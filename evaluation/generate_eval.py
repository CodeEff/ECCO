import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test_cases_path', default='./data/codenet/generated_test_cases')
parser.add_argument('--judge_url', default='http://<YOUR_URL>:PORT')
parser.add_argument('--input_path', default=None)
parser.add_argument('--out_path', default='./judge_eval/generate/')
parser.add_argument('--nrows', default=None, type=int)
parser.add_argument('--code_col_name', default='generated_codes')
parser.add_argument('--num_runs', type=int, default=3)
parser.add_argument('--num_tests', type=int, default=None)
args = parser.parse_args()

def calculate_percentile(generated_json_data, submission_details):
    eval_data={}

    for index,row in tqdm(generated_json_data.iterrows(), total=len(generated_json_data)):
        codes = row[args.code_col_name] # Can be a list of generated samples
        problem_id = row['problem_id']
        if problem_id not in submission_details['problem_id'].values:
            print(f"Problem {problem_id} not in submission details")
            continue

        if type(codes) == str: # If not a list 
            fast_codes = [codes] # Make it a singular list
        else:
            fast_codes = codes

        # Get the test cases
        problem_input_folder = args.test_cases_path + problem_id
        if not os.path.exists(problem_input_folder):
            print(f"Does not exist {problem_input_folder}")
            continue
        data = os.listdir(problem_input_folder)
        data = sorted(data)
        file_count = len(data)
        if file_count<2:
            print(f"Not enough test files for {problem_id}")
            continue

        data = os.listdir(problem_input_folder)
        input_files = [file for file in data if file.startswith("input")]
        input_files = sorted(input_files)

        # Stats for each test case
        num_tests = len(input_files)
        if num_tests > args.num_tests:
            num_tests = args.num_tests
        num_samples = len(fast_codes)

        output_valid = [True] * num_samples
        output_pass = {} # Dict {sample_id : True/False}
        output_memory = {} # Test sample no : dict{test_id: metric}
        output_run_times = {} # Test sample no : dict{test_id: metric}
        output_errors = {} # Test sample no: dict{test_id: error str}

        judge_url = args.judge_url

        # Get stats for all generated samples
        for sample_id, code in enumerate(codes): # For each sample
            sample_valid, sample_pass, sample_errors, sample_run_times, sample_memory = judge_submit(
                code, problem_id, args.test_cases_path, args.num_runs, synchronous=True, judge_url=judge_url, number_of_tests=args.num_tests
            )

            output_valid[sample_id] = sample_valid
            output_pass[sample_id] = sample_pass
            output_run_times[sample_id] = sample_run_times
            output_memory[sample_id] = sample_memory
            output_errors[sample_id] = sample_errors

        # ======= Pick the best generated sample ========
        # Pick the solution that is the most correct first
        most_correct_samples = [] # ids

        num_passed_test = {x[0]: len(x[1]) for x in output_pass.items()} # x[0] sample id, x[1] is the set of passed test_ids, and get num
        max_pass_rate = max(num_passed_test.values()) # Max pass rate number
        for sample_id, pass_rate in num_passed_test.items():
            if pass_rate == max_pass_rate:
                most_correct_samples.append(sample_id)
        
        # Then pick based on memory and space across all tests
        # This is a dict sample_id: {test_id: dict of {test_id: runtime}} # 2 levels of dict
        filtered_run_times = {sample_id: output_run_times.get(sample_id) for sample_id in most_correct_samples}
        filtered_memory = {sample_id: output_memory.get(sample_id) for sample_id in most_correct_samples}

        best_time_sample_id, best_time = min(filtered_run_times.items(), key=lambda x: sum(x[1].values())) # Tuple of (bestsampleid, best time)
        best_mem_sample_id, best_mem = min(filtered_memory.items(), key=lambda x: sum(x[1].values())) 

        best_sample_id = best_time_sample_id # picking best time sample as the best solution
        # =====================

        # All statistics
        values = {
            f'problem_id': problem_id,
            f'accepted': output_valid, # At least one sample accepted

            f'best_run_time': None, # float
            f'best_memory': None, # float
            f'runtime_percentile': None, # float
            f'mem_percentile': None, # float

            f'pass_rate': len(output_pass[best_sample_id]) / num_tests, # float

            # Metadata from runs
            f'run_time_all': output_run_times[best_sample_id], # List (for each sample)
            f'memory_all': output_memory[best_sample_id], # List
            f'pass_all': output_pass[best_sample_id], # List of all test ids
            f'errors_all': output_errors[best_sample_id]
        }

        # ==== Calculate percentile ========
        submission_runtimes = submission_details[submission_details['problem_id']== problem_id]['runtimes']
        submission_memory = submission_details[submission_details['problem_id'] == problem_id]['memories']
        submission_runtimes = sorted(submission_runtimes.iloc[0],reverse=True)
        submission_memory = sorted(submission_memory.iloc[0],reverse=True)

        if len(output_pass[best_sample_id]) == num_tests: # Passed all tests 
            # Get sum of times of for all tests that the best sample passed 
            out_time_passed = []
            out_mem_passed = []

            for passed_test_id in output_pass[best_sample_id]: # Gives test id
                out_time_passed.append(output_run_times[best_sample_id][passed_test_id])

                out_mem_passed.append(output_memory[best_sample_id][passed_test_id])

            total_time = sum(out_time_passed) / len(out_time_passed)
            total_memory = sum(out_mem_passed) / len(out_mem_passed)

            # ===== Search for percentile ======
            # Can optimize this with binary search
            time_percentile = None
            mem_percentile = None
            for i, sub_run in enumerate(submission_runtimes):
                if sub_run < total_time: # First greater
                    time_percentile = i / len(submission_runtimes)
                    break
            if not time_percentile:
                time_percentile = 1
            for i, sub_mem in enumerate(submission_memory):
                if sub_mem < total_memory: # First greater
                    mem_percentile = i / len(submission_memory)
                    break
            if not mem_percentile:
                mem_percentile = 1

            values.update({
                'best_run_time': total_time, 
                'best_memory': total_memory, 
                'runtime_percentile': time_percentile,
                'mem_percentile': mem_percentile
            })
            
        eval_data[index] = (values)

    speedup_data_values = eval_data
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    file_path = os.path.join(args.out_path, args.code_col_name + '_' + args.input_path.split("/")[-1])
    with open(file_path, 'w') as json_file:
        print('Saving evaluation results to', file_path)
        json.dump(speedup_data_values, json_file, indent=2)

if __name__ == '__main__':
    gen_code_data = pd.read_json(args.input_path, nrows=args.nrows, orient='records', lines=True)
    submission_dataset = load_dataset('CodeEff/ECCO', 'generate_eval', split='test')
    submission_data = submission_dataset.to_pandas()
    calculate_percentile(gen_code_data, submission_data)