import pandas as pd
import os
from time import time
import requests
import asyncio
    
def judge_eval_single_test(
        code, 
        test_input_path, 
        number_of_runs=1, 
        synchronous=True, 
        judge_url='http://<YOUR_URL>:PORT'
):
    """Evaluate code on a single test case with Judge

    Returns:
        token: Submission token for async submission
        url: to check status of async submissions
        post_res: POST requests' response (for syncronous has all the stats)
    """
    with open(test_input_path, "r") as in_f:
        test_case_input = in_f.read()

    test_output_path = test_input_path.replace("input", "output")

    with open(test_output_path, "r") as out_f:
        test_case_output = out_f.read()

    request_body = {
        'source_code': code,
        'language_id': 71, # For python3 (For python 2: lang_id 70)
        'stdin': test_case_input,
        'number_of_runs': number_of_runs,
        'expected_output': test_case_output,
        'wall_time_limit': 20, # Max possible time and memory limits
        'memory_limit': 512000
    }

    if synchronous:
        post_res = requests.post(judge_url+'/submissions/?wait=true', request_body)
    else:
        post_res = requests.post(judge_url+'/submissions', request_body)

    if 'token' in post_res:
        token = post_res['token']
    else:
        token = ''

    status_url = judge_url + '/submissions' + f'/{token}'
    return token, status_url, post_res

def judge_submit(
        code, 
        problem_id,
        test_cases_path, 
        number_of_runs=1, 
        synchronous=True, 
        judge_url='http://<YOUR_URL>:PORT',
        number_of_tests=None,
):
    """
        Run a code on all test cases corresponding to a problem id

        Returns:
            accept: Pass all test cases or not
            pass_tests: List of all passed test_ids
            errors: Dict of test_id: stdout for all failed test cases (Wrong answer and Code Error) 
            run_times: Dict of test_id: run time (for all tests that can execute)
            memory: Dict of memory: run time (for all tests that can execute)
    """
    pass_tests = [] # Set of all passed test cases
    memory = {} # Test id: metric
    run_times = {}
    errors = {}

    # Sanity check
    problem_input_folder = os.path.join(test_cases_path, problem_id)
    if not os.path.exists(problem_input_folder):
        print(f"Does not exist {problem_input_folder}")
        return None 
    
    data = os.listdir(problem_input_folder)
    file_count = len(data)
    if file_count < 2: # 1 test case has one input and output (2 files min). If < 2 then 1 test case not present 
        print(f"Not enough test files for {problem_id}")
        return None

    input_files = [file for file in data if file.startswith("input")]
    input_files = sorted(input_files) # Execute tests in order

    if number_of_tests and number_of_tests < len(input_files):
        input_files = input_files[:number_of_tests] # pick first num_tests input files

    accept = True # Valid code 

    for i, input_file_id in enumerate(input_files):
        input_file= os.path.join(problem_input_folder, input_file_id)

        # input_file_id is like input.0.txt 
        test_id = int(input_file_id.split('.')[1]) # Middle is id # Could also be the same as enumeration

        # ======== Run code file for this test case and save stats =========
        token, url, res = judge_eval_single_test(code, input_file, 
                                            number_of_runs=number_of_runs, synchronous=synchronous, judge_url=judge_url # Optional args
                                        )
        
        res = res.json()

        valid = ('status' in res and res['status']['id'] in [3, 4]) # Status 3 is accepted Status 4 is wrong answer 
        # All other startus are runtime errors

        if not valid: # Runtime Error or other error
            accept = False # Failed a test case 
            if 'status' in res: 
                errors[test_id] = res['stderr']

        else: # Valid (Correct or wrong answer)
            run_times[test_id] = float(res['time']) 
            memory[test_id] = float(res['memory'])

            if res['status']['id'] == 3: # If correct, recording which test case passed
                pass_tests.append(test_id)
                
            else: # Status 4 -> Wrong answer
                test_output_path = input_file.replace("input", "output")
                with open(test_output_path, "r") as out_f:
                    expected_output = out_f.read()
                    
                if not res['stdout']:
                    res['stdout'] = '' # Empty string to indicate no output

                errors[test_id] = 'Wrong Answer:\n' + res['stdout'] + '\nExpected output:\n' + expected_output + '\n'

    return accept, pass_tests, errors, run_times, memory

        
