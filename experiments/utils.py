def get_execution_feedback(accept, pass_tests, errors, run_times, memory):
    if len(errors) == 0: # Passed all test cases 
        feedback = f'Your solution was functionally CORRECT across ALL test cases!\n'
    elif len(pass_tests) > 0: # Passed at least one test case
        feedback = f'Your solution was INCORRECT and passed {len(pass_tests)} test cases.\n'
    else: # Passed no test cases
        feedback = f'Your solution was FULLY INCORRECT and passed 0 test cases. This could either be a flaw in logic or a syntax error. Please see error logs.\n'
            
    if len(pass_tests) > 0:
        feedback += '\nHere are the run time and memory that your code utilized for each test case\n'
        for test_id in run_times.keys(): # Has run time for passed as well as failed
            time, mem = run_times[test_id], memory[test_id]
            pass_or_fail_str = 'PASSED' if int(test_id) in pass_tests else 'FAILED'
            feedback += f'-- Stats for test case {test_id} --\n'
            feedback += f'Correct: {pass_or_fail_str}\nRun time: {time} s\nMemory: {mem} KB\n'

    if len(errors) > 0:
        feedback += 'Here are the error logs for the failed test cases\n'
        for test_id in errors.keys():
            # Test Input of failed test case NOT added to feedback
            feedback += f'-- Error log for failed test case {test_id} --\n'
            if errors[test_id]:
                feedback += errors[test_id] + '\n' # Wrong Answer: {} Expected Answer: {}
            else:
                feedback += '\n'

    return feedback
            
# =============== Prompt builders =================
def build_coder_prompts(row, engine=None, train=None, few_shot=0, instruct_version='instruct'):
    raw_prompt = engine.get_prompt(row['input'], few_shot, train, instruct=(instruct_version == 'instruct'), mode='coder') 
    prompt = engine.wrap_prompt_chat_template(raw_prompt) if instruct_version == 'instruct' else raw_prompt
    row['prompt'] = prompt
    return row

def build_feedback_prompts(row, try_col_name, engine=None, train=None, few_shot=0, instruct_version='instruct'):
    best_sample_id = -1 
    raw_prompt = engine.get_prompt(row[try_col_name][best_sample_id], few_shot, train, instruct=True, mode='feedback') 
    prompt = engine.wrap_prompt_chat_template(raw_prompt)
    row['feedback_prompt'] = prompt 
    return row

def build_refine_prompts(row, prev_try_col_name, feedback_col_name, engine=None, train=None, few_shot=0, instruct_version='instruct'):
    best_sample_id = -1 
    raw_prompt = engine.build_refine_prompt(row[prev_try_col_name][best_sample_id], row[feedback_col_name][best_sample_id], few_shot, train, instruct=True) 
    prompt = engine.wrap_prompt_chat_template(raw_prompt)
    row['refine_prompt'] = prompt 
    return row

def build_reflect_prompts(row, prev_try_col_name, exec_col_name, engine=None, train=None, few_shot=0, instruct_version='instruct'):
    best_sample_id = -1 
    raw_prompt = engine.build_reflect_prompt(row[prev_try_col_name][best_sample_id], row[exec_col_name][best_sample_id], few_shot, train, instruct=True) 
    prompt = engine.wrap_prompt_chat_template(raw_prompt)
    row['reflect_prompt'] = prompt 
    return row
    
def build_nl2code_prompts(row, engine=None, train=None, few_shot=0, instruct_version='instruct'):
    raw_prompt = engine.get_prompt(row['problem_description'], few_shot, train, instruct=(instruct_version == 'instruct'), mode='nl2code') 
    prompt = engine.wrap_prompt_chat_template(raw_prompt)
    row['prompt'] = prompt
    return row

def build_nl2code_feedback_prompts(row, try_col_name, engine=None, train=None, few_shot=0, instruct_version='instruct'):
    best_sample_id = -1 
    raw_prompt = engine.get_prompt(row[try_col_name][best_sample_id],  mode='nl2code_feedback') 
    prompt = engine.wrap_prompt_chat_template(raw_prompt)
    row['feedback_prompt'] = prompt 
    return row

def build_nl2code_refine_prompts(row, prev_try_col_name, feedback_col_name, engine=None, train=None, few_shot=0, instruct_version='instruct'):
    best_sample_id = -1 
    raw_prompt = engine.build_nl2code_refine_prompt(row[prev_try_col_name][best_sample_id], row[feedback_col_name][best_sample_id]) 
    prompt = engine.wrap_prompt_chat_template(raw_prompt)
    row['refine_prompt'] = prompt 
    return row