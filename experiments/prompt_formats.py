# The first {} corresponds to the start of the prompt (includes few shot), {} corresponds to slow code

CODER_INSTRUCTION='Optimize the python program below to be functionally equivalent but run faster and use less memory.\
            Wrap the optimized code in a block of 3 backticks (```).\n\n'
CODER_PROMPT_FORMAT = '{}\n\n## Program:\n{}\n\n## Optimized (Runtime and Space) version of Program above:'
CODER_FEW_SHOT_FORMAT = '## Program:\n{}\n\n## Optimized (Runtime and Space) version of Program above:\n{}'

FEEDBACK_INSTRUCTION='Give feedback in english for why the code solution below is incorrect or inefficient and how the program can be fixed.\n\n'
FEEDBACK_PROMPT_FORMAT = '{}\n\n## Candidate solution:\n{}\n\n## Feedback for incorrectnes/inefficiency and how it can be improved:'
FEEDBACK_FEW_SHOT_FORMAT = '## Candidate solution:\n{}\n\n##  Feedback for incorrectnes/inefficiency and how it can be improved:{}'

REFINE_INSTRUCTION='Refine the given incorrect or sub-optimal code solution based on the feedback specified below. Wrap the refined code in a block of 3 backticks (```)\n\n'
REFINE_PROMPT_FORMAT = '{}\n\n## Feedback to improve the code:\n{}\n\n## Refined code that includes optimizations specified in feedback:'
REFINE_FEW_SHOT_FORMAT = '## Candidate solution:\n{}\n\n##  Feedback for incorrectnes/inefficiency and how it can be improved:{}'

REFLECT_INSTRUCTION = 'Based on the execution results, reflect on why the code solution below was incorrect or inefficient and how the program can be fixed.\n\n{}'
REFLECT_PROMPT_FORMAT = '{}\n\n## Execution Results:\n{}\n\n## Reflection on incorrectnes/inefficiency and how it can be improved:'
REFLECT_FEW_SHOT_FORMAT = '## Execution Results:\n{}\n\n##  Reflection on incorrectnes/inefficiency and how it can be improved:{}'

NL2CODE_INSTRUCTION='Write a python code which is efficient in terms of runtime and memory usage for the following problem description.\
            Wrap the optimized code in a block of 3 backticks (```).\n\n'
NL2CODE_PROMPT_FORMAT = '{}\n\n## Details:\n{}\n\n## Solution:'
NL2CODE_FEW_SHOT_FORMAT = '## Details:\n{}\n\n## Solution:{}'

NL2CODE_FEEDBACK_INSTRUCTION='Give feedback in english for why the code solution below is incorrect or inefficient and how the program can be fixed based on the problem description.\n{}'
FEEDBACK_NL2CODE_PROMPT_FORMAT = '{}\n\n## Candidate solution:\n{}\n\n## Feedback for incorrectnes/inefficiency and how it can be improved:'

NL2CODE_REFINE_INSTRUCTION='Refine the given incorrect or sub-optimal code solution based on the feedback specified below. Wrap the refined code in a block of 3 backticks (```)\n\n## Sub-optimal soliution:\n{}'