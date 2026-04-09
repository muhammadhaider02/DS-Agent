import json
import random

RP_PATH     = "../development/MLAgentBench/benchmarks/{}/scripts/research_problem.txt"
PYTHON_PATH = "../development/MLAgentBench/benchmarks/{}/env/train.py"
CASE_PATH   = "./experience_replay/{}.py"


# DS-Agent: retrieved case(s) + target task
FEW_SHOT_PROMPT = """Here are some example cases that solve machine learning tasks:
{}
Now please solve the following machine learning task based on the example cases above.
[Task]
{}
[train.py] ```python
{}
```
Start the python code with \"```python\". Please ensure the completeness of the code so that it can be run without additional modifications.
"""

# Format for each retrieved example case
CASE_PROMPT = """[Task]
{}
[train.py] ```python
{}
```
[Solution] ```python
{}
```
"""


def get_task(task: str):
    """Return (research_problem, starter_train_py) for a deployment task."""
    with open(RP_PATH.format(task)) as f:
        rp = f.read()
    with open(PYTHON_PATH.format(task)) as f:
        code = f.read()
    return rp, code


def get_case(task: str) -> str:
    """Return a formatted CASE_PROMPT string for a development task."""
    with open(RP_PATH.format(task)) as f:
        rp = f.read()
    with open(PYTHON_PATH.format(task)) as f:
        code = f.read()
    with open(CASE_PATH.format(task)) as f:
        case = f.read()
    return CASE_PROMPT.format(rp, code, case)


def get_prompt(task: str, context_num: int = 1,
               strategy: str = "retrieval") -> str:
    """
    Build and return the full prompt for the DS-Agent adapter.

    Parameters
    ----------
    task         : deployment task name
    context_num  : number of retrieved cases to include (default 1)
    strategy     : "retrieval" (ranked) or "random" (random sample)
    """
    rp, code = get_task(task)

    if context_num < 1:
        raise ValueError(
            "DS-Agent requires at least 1 in-context example (context_num >= 1). "
            "Zero-shot is not supported."
        )

    with open("./config/similarity_ranking.json") as f:
        ranking_dictionary = json.load(f)

    if strategy == "retrieval":
        selected_tasks = ranking_dictionary[task][:context_num]
    elif strategy == "random":
        selected_tasks = random.sample(ranking_dictionary[task], k=context_num)
    else:
        raise NotImplementedError(
            f"Strategy '{strategy}' is not supported. Use 'retrieval' or 'random'."
        )

    examples = "".join(get_case(t) for t in selected_tasks)
    return FEW_SHOT_PROMPT.format(examples, rp, code)


if __name__ == '__main__':
    # Quick smoke-test
    p = get_prompt("cirrhosis-outcomes", context_num=1, strategy="retrieval")
    print(p)