import argparse
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from prompt import get_prompt

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

if not DEEPSEEK_API_KEY:
    raise EnvironmentError(
        "DEEPSEEK_API_KEY not found. Please set it in the .env file."
    )

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# ── Task definitions ────────────────────────────────────────────────────────────
DEPLOYMENT_TASKS = [
    'smoker-status', 'mohs-hardness', 'bitcoin-price-prediction', 'heartbeat',
    'webmd-reviews', 'cirrhosis-outcomes', 'software-defects', 'hotel-reviews',
    'electricity', 'detect-ai-generation', 'weather', 'self-regulation-scp1',
    'uwave-gesture-library', 'traffic', 'boolq', 'crab-age',
    'concrete-strength', 'jigsaw'
]

# ── Token log path ─────────────────────────────────────────────────────────────
TOKEN_LOG_DIR = "./token_logs"


def get_args():
    parser = argparse.ArgumentParser()
    # Model Information
    parser.add_argument("--llm", default="deepseek-reasoner",
                        choices=["deepseek-chat", "deepseek-reasoner"],
                        help="DeepSeek model to use")
    parser.add_argument("--task", default="detect-ai-generation",
                        help="ML task name or 'all'")
    # Context Configuration — DS-Agent always uses retrieval + shot=1
    parser.add_argument("--shot", default=1, type=int,
                        help="Number of in-context examples (1 for DS-Agent)")
    parser.add_argument("--retrieval", default=False,
                        action='store_true',
                        help="Activate retrieval (DS-Agent mode)")
    # Generation Configuration
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--trials", default=10, type=int,
                        help="Number of generation trials per task")
    # Timeout
    parser.add_argument("--timeout", default=120, type=int,
                        help="Seconds before an API call is considered hung (default: 120)")
    args = parser.parse_args()
    return args


def _update_token_file(token_file: str, prompt_tokens: int, completion_tokens: int):
    """Read-modify-write a token count file."""
    try:
        with open(token_file, "r") as f:
            lines = f.read().strip().splitlines()
            existing_input  = int(lines[0].split("=")[1])
            existing_output = int(lines[1].split("=")[1])
    except (FileNotFoundError, IndexError, ValueError):
        existing_input = existing_output = 0
    with open(token_file, "w") as f:
        f.write(f"input={existing_input + prompt_tokens}\n")
        f.write(f"output={existing_output + completion_tokens}\n")


def log_token_usage(llm: str, task: str, prompt_tokens: int, completion_tokens: int):
    """Log tokens at two levels: cumulative per-model AND per-task."""
    os.makedirs(TOKEN_LOG_DIR, exist_ok=True)
    safe_llm  = llm.replace("/", "_").replace("-", "_")
    safe_task = task.replace("-", "_")

    # Cumulative across all tasks for this model
    _update_token_file(
        os.path.join(TOKEN_LOG_DIR, f"tokens_{safe_llm}.txt"),
        prompt_tokens, completion_tokens
    )
    # Per-task breakdown
    _update_token_file(
        os.path.join(TOKEN_LOG_DIR, f"tokens_{safe_llm}__{safe_task}.txt"),
        prompt_tokens, completion_tokens
    )


def log_to_file(log_file: str, prompt: str, completion: str,
                prompt_tokens: int, completion_tokens: int):
    """Write prompt + completion + token counts to the finetune-log file."""
    with open(log_file, "wt", encoding="utf-8") as f:
        f.write(prompt)
        f.write("\n[This is a split string for finetuning]\n")
        f.write(completion)
        f.write("\n[This is a split string for counting tokens]\n")
        f.write(f"Prompt: {prompt_tokens}, Completion: {completion_tokens}")


def generation(prompt: str, llm: str, temperature: float = 0.7,
               log_file: str = None, timeout: int = 120):
    """
    Call the DeepSeek API and return the extracted Python code block.

    - deepseek-reasoner: no stop sequences, large max_tokens to avoid truncation.
    - deepseek-chat    : standard settings.
    """
    is_reasoner = (llm == "deepseek-reasoner")

    # DeepSeek-R1 produces long chain-of-thought before the answer;
    # give it enough room so the final code block is never cut off.
    max_tokens = 16000 if is_reasoner else 4000

    request_params = {
        "model":       llm,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    # deepseek-reasoner does NOT support the `stop` parameter
    if not is_reasoner:
        request_params["stop"] = None

    completion     = None
    raw_completion = ""
    prompt_tokens  = 0
    comp_tokens    = 0

    for attempt in range(50):
        try:
            messages = [{"role": "user", "content": prompt}]
            response = deepseek_client.chat.completions.create(
                messages=messages,
                timeout=timeout,
                **request_params,
            )

            raw_completion = response.choices[0].message.content or ""

            # Record actual token usage from the API response
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                comp_tokens   = response.usage.completion_tokens

            # Extract the Python code block
            if "```python" in raw_completion:
                block = raw_completion.split("```python")[1].split("```")[0]
                if block.strip():
                    completion = block
                    break
            else:
                # Fallback: treat the whole response as code if no fence found
                if raw_completion.strip():
                    completion = raw_completion
                    break

        except Exception as e:
            print(f"===== Retry: {attempt + 1} =====")
            print(f"Error calling API: {e}")
            time.sleep(2)

    if not completion:
        completion = ""
    print(completion)

    if log_file:
        log_to_file(log_file, prompt, raw_completion, prompt_tokens, comp_tokens)

    return completion, prompt_tokens, comp_tokens


if __name__ == '__main__':
    args = get_args()

    # Load tasks
    if args.task == "all":
        tasks_to_solve = DEPLOYMENT_TASKS
    else:
        assert args.task in DEPLOYMENT_TASKS, \
            f"Unknown task '{args.task}'. Must be one of DEPLOYMENT_TASKS or 'all'."
        tasks_to_solve = [args.task]

    # Output directory path  (mirrors original naming convention)
    prefix = f"{args.llm}_{args.retrieval}_{args.shot}"
    pathname   = f"./codes/{prefix}"
    finetune_dir = f"./codes/{prefix}/finetune_log"
    os.makedirs(pathname, exist_ok=True)
    os.makedirs(finetune_dir, exist_ok=True)

    total_prompt_tokens = 0
    total_comp_tokens   = 0

    for task in tasks_to_solve:
        print(f"\n{'='*60}")
        print(f"Processing Task: {task}")
        print(f"{'='*60}")

        tmp_pathname    = f"{pathname}/{task}"
        temp_finetunedir = f"{finetune_dir}/{task}"
        os.makedirs(tmp_pathname, exist_ok=True)
        os.makedirs(temp_finetunedir, exist_ok=True)

        for idx in range(args.trials):
            print(f"  Trial {idx+1}/{args.trials} ...")
            prompt = get_prompt(
                task,
                context_num=args.shot,
                strategy="retrieval" if args.retrieval else "random",
            )
            response, pt, ct = generation(
                prompt, args.llm,
                temperature=args.temperature,
                log_file=f"{temp_finetunedir}/{idx}.txt",
                timeout=args.timeout,
            )
            total_prompt_tokens += pt
            total_comp_tokens   += ct
            log_token_usage(args.llm, task, pt, ct)

            filename = f"{tmp_pathname}/train_{idx}.py"
            with open(filename, "wt", encoding="utf-8") as file:
                file.write(response)

    print(f"\n{'='*60}")
    print(f"Run complete.")
    print(f"Total prompt tokens : {total_prompt_tokens}")
    print(f"Total completion tokens: {total_comp_tokens}")
    print(f"Token log dir: {TOKEN_LOG_DIR}")
    print(f"{'='*60}")