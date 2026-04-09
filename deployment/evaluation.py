import os
import shutil
import argparse
import pandas as pd
from execution import execute_script

DEPLOYMENT_TASKS = [
    'smoker-status', 'mohs-hardness', 'bitcoin-price-prediction', 'heartbeat',
    'webmd-reviews', 'cirrhosis-outcomes', 'software-defects', 'hotel-reviews',
    'electricity', 'detect-ai-generation', 'weather', 'self-regulation-scp1',
    'uwave-gesture-library', 'traffic', 'boolq', 'crab-age',
    'concrete-strength', 'jigsaw'
]


def get_args():
    parser = argparse.ArgumentParser()
    # Path to the generated codes folder (must match the prefix used in generate.py)
    parser.add_argument("--path",    default="deepseek-reasoner_True_1",
                        help="Code path prefix (e.g. deepseek-reasoner_True_1)")
    parser.add_argument("--task",    default="electricity",
                        help="Task name or 'all'")
    parser.add_argument("--trials",  default=10, type=int,
                        help="Number of generated trials to evaluate")
    parser.add_argument("--device",  default="0", type=str,
                        help="CUDA device id")
    parser.add_argument("--timeout", default=3600, type=int,
                        help="Per-script timeout in seconds (default 3600 = 1h). "
                             "Set 0 to disable.")
    args = parser.parse_args()
    # 0 means no timeout — pass None to execute_script
    if args.timeout == 0:
        args.timeout = None
    return args


if __name__ == '__main__':
    args = get_args()

    # Load tasks
    if args.task == "all":
        tasks_to_evaluate = DEPLOYMENT_TASKS
    else:
        assert args.task in DEPLOYMENT_TASKS, \
            f"Unknown task '{args.task}'. Must be in DEPLOYMENT_TASKS or 'all'."
        tasks_to_evaluate = [args.task]

    for task in tasks_to_evaluate:
        print(f"\n{'='*60}")
        print(f"Evaluating Task: {task}")
        print(f"{'='*60}")

        # ── Set up workspace ──────────────────────────────────────────────────
        work_dir = f"./workspace/{args.path}/{task}"
        os.makedirs(work_dir, exist_ok=True)

        bench_env = f"../development/MLAgentBench/benchmarks/{task}/env"
        if os.path.exists(bench_env):
            shutil.copytree(bench_env, work_dir, symlinks=True, dirs_exist_ok=True)

        codes_dir = f"./codes/{args.path}/{task}"
        if os.path.exists(codes_dir):
            shutil.copytree(codes_dir, work_dir, symlinks=True, dirs_exist_ok=True)

        # ── Detect result pattern from submission.py ──────────────────────────
        submission_path = f"{work_dir}/submission.py"
        if not os.path.exists(submission_path):
            print(f"  [SKIP] submission.py not found for task '{task}'")
            continue

        last_line = None
        with open(submission_path) as f:
            for line in f:
                if "print" in line:
                    last_line = line

        if not last_line:
            print(f"  [SKIP] No print statement found in submission.py for '{task}'")
            continue

        is_dual_metric = ("MSE" in last_line and "MAE" in last_line)
        results = [[], []] if is_dual_metric else []
        pattern = last_line.split("\"")[1].split(":")[0]

        # ── Result output path ────────────────────────────────────────────────
        result_dir      = f"results/{args.path}"
        os.makedirs(result_dir, exist_ok=True)
        result_filename = f"{result_dir}/{task}.txt"

        # ── Run each generated trial ──────────────────────────────────────────
        for idx in range(args.trials):
            filename = f"train_{idx}.py"
            print(f"  Running trial {idx+1}/{args.trials}: {filename}")

            log = execute_script(
                filename,
                work_dir=work_dir,
                device=args.device,
                timeout=args.timeout,
            )

            if pattern in log:
                if is_dual_metric:
                    results[0].append(float(log.split(pattern)[1].split(":")[1].split(",")[0]))
                    results[1].append(float(log.split(pattern)[1].split(":")[2].strip(",.\n ")))
                else:
                    results.append(float(log.split(pattern)[1].split(":")[1].strip(",.\n ")))
            else:
                # Script failed or timed out — record sentinel
                print(f"    [FAIL] Pattern not found in output for trial {idx}")
                if is_dual_metric:
                    results[0].append(-1.0)
                    results[1].append(-1.0)
                else:
                    results.append(-1.0)

        # ── Save results ──────────────────────────────────────────────────────
        if is_dual_metric:
            df = pd.DataFrame(results)
        else:
            df = pd.DataFrame(results).transpose()
        df.to_csv(result_filename, index=False, header=False)
        print(f"  Results saved to {result_filename}")
        print("=" * 60)