#!/bin/bash
# ============================================================
# DS-Agent Full Benchmark Runner (Round-Robin, 12 Tasks x 5 Runs)
# Usage: ./run_all.sh
# Recommended: nohup ./run_all.sh > benchmark_output.log 2>&1 &
# ============================================================

# set -e removed: we don't want the whole benchmark to die if one task is killed

TASKS=(
    "feedback"
    "airline-reviews"
    "textual-entailment"
    "chatgpt-prompt"
    "ett-m2"
    "ili"
    "handwriting"
    "ethanol-concentration"
    "media-campaign-cost"
    "wild-blueberry-yield"
    "smoker-status"
    "enzyme-substrate"
)

NUM_ROUNDS=1

BASE_CMD="python runner.py \
    --llm-name deepseek-reasoner \
    --edit-script-llm-name deepseek-chat \
    --fast-llm-name deepseek-chat \
    --ranking-model deepseek-chat"

echo "========================================================"
echo "  DS-Agent Benchmark: ${#TASKS[@]} tasks x $NUM_ROUNDS rounds"
echo "  Total runs: $(( ${#TASKS[@]} * NUM_ROUNDS ))"
echo "  Started at: $(date)"
echo "========================================================"

for (( round=1; round<=NUM_ROUNDS; round++ )); do
    echo ""
    echo "========================================================"
    echo "  ROUND $round of $NUM_ROUNDS"
    echo "  $(date)"
    echo "========================================================"

    for task in "${TASKS[@]}"; do
        echo ""
        echo "  --> Task: $task | Round $round/$NUM_ROUNDS | $(date)"
        echo "--------------------------------------------------------"

        timeout 18000 $BASE_CMD --task "$task"

        # Brief pause to clear GPU memory between runs
        sleep 15
    done
done

echo ""
echo "========================================================"
echo "  ALL $(( ${#TASKS[@]} * NUM_ROUNDS )) RUNS COMPLETE"
echo "  Finished at: $(date)"
echo "========================================================"
