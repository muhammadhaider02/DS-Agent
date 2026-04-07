#!/bin/bash
# ============================================================
# DS-Agent Group 1: feedback, airline-reviews, textual-entailment
# Usage: nohup ./run_group1.sh > group1_output.log 2>&1 &
# ============================================================

TASKS=(
    "feedback"
    "airline-reviews"
    "textual-entailment"
)

NUM_ROUNDS=1

BASE_CMD="python runner.py \
    --llm-name deepseek-reasoner \
    --edit-script-llm-name deepseek-chat \
    --fast-llm-name deepseek-chat \
    --ranking-model deepseek-chat"

echo "========================================================"
echo "  DS-Agent Group 1: ${#TASKS[@]} tasks x $NUM_ROUNDS rounds"
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

        timeout 18000 $BASE_CMD --task "$task" || true

        sleep 15
    done
done

echo ""
echo "========================================================"
echo "  GROUP 1 COMPLETE: $(( ${#TASKS[@]} * NUM_ROUNDS )) runs done"
echo "  Finished at: $(date)"
echo "========================================================"
