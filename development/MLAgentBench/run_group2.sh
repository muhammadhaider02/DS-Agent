#!/bin/bash
# ============================================================
# DS-Agent Group 2: chatgpt-prompt, ett-m2, ili
# Usage: nohup ./run_group2.sh > group2_output.log 2>&1 &
# ============================================================

TASKS=(
    "chatgpt-prompt"
    "ett-m2"
    "ili"
)

NUM_ROUNDS=4

BASE_CMD="python runner.py \
    --llm-name deepseek-reasoner \
    --edit-script-llm-name deepseek-chat \
    --fast-llm-name deepseek-chat \
    --ranking-model deepseek-chat"

echo "========================================================"
echo "  DS-Agent Group 2: ${#TASKS[@]} tasks x $NUM_ROUNDS rounds"
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

        $BASE_CMD --task "$task" || true

        sleep 15
    done
done

echo ""
echo "========================================================"
echo "  GROUP 2 COMPLETE: $(( ${#TASKS[@]} * NUM_ROUNDS )) runs done"
echo "  Finished at: $(date)"
echo "========================================================"
