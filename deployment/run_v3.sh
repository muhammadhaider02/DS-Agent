#!/bin/bash

TASKS=(
    "smoker-status"
    "mohs-hardness"
    "bitcoin-price-prediction"
    "heartbeat"
    "webmd-reviews"
    "cirrhosis-outcomes"
    "software-defects"
    "hotel-reviews"
    "electricity"
    "detect-ai-generation"
    "weather"
    "self-regulation-scp1"
    "uwave-gesture-library"
    "traffic"
    "boolq"
    "crab-age"
    "concrete-strength"
    "jigsaw"
)

LLM="deepseek-chat"
PATH_PREFIX="${LLM}_True_1"
TRIALS=10
GEN_TIMEOUT=600
EVAL_TIMEOUT=3600

echo "============================================================"
echo "  DS-Agent Deployment — DeepSeek-V3"
echo "  Tasks: ${#TASKS[@]}  |  Trials: ${TRIALS}"
echo "  Started: $(date)"
echo "============================================================"

PASS=0
FAIL=0

for task in "${TASKS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  Task: ${task}  |  $(date)"
    echo "------------------------------------------------------------"

    echo "  [1/2] Generating code..."
    python -u generate.py \
        --llm "${LLM}" \
        --task "${task}" \
        --shot 1 \
        --retrieval \
        --trials "${TRIALS}" \
        --timeout "${GEN_TIMEOUT}"
    GEN_EXIT=$?

    if [ ${GEN_EXIT} -ne 0 ]; then
        echo "  [WARN] Generation failed for '${task}' (exit ${GEN_EXIT}). Skipping evaluation."
        (( FAIL++ ))
        sleep 5
        continue
    fi

    echo "  [2/2] Evaluating..."
    python -u evaluation.py \
        --path "${PATH_PREFIX}" \
        --task "${task}" \
        --trials "${TRIALS}" \
        --device 0 \
        --timeout "${EVAL_TIMEOUT}"
    EVAL_EXIT=$?

    if [ ${EVAL_EXIT} -ne 0 ]; then
        echo "  [WARN] Evaluation failed for '${task}' (exit ${EVAL_EXIT})."
        (( FAIL++ ))
    else
        echo "  [OK] '${task}' done."
        (( PASS++ ))
    fi

    sleep 5
done

echo ""
echo "============================================================"
echo "  V3 PIPELINE COMPLETE"
echo "  Passed: ${PASS}  |  Failed/Skipped: ${FAIL}"
echo "  Finished: $(date)"
echo "============================================================"
