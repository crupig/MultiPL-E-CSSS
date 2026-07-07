DEVICE=$1
MODEL_PATH=$2
PID=${3:-None}
BATCH_SIZE=50
NUM_SAMPLES=2
MAX_NEW_TOKENS=512

IDS_FILE="../../constants/ids_train_val_test.json"
JSON_CONTENT=$(cat "$IDS_FILE")

CUDA_VISIBLE_DEVICES=$DEVICE python3 automodel_vllm_testcases.py \
    --name $MODEL_PATH \
    --root-dataset mbpp \
    --lang java \
    --temperature 1.0 \
    --batch-size $BATCH_SIZE \
    --completion-limit $NUM_SAMPLES \
    --max-tokens $MAX_NEW_TOKENS \
    --top-p 0.95 \
    --output-dir-prefix ./test-generations \
    --prompt-prefix "Provide up to 10 assert statements in Java aimed at testing the correctness of the following coding task. The assert statements should be based on the problem description and should cover various edge cases. Provide only the assert statements, no other text. Provide the assert statements in a markdown code block:" \
    --all_ids_dict "$JSON_CONTENT" \
    --split test # train, val, test, or all


# capture exit code immediately
EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ] && [ "$PID" != "None" ]; then
    touch .success_flag
fi