DEVICE=$1
MODEL_PATH=$2
PID=${3:-None}
BATCH_SIZE=5
NUM_SAMPLES=5
MAX_NEW_TOKENS=512

IDS_FILE="../../constants/ids_train_val_test.json"
JSON_CONTENT=$(cat "$IDS_FILE")

# python3 automodel.py \
CUDA_VISIBLE_DEVICES=$DEVICE python3 automodel_with_logprobs.py \
    --name $MODEL_PATH \
    --root-dataset mbpp \
    --lang java \
    --temperature 1.0 \
    --batch-size $BATCH_SIZE \
    --completion-limit $NUM_SAMPLES \
    --max-tokens $MAX_NEW_TOKENS \
    --top-p 0.95 \
    --output-dir-prefix ./generations \
    --pid $PID \
    --all_ids_dict "$JSON_CONTENT" \
    --split all # train, val, test, or all


# capture exit code immediately
EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ] && [ "$PID" != "None" ]; then
    touch .success_flag
fi