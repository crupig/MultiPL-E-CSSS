DEVICE=$1
MODEL_PATH=$2
PID=${3:-None}
BATCH_SIZE=5
NUM_SAMPLES=5
MAX_NEW_TOKENS=512

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
   --pid $PID

# capture exit code immediately
EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ] && [ "$PID" != "None" ]; then
    touch .success_flag
fi