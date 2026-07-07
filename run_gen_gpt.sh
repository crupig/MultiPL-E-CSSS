MODEL_PATH=$1
NUM_SAMPLES=10
BATCH_SIZE=10

python3 openai_model_fixed.py \
    --model $MODEL_PATH \
    --prompt-prefix "Write a solution for the following Java method. Only output, in a \```java \``` block, the whole implementation of the method starting from its signature, and no other text." \
    --root-dataset mbpp \
    --lang java \
    --batch-size $BATCH_SIZE \
    --temperature 1.0 \
    --completion-limit $NUM_SAMPLES \
    --output-dir-prefix ./generations
