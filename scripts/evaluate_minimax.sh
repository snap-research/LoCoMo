# sets necessary environment variables
source scripts/env.sh

# Evaluate MiniMax-M2.5
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
    --model minimax-m2.5 --batch-size 10

# Evaluate MiniMax-M2.5-highspeed (204K context, faster inference)
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
    --model minimax-m2.5-highspeed --batch-size 10
