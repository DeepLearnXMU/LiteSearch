MODEL=/path/to/base/model
TRAIN_DATA=/path/to/train/data
EVAL_DATA=/path/to/eval/data
DS_CONFIG=/path/to/ds_config.json
ACC_CONFIG=/path/to/accelerate_config.yaml
OUTPUT_DIR=/path/to/out/dir

accelerate launch --config_file=$ACC_CONFIG train_llama.py $MODEL $DS_CONFIG $TRAIN_DATA $EVAL_DATA $OUTPUT_DIR