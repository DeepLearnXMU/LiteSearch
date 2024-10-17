#!/bin/bash
SERVER_DIR=$1
GPU_ID=$2

echo $SERVER_DIR
echo $GPU_ID

export MODEL_PATH="path/to/value/model"

CUDA_VISIBLE_DEVICES=$GPU_ID uvicorn --app-dir "${SERVER_DIR}" server_hf:app --host 0.0.0.0 --port 9009 >& value_log &