#!/bin/bash
# Pre-process raw features offline

### Some samples
# bash scripts/dataset/make_dataset.sh --mode "(t4:sample2308),(v100:sample2308)" -y
# function usage_prompt {
#     echo "Usage: bash xxx.sh ..."
#     exit
# }
# if [ -z $1 ]; then
#     usage_prompt
# fi

PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH
export PATH=$PATH:/usr/local/cuda/bin
export CUBLAS_WORKSPACE_CONFIG=:4096:8
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
SAMPLE_NUM=2308
SAMPLE_NUM=200
remain_arg=${@:1}

python3 main.py \
    --source_data tir \
    -o make_dataset \
    --mode sample${SAMPLE_NUM} \
    -i .workspace/ast_ansor $remain_arg

