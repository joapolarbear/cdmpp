#!/bin/bash
# Scripts to extract raw features from Tenset dataset

### Some samples
# bash scripts/gen_raw_feature.sh ast_ansor t4
function usage_prompt {
    echo "Usage: bash xxx.sh [OUTPUT_DIR_NAME] ..."
    exit
}
if [ -z $1 ]; then
    usage_prompt
fi

PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH
export PATH=$PATH:/usr/local/cuda/bin
export CUBLAS_WORKSPACE_CONFIG=:4096:8
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

mkdir -p .workspace
OUTPUT_DIR_NAME=$1
DEVICE_MODEL=$2
remain_arg=${@:3}

### Load Tenset data and generate raw features
# set split_size to 0 to generate a feature file for each task
# set -j 8 to load data in parallel with 8 threads

python3 metalearner/feature/tenset_dataload.py \
        -o .workspace/${OUTPUT_DIR_NAME}/$DEVICE_MODEL \
        -i 3rdparty/tenset/scripts/dataset/measure_records/$DEVICE_MODEL/ \
        -j 8 \
        ${remain_arg}

### Some analysis and post-process task2split and split2task mapping
python3 metalearner/feature/test_task_files.py \
    .workspace/${OUTPUT_DIR_NAME}/$DEVICE_MODEL

