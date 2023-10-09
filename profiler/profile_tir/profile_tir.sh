#!/bin/bash 
set -x
PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH
export TF_ENABLE_ONEDNN_OPTS=0

# BATCH_SIZE_TO_TEST=(4 32 128)
BATCH_SIZE_TO_TEST=(4)
for (( i=0; i<${#BATCH_SIZE_TO_TEST[@]}; i+=1 )); do
    CUDA_VISIBLE_DEVICES=6 python3 profiler/profile_tir/profile_tir.py \
        -o profile \
        -m .workspace/onnx_models \
        -w .workspace/profile --per_task \
        --trial_num 500 \
        -b ${BATCH_SIZE_TO_TEST[i]}
done