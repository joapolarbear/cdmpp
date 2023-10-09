#!/bin/bash

PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH

TASK_NUMBER=200
EXP_TIMES=3

RST_DIR=".workspace/tiramisu"
mkdir -p $RST_DIR/logs
mkdir -p $RST_DIR/cost_models

# PLATFORM="cuda"
# PLATFORM="llvm"
PLATFORM=$1
GPU_DEVICES=('t4' 'a100' 'p100' 'v100' 'k80')
CPU_DEVICES=('e5-2673'  'epyc-7452'  'graviton2')
if [[ -L dataset ]]; then
    rm dataset
fi
if [[ ${PLATFORM} == "llvm" ]]; then
    DEVICES=(${CPU_DEVICES[@]})
    ln -sf $PWD/dataset_cpu dataset
elif [[ ${PLATFORM} == "cuda" ]]; then
    DEVICES=(${GPU_DEVICES[@]})
    ln -sf $PWD/dataset_gpu dataset
else
    echo "Invalid platform $PLATFORM"
fi
echo "$PLATFORM,$DEVICES"

for DEVICE in "${DEVICES[@]}"; do
    for (( i=1; i<=${EXP_TIMES}; i++ )); do
        echo "Device ${DEVICE}, iter ${i}"
        LOG_PATH=${RST_DIR}/logs/tiramisu_${DEVICE}_${i}_${TASK_NUMBER}.txt
        bash scripts/train.sh run -y \
            --mode sample${TASK_NUMBER} \
            --tb_logdir .workspace/runs/tiramisu \
            --tiramisu --gpu_model ${DEVICE} 2>&1 | tee ${LOG_PATH}

        mv .workspace/runs/tiramisu ${RST_DIR}/cost_modes/tiramisu_${DEVICE}_${i}_${TASK_NUMBER}
    done
done