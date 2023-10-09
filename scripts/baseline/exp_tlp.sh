#!/bin/bash
# set -x

PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH

TASK_NUMBER=200
EXP_TIMES=3
cd 3rdparty/tlp/scripts
RST_DIR=".workspace"
mkdir -p ${RST_DIR}/logs
mkdir -p ${RST_DIR}/cost_models

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

rm ${RST_DIR}/dataset_${TASK_NUMBER}_train_and_val.pkl
rm ${RST_DIR}/dataset_${TASK_NUMBER}_test.pkl
for DEVICE in "${DEVICES[@]}"; do
    for (( i=1; i<=${EXP_TIMES}; i++ )); do
        echo "Cross-model: device ${DEVICE}, Exp iter ${i}"
        
        python3 tlp_make_dataset.py \
            --files_cnt=${TASK_NUMBER} \
            --json_files_path=dataset/measure_records/${DEVICE} --platform=${PLATFORM} \
            --save_name ${RST_DIR}/dataset

        COST_MODEL_DIR=${RST_DIR}/cost_models/tlp_${DEVICE}_${i}_${TASK_NUMBER}
        LOG_PATH=${RST_DIR}/logs/tlp_${DEVICE}_${i}_${TASK_NUMBER}.txt

        python3 tlp_train.py \
            --save_folder=${COST_MODEL_DIR} \
            --dataset=${RST_DIR}/dataset_${TASK_NUMBER}_train_and_val.pkl \
            --step_size=40 --fea_size=20 2>&1 | tee ${LOG_PATH}

        python3 tlp_eval.py \
            --test_dataset_name=${RST_DIR}/dataset_${TASK_NUMBER}_test.pkl \
            --load_name=${COST_MODEL_DIR}/tlp_model_49.pkl \
            --platform=${PLATFORM} 2>&1 | tee -a ${LOG_PATH}

        rm ${RST_DIR}/dataset_${TASK_NUMBER}_train_and_val.pkl
        rm ${RST_DIR}/dataset_${TASK_NUMBER}_test.pkl

    done
done  