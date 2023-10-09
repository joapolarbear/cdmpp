#!/bin/bash
set -x

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

### For cross device learning
function single_device_make_dataset {
    python3 tlp_make_dataset.py \
        --files_cnt=${TASK_NUMBER} \
        --json_files_path=dataset/measure_records/$1 \
        --platform=${PLATFORM} \
        --save_name ${RST_DIR}/dataset_$1
}

for DEVICE in "${DEVICES[@]}"; do
    echo "Create dataset for device ${DEVICE}"
    single_device_make_dataset ${DEVICE}
done

for DEVICE in "${DEVICES[@]}"; do
    for (( i=1; i<=${EXP_TIMES}; i++ )); do
        TARGET_DEVICE=${DEVICE}
        echo "Cross-device: to device ${TARGET_DEVICE}, Exp iter ${i}"
        SOURCE_DATA=""
        MLT_HEAD_LIST=""
        DEVICE_CNT=0
        for SOURCE_DEVICE in "${DEVICES[@]}"; do
            if [[ ${SOURCE_DEVICE} != ${TARGET_DEVICE} ]]; then
                SOURCE_DATA="${SOURCE_DATA} ${RST_DIR}/dataset_${SOURCE_DEVICE}_${TASK_NUMBER}_train_and_val.pkl"
                DEVICE_CNT=$((${DEVICE_CNT}+1))
                if [[ -z ${MLT_HEAD_LIST} ]]; then
                    MLT_HEAD_LIST=${DEVICE_CNT}
                else
                    MLT_HEAD_LIST=${MLT_HEAD_LIST},${DEVICE_CNT}
                fi
            fi
        done
        SOURCE_DATA="${SOURCE_DATA} ${RST_DIR}/dataset_${TARGET_DEVICE}_${TASK_NUMBER}_train_and_val.pkl"
        MLT_HEAD_LIST=${MLT_HEAD_LIST},0
        
        COST_MODEL_DIR=${RST_DIR}/cost_models/mtl_tlp_${TARGET_DEVICE}_${i}_${TASK_NUMBER}
        LOG_PATH=${RST_DIR}/logs/mtl_tlp_${TARGET_DEVICE}_${i}_${TASK_NUMBER}.txt

        echo ${MLT_HEAD_LIST}
        echo ${SOURCE_DATA}
        echo ${COST_MODEL_DIR}
        echo ${LOG_PATH}

        python3 mtl_tlp_make_dataset.py \
            --union_datasets ${SOURCE_DATA} \
            --save_name ${RST_DIR}/union_dataset.pkl
        
        # Train the cost model 
        python3 mtl_tlp_train.py \
            --dataset=${RST_DIR}/union_dataset.pkl \
            --save_folder=${COST_MODEL_DIR} \
            --mtl_head_list=$MLT_HEAD_LIST \
            --step_size=40 --fea_size=20 \
            2>&1 | tee ${LOG_PATH}

        # Test the cost model
        python3 tlp_eval.py \
            --test_dataset_name=${RST_DIR}/dataset_${TARGET_DEVICE}_${TASK_NUMBER}_test.pkl \
            --load_name=${COST_MODEL_DIR}/mtl_tlp_model_49.pkl \
            --platform=${PLATFORM} 2>&1 | tee -a ${LOG_PATH}

    done
done



