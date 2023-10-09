#!/bin/bash
# set -x

PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH

TASK_NUMBER=200
EXP_TIMES=3
cd 3rdparty/tlp/scripts
RST_DIR=".workspace"
mkdir -p ${RST_DIR}/logs
mkdir -p ${RST_DIR}/cost_models

GPU_DEVICES=('t4' 'k80' 'a100' 'p100' 'v100' )
CPU_DEVICES=('e5-2673'  'epyc-7452'  'graviton2')

### For cross device learning, unify the seq_len and emb_size for CPU and GPUs
function single_device_make_dataset {
    if [[ -f ${RST_DIR}/dataset_${1}_${TASK_NUMBER}_train_and_val.pkl && -f ${RST_DIR}/dataset_${1}_${TASK_NUMBER}_test.pkl ]]; then
        echo "[Dataset] Dataset for devoce $1 has been created"
    else
        echo "[Dataset] Create dataset for device ${DEVICE}"
        python3 tlp_make_dataset.py \
            --files_cnt=${TASK_NUMBER} \
            --json_files_path=dataset/measure_records/$1 \
            --platform=${2} \
            --save_name ${RST_DIR}/dataset_$1 \
            --crop_seq_len 25 \
            --crop_emb_size 22
    fi
}

if [[ -L dataset ]]; then
    rm dataset
fi
ln -sf $PWD/dataset_gpu dataset
for DEVICE in "${GPU_DEVICES[@]}"; do
    single_device_make_dataset ${DEVICE} cuda
done
rm dataset && ln -sf $PWD/dataset_cpu dataset
for DEVICE in "${CPU_DEVICES[@]}"; do
    single_device_make_dataset ${DEVICE} llvm
done
rm dataset


GPU_SOURCE_DATA=""
GPU_MLT_HEAD_LIST=""
GPU_DEVICE_CNT=0
for SOURCE_DEVICE in "${GPU_DEVICES[@]}"; do
    GPU_SOURCE_DATA="${GPU_SOURCE_DATA} ${RST_DIR}/dataset_${SOURCE_DEVICE}_${TASK_NUMBER}_train_and_val.pkl"
    GPU_DEVICE_CNT=$((${GPU_DEVICE_CNT}+1))
    if [[ -z ${GPU_MLT_HEAD_LIST} ]]; then
        GPU_MLT_HEAD_LIST=${GPU_DEVICE_CNT}
    else
        GPU_MLT_HEAD_LIST=${GPU_MLT_HEAD_LIST},${GPU_DEVICE_CNT}
    fi
done

ln -sf $PWD/dataset_cpu dataset
for TARGET_DEVICE in "${CPU_DEVICES[@]}"; do
    for (( i=1; i<=${EXP_TIMES}; i++ )); do
        echo ""
        echo "GPUs to CPU: to device ${TARGET_DEVICE}, Exp iter ${i}"
        SOURCE_DATA=${GPU_SOURCE_DATA}
        MLT_HEAD_LIST=${GPU_MLT_HEAD_LIST}
        DEVICE_CNT=${GPU_DEVICE_CNT}
        for SOURCE_DEVICE in "${CPU_DEVICES[@]}"; do
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
        
        COST_MODEL_DIR=${RST_DIR}/cost_models/gpu2cpu_tlp_${TARGET_DEVICE}_${i}_${TASK_NUMBER}
        LOG_PATH=${RST_DIR}/logs/gpu2cpu_tlp_${TARGET_DEVICE}_${i}_${TASK_NUMBER}.txt

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
            --mtl_head_list=${MLT_HEAD_LIST} \
            2>&1 | tee ${LOG_PATH}

        # Test the cost model
        python3 tlp_eval.py \
            --test_dataset_name=${RST_DIR}/dataset_${TARGET_DEVICE}_${TASK_NUMBER}_test.pkl \
            --load_name=${COST_MODEL_DIR}/mtl_tlp_model_49.pkl \
            --platform=llvm 2>&1 | tee -a ${LOG_PATH}

    done
done
rm dataset


