#!/ban/bash
# set -x

TASK_NUMBER=200
EXP_TIMES=1

RST_DIR=.workspace/ablation/pe

mkdir -p ${RST_DIR}/logs
mkdir -p ${RST_DIR}/cost_models

ALL_DEVICES=("t4" "v100" "a100" "p100" "k80")

LEAF_NODE_NO=5
FILTER_STR=110
DISABLE_PE_STR="--disable_pe"
# DISABLE_PE_STR=""
OUTPUT_NORM_METHOD=0

for DEVICE in ${ALL_DEVICES[@]}; do
    DATA_MODE=${DEVICE}:sample${TASK_NUMBER}
    PREPROCESS_DATA_DIR=tmp/dataset-ave_lb_0_0-filters${FILTER_STR}
    if [[ ${DISABLE_PE_STR} == "--disable_pe" ]]; then
        PREPROCESS_DATA_DIR=${PREPROCESS_DATA_DIR}-not_use_pe
    fi
    PREPROCESS_DATA_DIR=${PREPROCESS_DATA_DIR}/$DATA_MODE
    if [[ ${LEAF_NODE_NO} != 5 ]]; then
        PREPROCESS_DATA_DIR=${PREPROCESS_DATA_DIR}-leaf_node_no_${LEAF_NODE_NO}
    fi
    echo "Preprocessed data dir: ${PREPROCESS_DATA_DIR}"

    for (( i=1; i<=${EXP_TIMES}; i++ )); do
    
        LOG_PATH=${RST_DIR}/logs/${DEVICE}_${i}_${TASK_NUMBER}_disable_pe.txt
        TB_DIR=${RST_DIR}/cost_models/${DEVICE}_${i}_${TASK_NUMBER}_disable_pe

        echo "LOG_PATH: ${LOG_PATH}"
        echo "TB_DIR: ${TB_DIR}"

        bash scripts/train.sh run -y \
            --mode "${DATA_MODE}" \
            -c tmp/search_trial_20221119_1575.yaml \
            -t ${TB_DIR} \
            --output_norm_method ${OUTPUT_NORM_METHOD} \
            --leaf_node_no ${LEAF_NODE_NO} \
            --load_cache \
            --disable_pe \
            --filters ${FILTER_STR} ${DISABLE_PE_STR} \
            -E 100 2>&1 | tee ${LOG_PATH}
        
    done
    rm -r ${PREPROCESS_DATA_DIR}
done
