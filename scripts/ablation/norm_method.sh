#!/ban/bash
# set -x

SAMPLE_NUM=2308
# SAMPLE_NUM=200
EXP_TIMES=1

RST_DIR=.workspace/ablation/norm_method

mkdir -p ${RST_DIR}/logs
mkdir -p ${RST_DIR}/cost_models

ALL_DEVICES=("t4" "v100" "a100" "p100" "k80")
OUTPUT_NORM_METHODS=(0 1 2 3 4 5 6 7)

LEAF_NODE_NO=5
FILTER_STR=110
# DISABLE_PE_STR="--disable_pe"
DISABLE_PE_STR=""

for DEVICE in ${ALL_DEVICES[@]}; do
    DATA_MODE=${DEVICE}:sample${SAMPLE_NUM}
    PREPROCESS_DATA_DIR=tmp/dataset-ave_lb_0_0-filters${FILTER_STR}
    if [[ ${DISABLE_PE_STR} == "--disable_pe" ]]; then
        PREPROCESS_DATA_DIR=${PREPROCESS_DATA_DIR}-not_use_pe
    fi
    PREPROCESS_DATA_DIR=${PREPROCESS_DATA_DIR}/$DATA_MODE
    if [[ ${LEAF_NODE_NO} != 5 ]]; then
        PREPROCESS_DATA_DIR=${PREPROCESS_DATA_DIR}-leaf_node_no_${LEAF_NODE_NO}
    fi
    echo "Preprocessed data dir: ${PREPROCESS_DATA_DIR}"

    for OUTPUT_NORM_METHOD in ${OUTPUT_NORM_METHODS[@]}; do
    echo "Output_norm_method: ${OUTPUT_NORM_METHOD}"
    for (( i=1; i<=${EXP_TIMES}; i++ )); do
    
        LOG_PATH=${RST_DIR}/logs/${DEVICE}_${i}_${SAMPLE_NUM}_${OUTPUT_NORM_METHOD}.txt
        TB_DIR=${RST_DIR}/cost_models/${DEVICE}_${i}_${SAMPLE_NUM}_${OUTPUT_NORM_METHOD}

        echo "LOG_PATH: ${LOG_PATH}"
        echo "TB_DIR: ${TB_DIR}"

        bash scripts/train.sh run -y \
            --mode "${DATA_MODE}" \
            -c tmp/search_trial_20221119_1575.yaml \
            -t ${TB_DIR} \
            --output_norm_method ${OUTPUT_NORM_METHOD} \
            --leaf_node_no ${LEAF_NODE_NO} \
            --load_cache \
            --filters ${FILTER_STR} ${DISABLE_PE_STR} \
            -S 100000 2>&1 | tee -a ${LOG_PATH}
        
    done
    done
    # rm -r ${PREPROCESS_DATA_DIR}
done
