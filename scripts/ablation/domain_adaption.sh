#!/ban/bash
# set -x

TASK_NUMBER=200
EXP_TIMES=1

RST_DIR=.workspace/ablation/domain_adaption

mkdir -p ${RST_DIR}/logs
mkdir -p ${RST_DIR}/cost_models

ALL_DEVICES=("t4" "v100" "a100" "p100" "k80")
# mse+mape
ALL_DEVICES=("p100" "k80")
DOMAIN_DIFF_METRICS=("cmd" "mmd")
DOMAIN_DIFF_METRICS=("2cmd")

LEAF_NODE_NO=5
FILTER_STR=110
# DISABLE_PE_STR="--disable_pe"
DISABLE_PE_STR=""
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

    for DOMAIN_DIFF_METRIC in ${DOMAIN_DIFF_METRICS[@]}; do
    echo "DOMAIN_DIFF_METRIC: ${DOMAIN_DIFF_METRIC}"
    for (( i=1; i<=${EXP_TIMES}; i++ )); do
    
        LOG_PATH=${RST_DIR}/logs/${DEVICE}_${i}_${TASK_NUMBER}_${DOMAIN_DIFF_METRIC}.txt
        TB_DIR=${RST_DIR}/cost_models/${DEVICE}_${i}_${TASK_NUMBER}_${DOMAIN_DIFF_METRIC}

        echo "LOG_PATH: ${LOG_PATH}"
        echo "TB_DIR: ${TB_DIR}"

        if [[ ${DOMAIN_DIFF_METRIC} == "none" ]]; then
            DOMAIN_DIFF_ARG=""
        else
            DOMAIN_DIFF_ARG="--domain_diff_metric ${DOMAIN_DIFF_METRIC}"
        fi

        bash scripts/train.sh run -y \
            --mode "${DATA_MODE}" \
            -c tmp/search_trial_20221119_1575.yaml \
            -t ${TB_DIR} \
            --output_norm_method ${OUTPUT_NORM_METHOD} \
            --leaf_node_no ${LEAF_NODE_NO} \
            --load_cache \
            ${DOMAIN_DIFF_ARG} \
            --filters ${FILTER_STR} ${DISABLE_PE_STR} \
            --domain_diff_metric ${DOMAIN_DIFF_METRIC} \
            -E 100 2>&1 | tee -a ${LOG_PATH}
        
    done
    done
    
done
