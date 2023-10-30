#!/bin/bash
# set -x

SAMPLE_NUM=2308
SAMPLE_NUM=200
EXP_TIMES=1

RST_DIR=.workspace/cross_model_finetune
mkdir -p ${RST_DIR}/logs
mkdir -p ${RST_DIR}/cost_models

ALL_DEVICES=("t4" "k80"  "a100" "p100" "v100")
NETWORKS=("resnet_18" "mobilenet_v2" "bert_tiny")

ALL_DEVICES=("t4")
NETWORKS=("resnet_18" "mobilenet_v2" "bert_tiny")
DOMAIN_DIFF_METRIC="none"

LEAF_NODE_NO=5
FILTER_STR=110
# DISABLE_PE_STR="--disable_pe"
DISABLE_PE_STR=""
OUTPUT_NORM_METHOD=0

for DEVICE in ${ALL_DEVICES[@]}; do
    PREPROCESS_DATA_DIR=tmp/dataset-ave_lb_0_0-filters${FILTER_STR}
    if [[ ${DISABLE_PE_STR} == "--disable_pe" ]]; then
        PREPROCESS_DATA_DIR=${PREPROCESS_DATA_DIR}-not_use_pe
    fi
    PREPROCESS_DATA_DIR=${PREPROCESS_DATA_DIR}/$DATA_MODE
    if [[ ${LEAF_NODE_NO} != 5 ]]; then
        PREPROCESS_DATA_DIR=${PREPROCESS_DATA_DIR}-leaf_node_no_${LEAF_NODE_NO}
    fi
    echo "Preprocessed data dir: ${PREPROCESS_DATA_DIR}"

    pretrained_model_path=.workspace/runs/20221119_autotune_trial_1575-fix_batch_first_bug-${DEVICE}
    if [[ ! -d ${pretrained_model_path} ]]; then
        echo "pretrained_model_path: ${pretrained_model_path} does not exists"
        exit 0
    fi
    
    for NETWORK in ${NETWORKS[@]}; do
    echo "NETWORK: ${NETWORK}"
    for (( i=1; i<=${EXP_TIMES}; i++ )); do
        if [[ ${DOMAIN_DIFF_METRIC} == "none" ]]; then
            DOMAIN_DIFF_ARG=""
            LOG_PATH=${RST_DIR}/logs/${DEVICE}_${i}_${SAMPLE_NUM}_${NETWORK}.txt
            TB_DIR=${RST_DIR}/cost_models/${DEVICE}_${i}_${SAMPLE_NUM}_${NETWORK}
        else
            LOG_PATH=${RST_DIR}/logs/${DEVICE}_${i}_${SAMPLE_NUM}_${NETWORK}_${DOMAIN_DIFF_METRIC}.txt
            TB_DIR=${RST_DIR}/cost_models/${DEVICE}_${i}_${SAMPLE_NUM}_${NETWORK}_${DOMAIN_DIFF_METRIC}
            DOMAIN_DIFF_ARG="--domain_diff_metric ${DOMAIN_DIFF_METRIC}"
        fi
        
        DATA_MODE=${DEVICE}:sample200.by_net:test-${NETWORK}_bs1
        echo "mode: ${DATA_MODE}"
        echo "LOG_PATH: ${LOG_PATH}"
        echo "TB_DIR: ${TB_DIR}"

        ### Training
        # bash scripts/train.sh run -y \
        #     --mode "${DATA_MODE}" \
        #     -c tmp/search_trial_20221119_1575.yaml \
        #     --output_norm_method ${OUTPUT_NORM_METHOD} \
        #     --filters ${FILTER_STR} ${DISABLE_PE_STR} \
        #     --leaf_node_no ${LEAF_NODE_NO} \
        #     ${DOMAIN_DIFF_ARG} \
        #     -t ${pretrained_model_path} --load_cache \
        #     --finetune_cache_dir ${TB_DIR} \
        #     --finetune_datapath ${DATA_MODE} 2>&1 | tee -a ${LOG_PATH}
        
        ### Analyze the resutls
        # bash scripts/train.sh analyze -y \
        #     --mode "${DATA_MODE}" \
        #     -c tmp/search_trial_20221119_1575.yaml \
        #     --output_norm_method ${OUTPUT_NORM_METHOD} \
        #     --filters ${FILTER_STR} ${DISABLE_PE_STR} \
        #     --leaf_node_no ${LEAF_NODE_NO} \
        #     --cache_dir ${TB_DIR}/cm/BaseLearner/best \
        #     2>&1 | tee -a ${LOG_PATH}
        
        if [[ ${DOMAIN_DIFF_METRIC} == "none" ]]; then
            python3 scripts/exp/cmd2error.py "cm_finetune" \
                ${TB_DIR}/cm/BaseLearner/best \
                ${RST_DIR}/${DEVICE}_${i}_${SAMPLE_NUM}_${NETWORK}_dist_visual.pdf
        else
            python3 scripts/exp/cmd2error.py "cm_finetune" \
                ${TB_DIR}/cm/BaseLearner/best \
                ${RST_DIR}/${DEVICE}_${i}_${SAMPLE_NUM}_${NETWORK}_${DOMAIN_DIFF_METRIC}_dist_visual.pdf
        fi
    done
    done
    # rm -r ${PREPROCESS_DATA_DIR}
done