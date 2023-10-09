#!/bin/bash
set -x

ALL_DEVICES=("v100", "t4", "a100" "p100" "k80")
ALL_DEVICES=("t4")

if [[ $1 == "none" ]]; then
    ### Fix the bug of batch first
    for DEVICE in ${ALL_DEVICES[@]}; do
        bash scripts/train.sh run -y \
            --mode "${DEVICE}:sample200" \
            -c tmp/search_trial_20221119_1575.yaml \
            --output_norm_method 0 \
            --filters 110 \
            -t .workspace/runs/20221119_autotune_trial_1575-fix_batch_first_bug-${DEVICE}
        
        rm -r tmp/dataset-ave_lb_0_0-filters_1_1_0/${DEVICE}:sample200
    done
    exit 0
elif [[ $1 == "mmd" ]]; then
    ### Apply MMD-based regulizer
    for DEVICE in ${ALL_DEVICES[@]}; do
        bash scripts/train.sh run -y \
            --mode "${DEVICE}:sample200" \
            -c tmp/search_trial_20221119_1575-use_cmd.yaml \
            --output_norm_method 0 \
            --filters 110 \
            -t .workspace/runs/20221119_autotune_trial_1575-fix_batch_first_bug-${DEVICE}-mmd \
            $@
        
        rm -r tmp/dataset-ave_lb_0_0-filters_1_1_0/${DEVICE}:sample200
    done
    exit 0
fi

exit 0

######### back up
DEVICE="t4"

# DATA_MODE="${DEVICE}:network-resnet_50_-resnet_18_-inception_v3_.by_net:test-resnet_18_"
DATA_MODE="${DEVICE}:sample200.by_net:test-resnet3d_18"
# DATA_MODE="${DEVICE}:sample200.by_net:test-bert_medium"
# DATA_MODE="${DEVICE}:sample200.by_net:train-all,test-resnet_18"

USE_GRAD_CLIP=0
FILTER_STR=110

if [ ${FILTER_STR} == "111" ]; then 
    FILTER_SUFFIX="-rm_outlier"
else
    FILTER_SUFFIX=""
fi
if [ ${USE_GRAD_CLIP} == '1' ]; then
    DEFAULT_TBLOG_DIR=".workspace/runs/cmpp-${DEVICE}${FILTER_SUFFIX}-grad_clip2"
else
    DEFAULT_TBLOG_DIR=".workspace/runs/cmpp-${DEVICE}${FILTER_SUFFIX}2"
fi

# bash scripts/train.sh run -y \
#     --mode "${DATA_MODE}" \
#     -c tmp/search_trial_20221119_1575.yaml \
#     --output_norm_method 0 \
#     --filters ${FILTER_STR} \
#     -t "${DEFAULT_TBLOG_DIR}-same_domain" \
#     $@

### Fix the "mode" format, without cmd
bash scripts/train.sh run -y \
    --mode "${DATA_MODE}" \
    -c tmp/search_trial_20221119_1575.yaml \
    --output_norm_method 0 \
    --filters ${FILTER_STR} \
    -t "${DEFAULT_TBLOG_DIR}" \
    $@

### Fix the "mode" format, with cmd
# bash scripts/train.sh run -y \
#     --mode "${DATA_MODE}" \
#     -c tmp/search_trial_20221119_1575-use_cmd.yaml \
#     --output_norm_method 0 \
#     --filters ${FILTER_STR} \
#     -t "${DEFAULT_TBLOG_DIR}-cmd" \
#     $@

### fine-tuning, do not train from scratch
# bash scripts/train.sh run -y \
#     --mode "${DATA_MODE}" \
#     -c tmp/search_trial_20221119_1575.yaml \
#     --output_norm_method 0 \
#     --filters ${FILTER_STR} \
#     -t "${DEFAULT_TBLOG_DIR}" \
#     --load_cache \
#     --finetune_cache_dir "${DEFAULT_TBLOG_DIR}-cmd" \
#     --finetune_cfg tmp/search_trial_20221119_1575-use_cmd.yaml \
#     --finetune_datapath "mode:${DATA_MODE}" \
#     $@

### Fix the "mode" format, with mmd
# bash scripts/train.sh run -y \
#     --mode "${DATA_MODE}" \
#     -c tmp/search_trial_20221119_1575-use_cmd.yaml \
#     --output_norm_method 0 \
#     --filters ${FILTER_STR} \
#     -t "${DEFAULT_TBLOG_DIR}-mmd" \
#     $@

############################## Train cost model for different leaf node no
DEVICE=t4
LEAF_NODE_NO=3
bash scripts/train.sh run -y \
    --mode "${DEVICE}:sample2308" \
    -c tmp/search_trial_20221119_1575.yaml \
    --output_norm_method 0 --filters 110 \
    -t .workspace/runs/cmpp-${DEVICE}-N_leaf_${LEAF_NODE_NO} \
    --leaf_node_no ${LEAF_NODE_NO}



LEAF_NODE_NOS=("4" "6" "11" "5")
for LEAF_NODE_NO in ${LEAF_NODE_NOS[@]}; do
    DEVICE=t4
    pretrained_model_path=.workspace/runs/20221119_autotune_trial_1575-fix_batch_first_bug-${DEVICE}
    finetune_cache_dir=${pretrained_model_path}-N_leaf_${LEAF_NODE_NO}
    mode=${DEVICE}:sample512
    echo $finetune_cache_dir
    echo ${LEAF_NODE_NO}
    nohup bash scripts/train.sh run -y \
        --mode $mode \
        -c tmp/search_trial_20221119_1575.yaml \
        -t ${pretrained_model_path} --load_cache \
        --finetune_cache_dir ${finetune_cache_dir} \
        --finetune_datapath ${mode} \
        --leaf_node_no ${LEAF_NODE_NO} >> log.txt 2>&1 

    rm -rf ${pretrained_model_path}/finetune_dataset-ave_lb_0_0-filters_1_1_0
done