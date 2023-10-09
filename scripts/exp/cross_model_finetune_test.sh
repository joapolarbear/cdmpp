#!/bin/bash
set -x

DEVICE=t4
DATA_MODE="${DEVICE}:sample200.by_net:test-resnet3d_18"
FILTER_STR=110
DEFAULT_TBLOG_DIR=".workspace/runs/20221119_autotune_trial_1575-fix_batch_first_bug-${DEVICE}"

bash scripts/train.sh run -y \
    --mode "${DEVICE}:sample200" \
    -c tmp/search_trial_20221119_1575.yaml \
    --output_norm_method 0 \
    --filters ${FILTER_STR} \
    -t "${DEFAULT_TBLOG_DIR}" \
    --load_cache \
    --finetune_cache_dir "${DEFAULT_TBLOG_DIR}-cmd" \
    --finetune_cfg tmp/search_trial_20221119_1575-use_cmd.yaml \
    --finetune_datapath "${DATA_MODE}" \
    $@