#!/bin/bash
# DEVICES=('k80' 't4' 'a100' 'p100' 'v100')
SAMPLE_NUM=2308
SAMPLE_NUM=200
DEVICES=('t4')
for DEVICE in "${DEVICES[@]}"; do
    bash scripts/dataset/gen_raw_feature.sh ast_ansor ${DEVICE} --ed ${SAMPLE_NUM} -j 1
done
# bash scripts/dataset/make_dataset.sh -y --mode sample${SAMPLE_NUM} -c tmp/search_trial_20221119_1575.yaml