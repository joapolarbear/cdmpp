#!/bin/bash
SAMPLE_NUM=2308
SAMPLE_NUM=200
### origin and disable CMD
bash scripts/train.sh make_dataset_run -y \
    --mode sample${SAMPLE_NUM},network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8 \
    -c tmp/search_trial_20221119_1575.yaml \
    --output_norm_method 0 \
    -t .workspace/runs/20221119_autotune_trial_1575-ours_minus_cmd

### + disable Box-Cox Transformation
bash scripts/train.sh make_dataset_run -y \
    --mode sample${SAMPLE_NUM},network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8 \
    -c tmp/search_trial_20221119_1575.yaml \
    --output_norm_method 7 \
    -t .workspace/runs/20221119_autotune_trial_1575-ours_minus_cmd_box-cox

### + disable Hybrid Loss
bash scripts/train.sh make_dataset_run -y \
    --mode sample${SAMPLE_NUM},network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8 \
    -c tmp/search_trial_20221119_1575-ours_minus_cmd_box-cox_hybrid.yaml \
    --output_norm_method 7 \
    -t .workspace/runs/20221119_autotune_trial_1575-ours_minus_cmd_box-cox_hybrid

### + disable PE
bash scripts/train.sh make_dataset_run -y \
    --mode sample${SAMPLE_NUM},network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8 \
    -c tmp/search_trial_20221119_1575-ours_minus_cmd_box-cox_hybrid.yaml \
    --output_norm_method 7 \
    -t .workspace/runs/20221119_autotune_trial_1575-ours_minus_cmd_box-cox_hybrid_pe \
    --disable_pe