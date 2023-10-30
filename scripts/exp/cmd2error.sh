#!/bin/bash
<<'COMMENTS'
Scripts to dump training and test result and 
analyze the relationship between CMD/MMD and MAPE
```
COMMENTS

PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH
export PATH=$PATH:/usr/local/cuda/bin
export CUBLAS_WORKSPACE_CONFIG=:4096:8
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

mkdir -p .workspace/verify

SAMPLE_NUM=2308
remain_arg=${@}

### Cache the training and test set
bash scripts/train.sh analyze -y \
    --mode sample${SAMPLE_NUM},network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8 \
    --output_norm_method 0 --filters 110 \
    -c tmp/search_trial_20221119_1575.yaml \
    --cache_dir tmp/cm/20221119_autotune_trial_1575-y_norm_0-t4-keep_outliers/cm/BaseLearner

bash scripts/train.sh analyze -y \
    --mode "(t4(train):sample${SAMPLE_NUM},network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8),(p100(test):sample200,network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8)" \
    -c tmp/search_trial_20221119_1575.yaml \
    --output_norm_method 0 --filters 110 \
    --cache_dir tmp/cm/cdpp_t4_to_p100/cm/BaseLearner/best

# Then analyze the results
python3 scripts/exp/cmd2error.py "cmd2error" tmp/20221119_autotune_trial_1575-y_norm_0-t4-keep_outliers/cm/BaseLearner

