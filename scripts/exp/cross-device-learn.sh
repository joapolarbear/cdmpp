#!/bin/bash

set -x
SAMPLE_NUM=2308

################## Single-device to single-device prediction ##################
SOURCE_DEVICE=$1
TARGET_DEVICE=$2
# SOURCE_DEVICE=p100
# TARGET_DEVICE=t4

# bash scripts/train.sh make_dataset_run -y \
#     --mode "(p100(train):sample${SAMPLE_NUM}),(t4(test):sample${SAMPLE_NUM},network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8)" \
#     -c tmp/search_trial_20221119_1575.yaml \
#     --output_norm_method 0 --filters 110 --load_cache \
#     -t .workspace/runs/20221119_autotune_trial_1575-y_norm_0-t4-keep_outliers \
#     --finetune_cache_dir .workspace/runs/cdpp_p100_to_t4

bash scripts/train.sh make_dataset_run -y \
    --mode "(${SOURCE_DEVICE}(train):sample${SAMPLE_NUM},network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8),(${TARGET_DEVICE}(test):sample${SAMPLE_NUM},network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8)" \
    -c tmp/search_trial_20221119_1575-use_cmd.yaml \
    --output_norm_method 0 --filters 110 --load_cache \
    -t .workspace/runs/20221119_autotune_trial_1575-y_norm_0-${SOURCE_DEVICE}-keep_outliers \
    --finetune_cache_dir .workspace/runs/cdpp_${SOURCE_DEVICE}_to_${TARGET_DEVICE}

################## Cross-device pre-training and finetuning to single-device prediction ##################

device=e5-2673
pretrained_model_path=.workspace/runs/cdpp_to_${device}-pretrain
mode=""
ALL_DEVICES=(a100 v100 p100 k80 t4 e5-2673 epyc-7452 graviton2 platinum-8272)
for train_device in ${ALL_DEVICES[@]}; do
    if [[ $train_device != $device ]]; then
        if [[ -z $mode ]]; then
            mode="${train_device}:sample${SAMPLE_NUM}"
        else
            mode="${mode},${train_device}:sample${SAMPLE_NUM}"
        fi
    fi
done
echo $mode
echo $pretrained_model_path
bash scripts/train.sh run -y \
    --mode $mode \
    -c tmp/search_trial_20221119_1575.yaml \
    -t ${pretrained_model_path} 

### Fine-tuning
# device=graviton2
# pretrained_model_path=.workspace/runs/cdpp_to_${device}-pretrain-fix_batch_first_bug
finetune_cache_dir=${pretrained_model_path}-finetune_${device}
mode=${device}:sample${SAMPLE_NUM}
echo $mode
echo $pretrained_model_path
echo $finetune_cache_dir

bash scripts/train.sh run -y \
    --mode sample10 \
    -c tmp/search_trial_20221119_1575.yaml \
    -t ${pretrained_model_path} --load_cache \
    --finetune_cache_dir ${finetune_cache_dir} \
    --finetune_datapath ${mode}

### Analyze latent representation
mkdir -p .workspace/cross_device
pretrained_model_path=".workspace/runs/cdpp_to_epyc-7452-pretrain-finetune_epyc-7452"
# pretrained_model_path=".workspace/runs/cdpp_to_epyc-7452-pretrain"
ALL_DEVICES=(a100 t4 epyc-7452)
for device in ${ALL_DEVICES[@]}; do
    bash scripts/train.sh analyze -y \
        --mode "${device}:sample${SAMPLE_NUM}" \
        -c tmp/search_trial_20221119_1575.yaml \
        --cache_dir ${pretrained_model_path}/cm/BaseLearner/

    mv ${pretrained_model_path}/cm/BaseLearner/training_latent.pickle .workspace/cross_device/train_latent_${device}.pickle
    mv ${pretrained_model_path}/cm/BaseLearner/test_latent.pickle .workspace/cross_device/test_latent_${device}.pickle
done

python3 scripts/exp/cmd2error.py cd_finetune \
    .workspace/cross_device \
    .workspace/cross_device/dist_visual.pdf

