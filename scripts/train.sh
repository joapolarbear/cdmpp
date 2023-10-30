#!/bin/bash
<<'COMMENTS'
# Scripts to train, tune and analyze
### Some examples
```
bash scripts/train.sh tune --tune_method gen_cfg
```

### Training
```
bash scripts/train.sh run \
    --mode sample2308 \
    -i .workspace/ast_ansor  --tb_logdir .workspace/runs/autotune_trial_5_fine_tune --ave_lb 0 --output_norm_method log  -y

bash scripts/train.sh run \
    --mode sample2308,network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8 --ave_lb 0
```
Or you can add `--load_cache` to load cached cost model

### Tune
```
bash scripts/train.sh tune -y \
    --tune_method optuna \                               
    --mode sample2308 --output_norm_method log --ave_lb 0
```

### Check the tuning results
```
bash scripts/train.sh tune -y \
    --tune_method optuna,monitor,ast_ansor_sample2308,network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8_mo \
    -i .workspace/ast_ansor\
    --mode sample2308 > log.txt
```
### analyze
```
bash scripts/train.sh analyze --mode sample2308 -i .workspace/ast_ansor --cache_dir test_cm --output_norm_method log --ave_lb 0 -y
```
COMMENTS

function usage_prompt {
    echo "Usage: bash xxx.sh [OPTION] ..."
    echo "Launch a training process"
    echo
    echo "  - run,                       Training from the scratch"
    echo "  - load,                      Training based on previous ckpt"
    echo "  - tune,                      Run an autotune process"
    echo 
    exit
}
if [ -z $1 ]; then
    usage_prompt
fi

if [[ ! -z ${CDPP_DEBUG} && ${CDPP_DEBUG} == '1' ]]; then
    DEBUG_STR="-m pdb"
else
    DEBUG_STR=""
fi

PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH
export PATH=$PATH:/usr/local/cuda/bin
export CUBLAS_WORKSPACE_CONFIG=:4096:8
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

mkdir -p .workspace
# if [ ! -d ".workspace/tenset" ]; then
#     cd .workspace
#     hdfs dfs -get /usr/hphu/0tmp/tenset.zip && unzip tenset && rm tenset.zip
#     hdfs dfs -get /usr/hphu/0tmp/None-norm-info.pickle && mv None-norm-info.pickle tenset/
#     cd .. 
# fi
mkdir -p .workspace/autotune

SAMPLE_NUM=2308
remain_arg=${@:2}

### ------ Start Training
if [[ $1 =~ ^run.* || $1 =~ ^train.* ]]; then
    # rm -rf .workspace/runs
    python3 ${DEBUG_STR} main.py \
        --log_level info \
        --force_load_data \
        --source_data tir \
        -o $1 \
        -i .workspace/ast_ansor $remain_arg
elif [[ $1 =~ ^make_dataset_run.* ]]; then
    python3 main.py \
        --source_data tir \
        -o make_dataset \
        --mode sample${SAMPLE_NUM} \
        -i .workspace/ast_ansor $remain_arg
        
    python3 ${DEBUG_STR} main.py \
        --log_level info \
        --force_load_data \
        --source_data tir \
        -o $1 \
        -i .workspace/ast_ansor $remain_arg
elif [[ $1 =~ ^metadata.* ]]; then
    python3 main.py \
        --source_data tir \
        -o metadata \
        --mode sample${SAMPLE_NUM} \
        -i .workspace/ast_ansor $remain_arg
elif [[ $1 =~ ^analyze.* ]]; then
    python3 ${DEBUG_STR} main.py \
        --log_level info \
        --source_data tir \
        --force_load_data \
        -o $1 \
        -i .workspace/ast_ansor $remain_arg
elif [[ $1 =~ ^sche_search.* ]]; then
    python3 ${DEBUG_STR} main.py \
        --log_level info \
        --source_data tir \
        --force_load_data \
        -o $1 \
        -i .workspace/ast_ansor $remain_arg
# elif [ $1 =~ ^tune.* ]; then
#     # rm -rf .workspace/runs
#     python3 ${DEBUG_STR} main.py \
#         --log_level info \
#         --source_data tir \
#         --force_load_data \
#         -o $1 \
#         -i .workspace/ast_ansor $remain_arg
# elif [ $1 =~ ^test.* ]; then
#     python3 ${DEBUG_STR} main.py \
#         --log_level info \
#         --source_data tir \
#         --force_load_data \
#         -o $1 \
#         -i .workspace/ast_ansor $remain_arg
# elif [ $1 =~ ^finetune.* ]; then
#     # rm -rf .workspace/runs
#     remain_arg=${@:3}
#     python3 ${DEBUG_STR} main.py \
#         --log_level info \
#         --source_data tir \
#         --force_load_data \
#         -o $1 -i .workspace/ast_ansor \
#         -w $2 \
#         --tune_method fine_tune \
#         $remain_arg
else
    usage_prompt
fi

# bash scripts/train.sh analyze
# bash scripts/train.sh run
# bash scripts/train.sh run --debug
# bash scripts/train.sh run --mode single
# bash scripts/train.sh run --mode single --metric_learner 3
# bash scripts/train.sh tune --tune_method bo
# bash scripts/train.sh tune --tune_method bo --mode single
# bash scripts/train.sh tune --tune_method bo --mode single --metric_learner 3
# bash scripts/train.sh finetune .workspace/autotune
# bash scripts/train.sh finetune .workspace/autotune --mode single