#!/bin/bash
<<'COMMENTS'
# Scripts to train, tune and analyze
### Some examples

### Replay
```
bash scripts/replay.sh -y \
    --cache_dir .workspace/runs/search_trial_17_no_pe_fix/cm/BaseLearner \
    --replay_mode replay
```

### Measure each task in a network
```
--replay_mode measure_by_task
```

### Measure a network
```
... --replay_mode measure
```

### Replay via profile
```
... --replay_mode replay_via_profile
```

### Decide schedules for the target network
```
... --replay_mode prepare
```

COMMENTS

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

remain_arg=${@:1}

python3 ${DEBUG_STR} main.py \
        --log_level info \
        --source_data tir \
        -o replay \
        -i .workspace/ast_ansor $remain_arg
