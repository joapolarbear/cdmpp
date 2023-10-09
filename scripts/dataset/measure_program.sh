#!/bin/bash
<<'COMMENTS'
# Measure the cost of tensor programs
### Some examples
```
bash scripts/dataset/measure_program.sh test_measure cuda v100
```

```
COMMENTS

function usage_prompt {
    echo "Usage: bash xxx.sh [OPTION] ..."

}
if [ -z $1 ]; then
    usage_prompt
fi

export PATH=$PATH:/usr/local/cuda/bin
set -x
PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

option=$1
device=$2
model=$3
remain_arg=${@:4}
echo $remain_arg

python3  $PROJECT_PATH/3rdparty/tenset/scripts/measure_programs.py \
    -o $option --target "${device} --model=${model}" $remain_arg