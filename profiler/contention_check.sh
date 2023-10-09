#!/bin/bash
set -e

if [ "$#" -ge 3 ]; then
    TARGET_OP="$1"
    export FINAL_TRACE_DIR=`realpath $2`
    export OP_CFG_CACHE_PATH=`realpath $3`
else
    echo "Usage: bash xxx.sh <target_op> <final_trace_dir> <cfg_path>"
    exit
fi

export PROJECT_DIR=$(dirname $0)
### We only focus on op level traces, when checking the op contention
export PROFILE_NSYS=0
export PROFILE_CUTLASS=0
bash $PROJECT_DIR/profile_op_nsys_cutlass.sh $TARGET_OP