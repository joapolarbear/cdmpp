#!/bin/bash
set -e
if [ "$#" -ge 1 ]; then
    TARGET_OP="$1"
else
    TARGET_OP="MatMul"
fi
export PROJECT_DIR=$(dirname $0)
export PYTHONPATH="${PYTHONPATH}:`realpath $PROJECT_DIR/..`"

GPU_MODEL_NAME=($(CUDA_VISIBLE_DEVICES=0 nvidia-smi --query-gpu=name --format=csv,noheader | tr " " _))
GPU_MODEL_NAME=${GPU_MODEL_NAME[0]}
export FINAL_TRACE_DIR=${FINAL_TRACE_DIR:-"/home/tiger/op_nsys_cutlass"}
export GPU_TRACE_DIR=$FINAL_TRACE_DIR/$GPU_MODEL_NAME
export OP_CFG_CACHE_PATH=${OP_CFG_CACHE_PATH:-$GPU_TRACE_DIR/${TARGET_OP}_cfgs.txt}
export PROFILE_NSYS=${PROFILE_NSYS:-1}
export PROFILE_CUTLASS=${PROFILE_CUTLASS:-1}

export LINE_NUMBER=`wc -l < $OP_CFG_CACHE_PATH`
export OP_TRACE_PATH=$GPU_TRACE_DIR/${TARGET_OP}_op.txt
export NSYS_TRACE_PATH=$GPU_TRACE_DIR/${TARGET_OP}_nsys.txt
export CUTLASS_TRACE_PATH=$GPU_TRACE_DIR/${TARGET_OP}_cutlass.txt
export NSYS_PROFILE_PATH=$HOME/nsys.qdrep
export NSYS_PROFILE_STDOUT=$HOME/nsys.log
echo "GPU Model: $GPU_MODEL_NAME, OP_TYPE: $TARGET_OP"
echo "Use profile cfg file:    $OP_CFG_CACHE_PATH ($LINE_NUMBER lines)"
echo "Store OP traces at:      $OP_TRACE_PATH"
echo "Store NSYS traces at:    $NSYS_TRACE_PATH"
echo "Store CUTLASS traces at: $CUTLASS_TRACE_PATH"
if [ ! -d  $GPU_TRACE_DIR ]; then
    mkdir -p $GPU_TRACE_DIR
fi
if [ -f  $OP_TRACE_PATH ]; then rm $OP_TRACE_PATH; fi
if [ -f  $NSYS_TRACE_PATH ]; then rm $NSYS_TRACE_PATH; fi
if [ -f  $CUTLASS_TRACE_PATH ]; then rm $CUTLASS_TRACE_PATH; fi

### Generate Configs
if [ ! -f $OP_CFG_CACHE_PATH ]; then
    echo "Generate config file at $OP_CFG_CACHE_PATH ..."
    python3 $PROJECT_DIR/profile_op/gen_data.py \
        --op $TARGET_OP \
        --cfg_cache_path $OP_CFG_CACHE_PATH
else
    echo "Config file $OP_CFG_CACHE_PATH already exists"
fi
cp $OP_CFG_CACHE_PATH $GPU_TRACE_DIR

function enable_nsys {
    CMD_PREFIX=" nsys profile --trace=cuda,nvtx,cublas,openmp,osrt,cudnn -c cudaProfilerApi --stop-on-range-end true -f true -o $NSYS_PROFILE_PATH"
    CMD_SUFFIX=" --nsys "
}

function disable_nsys {
    CMD_PREFIX=""
    CMD_SUFFIX=""
}

id=1
while IFS= read -r line; do
    _time=`date`
    echo "$_time: $id/$LINE_NUMBER"
    ### Profile OP
    disable_nsys
    $CMD_PREFIX python3 $PROJECT_DIR/profile_op/gen_data.py \
        --op $TARGET_OP \
        --cfg_str "$line" \
        --cache_path $OP_TRACE_PATH \
        $CMD_SUFFIX

    if [ "${PROFILE_NSYS}" = "1" ]; then
        ### Nsys
        enable_nsys
        $CMD_PREFIX python3 $PROJECT_DIR/profile_op/gen_data.py \
            --op $TARGET_OP \
            --cfg_str "$line" \
            $CMD_SUFFIX
        nsys stats --report gpusum,gpumemsizesum,cudaapisum \
            --format csv $NSYS_PROFILE_PATH > $NSYS_PROFILE_STDOUT
        python3 $PROJECT_DIR/profile_nsys/nsys_parser.py \
            $NSYS_PROFILE_STDOUT \
            --ops $TARGET_OP \
            --save_path $NSYS_TRACE_PATH
    fi

    if [ "${PROFILE_CUTLASS}" = "1" ]; then
        ### CUTLASS
        python3 $PROJECT_DIR/cutlass/cutlass_profiler.py \
            --cfg_str "$line" \
            --ops $TARGET_OP \
            --target_gpu $GPU_MODEL_NAME \
            --cache_path $CUTLASS_TRACE_PATH
    fi

    id=$(($id+1))
done < <(cat $OP_CFG_CACHE_PATH)