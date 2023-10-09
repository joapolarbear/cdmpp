#!/usr/bin/env bash
TRACE_PATH="/home/habana/hphu/traces"
mkdir -p ${TRACE_PATH}

export HABANA_PROFILE=1 
MODEL=$1
FRONTEND=$2 # Example: TF_1_5, ONNX


IFS=' ' read -r -a DEVICE_MODEL_NAME <<< $(hl-smi -Q name --format=csv,noheader | tr " " _)
DEVICE_MODEL_NAME=${DEVICE_MODEL_NAME[0]}
echo "------------------------------------------------------------------------------------------"
echo "GPU Model: $DEVICE_MODEL_NAME, DNN Model: $MODEL"
echo "------------------------------------------------------------------------------------------"

function func_run_resnet50 {
    # echo "func_run_resnet50"
    cd /home/habana/habanalabs/demos/demos/goya/resnet50_tf_perf
    python3 demo_resnet50_tf_perf_profile.py $batch_size >& stdout.log
}

function func_run_resnet18 {
    # echo "func_run_resnet50"
    cd /home/habana/hphu/resnet_18_example
    python3 demo.py $batch_size >& stdout.log
    mv host_profiling*.json default_profiling.json
}


function func_run_bert_base {
    # echo "func_run_bert_base"
    cd /home/habana/habanalabs/demos/demos/goya/BERT/auto_recipe_gen_tool
    #!/bin/bash

    export PYTHONPATH=$PYTHONPATH:$PWD
    export RECIPE_RUNNER_PATH=$(cd ../../recipe_runner && pwd)
    export REPEAT_EXECUTION_HABANA_BERT=False

    GRAPH=$SOFTWARE_DATA/mxnet/models/BERT/auto_recipe_gen/SQUAD/squad_frozen_graph.pb
    BATCH_SIZE=$batch_size
    MAX_SEQ_LEN=128
    RUN='build'
    RANDOM_DATA=1
    RECIPE_NAME=bert.recipe
    TEST='test'
    INPUTS="input_ids_1","segment_ids_1","input_mask_1"
    OUTPUTS="unstack"

    # cd SQUAD
    ENABLE_RAGGED_SOFTMAX_OPT=1 python SQUAD/demo-squad.py --frozen_graph $GRAPH --run $RUN --batch_size $BATCH_SIZE --max_seq_len $MAX_SEQ_LEN --random_data $RANDOM_DATA --recipe_name $RECIPE_NAME --input_names $INPUTS --output_names $OUTPUTS >& stdout.log
    # python SQUAD/demo-squad.py --run $TEST --recipe_name $RECIPE_NAME --batch_size $BATCH_SIZE --input_names $INPUTS --output_names $OUTPUTS
    # cd ..

    mv *default_profiling.json default_profiling.json && rm *default_profiling_host.json

}


DTYPE_TO_TRY=(auto)

if [ "$MODEL" = "ResNet50" ]; then
    FUNC_RUN_DNN_MODEL=func_run_resnet50
    BS_TO_TRY=(1 2 4 8 16 32 64 128 256 512 1024)
elif [ "$MODEL" = "BERT-Base" ]; then
    FUNC_RUN_DNN_MODEL=func_run_bert_base
    BS_TO_TRY=(1 2 4 8 16 32 64 128 256)
elif [ "$MODEL" = "ResNet18" ]; then
    FUNC_RUN_DNN_MODEL=func_run_resnet18
    BS_TO_TRY=(1 2 4 8 16 32 64 128 256 512 1024)
else
    echo "Invalid DNN Model: $MODEL"
    echo "Invalid DNN Model: $MODEL"
    echo "Currently supported models: ResNet50, BERT-Base, ResNet18"
    exit 0
fi

for(( id=0; id < "${#DTYPE_TO_TRY[@]}"; id++ ))
do
    DTYPE=${DTYPE_TO_TRY[$id]}
    for(( id_=0; id_ < "${#BS_TO_TRY[@]}"; id_++ ))
    do
        batch_size=${BS_TO_TRY[$id_]}

        echo "DTYPE=$DTYPE, BS=$batch_size"
        $FUNC_RUN_DNN_MODEL

        CUR_TRACE_PATH=$TRACE_PATH/${FRONTEND}_$MODEL/$DEVICE_MODEL_NAME/dtype_$DTYPE/BS_$batch_size/habana_profiler
        if [ ! -s $CUR_TRACE_PATH ]; then
            mkdir -p $CUR_TRACE_PATH
        fi
        mv stdout.log *.json $CUR_TRACE_PATH/
    done
done

