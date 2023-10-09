#!/usr/bin/env bash
### The launcher is similar to that of perf.git in the gitlab
#   which runs ResNet50 and BERT Large with different batch sizes
#   collect op level traces using TensorFlow Profiler
#   collect kernel level traces using NSYS

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TRACE_PATH="$HOME/traces"
mkdir -p ${TRACE_PATH}

MODEL=$1
PROFILE_MODE=$2
DTYPE_TO_TRY=(fp16 fp32)

if [ "$MODEL" = "ResNet50" ]; then
    BS_TO_TRY=(1 2 4 8 16 32 64 128 256 512 1024)
elif [ "$MODEL" = "BERT-Large" ]; then
    BS_TO_TRY=(1 2 4 8 16 32 64 128 256)
else
    echo "Invalid DNN Model: $MODEL"
    echo "Currently supported models: ResNet50, BERT-Large"
    exit 0
fi

if [ "$PROFILE_MODE" = "nsys" ]; then
    PROFILE_CMD="--enable_nsys True"
elif [ "$PROFILE_MODE" = "tf" ]; then
    PROFILE_CMD=" "
else
    echo "Invalid PROFILE_MODE: $PROFILE_MODE"
    echo "Currently supported profile modes: nsys, tf"
    exit 0
fi

GPU_MODEL_NAME=($(CUDA_VISIBLE_DEVICES=0 nvidia-smi --query-gpu=name --format=csv,noheader | tr " " _))
GPU_MODEL_NAME=${GPU_MODEL_NAME[0]}
echo "------------------------------------------------------------------------------------------"
echo "GPU Model: $GPU_MODEL_NAME, DNN Model: $MODEL, Profile Mode: $PROFILE_MODE"
echo "------------------------------------------------------------------------------------------"

function func_trace_name {
    TRACE_NAME="nsys"
    NSYS_PROFILE_PATH=${TRACE_NAME##*/}".qdrep"
    NSYS_PROFILE_STDOUT=${TRACE_NAME##*/}".log"
    NSYS_PROFILE_SQLITE=${TRACE_NAME##*/}".sqlite"
    NSYS_JSON_FILE=${TRACE_NAME##*/}".json"
}

function func_upload_rst {
    FINAL_TRACE_DIR="$HOME/final_traces"
    CUR_TRACE_DIR=${FINAL_TRACE_DIR}/traces/TF_2_4_$MODEL/$GPU_MODEL_NAME/dtype_$DTYPE/BS_$batch_size/$PROFILE_MODE
    if [ ! -s $CUR_TRACE_DIR ]; then
        mkdir -p $CUR_TRACE_DIR
    fi
    mv "$TRACE_PATH"/* ${CUR_TRACE_DIR}/
}

function func_run_resnet50 {
    # echo "func_run_resnet50"

    cd $CWD/image_classification/tensorflow2
    if [ "$PROFILE_MODE" = "nsys" ]; then
        PRE_CMD=" nsys profile --trace=cuda,nvtx,cublas,openmp,osrt,cudnn -c cudaProfilerApi --stop-on-range-end true -f true -o $NSYS_PROFILE_PATH"
    else
        PRE_CMD=""
    fi
    TF_GPU_THREAD_MODE=gpu_private $PRE_CMD python3 resnet_ctl_imagenet_main.py \
        --worker_hosts=$YOU_KNOW_WHO_WORKER_HOSTS \
        --task_index=$YOU_KNOW_WHO_ID \
        --base_learning_rate=9.5 --lr_schedule=polynomial --weight_decay=0.0002 --num_accumulation_steps=1 \
        --distribution_strategy=mirrored --label_smoothing=0.1 \
        --optimizer=LARS --tf_gpu_thread_mode=gpu_private --all_reduce_alg=nccl \
        --datasets_num_private_threads=96 --enable_device_warmup --enable_eager \
        --epochs_between_evals=4 --eval_dataset_cache --eval_offset_epochs=2 --eval_prefetch_batchs=192 --skip_eval True \
        --num_gpus=1 --model_dir=$TRACE_PATH \
        --noreport_accuracy_metrics --single_l2_loss_op \
        --training_dataset_cache --training_prefetch_batchs=32 --log_steps=10 --verbosity=0 \
        --batch_size=$batch_size --dtype=$DTYPE --use_synthetic_data True \
        --steps_per_loop=10 --train_epochs=1 --warmup_epochs=1 --train_steps 100 \
        --profile_steps 50,60 $PROFILE_CMD \
        >& stdout.log
}

function func_run_bert_large {
    # echo "func_run_bert_large"
    
    cd $CWD/language_model/tensorflow2/bert
    if [ "$PROFILE_MODE" = "nsys" ]; then
        PRE_CMD=" nsys profile --trace=cuda,nvtx,cublas,openmp,osrt,cudnn -c cudaProfilerApi --stop-on-range-end true -f true -o $NSYS_PROFILE_PATH"
    else
        PRE_CMD=""
    fi
    TF_GPU_THREAD_MODE=gpu_private $PRE_CMD python3 run_pretraining.py \
        --worker_hosts=$YOU_KNOW_WHO_WORKER_HOSTS \
        --task_index=$YOU_KNOW_WHO_ID \
        --bert_config_file=hdfs://xxx/bert/uncased_L-24_H-1024_A-16.json \
        --init_checkpoint=hdfs://xxx/bert/model.ckpt-28252 \
        --max_predictions_per_seq=76 --max_seq_length=512 \
        --all_reduce_alg=nccl --beta_1=0.91063 --beta_2=0.96497 --device_warmup \
        --learning_rate=0.00035221 --loss_scale=dynamic --optimizer_type=lamb --verbosity=0 \
        --num_gpus=1 --model_dir=$TRACE_PATH \
        --scale_loss --num_accumulation_steps=1 --steps_before_eval_start=3948 \
        --use_synthetic_data True \
        --num_steps_per_epoch=8000 --num_train_epochs=1 --steps_per_loop=10 --stop_steps=100  --warmup_steps=10 \
        --dtype=$DTYPE --train_batch_size=$batch_size --eval_batch_size=$batch_size \
        --profile_steps 50,60 $PROFILE_CMD \
        >& stdout.log
}

if [ "$MODEL" = "ResNet50" ]; then
    FUNC_RUN_DNN_MODEL=func_run_resnet50
elif [ "$MODEL" = "BERT-Large" ]; then
    FUNC_RUN_DNN_MODEL=func_run_bert_large
else
    echo "Invalid DNN Model: $MODEL"
    exit 0
fi

for(( id=0; id < "${#DTYPE_TO_TRY[@]}"; id++ ))
do
    DTYPE=${DTYPE_TO_TRY[$id]}
    for(( id_=0; id_ < "${#BS_TO_TRY[@]}"; id_++ ))
    do
        batch_size=${BS_TO_TRY[$id_]}
        func_trace_name

        echo "DTYPE=$DTYPE, BS=$batch_size"
        $FUNC_RUN_DNN_MODEL

        if [ "$PROFILE_MODE" = "nsys" ]; then
            if [ -e $NSYS_PROFILE_PATH ]; then
                if [ ! -d "${TRACE_PATH}" ]; then
                    mkdir -p "${TRACE_PATH}"
                else
                    rm -rf ${TRACE_PATH}/*
                fi

                nsys stats --report gpusum,gpumemsizesum,cudaapisum --format csv $NSYS_PROFILE_PATH > $NSYS_PROFILE_STDOUT

                python3 nsys_parser.py $NSYS_PROFILE_STDOUT --save_path $TRACE_PATH

                python3 nsys2json.py $NSYS_PROFILE_SQLITE > $NSYS_JSON_FILE

                mv stdout.log $NSYS_PROFILE_PATH $NSYS_PROFILE_SQLITE $NSYS_PROFILE_STDOUT $NSYS_JSON_FILE $TRACE_PATH

                func_upload_rst
            else
                echo "Nsight System profile failed." >&2
            fi
        else
            mv stdout.log $TRACE_PATH
            cd $TRACE_PATH
            dirname=$(ls plugins/profile/)
            cp plugins/profile/$dirname/*.trace.json.gz trace.json.gz
            cp plugins/profile/$dirname/*.memory_profile.json.gz memory_profile.json.gz
            rm -rf plugins events*
            func_upload_rst
        fi

    done
done

