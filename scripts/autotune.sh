#!/bin/bash
# Scripts to launch autotune processes
function usage_prompt {
    echo "Usage: bash xxx.sh [OPTION] ..."
    echo "Launch autotune processes"
    echo
    echo "  - bash xxx.sh bo,                       Launch a Bayesian Optimization Auto-tune Process"
    echo "  - bash xxx.sh file_grid <start_id>,     Launch Grid Search Processes on each available GPU,"
    echo "                                          Optional argument start_id denotes the first group configs"
    echo "                                          to test."
    echo "  - bash xxx.sh fine_tune <process_id>,   Fine tune a process"
    echo "  - bash xxx.sh kill,                     Kill all search processes"
    echo "  - bash xxx.sh check <process_id>,       Check the progress of a particular process"
    echo 
    exit
}
if [ -z $1 ]; then
    usage_prompt
fi

# cd && cd cross-device-perf-predictor
# git pull && git submodule init && git submodule update
# pip3 install -r requirements.txt 

PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

mkdir -p .fig && mkdir -p .workspace
if [[ ! -d ".workspace/tenset" && $1 == "file_grid" ]]; then
    cd .workspace
    hdfs dfs -get /usr/hphu/0tmp/tenset.zip && unzip tenset && rm tenset.zip
    hdfs dfs -get /usr/hphu/0tmp/None-norm-info.pickle && mv None-norm-info.pickle tenset/
    cd .. 
fi
mkdir -p .workspace/autotune

function process_ws {
    process_id=$1
    echo ".workspace/autotune/process${process_id}"
}

function tune_process {
    process_id=$1
    gpu_id=$2
    work_dir=$(process_ws ${process_id})
    echo "Search process ${process_id} on GPU ${gpu_id}"

    if [ ! -d ${work_dir}/all_cfgs ]; then
        mkdir -p ${work_dir}/all_cfgs
        cp .workspace/autotune/all_cfgs/${process_id}.json ${work_dir}/all_cfgs/
    fi
    CUDA_VISIBLE_DEVICES=${gpu_id} setsid nohup python3 main.py --log_level info \
        --metric_learner 2 --source_data tir --force_load_data \
        -o tune \
        -i .workspace/ansor \
        -w ${work_dir} > ${work_dir}/nohup.txt 2>&1 &
}

function fine_tune_process {
    process_id=$1
    gpu_id=$process_id
    work_dir=$(process_ws ${process_id})
    tb_logdir=$2
    echo "Finetune process ${process_id} on GPU ${gpu_id}, tb_logdir=$tb_logdir"

    CUDA_VISIBLE_DEVICES=${gpu_id} setsid nohup python3 -u main.py \
        --log_level info \
        --metric_learner 2 \
        --source_data tir \
        --force_load_data \
        -o tune \
        -i .workspace/ansor \
        -w ${work_dir} \
        --tune_method fine_tune \
        --tb_logdir $tb_logdir \
        > ${work_dir}/nohup.txt 2>&1 &
    # CUDA_VISIBLE_DEVICES=${gpu_id} python3 main.py \
    #     --log_level info \
    #     --metric_learner 2 \
    #     --source_data tir \
    #     --force_load_data \
    #     -o tune \
    -i .workspace/ansor \
    #     -w ${work_dir} \
    #     --tune_method fine_tune \
    #     --tb_logdir $tb_logdir 
}

function kill_process {
    if [ -z $1 ]; then
        echo "Kill all running processes"
        ps -ef | grep " python3 main.py" | grep -v grep | awk '{print $2}' | xargs kill -9
        return
    fi
    process_id=$1
    work_dir=$(process_ws ${process_id})
    echo "Kill process ${process_id}"
    ps -ef | grep "${work_dir}" | grep -v grep | awk '{print $2}' | xargs kill -9
}

function check_process {
    if [ -z $1 ]; then
        echo "All running processes"
        ps -ef | grep " python3 main.py" | grep -v grep 
        return
    fi
    process_id=$1
    work_dir=$(process_ws ${process_id})
    echo "Process ${process_id}:"
    tail -n 10 ${work_dir}/tuning.log
}

### ------ start search processes
remain_arg=${@:2}
if [ $1 = "bo" ]; then
    ### Bo search
    setsid nohup python3 main.py --log_level info \
            --metric_learner 2 --source_data tir --force_load_data \
            -o tune \
            -i .workspace/ansor \
            -w .workspace/autotune/ \
            --tune_method bo > nohup.txt 2>&1 &
elif [ $1 = "file_grid" ]; then
    ### Grid search
    if [ ! -d ".workspace/autotune/" ]; then
        hdfs dfs -get /usr/hphu/0tmp/all_cfgs && mv all_cfgs .workspace/autotune/
    fi

    if [ -n $2 ]; then
        file_base=$2
    else
        file_base=0
    fi

    for(( id=0; id < ${GPU_NUM}; id++ )); do
        file_id=$(($file_bias+$id))
        tune_process $file_id $id
    done
elif [ $1 = "optuna" ]; then
    python3 main.py \
        --log_level info \
        --source_data tir \
        --force_load_data \
        -o tune \
        -i .workspace/ansor \
        -w .workspace/autotune/ \
        --tune_method optuna \
        $remain_arg
elif [ $1 = "fine_tune" ]; then
    fine_tune_process $2 $3
elif [ $1 = "kill" ]; then
    ### Kill them
    kill_process $2
elif [ $1 = "check" ]; then
    check_process $2
else
    usage_prompt
fi


# PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH
# GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# for(( id=0; id < ${GPU_NUM}; id++ )); do
#     cat .workspace/autotune/process${id}/tuning.log
#     # cat .workspace/autotune/process3/best/best_cfg.json
# done
