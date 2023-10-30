#!/bin/bash
set -x
<<'COMMENTS'

### Write data to xxxxdrive
cd /mnt/bd/tenset-data/
hdfs dfs -get /usr/hphu/0data/cdpp/tenset/tenset_makedataset_t4.pkl
hdfs dfs -get /usr/hphu/0data/cdpp/tenset/dataset_gpu_v3.3.zip
unzip dataset_gpu_v3.3.zip && rm dataset_gpu_v3.3.zip

### Download cross-device-perf-predictor repo
cd && cd cross-device-perf-predictor
git submodule init && git submodule update
PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH
pip3 install -r requirements.txt 

### Reinstall old version of tvm that Tenset needs
sudo -i
cd /root/ && rm -rf tvm
git clone --recursive https://github.com/joapolarbear/tvm tvm
cd tvm && mkdir build && cp cmake/config.cmake.cuda build/config.cmake
git checkout tenset && git submodule update
export TVM_LOG_DEBUG="ir/transform.cc=1;relay/ir/transform.cc=1"
cd build && cmake .. && make -j 16
cd ../python; python3 setup.py install
exit

###
pip3 uninstall -y torchvision torch
sudo pip3 install torch==1.8.0 \
    torchvision==0.9.0

### Data
ln -sf /mnt/bd/tenset-data/dataset_gpu 3rdparty/tenset/scripts/dataset
ln -sf /mnt/bd/tenset-data/tenset_makedataset_t4.pkl .workspace/tenset_makedataset_t4.pkl
COMMENTS

PROJECT_PATH=$PWD && export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH

### Run tenset scripts
SAMPLE_NUM=2308
# SAMPLE_NUM=200
EXP_TIMES=3
cd 3rdparty/tenset/scripts
RST_DIR=".workspace"
mkdir -p ${RST_DIR}/logs
mkdir -p ${RST_DIR}/cost_models
## Pre-process the measured records in Tenset, e.g., all records of t4
# python3 make_dataset.py \
#     dataset/measure_records/t4/\(\[0013c369f74ad81dbdce48d38fd42748,1,14,14,512,3,3,16,512,1,1,1,512,1,14,14,512\],cuda\).json \
#     --out-file .workspace/tenset_dataset.pkl
    # --sample-in-files 1 \

# PLATFORM="cuda"
# PLATFORM="llvm"
PLATFORM=$1
GPU_DEVICES=('t4' 'a100' 'p100' 'v100' 'k80')
CPU_DEVICES=('e5-2673'  'epyc-7452'  'graviton2')
if [[ -L dataset ]]; then
    rm dataset
fi
if [[ ${PLATFORM} == "llvm" ]]; then
    DEVICES=(${CPU_DEVICES[@]})
    ln -sf $PWD/dataset_cpu dataset
elif [[ ${PLATFORM} == "cuda" ]]; then
    DEVICES=(${GPU_DEVICES[@]})
    ln -sf $PWD/dataset_gpu dataset
else
    echo "Invalid platform $PLATFORM"
fi
echo "$PLATFORM,$DEVICES"

for DEVICE in "${DEVICES[@]}"; do
    for (( i=1; i<=${EXP_TIMES}; i++ )); do
        echo "device ${DEVICE}, Exp iter ${i}"
        rm -rf ${RST_DIR}/tenset_dataset.pkl
        python3 make_dataset.py \
            --device ${DEVICE} \
            --data-size ${SAMPLE_NUM} \
            --out-file ${RST_DIR}/tenset_dataset.pkl

        LOG_PATH=${RST_DIR}/logs/xgb_${DEVICE}_${i}_${SAMPLE_NUM}.txt
        
        python3 train_model.py \
            --train-ratio 0.8 \
            --models xgb@random \
            --dataset ${RST_DIR}/tenset_dataset.pkl 2>&1 | tee ${LOG_PATH}
        
        mv xgb.pkl ${RST_DIR}/cost_models/xgb_${DEVICE}_${i}_${SAMPLE_NUM}.pkl
    done
done