
#!/bin/bash

GPU_MODEL_NAME=($(CUDA_VISIBLE_DEVICES=0 nvidia-smi --query-gpu=name --format=csv,noheader | tr " " _))
GPU_MODEL_NAME=${GPU_MODEL_NAME[0]}

dirname=`dirname $0`
capability=`python3 $dirname/../../utils/device_info.py $GPU_MODEL_NAME`

export CUDACXX=`which nvcc`
cutlass_profiler --help
if [ $? = '0' ]; then 
    echo "CUTLASS has been installed"
else
    
    cd
    if [ ! -d cutlass ]; then
        git clone https://github.com/joapolarbear/cutlass
    fi
    cd cutlass
    if [ -d build ]; then
        cd build && rm -rf ./*
    else
        mkdir build && cd build
    fi

    # real    0m8.013s
    cmake .. -DCUTLASS_NVCC_ARCHS=${capability} -DCUTLASS_LIBRARY_KERNELS=all
    # real    9m6.230s
    make cutlass_profiler -j12
    make 20_find_op_kernels -j
    sudo ln -sf $PWD/tools/profiler/cutlass_profiler /usr/bin/
    
fi
