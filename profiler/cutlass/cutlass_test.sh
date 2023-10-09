#!/usr/bin/
### Profile CUTLASS kernels
#   Usage: bash cutlass_test.sh gemm,conv2d,sparsegemm,conv3d
if (( $# != 1 )); then
        echo "Usage: bash cutlass_test.sh gemm,conv2d,sparsegemm,conv3d"
        exit 
fi

set -x

GPU_MODEL_NAME=($(CUDA_VISIBLE_DEVICES=0 nvidia-smi --query-gpu=name --format=csv,noheader | tr " " _))
GPU_MODEL_NAME=${GPU_MODEL_NAME[0]}

###########################################################################
### Check and install CUTLASS
dirname=`dirname $0`
bash $dirname/install_cutlass.sh

### Profile
# cutlass_profiler --operation=Gemm --help
# cutlass_profiler --operation=Conv3d --help
# cutlass_profiler --operation=Conv2d --help
# cutlass_profiler --operation=SparseGemm --help
# --kernels=cutlass_simt_sgemm_128x128 --m=4352 --n=4096 --k=256 
# --save-workspace=incorrect
# cutlass_profiler --operation=Gemm --output $OUTPUT_PATH --append=1

###########################################################################
### Prepare directories
output_dir=cutlass_test_$GPU_MODEL_NAME
cd && mkdir $output_dir && cd $output_dir
hdfs_dir=/usr/hphu/0data/cutlass_test/$output_dir/
hdfs dfs -test -e ${hdfs_dir}
if [ $? != "0" ]; then
    hdfs dfs -mkdir -p ${hdfs_dir}
fi

function upload2hdfs {
    hdfs dfs -put -f $1 ${hdfs_dir}
}

IFS=',' arg_list=($1); unset IFS

###########################################################################

### Gemm
function profile_gemm {
    echo "Profile $1"
    rm $GPU_MODEL_NAME.$1.csv
    test_id=0
    for m_shape in 8 16 32 64 128 256 512; do
        for n_shape in 8 16 32 64 128 256 512; do
            for k_shape in 8 16 32 64 128 256 512; do
                echo "Test $test_id: M=$m_shape, N=$n_shape, K=$k_shape"
                cutlass_profiler --operation=$1 \
                    --m=$m_shape --n=$n_shape --k=$k_shape --beta=0,1 \
                    --accumulator-type=f16,f32 --A=f16,f32 --B=f16,f32 --C=f16,f32 \
                    --providers=cutlass --profiling-iterations=10 --warmup-iterations=10 --verbose=0 \
                    --output=$GPU_MODEL_NAME --append=1
                upload2hdfs $GPU_MODEL_NAME.$1.csv
                test_id=$(($test_id+1))
            done
        done
    done
    echo "Test $test_id: M=1024:4096:1024, N=1024:4096:1024, K=1024:4096:1024"
    cutlass_profiler --operation=$1 \
        --m=1024:4096:1024 --n=1024:4096:1024 --k=1024:4096:1024 --beta=0,1 \
        --accumulator-type=f16,f32 --A=f16,f32 --B=f16,f32 --C=f16,f32 \
        --providers=cutlass --profiling-iterations=10 --warmup-iterations=10 --verbose=0 \
        --output=$GPU_MODEL_NAME --append=1
    upload2hdfs $GPU_MODEL_NAME.$1.csv
}

if [[ ${arg_list} =~ (^|[[:space:]])"gemm"($|[[:space:]]) ]]; then
    profile_gemm "gemm"
fi

if [[ ${arg_list} =~ (^|[[:space:]])"sparsegemm"($|[[:space:]]) ]]; then
    profile_gemm "sparsegemm"
fi


### Convolutional OPs
INPUT_SIZE_LIMIT=$((1024*1024*512))
function profile_conv {
    echo "Profile $1"
    $GPU_MODEL_NAME.$1.csv
    test_id=0
    for h_w_shape in 16 64 128 256; do
    for c_shape in 64 128 256 512 1024; do
        for k_shape in 64 128 256 512 1024; do
        for kernel_shape in 1 3; do
            for pad in 0; do for stride in 1 2; do for batch_size in 16 32 64 128 256; do
            max_input_size=$(($h_w_shape*$h_w_shape*$c_shape*$batch_size*4))
            echo "Test $test_id: N=$batch_size, H/W=$h_w_shape, C=$c_shape, K=$k_shape, R/S=$kernel_shape"
            if [ $max_input_size -ge $INPUT_SIZE_LIMIT ]; then
                echo "Extremely large problem, skip"
            else
                cutlass_profiler --operation=$1 \
                    --beta=0,1 \
                    --Activation=f16,f32 --Filter=f16,f32 --Output=f16,f32 --accumulator-type=f16,f32 \
                    --n=$batch_size --h=$h_w_shape--w=$h_w_shape --c=$c_shape \
                    --k=$k_shape --r=$kernel_shape --s=$kernel_shape \
                    --pad_h=$pad --pad_w=$pad \
                    --stride::h=$stride --stride::w=$stride \
                    --dilation::h=1 --dilation::w=1 \
                    --providers=cutlass --profiling-iterations=10 --warmup-iterations=10 --verbose=0 \
                    --append=1 --output=$GPU_MODEL_NAME
                upload2hdfs $GPU_MODEL_NAME.$1.csv
            fi
            test_id=$(($test_id+1))
            done; done; done
        done
        done
    done
    done
}
if [[ ${arg_list} =~ (^|[[:space:]])"conv2d"($|[[:space:]]) ]]; then
    profile_conv "conv2d"
fi

if [[ ${arg_list} =~ (^|[[:space:]])"conv3d"($|[[:space:]]) ]]; then
    profile_conv "conv3d"
fi