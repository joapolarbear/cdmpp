FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
WORKDIR /root/

# UPDATE for NVIDIA
RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update && apt-get install -y libffi-dev python-dev git \
    libevent-dev cmake ssh \
    zip ibverbs-providers \
    llvm-dev zsh && \
    pip install --upgrade \
    pip install numpy==1.19.5 \
    ujson \
    networkx \
    xlsxwriter

RUN pip install requests \
    intervaltree \
    scapy xgboost \
    seaborn pymc3 \
    toposort \
    decorator attrs tornado psutil xgboost cloudpickle \
    onnx onnxoptimizer \
    transformers \
    pyyaml \
    jinja2 \
    prettytable \
    PyInquirer \
    bayesian-optimization \
    scipy==1.4.1 \
    scikit-learn \
    tqdm cvxpy python-igraph cvxopt && \
    ln -sf /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcusolver.so /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcusolver.so.10 && \
    ln -sf /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcupti.so /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcupti.so.11.0
        # tensorboard tensorflow-gpu \

RUN git clone --recursive https://github.com/joapolarbear/tvm tvm && \
    cd tvm && mkdir build && cp cmake/config.cmake.cuda build/config.cmake && \
    export TVM_LOG_DEBUG="ir/transform.cc=1;relay/ir/transform.cc=1" && \
    cd build && cmake .. && make -j 16 && \
    cd ../python; python3 setup.py install

RUN git clone https://github.com/joapolarbear/dpro.git && cd dpro && bash setup.sh 

# Edit `build/config.cmake`
# set(USE_CUDA ON)
# set(USE_GRAPH_EXECUTOR ON)
# set(USE_PROFILER ON)
# set(USE_RELAY_DEBUG ON)
# set(USE_LLVM ON)
