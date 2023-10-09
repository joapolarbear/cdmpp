#!/bin/bash
python3 -c "import habitat"
if [ $? -ne "0" ]; then
    echo "Habitat is not installed, install it"
    apt-get update --fix-missing && \
    apt-get install --no-install-recommends -y \
        software-properties-common sudo \
        python3-pip python3-setuptools python3-dev \
        wget bzip2 ca-certificates libssl-dev \
        libjpeg-dev zlib1g-dev
    # rm -rf /var/lib/apt/lists/*

    pip3 install wheel PyYAML \
        torch==1.4.0 \
        torchvision==0.5.0 \
        pandas==1.1.2 \
        tqdm==4.49.0

    ### Download and install cmake-3.17
    wget "https://github.com/Kitware/CMake/releases/download/v3.17.0-rc1/cmake-3.17.0-rc1.tar.gz" -O /opt/cmake-3.17.0-rc1.tar.gz && \
    cd /opt && tar xzf cmake-3.17.0-rc1.tar.gz && cd /opt/cmake-3.17.0-rc1 && \
    ./bootstrap && make -j 16 && make install

    ### Clone habitat
    cd $HOME && git clone --recurse-submodules https://github.com/geoffxy/habitat

    ### Download pretrained habitat model
    wget https://zenodo.org/record/4876277/files/habitat-models.tar.gz?download=1 -O habitat-models.tar.gz 
    bash habitat/analyzer/extract-models.sh habitat-models.tar.gz


    export CUDA_HOME=/usr/local/cuda
    export CUPTI_DIR=/usr/local/cuda/targets/x86_64-linux
    ### habitat installer only finds CUPTI under some specific directories
    ln -sf /usr/local/cuda/targets/x86_64-linux/lib /usr/local/cuda/extras/CUPTI/lib64

    ### Note to check the SO_NAME in install-dev.sh
    #   habitat fixes the .so name, if not found, an error will be raised
    unameOut="$(uname -s)"
    case "${unameOut}" in
        Linux*)     machine=Linux;;
        Darwin*)    machine=Mac;;
        CYGWIN*)    machine=Cygwin;;
        MINGW*)     machine=MinGw;;
        *)          machine="UNKNOWN:${unameOut}"
    esac
    echo ${machine}
    if [ ${machine} = "Mac" ]; then
        sed_label="\".sh\""
    else
        sed_label=""
    fi
    IFS=' ' python3_split=(`python3 -V`); unset IFS
    IFS='.' version_numbers=(${python3_split[1]}); unset IFS
    major="${version_numbers[0]}"
    minor="${version_numbers[1]}"
    sed -i $sed_label "s/cpython-.*m-/cpython-${major}${minor}m-/g" habitat/analyzer/install-dev.sh

    ### Install habitat package
    export PATH=/usr/local/cuda/bin:$PATH
    bash habitat/analyzer/install-dev.sh
else echo "Habitat has been installed"; fi

### Run examples
python3 habitat/experiments/run_experiment.py V100
# cd habitat/experiments
# bash gather_raw_data.sh V100
# bash process_raw_data.sh