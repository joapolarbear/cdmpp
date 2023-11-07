# Introduction
CDMPP is a framework to accurately predict the latency of tensor programs from different DNN models on various devices

## Code Organization
The repository is organized as follows:

- `main.py`: the main entry of this project.
- `tvm_helper/`: contain the code related to TIR feature extraction
- `scritps/`: contain scripts to for this project
  - `train.py`: base script set launch a training job
  - `replay.py`: base script to use a pre-trained cost model for replay
  - `autotune.sh`: script to run the auto-tuner
  - `ablation/`: contain script to run the albation study experiments
  - `applications/`: contain script to run applications of our cost model, e.g., schedule search in TVM
  - `baseline/`: contain script to run baselines
  - `cross_model_finetune/`: contain scripts to run cross-model finetuning experimetns
  - `dataset/`: contain scripts to extract features and pre-process data
  - `end2end`: contain code to run end2end experiments
- `profiler`: contain code related to profiling, e.g., profile the latency of 
  - GPU kernels
  - operators
  - TIR tensor programs
  - habana kernels
  - cutlass kernels.
- `legacy/`: back up code
- `help/`: contain some auxilary codes, including
  - combine mutiple trace files
  - collect all label values from a large dataset
  - generate config files in YAML format required by our cost model from the auto-tuner search results
  - monitor the GPU utilization
  - parse runtime info from tensorboard trace files
- `end2end/`: contain code related to end2end experiments, including end2end performance replaying, measure the real performance and perform end2end performance prediction
- `docs/`: contain docs
- `docker/`: contain a DOCKERFILE to build an image for this project
- `configs/`: contain YAML files to specify the configuration of the experiment
- `utils`: utilization code, including
  - device information
  - code to parse the configuration files
  - different metric functions
- **`metalearner/`**: main part of the prediction framework
  - `analyze/`: contain the code to analyze experiment results
  - `autotune/`: contain the code of the auto-tuner
  - `clustering/`: contain the code of the clustering-based sampling strategy for cross-device fine-tuning
  - `data/`: contain the code of dataloading and preprocessing
  - `feature/`: contain the code of feature extraction
  - `learner/`: contain the code to train the cost model
  - `model/`: contain the code of the model architecture




# Setup

## Hardware dependencies
The current implementation of CDMPP requires GPUs to run the cost model. 

## Software dependencies
- customized TVM: https://github.com/joapolarbear/tvm
- dPRO: https://github.com/joapolarbear/dpro
- CUDA driver version >=450.80.02 (Linux) / 452.39 (Windows)

## Launch a container
Pull our docker image
```
docker pull haaanpeng/cdmpp:eurosys_ae
```

Launch the container
```bash
docker run -it --runtime=nvidia --shm-size 32768m --name hphu-test haaanpeng/cdmpp:eurosys_ae /bin/bash
```

Download the source code and install dependencies.
```bash
cd
git clone --recursive https://github.com/joapolarbear/cdmpp && cd cdmpp
bash setup.sh
```
---

# Usage

## Prepare the Dataset

### Download and unzip
You can choose to use either the CPU part or the GPU part. See [Tenset Dataset](https://github.com/tlc-pack/tenset/blob/main/docs/get_started_with_cost_model_experiments.md) to download the dataset accordingly. Our profiled dataset for A100, V100 and P100 will be available at [DOI:10.6084/m9.figshare.24156084](https://figshare.com/articles/dataset/cdmpp-data/24156084)

### An example of T4 GPU
Here we show an example of downloading the dataset of NVIDIA T4.
1. Change directory to `<The root directory of cdmpp>/3rdparty/tenset/scripts/`
2. Download
  - You can download it from Google Drive with the link [dataset_gpu_v3.3.zip](https://drive.google.com/file/d/1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK/view?usp=sharing)
  - Or you can use the command line
    ```
    pip3 install gdown
    gdown https://drive.google.com/uc?id=1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK
    ```
3. Unzip  
  - Put `dataset_gpu_v3.3.zip` under `<The root directory of cdmpp>/3rdparty/tenset/scripts/` and run `unzip dataset_gpu_v3.3.zip`.
  - A new folder `dataset_gpu` will appear in `<The root directory of cdmpp>/3rdparty/tenset/scripts/`. Make `dataset` as a softlink to it
  by `ln -s <The root directory of cdmpp>/3rdparty/tenset/scripts/dataset_gpu dataset`.

### An example of AMD EPYC 7452 CPU
Here we show an example to download the dataset of AMD EPYC 7452 CPU.
1. Change directory to `<The root directory of cdmpp>/3rdparty/tenset/scripts/`
2. Download
  - You can download it from Google Drive with the link [dataset_cpu_v3.3.zip](https://drive.google.com/file/d/1JQwGEe8jCpuhZPnUxO0Sb1CJJ06uevy6/view?usp=sharing)
  - Or you can use the command line
    ```
    pip3 install gdown
    gdown https://drive.google.com/uc?id=1JQwGEe8jCpuhZPnUxO0Sb1CJJ06uevy6
    ```
3. Unzip  
  - Put `dataset_cpu_v3.3.zip` under `<The root directory of cdmpp>/3rdparty/tenset/scripts/` and run `unzip dataset_cpu_v3.3.zip`.
  - A new folder `dataset_cpu` will appear in `<The root directory of cdmpp>/3rdparty/tenset/scripts/`. Make `dataset` as a softlink to it
  by `ln -s <The root directory of cdmpp>/3rdparty/tenset/scripts/dataset_cpu dataset`.

In the above process, if `dataset` already exists, just run `mv dataset_cpu/measure_records/* dataset/measure_records/`

After the above processes, you will see several directories under `<The root directory of cdmpp>/3rdparty/tenset/scripts/dataset/measure_records` as follows
```bash
measure_records
  |-t4
  |-k80
```
Note that each directory name represents a specific device and we will use those device names as flags to specify which device we will use to extract features or run training.

### Feature Extraction
Next, we will extract features for the dataset of each device. Make sure that you have put the profiled dataset under `3rdparty/tenset/scripts/dataset/measure_records/$DEVICE_MODEL/`, where `$DEVICE_MODEL` is the device whose dataset you want to extract from. Then, you can run the following commands to extract features.
```bash
cd && cd cdmpp
bash scripts/dataset/gen_raw_feature_all.sh
``` 
By default, the extracted features will be stored at `.workspace/ast_ansor/$DEVICE_MODEL`.
The process of extracting features and data preprocessing may take around 10~20 minutes for the dataset of each device.
If you want to extract features for other devices, just modify `DEVICES` in the `gen_raw_feature_all.sh`. 
By default, `'t4'` is used. Take AMD EPYC 7452 CPU for example, you can modify `DEVICES` as `DEVICES=('epyc-7452')`. 

### Data Preprocessing [Optional]
Run the following commands to preprocess the dataset
``` bash
bash scripts/dataset/make_dataset.sh
```
The preprocessed data will be stored under the `tmp/` directory. You can also skip this process since this can be done automatically before training starts, i.e., when the preprocessed dataset is required to be used for the first time. 


## Training
We will the configuration file `tmp/search_trial_20221119_1575.yaml`, which contains hyper-parameters found by our auto-tuner, to run the following experiments.

### Cross-model training
```
bash scripts/exp/cross_model.sh none
```
Similar to feature extraction, you can modify `DEVICES` in the `gen_raw_feature_all.sh` to run training on the dataset from other devices, e.g., Take AMD EPYC 7452 CPU for example, you can modify `DEVICES` as `DEVICES=('epyc-7452')`. 

### Cross-device training
```
bash scripts/exp/cross_device.sh
```

### End2end

```
bash scripts/end2end/recur_test.py
```

# Citation
If you find this project intriguing and wish to cite it, please utilize the following citation:
```
@inproceedings{hanpeng2024cdmpp,
  title={{CDMPP: A Device-Model Agnostic Framework for Latency Prediction of Tensor Programs}},
  author={Hanpeng, Hu and Junwei, Su and Juntao, Zhao and Yanghua, Peng and Yibo, Zhu and Haibin, Lin and Chuan, Wu},
  booktitle={Proceedings of the Nineteenth EuroSys Conference},
  year={2024}
}
```

# License
© Contributors Licensed under an [Apache-2.0](LICENSE) license.
