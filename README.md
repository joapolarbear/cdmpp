# Configuration
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

## Prepare the Dataset

### Download and unzip
You can choose to use either the CPU part or the GPU part. See [Tenset Dataset](https://github.com/tlc-pack/tenset/blob/main/docs/get_started_with_cost_model_experiments.md) to download the dataset accordingly. Our profiled dataset for A100, V100 and P100 will be available at [DOI:10.6084/m9.figshare.24156084](https://figshare.com/articles/dataset/cdmpp-data/24156084)

### An example for T4
Here we show an example to download the dataset of NVDIA T4.
1. Change directory to `<The root directory of cdmpp>/3rdparty/tenset/scripts/`
2. Download
  - You can download it from google drive with the link [dataset_gpu_v3.3.zip](https://drive.google.com/file/d/1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK/view?usp=sharing)
  - Or you can use the command line
    ```
    pip3 install gdown
    gdown https://drive.google.com/uc?id=1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK
    ```
3. Unzip  
  Put `dataset_gpu_v3.3.zip` under `<The root directory of cdmpp>/3rdparty/tenset/scripts/` and run `unzip dataset_gpu_v3.3.zip`.
  A new folder `dataset_gpu` will appear in `<The root directory of cdmpp>/3rdparty/tenset/scripts/`. Make `dataset` as a softlink to it
  by `ln -s <The root directory of cdmpp>/3rdparty/tenset/scripts/dataset_gpu dataset`.

### Feature Extraction
Next, we will extract features for the dataset of each device. Make sure that you have put the profiled dataset under `3rdparty/tenset/scripts/dataset/measure_records/$DEVICE_MODEL/`, where `$DEVICE_MODEL` is the device whose dataset you want to extract from. Then, you can run the following commands to extract features.
```bash
cd && cd cdmpp
bash scripts/dataset/gen_raw_feature_all.sh
``` 
By default, the extracted features will be stored at `.workspace/ast_ansor/$DEVICE_MODEL`.
The process of extracting features and data preprocessing may take around 10~20 minutes for the dataset of each device.

### Data Preprocessing [Optional]
Run the following commands to preprocess the dataset
``` bash
bash scripts/dataset/make_dataset.sh
```
The preprocessed data will be stored under the `tmp/` directory. You can also skip this process since this can be done automatically before training starts, i.e., when the preprocessed dataset is required to be used for the first time. 


# Training
We will the configuration file `tmp/search_trial_20221119_1575.yaml`, which contains hyper-parameters found by our auto-tuner, to run the following experiments.

## Cross-model training
```
bash scripts/exp/cross_model.sh none
```

## Cross-device training
```
bash scripts/exp/cross_device.sh
```

# End2end

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