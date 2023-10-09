
# Installation
Install dPRO first
```
cd $HOME
git clone --recursive https://github.com/joapolarbear/dpro.git
cd dpro && bash setup.sh
```
Then under the root path of this project, run the following commands to install dependencies
```
cd <path of this repo> && git pull
pip3 install -r requirements.txt
```
And set the environment variables
```
PROJECT_PATH=$PWD
export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH
```

Then, see the help info first
```
usage: Cross Model Performance Prediction [-h] [-o OPTION] [-i INPUT]
                                          [-w WORKSPACE]
                                          [--log_level LOG_LEVEL]
                                          [--force_load_data]
                                          [--source_data SOURCE_DATA]
                                          [--op_type OP_TYPE] [--debug]
                                          [--metric_learner METRIC_LEARNER]
                                          [--opt OPT] [--lr LR] [--wd WD]
                                          [--cm_path CM_PATH]
                                          [--tb_logdir TB_LOGDIR]
                                          [--mode MODE] [--load_cache]
                                          [--cache_dir CACHE_DIR]
                                          [--tune_method TUNE_METHOD]

optional arguments:
  -h, --help            show this help message and exit
  -o OPTION, --option OPTION
                        One of [profile|tir_info]
  -i INPUT, --input INPUT
                        Path to store the inputs
  -w WORKSPACE, --workspace WORKSPACE
                        Path to store the results
  --log_level LOG_LEVEL
                        logging level

Training Data:
  --force_load_data     If specified, force to load raw data
  --source_data SOURCE_DATA
                        One of op, tir, or not specified
  --op_type OP_TYPE     If set, focus on one op_type

Cost Model:
  --debug               debug step by step
  --metric_learner METRIC_LEARNER
                        Specify the Metric Learner, use -1 to list all
  --opt OPT             optimizer
  --lr LR               learning rate
  --wd WD               weight
  --cm_path CM_PATH     Cost Model Path
  --tb_logdir TB_LOGDIR
                        Tensorboard logdir
  --mode MODE           How to learn task files, e.g.,
                        single|cross|group|samplek
  --load_cache          Set to ture to load the cached model
  --cache_dir CACHE_DIR
                        The dir load the cached model

Auto Tune:
  --tune_method TUNE_METHOD
                        Tunning method, one of 'file_grid', 'bo', 'grid'
```

# Train and test the cost model
## Train the cross-domain cost model
```
rm -rf .workspace/runs && python3 main.py --log_level info --metric_learner 2 --source_data tir --lr 0.001 --force_load_data -o train -w .workspace
```

## Test the cached cost model
Here is an example to evaluate the cost model on the dataset from 10 sampled tasks 
```
bash scripts/train.sh run \
    --mode sample10 \
    --metric_learner 2 \
    --load_cache \
    --cache_dir <cached_model_dir>
```
`--metric_learner 2` denotes MLP and `--metric_learner 3` denotes XGB
By default, it will use the dataset at `.workspace/ansor/` (If using ansor's feature extractor), so please make sure there task files under this directory.

## End-to-end performance simulation
```
bash scripts/train.sh replay --cache_dir .workspace/runs/search_trial_17_no_pe_fix/cm/BaseLearner
```

# Profile data
## Profiling data for training
```
python3 profiler/profile_tir/profile_tir.py -o profile -m .workspace/onnx_models -w .workspace/profile
```

## Parse tenset data
```
python3 profiler/profile_tir/tenset_dataload.py -o .workspace/tenset/t4  -i 3rdparty/tenset/scripts/dataset/measure_records/t4/ -s 0 --reload
```

Do not save the data but only save the file2task mapping
```
python3 profiler/profile_tir/tenset_dataload.py -o .workspace/tenset_fake/t4  -i 3rdparty/tenset/scripts/dataset/measure_records/t4/ -s 0 --fake
```

Re-use the feature engineering in Ansor/Auto-scheduler
```
python3 profiler/profile_tir/tenset_dataload.py -o .workspace/ansor/t4  -i 3rdparty/tenset/scripts/dataset/measure_records/t4/ -s 0 --reload --feature ansor
```

Parse AST + Ansor features
```
python3 profiler/profile_tir/tenset_dataload.py -o .workspace/ast_ansor/t4  -i 3rdparty/tenset/scripts/dataset/measure_records/t4/ -s 0 --reload --feature ast
```


## Reshape features
```
python3 main.py --log_level info --metric_learner 2 --source_data tir -o reshape_feature
```

## Pre-process data and make dataset

```
python3 main.py \
  -o make_dataset \
  -i .workspace/ast_ansor \
  --source_data tir \
  --mode sample200
```
or 
```
bash scripts/train.sh make_dataset --mode sample200 -i .workspace/ast_ansor
env CDPP_DATASET_PATH=tmp/dataset_large_flops bash scripts/train.sh make_dataset --mode sample200 -i .workspace/ast_ansor
```
Where `CDPP_DATASET_PATH` is the directory under which the pre-processed results are stored