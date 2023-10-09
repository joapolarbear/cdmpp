import os
import json
import numpy as np
import torch

from metalearner.data.rawdata import parse_di
from metalearner.feature import is_feature_type, ALL_FEATURE_TYPE

from .my_learner import MyMatchNetLearner
from .matchnet_learner import MatchNetLearner
from .kernel_learner import KernelRegressionLearner
from .xgb import XGBLearner
from .attention_learner import AttentionLearner
from .base_learner import BaseLearner, NoEnoughBatchError

from utils.util import read_yes, task_repr, TrainTestPair
from utils.env import PROJECT_CFG

ALL_METRIC_LERNER = [
    MatchNetLearner,
    KernelRegressionLearner,
    MyMatchNetLearner,
    XGBLearner,
    AttentionLearner
]

def parse_metric_learner(learner_id, verbose=True):
    if learner_id == -1:
        print(" ** All Metric Learners **")
        for idx, learner in enumerate(ALL_METRIC_LERNER):
            print(" - {}: {}".format(idx, learner.__name__))
        exit(0)
    else:
        if is_feature_type(ALL_FEATURE_TYPE.ast_ansor):
            learner_cls = AttentionLearner
        else:
            assert learner_id < len(ALL_METRIC_LERNER)
            learner_cls = ALL_METRIC_LERNER[learner_id]
        if verbose:
            print("[Learner] Use the learner {}".format(learner_cls.__name__))
        return learner_cls

class InferBuffer:
    def __init__(self, learner, batch_size, assign_pred_rst):
        self.learner = learner
        self.buffer = []
        self.batch_size = batch_size
        self.assign_pred_rst = assign_pred_rst
    
    def add(self, _key, features):
        ''' Add the features of one query, identified by _key'''
        # norm_tir_feature = self.learner.data_meta_info.norm_input(features)  
        self.buffer.append((_key, features))
        if len(self.buffer) >= self.batch_size:
            self.infer()
            self.buffer = []
    
    def finalize(self):
        if len(self.buffer) > 0:
            self.infer()
            self.buffer = []
    
    def infer(self):
        features = np.array([_feature for _key, _feature in self.buffer])
        norm_tir_feature = self.learner.data_meta_info.norm_input(features)
        di = parse_di(self.buffer)
        preds = self.learner.predict(torch.Tensor(norm_tir_feature).cuda())
        for idx in range(len(self.buffer)):
            self.assign_pred_rst(self.buffer[idx][0], preds[idx])

def ret_learner(data_meta_info, learning_params, verbose):
    learner_cls = parse_metric_learner(PROJECT_CFG["metric_learner"], verbose=verbose)
    return learner_cls(
            data_meta_info = data_meta_info,
            cache_path=learning_params["cache_dir"],
            debug=learning_params["debug"],
            tb_log_dir=learning_params["tb_logdir"]
        )
    
def _metric_learning_w_tir_data_impl(
        ds_pair: TrainTestPair,
        data_meta_info,
        learning_params,
        verbose=True):
    assert learning_params["op_type"] is None, "--op_type should not be set when using TIR features"
    learner = ret_learner(data_meta_info, learning_params, verbose)
    learner.assign_config(PROJECT_CFG["cost_model"], verbose=False)

    # learning_params["load_cache"] = True
    # learning_params["max_epoch"] = int(1e4)
    
    ### Hyper-parameters specified using command line arguments have higher priorities
    if learning_params["batch_size"] is not None:
        learner.batch_size = learning_params["batch_size"]
    if learning_params["epoch"] is not None:
        learning_params["max_epoch"] = learning_params["epoch"]
    if learning_params["step"] is not None:
        learning_params["max_step"] = learning_params["step"]
    
    if learning_params["loss_func"] is not None:
        learner.loss_type = learning_params["loss_func"]
        learner._init_loss_func()
        print(f"[Learner Entry] Use loss type {learner.loss_type}")
    
    if learning_params["domain_diff_metric"] is not None:
        learner.domain_diff_metric = learning_params["domain_diff_metric"]

    return learner, learner.train(ds_pair, verbose=verbose, learning_params=learning_params)

class TaskRuntimeCache:
    def __init__(self, dir, cfg_name):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.cache_path = os.path.join(dir, f"task_runtime_cache_{cfg_name}.json")
        if os.path.exists(self.cache_path):
            x = input(f"Found existing cached task runtime info at {self.cache_path}? Make sure the same config is used: [Y/n]")
            if x.lower() in ["y", "yes", "1", ""]:
                pass
            else:
                exit(0)
            with open(self.cache_path, 'r') as fp:
                self.task_cache = json.load(fp)
        else:
            self.task_cache = {}
    
    def contain(self, task_id):
        return task_id in self.task_cache and \
            "mape_train" in self.task_cache[task_id] and \
            "mape_val" in self.task_cache[task_id]
    
    def record(self, task_id, info: dict):
        if task_id not in self.task_cache:
            self.task_cache[task_id] = {}
        for _key, _value in info.items():
            self.task_cache[task_id][_key] = _value
    
    def __getitem__(self, index):
        return self.task_cache[index]
    
    def save(self):
        with open(self.cache_path, 'w') as fp:
            json.dump(self.task_cache, fp, indent=4)

    def get_file_group_id(self, files):
        return task_repr(files)

