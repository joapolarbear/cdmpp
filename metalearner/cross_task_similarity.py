''' Used for domain-difference analysis
'''
import os
import numpy as np
import json
from typing import Tuple, Union
import prettytable
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from utils.util import warning, task_repr, sample_task_files, fig_base, TrainTestPair
import utils.env as cdpp_env

from metalearner.data.dataloader import (
    load_raw_data,
    MyDataSet
)
from metalearner.data.rawdata import (
    MIN_PER_TASK_SAMPLE_NUM,
    parse_metainfo,
    RawData
)
from metalearner.learner.base_learner import BaseLearner
from metalearner.learner import (
    _metric_learning_w_tir_data_impl,
    ret_learner
)
from tvm_helper.metadata import DataMetaInfo

DEBUG = True
DISABLE_CACHE = False
TARGET_TASK_ID = [0]
CONTINUOUS_LEARN = False
SENSITIVE_ANALYZE = True
TRAIN_CLASSIFIER = False

INVALID_ENTRIES = [0, 3, 6, 7, 12, 13, 15, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 94, 102, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146]

# TODO: deperacated
def evaluate_elementwise_mape(_learner: BaseLearner, dataset: Union[MyDataSet, Tuple]):
    if isinstance(dataset, MyDataSet):
        all_data = dataset.get_all_data()
    else:
        all_data = dataset
    val_x, val_y = _learner.data_to_train_device(*[torch.tensor(_data, dtype=torch.float32) for _data in all_data])
    outputs = _learner._inference(val_x)
    metrics_val = _learner.compute_metrics(outputs, val_y, element_wise_test=True)
    return metrics_val["element-wise-mape"]

# TODO: deperacated
def evaluate_y_func(_learner: BaseLearner, dataset: Union[MyDataSet, Tuple]):
    if isinstance(dataset, MyDataSet):
        all_data = dataset.get_all_data()
    else:
        all_data = dataset
    val_x, val_y = _learner.data_to_train_device(*[torch.tensor(_data, dtype=torch.float32) for _data in all_data])
    outputs = _learner.predict(val_x)
    return outputs


############################### Data loading ###############################

def _load_task_train_test_raw_data(root_path, _file, learning_params, _data_save_dir):
    os.makedirs(_data_save_dir, exist_ok=True)
    _train_data_path = os.path.join(_data_save_dir, "train.pickle")
    _val_data_path = os.path.join(_data_save_dir, "val.pickle")
    _metainfo_path = os.path.join(_data_save_dir, "metainfo.pickle")
    if not DISABLE_CACHE and os.path.exists(_train_data_path) and os.path.exists(_val_data_path):
        ### Load the existing data
        metainfo = DataMetaInfo.load_init(_metainfo_path)
        train_raw_data = RawData.load_init(_train_data_path, metainfo=metainfo)
        val_raw_data = RawData.load_init(_val_data_path, metainfo=metainfo)
    else:
        ### Re-generate the training and test data
        raw_data = load_raw_data(
            [os.path.join(root_path, _file)],
            learning_params,
            force=True,
            verbose=False)
        if raw_data.size < MIN_PER_TASK_SAMPLE_NUM:
            warning(f"Ignore {_file} since it only has {raw_data.size} samples")
            return None, None
        assert raw_data.metainfo is not None

        raw_data.preprocess(time_lb=learning_params["ave_lb"], verbose=False)
        if raw_data.size < MIN_PER_TASK_SAMPLE_NUM:
            warning(f"Ignore {_file} since it only has {raw_data.size} samples")
            return None, None

        train_raw_data, val_raw_data = raw_data.gen_train_test_data()

        assert train_raw_data.metainfo == val_raw_data.metainfo
        ### Cache the raw data
        train_raw_data.save(_train_data_path, save_metainfo=False)
        val_raw_data.save(_val_data_path, save_metainfo=False)
        train_raw_data.metainfo.save(_metainfo_path)

    assert not train_raw_data.normalized and not val_raw_data.normalized
    return train_raw_data, val_raw_data

def load_cross_task_data(files_to_test, single_task_save_dir, root_path, learning_params, cross_task_metainfo):
    cross_task_raw_pair = None
    for _file in files_to_test:
        _task_save_dir = os.path.join(single_task_save_dir, task_repr(_file))
        _data_save_dir = os.path.join(_task_save_dir, "data")
        train_raw_data, val_raw_data = _load_task_train_test_raw_data(
            root_path, _file, learning_params, _data_save_dir)
        if train_raw_data is None:
            continue
        assert not train_raw_data.normalized and not val_raw_data.normalized
        if cross_task_raw_pair is None:
            cross_task_raw_pair = TrainTestPair(train_raw_data, val_raw_data)
        else:
            cross_task_raw_pair.train.combine(train_raw_data, inplace=True, verbose=False)
            cross_task_raw_pair.val.combine(val_raw_data, inplace=True, verbose=False)

    assert not cross_task_raw_pair.train.normalized and not cross_task_raw_pair.val.normalized

    cross_task_raw_pair.train.metainfo = cross_task_raw_pair.val.metainfo = cross_task_metainfo
    cross_task_raw_pair.train.preprocess(time_lb=learning_params["ave_lb"], verbose=False)
    cross_task_raw_pair.val.preprocess(time_lb=learning_params["ave_lb"], verbose=False)
    if cross_task_raw_pair.train.size < MIN_PER_TASK_SAMPLE_NUM or \
            cross_task_raw_pair.val.size < MIN_PER_TASK_SAMPLE_NUM:
        raise ValueError("Not enough data")

    cross_task_data_dp = TrainTestPair(MyDataSet(cross_task_raw_pair.train), MyDataSet(cross_task_raw_pair.val))
    return cross_task_data_dp


############################### Main learning process ###############################

class CrossTaskContext:
    def __init__(self, files_to_test, config_name, root_path, learning_params):
        _cross_repr = task_repr(files_to_test)
        _cross_task_save_dir = os.path.join(".workspace", "verify", config_name, "cross_task", _cross_repr)
        self._cm_save_dir = os.path.join(_cross_task_save_dir, "cm")
        self._rst_save_dir = os.path.join(_cross_task_save_dir, "rst")
        os.makedirs(self._cm_save_dir, exist_ok=True)
        os.makedirs(self._rst_save_dir, exist_ok=True)

        ### NOTE !!!: need to make sure all tasks to use the same metainfo
        self.metainfo = parse_metainfo(
            [os.path.join(root_path, file) for file in files_to_test],
            learning_params, False, verbose=False)
        self._rst_path = os.path.join(self._rst_save_dir, "rst.json")


class SingleTaskContext:
    def __init__(self, _file, single_task_save_dir):
        _repr = task_repr(_file)
        _task_save_dir = os.path.join(single_task_save_dir, _repr)
        self._data_save_dir = os.path.join(_task_save_dir, "data")
        self._cm_save_dir = os.path.join(_task_save_dir, "cm")
        self._rst_save_dir = os.path.join(_task_save_dir, "rst")
        os.makedirs(self._data_save_dir, exist_ok=True)
        os.makedirs(self._cm_save_dir, exist_ok=True)
        os.makedirs(self._rst_save_dir, exist_ok=True)

        self._rst_path = os.path.join(self._rst_save_dir, "rst.json")


def single_task_learning(_file, single_task_save_dir, root_path,
        learning_params, single_task_stat, max_step, verbose=False):

    context = SingleTaskContext(_file, single_task_save_dir)
    
    ### Load training and val raw data
    train_raw_data, val_raw_data = _load_task_train_test_raw_data(
        root_path, 
        _file,
        learning_params,
        context._data_save_dir)

    if train_raw_data is None:
        single_task_stat.append(None)
        return
    
    ### convert raw data to dataset
    train_data = MyDataSet(train_raw_data)
    val_data = MyDataSet(val_raw_data)
    if not DISABLE_CACHE and os.path.exists(context._rst_path):
        print(f"Found cached resutls at {context._rst_path}")
        ### If the single-task learning job has be run, load the results directly
        with open(context._rst_path, "r") as fp:
            train_rst = json.load(fp)
    else:
        ### Training
        _learning_params = learning_params.copy()
        _learning_params.update({
            "cache_dir": context._cm_save_dir,
            "max_step": max_step
        })
        learner, (metrics_val, metrics_train) = _metric_learning_w_tir_data_impl(
                    TrainTestPair(train_data, val_data),
                    train_data.raw_data.metainfo,
                    _learning_params,
                    verbose=verbose)
        mape_val, mape_train = float(metrics_val["mape"]), float(metrics_train["mape"]) if metrics_train is not None else None
        
        train_rst = {
            "element_wise_test_mape": evaluate_elementwise_mape(learner, val_data).tolist(),
            "train_mape": mape_train,
            "test_mape": mape_val
        }
        with open(context._rst_path, "w") as fp:
            json.dump(train_rst, fp, indent=4)

    single_task_stat.append({
        "element_wise_test_mape": train_rst["element_wise_test_mape"],
        "train_mape": train_rst["train_mape"],
        "test_mape": train_rst["test_mape"],
        "val_data": val_data
    })
    print("Train " + str(train_rst["train_mape"]))
    print("Test " + str(train_rst["test_mape"]))
        
def cross_task_learning(files_to_test, config_name, root_path,
        learning_params, single_task_save_dir, max_step, 
        single_task_stat, verbose=False):
    print(f"\nCross {files_to_test} task learning")
    context = CrossTaskContext(files_to_test, config_name, root_path, learning_params)

    cross_task_learner = ret_learner(context.metainfo, learning_params, False)
    re_train = False
    try:
        cross_task_learner.load(path=context._cm_save_dir)
    except FileNotFoundError:
        ### The cost model does not exsits
        re_train = True
    _learning_params = learning_params.copy()

    if DEBUG:
        # cross_task_learner = None
        verbose = True
        if DISABLE_CACHE:
            re_train = True
        _learning_params["tb_logdir"] = "/root/cross-device-perf-predictor/.workspace/runs/test"
        # _learning_params["load_cache"] = True
        # re_train = True
        # max_step = 1e6

    if re_train:
        cross_task_data_dp = load_cross_task_data(
            files_to_test, single_task_save_dir, root_path, learning_params, context.metainfo)
        ### Training
        _learning_params.update({
            "cache_dir": context._cm_save_dir,
            "max_step": max_step
        })
        cross_task_learner, (metrics_val, metrics_train) = _metric_learning_w_tir_data_impl(
            cross_task_data_dp,
            context.metainfo,
            _learning_params,
            verbose=verbose)
        mape_val = float(metrics_val["mape"]) if metrics_val is not None else None
        mape_train = float(metrics_train["mape"]) if metrics_train is not None else None
        print("Train " + str(mape_train))
        print("Test " + str(mape_val))

        # _val_data = single_task_stat[0]["val_data"]
        # _val_data.raw_data.unfreeze()
        # _val_data.raw_data.metainfo = cross_task_metainfo
        # _val_data.finalize_rawdata()
        # cross_task_error = evaluate_elementwise_mape(cross_task_learner, _val_data)
        # print(np.array(cross_task_error).mean())

        train_rst = {
            "train_mape": mape_train,
            "test_mape": mape_val
        }
        with open(context._rst_path, "w") as fp:
            json.dump(train_rst, fp, indent=4)
    
    return cross_task_learner, context._rst_save_dir, context.metainfo


############################### Further analysis ###############################

def load_learned_model_data_patition(
        files_to_test,
        config_name,
        root_path,
        learning_params,
        single_task_save_dir):
    ''' Load the cached cost model and training dataset, test dataset
        Evaluate the prediction error of the test dataset, partition it to
        test_best and test_worst
    Return
    ------
    cross_task_learner: learner
        Learned cost model
    cross_task_data_dp: DataPair of MyDataSet
        Train and test dataset
    best_test_raw: MyDataSet
        Test samples that show the best perf
    worst_test_raw: MyDataSet
        Test samples that show the worst perf
    context: context
        Some information about paths
    '''
    context = CrossTaskContext(files_to_test, config_name, root_path, learning_params)

    ### Load cached model
    cross_task_learner = ret_learner(context.metainfo, learning_params, False)
    cross_task_learner.load(path=context._cm_save_dir)

    ### Load data and evaluate elementwise MAPE of the test data
    cross_task_data_dp = load_cross_task_data(
            files_to_test, single_task_save_dir, root_path, learning_params, context.metainfo)
    elementwise_mape = evaluate_elementwise_mape(cross_task_learner, cross_task_data_dp.val)
    test_error = elementwise_mape.mean()
    print(f"Training data size: {len(cross_task_data_dp.train)}, Test data size: {len(cross_task_data_dp.val)}, Test error {test_error}")

    ### Partition the test data according to MAPE
    mape_sorted_idx = np.argsort(elementwise_mape)
    sample_num = len(cross_task_data_dp.val)//3
    best_test_raw = cross_task_data_dp.val.raw_data.subset(mape_sorted_idx[:sample_num])
    worst_test_raw = cross_task_data_dp.val.raw_data.subset(mape_sorted_idx[-sample_num:])

    print(f"Avg Error of the BEST {sample_num} Samples D_best_test: {elementwise_mape[mape_sorted_idx[:sample_num]].mean()}")
    print(f"Avg Error of the WORST {sample_num} samples D_worst_test: {elementwise_mape[mape_sorted_idx[-sample_num:]].mean()}")

    return cross_task_learner, cross_task_data_dp, MyDataSet(best_test_raw), MyDataSet(worst_test_raw), context

def continuous_cross_task_learn(
        files_to_test,
        config_name,
        root_path,
        learning_params,
        single_task_save_dir,
        max_step, verbose=True):
    ''' 
        1. Train the model based on D_train, evaluate on D_test, get D_test_best and D_test_worst
        2. Train the model based on D_train + D_test_worst, evaluate on D_test
    '''
    cross_task_learner, cross_task_data_dp, best_test_data, worst_test_data, context = load_learned_model_data_patition(
        files_to_test,
        config_name,
        root_path,
        learning_params,
        single_task_save_dir)

    ### Construct new training data
    assert len(best_test_data) == len(worst_test_data)
    sample_num = len(best_test_data)
    new_train_data = cross_task_data_dp.train.combine(worst_test_data)
    new_train_data.finalize_rawdata()
    new_dp = TrainTestPair(new_train_data, cross_task_data_dp.val)
    print(f"Updated: Training data size: {len(new_dp.train)}, Test data size: {len(new_dp.val)}")

    ### Train with original Training data + Wost Test Data
    _learning_params = learning_params.copy()
    _learning_params.update({
            "test_data": None,
            "max_epoch": None,
            "max_step": max_step,
            "load_cache": False,
            "cache_dir": f"{context._cm_save_dir}_continuous",
            "tb_logdir": "/root/cross-device-perf-predictor/.workspace/runs/test"
        })
    cross_task_learner, (metrics_val, metrics_train) = _metric_learning_w_tir_data_impl(
            new_dp,
            context.metainfo,
            _learning_params,
            verbose=verbose)
    
    mape_train = float(metrics_train["mape"]) if metrics_train is not None else None
    mape_val = float(metrics_val["mape"]) if metrics_val is not None else None
    error4best_test = evaluate_elementwise_mape(cross_task_learner, best_test_data).mean()
    error4worst_test = evaluate_elementwise_mape(cross_task_learner, worst_test_data).mean()
    print(f"Training Error={mape_train}, Test Error={mape_val}")
    print(f"Avg Error of the BEST {sample_num} Samples D_best_test: {error4best_test}")
    print(f"Avg Error of the WORST {sample_num} samples D_worst_test: {error4worst_test}")

def sensitive_analyze(
        files_to_test,
        config_name,
        root_path,
        learning_params,
        single_task_save_dir,
        evaluate_y=True):

    cross_task_learner, cross_task_data_dp, best_test_data, worst_test_data, context = load_learned_model_data_patition(
        files_to_test,
        config_name,
        root_path,
        learning_params,
        single_task_save_dir)

    if False:
        import shap
        model = cross_task_learner.model
        cross_task_learner.monitor.clear_hooks()
            
        dataloader = torch.utils.data.DataLoader(
            cross_task_data_dp.train,
            batch_size=100,
            shuffle=True)
        x, y = cross_task_learner.data_to_train_device(*(next(iter(dataloader))[:2]))
        background = x

        # explain the model's predictions using SHAP
        explainer = shap.DeepExplainer(model, background)
        import code
        code.interact(local=locals())
        shap_values = explainer.shap_values(x[:1])
        print(shap_values)
        exit()

    test_x, test_y = cross_task_data_dp.val.get_all_data()

    if False:
        ### x or Y to error scatter
        elementwise_mape = evaluate_elementwise_mape(cross_task_learner, (test_x, test_y))
        import code
        code.interact(local=locals())

        def __plot(xaxis_data, xaxis_name):
            if isinstance(xaxis_name, list):
                assert isinstance(xaxis_data, list) and len(xaxis_data) == len(xaxis_name)
                fig = plt.figure(figsize=(12, 5))
                _base = fig_base(len(xaxis_name))
                for id, (_data, _name) in enumerate(zip(xaxis_data, xaxis_name)):
                    ax = fig.add_subplot(_base + id + 1)
                    ax.scatter(_data, elementwise_mape, alpha=0.3, edgecolors='none')
                    plt.xlabel(_name)
                    plt.ylabel("Error")    
            else:
                fig = plt.figure(figsize=(12, 5))
                ax = fig.add_subplot(111)
                ax.scatter(xaxis_data, elementwise_mape, alpha=0.3, edgecolors='none')
                plt.xlabel(xaxis_name)
                plt.ylabel("Error")
            plt.tight_layout()
            plt.savefig(os.path.join(f"xy2error.png"))
            plt.close()

        __plot(test_y, "Y")
        _data_ls, _name_ls = [], []
        for i in range(test_x.shape[1]):
            if i in INVALID_ENTRIES:
                continue
            _data_ls.append(test_x[:, i])
            _name_ls.append(f"x[{i}]")
            if len(_name_ls) >= 9:
                __plot(_data_ls, _name_ls)
                _data_ls, _name_ls = [], []
                input("continue")
        if len(_name_ls) > 0:
            __plot(_data_ls, _name_ls)
        exit()

    if True:
        ### x to y scatter
        import seaborn as sns
        train_x, train_y = cross_task_data_dp.train.get_all_data()
        best_test_x, _ = best_test_data.get_all_data()
        worst_test_x, _ = worst_test_data.get_all_data()
        def __plot(_input):
            fig = plt.figure(figsize=(12, 5))
            _base = fig_base(len(_input))
            for id, (train_data, test_data, test_best, test_worst, _name) in enumerate(_input):
                ax = fig.add_subplot(_base + id + 1)
                # ax.scatter(test_data, test_y, alpha=0.1, edgecolors='none', label="Test")
                # ax.scatter(train_data, train_y, alpha=0.1, edgecolors='none', label="Train")
                # plt.legend()
                
                ax.boxplot([test_data, test_best, test_worst, train_data], labels=["Test", "Test Best", "Test Worst", "Train"])
                plt.ylabel("Percentile")

                plt.xlabel(_name)
                
            plt.tight_layout()
            plt.savefig(os.path.join(f"x2y.png"))
            plt.close()

        _input = [] 
        for i in range(test_x.shape[1]):
            if i in INVALID_ENTRIES:
                continue
            _input.append((train_x[:, i], test_x[:, i], best_test_x[:, i], worst_test_x[:, i], f"x[{i}]"))
            if len(_input) >= 9:
                __plot(_input)
                _input = []
                input("continue")
        if len(_input) > 0:
            __plot(_input)
        exit()
    ### Rst dir
    worker_dir = ".workspace/sensitive_analyze"
    os.makedirs(worker_dir, exist_ok=True)
    print("\n" + "#"*100)
    if evaluate_y:
        rst_path = os.path.join(worker_dir, "rst_of_y.json")
        origin_value = evaluate_y_func(cross_task_learner, (test_x, test_y)).mean()
        print(f"Original average Y {origin_value * 100:.3f}")
    else:
        rst_path = os.path.join(worker_dir, "rst.json")
        origin_value = evaluate_elementwise_mape(cross_task_learner, (test_x, test_y)).mean()
        print(f"Original test error {origin_value * 100:.3f}")
    if os.path.exists(rst_path):
        with open(rst_path, 'r') as fp:
            all_rst = json.load(fp) 
    else:
        all_rst = {}
    modified = False
    noise_form_list = ["+", "-", "*", "/"]
    noise_scale_list = np.arange(1, 9) / 10.
    # noise_form_list = ["+"]
    # noise_scale_list = [0.1]

    ### Start to analyze sensitivity
    fig = plt.figure(figsize=(12, 10))
    # _fig_base = fig_base(4, row_first=True)
    _fig_base = 410
    colors = ["r", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "grey", "black"]

    entry_num = test_x.shape[1]
    sample_num = test_x.shape[0]
    for idx, noise_form in enumerate(noise_form_list):
        ax = fig.add_subplot(_fig_base+idx+1)
        if noise_form not in all_rst:
            all_rst[noise_form] = {}
            modified = True
        for scale_idx, noise_scale in enumerate(noise_scale_list):
            noise_scale_key = str(noise_scale)
            if noise_scale_key not in all_rst[noise_form]:
                all_rst[noise_form][noise_scale_key] = []
                for entry_id in range(entry_num):
                    X = test_x.copy()
                    noise = np.ones(sample_num) * noise_scale
                    if noise_form == "+":
                        X[:, entry_id] += noise
                    elif noise_form == "-": 
                        X[:, entry_id] -= noise
                    elif noise_form == "*":
                        X[:, entry_id] *= (1 + noise)
                    elif noise_form == "/":
                        X[:, entry_id] *= (1 - noise)

                    if evaluate_y:
                        new_value = evaluate_y_func(cross_task_learner, (X, test_y)).mean()
                        print(f"Entry {entry_id}, noise {noise_form} {noise_scale} ==> Y {new_value * 100:.3f}")
                    else:
                        new_value = evaluate_elementwise_mape(cross_task_learner, (X, test_y)).mean()
                        print(f"Entry {entry_id}, noise {noise_form} {noise_scale} ==> Error {new_value * 100:.3f}")
                    all_rst[noise_form][noise_scale_key].append(float(new_value - origin_value))
                modified = True
            error_decay = all_rst[noise_form][noise_scale_key]
            ax.plot(np.arange(entry_num), error_decay, c=colors[scale_idx], label=f"noise scale={noise_scale}")
        plt.xlabel("Feature Entry ID")
        if evaluate_y:
            plt.ylabel("Predicted Y")
        else:
            plt.ylabel("Test Error Decay (%)")
        if idx == 0:
            plt.legend(bbox_to_anchor=(0., 1.5, 0.8, .102), ncol=len(all_rst[noise_form])//2)
        if noise_form == "*":
            plt.title(f"X[entry] *= 1+noise_scale")
        elif noise_form == "/":
            plt.title(f"X[entry] *= 1-noise_scale")
        else:
            plt.title(f"X[entry] {noise_form}= noise_scale")
    plt.tight_layout()
    if evaluate_y:
        plt.savefig(os.path.join(worker_dir, "senstive_y.png"))
    else:
        plt.savefig(os.path.join(worker_dir, "senstive.png"))

    ### Cache the results
    if modified:
        with open(rst_path, 'w') as fp:
            json.dump(all_rst, fp, indent=4)
    
    ### Statistic the sensitive for each entry
    sensitive_table = []
    for noise_form in all_rst.keys():
        for noise_scale, sensitive_list in all_rst[noise_form].items():
            sensitive_array = np.abs(np.array(sensitive_list))
            sensitive_table.append(sensitive_array / np.sum(sensitive_array))
    sensitive = np.array(sensitive_table).mean(0)

    ### Split the test data
    best_test_x, _ = best_test_data.get_all_data()
    worst_test_x, _ = worst_test_data.get_all_data()
    final_rst = {
        "sensitive": list(sensitive)
    }

    ### Analyze the distribution of each entry
    train_x, _ = cross_task_data_dp.train.get_all_data()
    def distribution_by_entry(name, _data):
        assert len(_data.shape) == 2
        entry_mean = np.maximum(0, _data.mean(axis=0))
        entry_std = _data.std(axis=0)
        sample_size = _data.shape[0]
        final_rst[name] = {
            "entry_mean": list(entry_mean),
            "entry_std": list(entry_std),
            "sample_num": int(sample_size)
        }
    import code
    code.interact(local=locals())
    distribution_by_entry("train", train_x)
    distribution_by_entry("best_test", best_test_x)
    distribution_by_entry("worst_test", worst_test_x)

    with open(os.path.join(os.path.dirname(rst_path), f"stat_{os.path.basename(rst_path)}"), 'w') as fp:
        json.dump(final_rst, fp, indent=4)

def train_classifier(
        files_to_test,
        config_name,
        root_path,
        learning_params,
        single_task_save_dir,
        classify_model="random_forest"):

    cross_task_learner, cross_task_data_dp, best_test_data, worst_test_data, context = load_learned_model_data_patition(
        files_to_test,
        config_name,
        root_path,
        learning_params,
        single_task_save_dir)

    ### Prepare dataset for classsification
    from utils.util import random_select

    best_val_raw, best_test_raw = random_select(best_test_data.raw_data, int(len(best_test_data)*0.95))
    best_val_data, best_test_data = MyDataSet(best_val_raw), MyDataSet(best_test_raw)

    worst_val_raw, worst_test_raw = random_select(worst_test_data.raw_data, int(len(best_test_data)*0.95))
    worst_val_data, worst_test_data = MyDataSet(worst_val_raw), MyDataSet(worst_test_raw)

    def _create_classify_dataset(_dataset: MyDataSet, label):
        _x, _ = _dataset.get_all_data()
        return np.concatenate((np.ones((_x.shape[0], 1)) * label, _x), axis=1)

    clf_y_x_best_val = _create_classify_dataset(best_val_data, 1)
    clf_y_x_worst_val = _create_classify_dataset(worst_val_data, 0)
    clf_y_x_pos = clf_y_x_best_val
    clf_y_x_neg = clf_y_x_worst_val

    clf_y_x_best_test = _create_classify_dataset(best_test_data, 1)
    clf_y_x_worst_test = _create_classify_dataset(worst_test_data, 0)

    if True:
        # clf_y_x_pos = np.concatenate((
        #     _create_classify_dataset(cross_task_data_dp.train, 1),
        #     clf_y_x_pos), axis=0)

        ### Down-sampling
        # clf_y_x_pos = random_select(clf_y_x_pos, clf_y_x_neg.shape[0])[0]

        ### Up-sampling
        while clf_y_x_neg.shape[0] < clf_y_x_pos.shape[0]:
            if clf_y_x_pos.shape[0] >= 2* clf_y_x_neg.shape[0]:
                clf_y_x_neg = np.concatenate((clf_y_x_neg, clf_y_x_neg), axis=0)
            else:
                clf_y_x_neg = np.concatenate((clf_y_x_neg, clf_y_x_neg[:(clf_y_x_pos.shape[0] - clf_y_x_neg.shape[0])]), axis=0)

    clf_y_x = np.concatenate((clf_y_x_pos, clf_y_x_neg), axis=0)
    print(f"Classification, training size = {clf_y_x.shape[0]} = {clf_y_x_pos.shape[0]} pos + {clf_y_x_neg.shape[0]} neg")

    def evaluate_accu(name, _test_y_x, infer_fn):
        _x = _test_y_x[:, 1:]
        _y = _test_y_x[:, 0]
        pred = infer_fn(_x)
        accu = accuracy_score(_y.flatten(), pred.flatten()) * 100
        print(f"{name} Accu: {accu:.3f}%")

    from sklearn.metrics import accuracy_score
    if classify_model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=None, random_state=0)
        clf.fit(clf_y_x[:, 1:], clf_y_x[:, 0])
        infer_fn = lambda x: clf.predict(x)
    elif classify_model == "xgb":
        from xgboost import XGBClassifier
        xgb_params = {
            "max_depth": 3,
            "gamma": 0.0001,
            "min_child_weight": 1,
            "subsample": 1.0,
            "eta": 0.3,
            "lambda": 1.00,
            "alpha": 0,
            "objective": "reg:linear",
            # "verbosity": 2
        }
        xgb = XGBClassifier(use_label_encoder=False, **xgb_params)
        xgb.fit(clf_y_x[:, 1:], clf_y_x[:, 0], 
            eval_metric="auc",
            eval_set=[(clf_y_x[:, 1:], clf_y_x[:, 0])],
            verbose=True)
        infer_fn = lambda x: xgb.predict(x)
    else:
        raise
    
    evaluate_accu("clssification train", clf_y_x, infer_fn)
    evaluate_accu("On val best", clf_y_x_best_val, infer_fn)
    evaluate_accu("On val worst", clf_y_x_worst_val, infer_fn)
    evaluate_accu("On test best", clf_y_x_best_test, infer_fn)
    evaluate_accu("On test worst", clf_y_x_worst_test, infer_fn)

def compare_single_cross_task_error(
        single_task_error,
        cross_task_error,
        task_name,
        save_dir,
        single_task_mape_pair: TrainTestPair
    ):
    ### visualize
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 6))
    _fig_base = 110
    ax = fig.add_subplot(_fig_base+1)
    x = np.arange(len(single_task_error))
    ax.plot(x, single_task_error, label="Single-task")
    ax.plot(x, cross_task_error, label="Cross-task")
    ax.plot(x, np.ones_like(x) * 0.1, '-', label="MAPE=0.1")
    plt.xlabel("Test Samples")
    plt.ylabel("MAPE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{task_name}_singleVScross_task_error.png"))
    plt.close()

    total_sample_num = len(single_task_error)
    _str = "#"*50 + f" {task_name} " + "#"*50
    _str += f"\nSingle task error: train MAPE={single_task_mape_pair.train*100:.3f} %,"
    _str += f"test MAPE={single_task_mape_pair.val*100:.3f} %, {total_sample_num} test samples\n"
    _str += "SE denotes the single-task error, CE denotes the corss-task error.\n"
    _str += f"Overall SE={np.array(single_task_error).mean()*100:.3f}%, CE={np.array(cross_task_error).mean()*100:.3f}%\n"

    table = prettytable.PrettyTable()
    table.field_names = ["l", "r", "SE<l", "l<=SE<r", "SE>r", "CE<l", "l<=CE<r", "CE>r", "flip"]

    for lower_before, upper_after in [(0.2, 0.1), (0.2, 0.05), (0.3, 0.1), (0.4, 0.1), (0.5, 0.1)]:
        cnt4single = [0, 0, 0]
        cnt4cross = [0, 0, 0]
        flip_cnt = 0.
        for i in range(total_sample_num):
            if single_task_error[i] >= lower_before:
                cnt4single[2] += 1
            elif single_task_error[i] >= upper_after:
                cnt4single[1] += 1
            else:
                cnt4single[0] += 1
            
            if cross_task_error[i] >= lower_before:
                cnt4cross[2] += 1
            elif cross_task_error[i] >= upper_after:
                cnt4cross[1] += 1
            else:
                cnt4cross[0] += 1

            if single_task_error[i] >= lower_before and cross_task_error[i] < upper_after:
                flip_cnt += 1
        cnt4single = np.array(cnt4single)
        cnt4cross = np.array(cnt4cross)
        row = [str(upper_after), str(lower_before)]
        row += [str(cnt) for cnt in cnt4single]
        row += [str(cnt) for cnt in cnt4cross]
        row += [str(flip_cnt)]
        table.add_row(row)

        row = ["", "Ratio"]
        row += [f"{cnt/total_sample_num*100:.3f}%" for cnt in cnt4single]
        row += [f"{cnt/total_sample_num*100:.3f}%" for cnt in cnt4cross]
        row += [f"{flip_cnt/total_sample_num*100:.3f}%"]
        table.add_row(row)
    
    table.align = 'l'
    table_str = _str + table.get_string()
    with open(os.path.join(save_dir, f"{task_name}.txt"), 'w') as fp:
        fp.write(table_str)
    print("\n" + table_str)

def verify_cross_task_similarity(
        _rst_save_dir, _TARGET_TASK_ID, single_task_stat,
        cross_task_metainfo, cross_task_learner, files_to_test,
        verbose=False):
    ''' 
        1. Learn a cost model for each task t, C_t, based on the training data D_t_train, 
           then we get D_t_test^good and D_t_test^bad with corresponding error E_t_test^good 
           and E_t_test^bad
        2. Train a cross-task cost model \hat{C}. Then test the performance on D_t_test^bad, 
           calculate the error \hat{E_t}_test^bad
        3. Compare E_t_test^bad to \hat{E_t}_test^bad, see if there any samples that have 
           better performance with \hat{C}, if so, partition is necessary
    '''
    ### Compare the single task error and cross-task error
    save_dir = _rst_save_dir
    for stat_idx, task_idx in enumerate(_TARGET_TASK_ID):
        if single_task_stat[stat_idx] is None:
            continue
        
        ### The cached dataset is normalized by the single-task metainfo, 
        # should clean the normalized data and re-normalize using the cross-task metainfo
        _val_data = single_task_stat[stat_idx]["val_data"]
        _val_data.raw_data.unfreeze()
        _val_data.raw_data.metainfo = cross_task_metainfo
        _val_data.finalize_rawdata()
        cross_task_error = evaluate_elementwise_mape(cross_task_learner, _val_data)

        single_task_mape_pair = TrainTestPair(
            single_task_stat[stat_idx]["train_mape"],
            single_task_stat[stat_idx]["test_mape"])
        task_name = task_repr(files_to_test[task_idx])
        single_task_error = single_task_stat[stat_idx]["element_wise_test_mape"]
        
        compare_single_cross_task_error(
            single_task_error,
            cross_task_error,
            task_name,
            save_dir,
            single_task_mape_pair
        )


############################### Cross_task_analyze entry ###############################

def cross_task_analyze_entry(trace_root_path, learning_params, verbose=False):

    ### Disable the filter for outlier removing
    cdpp_env.PROJECT_CFG["FILTERS"][2] = 0

    device2task = sample_task_files(trace_root_path, 
        learning_params["mode"], learning_params["gpu_model"])
    _device = device2task.devices[0]
    root_path = device2task[_device]["root_path"]
    files_to_test = device2task[_device]["tasks"]
    config_name = f"{cdpp_env.PROJECT_CFG['cfg_name']}"
    max_step = 300e3

    # max_step = 2e3
    # verbose = True
    # learning_params["load_cache"] = True

    ### Single task learning
    single_task_save_dir = os.path.join(".workspace", "verify", config_name, "single_task")
    single_task_stat = []

    _TARGET_TASK_ID = TARGET_TASK_ID if TARGET_TASK_ID is not None else list(range(len(files_to_test)))

    ### Store the loaded data in the following format
    # key=task_file_repr, value=(train_raw_data, test_raw_data)
    for task_idx, _file in enumerate(files_to_test):
        if task_idx not in _TARGET_TASK_ID:
            continue
        
        print(f"\n\nTask {_file} {task_idx+1}/{len(files_to_test)}")

        single_task_learning(
            _file, single_task_save_dir, root_path, learning_params, single_task_stat, max_step, verbose=verbose)
    
    ### Cross-task learning
    cross_task_learner, _rst_save_dir, cross_task_metainfo = cross_task_learning(
        files_to_test, config_name, root_path, learning_params,
        single_task_save_dir, max_step, single_task_stat)

    if CONTINUOUS_LEARN:
        continuous_cross_task_learn(files_to_test, config_name, root_path,
            learning_params, single_task_save_dir, max_step, verbose=True)
        exit(0)
    
    if SENSITIVE_ANALYZE:
        sensitive_analyze(files_to_test, config_name, root_path,
            learning_params, single_task_save_dir)
        exit(0)
    
    if TRAIN_CLASSIFIER:
        train_classifier(files_to_test, config_name, root_path,
            learning_params, single_task_save_dir)
        exit(0)
    
    verify_cross_task_similarity(
        _rst_save_dir, _TARGET_TASK_ID, single_task_stat,
        cross_task_metainfo, cross_task_learner, files_to_test)