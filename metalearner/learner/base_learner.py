import os
import numpy as np
from tqdm import tqdm
import time
import pickle
from datetime import datetime
from typing import Union, List, Tuple
import random
import yaml

import torch
import torch.nn as nn

from dpro import logger_utils

from tvm_helper.metadata import DataMetaInfo

from utils.util import (notify, warn_once, read_yes, warning, INFINITE_ERROR,
    LARGEST_MAPE, TrainTestPair, partition_data_based_on_train_rst)
import utils.env as cdpp_env
import utils.metrics as learning_metric

from metalearner.monitor import Monitor, visual_error, ProgressChecker
from metalearner.data.rawdata import ASTRawData
from metalearner.data.dataloader import (
    MyDataSet,
    LargeIterableDataset,
    WrapDataloader,
    LargeIterDataloader,
    load_iter_dataset,
    load_train_iter_dataset,
    load_val_iter_dataset
)
from metalearner.data.preprocess import make_dataset, DataPartitioner

from metalearner.feature import is_feature_type, ALL_FEATURE_TYPE
from metalearner.learn_utils import CMOutput, CMFeature


class NoEnoughBatchError(ValueError):
    pass

def weights_init_uniform(m):
    # classname = m.__class__.__name__
    # # for every Linear layer in a model..
    # if classname.find('Linear') != -1:
    #     # apply a uniform distribution to the weights and a bias=0
    #     # m.weight.data.uniform_(0.0, 1.0)
    #     # m.bias.data.fill_(0)
    #     nn.init.kaiming_normal_(m.weight)
    #     nn.init.kaiming_normal_(m.weight)

    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d, nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): weights_init_uniform(l)

class StepState:
    def __init__(self, step, train_loss, train_error, val_error):
        self.timestamp = time.time()
        self.step = step
        self.train_loss = train_loss
        self.train_error = train_error
        self.val_error = val_error

class StopChecker:
    def __init__(self):
        self.stats = []

        self.check_len = 10
        # self.time_budget = 1 * 3600
        self.time_budget = None

        ### Configurable variables to decide when to stop training
        self.max_step = None
        self.max_epoch = 5000
        
        self.stop = False
        self.st = time.time()

        self.best_error = None
        
    def record(self, step, train_loss, train_error, val_error):
        ''' Return True if this step has the smallest error
        '''
        self.stats.append(StepState(step, train_loss, train_error, val_error))
        if self.best_error is None or val_error < self.best_error:
            self.best_error = val_error
            return True
        else:
            return False
    
    def is_converged(self):
        if len(self.stats) < self.check_len:
            return False
        loss_list = [state.train_loss for state in self.stats]
        check_range = np.array(loss_list[-self.check_len:])
        std = np.std(check_range)
        avg = np.average(check_range)
        if avg < 10 and (std / avg) < 1e-4:
            self.stop = True
            notify("Stop training because training converges")
            return True
        else:
            return False
    
    def is_timeout(self):
        if self.time_budget is not None and (time.time() - self.st) >= self.time_budget:
            self.stop = True
            notify(f"Stop training because timeout {self.time_budget/3600}h")
            return True
        else:
            return False
    
    ### TODO (delete load and stop)
    def load(self, _path):
        with open(os.path.join(_path, "learning_state.pickle"), "rb") as f:
            self.time_budget, self.max_epoch = pickle.load(f)
    
    def save(self, _path):
        with open(os.path.join(_path, "learning_state.pickle"), "wb") as f:
            pickle.dump([
                    self.time_budget,
                    self.max_epoch
                ], f)
    
    def prune_errors(self, error_pair):
        idx = len(error_pair) - 1
        while idx > 0:
            if error_pair[idx][1] > 10:
                return error_pair[idx+1:]
            idx -= 1
        return error_pair

    def forecasted_error(self, max_step=None, max_time=None, yaxis="train_error"):
        ''' Forecast the error in a specific step or time, or when training converges

        * Note: max_time should be relative time in second
        '''
        if len(self.stats) < 10:
            return LARGEST_MAPE, None
        assert max_step is None or max_time is None, \
            "max_step and max_time can not be set at the same time"
        if max_step is not None:
            x = [state.step for state in self.stats]
            target_x = max_step
        elif max_time is not None:
            x = [state.timestamp - self.st for state in self.stats]
            target_x = max_time
        else:
            ### No step number or timepoint is specified, predict the error when training converges
            x = [state.step for state in self.stats]
            target_x = None
        
        y = [state.__dict__[yaxis] for state in self.stats]
        scalars = self.prune_errors(list(zip(x, y)))
        if len(scalars) <= 2:
            estimated_error = LARGEST_MAPE
        else:
            estimated_error = learning_metric.forecast_convergence_value(scalars, x=target_x)
        return float(estimated_error), scalars

    def check_epoch_step_num(self, monitor: Monitor):
        if self.max_step is not None and monitor.train_step > self.max_step:
            self.stop = True
            notify(f"Stop training because max train step {self.max_step} is met")
        elif self.max_epoch is not None and monitor.epoch_cnt > self.max_epoch:
            self.stop = True
            notify(f"Stop training because max train epoch {self.max_epoch} is met")
        else:
            self.stop = False

def trace_handler(p):
    # export trace data when traces ready (schedule cycle ends)
    p.export_chrome_trace("./tmp/traces/trace_" + str(p.step_num) + ".json")

class BaseLearner:
    def __init__(self, 
            data_meta_info=None, 
            cache_path=None,
            tb_log_dir=None,
            debug=False,
            log_level="info",
            no_prompt=False):

        self.init_cache_path(cache_path=cache_path, tb_log_dir=tb_log_dir, debug=debug)
        self.init_logger(log_level)
        self.model: Union[None, nn.Module] = None

        self.train_device = torch.device(os.environ.get('train_device', "cuda"))
        self.store_device = torch.device(os.environ.get('store_device', "cuda"))
        
        self.data_meta_info = data_meta_info
        if self.data_meta_info:
            self.input_len = self.data_meta_info.feature_len
        else:
            warning(f"Input shape is not specified, You must loaded a cached model, Or errors may occur")
            self.input_len = None

        ### Applly default configs
        self.assign_config(cdpp_env.PROJECT_CFG["cost_model"], verbose=False)

        if self.use_cmd_regular:
            if no_prompt:
                notify("[Learner] Use CMD regularization term")
            elif not read_yes("[Learner] Use CMD regularization term?"):
                exit(0)
                
        ### init with None
        self.optimizer = None
        self.scheduler = None
        
        ### TODO (delete)
        self.cmd_regularizer = None

        self.domain_diff_metric = None

        ### Used to stat training status
        self.stop_checker = StopChecker()

        ### Used for validation
        self.cached_data: dict[str, Union[None, List[Tuple], torch.utils.Dataloader]] = {"train": None, "val": None}

        self.profile = None
        # self.profile = torch.profiler.profile(
        #     schedule=torch.profiler.schedule(
        #         skip_first=10,
        #         wait=2,
        #         warmup=2,
        #         active=6,
        #         repeat=0
        #     ),
        #     on_trace_ready=trace_handler,
        #     with_stack=True
        # )
        if self.profile:
            warn_once("Torch Profiler is enabled.")

        self.verbose = True

        self.train_loss: float = INFINITE_ERROR  # Training loss
        self.use_grad_clip = False
        if self.use_grad_clip and not read_yes("Use gradient clipping ?"):
            exit(0)
    
    def init_cache_path(self, cache_path=None, tb_log_dir=None, debug=False):
        if cache_path:
            self.cache_path = cache_path
        elif tb_log_dir:
            self.cache_path = os.path.join(tb_log_dir, "cm", BaseLearner.__name__)
        else:
            self.cache_path = None
        if self.cache_path:
            self.best_cache_path = os.path.join(self.cache_path, "best")
            os.makedirs(self.best_cache_path, exist_ok=True)
        else:
            self.best_cache_path = None

        ### Used for monitoring the trainng process
        self.monitor = Monitor(tb_log_dir=tb_log_dir, debug=debug)
        self.batch_hooks = []

    def init_logger(self, log_level):
        if self.cache_path is not None:
            if not os.path.exists(self.cache_path):
                print("Creating Cache Directory {}".format(self.cache_path))
                os.makedirs(self.cache_path)
            self.log_level = log_level
            self.logger = logger_utils.SingleLogger(
                self.cache_path, datetime.now().strftime("%Y%m%d_%H:%M:%S"), logging_level="info")
        else:
            self.logger = None

    def init_scheduler(self, step_per_epoch):
        if self.lr_scheduler == "one_cycle":
            warn_once("OneCycleLR schedulers is used")
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.lr,
                steps_per_epoch=step_per_epoch,
                epochs=self.stop_checker.max_epoch)
        elif self.lr_scheduler == "cycle":
            ### https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.lr/10,
                max_lr=self.lr*10,
                cycle_momentum=False)
        elif self.lr_scheduler == "exp":
            warn_once("ExponentialLR schedulers is used")
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.9)
        elif self.lr_scheduler == "none":
            self.scheduler = None
        else:
            raise ValueError(self.lr_scheduler)

    def _log(self, str):
        if self.logger is None:
            return
        prefix = "Time {:5.3f} s".format(time.time() - self.stop_checker.st)
        # print(f"{prefix} - {str}")
        if self.log_level.lower() == "info":
            self.logger.info(f"{prefix} - {str}")
        else:
            self.logger.debug(f"{prefix} - {str}")

    def init_optimizer(self):
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr = self.lr,
                weight_decay = self.wd
            )
        elif self.opt_type == "SparseAdam":
            self.optimizer = torch.optim.SparseAdam(
                self.model.parameters(),
                lr = self.lr
            )
        elif self.opt_type == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr = self.lr,
                momentum = 0.9,
                dampening = 0.9,
                weight_decay = self.wd
            )
        elif self.opt_type == "rms":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters()
            )
        else:
            raise ValueError(self.opt_type)
    
    def init_device(self):
        self.normed_zero_output = None
        if self.data_meta_info is not None:
            if self.use_clip:
                self.normed_zero_output = torch.Tensor(
                        [self.data_meta_info.standardize_output(0)])
            self.output_avg = self.data_meta_info.output_avg
            self.output_std = self.data_meta_info.output_std
        else:
            self.output_avg = None
            self.output_std = None

        if self.train_device:
            if self.model:
                self.model.to(self.train_device)
            if self.normed_zero_output is not None:
                self.normed_zero_output = self.normed_zero_output.to(self.train_device)

    def _inference(self, x) -> CMOutput:
        outputs = self.model(x)
        if self.use_clip is not None and self.use_clip == "hard":
            outputs.hard_clip(self.normed_zero_output)
        return outputs
    
    def _init_loss_func(self):
        raise NotImplementedError()

    def _loss(self, outputs: CMOutput, feature, y, debug=False):
        raise NotImplementedError()

    def register_hooks_for_grads(self):
        pass
    
    ### Training and test
    def data_to_train_device(self, *tensor):
        if self.train_device:
            return [t.to(self.train_device) for t in tensor]
        else:
            return tensor

    def prepare_test_pair(self, dataset_or_x, y=None, verbose=True):
        ''' Take a dataset or a pair of x, y as input, return the test x, y pair on the target device'''
        if dataset_or_x is None:
            return None, None
        elif y is not None:
            x = dataset_or_x
        elif isinstance(dataset_or_x, MyDataSet):
            dataloader = torch.utils.data.DataLoader(dataset_or_x, batch_size=len(dataset_or_x), shuffle=True)
            x, y = next(iter(dataloader))
        elif isinstance(dataset_or_x, LargeIterableDataset):
            raise ValueError(type(dataset_or_x))
            val_len = len(list(dataset_or_x))
            dataloader = torch.utils.data.DataLoader(dataset_or_x, batch_size=val_len, shuffle=False)
            x, y = next(iter(dataloader))
        else:
            raise ValueError(type(dataset_or_x))

        if verbose:
            self._log(f"Test Data shape, x={x.shape}, y={y.shape}")

        return self.data_to_train_device(x, y)
    
    def register_test_data(self, dataset, verbose=True, synthetic=False):
        if synthetic:
            _, syn_x, syn_y = self.gen_synthetic_data()
            self.cached_data["val"] = (syn_x, syn_y)
        elif isinstance(dataset, (LargeIterableDataset, list)):
            self.cached_data["val"] = dataset
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
            val_x, val_y = next(iter(dataloader))
            self.cached_data["val"] = (CMFeature(val_x), val_y)
            if self.use_cmd_regular:
                self.enable_cmd_regularizer(val_x)
    
    def gen_dataloader(self, dataset, shuffle=True):
        if isinstance(dataset, MyDataSet):
            if len(dataset) < self.batch_size:
                warn_once(f"[BaseLearner] Data size is smaller than the batch size "
                        f", use full batch ?", others=f"{len(dataset)} < {self.batch_size}")
                dataloader = WrapDataloader(dataset, batch_size=len(dataset), shuffle=shuffle, drop_last=False)
            else:
                dataloader = WrapDataloader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)
        elif isinstance(dataset, list):
            dataloader = dataset
        elif isinstance(dataset, LargeIterableDataset):
            dataloader = LargeIterDataloader(dataset, self.batch_size, enable_up_sampling=self.enable_up_sampling, shuffle=shuffle)
        else:
            raise
        
        return dataloader
    
    def add_monotir_summary(self, outputs: CMOutput, feature, y, loss):
        ### Monitor the Training process
        raise NotImplementedError()

    def train_one_batch(self, feature: CMFeature, y: torch.Tensor):
        self.model.train()

        ### Register data for training error measurement
        if self.cached_data["train"] is None:
            self.cached_data["train"] = [(feature, y)]
        else:
            if len(self.cached_data["train"]) >= 10:
                self.cached_data["train"].pop(0)
            self.cached_data["train"].append((feature, y))

        outputs = self._inference(feature)
        loss = self._loss(outputs, feature, y)
        self.add_monotir_summary(outputs, feature, y, loss)

        self.optimizer.zero_grad()
        loss.backward()

        ### Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
        
        self.optimizer.step()

        for hook in self.batch_hooks:
            hook()
        # print(self.monitor.epoch_cnt, self.monitor.train_step, loss)
        
        ### Apply step-based scheduler
        if self.lr_scheduler == "cycle":
            self.scheduler.step()

        ### Summary, check and cache
        self.monitor.step()
        if self.monitor.is_cache:
            if self.monitor.epoch_cnt > 2:
                self.check_exit()
        
        ### debug
        # if self.monitor.train_step > 3:
        #     print(self.monitor.train_step)
        #     self.stop_training()
        if self.profile:
            self.profile.step()
        
        self.train_loss = float(loss.detach().data.cpu().numpy())
        return self.train_loss

    def train_one_epoch(self, train_data):
        train_dataloader = self.gen_dataloader(train_data)

        loss_train = INFINITE_ERROR
        self.monitor.epoch_start()

        for batch_idx, (feature, y) in enumerate(train_dataloader):
            if self.stop_checker.stop:
                return loss_train

            feature, y = self.data_to_train_device(feature, y)

            loss_train = self.train_one_batch(feature, y)
            # self._log("Epoch {:3.0f}, Step {:3.0f} - Loss {:.12f}, Memory {:.3f}".format(
            #     self.monitor.epoch_cnt,
            #     self.monitor.train_step,
            #     loss_train.detach(),
            #     torch.cuda.memory_allocated(self.train_device) / (1024**3)))

        self.monitor.epoch_end()

        return loss_train
    
    def gen_synthetic_data(self):
        batch_num = int(1e6 / self.batch_size)

        if is_feature_type(ALL_FEATURE_TYPE.ast_ansor):
            fix_seq_len = cdpp_env.PROJECT_CFG.get("FIX_SEQ_LEN", None)
            max_seq_len = max(fix_seq_len) if fix_seq_len else cdpp_env.PROJECT_CFG["MAX_SEQ_LEN"]
            input_shape = [self.batch_size, max_seq_len, self.input_len]
        else:
            input_shape = [self.batch_size, self.input_len]
        output_shape = [self.batch_size, 1]
        feature = CMFeature(torch.ones(input_shape, dtype=torch.float))
        y = torch.ones(output_shape, dtype=torch.float)
        return batch_num, feature, y

    def synthetic_one_epoch(self, move_once=True):
        loss_train = None
        batch_num, syn_x, syn_y = self.gen_synthetic_data()

        self.monitor.epoch_start()
        if move_once:
            _syn_x, _syn_y = self.data_to_train_device(syn_x, syn_y)   
            for batch_idx in range(batch_num):
                if self.stop_checker.stop:
                    return loss_train
                loss_train = self.train_one_batch(_syn_x, _syn_y)
        else:
            ''' Introduce memory_copy overhead for each batch'''
            for batch_idx in range(batch_num):
                if self.stop_checker.stop:
                    return loss_train
                _syn_x, _syn_y = self.data_to_train_device(syn_x, syn_y)
                loss_train = self.train_one_batch(_syn_x, _syn_y)
        self.monitor.epoch_end()

        return loss_train
    
    def _check_load_cache(self, learning_params, ds_pair):
        if not learning_params.get("load_cache", False):
            return ds_pair

        try:
            self.load()
        except FileNotFoundError:
            warning(f"No cached cost model found under {self.cache_path}")
            return ds_pair

        finetune_cache_dir = learning_params.get("finetune_cache_dir", None)
        if finetune_cache_dir is not None:
            notify(f"[Finetune] New cache/tb dir: {finetune_cache_dir}")
            self.init_cache_path(tb_log_dir=finetune_cache_dir)

            if learning_params["finetune_cfg"] is not None:
                assert os.path.exists(learning_params["finetune_cfg"]), (f"Finetune "
                    f"path {learning_params['finetune_cfg']} doesn't exist")
                notify(f"[Finetune] New config from {learning_params['finetune_cfg']}")
                cdpp_env.PROJECT_CFG.update(cdpp_env.read_yaml_cfg(learning_params["finetune_cfg"]))
                self.assign_config(cdpp_env.PROJECT_CFG["cost_model"], verbose=True)
            
            if learning_params["finetune_datapath"] is not None:
                if learning_params["finetune_datapath"].endswith(".npy"):
                    assert os.path.exists(learning_params["finetune_datapath"])
                    notify(f"[Finetune] new traing/test data from {learning_params['finetune_datapath']}")
                    xydata = None
                    for _path in learning_params["finetune_datapath"].split(","):
                        _xydata = np.load(_path, allow_pickle=True)
                        if xydata is None:
                            xydata = _xydata
                        else:
                            xydata = np.concatenate((xydata, _xydata), axis=0)
                    raw_data = ASTRawData(xydata, self.data_meta_info, disable_filter=False)
                    dataset = MyDataSet(raw_data)
                    ds_pair = TrainTestPair(dataset, dataset)
                else:
                    notify(f"[Finetune] new traing/test data with  {learning_params['finetune_datapath']}")
                    _before_env = {"mode": learning_params["mode"]}
                    if "CDPP_DATASET_PATH" in os.environ:
                        _before_env["CDPP_DATASET_PATH"] = os.environ["CDPP_DATASET_PATH"]

                    os.environ["CDPP_DATASET_PATH"] = os.path.join(finetune_cache_dir, "finetune_dataset")
                    _mode = learning_params["finetune_datapath"]
                    if "|" in _mode:
                        ### Specify the training set and test set separately
                        
                        ### Training set
                        learning_params["mode"] = _mode.split("|")[0]
                        if not os.path.exists(DataPartitioner.tmp_dataset_dir(learning_params)):
                            make_dataset(learning_params, self.data_meta_info)
                        train_data = load_train_iter_dataset(learning_params, self.data_meta_info)

                        ### validation set
                        learning_params["mode"] = _mode.split("|")[1]
                        if not os.path.exists(DataPartitioner.tmp_dataset_dir(learning_params)):
                            make_dataset(learning_params, self.data_meta_info)
                        val_data = load_val_iter_dataset(learning_params, self.data_meta_info)

                        ds_pair = TrainTestPair(train_data, val_data)

                    else:
                        learning_params["mode"] = _mode
                        if not os.path.exists(DataPartitioner.tmp_dataset_dir(learning_params)):
                            make_dataset(learning_params, self.data_meta_info)
                        ds_pair, _ = load_iter_dataset(learning_params)

                    os.environ.update(_before_env)
                    learning_params["mode"] = _before_env["mode"]
            
            self.monitor.log_checker = ProgressChecker(10, step_based=True)
            self.monitor.summary_checker = ProgressChecker(10, step_based=True)
            learning_params["max_step"] = 5e3
            
        return ds_pair

    def _set_monitor(self, ds_pair):
        ### Set the monitor
        if isinstance(ds_pair.train, MyDataSet):
            self.monitor.log_checker = ProgressChecker(10, step_based=False)
            self.monitor.cache_checker = ProgressChecker(100, step_based=False)
            step_per_epoch = len(ds_pair.train)
        elif isinstance(ds_pair.train, list):
            self.monitor.log_checker = ProgressChecker(1, step_based=False)
            self.monitor.cache_checker = ProgressChecker(100, step_based=False)
            step_per_epoch = len(ds_pair.train)
        elif isinstance(ds_pair.train, LargeIterableDataset):
            self.monitor.log_checker = ProgressChecker(1000, step_based=True)
            self.monitor.cache_checker = ProgressChecker(1000, step_based=True)
            self.monitor.summary_checker = ProgressChecker(1000, step_based=True)
            self.stop_checker.time_budget = None
            step_per_epoch = int(2184056 / self.batch_size)
        else:
            raise
        return step_per_epoch

    def train(self, ds_pair: TrainTestPair, verbose=True, learning_params={}):

        self.verbose = verbose

        if self.verbose:
            self._log(f"Training size: {len(ds_pair.train)}, Test size: {len(ds_pair.val)}")

        ### Set the monitor
        step_per_epoch = self._set_monitor(ds_pair)

        ### Check to whether to load the cached model
        ds_pair = self._check_load_cache(learning_params, ds_pair)

        ### Set stop condition, must be executed after self._set_monitor()
        if "finetune_epoch" in learning_params:
            ### init the stop checker for continuous learning, i.e., call self.train multiple times
            self.stop_checker.max_epoch = self.monitor.epoch_cnt + learning_params["finetune_epoch"] 
        elif "max_epoch" in learning_params:
            self.stop_checker.max_epoch = learning_params["max_epoch"]
        if "max_step" in learning_params:
            self.stop_checker.max_step = learning_params["max_step"]
        self.stop_checker.check_epoch_step_num(self.monitor)

        ### Register hooks
        self.batch_hooks = [
            lambda: self.check_summary_log_cache(verbose=verbose),
            lambda: self.stop_checker.check_epoch_step_num(self.monitor)
        ]
        self.register_hooks_for_grads()

        self.init_optimizer()
        self.init_scheduler(step_per_epoch)
        self.register_test_data(ds_pair.val, verbose=verbose, 
            synthetic=learning_params.get("synthetic", False))
    
        ### TODO (huhanpeng) delete
        # self.monitor.log_checker = ProgressChecker(1, step_based=True)
        # self.monitor.summary_checker = ProgressChecker(10, step_based=True)
        # # self.lr_scheduler = "none"
        
        ### Start to train
        if verbose:
            print(f"\n[{self.__class__.__name__}] Start training with the following config: ")
            self.summary()
            self._log(f"Start to training, max_epoch={self.stop_checker.max_epoch}, max_step={self.stop_checker.max_step}")

        loss_train = None
        while not self.stop_checker.stop:
            if learning_params.get("synthetic", False):
                loss_train = self.synthetic_one_epoch()
            else:
                loss_train = self.train_one_epoch(ds_pair.train)
            if self.lr_scheduler != "none" and self.lr_scheduler != "cycle":
                self.scheduler.step()
        
        self.stop_training()

        ### Evaluation
        outputs_train, metrics_train, _ = self.forward_compute_metrics(self.cached_data["train"])
        outputs_val, metrics_val, _ = self.forward_compute_metrics(
            self.cached_data["val"], element_wise_test=("element_wise_test_mape" in learning_params))

        if loss_train is not None and metrics_val is not None:
            self.monitor.summary([
                ("scalar", "MAPE/Val", metrics_val["mape"]),
                # ("scalar", "Loss/Val", metrics_val["loss"])
            ])

        if verbose:
            self._log("Train: " + str(metrics_train))
            self._log("Val: " + str(metrics_val))
        
        ### Returned values
        if learning_params is not None:
            if "test_data" in learning_params:
                learning_params["test_data"] = self.cached_data["val"]
            if "element_wise_test_mape" in learning_params:
                learning_params["element_wise_test_mape"] = metrics_val["element-wise-mape"]
        
        return metrics_val, metrics_train
    
    def loss(self, outputs: CMOutput, dataset_or_x, y=None):
        feature, y = self.prepare_test_pair(dataset_or_x, y, verbose=False)
        return self._loss(outputs, feature, y)

    def predict(self, feature):
        self.model.eval()
        outputs = self._inference(feature)
        outputs.preds = outputs.preds.cpu().detach().numpy()
        return outputs.de_standardize(self.data_meta_info)

    def compute_metrics(self, outputs: CMOutput, labels, element_wise_test=False):
        de_norm_preds = self.data_meta_info.de_standardize_output(outputs.preds.data.cpu().numpy().flatten())
        de_norm_true = self.data_meta_info.de_standardize_output(labels.data.cpu().numpy().flatten())
        metrics = {}
        metrics["mape"] = float(learning_metric.metric_mape(de_norm_preds, de_norm_true))
        metrics["rmse"] = float(learning_metric.metric_rmse(de_norm_preds, de_norm_true))
        learning_metric.metric_mape(outputs.preds.data.cpu().numpy().flatten(), labels.data.cpu().numpy().flatten())
        error_bounds = [0.2, 0.1, 0.05]
        error_accuracy = learning_metric.metric_error_accuracy(de_norm_preds, de_norm_true, error_bounds)
        for idx, error_bound in enumerate(error_bounds):
            metrics[f"{error_bound*100:.0f}%accuracy"] = float(error_accuracy[idx])
        if element_wise_test:
            metrics["element-wise-mape"] = learning_metric.metric_elementwise_mape(
                de_norm_preds, de_norm_true
            )
        return metrics

    def forward_compute_metrics(self, input_data, element_wise_test=False):
        if input_data is None:
            return None, None, None
        self.model.eval()
        outputs: Union[None, CMOutput] = None
        labels = None
        if isinstance(input_data, (torch.utils.data.IterableDataset, list)):
            # assert isinstance(input_data, list)
            if isinstance(input_data, list):
                dataloader = input_data
            else:
                dataloader = self.gen_dataloader(input_data, shuffle=False)
            # print(f"Memory usage at the beginning of validation: {torch.cuda.memory_allocated(self.train_device)/(1024**3):.3f}GB")
            try:
                sample_cnt = 0
                for batch_idx, (_feature, _y) in enumerate(dataloader):
                    _feature, _y = self.data_to_train_device(_feature.detach(), _y.detach())
                    # print(f"Memory usage before validation step {batch_idx}: {torch.cuda.memory_allocated(self.train_device)/(1024**3):.3f}GB")
                    _outputs = self._inference(_feature)
                    _outputs.preds = _outputs.preds.detach().cpu()
                    if _outputs.embedding is not None:
                        _outputs.embedding = _outputs.embedding.detach().cpu()
                    _y = _y.cpu()
                    if outputs is None:
                        outputs = _outputs
                        labels = _y
                    else:
                        outputs.concat_to(_outputs)
                        labels = torch.cat((labels, _y), axis=0)
                    # print(f"Memory usage after validation step {batch_idx}: {torch.cuda.memory_allocated(self.train_device)/(1024**3):.3f}GB")
                    sample_cnt += len(_y)
                    # if sample_cnt >= 4*32*128:
                    #     warn_once(f"Too many data, only use {sample_cnt} samples")
                    #     break
            except Exception as e:
                warning(f"Error in forward_compute_metrics: {repr(e)}")
                # import code
                # code.interact(local=locals())
                fake_tensor = torch.tensor([1000.])
                return CMOutput(preds=fake_tensor, embedding=fake_tensor), {"loss": 1000, "mape": 1000}, fake_tensor
            if outputs is None:
                raise NoEnoughBatchError(f"Batch size {self.batch_size} is too large for the dataset")
                import code
                code.interact(local=locals())
            metrics = self.compute_metrics(outputs, labels, element_wise_test=element_wise_test)
            # metrics["loss"] = float(self._loss(outputs, None, labels).data.cpu().numpy())
        elif isinstance(input_data, tuple):
            x, y = self.data_to_train_device(*input_data)
            outputs = self._inference(x)
            metrics = self.compute_metrics(outputs, y, element_wise_test=element_wise_test)
            # metrics["loss"] = float(self.loss(outputs, x, y).data.cpu().numpy())
            labels = y
        else:
            raise ValueError(f"Invalid input data format {type(input_data)}")

        return outputs, metrics, labels

    def check_summary_log_cache(self, verbose):
        ### Summary, check and cache
        if self.monitor.is_summary or (self.monitor.is_log and verbose):
            outputs_train, metrics_train, _ = self.forward_compute_metrics(self.cached_data["train"])
            outputs_val, metrics_val, labels_val = self.forward_compute_metrics(self.cached_data["val"])
    
            if self.monitor.is_summary:
                self.monitor.summary([
                    ("scalar", "MAPE/Val", metrics_val["mape"]),
                    ("scalar", "MAPE/Train", metrics_train["mape"]),
                    ("scalar", "Loss/Train", self.train_loss),
                    ("histogram", "Label/True/Val", labels_val),
                    ("histogram", "Label/Predicted/Val", outputs_val.preds),
                ])
                if metrics_val and metrics_val["mape"] < LARGEST_MAPE:
                    is_best = self.stop_checker.record(
                                self.monitor.train_step,
                                self.train_loss,
                                metrics_train["mape"],
                                metrics_val["mape"])
                    if self.best_cache_path and is_best:
                        self.save(self.best_cache_path)
                        with open(os.path.join(self.best_cache_path, "log.txt"), 'w') as fp:
                            fp.write(f"Keep best model with error {metrics_val['mape']} at "
                                    f"epoch {self.monitor.epoch_cnt},step {self.monitor.train_step}")
        
            if verbose and self.monitor.is_log:
                self._log("Epoch {:3.0f} step {} bs {} - loss_train={:.12f}, {} ".format(
                    self.monitor.epoch_cnt,
                    self.monitor.train_step,
                    self.batch_size,
                    self.train_loss,
                    metrics_val
                ))
        if self.monitor.is_cache:
            self.save()
      
    ### Save models
    def check_exit(self):
        self.stop_checker.is_converged()
        self.stop_checker.is_timeout()

    def stop_training(self):
        self.save()
        self.monitor.close()

    def save(self, path=None):
        _path = path if path is not None else self.cache_path
        # print(f"Save model to {_path}")
        if _path is None:
            return
    
        ### Remove all hooks
        self.monitor.clear_hooks()

        with open(os.path.join(_path, "base_learner.pickle"), "wb") as f:
            pickle.dump([
                    self.input_len,
                    self.lr,
                    self.wd,
                    self.opt_type
                    ], f)
        self.monitor.save(_path)
        self.stop_checker.save(_path)
        self.data_meta_info.save(os.path.join(_path, "metainfo.pickle"))
        
        with open(os.path.join(_path, f"project_cfg.yaml"), 'w') as yaml_fp:
            yaml_fp.write(yaml.dump(cdpp_env.PROJECT_CFG, default_flow_style=False))

        if self.verbose:
            notify(f"Save model to {_path}, epoch {self.monitor.epoch_cnt}, step {self.monitor.train_step}")
        
    def load(self, path=None):
        _path = path if path is not None else self.cache_path
        if _path is None:
            raise ValueError("No path is given to load cached cost model")
        
        project_cfg_path = os.path.join(_path, f"project_cfg.yaml")
        if os.path.exists(project_cfg_path):
            cur_fix_seq_len = cdpp_env.PROJECT_CFG.get("FIX_SEQ_LEN", None)
            cdpp_env.PROJECT_CFG = cdpp_env.read_yaml_cfg(project_cfg_path)
            cdpp_env.PROJECT_CFG["CUR_FIX_SEQ_LEN"] = cur_fix_seq_len
            self.assign_config(cdpp_env.PROJECT_CFG["cost_model"], verbose=True)

        with open(os.path.join(_path, "base_learner.pickle"), "rb") as f:
            self.input_len, \
                self.lr, \
                self.wd, \
                self.opt_type = pickle.load(f)
        self.monitor.load(_path)
        self.stop_checker.load(_path)
        # self.data_meta_info = DataMetaInfo.load_init(
        #     os.path.join(_path, "metainfo.pickle"))

        if not os.path.exists(project_cfg_path):
            warning(f"[Deperacated] Project CFG file {project_cfg_path} does not Exists! "
                "Make sure the correct CFG is used.")
        notify(f"Load model from {_path}, epoch {self.monitor.epoch_cnt}, step {self.monitor.train_step}")

    @staticmethod
    def load_init(_path, child_cls):
        cdpp_env.PROJECT_CFG = cdpp_env.read_yaml_cfg(os.path.join(_path, f"project_cfg.yaml"))

        data_meta_info = DataMetaInfo.load_init(
            os.path.join(_path, "metainfo.pickle"))
        learner = child_cls(data_meta_info = data_meta_info, no_prompt=True)
        with open(os.path.join(_path, "base_learner.pickle"), "rb") as f:
            learner.input_len,\
                learner.lr,\
                learner.wd,\
                learner.opt_type = pickle.load(f)
        learner.monitor.load(_path)
        learner.stop_checker.load(_path)
        learner.assign_config(cdpp_env.PROJECT_CFG["cost_model"], verbose=True)
        learner.init_cache_path(cache_path=_path)
        notify(f"Load model from {_path}, epoch {learner.monitor.epoch_cnt}, step {learner.monitor.train_step}")
        return learner
    
    def dataset2xylist(self, _dataset):
        _dataloader = self.gen_dataloader(_dataset, shuffle=False)
        xy_list = []
        sample_cnt = 0
        for _feature, _y in _dataloader:
            xy_list.append((_feature, _y))
            sample_cnt += len(_y)
            # if sample_cnt >= 4*32*128:
            #     warn_once(f"Too many data, only use {sample_cnt} samples")
            #     break
        return xy_list, sample_cnt

    def test_on_dataset(self, _dataset, filename=None, latent=False):
        print(f"[Analyze] test the learner to generate {filename} dataset ... ")

        input_data, sample_cnt = self.dataset2xylist(_dataset)
        outputs, _metrics, _ = self.forward_compute_metrics(input_data)
        
        if filename is None:
            return _metrics, None

        with open(os.path.join(self.cache_path, f"{filename}_latent.pickle"), 'wb') as fp:
            pickle.dump([outputs.embedding.data.cpu().numpy()], fp)
        print(f"Save latent variables under {self.cache_path}")

        notify(f"{filename} ({sample_cnt}):" + str(_metrics))
        if "." not in filename:
            filename = f"{filename}.pickle"
        assert self.cache_path is not None
        _path = os.path.join(self.cache_path, filename)
        if True or not os.path.exists(_path):
            _X, _Y = zip(*input_data)
            _X = np.concatenate([_x.x_tir.data.cpu().numpy() for _x in _X], axis=0)
            _Y = np.concatenate([_y.data.cpu().numpy() for _y in _Y], axis=0)

            partition_data_based_on_train_rst(
                _X, _Y,
                outputs.preds.data.cpu().numpy(),
                self.data_meta_info, 
                _path)
        return _metrics, _path

        ### Read data
        with open("latent.pickle", "rb") as f:
            ret = pickle.load(f)

    def enable_cmd_regularizer(self, x):
        raise NotImplementedError()

    def para_cnt(self):
        ''' Given a config, return the number of the model
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        para_cnt = sum([np.prod(p.size()) for p in model_parameters])

        return para_cnt
    
    def assign_config(self, config, verbose=True):
        ''' Assign a configuration to the learner, including hyper-parameters and neural architectures
        
        ** NOTE: if this method is inherited, this method must be called BEFORE child class's methods **
        Because we need to call `torch.manual_seed(0)` to control the randomness before model initialization.
        
        '''
        
        ### Control the randomness for each trial
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        for attr in ["batch_size", "lr", "wd", "opt_type", "loss_type"]:
            if attr in config:
                self.__dict__[attr] = config[attr]
        
        if "lr_scheduler" in config:
            self.lr_scheduler = config["lr_scheduler"]
            if self.lr_scheduler is None:
                self.lr_scheduler = "none"
        
        self.use_cmd_regular = config["USE_CMD_REGULAR"]
        self.enable_up_sampling = config["enable_up_sampling"]
        self.use_clip = config["use_clip"]
        
        self._init_loss_func()
    
    def summary(self):
        print("################# Learner Summary #################")
        print(f" - bs={self.batch_size}, lr={self.lr}, LRScheduler={self.lr_scheduler}")
        print(f" - wd={self.wd}, opt={self.opt_type}, Loss: {self.loss_type}")
        print(f" - use_cmd_regular={self.use_cmd_regular}")
        print(f" ==> Parameter count: {self.para_cnt()}")
    