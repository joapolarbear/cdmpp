import os
import json
import time
import shutil
from multiprocessing import Lock
from typing import Union, Tuple

from utils.util import good_new, prompt, INFINITE_ERROR, read_yes, LARGEST_MAPE, TrainTestPair

from metalearner.learner import ret_learner, NoEnoughBatchError
from metalearner.model import TrainInfinityError

MAX_TRIAL_TIME = 12 * 3600


class TuneState:
    def __init__(self, workspace):
        self.best_target = None
        self.best_learner = None
        self.workspace = workspace
        self.best_model_dir = os.path.join(self.workspace, "best/cm")
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)
        
        self.log_filename = os.path.join(self.workspace, "tuning.log")
        self.log_fp = open(self.log_filename, 'a')
        self.best_config_filename = os.path.join(self.workspace, "best/best_cfg.json")
        self.best_config_fp = open(self.best_config_filename, "w")

        self.lock = Lock()

    def check_best(self, target, learner, _config, **kwargs):
        if target is None:
            return
        self.lock.acquire()
        suffix = f" ({_config['progress']})" if "progress" in _config else ""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if self.best_target is None or target < self.best_target:
            self.best_target = target
            self.best_learner = learner
            self.best_learner.save(self.best_model_dir)
            
            dump_info = {
                "state": 1,
                "time": ts,
                "best_target": self.best_target,
                "target": target,
                "config": _config,
            }
            dump_info.update(kwargs)

            msg = ts + " - " + \
                f"1. Update the Best Target: {self.best_target}: {_config}" + suffix
            good_new(f"{self.workspace}: {msg}")
            json.dump(_config, self.best_config_fp, indent=4)
            self.best_config_fp.flush()
        else:
            dump_info = {
                "state": 0,
                "time": ts,
                "best_target": self.best_target,
                "target": target,
                "config": _config,
            }
            dump_info.update(kwargs)

            msg = ts + " - " + \
                f"2. Keep the Best Target {self.best_target} ({target})" + suffix
            prompt(f"{self.workspace}: {msg}")
        self.log(json.dumps(dump_info))
        self.lock.release()
    
    def log(self, msg):
        self.log_fp.write(msg + "\n")
        self.log_fp.flush()


class AutoTuner:
    def __init__(self,
            data_meta_info,
            ds_pair: TrainTestPair,
            learning_params):
        self.learn_method = learning_params["metric_learner"]
        self.data_meta_info = data_meta_info
        self.ds_pair = ds_pair
        self.tune_state = TuneState(learning_params["workspace"])
        self.workspace = learning_params["workspace"]
        self.learning_params = learning_params

        self.multi_object = bool(int(os.environ.get("CDPP_MULTI_OBJECTIVE_TUNE", '0')))
        if self.multi_object:
            prompt("Multi-Objective Optimization is Enabled")

    def target_func(self, _config, reproduce=False):
        self.learning_params.update({
            "cache_dir": None,
            "debug": False,
            "tb_logdir": None
        })
        verbose = False
        trial_id = _config.get("trial_id", None)

        if reproduce:
            verbose = True
            if "trial_id" in _config:
                self.learning_params["tb_logdir"] = f".workspace/runs/autotune_trial_{_config['trial_id']}_reproduce"
            else:
                self.learning_params["tb_logdir"] = ".workspace/runs/test"
        else:
            if trial_id is not None:
                self.learning_params["tb_logdir"] = f".workspace/runs/autotune_trial_{trial_id}"
                print(f"\nTrial {trial_id}, tensorboard dir: {self.learning_params['tb_logdir']}, config: {_config}")
            
        learner = ret_learner(self.data_meta_info, self.learning_params, verbose=False)
        learner.assign_config(_config, verbose=verbose)

        target: Union[Tuple, float] = LARGEST_MAPE

        if True:
            ### For large dataset
            # print("Train 160e3 steps ...")
            max_step = 300e3
            # max_step = 3
            if max_step < 1e3:
                if not read_yes(f"Are you sure to train for such a few steps: {max_step}"):
                    exit(0)
            try:
                metrics_val, metrics_train = learner.train(
                    self.ds_pair,
                    verbose=verbose,
                    learning_params={"max_step": max_step}
                    # learning_params={"max_epoch": 200}
                    )
                # estimated_mape_val, hist_errors = learner.stop_checker.forecasted_error(max_time=MAX_TRIAL_TIME)
                # smoothed_train_mape, _ = learner.stop_checker.forecasted_error(max_step=max_step, yaxis="train_error")
                # smoothed_val_mape, _ = learner.stop_checker.forecasted_error(max_step=max_step, yaxis="val_error")
                if self.multi_object:
                    # target = (smoothed_train_mape, smoothed_val_mape)
                    target = (metrics_train["mape"], metrics_val["mape"])
                else:
                    # target = smoothed_train_mape
                    target = metrics_train["mape"]
            except (TrainInfinityError, NoEnoughBatchError):
                    target = (LARGEST_MAPE, LARGEST_MAPE) if self.multi_object else LARGEST_MAPE
        else:
            USE_ESTIMATE_ERROR = False
            LEAST_EPOCH_NUM = 1
            TRAIN_ERROR_AS_TARGET = False
            mape_val = mape_train = target = INFINITE_ERROR
            if len(self.ds_pair.train) < learner.batch_size:
                estimated_mape_val = hist_errors = None
                learner.stop_training()
            else:
                try:
                    learner.init_optimizer()
                    val_x, val_y = learner.prepare_test_pair(self.ds_pair.val, verbose=verbose)
                    for _ in range(LEAST_EPOCH_NUM):
                        loss_train = learner.train_one_epoch(self.ds_pair.train)
                    _, metrics_train, _ = learner.forward_compute_metrics(learner.cached_data["train"])
                    _, metrics_val, _ = learner.forward_compute_metrics((val_x, val_y))
                    while True:
                        if TRAIN_ERROR_AS_TARGET:
                            target = metrics_train["mape"]
                            estimated_mape_val = hist_errors = None
                        elif USE_ESTIMATE_ERROR:
                            estimated_mape_val, hist_errors = learner.stop_checker.forecasted_error(max_time=MAX_TRIAL_TIME)
                            target = estimated_mape_val
                        else:
                            estimated_mape_val = hist_errors = None
                            target = metrics_val["mape"]
                        if target is None:
                            ### Collected historical error is not enough to forecast the convergence error
                            if learner.stop_checker.stop:
                                break
                        else:
                            break
                        loss_train = learner.train_one_epoch(self.ds_pair.train)
                        _, metrics_train, _ = learner.forward_compute_metrics(learner.cached_data["train"])
                        _, metrics_val, _ = learner.forward_compute_metrics((val_x, val_y))
                    learner.stop_training()
                except TrainInfinityError:
                    estimated_mape_val = hist_errors = None
                    learner.stop_training()

            if self.tune_state:
                self.tune_state.check_best(target, learner, _config,
                    mape_val=mape_val,
                    estimated_mape_val=estimated_mape_val,
                    mape_train=mape_train,
                    epoch_cnt=learner.monitor.epoch_cnt,
                    train_step=learner.monitor.train_step,
                    hist_errors=hist_errors)

        ### check_remove_tensorboard
        if (isinstance(target, tuple) and target[0] > 1) or (isinstance(target, float) and target > 1):
            if self.learning_params["tb_logdir"] and os.path.exists(self.learning_params["tb_logdir"]):
                shutil.rmtree(self.learning_params["tb_logdir"], ignore_errors=True)
                os.system(f"rm -r {self.learning_params['tb_logdir']}")

        if reproduce:
            exit(0)
            pass

        return target

