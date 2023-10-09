import os
import numpy as np

import torch
import xgboost as xgb

from utils.util import notify, partition_data_based_on_train_rst
import utils.metrics as learning_metric

from metalearner.learner.base_learner import BaseLearner
from metalearner.data.dataloader import (
    MyDataSet,
)
from metalearner.analyze import (
    analyze_train_test_rst,
    plot_xgb_history
)
from metalearner.learn_utils import CMOutput

DefaultXGBConfig = {
    "max_depth": 3,
    "gamma": 0.0001,
    "min_child_weight": 1,
    "subsample": 1.0,
    "eta": 0.3,
    "lambda": 1.00,
    "alpha": 0,
    "objective": "reg:linear",
}

def make_metric(meta_info):
    def wrap_mape(predt: np.ndarray, data: xgb.DMatrix):
        label = data.get_label()
        de_norm_true = meta_info.de_standardize_output(label)
        de_norm_preds = meta_info.de_standardize_output(predt)
        return "MAPE", learning_metric.metric_mape(de_norm_preds, de_norm_true)
    return wrap_mape

class SPELoss:
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix):
        '''Compute the gradient square percentage error.'''
        y = dtrain.get_label()
        return 2 * (predt / y -1) / y

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix):
        '''Compute the hessian for square percentage error.'''
        y = dtrain.get_label()
        return 2 / np.power(y, 2)

    def loss(predt: np.ndarray, dtrain: xgb.DMatrix):
        '''
        '''
        grad = SPELoss.gradient(predt, dtrain)
        hess = SPELoss.hessian(predt, dtrain)
        return grad, hess

class APELoss:
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix):
        '''Compute the gradient square percentage error.'''
        y = dtrain.get_label()
        grad = 1. / y
        grad[predt < y] *= -1
        return grad

    def loss(predt: np.ndarray, dtrain: xgb.DMatrix):
        '''
        '''
        grad = APELoss.gradient(predt, dtrain)
        hess = np.zeros_like(predt)
        return grad, hess

class EmptyLoss:
    def loss(predt: np.ndarray, dtrain: xgb.DMatrix):
        grad = np.ones_like(predt)
        hess = np.ones_like(predt)
        return grad, hess

class XGBLearner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super(XGBLearner, self).__init__(*args, **kwargs)

        self.xgb_params = {
            "max_depth": 3,
            "gamma": 0.0001,
            "min_child_weight": 1,
            "subsample": 1.0,
            "eta": 0.3,
            "lambda": 1.00,
            "alpha": 0,
            "objective": "reg:linear",
        }
        self.xgb_params["verbosity"] = 0
        self.xgb_params['nthread'] = 4
        self.log_interval = 25

        self.bst = None

        self.init_device()

    def init_device(self):
        return
        if self.use_cuda != -1:
            raise NotImplementedError("Init the model with cuda here")

    def train(self, train_data, val_data, verbose=True, learning_params=None):
        if False:
            analyze_train_test_rst(
                os.path.join(self.cache_path, "rst_on_train_data.pickle"),
                os.path.join(self.cache_path, "rst_on_test_data.pickle")
            )
            exit(0)

        val_x, val_y = self.prepare_test_pair(val_data, verbose=verbose)
        val_x = val_x.numpy()
        val_y = val_y.numpy()
        dtest = xgb.DMatrix(val_x, label=val_y)

        assert isinstance(train_data, MyDataSet)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)
        train_x, train_y, train_di = next(iter(train_dataloader))
        train_x = train_x.numpy()
        train_y = train_y.numpy()
        dtrain = xgb.DMatrix(train_x, label=train_y)

        custom_results: dict[str, dict[str, list[float]]] = {}
        if self.loss_type == "MSPE":
            self.bst = xgb.train(
                self.xgb_params,
                dtrain,
                feval=make_metric(self.data_meta_info),
                evals_result=custom_results,
                evals=[(dtrain, 'Train'), (dtest, 'Test')],
                num_boost_round=1000,
                verbose_eval=False,
                callbacks=[],
                obj=SPELoss.loss
            )
        elif self.loss_type == "MSE":
            self.bst = xgb.train(
                self.xgb_params,
                dtrain,
                feval=make_metric(self.data_meta_info),
                evals_result=custom_results,
                evals=[(dtrain, 'Train'), (dtest, 'Test')],
                num_boost_round=1000,
                verbose_eval=False,
                callbacks=[]
            )
        else:
            raise ValueError()
        # plot_xgb_history(custom_results)
        
        self.save()

        outputs_train = self._inference(train_x)
        metrics_train = self.compute_metrics(outputs_train, train_y)
        
        outputs_val = self._inference(val_x)
        metrics_val = self.compute_metrics(outputs_val, val_y, element_wise_test=(learning_params is not None))

        if self.cache_path:
            partition_data_based_on_train_rst(train_x, train_y, outputs_train.preds, self.data_meta_info,
                path=os.path.join(self.cache_path, "rst_on_train_data.pickle"))
            partition_data_based_on_train_rst(val_x, val_y, outputs_val.preds, self.data_meta_info,
                path=os.path.join(self.cache_path, "rst_on_test_data.pickle"))
        
            import json
            with open(os.path.join(self.cache_path, "training_log.json"), 'w') as fp:
                json.dump(custom_results, fp, indent=4)

        if verbose:
            notify("Training: " + str(metrics_train))
            notify("Validation: " + str(metrics_val))

        if learning_params is not None:
            if "test_data" in learning_params:
                learning_params["test_data"] = (val_x, val_y)
            if "element_wise_test_mape" in learning_params:
                learning_params["element_wise_test_mape"] = metrics_val["element-wise-mape"]

        return metrics_val, metrics_train


    def _inference(self, x: np.ndarray) -> CMOutput:
        dtest = xgb.DMatrix(x)
        preds = self.bst.predict(dtest)
        return CMOutput(preds)
    
    def _loss(self, outputs: CMOutput, x: np.ndarray, y: np.ndarray, debug=False):
        raise NotImplementedError("loss function")
    
    def save(self, path=None):
        _path = path if path is not None else self.cache_path
        if _path is None:
            return
        super(XGBLearner, self).save(_path)
        self.bst.save_model(os.path.join(_path, "xgb.model"))

    def load(self, path=None):
        _path = path if path is not None else self.cache_path
        super(XGBLearner, self).load(_path)
        self.bst = xgb.Booster({"nthread": 4})
        self.bst.load_model(os.path.join(_path, "xgb.model"))
        self.init_device()
    
    def para_cnt(self):
        ''' Given a config, return the number of the model
        '''
        return None
        
        model_parameters = filter(lambda p: p.requires_grad, self.match_net.parameters())
        para_cnt = sum([np.prod(p.size()) for p in model_parameters])

        return para_cnt
        
    def register_hooks_for_grads(self):
        return

    def assign_config(self, config, verbose=True):
        super(XGBLearner, self).assign_config(config, verbose=verbose)
        
        self.xgb_params["max_depth"] = config.get("max_depth", DefaultXGBConfig["max_depth"])
        self.xgb_params["gamma"] = config.get("gamma", DefaultXGBConfig["gamma"])
        self.xgb_params["min_child_weight"] = config.get("min_child_weight", DefaultXGBConfig["min_child_weight"])
        self.xgb_params["subsample"] = config.get("subsample", DefaultXGBConfig["subsample"])
        self.xgb_params["eta"] = config.get("eta", DefaultXGBConfig["eta"])

        if verbose:
            print(f"Update the Model Architecture")
            print(self.xgb_params)
            print("\n")
        self.init_device()
