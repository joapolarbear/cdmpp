import os
import numpy as np

import torch
from torch._C import Value

from tvm_helper.tir_helper import ALL_TIR_OPS
from utils.util import notify, warning, read_yes
import utils.env as cdpp_env

from metalearner.model import MyMatchNet, ReverseLayerF, CMDRegular
from metalearner.learner.base_learner import BaseLearner, weights_init_uniform
from metalearner.learn_utils import CMOutput

class MyMatchNetLearner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super(MyMatchNetLearner, self).__init__(*args, **kwargs)

        self.model = MyMatchNet(input_len=self.input_len)
        self.init_device()
        self.model.apply(weights_init_uniform)

        self.enable_similarity = cdpp_env.PROJECT_CFG["cost_model"]["enable_similarity"]
        self.enable_adversarial = cdpp_env.PROJECT_CFG["cost_model"]["enable_adversarial"]

        ### Debug
        self.enable_entry_regular = False
        if self.enable_entry_regular:
            if not read_yes("Use Entry-sensitivity-based Regular"):
                exit(0)
            import json
            worker_dir = ".workspace/sensitive_analyze"
            rst_path = os.path.join(worker_dir, "stat_rst_of_y.json")
            with open(rst_path, 'r') as fp:
                stat_rst = json.load(fp)
            self.entry_regular_weight = torch.tensor(np.array((stat_rst["sensitive"]))).to(self.train_device)
        else:
            self.entry_regular_weight = None

    def _loss(self, outputs: CMOutput, x, y):
        ### Only evaluate loss value, do not monitor
        
        ### TODO (huhanpeng) remove the assertation
        assert outputs.preds.shape == y.shape
        ### Select the loss function to use
        if self.loss_type == "MSE":
            loss = torch.nn.MSELoss()(outputs.preds, y)
        elif self.loss_type == "MSPE":
            loss = torch.mean(torch.pow((outputs.preds - y) / y, 2))
        elif self.loss_type == "MAPE":
            loss = torch.mean(torch.abs((outputs.preds - y) / y))
        else:
            raise ValueError(self.loss_type)
        
        if self.use_clip is not None and self.use_clip == "soft":
            residual = torch.nn.ReLU()(self.normed_zero_output - outputs.preds)
            if self.use_mse_loss:
                _loss = torch.linalg.norm(residual)
            else:
                _loss = torch.mean(residual / y)
            loss += _loss

        ### CMD egularizer
        if self.cmd_regularizer:
            _loss = self.cmd_regularizer(outputs.embedding)
            loss += _loss
        
        if self.enable_entry_regular:
            loss += 0.1 * self.model.regularizer(self.entry_regular_weight)

        if self.enable_adversarial:
            p = float(self.monitor.train_step) / (100 * 40206*128 / self.batch_size)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            ReverseLayerF.apply(outputs.embedding, alpha)
            loss += torch.nn.MSELoss()(
                self.model.adversarial(outputs.embedding))

        if self.enable_similarity:
            loss2 = self.model.disc(outputs.embedding, y)
            beta = 1./101.
            beta = 0.9
            loss = (1 - beta) * loss + beta * loss2
        
        return loss

    def add_monotir_summary(self, outputs: CMOutput, x, y, loss):
        ### Monitor the Training process
        if  self.monitor.monitor_dict is not None or self.monitor.is_summary:
            _x = x.data.cpu().numpy()
            _y = self.data_meta_info.de_standardize_output(y.data.cpu().numpy())
            _preds = self.data_meta_info.de_standardize_output(outputs.preds.data.cpu().numpy())
            _mape = np.abs(_preds - _y) / _y

            if self.monitor.monitor_dict is not None:
                _monitor_dict = {
                    "X": _x,
                    "Y": _y,
                    "embedded_x": outputs.embedding,
                    "Y_predicted": _preds,
                    "MAPE": _mape,
                    "Scalar/loss": loss,
                }

                # if self.use_clip is not None and self.use_clip == "soft":
                #     _monitor_dict["residual"] = residual
                
                for name, parameter in self.model.named_parameters():
                    if not parameter.requires_grad: continue
                    if name == "InputLayer.layers.0.weight":
                        _monitor_dict["input_layer_weight"] = parameter
                self.monitor.add_monitor(_monitor_dict)
        
            if self.monitor.is_summary:
                summary_list = [
                    ("image", "Feature/Embedded", outputs.embedding),
                    ("image", "Feature/X", _x),
                    # ("image", "Label/True", _y),
                    # ("image", "Label/Predicted", _preds),
                    # ("histogram", "Label/Predicted", _preds),
                    # ("histogram", "Label/True", _y),
                    ("histogram", "MAPE", _mape),
                ]
                self.monitor.summary(summary_list)

    def save(self, path=None):
        _path = path if path is not None else self.cache_path
        if _path is None:
            return
        super(MyMatchNetLearner, self).save(_path)
        torch.save(self.model, os.path.join(_path, "mymatch_net.torch"))

    def load(self, path=None):
        _path = path if path is not None else self.cache_path
        super(MyMatchNetLearner, self).load(_path)

        self.model = torch.load(os.path.join(_path, "mymatch_net.torch"))

        ### Backward compatibility
        cur_model = MyMatchNet(input_len=self.model.input_len)
        if self.model.__dict__.keys() != cur_model.__dict__.keys():
            ### self.model is out of date
            try:
                cur_model.convert_old_model(self.model)
            except:
                import code
                code.interact(local=locals())
                raise
            self.model = cur_model
            warning("Load OLD version of models")

        self.init_device()

    @staticmethod
    def load_init(_path):
        learner = BaseLearner.load_init(_path, MyMatchNetLearner)
        learner.model = torch.load(os.path.join(_path, "mymatch_net.torch"))
        learner.init_device()
        return learner

    def register_hooks_for_grads(self):
        self.model.register_hooks_for_grads_weights(self.monitor)
        
    def enable_cmd_regularizer(self, x):
        def repr_func(_input):
            return self.model.embedding_block(_input)
        self.cmd_regularizer = CMDRegular(x, repr_func)
    
    def assign_config(self, config, verbose=True):
        super(MyMatchNetLearner, self).assign_config(config, verbose=verbose)
        self.model.assign_config(config)
        for attr in ["enable_similarity", "use_clip", "enable_up_sampling"]:
            if attr in config:
                self.__dict__[attr] = config[attr]
        self.init_device()

        if verbose:
            self.summary()

    def summary(self):
        super().summary()
        print(self.model.model_dumps())
        print(f"Learner: enable_similarity_loss: {self.enable_similarity}, Clip: {self.use_clip}, Sampling: {self.enable_up_sampling}")
        if self.normed_zero_output is not None:
            print(f"The normed zero {self.normed_zero_output.data.cpu().numpy()}")
        print("\n")