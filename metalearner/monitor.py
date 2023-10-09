import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pickle
from typing import Union
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.util import fig_base

def monitor_visual(path, data_dict):
    print(type(data_dict))
    if data_dict is None:
        return 1

    scalar_keys = set()
    for idx, _key in enumerate(sorted(data_dict.keys())):
        data = data_dict[_key]
        if _key.startswith("Scalar/") or isinstance(data, float) or len(data.shape) == 0:
            scalar_keys.add(_key)
    non_scalar_keys = [_k for _k in data_dict.keys() if _k not in scalar_keys]
    
    fig = plt.figure(figsize=(16, 12))
    fig_num = len(non_scalar_keys) + int(len(scalar_keys) > 0)
    if fig_num == 0:
        return 1
    _fig_base = fig_base(fig_num)

    for idx, _key in enumerate(sorted(non_scalar_keys)):
        data = data_dict[_key]
        fig_idx = idx + 1
        ax = fig.add_subplot(_fig_base+fig_idx)
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()

        if len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[1] == 1):
            data = data.reshape(data.shape[0])
            ax = sns.lineplot(x=np.arange(data.shape[0]), y=data)
            plt.xlabel("# of Samples", fontsize=16)
            plt.ylabel(_key, fontsize=16)
        elif len(data.shape) == 2:
            ax = sns.heatmap(data, 
                # cmap="RdBu_r"
                # cmap="gray_r"
                cmap="YlGnBu_r"
                # cmap="OrRd"
                )
            plt.xlabel("# of Features", fontsize=16)
            plt.ylabel("# of Samples", fontsize=16)
        elif len(len(data.shape) == 0):
            ### Scalar
            raise
        else:
            raise

        plt.title(_key, fontsize=16)
        # ax = sns.heatmap(model_para_list[0], mask=mask, cmap="YlGnBu")

    ax = fig.add_subplot(_fig_base+fig_num)
    y_width = 10
    for idx, _key in enumerate(sorted(scalar_keys)):
        __key = _key.split("Scalar/")[1] if _key.startswith("Scalar/") else _key
        print(f"{__key}={data_dict[_key]}")
        # plt.text(x=0, y=idx*y_width,
        #     s=f"{__key}={data_dict[_key]}",
        #     fontsize=24,
        #     alpha=0.5,
        #     color='r')

    plt.tight_layout()
    save_dir = os.path.join(path, ".fig/metalearner")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "me_learner.png"))
    plt.close()
    x = input("Continue?")
    return int(x) if x.isdigit() else 1

def visual_error(preds, labels, path):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()
    fig = plt.figure(figsize=(8, 5))
    _fig_base = fig_base(1)
    ax = fig.add_subplot(_fig_base+1)
    ax.scatter(preds, labels, label="Predicted", alpha=0.5, edgecolors='none')
    ax.plot(labels, labels, c="y", label="Ideal")
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("Measured", fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    save_dir = os.path.join(path, ".fig/metalearner")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "cross_domain_error.png"))
    plt.close()

class ProgressChecker:
    def __init__(self, every, step_based=True):
        self.every = every
        self.step_based = step_based

    def check(self, step_cnt, batch_cnt, epoch):
        if self.step_based:
            return step_cnt % self.every == 0
        else:
            return batch_cnt == 1 and epoch % self.every == 0
    
    def __str__(self):
        return f"[ProgressChecker] Every {self.every} {'step' if self.step_based else 'epoch'}"

class Monitor:
    def __init__(self, tb_log_dir=None, debug=False):
        self.monitor_dict: Union[None, dict] = {} if debug else None
        if tb_log_dir is not None:
            self.summary_writer = SummaryWriter(log_dir=tb_log_dir)
        else:
            self.summary_writer = None
            tb_log_dir = ".workspace/runs/default"

        ### delete backward hook to visual net
        # self.summary_writer.add_graph(self.match_net, support_x)

        ### Used for hooks
        self.hooks = []

        self.train_step = 1
        self.batch_cnt = 1
        self.epoch_cnt = 1

        self.epoch_ts: Union[None, float] = None
        self.batch_ts: Union[None, float] = None
        self.epoch_durs = []
        self.batch_durs = []

        ### For monitor
        self.next_step = 1
        self.last_step = 1

        self.log_checker = ProgressChecker(1000)
        self.summary_checker = ProgressChecker(1, False)
        self.cache_checker = ProgressChecker(10, False)
    
    def summary_tensor(self, tensor, name):
        if not isinstance(tensor, np.ndarray):
            tensor = tensor.data.cpu().numpy()
        if len(tensor.shape) == 1:
            ### histogram
            self.summary_writer.add_histogram(
                name, tensor,
                global_step=self.train_step,
                walltime=None)
        else:
            dataformats = 'HW' if len(tensor.shape) == 2 else 'CHW'
            self.summary_writer.add_image(
                name, tensor,
                global_step=self.train_step,
                dataformats=dataformats)
                
    def make_bw_hook_fn(self, name):
        def hook_fn(module, ginp, gout):
            if self.summary_writer is None:
                return
            if self.is_summary:
                try:
                    for idx in range(len(ginp)):
                        if ginp[idx] is None or len(ginp[idx].shape) < 2:
                            continue
                        self.summary_tensor(ginp[idx], f"Grads/{name}_{idx}")
                    
                    ### debug
                    # for idx in range(len(gout)):
                    #     if gout[idx] is None or len(gout[idx].shape) < 2:
                    #         continue
                    #     self.summary_tensor(gout[idx], f"GradsOutput/{name}_{idx}")

                    for para_name, param in module.named_parameters():
                        if param.requires_grad:
                            if len(param.shape) < 2:
                                continue
                            self.summary_tensor(param, f"Weights/{name}/{para_name}")
                except:
                    pass
            if self.monitor_dict is not None:
                for idx in range(len(ginp)):
                    self.monitor_dict[f"Grads/{name}_{idx}"] = ginp[idx].data.cpu().numpy()

        return hook_fn
    
    def make_fw_hook_fn(self, name):
        def hook_fn(module, _input, _output):
            if self.summary_writer is None:
                return
            if self.is_summary:
                try:
                    if not isinstance(_input, tuple):
                        _input = (_input,)
                    for idx in range(len(_input)):
                        if _input[idx] is None:
                            continue
                        self.summary_tensor(_input[idx], f"_Input/{name}_{idx}")

                    if not isinstance(_output, tuple):
                        _output = (_output,)
                    for idx in range(len(_output)):
                        if _output[idx] is None:
                            continue
                        self.summary_tensor(_output[idx], f"_Output/{name}_{idx}")
                except:
                    pass
            if self.monitor_dict is not None:
                for idx in range(len(_output)):
                    self.monitor_dict[f"Activations/{name}_{idx}"] = _output[idx].data.cpu().numpy()

        return hook_fn

    def register_bw_hook(self, module, name, compatible=False):
        if not isinstance(module, torch.nn.Module):
            print(f"[Warning] Fail to register a bw hook for {name} with type{type(module)}")
        else:
            if compatible:
                hook_handler = module.register_backward_hook(
                    self.make_bw_hook_fn(name))
            else:
                hook_handler = module.register_full_backward_hook(
                    self.make_bw_hook_fn(name))
            self.hooks.append(hook_handler)
    
    def register_fw_hook(self, module, name):
        if not isinstance(module, torch.nn.Module):
            print(f"[Warning] Fail to register a fw hook for {name} with type{type(module)}")
        else:
            hook_handler = module.register_forward_hook(
                self.make_fw_hook_fn(name))
            self.hooks.append(hook_handler)
    
    def epoch_start(self):
        self.batch_cnt = 1
        self.summary([("text", "Epoch/Train", str(self.epoch_cnt))])
    
    def epoch_end(self):
        self.epoch_cnt += 1

    def step(self):
        self.train_step += 1
        self.batch_cnt += 1
        
        if self.monitor_dict is None:
            self.last_step = self.train_step
        elif self.train_step - self.last_step >= self.next_step:
            self.next_step = self.visual(".")
            self.last_step = self.train_step
    
    @property
    def is_summary(self):
        return self.summary_checker.check(self.train_step, self.batch_cnt, self.epoch_cnt)
    
    @property
    def is_log(self):
        # print("is_log", self.train_step, self.batch_cnt, self.epoch_cnt, self.log_checker.every, self.log_checker.step_based)
        return self.log_checker.check(self.train_step, self.batch_cnt, self.epoch_cnt)
        
    @property
    def is_cache(self):
        return self.cache_checker.check(self.train_step, self.batch_cnt, self.epoch_cnt)
    
    def summary(self, summary_list):
        if self.summary_writer is None:
            return
        for summary_type, name, tensor in summary_list:
            if tensor is None:
                continue
            if summary_type == "scalar":
                self.summary_writer.add_scalar(name, tensor, global_step=self.train_step, walltime=None)
            elif summary_type == "histogram":
                self.summary_writer.add_histogram(name, tensor, global_step=self.train_step, walltime=None)
            elif summary_type == "image":
                if len(tensor.shape) == 2:
                    self.summary_writer.add_image(name, tensor, global_step=self.train_step, dataformats='HW')
                elif len(tensor.shape) == 3:
                    self.summary_writer.add_image(name, tensor[:3], global_step=self.train_step, dataformats='CHW')
            elif summary_type == "graph":
                model, input_to_model = name, tensor
                self.summary_writer.add_graph(model, input_to_model)
            elif summary_type == "text":
                self.summary_writer.add_text(name, tensor, global_step=self.train_step)
            else:
                raise ValueError(f"Invalid summary type: {summary_type}")
        self.summary_writer.flush()
    
    def close(self):
        if self.summary_writer is not None:
            self.summary_writer.close()
    
    def add_monitor(self, _monitor_dict):
        if self.monitor_dict is None:
            return
        self.monitor_dict.update(_monitor_dict)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def save(self, _path):
        with open(os.path.join(_path, "monitor.pickle"), "wb") as f:
            pickle.dump([
                self.train_step,
                self.batch_cnt,
                self.epoch_cnt
                ], f)

    def load(self, _path):
        with open(os.path.join(_path, "monitor.pickle"), "rb") as f:
            self.train_step, self.batch_cnt, self.epoch_cnt = pickle.load(f)

    def visual(self, _path):
        monitor_visual(_path, self.monitor_dict)