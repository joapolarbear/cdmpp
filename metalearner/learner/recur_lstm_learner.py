
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Union

import torch
from torch.optim import AdamW

from metalearner.data.rawdata import load_raw_data
from metalearner.data.ast_dataloader import load_ast_dataset
from metalearner.model.recur_net import Model_Recursive_LSTM_v2
from metalearner.learner.base_learner import BaseLearner
from utils.util import warn_once, notify, read_yes, TrainTestPair
from metalearner.learn_utils import CMOutput

from os import environ
environ['train_device'] = 'cuda:0' # training device: 'cpu' or 'cuda:X'
environ['store_device'] = 'cuda:0' # Data storing device:  'cpu' or 'cuda:X'
train_device= torch.device(environ.get('train_device', "cuda"))
store_device= torch.device(environ.get('store_device', "cuda"))

def mape_criterion(outputs, labels):
    eps = 1e-5
    return 100 * torch.mean(torch.abs((labels - outputs)/(labels + eps)))

def mse_criterion(outputs, labels):
    return torch.nn.MSELoss()(labels, outputs)

class RecurLSTMLearner(BaseLearner):
    def __init__(self, disable_norm, loss_func_name,
            residual=False, input_size=164, **kwargs):
        super(RecurLSTMLearner, self).__init__(**kwargs)

        if loss_func_name.lower() == "mape":
            self.loss_func = mape_criterion
        elif loss_func_name.lower() == "mse":
            self.loss_func = mse_criterion
        else:
            raise ValueError()
        self.loss_func_name = loss_func_name

        self.model = Model_Recursive_LSTM_v2(input_size, disable_norm, residual=residual, drops=[0.112, 0.112, 0.112, 0.112])
        print(f"Model parameter count: {self.model.para_cnt()}")
        self.init_device()

        self.lr_scheduler = 'one_cycle'

    def init_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), weight_decay=0.375e-2)
    
    def _inference(self, inputs) -> CMOutput:
        outputs = self.model(inputs)
        # inputs = (inputs[0], inputs[1].to(original_device))
        return CMOutput(outputs)
    
    def _loss(self, outputs: CMOutput, labels):
        loss = self.loss_func(outputs.preds, labels)
        return loss

    def add_monotir_summary(self, outputs: CMOutput, x, y, loss):
        if self.monitor.is_summary:
            self.monitor.summary([
                ("scalar", "Loss/Train", loss),
            ])
        
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
            self.model.to(self.train_device)
            if self.normed_zero_output is not None:
                self.normed_zero_output = self.normed_zero_output.to(self.train_device)

    def data_to_train_device(self, inputs, labels):
        if self.train_device and labels.device != self.train_device:
            return (inputs[0], inputs[1].to(train_device)), labels.to(train_device)
        else:
            return inputs, labels

    def prepare_test_pair(self, val_data, verbose=True):
        val_xs, val_ys, val_dis = zip(*val_data)
        return val_xs, val_ys, val_dis
        
    def train_one_batch(self, inputs, labels):
        self.model.train()
        # original_device = labels.device
        inputs, labels = self.data_to_train_device(inputs, labels)
        self.cached_data["train"] = (inputs, labels)
        # zero the parameter gradients
        self.optimizer.zero_grad()
        ### Forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = self._inference(inputs)  

            assert outputs.preds.shape == labels.shape
            loss = self._loss(outputs, labels)
            self.add_monotir_summary(outputs, inputs, labels, loss)
            if self.loss_func_name != "mape":
                mape = mape_criterion(outputs.preds, labels)
            else:
                mape = loss
            ### Backward + optimize only if in training phase
            loss.backward()
            self.optimizer.step()
        
        for hook in self.batch_hooks:
            hook()
        # print(self.monitor.epoch_cnt, self.monitor.train_step, loss)
        ### Summary, check and cache
        self.monitor.step()
        if self.monitor.is_cache:
            if self.monitor.epoch_cnt > 2:
                self.check_exit()
        # labels = labels.to(original_device)

        # ### debug
        # if self.monitor.train_step > 3:
        #     print(self.monitor.train_step)
        #     self.stop_training()
        
        return loss

    def train_model(self, dataloader_dict, num_epochs=100, log_every=5):
        self.stop_checker.max_epoch = num_epochs
        self.model.register_hooks_for_grads_weights(self.monitor)
        self.train(TrainTestPair(dataloader_dict["train"], dataloader_dict["val"]))

    def loss(self, outputs: CMOutput, dataset_or_x, y):
        return self._loss(outputs, y)

    def predict(self, *args, **kwargs):
        self.model.eval()
        return super().predict(*args, **kwargs)
    
    def forward_compute_metrics(self, input_data, element_wise_test=False):
        self.model.eval()
        all_preds = []
        all_labels = []
        if isinstance(input_data, torch.utils.data.DataLoader):
            for batch_x, batch_y, _ in input_data:
                for idx in range(len(batch_x)):
                    _x, _y = batch_x[idx], batch_y[idx]
                    _x, _y = self.data_to_train_device(_x, _y)
                    outputs = self._inference(_x)
                    all_preds.append(outputs)
                    all_labels.append(_y)
        elif isinstance(input_data, tuple):
            X, Y = input_data
            # TODO: Naive approach to adapt to the tiramisu
            # case for X Y didn't have the same shape
            if len(X) != len(Y):
                _x, _y = X, Y
                _x, _y = self.data_to_train_device(_x, _y)
                outputs = self._inference(_x)
                all_preds.append(outputs)
                all_labels.append(_y)
            else:
                for idx in range(len(Y)):
                    _x, _y = X[idx], Y[idx]
                    _x, _y = self.data_to_train_device(_x, _y)
                    outputs = self._inference(_x)
                    all_preds.append(outputs)
                    all_labels.append(_y)
        # tiramisu, x,y value paris
        elif isinstance(input_data, list):
            for idx, (_x, _y) in enumerate(input_data):
                _x, _y = self.data_to_train_device(_x, _y)
                outputs = self._inference(_x)
                all_preds.append(outputs)
                all_labels.append(_y)
        else:
            raise ValueError()
        outputs = CMOutput.concat(all_preds, 0)
        labels = torch.cat(all_labels, 0)
        metrics = self.compute_metrics(outputs, labels, element_wise_test=element_wise_test)
        metrics["loss"] = float(self.loss(outputs, None, labels).data.cpu().numpy())
        return outputs, metrics, labels
    
    def get_results_df(self, batches_list, log=False):   
        df = pd.DataFrame()
        self.model.eval()
        torch.set_grad_enabled(False)
        all_outputs=[]
        all_labels=[]

        for k, (inputs, labels, di) in tqdm(list(enumerate(batches_list))):
            original_device = labels.device
            inputs, labels = self.data_to_train_device(inputs, labels)
            outputs = self.model(inputs)
            assert outputs.shape == labels.shape
            all_outputs.append(outputs)
            all_labels.append(labels)
            inputs = (inputs[0], inputs[1].to(original_device))
            labels = labels.to(original_device)
        preds = torch.cat(all_outputs).cpu().detach().numpy().reshape((-1,))
        labels = torch.cat(all_labels).cpu().detach().numpy().reshape((-1,))
        preds = np.around(np.abs(preds), decimals=6)
        labels = np.around(labels, decimals=6)

        ### Denormalization
        preds = self.data_meta_info.de_standardize_output(preds)
        labels = self.data_meta_info.de_standardize_output(labels)
                                                
        assert preds.shape == labels.shape 
        df['prediction'] = np.array(preds)
        df['labels'] = np.array(labels)
        df['abs_diff'] = np.abs(preds - labels)
        df['RMSE'] = np.sqrt(np.abs(np.power(preds-labels, 2)))
        df['MAPE'] = np.abs(df.labels - df.prediction) / df.labels * 100
        
        describe = df.describe()
        # print(df)
        print(describe)

        return df

    def save(self, path=None):
        _path = path if path is not None else self.cache_path
        if _path is None:
            return
        super(RecurLSTMLearner, self).save(_path)
        torch.save(self.model, os.path.join(_path, "Model_Recursive_LSTM_v2.torch"))

    def load(self, path=None):
        _path = path if path is not None else self.cache_path
        super(RecurLSTMLearner, self).load(_path)
        self.model = torch.load(os.path.join(_path, "Model_Recursive_LSTM_v2.torch"))
        self.init_device()
    
    def _init_loss_func(self):
        warn_once(f"{self.__class__.__name__} init loss function when the object is created")
        

def test_tiramisu(files_or_dir, learning_params):
    if isinstance(files_or_dir, str):
        root_path, _, files = list(os.walk(files_or_dir))[0]
        files = [os.path.join(root_path, f) for f in files if f.endswith(".npy")]
    elif isinstance(files_or_dir, list):
        files = files_or_dir
    elif isinstance(files_or_dir, dict): # fix: tiramisu
        files = files_or_dir['tasks'][1]
    else:
        raise
    raw_data = load_raw_data(files, learning_params, verbose=True, force=True)
    print(f"Raw data size (before pre-processing): {raw_data.size}")
    raw_data.preprocess(time_lb=learning_params["ave_lb"], verbose=True)
    print(f"Raw data size (after pre-processing): {raw_data.size}")
    if learning_params["disable_norm"]:
        warn_once("Disable normalization")
        raw_data.metainfo.to_norm_input = False
        raw_data.metainfo.to_norm_output = False
    else:
        raw_data.metainfo.tsfm_hub.parse_y_norm_method("std", None, {
            "avg": raw_data.metainfo.output_avg, "std": raw_data.metainfo.output_std})
        raw_data.metainfo.tsfm_hub.parse_x_norm_method("min-max", None, {
            "min": raw_data.metainfo.input_min, "max": raw_data.metainfo.input_max})
        raw_data.metainfo.tsfm_hub.print()

    dataset, val_batches_list, val_batches_indices, train_batches_list, train_batches_indices = load_ast_dataset(raw_data, train_device, store_device)

    ### debug
    # avg, std, flops, ast_features, node_ids, serialized_tree = xydata[0]
    # ast = AST.deserialize_tree(serialized_tree)
    # print(node_ids)
    # print(serialized_tree)
    # print(ast)
    
    bl_dict = {'train': train_batches_list, 'val': val_batches_list}
    # here, please careful that LSTM learning rate could be different from the previous. Change input config if necessary.
    learner = RecurLSTMLearner(
        learning_params["disable_norm"], learning_params["loss_func"],
        residual=learning_params["residual"],
        data_meta_info = raw_data.metainfo,
        cache_path=learning_params["cache_dir"],
        debug=learning_params["debug"],
        tb_log_dir=learning_params["tb_logdir"],
        )

    if learning_params["load_cache"]:
        if not read_yes(f"Load cost model at {learner.cache_path}"):
                exit(0)
        learner.load()

    learner.train_model(dataloader_dict=bl_dict, num_epochs=10000, log_every=1)
    # Basic results on the test and validation set
    val_df = learner.get_results_df(val_batches_list)
