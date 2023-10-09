import numpy as np
from typing import Union
import os
import re

import torch
from torch.nn.utils.rnn import pad_sequence

from metalearner.learner.base_learner import BaseLearner, ProgressChecker
from metalearner.learn_utils import CMOutput
from metalearner.data.dataloader import (MyDataSet, LargeIterableDataset,
    LargeIterDataloader, WrapDataloader)
from metalearner.model.attention_net import AttentionNet, AttentionNetV2

from utils.util import read_yes, warn_once, warning
import utils.env as cdpp_env
from utils.metrics import metric_cmd, metric_mmd


def test_pad_sequence():
    import torch
    from torch.nn.utils.rnn import pad_sequence
    a = torch.ones(3, 10)
    b = torch.ones(2, 10) * 2
    c = torch.ones(4, 10) * 3
    pad_sequence([a, b, c]).size()
    
def collate_fn_pad_impt(list_pairs_seq_target):
    ### Perform padding, refer to https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/16
    seqs = [seq for seq, _, _ in list_pairs_seq_target]
    targets = [target for _, target, _ in list_pairs_seq_target]
    dis = [di for _, _, di in list_pairs_seq_target]
    seqs_padded_batched = pad_sequence(seqs)   # will pad at beginning of sequences
    targets_batched = torch.stack(targets)
    dis_batched = torch.stack(dis)
    ### by default, targets_batched is not batch first, the shape is (L_seq, B, d_entry)
    assert seqs_padded_batched.shape[1] == len(targets_batched)
    return seqs_padded_batched, targets_batched, dis_batched

def _mse_loss(y_pred, y, weight=None):
    if weight is None:
        return torch.nn.MSELoss()(y_pred, y)
    else:
        return torch.mean(weight * torch.pow(y_pred - y, 2))

def _mspe_loss(y_pred, y, weight=None):
    if weight is None:
        return torch.mean(torch.pow((y_pred - y) / y, 2))
    else:
        return torch.mean(weight * torch.pow((y_pred - y) / y, 2))

def _mape_loss(y_pred, y, weight=None):
    if weight is None:
        return torch.mean(torch.abs((y_pred - y) / y))
    else:
        return torch.mean(weight * torch.abs((y_pred - y) / y))

def _mse_mape_loss(y_pred, y, weight=None):
    return 1000 * _mse_loss(y_pred, y, weight) + _mape_loss(y_pred, y, weight)

def _sme_loss(y_pred, y, weight=None):
    _mean = (torch.sum(y_pred) - torch.sum(y)) / len(y_pred)
    if weight is None:
        return torch.pow(_mean, 2)
    else:
        return torch.pow(weight *_mean, 2)

def _mean_power_error(k):
    def __loss(y_pred, y, weight=None):
        if weight is None:
            return torch.mean(torch.pow(y_pred - y, k))
        else:
            return torch.mean(weight * torch.pow(y_pred - y, k))
    return __loss

_loss_func_dict = {
    "mse": torch.nn.MSELoss(),
    "mspe": _mspe_loss,
    "mape": _mape_loss
}

class AttentionLearner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super(AttentionLearner, self).__init__(*args, **kwargs)

        if self.input_len:
            self.model = AttentionNetV2(self.input_len)
        else:
            self.model = None
        self.init_device()
        # self.model.apply(weights_init_uniform)

        ### Used for padding length decision
        self.fix_seq_len = cdpp_env.PROJECT_CFG.get("FIX_SEQ_LEN", None)
        if self.fix_seq_len:
            self.max_seq_len = max(self.fix_seq_len)
        else:
            self.max_seq_len = cdpp_env.PROJECT_CFG["MAX_SEQ_LEN"]
        self.collate_fn_pad = None if self.max_seq_len else collate_fn_pad_impt
        
        ### Init loss func
        self.loss_func = None
        self._init_loss_func()

        self.cost_sensitive_loss = bool(int(os.environ.get("CDPP_COST_SENSITIVE_LOSS", '0')))
        if self.cost_sensitive_loss:
            if not read_yes("CDPP_COST_SENSITIVE_LOSS is enabled?"):
                exit(0)
    
    def _init_loss_func(self):
        ### Select the loss function to use
        if self.loss_type.lower() in _loss_func_dict:
            self.loss_func = _loss_func_dict[self.loss_type.lower()]
        elif self.loss_type.lower() == "mse+mape":
            self.loss_func = _mse_mape_loss
        elif self.loss_type.lower() == "sme":
            self.loss_func = _sme_loss
        elif self.loss_type.lower().startswith("mpowere"):
            power = int(self.loss_type.split("-")[1])
            self.loss_func = _mean_power_error(power)
        elif "+" in self.loss_type:
            _hybrid_loss_weight = {}
            for _loss_func_name in self.loss_type.lower().split("+"):
                match = re.search(r"(?P<weight>\d*)(?P<type>(mse|mape|mspe))", _loss_func_name)
                assert match is not None, _loss_func_name
                rst = match.groupdict()
                _hybrid_loss_weight[rst["type"]] = float(rst["weight"]) if len(rst["weight"]) > 0 else 0
            assert len(_hybrid_loss_weight) > 0, (_hybrid_loss_weight, self.loss_type)

            def _hybrid_loss(y_pred, y, sample_weight=None):
                ret = None
                for _loss_func_name in _loss_func_dict.keys():
                    _weight = _hybrid_loss_weight.get(_loss_func_name, 0)
                    if _weight <= 0:
                        continue
                    if ret is None:
                        ret = _weight * _loss_func_dict[_loss_func_name](y_pred, y, sample_weight)
                    else:
                        ret += _weight * _loss_func_dict[_loss_func_name](y_pred, y, sample_weight)
                return ret

            self.loss_func = _hybrid_loss
        else:
            raise ValueError(self.loss_type)
    
    def prepare_test_pair(self, dataset_or_x, y=None, verbose=True):
        if dataset_or_x is None:
            return None, None
        elif y is not None:
            x = dataset_or_x
        elif isinstance(dataset_or_x, MyDataSet):
            dataloader = torch.utils.data.DataLoader(
                dataset_or_x, batch_size=len(dataset_or_x), shuffle=True, collate_fn=self.collate_fn_pad)
            x, y = next(iter(dataloader))
        elif isinstance(dataset_or_x, LargeIterableDataset):
            val_len = len(list(dataset_or_x))
            dataloader = torch.utils.data.DataLoader(
                dataset_or_x, batch_size=val_len, shuffle=False, collate_fn=self.collate_fn_pad)
            x, y = next(iter(dataloader))
        else:
            raise ValueError(type(dataset_or_x))

        if verbose:
            self._log(f"Test Data shape, x={x.shape}, y={y.shape}")

        return self.data_to_train_device(x, y)

    def gen_dataloader(self, dataset, shuffle=True):
        if isinstance(dataset, MyDataSet):
            if len(dataset) < self.batch_size:
                warn_once(f"[BaseLearner] Data size ({len(dataset)}) is smaller than the batch size "
                        f"({self.batch_size}), use full batch ?")
                dataloader = WrapDataloader(dataset, batch_size=len(dataset),
                    shuffle=True, drop_last=False, collate_fn=self.collate_fn_pad)
            else:
                dataloader = WrapDataloader(dataset, batch_size=self.batch_size,
                    shuffle=True, drop_last=False, collate_fn=self.collate_fn_pad)
        elif isinstance(dataset, LargeIterableDataset):
            dataloader = LargeIterDataloader(dataset, self.batch_size, 
                enable_up_sampling=self.enable_up_sampling,
                shuffle=shuffle)
        else:
            raise ValueError(f"Invalid dataset type {type(dataset).__name__}")
        
        return dataloader

    def _sample_test_embedding(self, _iterator, sample_num=2048):
        sample_cnt = 0
        outputs = None
        for batch_idx, (_feature, _y) in enumerate(_iterator):
            _feature, _y = self.data_to_train_device(_feature, _y)
            _outputs = self._inference(_feature)
            if outputs is None:
                outputs = _outputs
            else:
                outputs.embedding = torch.cat((outputs.embedding, _outputs.embedding), axis=0)
            sample_cnt += len(_y)
            if sample_cnt >= sample_num:
                break
        return outputs.embedding
                
    def _loss(self, outputs: CMOutput, feature, y):
        ### Only evaluate loss value, do not monitor
        
        ### TODO (huhanpeng) remove the assertation
        assert outputs.preds.shape == y.shape, (outputs.preds.shape, y.shape)

        if self.cost_sensitive_loss:
            weight = torch.sigmoid(y)
            loss = self.loss_func(outputs.preds, y, weight)
        else:
            loss = self.loss_func(outputs.preds, y)

        UPDATE_STEP_INTERVAL = 1
        USE_STALE_EMBED = True
        USE_MMD = True
        ALPHA = 1
        SAMPLE_NUM = 2048

        if USE_MMD:
            _metric_func = metric_mmd
        else:
            _metric_func = metric_cmd

        if self.use_cmd_regular or self.domain_diff_metric:
            if self.domain_diff_metric:
                if self.domain_diff_metric.endswith("cmd"):
                    _metric_func = metric_cmd
                    _alpha = self.domain_diff_metric.split("cmd")[0]
                elif self.domain_diff_metric.endswith("mmd"):
                    _metric_func = metric_mmd
                    _alpha = self.domain_diff_metric.split("mmd")[0]
                else:
                    raise ValueError(self.domain_diff_metric)
                if len(_alpha) > 0:
                    ALPHA = eval(_alpha.replace("_", "."))

            ### Update Z_tir_val
            if "z_tir_val" not in self.cached_data or self.cached_data["z_tir_val"]["cnt"] >= UPDATE_STEP_INTERVAL:
                test_dataloader = self.gen_dataloader(self.cached_data["val"], shuffle=True)
                test_embedding = self._sample_test_embedding(test_dataloader, sample_num=SAMPLE_NUM)
                self.cached_data["z_tir_val"] = {"data": test_embedding, "cnt": 1}
                __update_regular = True
            else:
                # assert self.cached_data["z_tir_val"]["cnt"] < UPDATE_STEP_INTERVAL:
                test_embedding = self.cached_data["z_tir_val"]["data"]
                self.cached_data["z_tir_val"]["cnt"] += 1
                __update_regular = False
                
            if USE_STALE_EMBED or __update_regular:
                # train_embedding = self._sample_test_embedding(self.cached_data["train"])
                train_embedding = outputs.embedding
                # test_embedding = test_embedding.to(train_embedding.device)
                # regular = _metric_func(train_embedding, test_embedding).to(loss.device)
                regular = _metric_func(train_embedding, test_embedding)
                loss += ALPHA * regular

        return loss

    def add_monotir_summary(self, outputs: CMOutput, feature, y, loss):
        ### Monitor the Training process
        if  self.monitor.monitor_dict is not None or self.monitor.is_summary:
            _x = feature.x_tir.data.cpu().numpy()
            _y = self.data_meta_info.de_standardize_output(y.data.cpu().numpy())
            _preds = self.data_meta_info.de_standardize_output(outputs.preds.data.cpu().numpy())
            _mape = np.abs(_preds - _y) / _y

            if self.monitor.monitor_dict is not None:
                _monitor_dict = {
                    "X": _x,
                    "Y": _y,
                    "Y_predicted": _preds,
                    "MAPE": _mape,
                    "Scalar/loss": loss,
                }

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
    
    def _save_paths(self, _path, N_leaf):
        ### Sharable modules
        tsfm_encoder_path = os.path.join(_path, "tsfm_encoder.torch")
        device_encoder_path = os.path.join(_path, "device_encoder.torch")
        decoder_path = os.path.join(_path, "decoder.torch")
        ### Not sharable mudule
        flatten_layer_path = os.path.join(_path, f"flatten_layer_{N_leaf}.torch")
        return tsfm_encoder_path, device_encoder_path, decoder_path, flatten_layer_path

    def save(self, path=None):
        _path = path if path is not None else self.cache_path
        if _path is None:
            return
        super(AttentionLearner, self).save(_path)
        assert isinstance(self.model, AttentionNetV2)
        tsfm_encoder_path, device_encoder_path, decoder_path, \
            flatten_layer_path = self._save_paths(_path, self.model.max_seq_len)

        torch.save(self.model.tsfm_encoder, tsfm_encoder_path)
        torch.save(self.model.flatten_layer, flatten_layer_path)
        torch.save(self.model.device_encoder, device_encoder_path)
        torch.save(self.model.decoder, decoder_path)
    
    @staticmethod
    def _load(learner, _path):
        _model = AttentionNetV2(learner.input_len)

        tsfm_encoder_path, device_encoder_path, decoder_path, \
            flatten_layer_path = learner._save_paths(_path, _model.max_seq_len)
        
        old_version_model: Union[None, AttentionNet] = None
        ## Load the shared modules first
        if os.path.exists(tsfm_encoder_path) and os.path.exists(device_encoder_path) \
                and os.path.exists(decoder_path):
            _model.tsfm_encoder = torch.load(tsfm_encoder_path)
            _model.device_encoder = torch.load(device_encoder_path)
            _model.decoder = torch.load(decoder_path)
        else:
            ## Compatible to the old version of cost models
            old_model_path = os.path.join(_path, "attention_net.torch")
            if not os.path.exists(old_model_path):
                raise ValueError(f"Fail to find a cost model under {_path}")
            warning("Load cost model from old version, which will be depracated")
            old_version_model = torch.load(old_model_path)
            ### Copy sharable parts
            _model.tsfm_encoder.encoder = old_version_model.encoder
            _model.device_encoder.device_encoder_linear = old_version_model.mlp.DeviceEmbed
            _model.device_encoder.BN = old_version_model.mlp.BN
            _model.decoder.regression = old_version_model.mlp.Regression
            _model.decoder.output_layer = old_version_model.mlp.OutputLayer

        ## Load the leaf_node_specified regressor if exist
        if os.path.exists(flatten_layer_path):
            _model.flatten_layer = torch.load(flatten_layer_path)
        elif old_version_model is not None and old_version_model.max_seq_len == _model.max_seq_len:
            _model.flatten_layer.input_layer = old_version_model.mlp.InputLayer
            _model.flatten_layer.embed_layer = old_version_model.mlp.Embedding
        else:
            warning(f"Fail to find the regreesor for N_leaf={_model.max_seq_len}")

        learner.model = _model
        learner.init_device()

    def load(self, path=None):
        _path = path if path is not None else self.cache_path
        super(AttentionLearner, self).load(_path)
        AttentionLearner._load(self, _path)

    @staticmethod
    def load_init(_path):
        learner = BaseLearner.load_init(_path, AttentionLearner)
        AttentionLearner._load(learner, _path)
        return learner

    def assign_config(self, config, verbose=True):
        super(AttentionLearner, self).assign_config(config, verbose=verbose)
        if self.model is not None:
            self.model.assign_config(config)
            self.init_device()

        if verbose:
            print(f"\n[{self.__class__.__name__}] Assign new config ... ")
            self.summary()
    
    def summary(self):
        super().summary()
        print(self.model.model_dumps())
        print("\n")