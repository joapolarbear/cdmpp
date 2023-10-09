from typing import Union, List
import torch
import torch.nn as nn
from torch.autograd import Function

from metalearner.data.dataloader import get_di_len, MyDataSet
from metalearner.learn_utils import CMOutput

from utils.metrics import metric_cmd
import utils.env as cdpp_env
from utils.device_info import DEVICE_FEATURE_LEN
from utils.util import read_yes, notify

from .base_net import MLP, TrainInfinityError

class CMDRegular(nn.Module):
    def __init__(self, val_data: MyDataSet, repr_func, weight=1.):
        super(CMDRegular, self).__init__()
        self.val_data = val_data
        self.repr_func = repr_func
        self.weight = weight

    def forward(self, train_repr: torch.tensor):
        self.val_repr = self.repr_func(self.val_data)
        cmd = metric_cmd(train_repr, self.val_repr)
        return cmd * self.weight

class EuclidianDistance(nn.Module):
    def __init__(self):
        super(EuclidianDistance, self).__init__()

    def forward(self, support_set, query_set):
        '''
        support_set: shape = (B1, N)
        query_set: shape = (B2, N)
        Output shape = (B1, B2)
        '''
        return torch.nn.functional.softmax(
            torch.cdist(support_set, query_set), dim=1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.Distance = EuclidianDistance()

        self.cos = nn.CosineSimilarity(dim=1)
    
    def forward(self, embedded_support_x, support_y):
        featrue_distance = self.Distance(embedded_support_x, embedded_support_x)
        label_distance = self.Distance(support_y, support_y)

        ### Try 1
        # similarity_loss = featrue_distance / torch.sigmoid(label_distance)
        # similarity_loss = 1 / self.cos(featrue_distance, label_distance)
        # return torch.norm(similarity_loss)

        ### Try 2
        # featrue_distance = self.cos(embedded_support_x, embedded_support_x)
        # label_distance = self.cos(support_y, support_y)
        # similarity_loss = torch.nn.functional.mse_loss(featrue_distance, label_distance)
        # return similarity_loss

        ### Try 3
        cos_distance = self.cos(featrue_distance, label_distance)

        # print(embedded_support_x.shape)
        # print(support_y.shape)
        # print(featrue_distance.shape)
        # print(label_distance.shape)
        # raise

        return 1e4 * torch.norm(1 - cos_distance)

class ReverseLayerF(Function):
    @staticmethod
    def forward(self, x, alpha):
        self.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        output = grad_output.neg() * self.alpha
        return output, None

class MyMatchNet(nn.Module):
    def __init__(self, input_len):
        super(MyMatchNet, self).__init__()

        self.input_len = input_len
        self.disc = Discriminator()
        
        _config = cdpp_env.PROJECT_CFG["cost_model"]
        self.input_layer_unit: Union[int, None] = None
        self.embedded_layers: Union[None, List[int]] = None
        self.embed_feature_len: Union[int, None]  = None
        self.regression_layers: Union[None, List[int]] = None
        
        self.use_di = False
        self.dropout_rate: Union[None, float] = None
        self.use_residual = False
        self.is_cross_device = "add"

        self.assign_config(_config)
        self.init_model()

        if self.is_cross_device is not None:
            notify(f"Use device embeddings: {self.is_cross_device}?")
        #     if not read_yes(f"Use device embeddings: {self.is_cross_device}?"):
        #         exit(0)

    def init_model(self):
        self.InputLayer = MLP([self.input_len, self.input_layer_unit], activation=None)

        embedding_shapes = [self.input_layer_unit] + self.embedded_layers + [self.embed_feature_len]
        self.Embedding = MLP(embedding_shapes)

        if self.is_cross_device is not None:
            self.DeviceEmbed = MLP([DEVICE_FEATURE_LEN] + [128] * 3 + [self.embed_feature_len])
            if self.is_cross_device == "cat":
                regression_shapes = [2 * self.embed_feature_len] + self.regression_layers
            else:
                regression_shapes = [1 * self.embed_feature_len] + self.regression_layers
        else:
            self.DeviceEmbed = None
            regression_shapes = [self.embed_feature_len] + self.regression_layers

        self.Regression = MLP(regression_shapes)
        self.BN = torch.nn.BatchNorm1d(self.embed_feature_len, affine=False)
        self.OutputLayer = MLP([regression_shapes[-1], 1], activation=None, bn=False)

        if self.use_di:
            self.DiLearner = MLP([get_di_len()] + [128] * 3 + [self.embed_feature_len])
        else:
            self.DiLearner = None
        
        # print(len(self.Regression.layers))
        # print(self.Regression.layers[0].weight)
        # print(self.Regression.layers[0].bias)
        self.adversarial = MLP([self.embed_feature_len, 1])
        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = None
        
        if self.use_residual:
            self.residual_layer = MLP([self.input_len, self.embed_feature_len])
        else:
            self.residual_layer = None

    def model_dumps(self):
        _str = f"**************** {self.__class__.__name__} Summary ****************"
        _str += f"\n InputLayer: {self.input_len} -> {self.input_layer_unit}"
        _str += f"\n Embedded Layer: == {self.embedded_layers} ==> {self.embed_feature_len}"
        if self.dropout:
            _str += f"\n Dropout(p={self.dropout_rate})"
        if self.is_cross_device is not None:
            _str += f"\n DeviceEmbed Layer: == {DEVICE_FEATURE_LEN} ==> ... ==> {self.embed_feature_len} + {self.is_cross_device}"
        _str += f"\n Regressor: == {self.regression_layers} ==> 1"
        _str += "\n**********************************************"
        return _str

    def embedding_block(self, x_tir):
        ### Embedding
        x = self.InputLayer(x_tir)
        embedded_x = self.Embedding(x)

        if self.use_residual:
            embedded_x = embedded_x + self.residual_layer(x_tir)
        
        return embedded_x

    def forward(self, feature):
        embedded_x = self.embedding_block(feature.x_tir)

        if self.use_di:
            raise NotImplementedError("Should integrate di into x")
            embedded_di = self.DiLearner(feature.x_tir)
            embedded_x = torch.mul(embedded_x, embedded_di)
        
        if self.is_cross_device is not None and feature.x_device is not None:
            z_device = self.DeviceEmbed(feature.x_device)

            embedded_x = self.BN(embedded_x)
            z_device = self.BN(z_device)

            ### Aggregate z_tir and z_device
            if self.is_cross_device == "mul":
                embedded_x = torch.mul(embedded_x, z_device)
            elif self.is_cross_device == "cat":
                embedded_x = torch.cat((embedded_x, z_device), 1)
            elif self.is_cross_device == "add":
                embedded_x = embedded_x + z_device
            else:
                raise ValueError(self.is_cross_device)

            # embedded_x = self.BN(embedded_x)

        if self.dropout:
            _output = self.Regression(self.dropout(embedded_x))
        else:
            _output = self.Regression(embedded_x)

        preds = self.OutputLayer(_output)
        if torch.any(torch.isnan(preds)) or torch.any(torch.isinf(preds)):
            raise TrainInfinityError("Found Nan in preds") # used for autotune
            # print("Found Nan/Inf in preds")
            # import pdb; pdb.set_trace()
        return CMOutput(preds=preds, embedding=embedded_x)
    
    def register_hooks_for_grads_weights(self, monitor):
        monitor.register_bw_hook(self.Embedding, "Embedding")
        if self.DiLearner is not None:
            monitor.register_bw_hook(self.DiLearner, "DiLearner")
        monitor.register_bw_hook(self.Regression, "Regression")

    def make_fw_hook_fn(self, name, _rst):
        def hook_fn(module, _input, _output):
            try:
                if not isinstance(_output, tuple):
                    _output = (_output,)
                for idx in range(len(_output)):
                    if _output[idx] is None:
                        continue
                    output_name = f"_Output/{name}/{idx}"
                    # print(output_name)
                    _rst.append((output_name, _output[idx].data.cpu().numpy()))
            except:
                pass
        return hook_fn
        
    def check_latent(self, ret):
        
        def _register_fw_hook(module, name):
            module.register_forward_hook(self.make_fw_hook_fn(name, ret))

        _register_fw_hook(self.Embedding, "Embedding")
        for idx, mlp_layer in enumerate(self.Regression.layers):
            _register_fw_hook(mlp_layer, f"Regression/{idx}")
    
    def assign_config(self, config):
        for attr in ["input_layer_unit", "dropout_rate", "use_di", "embed_feature_len", "use_residual"]:
            if attr in config:
                self.__dict__[attr] = config[attr]

        if "embedded_layers" in config:
            self.embedded_layers = config["embedded_layers"]
        elif "embed_layer_unit" in config and "embed_layer_num" in config:
            self.embedded_layers = [config["embed_layer_unit"]] * config["embed_layer_num"]
        
        if "regression_layers" in config:
            self.regression_layers = config["regression_layers"]
        elif "mlp_layer_unit" in config and "mlp_layer_num" in config:
            self.regression_layers = [config["mlp_layer_unit"]] * config["mlp_layer_num"]

        self.init_model()
    
    def convert_old_model(self, old_model):
        ''' Make the model backward compatible to old version'''
        for attr in ["input_layer_unit", "dropout_rate", "use_di", "embed_feature_len", "use_residual"]:
            if attr in old_model.__dict__:
                self.__dict__[attr] = old_model.__dict__[attr]
            elif attr == "use_residual":
                if "residual" in old_model.__dict__:
                    self.__dict__[attr] = old_model.residual
                else:
                    raise ValueError(f"Fail to find attr {attr} in old version of model")
            elif attr == "embed_feature_len":
                if "embedded_feature_len" in old_model.__dict__:
                    self.__dict__[attr] = old_model.embedded_feature_len
                else:
                    raise ValueError(f"Fail to find attr {attr} in old version of model")
            else:
                raise ValueError(f"Fail to find attr {attr} in old version of model")

        if "embedded_layers" in old_model.__dict__:
            self.embedded_layers = old_model.embedded_layers
        elif "embed_layer_unit" in old_model.__dict__ and "embed_layer_num" in old_model.__dict__:
            self.embedded_layers = [old_model.embed_layer_unit] * old_model.embed_layer_num
        
        if "regression_layers" in old_model.__dict__:
            self.regression_layers = old_model.regression_layers
        elif "mlp_layers" in old_model.__dict__:
            self.regression_layers = old_model.mlp_layers
        elif "mlp_layer_unit" in old_model.__dict__ and "mlp_layer_num" in old_model.__dict__:
            self.regression_layers = [old_model.mlp_layer_unit] * old_model.mlp_layer_num

        self.InputLayer = old_model.InputLayer
        self.Embedding = old_model.Embedding
        self.Regression = old_model.Regression
        self.OutputLayer = old_model.OutputLayer
        self.DiLearner = old_model.DiLearner
        self.adversarial = old_model.adversarial
        self.dropout = old_model.dropout
        self.residual_layer = old_model.residual_layer if "residual_layer" in old_model.__dict__ else None

    def regularizer(self, weight=None):
        l2_regularization = None
        for param in self.InputLayer.parameters():
            assert isinstance(weight, torch.Tensor)
            assert param.shape[-1] == self.input_len

            if weight is not None:
                assert weight.shape[0] == self.input_len
                layer_regular = torch.norm(param * weight, 2)**2
            else:
                layer_regular = torch.norm(param * weight, 2)**2

            if l2_regularization:
                l2_regularization += layer_regular
            else:
                l2_regularization = layer_regular
                
            break
        return l2_regularization

