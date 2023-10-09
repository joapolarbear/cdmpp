from typing import Union, List
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from metalearner.learn_utils import CMOutput, CMFeature
import utils.env as cdpp_env
from utils.device_info import DEVICE_FEATURE_LEN
from utils.util import warn_once
from .base_net import MLP, TrainInfinityError
from .cross_domain_net import MyMatchNet

class AttentionNet(nn.Module):
    def __init__(self, input_len):
        warn_once("AttentionNet will be deprecated")

        super(AttentionNet, self).__init__()

        ### Used for padding length decision
        self.fix_seq_len = cdpp_env.PROJECT_CFG.get("FIX_SEQ_LEN", None)
        if self.fix_seq_len:
            self.max_seq_len = max(self.fix_seq_len)
        else:
            self.max_seq_len = cdpp_env.PROJECT_CFG["MAX_SEQ_LEN"]

        self.input_len = input_len
        self.hidden_state_len = self.input_len if self.max_seq_len is None else (self.input_len * self.max_seq_len)

        self.nhead = 4 if input_len % 4 == 0 else 5
        self.tsfm_layer_num = cdpp_env.PROJECT_CFG["cost_model"].get("tsfm_layer_num", 6)

        self.mlp = MyMatchNet(self.hidden_state_len)
        self.init_model()
        
    def init_model(self):
        # 0: baseline:
        # TODO(huhanpeng): improve the code, currently, when applying a config, init is called for multiple times
        self.mlp.init_model()
        
        # 1: torch transformer encoder only
        encode_layer = nn.TransformerEncoderLayer(d_model=self.input_len, nhead=self.nhead, batch_first=True)
        if "tsfm_layer_num" in self.__dict__:
            self.encoder = nn.TransformerEncoder(encode_layer, num_layers=self.tsfm_layer_num)
        else:
            ### Adapt to old version
            warn_once("Define TransformerEncoder using the constant 6 will be deprecated")
            self.encoder = nn.TransformerEncoder(encode_layer, num_layers=6)

        # 2: torch transformer
        # self.transformer = nn.Transformer(d_model=self.input_len, nhead=self.nhead)

        # 3: hugging face transformers
        # BASE_MODEL = "camembert-base"
        # self.transformer = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

        # self.Regression = MLP([self.hidden_state_len, 1])

    def forward(self, feature):
        # import code
        # code.interact(local=locals())

        # 0: baseline
        # x = feature.x_tir.sum(dim=0)
        # output = self.mlp(x)
        # encoding = None
        # return output

        # 1: torch transformer encoder only
        encoding = self.encoder(feature.x_tir)
        if self.max_seq_len:
            ### Batch size first is True
            bs = encoding.shape[0]
            encoding = encoding.reshape(bs, -1)
        else:
            ### Batch size first is False
            encoding = encoding.sum(dim=0)
        # output = self.Regression(encoding)

        # 2: torch transformer
        # tgt = torch.ones((1, x.shape[1], self.input_len)).to(x.device)
        # encoding = self.transformer(x, tgt).view(x.shape[1], -1)
        # output = self.Regression(encoding)

        # print(encoding.shape)
        # 3: hugging face transformers
        # output = self.transformer(x)

        # 0+1: TransformerEncoder+MLP
        output = self.mlp(CMFeature(encoding, x_device=feature.x_device))
        return output

        return CMOutput(preds=output, embedding=encoding)
    
    def model_dumps(self):
        _str = f"**************** {self.__class__.__name__} Summary ****************\n"
        _str += f" Transformer: d_model={self.input_len}, nhead={self.nhead}, tsfm_layer={self.tsfm_layer_num}\n"
        # _str += f"\n Regressor: == {self.Regression.layers[0].in_features} ==> 1"
        _str += self.mlp.model_dumps()
        return _str
    
    def assign_config(self, config):
        self.mlp.assign_config(config)

        ### tune hyper-parameters
        if "tsfm_layer_num" in config:
            self.tsfm_layer_num = config["tsfm_layer_num"]
        self.init_model()


class AttentionNetEncoder(nn.Module):
    def __init__(self, input_len):
        super(AttentionNetEncoder, self).__init__()

        self.input_len = input_len
    
        self.nhead = 4 if input_len % 4 == 0 else 5
        self.tsfm_layer_num = cdpp_env.PROJECT_CFG["cost_model"].get("tsfm_layer_num", 6)

        self.init_model()
    
    def init_model(self):
        encode_layer = nn.TransformerEncoderLayer(d_model=self.input_len, nhead=self.nhead, batch_first=True)
        if "tsfm_layer_num" in self.__dict__:
            self.encoder = nn.TransformerEncoder(encode_layer, num_layers=self.tsfm_layer_num)
        else:
            ### Adapt to old version
            warn_once("Define TransformerEncoder using the constant 6 will be deprecated")
            self.encoder = nn.TransformerEncoder(encode_layer, num_layers=6)

    def forward(self, feature):
        # 1: torch transformer encoder only
        encoding = self.encoder(feature.x_tir)
        return encoding

    def model_dumps(self):
        _str = f"**************** {self.__class__.__name__} Summary ****************\n"
        _str += f" Transformer: d_model={self.input_len}, nhead={self.nhead}, tsfm_layer={self.tsfm_layer_num}"
        return _str
    
    def assign_config(self, config):
        ### tune hyper-parameters
        if "tsfm_layer_num" in config:
            self.tsfm_layer_num = config["tsfm_layer_num"]
        self.init_model()

class Flatten(nn.Module):
    ''' Not sharable across N-leaf'''
    def __init__(self, input_len):
        super(Flatten, self).__init__()

        self.input_len = input_len

        self.input_layer_unit: Union[int, None] = None
        self.embedded_layers: Union[None, List[int]] = None
        self.embed_feature_len: Union[int, None]  = None

        _config = cdpp_env.PROJECT_CFG["cost_model"]
        self.assign_config(_config)
        self.init_model()
    
    def init_model(self):
        self.input_layer = MLP([self.input_len, self.input_layer_unit], activation=None)
        embedding_shapes = [self.input_layer_unit] + self.embedded_layers + [self.embed_feature_len]
        self.embed_layer = MLP(embedding_shapes)

    def forward(self, encoding):
        embedded_x = self.input_layer(encoding)
        embedded_x = self.embed_layer(embedded_x)
        return embedded_x

    def model_dumps(self):
        _str = f"**************** {self.__class__.__name__} Summary ****************\n"
        _str += f" InputLayer: {self.input_len} -> {self.input_layer_unit}\n"
        _str += f" Embedded Layer: == {self.embedded_layers} ==> {self.embed_feature_len}"
        return _str

    def assign_config(self, config):
        if "embedded_layers" in config:
            self.embedded_layers = config["embedded_layers"]
        elif "embed_layer_unit" in config and "embed_layer_num" in config:
            self.embedded_layers = [config["embed_layer_unit"]] * config["embed_layer_num"]

        for attr in ["input_layer_unit", "embed_feature_len"]:
            if attr in config:
                self.__dict__[attr] = config[attr]

class DeviceEncoder(nn.Module):
    def __init__(self, is_cross_device="add"):
        super(DeviceEncoder, self).__init__()

        self.is_cross_device = is_cross_device

        ### Decide the shapes of the OUTPUT of this module
        self.embed_feature_len: Union[int, None]  = None

        _config = cdpp_env.PROJECT_CFG["cost_model"]
        self.assign_config(_config)
        self.init_model()
    
    def init_model(self):
        if self.is_cross_device is not None:
            self.device_encoder_linear = MLP([DEVICE_FEATURE_LEN] + [128] * 3 + [self.embed_feature_len])
        else:
            self.device_encoder_linear = None
        self.BN = torch.nn.BatchNorm1d(self.embed_feature_len, affine=False)

    def forward(self, embedded_x, x_device):
        if self.is_cross_device is not None and x_device is not None:
            z_device = self.device_encoder_linear(x_device)

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
        return embedded_x

    def model_dumps(self):
        _str = f"**************** {self.__class__.__name__} Summary ****************\n"
        if self.is_cross_device is not None:
            _str += f" DeviceEmbed Layer: == {DEVICE_FEATURE_LEN} ==> ... ==> {self.embed_feature_len} + {self.is_cross_device}"
        else:
            _str += " None"
        return _str

    def assign_config(self, config):
        if "embed_feature_len" in config:
            self.embed_feature_len = config["embed_feature_len"]


class Decoder(nn.Module):
    def __init__(self, is_cross_device="add"):
        super(Decoder, self).__init__()

        self.is_cross_device = is_cross_device

        ### Decide the shapes of the INPUT of this module
        self.embed_feature_len: Union[int, None]  = None

        self.regression_layers: Union[None, List[int]] = None
        self.dropout_rate: Union[None, float] = None

        _config = cdpp_env.PROJECT_CFG["cost_model"]
        self.assign_config(_config)
        self.init_model()

        self.embedding = None
    
    def init_model(self):
        if self.is_cross_device is not None:
            if self.is_cross_device == "cat":
                regression_shapes = [2 * self.embed_feature_len] + self.regression_layers
            else:
                regression_shapes = [1 * self.embed_feature_len] + self.regression_layers
        else:
            regression_shapes = [self.embed_feature_len] + self.regression_layers

        self.regression = MLP(regression_shapes)
        self.output_layer = MLP([regression_shapes[-1], 1], activation=None, bn=False)

        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = None

    def forward(self, embedded_x):
        if self.dropout:
            _output = self.regression(self.dropout(embedded_x))
        else:
            _output = self.regression(embedded_x)
        
        self.embedding = _output
        
        preds = self.output_layer(_output)
        # if torch.any(torch.isnan(preds)) or torch.any(torch.isinf(preds)):
        #     raise TrainInfinityError("Found Nan in preds")
        preds = torch.nan_to_num(preds)

        return preds

    def model_dumps(self):
        _str = f"**************** {self.__class__.__name__} Summary ****************\n"
        if self.dropout:
            _str += f" Dropout(p={self.dropout_rate})"
        _str += f" Regressor: == {self.regression_layers} ==> 1"
        return _str
    
    def assign_config(self, config):
        if "regression_layers" in config:
            self.regression_layers = config["regression_layers"]
        elif "mlp_layer_unit" in config and "mlp_layer_num" in config:
            self.regression_layers = [config["mlp_layer_unit"]] * config["mlp_layer_num"]

        if "dropout_rate" in config:
            self.dropout_rate = config["dropout_rate"]
        
        if "embed_feature_len" in config:
            self.embed_feature_len = config["embed_feature_len"]


class AttentionNetV2(nn.Module):
    def __init__(self, input_len):
        super(AttentionNetV2, self).__init__()

        self.input_len = input_len

        ### Used for padding length decision
        ## TODO (huhanpeng): since we will load all config from the cached model
        # the N_leaf required by the latest trial does not work, so we use 
        # CUR_FIX_SEQ_LEN to store the N_leaf required by the latest trial
        self.fix_seq_len = cdpp_env.PROJECT_CFG.get("CUR_FIX_SEQ_LEN", None)
        if self.fix_seq_len is None:
            self.fix_seq_len = cdpp_env.PROJECT_CFG.get("FIX_SEQ_LEN", None)
        if self.fix_seq_len:
            self.max_seq_len = max(self.fix_seq_len)
        else:
            self.max_seq_len = cdpp_env.PROJECT_CFG["MAX_SEQ_LEN"]
        self.hidden_state_len = self.input_len if self.max_seq_len is None else (self.input_len * self.max_seq_len)
        
        # Encoder:
        ## 1: Transformer Encoder
        self.tsfm_encoder = AttentionNetEncoder(input_len)
        ## 2: N_leaf-specific linear
        self.flatten_layer = Flatten(self.hidden_state_len)
        ## 3: device encoder
        self.is_cross_device = "add"
        self.device_encoder = DeviceEncoder(self.is_cross_device)
        
        # Decoder
        self.decoder = Decoder(self.is_cross_device)

        self.init_model()

    def init_model(self):
        # Encoder:
        ## 1: Transformer Encoder
        # TODO(huhanpeng): improve the code, currently, when applying a config, init is called for multiple times
        self.tsfm_encoder.init_model()
        ## 2: N_leaf-specific linear
        self.flatten_layer.init_model()
        ## 3: device encoder
        self.device_encoder.init_model()

        self.decoder.init_model()
        
    def forward(self, feature):
        ## 1: torch transformer encoder only
        encoding = self.tsfm_encoder(feature)
        if self.max_seq_len:
            ### Batch size first is True
            bs = encoding.shape[0]
            encoding = encoding.reshape(bs, -1)
        else:
            ### Batch size first is False
            encoding = encoding.sum(dim=0)
        
        ## 2: N_leaf-specific linear
        embedded_x = self.flatten_layer(encoding)

        ## 3: device-encoder
        embedded_x = self.device_encoder(embedded_x, feature.x_device)
        
        # Decoder
        preds = self.decoder(embedded_x)

        # return CMOutput(preds=preds, embedding=embedded_x)
        return CMOutput(preds=preds, embedding=self.decoder.embedding)
    
    def model_dumps(self):
        _str = f"**************** {self.__class__.__name__} Summary ****************\n"
        _str += f" - N_leaf={self.max_seq_len}\n"
        _str += self.tsfm_encoder.model_dumps() + "\n"
        _str += self.flatten_layer.model_dumps() + "\n"
        _str += self.device_encoder.model_dumps() + "\n"
        _str += self.decoder.model_dumps() + "\n"
        _str += "**********************************************"

        return _str
    
    def assign_config(self, config):
        self.tsfm_encoder.assign_config(config)
        self.flatten_layer.assign_config(config)
        self.device_encoder.assign_config(config)
        self.decoder.assign_config(config)
        self.init_model()