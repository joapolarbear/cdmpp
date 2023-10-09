import pickle
import numpy as np
import os

from utils.util import read_yes, warning, load_pickle
import utils.env as cdpp_env

from metalearner.data.map2norm import TransformerHub, get_method_id

import torch

class DataMetaInfo:
    def __init__(self, xydata=None, pre_add_dims=None, to_norm_output=True):
        # TODO: should save output_norm_method when metainfo is saved
        self.output_norm_method = cdpp_env.PROJECT_CFG['OUTPUT_NORM_METHOD']
        self.input_norm_method = cdpp_env.PROJECT_CFG.get('INPUT_NORM_METHOD', "min-max")
        if xydata is None:
            return
        self.pre_add_dims = pre_add_dims
        self.total_fea_len = xydata.shape[1]

        _feature = self.feature(xydata)
        self.input_max = np.max(_feature, axis=0) + 1e-6
        self.input_min = np.min(_feature, axis=0) + 1e-6
        self.output_avg = np.average(xydata[:, 0].astype(float))
        self.output_std = np.std(xydata[:, 0].astype(float))

        self.to_norm_output = to_norm_output
        self.tsfm_hub = TransformerHub()
        self.init_default()

        self.percentile_dict = None
    
    def init_default(self):
        self.to_norm_input = True

        # self.to_norm_input = False
        # self.to_norm_output = False

        ### Check if tsfm_hub is assigned correctly
        # self.tsfm_hub.check_y_norm_method(self.output_norm_method)

        self.log_trans_dim_idx = None
        if not self.to_norm_input:
            if not read_yes("Are you sure not to norm input?"):
                exit(0)
    
    def has_std(self):
        return self.pre_add_dims and "std" in self.pre_add_dims

    def update(self, input_max, output_avg, output_std, to_norm_output=True):
        self.input_max = input_max
        self.output_avg = output_avg
        self.output_std = output_std

        self.to_norm_output = to_norm_output
    
    def register_log_tran_dims(self, log_trans_dim_idx):
        self.log_trans_dim_idx = log_trans_dim_idx

    def norm_input(self, _input):
        if not self.to_norm_input:
            return _input
        if self.log_trans_dim_idx is not None:
            raise NotImplementedError("Not adapt to Distribution Transformers")
            _input_max = np.copy(self.input_max)
            _input_min = np.copy(self.input_min)
            for dim in self.log_trans_dim_idx:
                _input[:, dim] = np.log(_input[:, dim]+1)
                _input_max[dim] = np.log(_input_max[dim]+1)
                _input_min[dim] = np.log(_input_min[dim]+1) 
            return (_input - _input_min) / (_input_max - _input_min + 1e-6)
        else:
            return self.tsfm_hub.get_distribution_tsfm("x").transform(_input)
            
    def de_norm_input(self, _input):
        if not self.to_norm_input:
            return _input
        if self.log_trans_dim_idx is not None:
            raise NotImplementedError("Not adapt to Distribution Transformers")
            _input_max = np.copy(self.input_max)
            _input_min = np.copy(self.input_min)
            for dim in self.log_trans_dim_idx:
                _input_max[dim] = np.log(_input_max[dim]+1)
                _input_min[dim] = np.log(_input_min[dim]+1)
            _input = _input * (_input_max - _input_min + 1e-6) + _input_min
            for dim in self.log_trans_dim_idx:
                _input[:, dim] = np.exp(_input[:, dim]) - 1
            return _input
        else:
            return self.tsfm_hub.get_distribution_tsfm("x").inverse_transform(_input) 
        
    def standardize_output(self, _output):
        return self.tsfm_hub.get_distribution_tsfm("y").transform(_output)

        if self.output_norm_method.startswith("cls_"):
            class_base = float(self.output_norm_method.split("cls_")[1])
            return (_output >= class_base).astype(int)
        else:
            raise ValueError(self.output_norm_method)
    
    def de_standardize_output(self, _output):
        return self.tsfm_hub.get_distribution_tsfm("y").inverse_transform(_output)

        if self.output_norm_method.startswith("cls_"):
            raise NotImplementedError("Do not support de-standardization for discretized Y")
        else:
            raise ValueError(self.output_norm_method)
    
    @staticmethod
    def metapath2tsfm_dir(path):
        dir_basename = ".".join(os.path.basename(path).split(".")[:-1] + ["distribution_transform"])
        return os.path.join(os.path.dirname(path), dir_basename)
        
    @staticmethod
    def _load(path):
        ret = load_pickle(path)
        tsfm_hub = TransformerHub()
        if len(ret) == 10:
            raise ValueError("The cost model is out of date, missing input_min, please update it")
            input_max, output_avg, \
                output_std, pre_add_dims, \
                total_fea_len, to_norm_output, \
                log_trans_dim_idx, percentile_dict, \
                di_avg, di_std = ret
            input_min = None
            old_version = True
        elif len(ret) == 11:
            input_max, input_min, output_avg, \
                output_std, pre_add_dims, \
                total_fea_len, to_norm_output, \
                log_trans_dim_idx, percentile_dict, \
                di_avg, di_std = ret
            warning(f"To be deprecated, separate fitter method from DataMetaInfo")
            tsfm_hub.parse_y_norm_method("log", None)
            tsfm_hub.parse_x_norm_method("min-max", None, {"min": input_min, "max": input_max})
            old_version = True
        elif len(ret) == 12:
            input_max, input_min, output_avg, \
                output_std, pre_add_dims, \
                total_fea_len, to_norm_output, \
                log_trans_dim_idx, percentile_dict, \
                di_avg, di_std, y2norm_fitter = ret
            warning(f"To be deprecated, separate fitter method from DataMetaInfo")
            tsfm_hub.parse_y_norm_method("log", None)
            tsfm_hub.parse_x_norm_method("min-max", None, {"min": input_min, "max": input_max})
            old_version = True
        elif len(ret) == 9:
            input_max, input_min, output_avg, \
                output_std, pre_add_dims, \
                total_fea_len, to_norm_output, \
                log_trans_dim_idx, percentile_dict = ret
            output_norm_method = cdpp_env.PROJECT_CFG['OUTPUT_NORM_METHOD']
            input_norm_method = cdpp_env.PROJECT_CFG.get('INPUT_NORM_METHOD', "min-max")
            tsfm_hub.load(DataMetaInfo.metapath2tsfm_dir(path),
                target2method={"x": input_norm_method, "y": output_norm_method})
            old_version = False
        else:
            raise ValueError()

        return input_max, input_min, output_avg, \
                output_std, pre_add_dims, \
                total_fea_len, to_norm_output, \
                log_trans_dim_idx, percentile_dict, \
                tsfm_hub, old_version

    def reload_tsfm_hub(self, path):
        self.tsfm_hub.load(DataMetaInfo.metapath2tsfm_dir(path),
            target2method={"x": self.input_norm_method, "y": self.output_norm_method})

    def load(self, path):
        self.input_max, self.input_min, self.output_avg, \
                self.output_std, self.pre_add_dims, \
                self.total_fea_len, self.to_norm_output, \
                self.log_trans_dim_idx, self.percentile_dict, \
                self.tsfm_hub, old_version = DataMetaInfo._load(path)
        if old_version:
            self.save(path)
        self.init_default()
    
    @staticmethod
    def load_init(path):
        dm = DataMetaInfo()
        dm.input_max, dm.input_min, dm.output_avg, \
                dm.output_std, dm.pre_add_dims, \
                dm.total_fea_len, dm.to_norm_output, \
                dm.log_trans_dim_idx, dm.percentile_dict, \
                dm.tsfm_hub, old_version = DataMetaInfo._load(path)
        if old_version:
            dm.save(path)
        dm.init_default()
        return dm

    def save(self, path):
        with open(path, "wb") as fp:
            pickle.dump([
                self.input_max,
                self.input_min,
                self.output_avg,
                self.output_std,
                self.pre_add_dims,
                self.total_fea_len,
                self.to_norm_output,
                self.log_trans_dim_idx,
                self.percentile_dict
            ], fp)
        
        self.tsfm_hub.save(DataMetaInfo.metapath2tsfm_dir(path))

    @property
    def feature_end_idx(self):
        return self.total_fea_len

    @property
    def feature_start_idx(self):
        ### TODO (huhanpeng)
        return len(self.pre_add_dims)

        ### assert "FLOPs" is the first additional feature 
        # in front of the customized tir features
        if "FLOPs" in self.pre_add_dims:
            return self.pre_add_dims.index("FLOPs")
        else:
            return len(self.pre_add_dims)
    
    def feature(self, xydata):
        if len(xydata.shape) == 1:
            return xydata[self.feature_start_idx:self.feature_end_idx]
        elif len(xydata.shape) == 2:
            return xydata[:, self.feature_start_idx:self.feature_end_idx]
        else:
            raise ValueError()
    
    def output(self, xydata):
        if len(xydata.shape) == 1:
            return xydata[0].astype(float)
        elif len(xydata.shape) == 2:
            return xydata[:, 0].astype(float)
        else:
            raise ValueError()

    @property
    def feature_len(self):
        return self.input_max.shape[0]


class ASTMetaInfo(DataMetaInfo):
    ''' In this case, each data sample is an array of 
        [avg, std, flops, ast_features, node_ids, serialized_tree], where the last 3 entries are unique to AST-based features
            * ast_features: an N_leaf x N_entry array
            * node_ids: an N_leaf array denotes node_ids of leaf nodes corresponding to ast_features
            * serialized_tree: an 1D array
    '''
    def __init__(self, xydata=None, pre_add_dims=None, to_norm_output=True):
        self.ast_feature_dim = 3

        super(ASTMetaInfo, self).__init__(xydata=xydata, 
            pre_add_dims=pre_add_dims, to_norm_output=to_norm_output)
    
    def register_log_tran_dims(self, log_trans_dim_idx):
        raise NotImplementedError("ASTMetaInfo does NOT support log-transformation, since the data has been log-trainsformed")

    def norm_input(self, _input):
        if not self.to_norm_input:
            return _input
        assert self.log_trans_dim_idx is None
        return self.tsfm_hub.get_distribution_tsfm("x").transform(_input)
    
    def de_norm_input(self, _input):
        if not self.to_norm_input:
            return _input
        assert self.log_trans_dim_idx is None
        return self.tsfm_hub.get_distribution_tsfm("x").inverse_transform(_input)
    
    def feature(self, xydata):
        if len(xydata.shape) == 1:
            return xydata[self.ast_feature_dim]
        elif len(xydata.shape) == 2:
            return np.concatenate(xydata[:, self.ast_feature_dim], axis=0)
        else:
            raise ValueError()
    
    @property
    def feature_end_idx(self):
        raise NotImplementedError("Does NOT support")
    
    @property
    def feature_start_idx(self):
        raise NotImplementedError("Does NOT support")
    
    @staticmethod
    def load_init(path):
        dm = ASTMetaInfo()
        dm.input_max, dm.input_min, dm.output_avg, \
                dm.output_std, dm.pre_add_dims, \
                dm.total_fea_len, dm.to_norm_output, \
                dm.log_trans_dim_idx, dm.percentile_dict, \
                dm.tsfm_hub, old_version = DataMetaInfo._load(path)
        dm.init_default()
        if old_version:
            dm.save(path)
        return dm
