import os
import json
import math
import numpy as np
from typing import Union, Dict
from utils.op_info import FULL_HEADERS
from utils.op_info import feature_encoder


def fig_base(fig_num, row_first=True):
    if row_first:
        row_num = math.ceil(math.sqrt(fig_num))
        col_num = math.ceil(fig_num / row_num)
    else:
        col_num = math.ceil(math.sqrt(fig_num))
        row_num = math.ceil(fig_num / col_num)
    return row_num * 100 + col_num * 10

def ret_model_name(s):
    split = s.split("_")
    return split[-1], ".".join(split[1:3])

def ret_dtype(s):
    return s.split("_")[-1]

def ret_bs(s):
    return int(s.split("_")[-1])

def bs2dir(bs):
    return "BS_{}".format(bs)

def convert_json2readable(trace_path, output_path):
    with open(trace_path, 'r') as fp:
        traces = json.load(fp)
    with open(output_path, 'w') as fp:
        json.dump(traces, fp, indent=4)
        

AXIS_UNIT = {
    "flops": "FLOPs",
    "size": "byte",
    "ai": "FLOP/byte",
    "arithmetic_intensity": "FLOP/byte",
    "perf": "FLOP/s",
    "performance": "FLOP/s",
    "ave": "ms"
}

def axis_label(dim):
    return "{} ({})".format(dim, AXIS_UNIT.get(dim.lower(), ""))

def config_unique_key(config):
    _key = ""
    for dim in ALL_DIMENSION:
        if len(_key) > 0:
            _key += "/"
        _key += "{}={}".format(dim, config[dim] if config[dim] else "?")
    return _key

def config_key2config(config_key):
    config = dict([tuple(x.split("=")) for x in config_key.split("/")])
    for dim in config.keys():
        if config[dim] == "?":
            config[dim] = None
    return config

def config_key_str(config_key):
    config = config_key2config(config_key)
    rst = ""
    for dim in ALL_DIMENSION:
        if config[dim]:
            if len(rst) > 0:
                rst += " "
            rst += "{}".format(config[dim])
    return rst

def str_convert(_str):
    try:
        return float(_str)
    except:
        return _str[1:-1]

def line2list(line):
    line = line.strip("\n")
    features = line[1:-1].split("], [")
    rst = []
    for feature in features:
        if feature.startswith("["):
            feature = feature[1:]
        if feature.endswith("]"):
            feature = feature[:-1]
        rst.append([str_convert(elem) for elem in feature.split(", ")])
    return rst

def line2dict(line):
    _dict = {}
    for key_value_str in re.findall('(\'[^,]+\': \'[^,]+\')', line[2:-2]):
        key, value = key_value_str.split("': '")
        _dict[key[1:]] = value[:-1]
    _dict["Operation"] = line[:-4].split("'Operation': '")[1]
    return _dict

def idw_average(nhbr_features, distance, weighted=True):
    ### Inverse distance weighting
    ### shape of nhbr_features = (# of ngbrs, # of dims)
    ### shape of distance = # of ngbrs
    if weighted:
        inverse_distance = 1 / (np.array(distance) + 1)
        weights = inverse_distance / np.sum(inverse_distance)
        ### reshape weights such that it has the same number of dimensions
        weighted_features = nhbr_features * weights[:, None]
        return np.average(weighted_features, axis=0)
    else:
        return np.average(nhbr_features, axis=0)

        
class Setting:
    def __init__(self, source_gpu, target_gpu,
            kernel_type=None,
            ave_lower_bound_ms=None,
            debug_kernel_list=None,
            target_op_type=None,
            target_dtype=None):
        self.source_gpu = source_gpu
        self.target_gpu = target_gpu
        self.kernel_type = kernel_type # one of gemm, conv2d_fprop
        self.ave_lower_bound = ave_lower_bound_ms
        self.debug_kernel_list = debug_kernel_list
        self.target_op_type = target_op_type
        self.target_dtype = target_dtype
    
    def __str__(self):
        _str = "{}_to_{}".format(self.source_gpu, self.target_gpu)
        if self.kernel_type is not None:
            _str += "_for_{}".format(self.kernel_type)
        if self.ave_lower_bound is not None:
            _str += "_ge{}ms".format(str(self.ave_lower_bound).replace(".", "_"))
        if self.target_op_type is not None:
            _str += "_{}".format(self.target_op_type)
        if self.target_dtype is not None:
            _str += "_{}".format(self.target_dtype)
        return _str

def gen_feature_id(feature):
    str_list = [str(int(v)) for v in feature[2:]]
    str_list[1] = "0"
    return ",".join(str_list)


class Filters:
    def __init__(self, filter=None):
        '''
        filter: a dict, 
            the key is the attribute, 
            value is a list of values of corresponding attribute we want to keep
            For attributes with no filters, collect data of all attributeds and calculate the average value
        '''
        self.filter = filter

    def serialize_filter(self, decode=False):
        ### convert a filter to a unique string
        ret = ""
        if self.filter is None:
            return ret
        for dim in sorted(self.filter.keys()):
            if len(self.filter[dim]) == 0:
                continue
            if decode:
                cvt = lambda x: x
            else:
                cvt = feature_encoder(dim)
            if len(ret) > 0:
                ret += "-"
            ret += "{}_{}".format(dim,
                                "_".join(sorted([str(cvt(v)) for v in self.filter[dim]])))
        return ret
    
    def not_in_filter(self, dim_name=None, target_value=None):
        ### Check whether to ignore target_value or not
        return dim_name in self.filter and len(self.filter[dim_name]) > 0 and target_value not in self.filter[dim_name]
    
    def check_filters(self):
        if "op_type" in self.filter and len(self.filter["op_type"]) > 1:
            raise ValueError("Currently we only support analyze data of one op type at one time, modify your filter: {}".format(
                self.filter["op_type"]))
    
    def apply_filters_to_op(self, op2xydata):
        ### Apply filter rules to op-level data
        ret = {}
        for op_type, _data in op2xydata.items():
            if self.not_in_filter(DIMENSION_NAME.op_type, op_type):
                continue
            ret[op_type] = []
            header = FULL_HEADERS[op_type]
            filter_converted = [None] * len(header)
            for dim, item in self.filter.items():
                if len(item) == 0:
                    filter_converted[header.index(dim)] = None
                else:
                    filter_converted[header.index(dim)] = [
                        feature_encoder(dim)(e) for e in item]
            for features in _data:
                keep = True
                for dim_idx, allowed_values in enumerate(filter_converted):
                    if allowed_values is None:
                        ### keep data if no filter rules are given for a specific dim
                        continue
                    if features[dim_idx] not in allowed_values:
                        keep = False
                        break
                if keep:
                    ret[op_type].append(features)
        return ret
    
class Scaler:
    def __init__(self, dump_path=None):
        self.upper = {}
        self.dump_path = dump_path
    
    def record(self, dim, value):
        if dim not in self.upper:
            self.upper[dim] = max(1, value)
        else:
            self.upper[dim] = max(value, self.upper[dim])
    
    def record_dims(self, dims, values):
        for dim, value in zip(dims, values):
            self.record(dim, value)
        
    def normalize(self, dim_list, features):
        ### Create a normalization_upper_list based on the dim_list first
        if isinstance(dim_list, list):
            norm_upper = np.array([self.upper[dim] for dim in dim_list])
        else:
            norm_upper = self.upper[dim_list]
        return features / norm_upper
    
    def denormalize(self, dim_list, norm_features):
        ### Create a normalization_upper_list based on the dim_list first
        if isinstance(dim_list, list):
            norm_upper = np.array([self.upper[dim] for dim in dim_list])
        else:
            norm_upper = self.upper[dim_list]
        return norm_features * norm_upper
    
    def combine(self, other):
        if other is None:
            return
        for dim in other.upper.keys():
            self.record(dim, other.upper[dim])
    
    def dump(self):
        with open(self.dump_path, 'w') as fp:
            json.dump(self.upper, fp)

    def load(self):
        if not os.path.exists(self.dump_path):
            print(f"[Scaler] dump path not exist, load nothing : {self.dump_path}")
            return
        else:
            print(f"[Scaler] successfully load norm_upper at {self.dump_path}")
        with open(self.dump_path, 'r') as fp:
            self.upper = json.load(fp)
