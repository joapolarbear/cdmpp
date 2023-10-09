import os
import json
import math
import numpy as np
import re
from collections import namedtuple

import pickle
from typing import Union, Dict

from utils.base import ALL_DIMENSION, DIMENSION_NAME, PROJECT_DIR
from utils.op_info import FULL_HEADERS
from utils.op_info import feature_encoder
from utils.device_info import ALL_GPU_MODEL

INFINITE_ERROR = 1e6
LARGEST_MAPE = 100
LARGEST_COST = 1e6
STD2AVG_UPPER_BOUND = 0.01

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

def fig_base(fig_num, row_first=True):
    if row_first:
        row_num = math.ceil(math.sqrt(fig_num))
        col_num = math.ceil(fig_num / row_num)
    else:
        col_num = math.ceil(math.sqrt(fig_num))
        row_num = math.ceil(fig_num / col_num)
    return row_num * 100 + col_num * 10

def range2str(_range):
    lower, upper = _range
    ret = []
    if lower is not None:
        ret.append(f"ge{lower:.1e}")
    if upper is not None:
        ret.append(f"le{upper:.1e}")
    assert len(ret) > 0
    return "_".join(ret)

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

PromptCache = {}

def read_yes(prompt, yes=False):
    if yes:
        return True
    if prompt in PromptCache:
        _input = PromptCache[prompt]
    else:
        _input = input(dpro.base.bcolors.CVIOLET + "\n"+prompt + " [Y/n]: " + dpro.base.bcolors.ENDC)
        PromptCache[prompt] = _input
    if len(_input) == 0 or _input.lower() in ['y', "yes", "1"]:
        return True
    else:
        return False

def warn_once(prompt, others=""):
    if prompt in PromptCache:
        pass
    else:
        PromptCache[prompt] = "y"
        if len(others) > 0:
            prompt += f", {str(others)}"
        warning(prompt)

def check_exit(prompt):
    if read_yes(prompt):
        return
    else:
        exit(0)

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

import dpro
def notify(_str):
    print(dpro.base.bcolors.CYELLOW + _str + dpro.base.bcolors.ENDC)

def warning(_str):
    print(dpro.base.bcolors.CRED + f"[Warning] {_str}" + dpro.base.bcolors.ENDC)

def good_new(_str):
    print(dpro.base.bcolors.CGREEN + _str + dpro.base.bcolors.ENDC)

def prompt(_str):
    print(dpro.base.bcolors.CBLUE + _str + dpro.base.bcolors.ENDC)

def random_select(origin_data, select_num, index=False):
    n_samples = len(origin_data)
    mask = np.zeros(n_samples, dtype=bool)
    # np.random.seed(0)
    pick_idx = np.random.choice(n_samples, math.ceil(select_num), replace=False)
    mask[pick_idx] = True
    if index:
        return mask, None
    else:
        return origin_data[mask], origin_data[~mask]

def load_pickle(path):
    with open(path, 'rb') as fp:
        ret = pickle.load(fp)
    return ret

def task_repr(task_file_name):
    if isinstance(task_file_name, list):
        if len(task_file_name) == 1:
            return task_repr(task_file_name[0])
        rst = sorted([os.path.basename(_f).split(".npy")[0] for _f in task_file_name])
        return f"{rst[0]}-to-{rst[-1]}"
    return os.path.basename(task_file_name).split(".npy")[0]


class SplitAttr:
    filtered = 0
    default = 0
    train = 0b001
    test = 0b010
    ALL = 0b111
    @staticmethod
    def _to_implt(t, code):
        if isinstance(t, int):
            return t | code
        elif isinstance(t, (np.ndarray, list)):
            t = np.array(t)
            return np.bitwise_or(t, np.ones_like(t, dtype=int) * code)
        else:
            raise NotImplementedError(f"Not implement for type {type(t)}")
    @staticmethod
    def _is_implt(t, code):
        if isinstance(t, int):
            return t & code != 0
        elif isinstance(t, (np.ndarray, list)):
            t = np.array(t)
            return np.bitwise_and(t, np.ones_like(t, dtype=int) * code).astype(bool)
        else:
            raise NotImplementedError(f"Not implement for type {type(t)}")

    @staticmethod
    def to_train(t):
        return SplitAttr._to_implt(t, SplitAttr.train)
    @staticmethod
    def to_test(t):
        return SplitAttr._to_implt(t, SplitAttr.test)
    @staticmethod
    def is_train(t):
        return SplitAttr._is_implt(t, SplitAttr.train)
    @staticmethod
    def is_test(t):
        return SplitAttr._is_implt(t, SplitAttr.test)

class Device2Task:
    def __init__(self):
        ''' Example
            "t4": {
                "tasks": [task1, task2],
                "root_path": root_path
                "attr": { 
                    "split": 0 # no specified split
                }
            }
        '''
        self.device2info: Dict[str, Dict] = {}
        self.data_split_mode = "default"

    def __len__(self):
        return len(self.device2info)
    
    @property
    def is_cross_device(self):
        return len(self.device2info) > 1
    
    @property
    def devices(self):
        return list(self.device2info.keys())
    
    def __getitem__(self, device):
        return self.device2info[device]
    
    def __setitem__(self, device, values):
        self.device2info[device] = values
    
    def __iter__(self):
        for device, _info in self.device2info.items():
            if "tasks" in _info:
                yield device, _info["tasks"]
            else:
                tasks = list(_info.get("train", []))
                tasks += list(_info.get("test", []))
                assert len(tasks) > 0, (device, _info)
                yield device, tasks
    
    def parse_device_split_mode(self, device):
        assert self.data_split_mode == "by_device"
        if "attr" not in self.device2info[device]:
            return SplitAttr.default
        return self.device2info[device]["attr"].get("split", SplitAttr.default)
    
    def parse_split_mode(self, device, task_file):
        if self.data_split_mode == "by_device":
            return self.parse_device_split_mode(device)
        elif self.data_split_mode == "default":
            return SplitAttr.default
        elif self.data_split_mode == "by_net":
            assert "tasks" not in self.device2info[device]
            if "train" in self.device2info[device] and task_file in self.device2info[device]["train"]:
                return SplitAttr.train
            elif "test" in self.device2info[device] and task_file in self.device2info[device]["test"]:
                return SplitAttr.test
            else:
                return SplitAttr.default
        else:
            raise ValueError(f"Invalid data split mode {self.data_split_mode}")

    def __str__(self):
        _str = ""
        for device in self.device2info:
            _str += f"{device}: "
            _str += f" {len(self.device2info[device].get('tasks', []))} default tasks"
            _str += f", {len(self.device2info[device].get('train', []))} train tasks"
            _str += f", {len(self.device2info[device].get('test', []))} test tasks"
            _str += "\n"
        return _str

class MODE_DEL:
    source2split = '.'
    inter_device = ','
    inner_device = '+'


def sample_task_files(feature_root_path: str, mode: str, gpu_model: str, absolute_path=True) -> Device2Task:
    ''' Parse the source of data and how to partition it

        1. mode = <source_mode>.<split_mode>
        2. <source_mode> = device_mode1,device_mode2,...,device_mode3
            device_mode1=device:sample_mode1+sample_mode2+...+sample_mode3
            1. sample_mode could be one of samplek, cross, list-A-B-C or network-A-B

            ### Sample 200 task files
            mode = "sample200"

            ### Sample 200 task files along with tasks related to some specific networks
            mode = "sample200,network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8"

            ### Specify some tasks
            mode = t4:list-0-1-2
        3. <split_mode> is optional, 
            1. if not set, shuffle the dataset and partition it randomly
            2. by_device: e.g., <split_mode> = train-t4-v100,test-a100
            3. by_net: e.g., <split_mode> = train-A-B,test-C
        4. Special rules
            1. Default rule: If train- is not specified but test- is given, the remaining devices 
               or nets that are not specified explicitly belong to the training set
            2. Empty rule: train- is given explicitly, but without concrete devices 
               or nets, it means the training set is empty
            3. All rule: with train-all, all tasks will be taken as the training data
    '''

    device2task = Device2Task()
    if MODE_DEL.source2split in mode:
        data_source_mode, data_split_mode = mode.split(MODE_DEL.source2split)
    else:
        data_source_mode = mode
        data_split_mode = None
    
    ### Parse partition index to guide how to partition the dataset
    data_split_index = {}
    if data_split_mode is not None:
        if data_split_mode.startswith("by_device:"):
            device2task.data_split_mode = "by_device"
            data_split_index["by_device"] = {}
            for _split_mode in data_split_mode.split("by_device:")[1].split(","):
                ### Example of _split_mode: train-t4-v100
                _split_mode_list = re.findall(r"t4|v100|a100|p100|k80|e5-2673|epyc-7452|graviton2|platinum-8272|hl-100", _split_mode)
                # _split_mode_list = _split_mode.split("-")
                data_split_index["by_device"][_split_mode.split("-")[0]] = _split_mode_list
            ### Check the default split attr
            if "train" in data_split_index["by_device"]:
                if "test" in data_split_index["by_device"]:
                    data_split_index["by_device"]["default"] = SplitAttr.default
                else:
                    data_split_index["by_device"]["default"] = SplitAttr.test
            elif "test" in data_split_index["by_device"]:
                data_split_index["by_device"]["default"] = SplitAttr.train
            else:
                raise ValueError(f"Both training and test split is not specified for {data_split_mode}")
        elif data_split_mode.startswith("by_net:"):
            device2task.data_split_mode = "by_net"
            data_split_index["by_net"] = {}
            for _split_mode in data_split_mode.split("by_net:")[1].split(","):
                ### Example of _split_mode: train-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8
                _split_mode_list = _split_mode.split("-")
                data_split_index["by_net"][_split_mode_list[0]] = _split_mode_list[1:]
            ### Check the default split attr
            if "train" in data_split_index["by_net"]:
                if "test" in data_split_index["by_net"]:
                    data_split_index["by_net"]["default"] = SplitAttr.default
                else:
                    data_split_index["by_net"]["default"] = SplitAttr.test
            elif "test" in data_split_index["by_net"]:
                data_split_index["by_net"]["default"] = SplitAttr.train
            else:
                raise ValueError(f"Both training and test split is not specified for {data_split_mode}")
        else:
            raise ValueError(f"Invalid data split mode {data_split_mode}")
    assert not ("by_device" in data_split_index and "by_net" in data_split_index), \
        "Can not split data by device and net at the same time"

    ### Get tasks and corresponding partition attr
    for _mode in data_source_mode.split(MODE_DEL.inter_device):
        if ":" in _mode:
            _device, _mode_detail = _mode.split(":")
        else:
            _device = gpu_model
            _mode_detail = _mode
        
        # assert _device.upper() in ALL_GPU_MODEL

        if "by_device" in data_split_index:
            root_path, files_to_test = _sample_task_files(os.path.join(
                feature_root_path, _device), _mode_detail, absolute_path=absolute_path)
            _split_attr = SplitAttr.default
            decided = False
            if "train" in data_split_index["by_device"] and (
                    "all" in data_split_index["by_device"]["train"] or
                    _device in data_split_index["by_device"]["train"]):
                _split_attr = SplitAttr.to_train(_split_attr)
                decided = True
            if "test" in data_split_index["by_device"] and (
                    "all" in data_split_index["by_device"]["test"] or
                    _device in data_split_index["by_device"]["test"]):
                _split_attr = SplitAttr.to_test(_split_attr)
                decided = True
            if not decided:
                _split_attr = data_split_index["by_device"]["default"]
        
            decided = False
            if SplitAttr.is_train(_split_attr):
                device2task[_device] = {"train": files_to_test} 
                decided = True
            if SplitAttr.is_test(_split_attr):
                device2task[_device] = {"test": files_to_test}
                decided = True
            if not decided:
                device2task[_device] = {}
                print (f"[Warning] do NOT decide partition for device {_device}")
                _split_attr = SplitAttr.default
            device2task[_device]["root_path"] = root_path
            device2task[_device]["attr"] = {"split": _split_attr}
        elif "by_net" in data_split_index:
            network2files = {}
            root_path, files_to_test = _sample_task_files(os.path.join(feature_root_path,
                _device), _mode_detail, absolute_path=absolute_path, network2files=network2files)
            device2task[_device] = {}
            for _network in network2files:
                for _bs in network2files[_network]:
                    _split_attr = SplitAttr.default
                    decided = False
                    if "train" in data_split_index["by_net"] and (
                            "all" in data_split_index["by_net"]["train"] or 
                            _network in data_split_index["by_net"]["train"] or 
                            f"{_network}_bs{_bs}" in data_split_index["by_net"]["train"]):
                        _split_attr = SplitAttr.to_train(_split_attr)
                        decided = True
                    if "test" in data_split_index["by_net"] and (
                            "all" in data_split_index["by_net"]["test"] or
                            _network in data_split_index["by_net"]["test"] or
                            f"{_network}_bs{_bs}" in data_split_index["by_net"]["test"]):
                        _split_attr = SplitAttr.to_test(_split_attr)
                        decided = True
                    if not decided:
                        _split_attr = data_split_index["by_net"]["default"]

                    decided = False
                    if SplitAttr.is_train(_split_attr):
                        if "train" not in device2task[_device]:
                            device2task[_device]["train"] = set()
                        device2task[_device]["train"] = device2task[_device]["train"].union(network2files[_network][_bs])
                        decided = True
                    if SplitAttr.is_test(_split_attr):
                        if "test" not in device2task[_device]:
                            device2task[_device]["test"] = set()
                        device2task[_device]["test"] = device2task[_device]["test"].union(network2files[_network][_bs])
                        decided = True
                    if not decided:
                        print(f"[Warning] Fail to decide partition for device {_device}'s {_network}")
                    
            device2task[_device]["root_path"] = root_path
        else:
            root_path, files_to_test = _sample_task_files(os.path.join(
                feature_root_path, _device), _mode_detail, absolute_path=absolute_path)
            device2task[_device] = {
                "tasks": files_to_test,
                "root_path": root_path,
                "attr": {}
            }

    return device2task

def _sample_task_files(device_feature_path, mode, absolute_path=False, network2files=None):
    assert os.path.exists(device_feature_path), f"{device_feature_path} does NOT exist"
    root_path, _, files = list(os.walk(device_feature_path))[0]
    _files_to_test = sorted([f for f in files if f.endswith(".npy")], key=lambda f: int(f.split(".")[0]))
    if absolute_path:
        _files_to_test = [os.path.join(root_path, f) for f in _files_to_test]
        
    assert os.path.exists(os.path.join(root_path, "network2split.json")), root_path
    with open(os.path.join(root_path, "network2split.json"), 'r') as fp:
        network2split = json.load(fp)
    
    with open(os.path.join(root_path, "split_stat.json"), 'r') as fp:
        split_stat = json.load(fp)
        
    files_to_test = []
    for _mode in mode.split(MODE_DEL.inner_device):
        if _mode == "cross":
            files_to_test = _files_to_test
        elif _mode.startswith("sample"):
            sample_num = int(_mode.split("sample")[1])
            files_to_test = _files_to_test[:sample_num]
        elif _mode.startswith("random"):
            sample_num = min(int(_mode.split("random")[1]), len(_files_to_test))
            selected_idx = np.random.choice(len(_files_to_test), sample_num, replace=False)
            files_to_test = np.array(_files_to_test)[selected_idx].tolist()
        elif _mode.startswith("list-"):
            match = re.search(r"^list-(?P<_list>(\w+)(- *\w+)*)$", _mode)
            assert match, f"Invalide mode {_mode}"
            for _file in match["_list"].replace(" ", "").split("-"):
                if not _file.endswith(".npy"):
                    _file += ".npy"
                assert _file in files, f"File {_file} does NOT exist at {root_path}"
                files_to_test.append(os.path.join(root_path, _file) if absolute_path else _file)
        elif _mode.startswith("network"):
            ### E.g., _mode="network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8"
            additional_files = set()
            network_bs_str = _mode[_mode.find("network")+len("network")+1:].split(",")[0]
            for network in network_bs_str.split("-"):
                # match = re.search(r"^(?P<network_name>(resnet)_(18|50))_bs(?P<bs>\d+)$", network)
                match = re.search(r"^(?P<network_name>\w+)_(bs(?P<bs>\d+))?$", network)
                rst = match.groupdict()
                network_name = rst["network_name"]
                if rst["bs"] is None:
                    ### Fuzzy match
                    split_files = []
                    for _bs in network2split[network_name]:
                        split_files += network2split[network_name][_bs]
                else:
                    split_files = network2split[network_name][rst["bs"]]
                ### Convert to file path and remove those tasks that are not well profiled
                if absolute_path:
                    split_files = [os.path.join(root_path, f"{_file}.npy") for _file in split_files 
                        if split_stat[str(_file)]["shape"] is not None]
                else:
                    split_files = [f"{_file}.npy" for _file in split_files 
                        if split_stat[str(_file)]["shape"] is not None]
                print(f"[Dataset] Found {len(split_files)} tasks from network={network_name}, bs={rst['bs']}")
                additional_files = additional_files.union(split_files)
            to_add = list(additional_files.difference(set(files_to_test)))
            print(f"[Dataset] Among {len(additional_files)} new tasks, *{len(to_add)}* is not in sampled tasks ")
            files_to_test += to_add
        else:
            raise ValueError(f"Invalid mode {_mode}")

    if len(files_to_test) == 0:
        warning(f"No task files are selected under {device_feature_path}, "
                f"given {sample_num} sample_num and {mode} mode")

    if network2files is not None:
        for _file in files_to_test:
            task_id = os.path.basename(_file).split(".npy")[0]
            for network_str in split_stat[task_id]["networks"]:
                ### e.g., network_str = "(densenet_121,[(1,3,224,224)]),cuda"
                rst = re.search(r"^\((?P<network_name>\w+),\[\((?P<bs>\d+),.+\)\]\),(?P<device>\w+)$", network_str).groupdict()
                network = rst["network_name"]
                bs = int(rst["bs"])
                if network not in network2files:
                    network2files[network] = {}
                if bs not in network2files[network]:
                    network2files[network][bs] = set()
                network2files[network][bs].add(_file)

    return root_path, files_to_test

def test_sample_task_files(feature_root_path, learning_params):
    for mode in [
        "t4:sample200+network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8,v100:sample200",
        "t4:sample200+network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8,v100:sample200.by_device:train-t4,test-v100",
        "t4:sample200+network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8,v100:sample200.by_device:test-v100",
        "t4:sample200+network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8,v100:sample200.by_device:train-,test-v100",
    ]:
        # import pdb; pdb.set_trace()
        print("mode: ", mode)
        device2task = sample_task_files(feature_root_path, mode, learning_params["gpu_model"])
        print(str(device2task))

### Dump and test training and test rst
def partition_data_based_on_train_rst(x, y, y_pred, meta_info, path=None):
    de_norm_true = meta_info.de_standardize_output(y).flatten()
    de_norm_preds = meta_info.de_standardize_output(y_pred).flatten()

    elementwise_mape = np.abs(de_norm_true - de_norm_preds) / de_norm_true
    mape_sorted_idx = np.argsort(elementwise_mape)
    sample_num = 100 if len(x) > 200 else len(x)//2
    best_samples = mape_sorted_idx[:sample_num]
    worst_samples = mape_sorted_idx[-sample_num:]
    stat_rst = {
        "MAPE": np.average(elementwise_mape),
        f"MAPE_of_Best_{sample_num}": np.average(elementwise_mape[best_samples]),
        f"MAPE_of_Worst_{sample_num}": np.average(elementwise_mape[worst_samples]),
        "Y": de_norm_true,
        "X": x,
        "Y_pred": de_norm_preds,
        f"best_{sample_num}_idxs": best_samples,
        f"worst_{sample_num}_idxs": worst_samples
    }
    if path is not None:
        with open(path, "wb") as fp:
            pickle.dump(stat_rst, fp)

def retrieve_partition_data_rst(path):
    ''' 
    Return
    ------
    stat_rst: dict
    {
        "MAPE": <MAPE of all data>,
        "MAPE_of_Best_<k>": <MAPE of the top k best sampes>,
        "MAPE_of_Worst_<k>": <MAPE of the top k worst samples>,
        "Y": y,
        "X": x,
        "Y_pred": y_pred,
        f"best_<k>_idxs": <Sample indexes of the top k best sampes>,
        f"worst_<k>_idxs": <Sample indexes of the top k worst samples>
    }
    '''
    stat_rst = load_pickle(path)
    return stat_rst

def retrieve_latent_repr(path):
    ''' latent: List[Tuple[str, np.ndarray]]
        A list of latent representations, each latent representation is a pair of name and np.ndarray
        
        Return
        ------
        latent: Mapping[str, np.ndarray]
    '''
    if os.path.exists(path):
        latent = load_pickle(path)
        # latent = dict(latent)
        latent = latent[0]
    else:
        latent = None
    return latent


class Sample:
    ''' One or multiple samples in the form of x_tir, y, and x_device
    '''
    def __init__(self, x_tir: np.array, y: np.array, x_device=None):
        if x_device is None or len(x_device) == 0:
            self.is_cross_device = False
        else:
            self.is_cross_device = True

        if len(x_tir.shape) >= 2:
            assert len(x_tir) == len(y)
            if self.is_cross_device:
                assert len(x_tir) == len(x_device), (x_tir.shape, x_device.shape)
            self._size = len(x_tir)
        else:
            self._size = 1
        self.x_tir = x_tir
        self.y = y
        self.x_device = x_device
        
    @property
    def size(self):
        return self._size

    def __getitem__(self, index):
        return Sample(self.x_tir[index], self.y[index],
            self.x_device[index] if self.is_cross_device else self.x_device)
    
    def clear_samples(self, index, inplace=True):
        assert self.size > 1
        if isinstance(index, int):
            index = np.array(int(index))
        elif isinstance(index, (np.ndarray, list)):
            pass
        elif isinstance(index, set):
            index = list(index)
        
        
        mask = np.ones_like(self.y, dtype=bool)
        mask[index] = False
        _x_device = self.x_device[mask] if self.is_cross_device else self.x_device
        if inplace:
            self.x_tir, self.y, self.x_device = self.x_tir[mask], self.y[mask], _x_device
        else:
            return Sample(self.x_tir[mask], self.y[mask], _x_device)

        self._size = len(self.y)

class TrainTestPair:
    def __init__(self, train, val):
        self.train = train
        self.val = val
