import os
import queue
import threading
import itertools
import random
import numpy as np
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # default level，display all information
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # only display warning and Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # only display Error
import tensorflow as tf

def gen_config(_range_dict, size, attrs):
    ret = [None] * len(attrs)
    for attr, spec in _range_dict.items():
        attr_idx = attrs.index(attr)
        if isinstance(spec, list):
            ret[attr_idx] = random.choices(spec, k=size)
        elif isinstance(spec, dict):
            if "ave" in spec:
                values = []
                while len(values) < size:
                    tmp = np.random.normal(
                        loc=spec["ave"], scale=spec["stdev"], size=size).astype(int)
                    for v in tmp:
                        if (spec["max"] and v > spec["max"]) or (spec["min"] and v < spec["min"]):
                            continue
                        values.append(v)
                ret[attr_idx] = values
            else:
                ret[attr_idx] = np.random.randint(
                    low=spec["min"], high=spec["max"], size=size, dtype=int)
    return ret

def gen_config_v2(_range_dict, attrs):
    tmp = [_range_dict[attr] for attr in attrs]
    return list(itertools.product(*tmp))

def config2str(config):
    _str = ""
    for _dim in config.keys():
        if len(_str) > 0:
            _str += ","
        _str += "{}={}".format(_dim, config[_dim])
    return _str

def str2config(config_str):
    _config = dict([_config.split("=") for _config in config_str.split(",")])
    for _dim in _config.keys():
        if _dim == "dtype":
            if "float32" in _config["dtype"] or "fp32" in _config["dtype"]:
                _config["dtype"] = tf.dtypes.float32
            elif "float16" in _config["dtype"] or "fp16" in _config["dtype"]:
                _config["dtype"] = tf.dtypes.float16
            else:
                raise ValueError(config_str)
        elif _dim == "padding":
            _config[_dim] = _config[_dim]
        else:
            _config[_dim] = int(float(_config[_dim]))
    return _config

class ConfigGenThread(threading.Thread):
    def __init__(self, range_dict, check_valid_fn=None):
        super(ConfigGenThread, self).__init__()
        self.go = True
        self.config_queue = queue.Queue()
        self.range_dict = range_dict
        self.check_valid_fn = check_valid_fn

    # def run(self):
    #     while self.go:
    #         if self.config_queue.qsize() < SAMPLE_SIZE / 10:
    #             ### Add new configs
    #             configs = gen_config(self.range_dict, SAMPLE_SIZE, self.attrs)
    #             for idx in range(SAMPLE_SIZE):
    #                 self.config_queue.put([attr_values[idx]
    #                                        for attr_values in configs])
    #             print("config_gen generates {} configs".format(SAMPLE_SIZE))
    #         time.sleep(0.5)

    def run(self):
        attrs = sorted(self.range_dict.keys())
        configs = gen_config_v2(self.range_dict, attrs)

        np.random.shuffle(configs)

        for config in configs:
            config = dict([(attrs[idx], value)
                             for idx, value in enumerate(list(config))])

            if self.check_valid_fn and not self.check_valid_fn(config):
                # print("[Warning] Ignore the config to avoid OOM: {}".format(config_str))
                continue
            else:
                self.add_cfg(config)

        print("config_gen generates {} configs".format(self.config_queue.qsize()))
    
    def add_cfg(self, config):
        self.config_queue.put(config)

    def join(self):
        self.go = False
        super(ConfigGenThread, self).join()
    
    def get(self):
        return self.config_queue.get(block=True, timeout=60)

    def empty(self):
        return self.config_queue.empty()

    def size(self):
        return self.config_queue.qsize()
    
    def dump(self, cache_path):
        with open(cache_path, 'w') as fp:
            first=True
            while not self.empty():
                _config = self.get()
                if not first:
                    fp.write("\n")
                fp.write(config2str(_config))
                first = False
    
    def load(self, cache_path):
        with open(cache_path, 'r') as fp:
            lines = fp.readlines()
            for config_str in lines:
                self.config_queue.put(str2config(config_str))
    
