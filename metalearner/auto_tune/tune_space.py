import os
import json

from utils.util import prompt

def int_convert(x):
    return int(x)

def pow_convert(x):
    return pow(10, int(x))

class AutotuneSpace:
    def __init__(self):
        self.validator = []

        ### Used for Grid Search, manually list all possible values for each 
        # hyper-parameter
        self.configs = {
            "embed_layer_unit": [128, 512, 1024],
            # "embed_layer_num": [1, 4, 32],
            "embed_layer_num": [1, 3, 5],
            "mlp_layer_unit": [128, 512, 1024],
            # "mlp_layer_num": [0, 4, 32],
            "mlp_layer_num": [1, 3, 5],
            "embed_feature_len": [128, 512, 1024],
            # "opt_type": ["adam", "sgd", "rms"],
            "opt_type": ["adam"],
            "wd": [1e-04, 1, 0.01],
            "lr": [0.1, 0.001, 1e-07, 1e-05],
            "batch_size": [32, 8, 128],
        }
    
    def register_config_filter(self, check_func):
        self.validator.append(check_func)
    
    def check_config(self, config):
        for check_func in self.validator:
            if not check_func(config):
                return False
        return True


class GridAutotuneSpace(AutotuneSpace):
    def __init__(self):
        super(GridAutotuneSpace, self).__init__()

        ### The fommer one is closer to the root node in the traverse tree
        self.config_names = [   
            "embed_layer_unit",
            "embed_layer_num",
            "mlp_layer_unit",
            "mlp_layer_num",
            "embed_feature_len",
            "opt_type",
            "wd",
            "lr",
            "batch_size",
        ]

        self.config_lens = [len(self.configs[cfg_name]) for cfg_name in self.config_names] 
        self.config_len = 1
        for l in self.config_lens:
            self.config_len *= l

        self.config_len_hierachy = []
        tmp = 1
        for _len in self.config_lens[::-1]:
            self.config_len_hierachy.append(tmp)
            tmp *= _len
        self.config_len_hierachy = self.config_len_hierachy[::-1]

        self.cur_cfg_id = 0

    @property
    def len(self):
        return self.config_len

    def __getitem__(self, cfg_id):
        ret_cfg = {}
        res = cfg_id
        for idx, cfg_name in enumerate(self.config_names):
            _per_cfg_id = res // self.config_len_hierachy[idx]
            ret_cfg[cfg_name] = self.configs[cfg_name][_per_cfg_id]
            res = res % self.config_len_hierachy[idx]
        if not self.check_config(ret_cfg):
            return None
        return ret_cfg
    
    def ret_config_str(self, _config):
        return "-".join([f"{cfg_name}={_config[cfg_name]}" for cfg_name in self.config_names])

    def dump_all_configs(self, target_dir, split=1):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        all_cfgs = []
        for _ in range(split):
            all_cfgs.append([])
        cnt = 0
        for i in range(self.config_len):
            _config = self[i]
            if _config is not None:
                all_cfgs[i%split].append(_config)
                cnt += 1
        for split_id in range(split):
            file = os.path.join(target_dir, f"{split_id}.json")
            with open(file, 'w') as fp:
                json.dump(all_cfgs[split_id], fp)
        prompt(f"Successfully dump {cnt} configs to {split} files under {target_dir}")


class BOAutotuneSpace(AutotuneSpace):
    def __init__(self):
        super(BOAutotuneSpace, self).__init__()

        ### Used for BOA, list the upper and lower bounds of all possible values
        self.config_bound = {
            "hidden_layer_unit": (128, 1024),
            # "embed_layer_num": (1, 32),
            "embed_layer_num": (2, 8),
            "mlp_layer_num": (2, 8),
            "opt_type": (0, len(self.configs["opt_type"])),
            "wd": (-5, 1),
            "lr": (-8, 1),
            # "batch_size": (0, len(self.configs["batch_size"])),
            "batch_size": (4, 256)
        }

        self.config_convert = {
            "embed_layer_unit": int_convert,
            "embed_layer_num": int_convert,
            "mlp_layer_unit": int_convert,
            "mlp_layer_num": int_convert,
            "embed_feature_len": int_convert,
            "opt_type": lambda x: self.configs["opt_type"][
                int(x) % len(self.configs["opt_type"])],
            "wd": pow_convert,
            "lr": pow_convert,
            # "batch_size": lambda x: self.configs["batch_size"][
            #     int(x) % len(self.configs["batch_size"])],
            "batch_size": int_convert
        }

    @property
    def pbound_dict(self):
        return self.config_bound

    def convert(self, config_key, config_value):
        if config_key in self.config_convert:
            return self.config_convert[config_key](config_value)
        else:
            return config_value
    

class XGBAutotuneSpace(BOAutotuneSpace):
    def __init__(self):
        self.configs = {
            "max_depth": 3,
            "gamma": 0.0001,
            "min_child_weight": 1,
            "subsample": 1.0,
            "eta": 0.3,
            "lambda": 1.00,
            "alpha": 0,
        }

        ### Used for BOA, list the upper and lower bounds of all possible values
        self.config_bound = {
            "max_depth": (1, 100),
            "gamma": (-8, 1),
            "min_child_weight": (1, 3),
            "subsample": (0, 1),
            "eta": (0, 1)
        }

        self.config_convert = {
            "max_depth": int_convert,
            "gamma": pow_convert,
            "min_child_weight": int_convert
        }
