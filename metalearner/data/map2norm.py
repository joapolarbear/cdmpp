import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
import pickle
import os
from typing import Union, Dict, List
import json

import torch

from utils.util import warning, LARGEST_COST


class Transformer:
    def __init__(self, method):
        self.method_name = method

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

    def save(self, cache_dir):
        pass

    def load(self, cache_dir):
        pass

    @staticmethod
    def load_init(cache_dir, method_name):
        return Transformer("None")
    
    def __str__(self) -> str:
        return self.method_name


class MinMaxTransformer(Transformer):
    def __init__(self, _min, _max):
        super().__init__("min-max")
        assert isinstance(_min, (int, float, np.ndarray)), type(_min)
        assert isinstance(_max, (int, float, np.ndarray)), type(_max)
        self._min = _min
        self._max = _max

    def transform(self, X):
        return (X - self._min) / (self._max - self._min + 1e-6)

    def inverse_transform(self, X):
        return X * (self._max - self._min + 1e-6) + self._min

    def save(self, cache_dir):
        with open(os.path.join(cache_dir, f"{self.method_name}.pkl"), 'wb') as fp:
            pickle.dump([self._min, self._max], fp)
    
    def load(self, cache_dir):
        with open(os.path.join(cache_dir, f"{self.method_name}.pkl"), 'rb') as fp:
            self._min, self._max = pickle.load(fp)
    
    @staticmethod
    def load_init(cache_dir, method_name):
        with open(os.path.join(cache_dir, f"{method_name}.pkl"), 'rb') as fp:
            _min, _max = pickle.load(fp)
        return MinMaxTransformer(_min, _max)

    # def __str__(self) -> str:
    #     return f"{self.method_name}(min={self._min}, max={self._max})"


class StdTransformer(Transformer):
    def __init__(self, avg, std):
        super().__init__("std")
        self.avg = avg
        self.std = std

    def transform(self, X):
        return (X - self.avg) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.avg

    def save(self, cache_dir):
        with open(os.path.join(cache_dir, f"{self.method_name}.pkl"), 'wb') as fp:
            pickle.dump([self.avg, self.std], fp)
    
    def load(self, cache_dir):
        with open(os.path.join(cache_dir, f"{self.method_name}.pkl"), 'rb') as fp:
            self.avg, self.std = pickle.load(fp)
    
    @staticmethod
    def load_init(cache_dir, method_name):
        with open(os.path.join(cache_dir, f"{method_name}.pkl"), 'rb') as fp:
            avg, std = pickle.load(fp)
        return StdTransformer(avg, std)
    
    # def __str__(self) -> str:
    #     return f"{self.method_name}(avg={self.avg}, std={self.std})"


class LogTransformer(Transformer):
    def __init__(self):
        super().__init__("log")

    def transform(self, X):
        return np.log(X)

    def inverse_transform(self, X):
        if isinstance(X, torch.Tensor):
            return torch.exp(X)
        else:
            return np.exp(X)

    @staticmethod
    def load_init(cache_dir, method_name):
        return LogTransformer()
    

class DistTransformer(Transformer):
    def __init__(self, fitter, method):
        super().__init__(method)
        self.fitter = fitter
    
    def transform(self, X):
        if len(X.shape) == 2:
            return self.fitter.transform(X)
        elif len(X.shape) == 1:
            return self.fitter.transform(X.reshape((-1, 1))).flatten()
        elif len(X.shape) == 3:
            origin_shape = X.shape
            X = X.reshape((-1, origin_shape[-1]))
            X = self.fitter.transform(X)
            X = X.reshape(origin_shape)
            return X
        else:
            raise ValueError(f"Invalid data shape {X.shape}")
    
    def inverse_transform(self, X):
        X = X.astype("float64")
        if len(X.shape) == 2:
            ret = self.fitter.inverse_transform(X)
        elif len(X.shape) == 1:
            ret = self.fitter.inverse_transform(X.reshape((-1, 1))).flatten()
        elif len(X.shape) == 3:
            origin_shape = X.shape
            X = X.reshape((-1, origin_shape[-1]))
            X = self.fitter.inverse_transform(X)
            X = X.reshape(origin_shape)
            ret = X
        else:
            raise ValueError(f"Invalid data shape {X.shape}")
        ret = np.nan_to_num(ret, nan=1e-6, posinf=LARGEST_COST)
        return ret
    
    def save(self, cache_dir):
        with open(os.path.join(cache_dir, f"{self.method_name}.pkl"), 'wb') as fp:
            pickle.dump([self.fitter, self.method_name], fp)
    
    def load(self, cache_dir):
        with open(os.path.join(cache_dir, f"{self.method_name}.pkl"), 'rb') as fp:
            self.fitter, self.method_name = pickle.load(fp)
    
    @staticmethod
    def load_init(cache_dir, method_name):
        with open(os.path.join(cache_dir, f"{method_name}.pkl"), 'rb') as fp:
            fitter, fit_method = pickle.load(fp)
        return DistTransformer(fitter, fit_method)


METHOD_NAMES = ["Box-Cox", "Yeo-Johnson", "Quantile", "Quantile-Uniform", "min-max", "std", "log", "None"]
METHODS = [DistTransformer, DistTransformer, DistTransformer, DistTransformer,
    MinMaxTransformer, StdTransformer, LogTransformer, Transformer]
NO_ARG_METHOD_NAMES = ["log", "None"]

def get_method_id(method):
    if isinstance(method, str) and method.isdigit():
        method = int(method)
    if isinstance(method, str):
        return METHOD_NAMES.index(method)
    else:
        assert isinstance(method, int) and method < len(METHOD_NAMES), method
        return method

def _method_id2fitter(map_method_id):
    if map_method_id == 0:
        fitter = PowerTransformer(method="box-cox")
    elif map_method_id == 1:
        fitter = PowerTransformer(method="yeo-johnson")
    elif map_method_id == 2:
        rng = np.random.RandomState(304)
        # n_quantiles is set to the training set size rather than the default value
        # to avoid a warning being raised by this example
        fitter = QuantileTransformer(n_quantiles=500, 
            output_distribution="normal", random_state=rng)
    elif map_method_id == 3:
        rng = np.random.RandomState(304)
        fitter = QuantileTransformer(n_quantiles=500, 
            output_distribution="uniform", random_state=rng)
    else:
        raise ValueError(f"Invalid transformation method: {map_method_id}")
    return fitter

def get_y2norm_fitter_cls_name(map_method):
    if isinstance(map_method, str):
        assert map_method in METHOD_NAMES, (map_method, METHOD_NAMES)
        return map_method
    else:
        return METHOD_NAMES[map_method]

def try_load_cached_fitter(map_method, cache_dir):
    if isinstance(map_method, str):
        assert map_method in METHOD_NAMES, (map_method, METHOD_NAMES)
        map_method = METHOD_NAMES.index(map_method)
    
    cache_path = os.path.join(cache_dir, METHOD_NAMES[map_method])
    if not os.path.exists(cache_path):
        return None
    else:
        return DistTransformer.load_init(cache_dir)

def create_dist_transformer(map_method, X, tsfm_kwargs={}):
    map_method = get_method_id(map_method)

    if map_method <= 3:
        assert X is not None
        fitter = _method_id2fitter(map_method)
        if len(X.shape) == 2:
            fitter = fitter.fit(X)
        elif len(X.shape) == 1:
            fitter = fitter.fit(X.reshape((-1, 1)))
        else:
            raise ValueError(f"Invalid data shape {X.shape}")
        tsfm = DistTransformer(fitter, METHOD_NAMES[map_method])
    elif map_method == 4:
        ### Min max
        if "min" in tsfm_kwargs and "max" in tsfm_kwargs:
            _min = tsfm_kwargs["min"]
            _max = tsfm_kwargs["max"]
        else:
            _min = np.min(X, axis=0) + 1e-6
            _max = np.max(X, axis=0) + 1e-6
        tsfm = MinMaxTransformer(_min, _max)
    elif map_method == 5:
        ### Standardization
        if "avg" in tsfm_kwargs and "std" in tsfm_kwargs:
            avg = tsfm_kwargs["avg"]
            std = tsfm_kwargs["std"]
        else:
            avg = np.average(X, axis=0)
            std = np.std(X, axis=0)
        tsfm = StdTransformer(avg, std)
    elif map_method == 6:
        ### Log
        tsfm = LogTransformer()
    elif map_method == 7:
        ### non transformation
        tsfm = Transformer("None")
    else:
        raise ValueError(f"Invalid Transformation method {map_method}")

    return tsfm


class TransformerHub:
    def __init__(self):
        self.target2dist_tsfm: Dict[str, List[Transformer]] = {}
        self.target2id: Dict[str, int] = {}

    def register_fit(self, target, dist_tsfm: Transformer):
        if target not in self.target2dist_tsfm:
            self.target2dist_tsfm[target] = []
        cur_id = len(self.target2dist_tsfm[target])
        self.target2dist_tsfm[target].append(dist_tsfm)
        self.target2id[target] = cur_id

    def check_y_norm_method(self, expected_method):
        expected_method_name = METHOD_NAMES[get_method_id(expected_method)]
        if "y" not in self.target2dist_tsfm:
            return False
        cur_id = self.target2id["y"]
        used_method_name = self.target2dist_tsfm["y"][cur_id].method_name
        return expected_method_name == used_method_name
        # (f"Incorrect Y-norm method: expected {expected_method_name} VS {used_method_name}")

    def parse_y_norm_method(self, method, Y, tsfm_kwargs={}):
        map_method = get_method_id(method)
        self.register_fit("y", create_dist_transformer(map_method, Y, tsfm_kwargs))
        print(f"[Normalization] Register {METHOD_NAMES[map_method]}"
            f" transforms to Y of size {len(Y) if Y is not None else 0}")

    def parse_x_norm_method(self, method, X, tsfm_kwargs={}):
        map_method = get_method_id(method)
        self.register_fit("x", create_dist_transformer(map_method, X, tsfm_kwargs))
        print(f"[Normalization] Register {METHOD_NAMES[map_method]}"
            f" transforms to X of size {len(X) if X is not None else 0}")

    def get_distribution_tsfm(self, target):
        if target.startswith("x"):
            if target == "x" and target in self.target2dist_tsfm:
                cur_id = self.target2id[target]
                return self.target2dist_tsfm[target][cur_id]
            elif target in self.target2dist_tsfm:
                ### E.g. x0
                cur_id = self.target2id[target]
                return self.target2dist_tsfm[target][cur_id]
            else:
                ### Fail to find x0's transformation method, use default
                cur_id = self.target2id[target]
                return self.target2dist_tsfm["x"][cur_id]
        elif target == "y" and target in self.target2dist_tsfm:
            cur_id = self.target2id[target]
            return self.target2dist_tsfm[target][cur_id]
        raise ValueError(f"Fail to find transformation method for {target} "
            f"from {list(self.target2dist_tsfm.keys())}")

    def print(self):
        print(f"[Metainfo Transformation Hub]")
        for _key, _value in self.target2dist_tsfm.items():
            cur_id = self.target2id[_key]
            print(f"\t- For {_key}, use {str(_value[cur_id])}")

    def save(self, cache_root):
        for target, dist_tsfm_list in self.target2dist_tsfm.items():
            _cache_dir = os.path.join(cache_root, target)
            os.makedirs(_cache_dir, exist_ok=True)
            for dist_tsfm in dist_tsfm_list:
                dist_tsfm.save(_cache_dir)
        target2method_name = {}
        for target, cur_id in self.target2id.items():
            target2method_name[target] = self.target2dist_tsfm[target][cur_id].method_name
        with open(os.path.join(cache_root, "target2method_name.json"), 'w') as fp:
            json.dump(target2method_name, fp, indent=4)

    def load(self, cache_root, target2method={}):
        target2method_name_path = os.path.join(cache_root, "target2method_name.json")
        target2method_name = {}
        if os.path.exists(target2method_name_path):
            with open(target2method_name_path, 'r') as fp:
                target2method_name = json.load(fp)
        dirs = os.listdir(cache_root)
        for _dir in dirs:
            if _dir.endswith(".json"):
                continue
            target = _dir
            self.target2dist_tsfm[target] = []
            _cache_dir = os.path.join(cache_root, _dir)
            files = os.listdir(_cache_dir)
            for _file in files:
                method_id = get_method_id(_file.split(".pkl")[0])
                fitter = METHODS[method_id].load_init(_cache_dir, METHOD_NAMES[method_id])
                if target in target2method_name and \
                    target2method_name[target] == fitter.method_name:
                    self.target2id[target] = len(self.target2dist_tsfm)
                self.target2dist_tsfm[target].append(fitter)

        for target in self.target2dist_tsfm:
            if target in target2method:
                required_method_id = get_method_id(target2method[target])
                if METHOD_NAMES[required_method_id] in NO_ARG_METHOD_NAMES:
                    ### Methods with no arguments are not cached, directly construct the fitter
                    fitter = METHODS[required_method_id].load_init(_cache_dir, METHOD_NAMES[required_method_id])
                    self.target2id[target] = len(self.target2dist_tsfm[target])
                    self.target2dist_tsfm[target].append(fitter)
                else:
                    id_in_list = -1
                    for cur_id, fitter in enumerate(self.target2dist_tsfm[target]):
                        if fitter.method_name == METHOD_NAMES[required_method_id]:
                            id_in_list = cur_id
                            break
                    if id_in_list >= 0:
                        self.target2id[target] = id_in_list
                    else:
                        warning(f"Fail to specify transformation method for {target}:{target2method[target]}")
                        if target not in self.target2id:
                            self.target2id[target] = 0
            elif target in self.target2id:
                continue
            else:
                self.target2id[target] = 0
            
    def __getitem__(self, target):
        if target in self.target2dist_tsfm:
            cur_id = self.target2id[target]
            return self.target2dist_tsfm[target][cur_id]
        else:
            return None

    @property
    def has_y_tsfm(self):
        return "y" in self.target2dist_tsfm
            

    