import numpy as np
import math
import os
import re
from typing import Union, List
import pickle
from tqdm import tqdm

from tvm_helper.tir_helper import (
    FEATURE_PER_TIR_OP,
    ANNOTATION_DIM_LEN,
    feature_shape2d,
)

from utils.util import (notify, warning, random_select, 
    warn_once, read_yes, STD2AVG_UPPER_BOUND, Device2Task, Sample)
import utils.env as cdpp_env
from tvm_helper.metadata import DataMetaInfo, ASTMetaInfo
from tvm_helper.tir_helper import feature_lable_len, feature_shape2d

import metalearner.feature.feature as fea_info
from metalearner.feature import is_feature_type, ALL_FEATURE_TYPE

from .map2norm import TransformerHub, METHOD_NAMES

MIN_PER_TASK_SAMPLE_NUM = 48
TRAIN2ALL_RATIO = 0.8

def local_log(_str, level="info"):
    _str = f"[Data Preprocess] {_str}"
    if level == "info":
        print(_str)
    elif level == "notify":
        notify(_str)
    elif level.startswith("warn"):
        warn_once(_str)

def batch2x(batch, data_meta_info):
    return batch[:, data_meta_info.feature_start_idx:data_meta_info.feature_end_idx]

def parse_di(raw_data, metainfo, method="task_id"):
    if method == "task_id":
        if fea_info.FEATURE_INFO.task_id_idx:
            return raw_data[:, fea_info.FEATURE_INFO.task_id_idx].reshape(raw_data.shape[0], 1)
        else:
            return np.ones((raw_data.shape[0], 1))
    elif method == "di":
        x = batch2x(raw_data, metainfo)
        if not is_feature_type(ALL_FEATURE_TYPE.cus):
            return x
        _x = x.reshape(raw_data.shape[0], *feature_shape2d())
        return (np.average(_x, axis=2) > 1).astype(float)
    else:
        raise ValueError("di")

def random_split_data(all_data, ratio, index=False):
    ''' Split the input data to training data and validation data
    '''
    train_data, val_data = random_select(all_data, len(all_data)*ratio, index=index)
    if index:
        selected_indexs = train_data
        return selected_indexs, None
    else:
        return train_data, val_data if val_data.shape[0] > 0 else train_data

def trick_split_data_based_on_y(all_data, ratio, index=False):
    ''' Idea: sort y's first, suppose we have sorted y's, Y
        And required left to right ratio is (k-1):1, then we have
        Training data: Y[0:k-1] (k-1 is not included), Y[k:2k-1], Y[2k:3k-1]
        Test data: Y[k-1], Y[2k-1], ...
    '''
    sorted_idxs = np.argsort(all_data[:, 0])
    l2r_ratio = round(ratio / (1 - ratio))
    k = l2r_ratio + 1
    l_idxs = sorted_idxs[0:k-1]
    if k - 1 > len(sorted_idxs):
        r_idxs = l_idxs
    else:
        r_idxs = np.array([sorted_idxs[k-1]])
        anchor = k
        while anchor + k - 1 < len(sorted_idxs):
            l_idxs = np.concatenate((l_idxs, sorted_idxs[anchor:anchor+k-1]))
            r_idxs = np.concatenate((r_idxs, np.array([sorted_idxs[anchor+k-1]])))
            anchor += k
    if index:
        return l_idxs, r_idxs
    else:
        return all_data[l_idxs], all_data[r_idxs]

def remove_small_data(all_data, time_lb=None, verbose=True, index=False):
    if index:
        tmp = np.ones(all_data.shape[0], dtype=int)
        for idx, feature in enumerate(all_data):
            if feature[0] < time_lb:
                tmp[idx] = 0
        return tmp
    else:
        len_before = len(all_data)
        tmp = []
        for feature in all_data:
            if feature[0] >= time_lb:
                tmp.append(feature)
        all_data = np.array(tmp)
        if verbose:
            local_log(f"The avg>={time_lb}s filter takes effect: {len_before} ==> {len(all_data)}")
        return all_data

def remove_unstable_data(all_data, verbose=True, index=False):
    if index:
        tmp = np.ones(all_data.shape[0], dtype=int)
        for idx, feature in enumerate(all_data):
            if feature[1] / feature[0] > STD2AVG_UPPER_BOUND:
                tmp[idx] = 0
        return tmp
    else:
        len_before = len(all_data)
        tmp = []
        for feature in all_data:
            if feature[1] / feature[0] <= STD2AVG_UPPER_BOUND:
                tmp.append(feature)
        all_data = np.array(tmp)
        if verbose:
            local_log(f"The std/avg<{STD2AVG_UPPER_BOUND} filter takes effect: {len_before} ==> {len(all_data)}")
        return all_data

def group_downsample_data(all_data, verbose=True):
    grp_num = 10
    _max, _min = np.max(all_data[:, 0]), np.min(all_data[:, 0])
    if verbose:
        local_log(f"max avg {_max}, min avg {_min}, split to {grp_num} even groups")
    grp_span = (_max - _min) / grp_num
    grp_dict = {}
    for feature in all_data:
        _key = math.floor((feature[0] - _min) / grp_span)
        if _key not in grp_dict:
            grp_dict[_key] = []
        grp_dict[_key].append(feature)
    least_cnt = None
    least_bound = 100
    for _key in sorted(grp_dict.keys()):
        features = grp_dict[_key]
        if least_cnt is None or (len(features) > least_bound and len(features) < least_cnt):
            least_cnt = len(features)
        if verbose:
            local_log(f"{_key} has {len(features)} samples")
    if verbose:
        local_log(f"Downsample to {least_cnt} samples")
    tmp = None
    for _key in sorted(grp_dict.keys()):
        features = np.array(grp_dict[_key])
        if features.shape[0] < least_cnt:
            continue
        rst, _ = random_select(features, least_cnt)
        tmp = rst if tmp is None else np.concatenate((tmp, rst), axis=0)
    all_data = tmp
    return all_data

def register_log_feature_dim(all_data, data_meta_info, verbose=True):
    if len(all_data) == 0:
        return
    # log_trans_fea_per_tir_op = ["loop_steps", "para_steps", "unroll_steps"]
    log_trans_fea_per_tir_op = ["loop_steps"]
    log_trans_dim_per_tir_op = [FEATURE_PER_TIR_OP.index(fea)
        for fea in log_trans_fea_per_tir_op]
    log_trans_dim_per_tir_op += [len(FEATURE_PER_TIR_OP) + idx for idx in range(ANNOTATION_DIM_LEN)]
    feature_start_idx = data_meta_info.feature_start_idx

    log_trans_dim_idx = []
    idx = feature_start_idx
    while idx < all_data.shape[1]:
        for dim in log_trans_dim_per_tir_op:
            if idx+dim >= all_data.shape[1]:
                break
            log_trans_dim_idx.append(idx+dim-feature_start_idx)
        idx += len(FEATURE_PER_TIR_OP) + ANNOTATION_DIM_LEN
    data_meta_info.register_log_tran_dims(log_trans_dim_idx)
    if verbose:
        local_log("Register dimensions that will be applied with log-transformation: {}".format(
            str(log_trans_dim_idx)))

def remove_outliers(all_data, dim, verbose=True, data_meta_info=None, index=False):
    assert dim == 0
    # percentile_bound = (1, 99)
    percentile_bound = (10, 90)
    # percentile_bound = (25, 75)
    if data_meta_info and data_meta_info.percentile_dict is not None:
        lower_bound = data_meta_info.percentile_dict[percentile_bound[0]]
        upper_bound = data_meta_info.percentile_dict[percentile_bound[1]]
    else:
        local_log("Calculate percentiles using local data, may be not correct if data is loaded multiple times", level="warn")
        lower_bound = np.percentile(all_data[:, dim], percentile_bound[0])
        upper_bound = np.percentile(all_data[:, dim], percentile_bound[1])

    if index:
        tmp = np.ones(all_data.shape[0], dtype=int)
        for idx, feature in enumerate(all_data):
            if not (feature[dim] >= lower_bound and feature[dim] < upper_bound):
                tmp[idx] = 0
        return tmp
    else:
        len_before = len(all_data)
        tmp = []
        for feature in all_data:
            if feature[dim] >= lower_bound and feature[dim] < upper_bound:
                tmp.append(feature)
        if verbose:
            local_log(f"Remove outliers that are not in the range from {percentile_bound[0]}th to {percentile_bound[1]}th: {len_before} ==> {len(tmp)}")
        return np.array(tmp)

def sample_data_around_avg(all_data, data_meta_info, ratios=0.1, check="max"):
    def _internal_sample(_all_data, ratio):
        _input = _all_data[:, data_meta_info.feature_start_idx:data_meta_info.feature_end_idx]
        _input_avg = np.average(_input, axis=0)
        _input_std = np.std(_input, axis=0)
        # print(_input_avg)
        # print(_input_std)
        # elementwise_rst = (np.abs(_input - _input_avg) / (_input_avg))
        elementwise_rst = np.abs(_input - _input_avg) / (_input_std + 1e-6)
        if check == "max":
            rst_idxs = np.all((elementwise_rst <= ratio), axis=1)
        elif check == "l1norm":
            rst_idxs = (np.average(elementwise_rst, axis=1) <= ratio)
        else:
            raise
        return rst_idxs, elementwise_rst

    if isinstance(ratios, list):
        ratios = sorted(ratios, reverse=True)
        ret_data, ret_elementwise_rst, classifier = None, None, None
        cur_data = all_data
        for ratio in ratios:
            rst_idxs, elementwise_rst = _internal_sample(cur_data, ratio)
            print(f"ratio={ratio}: rst={sum(rst_idxs)}")
            if ret_data is None:
                ret_data, ret_elementwise_rst = all_data[rst_idxs], elementwise_rst[rst_idxs]
                classifier = np.ones(ret_data.shape[0]) * ratio
                cur_data = ret_data
            else:
                classifier[rst_idxs] = ratio
    else:
        classifier = None
        rst_idxs, elementwise_rst = _internal_sample(all_data, ratios)
        ret_data, ret_elementwise_rst = all_data[rst_idxs], elementwise_rst[rst_idxs]
    
    return ret_data, ret_elementwise_rst, classifier

def arg_filter(raw_data, metainfo, time_lb=None, verbose=True):
    ''' Calculate which data samples would be filtered out from `raw_data`

        NOTE: the first dimension of `raw_data` must be equalt to sample_num 
    '''
    indexs = np.ones(raw_data.shape[0], dtype=bool)

    ### Filter 1: using avg
    assert time_lb is not None
    if cdpp_env.PROJECT_CFG["FILTERS"][0] and time_lb is not None:
        _index = remove_small_data(raw_data, time_lb, verbose, index=True)
        indexs = np.logical_and(indexs, _index)

    ### Filter 2: using std
    if cdpp_env.PROJECT_CFG["FILTERS"][1] and metainfo.has_std():
        _index = remove_unstable_data(raw_data, verbose, index=True)
        indexs = np.logical_and(indexs, _index)

    ### Filter 3: remove outliers
    if cdpp_env.PROJECT_CFG["FILTERS"][2] and not is_feature_type(ALL_FEATURE_TYPE.debug) and len(raw_data) > 0:
        _index = remove_outliers(raw_data, 0, verbose, metainfo, index=True)
        indexs = np.logical_and(indexs, _index)

    return indexs

def filter_samples(raw_data, metainfo, time_lb=None, verbose=True):
    ''' Filter out some samples from `raw_data`

        NOTE: the first dimension of `raw_data` must be equalt to sample_num 
    '''
    ### Filter 1: using avg
    assert time_lb is not None
    if cdpp_env.PROJECT_CFG["FILTERS"][0] and time_lb is not None:
        raw_data = remove_small_data(raw_data, time_lb, verbose)

    ### Filter 2: using std
    if cdpp_env.PROJECT_CFG["FILTERS"][1] and metainfo.has_std():
        raw_data = remove_unstable_data(raw_data, verbose)

    ### Filter 3: remove outliers
    if cdpp_env.PROJECT_CFG["FILTERS"][2] and not is_feature_type(ALL_FEATURE_TYPE.debug) and len(raw_data) > 0:
        raw_data = remove_outliers(raw_data, 0, verbose, metainfo)
    
    return raw_data


class RawData:
    ''' 
    1. self.preprocess(...)
    2. (optional) self.shuffle(...) 
    2. self.freeze(...)
    3. self.normalize(...)
    '''
    def __init__(self, raw_data, metainfo,
            freezed=False,
            normalized=False,
            log_registerd=False):
        self.raw_data = raw_data
        self.metainfo = metainfo
        self.x: Union[None, np.ndarray, List] = None
        self.y: Union[None, np.ndarray] = None
        self.di: Union[None, np.ndarray] = None
        self._size = 0 if self.raw_data is None else len(self.raw_data)

        self.freezed = freezed
        self.normalized = normalized
        self.log_registerd = log_registerd

    @property
    def size(self):
        return self._size
    
    def __len__(self):
        return self._size

    def concatenate(self, target: np.ndarray):
        if self.raw_data is None:
            self.raw_data = target
        else:
            self.raw_data = np.concatenate(
                (self.raw_data, target), axis=0)

    def combine(self, others, inplace=False, verbose=True):
        assert self.log_registerd == others.log_registerd
        if inplace:
            if self.freezed:
                self.unfreeze()
            shape_before = self.raw_data.shape
            self.raw_data = np.concatenate((self.raw_data, others.raw_data), axis=0)
            if verbose:
                print(f"Merge data a {shape_before} and data b {others.raw_data.shape} --> {self.raw_data.shape}")
            self._size = 0 if self.raw_data is None else len(self.raw_data)
            return None
        else:
            new_raw_data = np.concatenate((self.raw_data, others.raw_data), axis=0)
            if verbose:
                print(f"Merge data a {self.raw_data.shape} and data b {others.raw_data.shape} --> {new_raw_data.shape}")
            return RawData(new_raw_data, self.metainfo, normalized=False, log_registerd=self.log_registerd)

    def shuffle(self):
        ### Assert shuffle is used before freeze the dataset
        ### And return the permutation of indexes
        assert not self.freezed
        p = np.random.permutation(len(self.raw_data))
        self.raw_data = self.raw_data[p]
        return p

    def freeze(self):
        ### Freeze the length of the dataset
        assert not self.normalized
        self.freezed = True
        self.x = batch2x(self.raw_data, self.metainfo)
        self.y = self.raw_data[:, 0]
        self.di = parse_di(self.raw_data, self.metainfo)
        return None
    
    def unfreeze(self):
        ### Freeze the length of the dataset
        assert self.freezed
        self.x = None
        self.y = None
        self.di = None
        self.freezed = False
        self.normalized = False
        self._size = 0 if self.raw_data is None else len(self.raw_data)

    def __iter__(self):
        assert self.freezed and self.normalized
        for idx in range(self.size):
            yield self.x[idx], self.y[idx], self.di[idx]
    
    def __getitem__(self, index):
        return RawData(self.raw_data[index], self.metainfo, log_registerd=self.log_registerd)

    def subset(self, cluster_ids: np.ndarray):
        return RawData(self.raw_data[cluster_ids, :], self.metainfo, log_registerd=self.log_registerd)
    
    def gen_train_test_data(self):
        # assert not self.freezed
        # train_raw_data, val_raw_data = random_split_data(self.raw_data, train2all_ratio)
        train_raw_data, val_raw_data = trick_split_data_based_on_y(self.raw_data, TRAIN2ALL_RATIO)
        train_raw_data = RawData(train_raw_data, self.metainfo, log_registerd=self.log_registerd)
        val_raw_data = RawData(val_raw_data, self.metainfo, log_registerd=self.log_registerd)
        return train_raw_data, val_raw_data
    
    def arg_train_test_part(self):
        # partition_indexes, _ = random_split_data(self.raw_data, TRAIN2ALL_RATIO, index=True)
        partition_indexes, _ = trick_split_data_based_on_y(
            self.raw_data, TRAIN2ALL_RATIO, index=True)
        return partition_indexes

    def slice_freezed(self, idx):
        return self.x[idx], self.y[idx], self.di[idx]

    def normalize(self):
        assert self.freezed and not self.normalized
        self.normalized = True
        if self.size == 0:
            return
        ### Normalize and standarize data
        self.x = self.metainfo.norm_input(self.x)
        self.y = self.metainfo.standardize_output(self.y)
        # self.di = self.metainfo.standardize_di(self.di)

    def preprocess(self, time_lb=None, verbose=True):
        assert not self.freezed

        self.raw_data = filter_samples(self.raw_data, self.metainfo, 
            time_lb=time_lb, verbose=verbose)

        ### Grouping and downsampling
        if False:
            self.raw_data = group_downsample_data(self.raw_data, verbose)
            # print(self.raw_data.shape)
        
        if not self.log_registerd and is_feature_type(ALL_FEATURE_TYPE.cus):
            register_log_feature_dim(self.raw_data, self.metainfo, verbose)
            self.log_registerd = True

        # check_exit("do you want to use force sampling")
        # self.raw_data = sample_data_around_avg(self.raw_data, self.metainfo, ratio=2)
        # print(len(self.raw_data))

    def arg_filter(self, time_lb=None, verbose=True):
        assert not self.freezed
        return arg_filter(self.raw_data, self.metainfo, time_lb=time_lb, verbose=verbose)

    @staticmethod
    def load_init(path, metainfo=None):
        with open(path, 'rb') as fp:
            raw_data, normalized, log_registerd = pickle.load(fp)
        if metainfo is None:
            metainfo_path = f"{path}.metainfo"
            metainfo = DataMetaInfo.load_init(metainfo_path)

        ret = RawData(raw_data, metainfo, normalized=normalized, log_registerd=log_registerd)
        # ret.freeze()
        return ret

    def save(self, path, save_metainfo=True):
        with open(path, 'wb') as fp:
            pickle.dump([self.raw_data, self.normalized, self.log_registerd], fp)
        if save_metainfo:
            metainfo_path = f"{path}.metainfo"
            self.metainfo.save(metainfo_path)
    
    def afterward_filter(self):
        if not read_yes("Filter samples after normalize"):
            return
        assert self.freezed and self.normalized
        print(f"Shape before: x:{self.x.shape}, y: {self.y.shape}")

        ### group based on arithmetic intensity
        idxs = (self.x[:, 147:157].mean(axis=1) > 0.5)
        self.x = self.x[idxs]
        self.y = self.y[idxs]
        self.di = self.di[idxs]
        self.raw_data = self.raw_data[idxs]
        
        print(f"Shape after: x:{self.x.shape}, y: {self.y.shape}")
    
    @property
    def flops(self):
        raise NotImplementedError()


def ast_positional_encoding(ast_features, node_ids, serialized_tree):
    ### Customized positional encodings for AST
    # refer to https://github.com/pytorch/pytorch/issues/24826
    seq_len, n_entry = ast_features.shape
    pe = np.zeros((seq_len, n_entry))
    for pos in range(seq_len):
        for i in range(0, n_entry, 2):
            ast_pos = list(serialized_tree).index(node_ids[pos])
            pe[pos, i] = \
                math.sin(ast_pos / (10000 ** ((2 * i) / n_entry)))
            pe[pos, i + 1] = \
                math.cos(ast_pos / (10000 ** ((2 * (i + 1)) / n_entry)))
    return pe


class ASTRawData(RawData):
    def __init__(self, raw_data, metainfo, freezed=False, 
            normalized=False, log_registerd=False, disable_filter=False):
        super(ASTRawData, self).__init__(raw_data, metainfo,
            freezed=freezed, normalized=normalized, log_registerd=log_registerd)
        
        self.pe = None

        if disable_filter:
            self.flop_bound = None
            self.fix_seq_len = None
            self.max_seq_len = None
        else:
            self.max_seq_len = cdpp_env.PROJECT_CFG["MAX_SEQ_LEN"] # including seq end id
            self.fix_seq_len = cdpp_env.PROJECT_CFG.get("FIX_SEQ_LEN", None)
            self.flop_bound = cdpp_env.PROJECT_CFG.get("FLOP_BOUND", None)

        self.use_pe = cdpp_env.PROJECT_CFG.get("USE_PE", True)

    @staticmethod
    def leaf_no(sample):
        avg, std, flops, ast_features, node_ids, serialized_tree = sample
        # print(ast_features.shape) # (N_seq, N_entry)
        leaf_no = ast_features.shape[0]
        return leaf_no

    def freeze(self):
        assert not self.normalized

        self.x = []
        self.y = []
        selected_idx = []

        if not self.use_pe:
            assert self.fix_seq_len
            for idx, _data in enumerate(self.raw_data):
                avg, std, flops, ast_features, node_ids, serialized_tree = _data
                # print(ast_features.shape) # (N_seq, N_entry)
                if ast_features is None:
                    import pdb; pdb.set_trace()
                leaf_no = ast_features.shape[0]

                if leaf_no not in self.fix_seq_len:
                    continue
                    
                if self.flop_bound:
                    if flops == -1:
                        continue
                    lower, upper = self.flop_bound
                    if lower and flops < lower:
                        continue
                    elif upper and flops >= upper:
                        continue

                self.x.append(ast_features)
                self.y.append(avg)
                selected_idx.append(idx)

            ### self.X's shape = (B, N_seq, N_entry)
            self.x = np.array(self.x)
            # self.pe = np.array(self.pe)
        else:
            self.pe = []
            self.end_token = np.ones([1, self.metainfo.feature_len], dtype=float)

            for idx, _data in enumerate(self.raw_data):
                avg, std, flops, ast_features, node_ids, serialized_tree = _data
                # print(ast_features.shape) # (N_seq, N_entry)
                if ast_features is None:
                    import pdb; pdb.set_trace()
                leaf_no = ast_features.shape[0]

                if self.fix_seq_len and leaf_no not in self.fix_seq_len:
                    continue
                    
                if self.flop_bound:
                    if flops == -1:
                        continue
                    lower, upper = self.flop_bound
                    if lower and flops < lower:
                        continue
                    elif upper and flops >= upper:
                        continue

                ### Positional encoding
                pe = ast_positional_encoding(ast_features, node_ids, serialized_tree)

                if self.fix_seq_len is None and self.max_seq_len:
                    ### Only take effect when FIX_SEQ_LEN is None and MAX_SEQ_LEN is given
                    assert self.max_seq_len > (leaf_no + 1), (self.max_seq_len, ast_features.shape)
                    padding = np.zeros([self.max_seq_len - leaf_no - 1, self.metainfo.feature_len], dtype=float)
                    ast_features = np.concatenate((ast_features, self.end_token, padding), axis=0)
                    pe = np.concatenate((pe, self.end_token, padding), axis=0)

                self.x.append(ast_features)
                self.pe.append(pe)
                self.y.append(avg)
                selected_idx.append(idx)

            ### self.X's shape = (B, N_seq, N_entry)
            if self.max_seq_len or self.fix_seq_len:
                self.x = np.array(self.x)
                self.pe = np.array(self.pe)
            else:
                self.x = np.array(self.x, dtype=object)
                self.pe = np.array(self.pe, dtype=object)

        self.y = np.array(self.y)
        self._size = len(self.y)
        self.freezed = True
        return selected_idx
    
    def __iter__(self):
        assert self.freezed and self.normalized
        for idx in range(self.size):
            yield self.x[idx].astype(float), self.y[idx].astype(float), 0
    
    def slice_freezed(self, idx):
        return self.x[idx].astype(float), self.y[idx].astype(float), 0

    def normalize(self):
        assert self.freezed and not self.normalized
        self.normalized = True
        if self.size == 0:
            return
        ### Normalize and standarize data
        self.x = np.array([self.metainfo.norm_input(_x) for _x in self.x], dtype=object)
        self.y = self.metainfo.standardize_output(self.y)

        ### Apply positional encodings
        if self.use_pe:
            self.x += self.pe
    
    def afterward_filter(self):
        raise NotImplementedError("Does NOT support afterward filter for AST rawdata")
    
    def __getitem__(self, index):
        return ASTRawData(self.raw_data[index], self.metainfo, log_registerd=self.log_registerd)

    def subset(self, cluster_ids: np.ndarray):
        return ASTRawData(self.raw_data[cluster_ids, :], self.metainfo, log_registerd=self.log_registerd)
    
    def gen_train_test_data(self):
        # assert not self.freezed
        # train_raw_data, val_raw_data = random_split_data(self.raw_data, train2all_ratio)
        train_raw_data, val_raw_data = trick_split_data_based_on_y(self.raw_data, TRAIN2ALL_RATIO)
        train_raw_data = ASTRawData(train_raw_data, self.metainfo, log_registerd=self.log_registerd)
        val_raw_data = ASTRawData(val_raw_data, self.metainfo, log_registerd=self.log_registerd)
        return train_raw_data, val_raw_data
    
    def max_leaf_no(self):
        rst = 0
        for _data in self.raw_data:
            avg, std, flops, ast_features, node_ids, serialized_tree = _data
            # print(ast_features.shape) # (N_seq, N_entry)
            rst = max(rst, ast_features.shape[0])
        return rst
    
    def leaf_no_dist(self):
        leaf_nos = []
        for _data in self.raw_data:
            avg, std, flops, ast_features, node_ids, serialized_tree = _data
            # print(ast_features.shape) # (N_seq, N_entry)
            leaf_nos.append(ast_features.shape[0])
        return leaf_nos
    
    @property
    def flops(self):
        if self.size == 0:
            return []
        # avg, std, flops, ast_features, node_ids, serialized_tree = _data
        return list(self.raw_data[:, 2])

def extract_task_id(file_name):
    match = re.search(r"_(?P<task_id>[\d]+).npy", file_name)
    if match:
        return int(match["task_id"])
    else:
        global ALL_TASK
        ### E.g. 't4_resnet50_cuda_bs-4_trial-num-500_task-0.npy' --> 
        split = file_name.split("_")
        return hash((split[1], split[4], split[-1].split(".")[0]))

def _load_debug_data():
    def func(a, b):
        return 3.2 * a + 2.4 * b + 12
    xydata = []
    def funcc(a, b):
        xydata.append([func(a, b), a, b])
    for a in range(30):
        for b in range(30):
            funcc(a, b)
    xydata = np.array(xydata)
    data_len = xydata.shape[1]
    return xydata, data_len

def _load_cus_data(files, learning_params, verbose=True):
    xydata = None

    data_len = feature_lable_len(fea_info.FEATURE_INFO._pre_add_dims) 
    for file in files:
        if not file.endswith(".npy"):
            continue
        if learning_params["feature_type"] is not None and learning_params["feature_type"] not in file:
            continue
        data = np.load(file)

        # if read_yes("Remove redundant features?"):
        #     element_wise_std = np.std(data, axis=0)
        #     print(element_wise_std != 0)
        #     data = data[:, element_wise_std != 0]
        # test_idx = [7, 2858, 3063]
        # xydata[test_idx, :]

        task_id = extract_task_id(file)
        data = np.insert(data, fea_info.FEATURE_INFO.task_id_idx, values=float(task_id), axis=1)

        ### check the data shape
        if learning_params["feature_type"] is not None and learning_params["feature_type"] == "cus-feature" \
            and data.shape[1] != data_len:
            # print(file, data.shape, data_len)
            continue
        if verbose:
            print(f"Read np.array of shape {data.shape} from {file}")
        if xydata is None:
            xydata = data
        else:
            xydata = np.concatenate((xydata, data), axis=0)
    
    if xydata is None:
        raise ValueError(f"No data from {files}")

    return xydata, data_len

def _load_ansor_data(files, verbose=True):
    xydata = None

    for file in files:
        if not file.endswith(".npy"):
            continue
        # if learning_params["feature_type"] is not None and learning_params["feature_type"] not in file:
        #     continue
        data = np.load(file, allow_pickle=True)
        data = data.astype("float")
        
        if verbose:
            print(f"Read np.array of shape {data.shape} from {file}")
        if xydata is None:
            xydata = data
        else:
            xydata = np.concatenate((xydata, data), axis=0)
    
    if xydata is None:
        raise ValueError(f"No data from {files}")
    
    return xydata, xydata.shape[1] - len(fea_info.FEATURE_INFO._pre_add_dims)

def _load_ast_ansor_data(files, verbose=True):
    ''' Load AST+Ansor features

    Return
    ------
    xydata: np.ndarray
        * Shape = (N_schedules, 6), the 6 column denotes avg, std, flops, ast_features, node_ids, serialized_tree
        * `ast_features` is an array of shape (|node_ids|, N_entry), N_entry=164 by default
        * `node_ids` is the ids of leaf nodes in the `serialized_tree`
    '''
    xydata = None

    for file in files:
        if not file.endswith(".npy"):
            continue
        data = np.load(file, allow_pickle=True)
        # data = data.astype("float")
        
        if verbose:
            print(f"Read np.array of shape {data.shape} from {file}")
        if xydata is None:
            xydata = data
        else:
            xydata = np.concatenate((xydata, data), axis=0)
    
    if xydata is None:
        raise ValueError(f"No data from {files}")
    
    return xydata, xydata[0][3].shape[1]

def load_raw_data(
        files,
        learning_params,
        force=False,
        verbose=True,
        # metainfo_path=None
        ):
    ''' Load all TIR data that meet the current shape requirement according to 
        the # of TIR OP and # of FEATURE_PER_TIR_OP
    
    Parameters
    ----------
    files: list of str
        The list of absolute paths of the profiling data
    metainfo_path: str
        Path to cache metainfo. Do not parse metainfo if it is set None
    force: boolean
        If set True, parse the metainfo, no matter whether there is 
        metainfo cached before
    '''
    assert len(files) == len(set(files)), "Files contain repeated ones"
    files = sorted(files)
    if is_feature_type(ALL_FEATURE_TYPE.debug):
        ### Debug
        warning("Debug data is used")
        xydata, data_len = _load_debug_data()
    if is_feature_type(ALL_FEATURE_TYPE.ast_ansor):
        # notify("AST+Ansor features are used")
        xydata, data_len = _load_ast_ansor_data(files, verbose=verbose)
    elif is_feature_type(ALL_FEATURE_TYPE.ansor):
        # notify("Ansor features are used")
        xydata, data_len = _load_ansor_data(files, verbose=verbose)
    elif is_feature_type(ALL_FEATURE_TYPE.cus):
        raise NotImplementedError()
        learning_params = None
        xydata, data_len = _load_cus_data(files, learning_params, verbose=verbose)
    else:
        raise ValueError(fea_info.FEATURE_INFO.feature_type)

    ### Construct the DatametaInfo
    re_cal = True
    data_meta_info = None
    # if not force and metainfo_path is not None and os.path.exists(metainfo_path):
    #     data_meta_info = DataMetaInfo.load_init(metainfo_path)
    #     if data_meta_info.input_max.shape[0] != data_len:
    #         pass
    #     else:
    #         re_cal = False
    # elif force:
    if force:
        re_cal = True
    # elif metainfo_path is None:
    #     re_cal = False

    if re_cal:
        if is_feature_type(ALL_FEATURE_TYPE.ast_ansor):
            data_meta_info = ASTMetaInfo(xydata, fea_info.FEATURE_INFO._pre_add_dims)
        else:
            data_meta_info = DataMetaInfo(xydata, fea_info.FEATURE_INFO._pre_add_dims)
        # if metainfo_path is not None:
        #     data_meta_info.save(metainfo_path)
        if verbose:
            print(f"Calculate output avg={data_meta_info.output_avg}, "
                    f"std={data_meta_info.output_std}")

    if is_feature_type(ALL_FEATURE_TYPE.ast_ansor) \
            and not learning_params["tiramisu"]:
        return ASTRawData(xydata, data_meta_info)
    else:
        return RawData(xydata, data_meta_info)

def parse_metainfo(files, learning_params, save=True, verbose=True, use_default_path=False):
    ''' Read all data in `files` and analyze the metainfo
    '''
    MetaInfoCls = ASTMetaInfo if is_feature_type(ALL_FEATURE_TYPE.ast_ansor) else DataMetaInfo

    if isinstance(files, Device2Task):
        ### Multiple devices
        _files = []
        for device, device_files in files:
            _files += device_files
        files = _files

    metainfo_path = None
    if save:
        if use_default_path:
            metainfo_path = os.path.join(learning_params["input"], f"metainfo.pickle")
        else:
            _mode_list = learning_params['mode'].split(".")
            assert len(_mode_list) == 1 or len(_mode_list) == 2, \
                f"Invalid mode {learning_params['mode']}"
            ### If the data mode consists of both the source pattern 
            # and the split pattern, we only focus on the source pattern
            metainfo_path = os.path.join(learning_params["input"], 
                f"metainfo-{_mode_list[0]}.pickle")
          
    ### Load metainfo
    re_cal = True
    if metainfo_path and os.path.exists(metainfo_path):
        try:
            data_meta_info = MetaInfoCls.load_init(metainfo_path)
            if not (is_feature_type(ALL_FEATURE_TYPE.cus) and data_meta_info.input_max.shape[0] != feature_len()):
                re_cal = False
        except FileNotFoundError:
            re_cal = True
        except:
            raise
            re_cal = True
    ### If not cached, load all data and calculate metainfo
    if re_cal:
        notify(f"Load data to stat metainfo --> {metainfo_path} ... ")
        data_meta_info = None
        data_meta_dict = {
            "output": None,
            "input_max": None,
            "input_min": None
        }

        files = [f for f in files if f.endswith(".npy")]

        if verbose:
            print(f"Read np.array from {files[0]} to {files[-1]}")

        for _file in tqdm(sorted(files), total=len(files)):
            data = np.load(_file, allow_pickle=True)
            if not is_feature_type(ALL_FEATURE_TYPE.ast_ansor):
                data = data.astype("float")

            if is_feature_type(ALL_FEATURE_TYPE.cus):
                task_id = extract_task_id(_file)
                data = np.insert(data, fea_info.FEATURE_INFO.task_id_idx, values=float(task_id), axis=1)
            if data_meta_info is None:
                data_meta_info = MetaInfoCls(data, fea_info.FEATURE_INFO._pre_add_dims)
                feature = data_meta_info.feature(data)
                data_meta_dict["input_max"] = np.max(feature, axis=0)
                data_meta_dict["input_min"] = np.min(feature, axis=0)
                data_meta_dict["output"] = data_meta_info.output(data)
            else:
                feature = data_meta_info.feature(data)
                data_meta_dict["input_max"] = np.max(
                    np.stack((np.max(feature, axis=0), data_meta_dict["input_max"])), axis=0)
                data_meta_dict["input_min"] = np.min(
                    np.stack((np.min(feature, axis=0), data_meta_dict["input_min"])), axis=0)
                data_meta_dict["output"] = np.concatenate((
                    data_meta_dict["output"],
                    data_meta_info.output(data)))

            del data
        
        data_meta_info.tsfm_hub = TransformerHub()
        data_meta_info.input_max = data_meta_dict["input_max"]
        data_meta_info.input_min = data_meta_dict["input_min"]
        data_meta_info.output_avg = np.average(data_meta_dict["output"])
        data_meta_info.output_std = np.std(data_meta_dict["output"])

        ### register Distribution Transformers
        if metainfo_path is not None:
            for method_id in range(len(METHOD_NAMES)):
                data_meta_info.tsfm_hub.parse_y_norm_method(method_id, data_meta_dict["output"])
                data_meta_info.tsfm_hub.save(DataMetaInfo.metapath2tsfm_dir(metainfo_path))
        data_meta_info.tsfm_hub.parse_y_norm_method(data_meta_info.output_norm_method, data_meta_dict["output"])
        data_meta_info.tsfm_hub.parse_x_norm_method("min-max", None, {
            "min": data_meta_info.input_min, "max": data_meta_info.input_max})

        ### Calculate percentiles
        percentile_dict = {}
        for percentile_value in [1, 5, 10, 25, 50, 90, 75, 95, 99]:
            percentile_dict[percentile_value] = np.percentile(data_meta_dict["output"], percentile_value)
        data_meta_info.percentile_dict = percentile_dict

        if metainfo_path is not None:
            data_meta_info.save(metainfo_path)
    else:
        notify(f"Load cached metainfo at {metainfo_path} ... ")
    if metainfo_path is not None:
        assert os.path.exists(metainfo_path), metainfo_path

    return data_meta_info