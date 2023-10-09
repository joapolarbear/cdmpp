'''Generate an index file for large dataset

Preprocess dataset in an offline manner and cache the results
$ bash scripts/train.sh run --mode sample200 -i .workspace/ast_ansor -o make_dataset
$ bash scripts/train.sh run --mode sample200 -i .workspace/ast_ansor -o make_dataset --learf_node_no 6
$ bash scripts/train.sh run --mode sample200  -i .workspace/ast_ansor -o make_dataset --ave_lb 0
'''
import pickle
import os
import json
import numpy as np
import math
from typing import Union, List
from tqdm import tqdm
import re
import copy

from utils.util import (sample_task_files, notify, 
    SplitAttr, range2str, warning, Sample, read_yes, TrainTestPair, Device2Task)
from utils.env import PROJECT_CFG
from utils.device_info import get_device_feature, DEVICE_FEATURE_LEN
from metalearner.feature import (
    ALL_FEATURE_TYPE,
    init_fea_info_via_data,
    is_feature_type
)
from metalearner.data.rawdata import (
    load_raw_data,
    filter_samples,
    trick_split_data_based_on_y,
    TRAIN2ALL_RATIO,
    parse_metainfo,
    RawData,
    ASTRawData,
    MIN_PER_TASK_SAMPLE_NUM
)
from tvm_helper.metadata import DataMetaInfo, ASTMetaInfo

def task_short_name(task_file_name, device=""):
    return device+ "_" + os.path.basename(task_file_name).split(".")[0]

def remove_vague_samples(samples_list: List):
    ''' Remove vague samples from k Sample instances
    '''
    if not read_yes("Remove samples with the same x but different y ??? "):
        exit(0)

    sample_num_list = [s.size for s in samples_list]
    sample_id_to_rm = {}

    def compare_samples(k1, ii1, k2, ii2):
        if np.all(samples_list[k1].x_tir[ii1] == samples_list[k2].x_tir[ii2]) and \
            (2 * (samples_list[k1].y[ii1] - samples_list[k2].y[ii2]) / (
                samples_list[k1].y[ii1] + samples_list[k2].y[ii2]) >= 0.05):
                ### Sample x, and y_diff is too large, remove the latter sample
                if k2 not in sample_id_to_rm:
                    sample_id_to_rm[k2] = set()
                sample_id_to_rm[k2].add(ii2)
    
    for k1 in range(len(samples_list) - 1):
        ### Compare samples in the same Sample instance
        for ii1 in range(sample_num_list[k1] - 1):
            for ii2 in range(ii1, sample_num_list[k1]):
                compare_samples(k1, ii1, k1, ii2)
        ### Compare samples in Other Sample instances
        for k2 in range(k1, len(samples_list)):
            for ii1 in range(sample_num_list[k1]):
                for ii2 in range(sample_num_list[k2]):
                    compare_samples(k1, ii1, k2, ii2)
    
    for k, samples in enumerate(samples_list):
        if k in sample_id_to_rm:
            print(f"Remove index {len(sample_id_to_rm[k])} samples from the {k}th dataset")
            samples.clear_samples(sample_id_to_rm[k])

class DataSplit:
    def __init__(self, verbose=True):
        ''' A dict mapping from task file to indexes
            the index is a 2d array of shape (N_sample, 1)
            For each sample, we use the following integers to denote the split of one sample
             * 0: filter out by the Filter
             * 1: Training set
             * 2: Validation/Test set
        '''
        self.file2splits = {}
        self.trace_root_path: Union[None, str] = None
        self.ave_lb: Union[None, float] = None
        self.verbose = verbose
    
    def local_log(self, *args, **kwargs):
        if self.verbose:
            print(f"[Pre-process]", *args, **kwargs)
           
    def save(self):
        assert self.trace_root_path is not None
        file_name = os.path.join(self.trace_root_path, "DataSplit.pickle")
        with open(file_name, 'wb') as fp:
            pickle.dump([self.file2splits, self.trace_root_path, self.ave_lb], fp)
        self.local_log(f"Dump DataSplit to {file_name}, with filter>{self.ave_lb}")

    def load(self):
        assert self.trace_root_path is not None
        file_name = os.path.join(self.trace_root_path, "DataSplit.pickle")
        with open(file_name, 'rb') as fp:
            self.file2splits, self.trace_root_path, self.ave_lb = pickle.load(fp)
        self.local_log(f"Load DataSplit from {file_name}, with filter>{self.ave_lb}")
    
    @staticmethod
    def load_init(data_dir):
        data_index = DataSplit()
        file_name = os.path.join(data_dir, "DataSplit.pickle")
        with open(file_name, 'rb') as fp:
            data_index.file2index, data_index.trace_root_path,\
                data_index.ave_lb = pickle.load(fp)
        # print(f"Load DataSplit from {file_name}, with filter>{data_index.ave_lb}")
        return data_index
    
    def gen_train_test_data(self, task_file_name, device, raw_data):
        short_name = task_short_name(task_file_name, device)
        splits = self.file2splits[short_name]
        train_raw_data = raw_data.subset(np.where(splits==SplitAttr.train)[0])
        test_raw_data = raw_data.subset(np.where(splits==SplitAttr.test)[0])
        return train_raw_data, test_raw_data
    
    def verify_index(self, task_file_name, device, raw_data):
        ''' Verify the index file corresponds to the raw_data'''
        short_name = task_short_name(task_file_name, device)
        splits = self.file2splits[short_name]
        assert len(splits) == raw_data.size

class ShardPaths:
    def __init__(self, train_x_tir, train_x_device, train_y, 
        val_x_tir, val_x_device, val_y):
        self.train_x_tir = train_x_tir
        self.train_x_device = train_x_device
        self.train_y = train_y
        self.val_x_tir = val_x_tir
        self.val_x_device = val_x_device
        self.val_y = val_y
    
    def tolist(self):
        return [self.train_x_tir, self.train_y, self.val_x_tir, self.val_y, 
            self.train_x_device, self.val_x_device]
    
    def abs_paths(self, dataset_path):
        return ShardPaths.from_list(
            [os.path.join(dataset_path, "none" if _f is None else f"{_f}.npy" if not _f.endswith(".npy") else _f) for _f in self.tolist()])
    
    @staticmethod
    def from_list(_list):
        if len(_list) == 4:
            train_x_tir, train_y, val_x_tir, val_y = _list
            train_x_device = val_x_device = None
        elif len(_list) == 6:
            train_x_tir, train_y, val_x_tir, val_y, train_x_device, val_x_device = _list
        else:
            raise ValueError(f"Invalid list {_list}")
        return ShardPaths(train_x_tir, train_x_device, train_y,
            val_x_tir, val_x_device, val_y)

class DataPartitioner:
    def __init__(self, data_meta_info=None, verbose=True):

        ''' self.global_partition_rst is an array to store the final partition results of 
            samples from all tasks, where each value is one kind of SplitAttr, e.g.
             * 0: filter out by the Filter
             * 0b001: Training set
             * 0b010: Test set
            NOTE that we allow a sample appear in both the training set and test set, by
            assigning the value of 0b001 | 0b010 = 0b011

            self.files2global_idxs is a dict mapping from task file to the global indexes (the 
            indexs in self.global_partition_rst)
        '''
        self.files2global_idxs = {}
        self.global_partition_rst = None

        self.trace_root_path: Union[None, str] = None
        self.ave_lb: Union[None, float] = None
        self.verbose = verbose

        self.files_to_test = None
        self.data_meta_info = data_meta_info
        self.is_cross_device = False

        self.file_shard_size = 64
        self.min_sample_size = MIN_PER_TASK_SAMPLE_NUM

        self.filter_via_feature_entry = PROJECT_CFG["FILTERS_VIA_FEATURE_ENTRY"]

    def local_log(self, *args, **kwargs):
        if self.verbose:
            print(f"[Pre-process]", *args, **kwargs)

    @staticmethod
    def default_dir(learning_params: dict):
        ''' Return the directory to store all cached data with the same configuration.
            Different data loading mode (including data source and partition method) share the same directory
        '''
        ret = os.environ.get("CDPP_DATASET_PATH", os.path.join("tmp", "dataset"))
        suffix = ""
        if learning_params["ave_lb"] != 0.001:
            suffix += f"-ave_lb_{learning_params['ave_lb']}"
        if PROJECT_CFG["FILTERS"] != [1, 1, 1]:
            suffix += f'-filters{PROJECT_CFG["FILTERS"][0]}{PROJECT_CFG["FILTERS"][1]}{PROJECT_CFG["FILTERS"][2]}'
        
        if PROJECT_CFG["FILTERS_VIA_FEATURE_ENTRY"]:
            suffix += f'-filter_via_feature_entry'
        
        flop_bound = PROJECT_CFG.get("FLOP_BOUND", None)
        if flop_bound is not None:
            suffix += f"-{range2str(flop_bound)}"

        use_pe = PROJECT_CFG.get("USE_PE", True)
        if not use_pe:
            suffix += f"-not_use_pe"
            
        return ret + suffix.replace('.', '_')

    @staticmethod
    def tmp_dataset_dir(learning_params: dict, limit=200):
        ''' Return the data directory for a specific combination of configuration and loading mode
        '''
        fix_seq_len = PROJECT_CFG.get("FIX_SEQ_LEN", None)

        if len(learning_params["mode"]) > limit:
            _path = os.path.join(DataPartitioner.default_dir(learning_params), learning_params["mode"][:limit] + "...")
        else:
            _path = os.path.join(DataPartitioner.default_dir(learning_params), learning_params["mode"])
        if fix_seq_len is None:
            _path = _path + "-all"
        elif len(fix_seq_len) == 1 and fix_seq_len[0] == 5:
            pass
        else:
            _path = _path + "-leaf_node_no_" + "_".join([str(v) for v in sorted(fix_seq_len)])
        
        return _path
    
    @staticmethod
    def metadata_path(learning_params: dict):
        return os.path.join(DataPartitioner.tmp_dataset_dir(learning_params),"data_meta_info.pickle")

    @staticmethod
    def shard_meta_path(learning_params: dict):
        return os.path.join(DataPartitioner.tmp_dataset_dir(learning_params), "shard_meta.json")
    
    @staticmethod
    def train_val_path_bases(shard_id, _dir=None, 
            output_norm_method="std", input_norm_method="min-max"):
        x_suffix = ""
        if input_norm_method != "min-max" and str(input_norm_method) != "4":
            x_suffix += f"-input_norm_{input_norm_method}"
        y_suffix = ""
        if output_norm_method != "std":
            y_suffix += f"-output_norm_{output_norm_method}"

        if _dir is None:
            return ShardPaths(
                f"train_x_{shard_id}{x_suffix}.npy",
                f"train_x_device_{shard_id}.npy",
                f"train_y_{shard_id}{y_suffix}.npy",
                f"val_x_{shard_id}{x_suffix}.npy",
                f"val_x_device_{shard_id}.npy",
                f"val_y_{shard_id}{y_suffix}.npy"
            )
        else:
            return ShardPaths(
                os.path.join(_dir, f"train_x_{shard_id}{x_suffix}.npy"),
                os.path.join(_dir, f"train_x_device_{shard_id}.npy"),
                os.path.join(_dir, f"train_y_{shard_id}{y_suffix}.npy"),
                os.path.join(_dir, f"val_x_{shard_id}{x_suffix}.npy"),
                os.path.join(_dir, f"val_x_device_{shard_id}.npy"),
                os.path.join(_dir, f"val_y_{shard_id}{y_suffix}.npy")
            )
    
    @staticmethod
    def parse_shard_id(basename):
        match = re.search(r"(train|val|test)_(x|y)_(?P<shard_id>\d+)(-)*", basename)
        return int(match.groupdict()['shard_id'])
    
    @staticmethod
    def load_metainfo(learning_params: dict):
        if is_feature_type(ALL_FEATURE_TYPE.ast_ansor):
            return ASTMetaInfo.load_init(DataPartitioner.metadata_path(learning_params))
        else:
            return DataMetaInfo.load_init(DataPartitioner.metadata_path(learning_params))

    @staticmethod
    def load_shard(learning_params: dict):
        
        dataset_path = DataPartitioner.check_tmp_dataset_path(learning_params)

        with open(DataPartitioner.shard_meta_path(learning_params), 'r') as fp:
            shard_meta = json.load(fp)

        metainfo = DataPartitioner.load_metainfo(learning_params)
        metainfo.input_norm_method = learning_params.get("input_norm_method", PROJECT_CFG["INPUT_NORM_METHOD"])
        metainfo.output_norm_method = learning_params.get("output_norm_method", PROJECT_CFG["OUTPUT_NORM_METHOD"])
        metainfo_path = DataPartitioner.metadata_path(learning_params)
        metainfo.reload_tsfm_hub(metainfo_path)

        new_shards = []
        for default_path_bases in shard_meta["shard_paths"]:
            shard_path = ShardPaths.from_list(default_path_bases)
            shard_id = DataPartitioner.parse_shard_id(shard_path.train_y) 
            assert shard_id == DataPartitioner.parse_shard_id(shard_path.val_y), default_path_bases
            new_path_bases = DataPartitioner.train_val_path_bases(shard_id,
                output_norm_method=metainfo.output_norm_method,
                input_norm_method=metainfo.input_norm_method)
            new_shards.append(new_path_bases)
        shard_meta["shard_paths"] = new_shards

        notify(f"Read preprocessed data under {dataset_path}: {len(shard_meta['shard_paths'])} shards of size {shard_meta['shard_size']}")

        return dataset_path, shard_meta, metainfo
    
    @staticmethod
    def check_tmp_dataset_path(learning_params: dict):
        dataset_dir = DataPartitioner.tmp_dataset_dir(learning_params)
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Fail to find the preprocessed dataset under {dataset_dir}")
        elif len(os.listdir(dataset_dir)) == 0:
            raise ValueError(f"Empty directory {dataset_dir}")
        return dataset_dir
    
    @staticmethod 
    def tmp_dataset_exists(learning_params: dict):
        dataset_dir = DataPartitioner.tmp_dataset_dir(learning_params)
        return os.path.exists(dataset_dir) and os.path.exists(DataPartitioner.shard_meta_path(learning_params))

    def calculate_split(self, learning_params, device2task: Device2Task):
        ### For default data partition method, load all data, shuffle, then perform partition together
        ### Create a global Y array without samples that have been filtered out and mapping their index to `self.global_partition_rst`
        all_labels_std: np.ndarray = np.empty((0, 2))
        all_predefined_modes = np.empty(0, dtype=int)
        all_x = np.empty((0, self.data_meta_info.feature_len))

        self.local_log(f"Start to create Data Splits, is_cross_device={self.is_cross_device}")
        self.files_to_test = []

        def __handler(_f, _device, _split_mode, _all_labels_std, _all_predefined_modes):
            short_name = task_short_name(_f, _device)
            if short_name not in self.files2global_idxs:
                self.files_to_test.append((_f, _device))
                raw_data = load_raw_data([_f], 
                    learning_params, force=False, verbose=False)
                
                predefined_modes = np.ones(raw_data.size, dtype=int) * _split_mode

                _labels_std = raw_data.raw_data[:, :2]
                ### By default, all samples are filtered out
                self.files2global_idxs[short_name] = np.arange(len(_all_labels_std), len(_all_labels_std)+raw_data.size)

                _all_labels_std = np.concatenate((_all_labels_std, _labels_std), axis=0)
                _all_predefined_modes = np.concatenate((_all_predefined_modes, predefined_modes), axis=0)

                if self.filter_via_feature_entry:
                    raise NotImplementedError()
                    if isinstance(raw_data, ASTRawData):
                        raw_data.fix_seq_len = None
                        ### add zero padding will not affect np.sum
                        raw_data.max_seq_len = 16
                    raw_data.freeze()
                    assert len(raw_data.x) == len(_labels_std), short_name
                    ### TODO: use np.sum to handle the multi-leaf node issue
                    # NOTE that np.sum is highly related to raw_data.max_seq_len = 16
                    _x = np.sum(raw_data.x, axis=1) if len(raw_data.x.shape) == 3 else raw_data.x
                    if len(_x.shape) != 2:
                        import code
                        code.interact(local=locals())   
                    all_x = np.concatenate((all_x, _x), axis=0)
                    assert all_x is not None, raw_data.raw_data                          
            else:
                # print(f"[Preprocess warning] {short_name} occurs repeatedly")
                global_idxs = self.files2global_idxs[short_name]
                if SplitAttr.is_train(_split_mode):
                    _all_predefined_modes[global_idxs] = SplitAttr.to_train(_all_predefined_modes[global_idxs])
                elif SplitAttr.is_test(_split_mode):
                    _all_predefined_modes[global_idxs] = SplitAttr.to_test(_all_predefined_modes[global_idxs])
                else:
                    raise ValueError()
            
            return _all_labels_std, _all_predefined_modes
        
        if device2task.data_split_mode == "default":
            for device, device_files in device2task:
                self.local_log(f"Device {device}, split_mode={device2task.data_split_mode} ... ")
                for _f in tqdm(device_files):
                    all_labels_std, all_predefined_modes = __handler(_f, device, SplitAttr.default, all_labels_std, all_predefined_modes)
            
            assert all_labels_std is not None and len(all_labels_std.shape) == 2
            Y_I = np.concatenate((all_labels_std, 
                np.expand_dims(np.arange(len(all_labels_std)), 1)), axis=1)

            ### Global preprocess
            ### NOTE that here Y_I is not got through raw_data.freeze --> raw_data.y, 
            #  because freeze will remove some samples, make the index not align to raw_data.raw_data
            Y_I_filter = filter_samples(Y_I, self.data_meta_info, time_lb=self.ave_lb, verbose=self.verbose)

            ### Decide trainig set and test set
            train_Y_I_filter, test_Y_I_filter = trick_split_data_based_on_y(Y_I_filter, TRAIN2ALL_RATIO, index=False)

            self.global_partition_rst = np.ones(len(all_labels_std), dtype=int) * SplitAttr.filtered
            self.global_partition_rst[train_Y_I_filter[:, -1].astype(int)] = SplitAttr.train
            self.global_partition_rst[test_Y_I_filter[:, -1].astype(int)] = SplitAttr.test
        else:
            if device2task.data_split_mode == "by_device":
                for device, device_files in device2task:
                    _split_mode = device2task.parse_device_split_mode(device)
                    self.local_log(f"Device {device}, split_mode={device2task.data_split_mode}, mode {_split_mode} ...")
                    for _f in tqdm(device_files):
                        all_labels_std, all_predefined_modes = __handler(_f, device, _split_mode, all_labels_std, all_predefined_modes)
            elif device2task.data_split_mode == "by_net":
                for device in device2task.devices:
                    self.local_log(f"Device {device}, split_mode={device2task.data_split_mode} ... ")
                    for _f in tqdm(device2task[device].get("train", [])):
                        all_labels_std, all_predefined_modes = __handler(_f, device, SplitAttr.train, all_labels_std, all_predefined_modes)
                    for _f in tqdm(device2task[device].get("test", [])):
                        all_labels_std, all_predefined_modes = __handler(_f, device, SplitAttr.test, all_labels_std, all_predefined_modes)
            else:
                raise ValueError()

            assert all_labels_std is not None and len(all_labels_std.shape) == 2
            Y_I = np.concatenate((all_labels_std, 
                np.expand_dims(np.arange(len(all_labels_std)), 1)), axis=1)
            
            ### Global preprocess
            ### NOTE that here Y_I is not got through raw_data.freeze --> raw_data.y, 
            #  because freeze will remove some samples, make the index not align to raw_data.raw_data
            Y_I_filter = filter_samples(Y_I, self.data_meta_info, time_lb=self.ave_lb, verbose=self.verbose)

            self.global_partition_rst = np.ones(len(all_labels_std), dtype=int) * SplitAttr.filtered
            self.global_partition_rst[Y_I_filter[:, -1].astype(int)] = SplitAttr.ALL

            assert len(all_predefined_modes) == len(all_labels_std)
            self.global_partition_rst = np.bitwise_and(self.global_partition_rst, all_predefined_modes)

        ### Filter out test samples based on training features
        if self.filter_via_feature_entry:
            raise NotImplementedError()
            self.local_log("Filter out test samples based on training features")
            assert isinstance(all_x, np.ndarray)
            assert len(all_x) == len(all_labels_std)
            
            train_x = all_x[Y_I_train[:, -1].astype(int)]
            test_x = all_x[Y_I_test[:, -1].astype(int)]

            train_x_percentile_by_feature = (
                np.percentile(train_x, 5, axis=0),
                np.percentile(train_x, 95, axis=0),
            )
            assert train_x_percentile_by_feature[0].shape == train_x[0].shape
            in_valide_feature_entry = [i for i in range(len(train_x_percentile_by_feature[0])) 
                if train_x_percentile_by_feature[0][i] == train_x_percentile_by_feature[1][i]]

            idx_in_Y_I_test = np.ones(len(test_x), dtype=bool)
            assert len(idx_in_Y_I_test) == len(Y_I_test), (len(idx_in_Y_I_test), Y_I_test.shape)
            for _idx in range(len(test_x)):
                for entry_id, test_entry in enumerate(test_x[_idx]):
                    if entry_id in in_valide_feature_entry:
                        continue

                    if test_entry < train_x_percentile_by_feature[0][entry_id] or \
                        test_entry > train_x_percentile_by_feature[1][entry_id]:
                        idx_in_Y_I_test[_idx] = False
                        break

            Y_I_test = Y_I_test[idx_in_Y_I_test]
            self.local_log(f"Y_I_test afterwards: {Y_I_test.shape}")  

    def gen_train_test_data(self, task_file_name, device, raw_data):
        short_name = task_short_name(task_file_name, device)
        splits = self.global_partition_rst[self.files2global_idxs[short_name]]
        train_raw_data = raw_data.subset(np.where(SplitAttr.is_train(splits))[0])
        test_raw_data = raw_data.subset(np.where(SplitAttr.is_test(splits))[0])
        return train_raw_data, test_raw_data

    def _partition_dataset(self, learning_params):
        ''' 1. Loading data
            2. file-level shuffle
            3. partition
            4. Sample-level shuffle (inner-shard shuffle)
        '''

        save_dir = DataPartitioner.tmp_dataset_dir(learning_params)
        if os.path.exists(save_dir):
            raise ValueError(f"Temporary Dataset path {save_dir} already exists")
        os.makedirs(save_dir)

        assert self.files_to_test is not None
        _files_to_test = np.array(self.files_to_test)

        ### file level shuffle
        np.random.shuffle(_files_to_test)

        file_shards = np.array_split(_files_to_test, 
            math.ceil(len(_files_to_test)/self.file_shard_size))
        
        self.local_log(f"Start to load and pre-process data ... is_cross_device={self.is_cross_device}")

        shard_paths = []

        data_size = TrainTestPair(0, 0)
        for shard_id, file_shard in tqdm(enumerate(file_shards), total=len(file_shards), disable=(not self.verbose)):
            train_data: Union[None, RawData, ASTRawData] = None
            val_data: Union[None, RawData, ASTRawData] = None
            train_device_data = np.empty((0, DEVICE_FEATURE_LEN))
            val_device_data = np.empty((0, DEVICE_FEATURE_LEN))
            for _f, device in file_shard:
                raw_data = load_raw_data([_f], learning_params, verbose=False)
                if raw_data.size < self.min_sample_size:
                    continue
                
                ### Use Global Data Index to partition the dataset
                # self.verify_index(_f, device, raw_data)
                train_raw_data, val_raw_data = self.gen_train_test_data(_f, device, raw_data)

                # if len(train_raw_data) == 0 or len(val_raw_data) == 0:
                #     continue

                assert not train_raw_data.freezed and not val_raw_data.freezed
                if train_data:
                    train_data.combine(train_raw_data, inplace=True, verbose=False)
                    val_data.combine(val_raw_data, inplace=True, verbose=False)
                else:
                    train_data = train_raw_data
                    val_data = val_raw_data

                if self.is_cross_device:
                    x_device: np.ndarray = get_device_feature(device)
                    if len(train_raw_data) > 0:
                        train_device_data = np.concatenate((train_device_data, 
                            np.tile(x_device, len(train_raw_data)).reshape(len(train_raw_data), -1)), axis=0)
                    if len(val_raw_data) > 0:
                        val_device_data = np.concatenate((val_device_data,
                            np.tile(x_device, len(val_raw_data)).reshape(len(val_raw_data), -1)), axis=0)
            
            if train_data is not None:
                train_data.metainfo = self.data_meta_info
                ### Sample-level shuffling
                perm = train_data.shuffle()
                selected_idx = train_data.freeze()
                train_data.normalize()
                if self.is_cross_device:
                    train_device_data = train_device_data[perm]
                    if selected_idx is not None:
                        train_device_data = train_device_data[selected_idx]
                    assert len(train_device_data) == len(train_data)
                data_size.train += len(train_data)
            
            if val_data is not None:
                val_data.metainfo = self.data_meta_info
                perm = val_data.shuffle()
                selected_idx = val_data.freeze()
                val_data.normalize()
                if self.is_cross_device:
                    val_device_data = val_device_data[perm]
                    if selected_idx is not None:
                        val_device_data = val_device_data[selected_idx]
                    assert len(val_device_data) == len(val_data)
                data_size.val += len(val_data)
            
            samples_list = []
            samples_list.append(Sample(train_data.x, train_data.y, train_device_data))
            samples_list.append(Sample(val_data.x, val_data.y, val_device_data))
            
            ### TODO: comment this
            # print("Size before: ", [s.size for s in samples_list])
            # remove_vague_samples(samples_list)
            # print("Size after: ", [s.size for s in samples_list])

            ### Save this shard
            _shard_paths = DataPartitioner.train_val_path_bases(shard_id,
                output_norm_method=PROJECT_CFG["OUTPUT_NORM_METHOD"],
                input_norm_method=PROJECT_CFG["INPUT_NORM_METHOD"])
            np.save(os.path.join(save_dir, _shard_paths.train_x_tir), samples_list[0].x_tir)
            np.save(os.path.join(save_dir, _shard_paths.train_y), samples_list[0].y)
            np.save(os.path.join(save_dir, _shard_paths.val_x_tir), samples_list[1].x_tir)
            np.save(os.path.join(save_dir, _shard_paths.val_y), samples_list[1].y)
            if self.is_cross_device:
                np.save(os.path.join(save_dir, _shard_paths.train_x_device), samples_list[0].x_device)
                np.save(os.path.join(save_dir, _shard_paths.val_x_device), samples_list[1].x_device)
            shard_paths.append(_shard_paths.tolist())

        self.local_log(f"Train data size: {data_size.train}, Test data size: {data_size.val}")
        self.data_meta_info.save(DataPartitioner.metadata_path(learning_params))
        with open(DataPartitioner.shard_meta_path(learning_params), 'w') as fp:
            json.dump({
                "shard_paths": shard_paths,
                "mode": learning_params["mode"],
                "shard_size": self.file_shard_size,
                "output_norm_method": PROJECT_CFG["OUTPUT_NORM_METHOD"],
                "input_norm_method": PROJECT_CFG["INPUT_NORM_METHOD"],
                "is_cross_device": self.is_cross_device,
                }, fp, indent=4)

        self.local_log(f"Pre-process done, results cached under {save_dir}")

    def _modify_x(self, metainfo, learning_params, shard_meta, default_dir):
        metainfo.output_norm_method = shard_meta["output_norm_method"]
        metainfo.input_norm_method = shard_meta["input_norm_method"]
        assert metainfo.input_norm_method == "min-max"
        metainfo_path = DataPartitioner.metadata_path(learning_params)
        metainfo.reload_tsfm_hub(metainfo_path)
        new_metainfo = copy.deepcopy(metainfo)
        new_metainfo.input_norm_method = int(PROJECT_CFG['INPUT_NORM_METHOD']) if PROJECT_CFG['INPUT_NORM_METHOD'].isdigit() else PROJECT_CFG['INPUT_NORM_METHOD']
        self.local_log(f"Preprocess X with {new_metainfo.input_norm_method} transform based on X with {metainfo.input_norm_method}")
        self.local_log("Original Metainfo")
        metainfo.tsfm_hub.print()

        use_data_to_reload_tsfm = True
        if use_data_to_reload_tsfm:
            ### Correct distribution transforms for Y
            X_to_fit = np.empty((0, metainfo.feature_len))
            for default_path_bases in shard_meta['shard_paths']:
                shard_paths = ShardPaths.from_list(default_path_bases)
                default_paths = shard_paths.abs_paths(default_dir)
                ### Load default Y
                train_x = np.load(default_paths.train_x_tir, allow_pickle=True).astype(float)
                if len(train_x.shape) == 3:
                    ### TODO (Huhanpeng): how to handle multiple leaf X
                    train_x = np.average(train_x, axis=1)
                else:
                    assert len(X_to_fit.shape) == len(train_x.shape), (len(X_to_fit.shape), len(train_x.shape))
                X_to_fit = np.concatenate((X_to_fit, train_x), axis=0)

            X_to_fit = metainfo.de_norm_input(X_to_fit)
            new_metainfo.tsfm_hub.parse_x_norm_method(new_metainfo.input_norm_method, X_to_fit)
        else:
            ### Use a unified metainfo, directly load corresonding transformer
            new_metainfo.reload_tsfm_hub(metainfo_path)

        self.local_log("New Metainfo")
        new_metainfo.tsfm_hub.print()
        del X_to_fit
        
        for default_path_bases in shard_meta['shard_paths']:
            shard_paths = ShardPaths.from_list(default_path_bases)
            default_paths = shard_paths.abs_paths(default_dir)

            ### Load default X
            train_x = np.load(default_paths.train_x_tir, allow_pickle=True).astype(float)
            val_x = np.load(default_paths.val_x_tir, allow_pickle=True).astype(float)

            ### Generate new X
            def _handle_x(X, batch_size = 512): 
                st = 0
                _X = None
                while st < len(X):
                    ed = st + batch_size
                    origin_X = metainfo.de_norm_input(X[st:ed])
                    batch_X = new_metainfo.norm_input(origin_X)
                    if not np.any(new_metainfo.de_norm_input(batch_X) == origin_X):
                        import code; code.interact(local=locals())
                    _X = batch_X if _X is None else np.concatenate((_X, batch_X), axis=0)
                    st += batch_size
                return _X
            
            train_x = _handle_x(train_x)
            val_x = _handle_x(val_x)
            
            ### Cache new X
            shard_id = DataPartitioner.parse_shard_id(shard_paths.train_x_tir) 
            assert shard_id == DataPartitioner.parse_shard_id(shard_paths.val_x_tir), default_path_bases
            new_path_bases = DataPartitioner.train_val_path_bases(shard_id,
                output_norm_method=PROJECT_CFG["OUTPUT_NORM_METHOD"],
                input_norm_method=PROJECT_CFG["INPUT_NORM_METHOD"])

            assert not os.path.exists(new_path_bases.train_x_tir), new_path_bases
            self.local_log(f"Save new X to {new_path_bases.train_x_tir} and {new_path_bases.val_x_tir}")
            np.save(os.path.join(default_dir, new_path_bases.train_x_tir), train_x)
            np.save(os.path.join(default_dir, new_path_bases.val_x_tir), val_x)
        new_metainfo.save(DataPartitioner.metadata_path(learning_params))

    def _modify_y(self, metainfo, learning_params, shard_meta, default_dir):
        metainfo.output_norm_method = shard_meta["output_norm_method"]
        metainfo.input_norm_method = shard_meta.get("input_norm_method", "min-max")
        metainfo_path = DataPartitioner.metadata_path(learning_params)
        metainfo.reload_tsfm_hub(metainfo_path)
        new_metainfo = copy.deepcopy(metainfo)
        new_metainfo.output_norm_method = int(PROJECT_CFG['OUTPUT_NORM_METHOD']) if PROJECT_CFG['OUTPUT_NORM_METHOD'].isdigit() else PROJECT_CFG['OUTPUT_NORM_METHOD']
        self.local_log(f"Preprocess Y with transform {new_metainfo.output_norm_method} based on Y with transform {metainfo.output_norm_method}")
        self.local_log("Original Metainfo")
        metainfo.tsfm_hub.print()

        use_data_to_reload_tsfm = False
        if use_data_to_reload_tsfm:
            ### Correct distribution transforms for Y
            Y_to_fit = np.empty((0))
            for default_path_bases in shard_meta['shard_paths']:
                shard_paths = ShardPaths.from_list(default_path_bases)
                default_paths = shard_paths.abs_paths(default_dir)
                ### Load default Y
                train_y = np.load(default_paths.train_y, allow_pickle=True).astype(float)
                # val_y = np.load(default_paths.val_y, allow_pickle=True).astype(float)
                assert len(Y_to_fit.shape) == len(train_y.shape), (len(Y_to_fit.shape), len(train_y.shape))
                Y_to_fit = np.concatenate((Y_to_fit, train_y), axis=0)
        
            Y_to_fit = metainfo.de_standardize_output(Y_to_fit)
            new_metainfo.tsfm_hub.parse_y_norm_method(new_metainfo.output_norm_method, Y_to_fit)
        else:
            ### Use a unified metainfo, directly load corresonding transformer
            new_metainfo.reload_tsfm_hub(metainfo_path)

        self.local_log("New Metainfo")
        new_metainfo.tsfm_hub.print()

        self.local_log("Origin Metainfo convert [1, 2] to ", metainfo.tsfm_hub["y"].transform(np.array([1, 2])))
        self.local_log("New Metainfo convert [1, 2] to ", new_metainfo.tsfm_hub["y"].transform(np.array([1, 2])))
        
        for default_path_bases in shard_meta['shard_paths']:
            shard_paths = ShardPaths.from_list(default_path_bases)
            default_paths = shard_paths.abs_paths(default_dir)

            ### Load default Y
            train_y = np.load(default_paths.train_y, allow_pickle=True).astype(float)
            val_y = np.load(default_paths.val_y, allow_pickle=True).astype(float)

            ### Generate new Y
            _train_y = new_metainfo.standardize_output(metainfo.de_standardize_output(train_y))
            _val_y = new_metainfo.standardize_output(metainfo.de_standardize_output(val_y))
            
            ### Cache new Y
            shard_id = DataPartitioner.parse_shard_id(shard_paths.train_y) 
            assert shard_id == DataPartitioner.parse_shard_id(shard_paths.val_y), default_path_bases
            new_path_bases = DataPartitioner.train_val_path_bases(shard_id,
                output_norm_method=PROJECT_CFG["OUTPUT_NORM_METHOD"],
                input_norm_method=PROJECT_CFG["INPUT_NORM_METHOD"])

            assert not os.path.exists(new_path_bases.train_y), new_path_bases
            self.local_log(f"Save new Y to {new_path_bases.train_y} and {new_path_bases.val_y}")
            np.save(os.path.join(default_dir, new_path_bases.train_y), _train_y)
            np.save(os.path.join(default_dir, new_path_bases.val_y), _val_y)
        new_metainfo.save(DataPartitioner.metadata_path(learning_params))
        
    def partition_dataset(self, learning_params):
        default_dir = DataPartitioner.tmp_dataset_dir(learning_params)
        shard_meta_path = DataPartitioner.shard_meta_path(learning_params)
        self.local_log(f"Preprocess data and store it under {default_dir}")
        if os.path.exists(shard_meta_path):
            
            with open(shard_meta_path, 'r') as fp:
                shard_meta = json.load(fp)
            
            if "input_norm_method" not in shard_meta:
                warning("[Deprecated] \"input_norm_method\" should be in shard_meta")
                shard_meta["input_norm_method"] = "min-max"
                
            target_shard_paths = DataPartitioner.train_val_path_bases(0,
                _dir = default_dir,
                output_norm_method=PROJECT_CFG["OUTPUT_NORM_METHOD"],
                input_norm_method=PROJECT_CFG["INPUT_NORM_METHOD"])

            try:
                metainfo = DataPartitioner.load_metainfo(learning_params)
            except:
                device2task = sample_task_files(learning_params["input"], learning_params["mode"],
                    learning_params["gpu_model"], abs=True)
                metainfo = parse_metainfo(device2task, learning_params, True)
                
            if not os.path.exists(target_shard_paths.train_x_tir):
                ### Avoid generate repeated Y if only X is different
                self._modify_x(metainfo, learning_params, shard_meta, default_dir)

            if not os.path.exists(target_shard_paths.train_y):
                ### Avoid generate repeated X if only Y is different
                self._modify_y(metainfo, learning_params, shard_meta, default_dir)
            
        else:
            self.trace_root_path = learning_params["input"]
            self.ave_lb = learning_params["ave_lb"]
            assert self.trace_root_path is not None
            # init_fea_info_via_data(self.trace_root_path)
            device2task = sample_task_files(self.trace_root_path, learning_params["mode"],
                learning_params["gpu_model"], absolute_path=True)
            print("Device2Task: \n", str(device2task))
            self.is_cross_device = device2task.is_cross_device

            if self.data_meta_info is None:
                self.data_meta_info = parse_metainfo(device2task, learning_params,
                    True, use_default_path=True)

            ### Loading data twice is necessary because we need to do some 
            # **global** pre-processing before partition
            self.calculate_split(learning_params, device2task)
            self._partition_dataset(learning_params)

    def save(self):
        raise NotImplementedError()

def test_data_index(learning_params):
    data_index = DataSplit()
    data_index.calculate_split(learning_params)

    data_index.save()
    del data_index

    data_index = DataSplit.load_init(learning_params["input"])
    train_cnt = test_cnt = 0
    for partition in data_index.file2index.values():
        train_cnt += len(np.where(partition==1)[0])
        test_cnt += len(np.where(partition==2)[0])
    print(train_cnt, test_cnt)

def check_dataset(learning_params):
    dataset_path, shard_meta, data_meta_info = DataPartitioner.load_shard(learning_params)

    ### check data_meta_info
    assert data_meta_info.output_norm_method == PROJECT_CFG["OUTPUT_NORM_METHOD"], data_meta_info.output_norm_method

    data_size = [0, 0]
    for shard_path_bases in shard_meta["shard_paths"]:
        # local_log(shard_path_bases)
        shard_path = shard_path_bases.abs_paths(dataset_path)
        train_y = np.load(shard_path.train_y, allow_pickle=True).astype(float)
        val_y = np.load(shard_path.val_y, allow_pickle=True).astype(float)
        data_size[0] += len(train_y)
        data_size[1] += len(val_y)
    print(f"Train data size: {data_size[0]}, Test data size: {data_size[1]}")

    from metalearner.data.map2norm import METHOD_NAMES, METHODS
    metainfo_path = DataPartitioner.metadata_path(learning_params)
    default_dir = DataPartitioner.tmp_dataset_dir(learning_params)
    sample_num = 1024

    ### check different version of x
    method2retrieve_x = []
    for method_id in range(len(METHOD_NAMES)):
        target_shard_paths = DataPartitioner.train_val_path_bases(0,
            _dir = default_dir, input_norm_method=method_id)
        if not os.path.exists(target_shard_paths.val_x_tir):
            method2retrieve_x.append(None)
            continue
        val_x = np.load(target_shard_paths.val_x_tir, allow_pickle=True).astype(float)
        val_x = val_x[:sample_num]
        _tmp_metainfo = copy.deepcopy(data_meta_info)
        _tmp_metainfo.input_norm_method = method_id
        _tmp_metainfo.output_norm_method = shard_meta["output_norm_method"]
        _tmp_metainfo.reload_tsfm_hub(metainfo_path)
        origin_val_x = _tmp_metainfo.de_norm_input(val_x)
        method2retrieve_x.append(origin_val_x)
    for method_id in range(len(METHOD_NAMES)-1):
        if method2retrieve_x[method_id] is None:
            continue
        for compare_to in range(method_id+1, len(METHOD_NAMES)):
            if method2retrieve_x[compare_to] is None:
                continue
            if np.all(method2retrieve_x[method_id] == method2retrieve_x[compare_to]):
                print(f"[Check X] Method {method_id} is CONSISTENT with {compare_to}")
            else:
                print(f"[Check X] Method {method_id} is CONFLICT with {compare_to}")
                import pdb; pdb.set_trace()

    ### check different version of y
    method2retrieve_y = []
    for method_id in range(len(METHOD_NAMES)):
        target_shard_paths = DataPartitioner.train_val_path_bases(0,
            _dir = default_dir, output_norm_method=method_id)
        if not os.path.exists(target_shard_paths.val_y):
            method2retrieve_y.append(None)
            continue
        val_y = np.load(target_shard_paths.val_y, allow_pickle=True).astype(float)
        val_y = val_y[:sample_num]
        _tmp_metainfo = copy.deepcopy(data_meta_info)
        _tmp_metainfo.input_norm_method = shard_meta.get("input_norm_method", "min-max")
        _tmp_metainfo.output_norm_method = method_id
        _tmp_metainfo.reload_tsfm_hub(metainfo_path)
        origin_val_y = _tmp_metainfo.de_standardize_output(val_y)
        method2retrieve_y.append(origin_val_y)
    for method_id in range(len(METHOD_NAMES)-1):
        if method2retrieve_y[method_id] is None:
            continue
        for compare_to in range(method_id+1, len(METHOD_NAMES)):
            if method2retrieve_y[compare_to] is None:
                continue
            if np.all(method2retrieve_y[method_id] == method2retrieve_y[compare_to]):
                print(f"[Check Y] Method {method_id} is CONSISTENT with {compare_to}")
            else:
                print(f"[Check Y] Method {method_id} is CONFLICT with {compare_to}")
                import pdb; pdb.set_trace()

def make_dataset(learning_params, data_meta_info=None, verbose=True):
    dataset_dir = DataPartitioner.tmp_dataset_dir(learning_params)
    shard_paths = DataPartitioner.train_val_path_bases(0,
        output_norm_method=PROJECT_CFG["OUTPUT_NORM_METHOD"],
        input_norm_method=PROJECT_CFG["INPUT_NORM_METHOD"],
        _dir = dataset_dir)
    if os.path.exists(shard_paths.train_x_tir) and os.path.exists(shard_paths.train_y):
        ### Both X and Y has been generated
        print(shard_paths.train_x_tir, shard_paths.train_y)
        check_dataset(learning_params)
        raise ValueError(f"{shard_paths.train_y} already exists")

    ### Check if the preprocessed data is available at the default directory
    tmp = os.environ.get("CDPP_DATASET_PATH", os.path.join("tmp", "dataset"))
    linked = False
    if tmp != os.path.join("tmp", "dataset"):
        os.environ["CDPP_DATASET_PATH"] = os.path.join("tmp", "dataset")
        default_dataset_dir = os.path.abspath(DataPartitioner.tmp_dataset_dir(learning_params))
        if os.path.exists(default_dataset_dir) and len(os.listdir(default_dataset_dir)) > 0:
            print(f"The preprocessed data is available at {default_dataset_dir}, make a symbol link")
            os.makedirs(os.path.dirname(dataset_dir), exist_ok=True)
            os.symlink(default_dataset_dir, dataset_dir)
            linked = True
        os.environ["CDPP_DATASET_PATH"] = tmp

    if not linked:
        data_part = DataPartitioner(data_meta_info=data_meta_info, verbose=verbose)
        data_part.partition_dataset(learning_params)