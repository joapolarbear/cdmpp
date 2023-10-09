import numpy as np
import random
import os
import scipy.stats
import math
from sklearn.cluster import KMeans
from typing import Tuple, Union
import time

import torch

from utils.util import (notify, warning, read_yes,
    sample_task_files, TrainTestPair)
from tvm_helper.metadata import DataMetaInfo, ASTMetaInfo
from tvm_helper.tir_helper import (
    feature_len,
    feature_shape2d
)
from metalearner.data.rawdata import (
    MIN_PER_TASK_SAMPLE_NUM,
    RawData,
    ASTRawData,
    extract_task_id,
    load_raw_data
)
from metalearner.learn_utils import CMFeature
from metalearner.data.preprocess import DataPartitioner, make_dataset

def get_di_len():
    return feature_shape2d()[1]

FILTER_DATA_AFTER_NORM = False

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_data: Union[ASTRawData, RawData]):
        self.raw_data = raw_data
        self.finalize_rawdata()

    def __getitem__(self, index):
        x, y, di = self.raw_data.slice_freezed(index)
        return torch.FloatTensor(x), torch.FloatTensor(np.array([y]))

    def __len__(self):
        return self.raw_data.size
    
    def combine(self, others):
        ### In place add 'others''s raw_data to this dataset
        if isinstance(others, MyDataSet):
            self.raw_data = self.raw_data.combine(others.raw_data)
        elif isinstance(others, RawData):
            self.raw_data = self.raw_data.combine(others)
        else:
            raise ValueError
    
    def finalize_rawdata(self):
        self.raw_data.freeze()
        self.raw_data.normalize()
        if FILTER_DATA_AFTER_NORM:
            self.raw_data.afterward_filter()
    
    def get_all_data(self):
        return self.raw_data.x, self.raw_data.y


class WrapDataloader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.dataloader = torch.utils.data.DataLoader(*args, **kwargs)
    
    def __iter__(self):
        for (batch_x, batch_y) in self.dataloader:
            yield CMFeature(batch_x), batch_y


class LargeIterableDataset(torch.utils.data.IterableDataset):
    ''' Load data in sequence '''
    def __init__(self, learning_params, is_train=True, data_meta_info=None, drop_last=False):
        ### Do not init LargeIterableDataset
        super(LargeIterableDataset, self).__init__()

        self.learning_params = learning_params
        self.is_train = is_train
        self.drop_last = drop_last

        # TODO: parsing data_meta_info as an argument should be deperacated
        assert data_meta_info is None

        if not DataPartitioner.tmp_dataset_exists(learning_params):
            print(f"[Dataloader] fail to find preprocessed data, make dataset first ... ")
            make_dataset(learning_params, data_meta_info=data_meta_info)
            ### Reset the random seed
            # torch.manual_seed(0)
            # random.seed(0)
            # np.random.seed(0)
        
        self.dataset_path, shard_meta, self.data_meta_info = DataPartitioner.load_shard(learning_params)
        self.task_shards = np.array(shard_meta["shard_paths"])
        self.is_cross_device = shard_meta.get("is_cross_device", False)
        # print("All shards", self.task_shards)
        self.shuffle()

        self.train_device = torch.device(os.environ.get('train_device', "cuda"))
        self.batch_size = None

        ts = time.time()
        # print(f"[Dataloader] all_shards for {'training' if self.is_train else 'test'} data: ")
        try:
            self.all_shards = [self.fetch_shard(_id, shard_path_bases) for _id, shard_path_bases in enumerate(self.task_shards)]
        except FileNotFoundError:
            print(f"[Dataloader] fail to find preprocessed data, modify dataset's y or x ... ")
            make_dataset(learning_params, data_meta_info=data_meta_info)
            ts = time.time()
            self.all_shards = [self.fetch_shard(_id, shard_path_bases) for _id, shard_path_bases in enumerate(self.task_shards)]
        print(f"[Dataloader] Take {time.time() - ts} s to load all {len(self.all_shards)} {'training' if self.is_train else 'test'} shard(s)")
        print(f"[Dataloader] transformation used: ")
        self.data_meta_info.tsfm_hub.print()
        
        self.sample_limit = -1
        # limits = [10e3, 30e3, 50e3, 70e3, 100e3, 300e3, 500e3, 700e3, 900e3, 1.2e6, 1.5e6, 2.0e6]
        # if self.is_train:
        #     # self.all_shards = self.all_shards[:1]
        #     self.sample_limit = limits[3]
        #     if not read_yes(f"Limit the size of training set to {self.sample_limit}"):
        #         exit(0)   

        self._size = None  

    def fetch_shard(self, shard_id, shard_path_bases):
        shard_path = shard_path_bases.abs_paths(self.dataset_path)
        if self.is_train:
            shard_x_tir_path = shard_path.train_x_tir
            shard_y_path = shard_path.train_y
            shard_x_device_path = shard_path.train_x_device
        else:
            shard_x_tir_path = shard_path.val_x_tir
            shard_y_path = shard_path.val_y
            shard_x_device_path = shard_path.val_x_device

        shard_x_tir = np.load(shard_x_tir_path, allow_pickle=True).astype(float)
        shard_y = np.load(shard_y_path, allow_pickle=True).astype(float)
        shard_x_device = None
        if self.is_cross_device:
            shard_x_device = np.load(shard_x_device_path, allow_pickle=True).astype(float)

        assert len(shard_x_tir) == len(shard_y)
        # print(f"\tShard {shard_id}:", os.path.basename(shard_x_tir_path), 
        #     os.path.basename(shard_y_path), os.path.basename(shard_x_device_path))
        return shard_x_tir, shard_y, shard_x_device
    
    def __iter__(self):
        assert isinstance(self.batch_size, int)
        residual_x = residual_y = residual_x_device = None
        sample_cnt = 0
        limit_met = False
        for shard_x_tir, shard_y, shard_x_device in self.all_shards:
            ### Iterate one shard
            shard_x_tir = torch.FloatTensor(shard_x_tir)
            shard_y = torch.FloatTensor(shard_y)
            if self.is_cross_device:
                shard_x_device = torch.FloatTensor(shard_x_device)

            if residual_x is not None:
                shard_x_tir = torch.cat((residual_x, shard_x_tir), axis=0)
                shard_y = torch.cat((residual_y, shard_y), axis=0)
                if self.is_cross_device:
                    shard_x_device = torch.cat((residual_x_device, shard_x_device), axis=0)

            for batch_id in range(len(shard_x_tir) // self.batch_size):
                batch_x_tir = shard_x_tir[(batch_id * self.batch_size):((batch_id+1) * self.batch_size)]
                batch_y = shard_y[(batch_id * self.batch_size):((batch_id+1) * self.batch_size)].reshape((self.batch_size, -1))
                if self.is_cross_device:
                    batch_x_device = shard_x_device[(batch_id * self.batch_size):((batch_id+1) * self.batch_size)]
                else:
                    batch_x_device = None
                batch_x = CMFeature(batch_x_tir, x_device=batch_x_device)
                yield batch_x.to(self.train_device), batch_y.to(self.train_device)

                sample_cnt += self.batch_size
                if self.sample_limit > 0 and sample_cnt >= self.sample_limit:
                    limit_met = True
                    break

            if limit_met:
                break
            
            end_sample_id = (len(shard_x_tir) // self.batch_size) * self.batch_size
            if end_sample_id < len(shard_x_tir):
                residual_x = shard_x_tir[end_sample_id:]
                residual_y = shard_y[end_sample_id:]
                if self.is_cross_device:
                    residual_x_device = shard_x_device[end_sample_id:]

        if not self.drop_last and len(residual_y) > 0:
            batch_x = CMFeature(residual_x, x_device=residual_x_device)
            batch_y = residual_y.reshape((-1, 1))
            yield batch_x.to(self.train_device), batch_y.to(self.train_device)

    def shuffle(self):
        np.random.shuffle(self.task_shards)
    
    def __len__(self):
        if self._size is None:
            prev_batch_size = self.batch_size
            self.batch_size = 128
            self._size = 0
            for batch_x, batch_y in iter(self):
                self._size += len(batch_y)
            self.batch_size = prev_batch_size
        return self._size

class LargeIterDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, enable_up_sampling=None, shuffle=True):
        self.dataset = dataset
        self.dataset.batch_size = batch_size

        if shuffle:
            self.dataset.shuffle()
        if enable_up_sampling is not None:
            self.dataset.enable_up_sampling = enable_up_sampling
    
    def __iter__(self):
        return iter(self.dataset)


def rawdata2dataset_pair(raw_data: Union[ASTRawData, RawData], verbose=True) -> TrainTestPair:
    ''' Convert raw data to training dataset and validation dataset
    '''
    if raw_data.size < MIN_PER_TASK_SAMPLE_NUM:
        return TrainTestPair(None, None)
    train_raw_data, val_raw_data = raw_data.gen_train_test_data()
    train_data = MyDataSet(train_raw_data)
    val_data = MyDataSet(val_raw_data)
    print("Training data size: {}, validation data size: {}".format(len(train_data), len(val_data)))
    return TrainTestPair(train_data, val_data)

def load_non_iter_dataset(files, learning_params, data_meta_info=None, verbose=True):
    raw_data = load_raw_data(files, learning_params, force=True, verbose=verbose)
    if raw_data.size < MIN_PER_TASK_SAMPLE_NUM:
        warning(f"Ignore {files} since it only has {raw_data.size} samples")
        return None, None
    # assert data_meta_info is None
    if data_meta_info:
        raw_data.metainfo = data_meta_info
    raw_data.preprocess(time_lb=learning_params["ave_lb"], verbose=verbose)
    ds_pair = rawdata2dataset_pair(raw_data, verbose=verbose)
    return ds_pair, raw_data.metainfo

def _get_all_Y(_dataloader: LargeIterDataloader):
    ret = np.empty((0, 1))
    for batch_x, batch_y in _dataloader:
        ret = np.concatenate((ret, batch_y.data.cpu().numpy()))
    return ret

def test_map_to_norm(train_data: LargeIterableDataset, val_data, data_meta_info):
    Y_train = data_meta_info.de_standardize_output(_get_all_Y(LargeIterDataloader(train_data, 256)))
    Y_test = data_meta_info.de_standardize_output(_get_all_Y(LargeIterDataloader(val_data, 256)))

    y_min = min(min(Y_train), min(Y_test))[0]
    y_max = max(max(Y_train), max(Y_test))[0]
    sample_num = int(min(min(1e4, len(Y_test)), len(Y_train)))
    print(f"Train size: {len(Y_train)}, Test size: {len(Y_test)}, Sample num: {sample_num}")
    print(f"Max Y: {y_max}, Min Y: {y_min}")

    ### Sampling
    # import pdb; pdb.set_trace()
    Y_pair = TrainTestPair(
        Y_train[np.random.choice(len(Y_train), sample_num, replace=False), :],
        Y_test[np.random.choice(len(Y_test), sample_num, replace=False), :],)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    fig_base = 330
    bin_num = 20

    fig_base += 1
    ax = fig.add_subplot(fig_base)
    plt.hist(Y_pair.train, bins=bin_num, range=(y_min, y_max))
    plt.xlabel("Y_train")
    plt.ylabel("Frequency")
    fig_base += 1
    ax = fig.add_subplot(fig_base)
    plt.hist(Y_pair.val, bins=bin_num, range=(y_min, y_max))
    plt.xlabel("Y_test")
    plt.ylabel("Frequency")

    from sklearn.preprocessing import PowerTransformer
    from sklearn.preprocessing import QuantileTransformer
    rng = np.random.RandomState(304)
    bc = PowerTransformer(method="box-cox")
    yj = PowerTransformer(method="yeo-johnson")
    # n_quantiles is set to the training set size rather than the default value
    # to avoid a warning being raised by this example
    qt = QuantileTransformer(
        n_quantiles=500, output_distribution="normal", random_state=rng
    )
    
    method_names = ["Box-Cox", "Yeo-Johnson", "Quantile"]
    method_fits = [bc, yj, qt]

    for _name, _fit in zip(method_names, method_fits):
        Y_test_fit = _fit.fit(Y_pair.train).transform(Y_pair.val)
        fig_base += 1
        ax = fig.add_subplot(fig_base)
        plt.hist(Y_test_fit, bins=bin_num,
            # range=(y_min, y_max)
            )
        plt.xlabel(f"Y_test with {_name} transforms")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("tmp/tmp5.png")
    plt.close()
    raise

def load_iter_dataset(learning_params):
    ''' Load iterable training and validation dataset based on `files`
        avoid OOM with a large dataset
    '''

    train_data = LargeIterableDataset(learning_params, is_train=True)
    val_data = LargeIterableDataset(learning_params, is_train=False)
    data_meta_info = train_data.data_meta_info

    # ds_pair.train.check_max_leaf_no()
    # ds_pair.val.check_max_leaf_no()
    # ds_pair.train.check_leaf_no_dist("./tmp/leaf_no_dist_train.png")
    # ds_pair.val.check_leaf_no_dist("./tmp/leaf_no_dist_val.png")
    # test_map_to_norm(train_data, val_data, data_meta_info)

    return TrainTestPair(train_data, val_data), data_meta_info

def load_train_iter_dataset(learning_params, data_meta_info):
    ''' Load iterable training and validation dataset based on `files`
        avoid OOM with a large dataset
    '''
    train_data = LargeIterableDataset(learning_params, 
        is_train=True, data_meta_info=data_meta_info)
    return train_data

def load_val_iter_dataset(learning_params, data_meta_info):
    ''' Load iterable training and validation dataset based on `files`
        avoid OOM with a large dataset
    '''
    val_data = LargeIterableDataset(learning_params,
        is_train=False, data_meta_info=data_meta_info)
    return val_data

def load_raw_data_w_cluster_id(
        files_or_dir,
        learning_params,
        task_sample_num=10,
        cluster_method="kmeans",
        verbose=True
        ):
    ### Prepare data
    if isinstance(files_or_dir, str):
        raise NotImplementedError("Do not support specifying sample_num")
        device2tasks_info = sample_task_files(files_or_dir, learning_params["mode"], 
            learning_params["gpu_model"], sample_num=task_sample_num, 
            absolute_path=True)[learning_params["gpu_model"]]
        root_path, files_to_test = device2tasks_info["root_path"], device2tasks_info["tasks"]
    elif isinstance(files_or_dir, list):
        assert task_sample_num is None
        files_to_test = files_or_dir
    else:
        raise
    print(f"Selected {len(files_to_test)} task files")
    
    raw_data = load_raw_data(
            files_to_test, learning_params,
            force=True, verbose=False)
    
    raw_data.preprocess(time_lb=learning_params["ave_lb"], verbose=verbose)
    raw_data.freeze()

    ### Clustering
    if cluster_method == "kmeans":
        cluster_num = 10
        kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(raw_data.x)
        cluster_rst = kmeans.labels_
        ### kmeans.labels_ is the clustering results, \in R^(xydata.shape[0])
        ### kmeans.cluster_centers_ contains the centroids of the result clusers, \in R^{n_clusters, xydata.shape[1]}
        ### Use kmeans.predict(_batch) to predict the cluster id of _batch
    elif cluster_method == "di":
        hash_table = {}
        cluster_rst = np.zeros(raw_data.size)
        for idx in range(raw_data.size):
            hashable_di = tuple(raw_data.di[idx].flatten())
            if hashable_di not in hash_table:
                hash_table[hashable_di] = {
                    "key": len(hash_table)
                }
            cluster_rst[idx] = hash_table[hashable_di]["key"]
    else:
        raise
    return raw_data, cluster_rst, files_to_test

def group_raw_data(cluster_rst):
    group_idx_dict = {}
    for idx in range(len(cluster_rst)):
        if cluster_rst[idx] not in group_idx_dict:
            group_idx_dict[cluster_rst[idx]] = []
        group_idx_dict[cluster_rst[idx]].append(idx)
    return group_idx_dict
