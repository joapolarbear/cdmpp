''' Test the Dataset and IterableDataset

Sample commands:
bash scripts/train.sh test --mode sample12 -i .workspace/ast_ansor      
'''
import time
import numpy as np

import torch

from metalearner.data.dataloader import (
    load_iter_dataset,
    load_non_iter_dataset,
)

from metalearner.data.rawdata import load_raw_data, parse_metainfo

def eval_traversal_time(name, dataset, batch_size):
    ''' Evaluate the time to traversal one dataset

    Return
    ------
    dur_init: time cost to initialize the dataloader
    dur_trav: time cost to traverse the dataset
    cnt: the size of the dataset
    '''
    ts = time.time()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=None)
    dur_init = time.time() - ts
    ts = time.time()
    cnt = 0
    for _ in dataloader:
        cnt += 1
    dur_trav = time.time() - ts
    print(f"{name}: bs={batch_size}, dataloader init time={dur_init}s, traverse time={dur_trav}s, data size={batch_size * cnt}, throughput={(batch_size * cnt)/dur_trav}/s")
    return dur_init, dur_trav, cnt
    

def test_dataset_and_iterable_dataset(files, learning_params, data_meta_info=None):
    ds_pair, _ = load_non_iter_dataset(files, learning_params,
            data_meta_info=data_meta_info, verbose=False)

    ds_pair_iterable, _ = load_iter_dataset(files, learning_params,
        data_meta_info=data_meta_info)

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 512]:
        # rst = eval_traversal_time("Dataset", ds_pair.train, batch_size=batch_size)
        rst_iterable = eval_traversal_time("IterableDataset", ds_pair_iterable.train, batch_size=batch_size)
        print("")

def test_load_raw_data_cost(files, learning_params):
    repeated_times = 10
    files = np.array(files)
    data_meta_info = parse_metainfo(files, learning_params, True)

    for i in range(len(files)):
        task_num = i + 1
        costs = []
        for _ in range(repeated_times):
            _files = files[np.random.choice(len(files), task_num, replace=False)]
            assert len(set(_files)) == task_num
            ts = time.time()
            raw_data = load_raw_data(_files, learning_params, verbose=False)
            t_load = time.time() - ts
            
            ts = time.time()
            train_raw_data, val_raw_data = raw_data.gen_train_test_data()
            t_part = time.time() - ts
            
            train_raw_data.metainfo = data_meta_info
            ts = time.time()
            train_raw_data.preprocess(time_lb=learning_params["ave_lb"], verbose=False)
            t_preprocess = time.time() - ts

            ts = time.time()
            train_raw_data.shuffle()
            t_shuffle = time.time() - ts

            ts = time.time()
            train_raw_data.freeze()
            t_freeze = time.time() - ts

            ts = time.time()
            train_raw_data.normalize()
            t_norm = time.time() - ts

            costs.append([t_load, t_part, t_preprocess, t_shuffle, t_freeze, t_norm])
        
        costs = np.array(costs)
        mean_costs = np.average(costs, axis=0)
                
        print(f"Cost of loading, partition, preprocessing, shuffle, freeze and normalization for {task_num} task files: {list(mean_costs)}")

def test_load_raw_data_freeze_cost(files, learning_params):
    ''' After finding `raw_data.freeze()` is the bottleneck to load data, further breakdown it
    '''
    repeated_times = 10
    files = np.array(files)
    data_meta_info = parse_metainfo(files, learning_params, True)

    for i in range(len(files)):
        task_num = i + 1
        costs = []
        for _ in range(repeated_times):
            _files = files[np.random.choice(len(files), task_num, replace=False)]
            assert len(set(_files)) == task_num
            raw_data = load_raw_data(_files, learning_params, verbose=False)
            train_raw_data, val_raw_data = raw_data.gen_train_test_data()
            train_raw_data.metainfo = data_meta_info
            train_raw_data.preprocess(time_lb=learning_params["ave_lb"], verbose=False)
            train_raw_data.shuffle()

            ### Start to break down
            ts = time.time()
            train_raw_data.freeze()
            t_freeze = time.time() - ts
            ### End of break down

            train_raw_data.normalize()

            costs.append([t_freeze])
        
        costs = np.array(costs)
        mean_costs = np.average(costs, axis=0)
                
        print(f"Cost of generate x, y, di for {task_num} task files: {list(mean_costs)}")

    


