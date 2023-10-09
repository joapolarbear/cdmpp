''' Analyze task files grouped by networks
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from metalearner.data.rawdata import load_raw_data
from utils.util import fig_base

def draw_box_plot(network2files: dict, learning_params):
    '''
    Parameters
    ----------
    network2files: dict
        A mapping between networks and corresponding sampled task files
    '''
    networks = list(network2files.keys())
    all_data_by_net = [[], []]

    all_avg = np.empty(0, dtype=float)
    all_flop_ct = np.empty(0, dtype=float)

    for _network in networks:
        raw_data = load_raw_data(network2files[_network], 
            learning_params, force=False, verbose=False)

        ### Not freeze first

        avg = raw_data.raw_data[:, 0]
        flop_ct = raw_data.raw_data[:, 2]
        all_data_by_net[0].append(avg)
        all_data_by_net[1].append(flop_ct)

        ### Accumulate the data
        all_avg = np.concatenate((all_avg, avg), axis=0)
        all_flop_ct = np.concatenate((all_flop_ct, flop_ct), axis=0)
    
    networks.append("ALL")
    all_data_by_net[0].append(all_avg)
    all_data_by_net[1].append(all_flop_ct)

    valid_idxs = [0, 1, 3, 7, 9, 10, 11, 12, 13]
    networks = np.array(networks)[valid_idxs]
    all_data_by_net[0] = np.array(all_data_by_net[0])[valid_idxs]
    all_data_by_net[1] = np.array(all_data_by_net[1])[valid_idxs]

    ### Visualize the results
    fig = plt.figure(figsize=(12, 6))
    _fig_base = fig_base(2, row_first=True)
    print(_fig_base)

    _fig_base += 1
    ax = fig.add_subplot(_fig_base)
    ax.boxplot(all_data_by_net[0], labels=networks)
    plt.xlabel("Distribution of Execution Time")
    plt.xticks(rotation=60)

    _fig_base += 1
    ax = fig.add_subplot(_fig_base)
    ax.boxplot(all_data_by_net[1], labels=networks)
    plt.xlabel("Distribution of Flop_ct")
    plt.xticks(rotation=60)

    plt.tight_layout()
    plt.savefig("tmp/network2files_analyze.png")

def dist_diff(network2files: dict, learning_params):
    '''
    Parameters
    ----------
    network2files: dict
        A mapping between networks and corresponding sampled task files
    '''
    from utils.metrics import metric_cmd, KMeansDiff, metric_mmd

    networks = list(network2files.keys())
    network_indices = []
    X = np.empty((0, 164), dtype=float)

    print("Load data and freeze ... ")
    for _network in tqdm(networks):
        raw_data = load_raw_data(network2files[_network], 
            learning_params, force=False, verbose=False)

        ### Not freeze first
        raw_data.freeze()

        _feature = raw_data.x.reshape((-1, 164))
        network_indices.append(np.arange(len(X), len(X)+len(_feature)))
        X = np.concatenate((X, _feature), axis=0)
    
    kmean_diff = KMeansDiff(X, k=100)

    for _id in range(len(networks)):
        _network = networks[_id]
        indices_in_X = network_indices[_id]
        
        if len(indices_in_X) == 0:
            print(f"Network {_network}, no data")
        else:
            mask = np.ones(len(X), dtype=bool)
            mask[indices_in_X] = False
            train_set = X[mask]
            test_set = X[indices_in_X]

            cmd = metric_cmd(train_set, test_set)
            # mmd = metric_mmd(train_set, test_set)
            kmeans = kmean_diff.diff2set(train_set, test_set)
            print(f"Network {_network}, diff2all, cmd={cmd}, kmeans={kmeans}")

def entry(feature_root_path: str, learning_params):
    from utils.util import _sample_task_files

    mode = learning_params["mode"]
    gpu_model = learning_params["gpu_model"]

    if "." in mode:
        data_source_mode, data_split_mode = mode.split(".")
    else:
        data_source_mode = mode
        data_split_mode = None

    ### Get tasks and corresponding partition attr
    for _mode in data_source_mode.split(","):
        if ":" in _mode:
            _device, _mode_detail = _mode.split(":")
        else:
            _device = gpu_model
            _mode_detail = _mode
        network2files = {}
        root_path, files_to_test = _sample_task_files(os.path.join(feature_root_path,
            _device), _mode_detail, absolute_path=True, network2files=network2files)
        
        # draw_box_plot(network2files, learning_params)
        dist_diff(network2files, learning_params)
