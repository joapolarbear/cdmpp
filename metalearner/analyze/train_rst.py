
import os
import numpy as np
import math
import json

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from tqdm import tqdm

import torch

from utils.util import (fig_base, TrainTestPair,
    retrieve_partition_data_rst, retrieve_latent_repr)
from utils.metrics import (
    metric_cmd,
    metric_mmd,
    centroid_diff,
    metric_mape,
    metric_elementwise_mape,
    l2diff
)

from metalearner.data.dataloader import MyDataSet
from metalearner.data.rawdata import load_raw_data

font = {"color": "darkred",
        "size": 13,
        "family": "serif"}

INVALID_ENTRIES = [0, 3, 6, 7, 12, 13, 15, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 94, 102, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146]

def distance_to_train(train_space, test_space, test_idx, distance_func):
    ''' For a specific test sample, find the cloest samples in the training space,
    Return the distance of the found sample and the test sample
    
    Parameters
    ---------
    train_space: np.ndarray of shape (N_sample, N_entry)
        Training space of input feature or latent representations
    test_space: np.ndarray of shape (N_sample, N_entry)
        Test sapce
    test_idx: int
        The index of target test samples in the test space
    distance_func: func(repr1, repr2):
        The function to measure the distance between two representations

    Returns
    -------
    distance: int
        The distance of the found sample and the test sample
    '''
    test_repr = test_space[test_idx]
    distance = math.inf
    for idx in range(train_space.shape[0]):
        train_repr = train_space[idx]
        _dist = distance_func(train_repr, test_repr)
        if _dist < distance:
            distance = _dist
    return distance

def shape_convert(_X):
    if len(_X.shape) == 2:
        return _X
    elif len(_X.shape) == 3:
        return np.sum(_X, axis=1)

def analyze_train_test_rst(rst_path_pair, latent_path_pair):
    ''' Analyze the training results with dumped training and test dataset
    '''
    from prettytable import PrettyTable

    learn_rst = TrainTestPair(
        retrieve_partition_data_rst(rst_path_pair.train),
        retrieve_partition_data_rst(rst_path_pair.val)
    )
    latent_pair = TrainTestPair(
        retrieve_latent_repr(latent_path_pair.train),
        retrieve_latent_repr(latent_path_pair.val))

    # monitor = Monitor(debug=True)
    # monitor.add_monitor(learn_rst.val)
    # monitor.visual(".")

    METRIC_DICT = {
        "CMD": metric_cmd,
        "centroid_distance": centroid_diff
    }
    
    if False:
        ### Calculate CMD and other metrics for D_train, D_test, D_best_test and D_worst_test
        print(f"Training datasize = {len(learn_rst.train['Y'])}")
        print(f"Test datasize = {len(learn_rst.val['Y'])}")
        print(f"Training error = {learn_rst.train['MAPE']}")
        print(f"Test error = {learn_rst.val['MAPE']}")
        print(f"Test error best = {learn_rst.val['MAPE_of_Best_100']}")
        print(f"Test error wost = {learn_rst.val['MAPE_of_Worst_100']}")
        X_train = learn_rst.train["X"].reshape(learn_rst.train["X"].shape[0], -1)
        X_val = learn_rst.val["X"].reshape(learn_rst.val["X"].shape[0], -1)
        for metric_name in sorted(METRIC_DICT.keys()):
            print(f"\n ########### Metric {metric_name} ###########")
            metric_func = METRIC_DICT[metric_name]
            rows = []
            rows.append(["", "Train2Test", "Train2BestTest", "Train2WorstTest"])
            rows.append([
                "Input X",
                metric_func(X_train, X_val),
                metric_func(X_train, X_val[learn_rst.val["best_100_idxs"]]),
                metric_func(X_train, X_val[learn_rst.val["worst_100_idxs"]])
            ])
            if latent_pair.train is not None and latent_pair.val is not None:
                for latent_name, latent_train in latent_pair.train.items():
                    latent_val = latent_pair.val[latent_name]
                    rows.append([
                        f"{latent_name}",
                        metric_func(latent_train, latent_val),
                        metric_func(latent_train, latent_val[learn_rst.val["best_100_idxs"]]),
                        metric_func(latent_train, latent_val[learn_rst.val["worst_100_idxs"]])
                    ])
            
            rows = np.array(rows).T.tolist()
            table = PrettyTable()
            table.field_names = rows[0]
            table.add_rows(rows[1:])
            print(table)

    if True:
        print("\nDomain differene analysis: x-axis=F_metric(D_train, D_test'), y-axis=E_test' ...")
        X_train = learn_rst.train["X"].reshape(learn_rst.train["X"].shape[0], -1)
        X_val = learn_rst.val["X"].reshape(learn_rst.val["X"].shape[0], -1)

        ### Sample a subset from D_test, D', of the same size as D_worst_test and D_best_test
        n_samples = len(learn_rst.val["Y_pred"])
        repeat_times = 100
        select_num = n_samples // repeat_times
        print(f"Sample {select_num}/{n_samples} test samples for {repeat_times} times ... ")

        val_ape = metric_elementwise_mape(learn_rst.val["Y_pred"], learn_rst.val["Y"])
        sorted_val_idx = np.argsort(val_ape)

        mape2dist_metrics = []
        for i in tqdm(range(repeat_times)):
            # pick_idx = np.random.choice(n_samples, math.ceil(select_num), replace=False)
            pick_idx = sorted_val_idx[(i*select_num):((i+1)*select_num)]
            Y_pred = learn_rst.val["Y_pred"][pick_idx]
            Y = learn_rst.val["Y"][pick_idx]
            mape = metric_mape(Y_pred, Y)
            cmd = metric_cmd(X_train, X_val[pick_idx])
            # mmd = metric_mmd(X_train, X_val[pick_idx])
            centroid_distance = centroid_diff(X_train, X_val[pick_idx])
            mape2dist_metrics.append((
                mape, cmd,
                # mmd,
                centroid_distance))
        mape2dist_metrics = np.array(mape2dist_metrics)

        # print(mape2dist_metrics)

        ### Visualize
        fig = plt.figure(figsize=(12, 5))
        _figbase = fig_base(4)
        X_axis_names = ["CMD", 
        # "MMD", 
        "Centroid Distance"]

        def fit_func(x, a, b):
            return a * x + b

        clrs = sns.color_palette("husl", 5)
        error = mape2dist_metrics[:, 0]
        for i in range(len(X_axis_names)):
            _figbase += 1
            ax = fig.add_subplot(_figbase)
            metric = mape2dist_metrics[:, i+1]
            ax.scatter(metric, error)

            ### plot the fit results
            popt, pcov = curve_fit(fit_func, metric, error)
            perr = np.sqrt(np.diag(pcov))
            x_axis = np.linspace(np.min(metric), np.max(metric), 10)
            ax.plot(x_axis, fit_func(x_axis, *popt), 'r')
            ax.fill_between(x_axis, fit_func(x_axis, *popt+perr),
                fit_func(x_axis, *popt-perr), alpha=0.3, facecolor=clrs[1])

            plt.xlabel(X_axis_names[i])
            plt.ylabel("Test Error")
        
        plt.tight_layout()
        plt.savefig("./tmp/cmd2error.png")
        plt.close()
        exit(0)

    if False:
        worker_dir = ".workspace/sensitive_analyze"
        os.makedirs(worker_dir, exist_ok=True)
        rst_path = os.path.join(worker_dir, "rst.json")
        if True:
            target_idx = [2, 5, 11, 14, 53, 57, 59, 66, 69, 70, 77, 82, 88, 89, 91, 98, 108, 147, 148, 149, 152, 153, 154, 156, 162]
            distance_weight = np.zeros(learn_rst.train["X"].shape[1])
            distance_weight[np.array(target_idx)] = 1
        elif os.path.exists(rst_path):
            with open(rst_path, 'r') as fp:
                sensitive_ana_rst = json.load(fp)
            influence_vector = None
            for norm_method, method_dict in sensitive_ana_rst.items():
                for noise_scale, _vector in method_dict.items():
                    if influence_vector is None:
                        influence_vector = np.array(_vector)
                    else:
                        influence_vector += _vector
            assert influence_vector is not None

            def stable_softmax(x):
                '''https://www.delftstack.com/zh/howto/numpy/numpy-softmax/'''
                y = np.exp(x - np.max(x))
                f_x = y / np.sum(np.exp(x))
                return f_x

            distance_weight = stable_softmax(influence_vector) 
        else:
            distance_weight = np.ones(learn_rst.train["X"].shape[1])
        ### Calculate Minimun-distance2D_train
        def weightedL2(a, b):
            q = a-b
            return np.sqrt((distance_weight*q*q).sum())
        
        elementwise_mape = metric_elementwise_mape(learn_rst.val["Y_pred"], learn_rst.val["Y"])
        rst_table = []
        first_ = True
        header = ["MAPE"]
        for val_idx in range(len(learn_rst.val["Y"])):
            mape = elementwise_mape[val_idx]
            row = [mape]
            if False:
                ### euclidean distance
                input_distance = distance_to_train(learn_rst.train["X"], learn_rst.val["X"], val_idx, l2diff)
                row.append(input_distance)
                if first_:
                    header.append("X")
            if True:
                ### with weight
                input_distance = distance_to_train(learn_rst.train["X"], learn_rst.val["X"], val_idx, weightedL2)
                row.append(input_distance)
                if first_:
                    header.append("Weighted_X")
            
            if False and latent_pair.train is not None and latent_pair.val is not None:
                for latent_name, latent_train in latent_pair.train.items():
                    if first_:
                        header.append(latent_name)
                    latent_val = latent_pair.val[latent_name]
                    latent_distance = distance_to_train(latent_train, latent_val, val_idx, l2diff)
                    row.append(latent_distance)

            rst_table.append(row)
            first_ = False
        print(header)
        rst_table = np.array(rst_table).T

        fig = plt.figure(figsize=(12, 6))
        _fig_base = fig_base(5, row_first=True)

        for idx in range(1, len(header)):
            ax = fig.add_subplot(_fig_base+idx)
            print(header[idx])
            ax.scatter(rst_table[0], rst_table[idx], label=str(header[idx][1:]),
                alpha=0.3, edgecolors='none')
            plt.xlabel("MAPE")
            plt.ylabel("Min Distance to Training X/Latent Repr")
            ax.legend()

        plt.savefig("tmp.png")

    if False:
        _find_cluster_metrics(learn_rst.train, learn_rst.val)

    if False:
        ### Compare X_test[i] to X_train one by one
        # TODO (delete), since those are distribution-to-distribution matrics
        mape_metric = []
        for i in range(len(learn_rst.val["Y"])):
            y = learn_rst.val["Y"][i]
            y_pred = learn_rst.val["Y_pred"][i]
            mape = np.abs(y - y_pred) / y
            _metrics = []
            for metric_name in sorted(METRIC_DICT.keys()):
                metric_func = METRIC_DICT[metric_name]
                _metric = metric_func(learn_rst.train["X"], learn_rst.val["X"][i])
                _metrics.append(_metric)
            mape_metric.append((mape, _metrics))
        mape_list, metrics_list = zip(*sorted(mape_metric, key=lambda x: x[0]))
        metrics_list = np.array(metrics_list)
        heads = list(sorted(METRIC_DICT.keys()))

        fig, axs = plt.subplots(4, 1)
        for idx, ax in enumerate(axs):
            ax.plot(mape_list, metrics_list[:, idx], label=heads[idx])
            ax.legend()

    if True:
        ### Visualize the distribution of Y_train and Y_val
        fig = plt.figure(figsize=(12, 10))
        _fig_base = fig_base(9, row_first=True)

        def metric_elementwise_se(Y_pred, Y):
            return np.power(Y_pred-Y, 2)
        
        def metric_elementwise_ae(Y_pred, Y):
            return Y_pred-Y

        train_Y = learn_rst.train["Y"]
        train_Y_pred = learn_rst.train["Y_pred"]
        train_ape = metric_elementwise_mape(train_Y_pred, train_Y)
        train_ae = metric_elementwise_ae(train_Y_pred, train_Y)

        val_Y = learn_rst.val["Y"]
        val_Y_pred = learn_rst.val["Y_pred"]
        val_ape = metric_elementwise_mape(val_Y_pred, val_Y)
        val_ae = metric_elementwise_ae(val_Y_pred, val_Y)

        _fig_base += 1
        ax = fig.add_subplot(_fig_base) 
        plt.title("Distribution of Y")
        ax.boxplot(
            [train_Y, val_Y, val_Y_pred, 
            val_Y_pred[learn_rst.val["best_100_idxs"]],
            val_Y_pred[learn_rst.val["worst_100_idxs"]]],
            labels=["Y_train", "Y_test", "Y_test_pred", "Y_test_pred_best", "Y_test_pred_worst"]
        )

        _fig_base += 1
        ax = fig.add_subplot(_fig_base) 
        plt.title("Train: Y to Y'")
        ax.scatter(train_Y, train_Y_pred, label="Train", alpha=0.3)
        ideal = np.linspace(min(train_Y), max(train_Y), num=5)
        ax.plot(ideal, ideal, 'r', label="linear")
        plt.xlabel("Y")
        plt.ylabel("Y_pred")
        plt.legend()

        _fig_base += 1
        ax = fig.add_subplot(_fig_base) 
        plt.title("Test: Y to Y'")
        ax.scatter(val_Y, val_Y_pred, label="Test", alpha=0.3)
        ideal = np.linspace(min(val_Y), max(val_Y), num=5)
        ax.plot(ideal, ideal, 'r', label="linear")
        plt.xlabel("Y")
        plt.ylabel("Y_pred")
        plt.legend()

        ##############

        _fig_base += 1
        ax = fig.add_subplot(_fig_base) 
        plt.title("Train: Histogram of Y-APE")
        x_min, x_max = np.min(train_Y), np.percentile(train_Y, 75)
        y_min, y_max = np.min(train_ape), np.percentile(train_ape, 90)
        x_bins = np.linspace(x_min, x_max, 100)
        y_bins = np.linspace(y_min, y_max, 100)
        plt.hist2d(train_Y, train_ape, bins =[x_bins, y_bins], cmap = plt.cm.nipy_spectral)
        plt.colorbar()
        ax.set_xlabel('Y') 
        ax.set_ylabel('APE') 

        # _fig_base += 1
        # ax = fig.add_subplot(_fig_base) 
        # plt.title("Train: Y to Absolute Percentage Error")
        # ax.scatter(train_Y, train_ape, label="Train", alpha=0.3)
        # plt.xlabel("Y")
        # plt.ylabel("APE")
        # plt.ylim(0, 5)
        # plt.legend()

        _fig_base += 1
        ax = fig.add_subplot(_fig_base) 
        plt.title("Test: Histogram of Y-APE")
        x_min, x_max = np.min(val_Y), np.percentile(val_Y, 75)
        y_min, y_max = np.min(val_ape), np.percentile(val_ape, 90)
        x_bins = np.linspace(x_min, x_max, 100)
        y_bins = np.linspace(y_min, y_max, 100)
        plt.hist2d(val_Y, val_ape, bins =[x_bins, y_bins], cmap = plt.cm.nipy_spectral)
        plt.colorbar()
        ax.set_xlabel('Y') 
        ax.set_ylabel('APE') 

        # _fig_base += 1
        # ax = fig.add_subplot(_fig_base) 
        # plt.title("Test: Y to Absolute Percentage Error")
        # ax.scatter(val_Y, val_ape, label="Test", alpha=0.3)
        # plt.xlabel("Y")
        # plt.ylabel("APE")
        # plt.ylim(0, 5)
        # plt.legend()

        _fig_base += 1
        ax = fig.add_subplot(_fig_base) 
        plt.title("Train: Histogram of Y-AE")
        x_min, x_max = np.min(train_Y), np.percentile(train_Y, 75)
        y_min, y_max = np.percentile(train_ae, 10), np.percentile(train_ae, 90)
        x_bins = np.linspace(x_min, x_max, 100)
        y_bins = np.linspace(y_min, y_max, 100)
        plt.hist2d(train_Y, train_ae, bins =[x_bins, y_bins], cmap = plt.cm.nipy_spectral)
        plt.colorbar()
        ax.set_xlabel('Y') 
        ax.set_ylabel('AE') 

        # _fig_base += 1
        # ax = fig.add_subplot(_fig_base) 
        # plt.title("Train: Y to Absolute Error")
        # ax.scatter(train_Y, train_ae, label="Train", alpha=0.3)
        # plt.xlabel("Y")
        # plt.ylabel("Absolute Error (s)")
        # plt.legend()

        _fig_base += 1
        ax = fig.add_subplot(_fig_base) 
        plt.title("Test: Histogram of Y-APE")
        x_min, x_max = np.min(val_Y), np.percentile(val_Y, 75)
        y_min, y_max = np.percentile(val_ae, 10), np.percentile(val_ae, 90)
        x_bins = np.linspace(x_min, x_max, 100)
        y_bins = np.linspace(y_min, y_max, 100)
        plt.hist2d(val_Y, val_ae, bins =[x_bins, y_bins], cmap = plt.cm.nipy_spectral)
        plt.colorbar()
        ax.set_xlabel('Y') 
        ax.set_ylabel('APE') 

        # _fig_base += 1
        # ax = fig.add_subplot(_fig_base) 
        # plt.title("Test: Y to Absolute Error")
        # ax.scatter(val_Y, val_ae, label="Test", alpha=0.3)
        # plt.xlabel("Y")
        # plt.ylabel("Absolute Error (s)")
        # plt.legend()

        #######
        _fig_base += 1
        ax = fig.add_subplot(_fig_base) 
        plt.scatter(val_ape, val_ae, alpha=0.3
        )
        plt.xlabel("APE")
        plt.ylabel("Absolute Error")
        plt.tight_layout()
        plt.savefig("tmp.png")
        plt.close()
            
    if False:
        ### Check the effect of each feature entry on the error
        X_train = shape_convert(learn_rst.train["X"])
        X_val = shape_convert(learn_rst.val["X"])
        X_val_best = X_val[learn_rst.val["best_100_idxs"]]
        X_val_worst = X_val[learn_rst.val["worst_100_idxs"]]

        def __plot(_input):
            fig = plt.figure(figsize=(12, 5))
            _base = fig_base(len(_input))
            for id, (train_data, test_data, test_best, test_worst, _name) in enumerate(_input):
                ax = fig.add_subplot(_base + id + 1)
                # ax.scatter(test_data, test_y, alpha=0.1, edgecolors='none', label="Test")
                # ax.scatter(train_data, train_y, alpha=0.1, edgecolors='none', label="Train")
                # plt.legend()
                
                ax.boxplot([test_data, test_best, test_worst, train_data], labels=["Test", "Test Best", "Test Worst", "Train"])
                plt.ylabel("Percentile")

                plt.xlabel(_name)
                
            plt.tight_layout()
            plt.savefig(os.path.join(f"x2y.png"))
            plt.close()

        _input = [] 
        for i in range(X_val.shape[1]):
            if i in INVALID_ENTRIES:
                continue
            _input.append((X_train[:, i], X_val[:, i], X_val_best[:, i], X_val_worst[:, i], f"x[{i}]"))
            if len(_input) >= 9:
                __plot(_input)
                _input = []
                input("continue")
        if len(_input) > 0:
            __plot(_input)
        exit()

    if False:
        ''' Feature entry based filter, filter out Test samples based on feature entry distribution in the Training dataset
        '''
        X_train = shape_convert(learn_rst.train["X"])
        X_val = shape_convert(learn_rst.val["X"])

        train_x_percentile_by_feature = (
            np.percentile(X_train, 5, axis=0),
            np.percentile(X_train, 95, axis=0),
        )
        assert train_x_percentile_by_feature[0].shape == X_train[0].shape
        in_valide_feature_entry = [i for i in range(len(train_x_percentile_by_feature[0])) 
            if train_x_percentile_by_feature[0][i] == train_x_percentile_by_feature[1][i]]

        pick_idx = np.ones(len(X_val), dtype=bool)
        for _idx in range(len(X_val)):
            for entry_id, test_entry in enumerate(X_val[_idx]):
                if entry_id in in_valide_feature_entry:
                    continue
                if test_entry < train_x_percentile_by_feature[0][entry_id] or \
                    test_entry > train_x_percentile_by_feature[1][entry_id]:
                    pick_idx[_idx] = False
                    break
        Y_pred = learn_rst.val["Y_pred"][pick_idx]
        Y = learn_rst.val["Y"][pick_idx]
        mape = metric_mape(Y_pred, Y)
        print(f"Feature Entry based filter, MAPE: {mape}")

        pick_idx = np.ones(len(X_val), dtype=bool)
        pick_idx[learn_rst.val["worst_100_idxs"]] = False
        Y_pred = learn_rst.val["Y_pred"][pick_idx]
        Y = learn_rst.val["Y"][pick_idx]
        mape = metric_mape(Y_pred, Y)
        print(f"Remove worst 100 samples, MAPE: {mape}")

    if False:
        ''' Define a metric based on the angle between two line segments connecting $X^{test}$ and two nearest samples in $D_{train}$
        '''
        X_train = shape_convert(learn_rst.train["X"])
        X_val = shape_convert(learn_rst.val["X"])
        print("Calculating the new metric ...")
        test2cos_theta = []
        test2min_dist = []
        test2mean_dist = []
        for test_sample_id in tqdm(range(len(X_val))):
            ### second smallest and the smallest
            smallest_X_train = [None, None]
            smallest_distance = [None, None]
            dist_sum = 0
            for train_sample_id in range(len(X_train)):
                distance = l2diff(X_val[test_sample_id], X_train[train_sample_id])
                dist_sum += distance
                if smallest_X_train[0] is None:
                    if smallest_X_train[1] is None:
                        smallest_X_train[1] = X_train[train_sample_id]
                        smallest_distance[1] = distance
                    elif distance < smallest_distance[1]:
                        smallest_X_train[0] = smallest_X_train[1]
                        smallest_distance[0] = smallest_distance[1]
                        smallest_X_train[1] = X_train[train_sample_id]
                        smallest_distance[1] = distance
                else:
                    if distance < smallest_distance[1]:
                        smallest_X_train[0] = smallest_X_train[1]
                        smallest_distance[0] = smallest_distance[1]
                        smallest_X_train[1] = X_train[train_sample_id]
                        smallest_distance[1] = distance
                    elif distance < smallest_distance[0]:
                        smallest_X_train[0] = X_train[train_sample_id]
                        smallest_distance[0] = distance

            a = smallest_distance[0]
            b = smallest_distance[1]
            c = l2diff(smallest_X_train[0], smallest_X_train[1])
            metric = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
            test2cos_theta.append(metric)
            test2min_dist.append(b)
            test2mean_dist.append(dist_sum/len(X_train))
        
        Y_pred = learn_rst.val["Y_pred"]
        Y = learn_rst.val["Y"]
        elementwise_mape = metric_elementwise_mape(Y_pred, Y)

        fig = plt.figure(figsize=(12, 6))

        ax = fig.add_subplot(221)
        ax.scatter(elementwise_mape, test2cos_theta)
        plt.xlabel("MAPE")
        plt.ylabel("Cos(theta)")

        ax = fig.add_subplot(222)
        ax.scatter(elementwise_mape, test2min_dist)
        plt.xlabel("MAPE")
        plt.ylabel("Min Distance to D_train")

        ax = fig.add_subplot(223)
        ax.scatter(elementwise_mape, test2mean_dist)
        plt.xlabel("MAPE")
        plt.ylabel("Mean Distance to D_train")

        plt.tight_layout()
        plt.savefig("tmp2.png")

        
def _find_cluster_metrics(train_rst, test_rst):
    ''' Automatically find a metric for clustering based on the learning results '''
    X_train = train_rst["X"]
    X_test = test_rst["X"]
    X_test_best = X_test[test_rst["best_100_idxs"]]
    X_test_worst = X_test[test_rst["worst_100_idxs"]]

    def pairwise_distance(X1, X2=None):
        distance_arr = []
        if len(X1) > 1000:
            X1 = X1[:1000]
        # if X2 is not None and len(X2) > 1000:
        #     X2 = X2[:1000]
        for i in range(len(X1)):
            if X2 is None:
                ### Calculate pairwise distance intra X1
                for j in range(i+1, len(X1)):
                    distance_arr.append(np.abs(X1[i] - X1[j]))
            else:
                ### Calculate pairwise distance between X1 and X2
                for j in range(len(X2)):
                    distance_arr.append(np.abs(X1[i] - X1[j]))
        return np.array(distance_arr)

    class LinearMatric:
        def avg_pairwise_distance(self, X1, X2=None):
            distance_arr = pairwise_distance(X1, X2)
            return np.average(distance_arr, axis=0)
        def select_indicator_entries(self, k):
            ### Find entries where X_train-intra-dist and X_train-X_test_worst-inter-dist differ most
            d_train = self.avg_pairwise_distance(X_train)
            d_train_test_best = self.avg_pairwise_distance(X_train, X_test_best)
            d_train_test_worst = self.avg_pairwise_distance(X_train, X_test_worst)
            # print(d_train.shape)
            # print(d_train_test_best.shape)
            # print(d_train_test_worst.shape)
            # np.argsort(d_train)[::-1][:10]
            # d_train[np.argsort(d_train)[::-1][:10]]
            # np.argsort(d_train_test_best)[::-1][:10]
            # d_train_test_best[np.argsort(d_train_test_best)[::-1][:10]]
            # np.argsort(d_train_test_worst)[::-1][:10]
            # d_train_test_worst[np.argsort(d_train_test_worst)[::-1][:10]]
            # return np.argsort(d_train - d_train_test_worst)[:k]
            return np.arange(X_train.shape[1])[(d_train - d_train_test_worst) < 0]
        def weight(k):
            return np.arange(k)[::-1] / np.sum(np.arange(k))
        def construct_classfier(self, k):
            indicator_entries = self.select_indicator_entries(k)
            # weight = cf.weight(k)
            W = np.ones_like(indicator_entries)
            def metric_func(X):
                if len(X.shape) == 1:
                    return np.sum(X[indicator_entries] * W)
                elif len(X.shape) == 2:
                    return np.sum(X[:, indicator_entries] * W, axis=1)
                else:
                    raise ValueError(X.shape)
            metrics_train = metric_func(X_train)
            metrics_test_best = metric_func(X_test_best)
            metrics_test_worst = metric_func(X_test_worst)

            import code
            code.interact(local=locals())

            for metrics in [metrics_train, metrics_test_best, metrics_test_worst]:
                print(np.min(metrics), np.median(metrics), np.mean(metrics), np.max(metrics))

            train_di = np.mean(metrics_train)
            test_worst_di = np.mean(metrics_test_worst)
            def classfier(X):
                if len(X.shape) == 1:
                    X = X.reshape((1, len(X)))
                di = metric_func(X)
                return (np.abs(di - train_di) < np.abs(di - test_worst_di)).astype(int)
            return classfier
    
    class DisjointDistanceClassifier():
        def pairwise_distance_range(self, X1, X2=None):
            distance_arr = pairwise_distance(X1, X2)
            return np.max(distance_arr, axis=0), np.min(distance_arr, axis=0)
        def select_indicator_entries(self):
            ### Find entries where X_train-intra-dist and X_train-X_test_worst-inter-dist do not intersect
            ### Assume the latter is larger, --> entries X_train-X_test_worst-inter-dist.min > X_train-intra-dist.max
            range_train = self.pairwise_distance_range(X_train)
            range_train_test_best = self.pairwise_distance_range(X_train, X_test_best)
            range_train_test_worst = self.pairwise_distance_range(X_train, X_test_worst)

    Y_test = test_rst["Y"]
    Y_pred_test = test_rst["Y_pred"]
    elementwise_mape = np.abs(Y_test - Y_pred_test) / Y_test

    cf = LinearMatric()
    clsf = cf.construct_classfier(10)
    di_test = clsf(X_test)
    print(np.mean(elementwise_mape[di_test.astype(bool)]))
    print(np.mean(elementwise_mape[~di_test.astype(bool)]))

    import code
    code.interact(local=locals())

def multitask_train_rst(files_to_test, learner, learning_params):
    ''' Test a cross-task learner on each task, record the relationship between the test error
        average time cost and flop_ct
    '''
    error_list = []
    flops_list = []
    avg_Y_list = []
    for f in files_to_test:
        ### Evaluate the error for each task
        raw_data = load_raw_data([f], learning_params,
            force=False, verbose=True)
        avg_Y_list.append(raw_data.metainfo.output_avg)

        ### Flops
        flops_set = set(raw_data.flops)
        assert len(flops_set) == 1, f"Schedulings in {f} have different flops"
        flops = list(flops_set)[0]
        flops_list.append(flops)

        ### Preprocess
        raw_data.metainfo = learner.data_meta_info
        raw_data.preprocess(time_lb=learning_params["ave_lb"], verbose=True)
        dataset = MyDataSet(raw_data)
        dataset.check_leaf_no_dist()

        if len(dataset) == 0:
            error_list.append(-1)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            X, Y, di = next(iter(dataloader))
            X, Y = learner.data_to_train_device(X, Y)
            outputs, metrics, labels = learner.forward_compute_metrics((X, Y))
            error_list.append(metrics["mape"])
        print(f"Final dataset size (filter according to leaf node no): {len(dataset)}, flops: {flops}, error: {error_list[-1]}\n")

    error_list = np.array(error_list)
    files_to_test = np.array(files_to_test)

    ### Visualize the per-task evaluated results

    ### 1. Order task according to their per-task errors
    ordering = error_list.argsort()
    print(f"")
    print(f"Evaluate the cost model {learner.cache_path} on {len(files_to_test)} tasks respectively")
    rows = []
    header = ["Order"]
    top_n = min(len(files_to_test)//2, 10000)
    header += [f"{n}" for n in range(top_n)]
    header += [f"-{n}" for n in range(top_n, 0, -1)]
    rows.append(header)
    row = ["Task"]
    row += [f"{files_to_test[ordering[n]]}" for n in range(top_n)]
    row += [f"{files_to_test[ordering[-n]]}" for n in range(top_n, 0, -1)]
    rows.append(row)

    row = ["Error"]
    row += [f"{error_list[ordering[n]]}" for n in range(top_n)]
    row += [f"{error_list[ordering[-n]]}" for n in range(top_n, 0, -1)]
    rows.append(row)
    
    from prettytable import PrettyTable
    rows = np.array(rows).T.tolist()
    table = PrettyTable()
    table.field_names = rows[0]
    table.add_rows(rows[1:])
    print(table)

    ### 2. plot avg-error and flops-error figures
    valid_idxs = np.where(error_list != -1)[0]
    _error_list = error_list[valid_idxs]
    _flops_list = np.array(flops_list)[valid_idxs]
    _avg_Y_list = np.array(avg_Y_list)[valid_idxs]
    
    fig = plt.figure(figsize=(12, 6))
    _fig_base = fig_base(3, row_first=True)

    ax = fig.add_subplot(_fig_base+1) 
    plt.title("Avg time cost to Per-task Error", fontsize=16)
    plt.scatter(_avg_Y_list, _error_list)
    plt.xlabel("Average Time Cost", fontsize=16)
    plt.ylabel("Per-task Error", fontsize=16)
    
    ax = fig.add_subplot(_fig_base+2) 
    plt.title("# of Flops to Per-task Error", fontsize=16)
    plt.scatter(_flops_list, _error_list)
    plt.xlabel("# of flops", fontsize=16)
    plt.ylabel("Per-task Error", fontsize=16)

    ax = fig.add_subplot(_fig_base+3) 
    plt.title("# of Flops to Avg time cost ", fontsize=16)
    plt.scatter(_flops_list, _avg_Y_list)
    plt.xlabel("# of flops", fontsize=16)
    plt.ylabel("Average Time Cost", fontsize=16)

    plt.tight_layout()
    plt.savefig("./tmp/per-task-avg-flops-error.png")