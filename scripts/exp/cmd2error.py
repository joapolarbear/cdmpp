''' Analyze the training results, i.e., analyze the relationship
    between domain difference and the test error
'''
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import pickle

from utils.util import (TrainTestPair, fig_base, 
    retrieve_partition_data_rst, retrieve_latent_repr)

from utils.metrics import (
    metric_cmd,
    metric_mmd,
    centroid_diff,
    metric_mape,
    metric_elementwise_mape,
    l2diff
)

fontsize = 36

def reduce_tick_num(num, axis="y"):
    if axis == "y":
        locs, labels = plt.yticks()
    else:
        locs, labels = plt.xticks()
    _min = min(locs)
    _max = max(locs)
    _mid = (_min + _max) / 2
    _range = (_max - _min)
    low = _mid - 1.1 * _range / 2
    high = _mid + 1.3 * _range / 2
    new_locs = np.arange(low, high, step=(high-low)/float(num))
    # new_ticks = (new_locs / 1e4).astype(int)
    if axis == "y":
        plt.yticks(new_locs, fontsize=fontsize-2)
    else:
        plt.xticks(new_locs, fontsize=fontsize-2)

def get_mape_distribution_shift(rst_path_pair, latent_path_pair):
    import time
    np.random.seed(int(time.time()))

    learn_rst = TrainTestPair(
        retrieve_partition_data_rst(rst_path_pair.train),
        retrieve_partition_data_rst(rst_path_pair.val)
    )
    latent_pair = TrainTestPair(
        retrieve_latent_repr(latent_path_pair.train),
        retrieve_latent_repr(latent_path_pair.val))

    X_tir_train = learn_rst.train["X"]
    X_tir_val = learn_rst.val["X"]
    Z_tir_train = latent_pair.train
    Z_tir_val = latent_pair.val

    if False:
        repr_train, repr_val = X_tir_train, X_tir_val
    else:
        repr_train, repr_val = Z_tir_train, Z_tir_val

    repr_train = repr_train.reshape(repr_train.shape[0], -1)
    repr_val = repr_val.reshape(repr_val.shape[0], -1)

    print("\nDomain differene analysis: x-axis=F_metric(D_train, D_test'), y-axis=E_test' ...")

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
        _X_repr_train = repr_train[np.random.choice(len(repr_train), select_num, replace=False)]
        cmd = metric_cmd(_X_repr_train, repr_val[pick_idx])
        # mmd = metric_mmd(_X_repr_train, repr_val[pick_idx])
        centroid_distance = centroid_diff(_X_repr_train, repr_val[pick_idx])
        mape2dist_metrics.append((
            mape, cmd,
            # mmd,
            centroid_distance))
    mape2dist_metrics = np.array(mape2dist_metrics)
    return mape2dist_metrics

def visualize_mape_to_distribution_shift(mape2dist_metrics, trial=0):

    ### Visualize
    X_axis_names = [
        "CMD", 
        # "MMD", 
        "Centroid Distance"
    ]

    def fit_func(x, a, b):
        return a * x + b

    clrs = sns.color_palette("husl", 5)
    error = mape2dist_metrics[:, 0] * 100
    for i in range(len(X_axis_names)):
        metric = mape2dist_metrics[:, i+1]
        fig = plt.figure(figsize=(12, 6))

        plt.scatter(metric, error, s=50)

        ### plot the fit results
        popt, pcov = curve_fit(fit_func, metric, error)
        perr = np.sqrt(np.diag(pcov))
        x_axis = np.linspace(np.min(metric), np.max(metric), 10)
        plt.plot(x_axis, fit_func(x_axis, *popt), 'r', linewidth=3, label="Smoothed")
        plt.fill_between(x_axis, fit_func(x_axis, *popt+perr),
            fit_func(x_axis, *popt-perr), alpha=0.3, facecolor=clrs[1])

        plt.xlabel(X_axis_names[i], fontsize=fontsize)
        plt.ylabel("Test Error (%)", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        reduce_tick_num(5)

        plt.tight_layout()
        # plt.savefig(f"./tmp/{X_axis_names[i].lower()}2error.pdf")
        plt.savefig(f"./tmp/{X_axis_names[i].lower().replace(' ', '_')}2error_{trial}.png")
        plt.close()

def visualize_train_test_distribution(cache_path, save_fig_path):
    data_embed_path = os.path.join(cache_path, "data_dim_reduct.pkl")
    if False and os.path.exists(data_embed_path):
        with open(data_embed_path, 'rb') as fp:
            train_len, val_len, data_all = pickle.load(fp)
    else:
        latent_path_pair = TrainTestPair(
            os.path.join(cache_path, "training_latent.pickle"),
            os.path.join(cache_path, "test_latent.pickle"))
        latent_pair = TrainTestPair(
            retrieve_latent_repr(latent_path_pair.train),
            retrieve_latent_repr(latent_path_pair.val))
        
        repr_train = latent_pair.train
        repr_val = latent_pair.val
        repr_train = repr_train.reshape(repr_train.shape[0], -1)
        repr_val = repr_val.reshape(repr_val.shape[0], -1)

        # For reproducability of the results
        np.random.seed(42)
        rndperm = np.random.permutation(len(repr_train))
        max_sample_num=1000

        data_train = repr_train[rndperm[:max_sample_num]]
        data_val = repr_val[:max_sample_num]

        train_len = len(data_train)
        val_len = len(data_val)

        data_all = np.concatenate((data_train, data_val), axis=0)
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init='random', verbose=1, perplexity=40, n_iter=300, random_state=0)
        data_all = tsne.fit_transform(data_all)

        with open(data_embed_path, 'wb') as fp:
            pickle.dump([train_len, val_len, data_all], fp)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.grid(True)
    plt.scatter(
        data_all[0:train_len, 0],
        data_all[0:train_len, 1],
        alpha=0.5,
        # edgecolors='none',
        s=100,
        label="Samples in the source networks"
    )
    plt.scatter(
        data_all[train_len:(train_len+val_len), 0],
        data_all[train_len:(train_len+val_len), 1],
        alpha=0.5,
        # edgecolors='none',
        s=100,
        label="Samples in the target network"
    )
    plt.xlabel('Dimension 0', fontsize=fontsize)
    plt.ylabel('Dimension 1', fontsize=fontsize)
    reduce_tick_num(4)
    reduce_tick_num(4, axis='x')
    plt.legend(fontsize=fontsize-2)
    plt.tight_layout()
    plt.savefig(save_fig_path)

def visualize_cross_device_distribution(cache_path, save_fig_path):
    from sklearn.manifold import TSNE

    data_embed_path = os.path.join(cache_path, "data_dim_reduct.pkl")
    if False and os.path.exists(data_embed_path):
        with open(data_embed_path, 'rb') as fp:
            train_len, val_len, data_all = pickle.load(fp)
    else:
        max_sample_num=1000
        devices = []
        data_range = []
        data_all = None 
        for _file in os.listdir(cache_path):
            if not (_file.startswith("train_latent") and _file.endswith(".pickle")):
                continue
            device = _file.split(".")[0][len("train_latent_"):]
            repr_train = retrieve_latent_repr(os.path.join(cache_path, _file))
            repr_train = repr_train.reshape(repr_train.shape[0], -1)

            # For reproducability of the results
            np.random.seed(0)
            rndperm = np.random.permutation(len(repr_train))
            data_train = repr_train[rndperm[:max_sample_num]]

            if data_all is None:
                data_range.append((0, len(data_train)))
                data_all = data_train
            else:
                data_range.append((len(data_all), len(data_all)+len(data_train)))
                data_all = np.concatenate((data_all, data_train), axis=0)
            
            devices.append(device)
            
        tsne = TSNE(n_components=2, init='random', verbose=1, perplexity=40, n_iter=300, random_state=0)
        data_all = tsne.fit_transform(data_all)

        with open(data_embed_path, 'wb') as fp:
            pickle.dump([devices, data_range, data_all], fp)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.grid(True)
    for device_id, device in enumerate(devices):
        st, ed = data_range[device_id]
        plt.scatter(
            data_all[st:ed, 0],
            data_all[st:ed, 1],
            alpha=0.5,
            # edgecolors='none',
            s=100,
            label=device.upper()
        )
    plt.xlabel('Dimension 0', fontsize=fontsize)
    plt.ylabel('Dimension 1', fontsize=fontsize)
    reduce_tick_num(4)
    reduce_tick_num(4, axis='x')
    plt.legend(fontsize=fontsize-2)
    plt.tight_layout()
    plt.savefig(save_fig_path)

if __name__ == "__main__":
    option = sys.argv[1]
    cache_path = sys.argv[2]
    save_fig_path = sys.argv[3]
    if option == "cmd2error":
        rst_path_pair = TrainTestPair(
            os.path.join(cache_path, "training.pickle"),
            os.path.join(cache_path, "test.pickle"))
        latent_path_pair = TrainTestPair(
            os.path.join(cache_path, "training_latent.pickle"),
            os.path.join(cache_path, "test_latent.pickle"))
        assert all([
            os.path.exists(rst_path_pair.train), os.path.exists(rst_path_pair.val),
        # os.path.exists(latent_path_pair.train), os.path.exists(latent_path_pair.val)
        ]), (f"Doesn't find training and test set under {cache_path}. "
            "Please dump traing and test set by runing `bash script/train.sh analyze` first")
        
        for i in range(10):
            mape2dist_metrics = get_mape_distribution_shift(rst_path_pair, latent_path_pair)
            visualize_mape_to_distribution_shift(mape2dist_metrics, trial=i)
    elif option == "cm_finetune":
        visualize_train_test_distribution(cache_path, save_fig_path)
    elif option == "cd_finetune":
        visualize_cross_device_distribution(cache_path, save_fig_path)
    else:
        raise