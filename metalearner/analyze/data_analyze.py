import os
import numpy as np
import pandas as pd
import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from utils.util import fig_base

from metalearner.data.dataloader import load_raw_data_w_cluster_id
from metalearner.data.rawdata import load_raw_data, sample_data_around_avg, RawData


def check_data_distribution(xydata: RawData, path=None):
    ''' Visualize the data distribution '''
    data_meta_info = xydata.metainfo
    assert "FLOPs" in data_meta_info.pre_add_dims
    flop_idx = data_meta_info.pre_add_dims.index("FLOPs")
    percentiles = [1, 99]

    max_flop = np.max(xydata[:, flop_idx])
    min_flop = np.min(xydata[:, flop_idx])
    if min_flop == max_flop:
        min_flop -= 0.5
        max_flop += 0.5
    xaxis = np.arange(min_flop, max_flop, (max_flop-min_flop)/10)

    fig = plt.figure(figsize=(8, 5))
    _fig_base = fig_base(2)

    ax = fig.add_subplot(_fig_base+1)
    ax.scatter(xydata[:, flop_idx], xydata[:, 0])
    for percentile in percentiles:
        value = np.percentile(xydata[:, 0], percentile)
        ax.plot(xaxis, [value for _ in xaxis], '-', label=f"{percentile}th")
    plt.xlabel("# of FLOPs", fontsize=16)
    plt.ylabel("Time", fontsize=16)
    plt.legend(fontsize=16)

    ax = fig.add_subplot(_fig_base+2)
    ax.hist(xydata[:, 0], bins = 520,
             color = 'blue', edgecolor = 'black')
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Count", fontsize=16)

    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_dir = os.path.join(path, ".fig/metalearner")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, "data_distribution.png"))
    plt.close()

def check_data_distribution_by_file(data_tuples, save_file_name, save_dir=None):
    ''' Visualize the data distribution for each task file

    Parameters
    ---------
    data_tuples: tuple of _file, xydata, error
    '''
    fig = plt.figure(figsize=(8, 5))
    _fig_base = fig_base(len(data_tuples))

    for idx, (_file, xydata, error) in enumerate(data_tuples):
        ax = fig.add_subplot(_fig_base+idx+1)
        ax.hist(xydata[:, 0], bins = 520,
                color = 'blue', edgecolor = 'black')
        
        plt.xlabel("Time", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        if error:
            plt.title(f"{_file} - Error={error}")
        else:
            plt.title(_file)

    plt.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, save_file_name))

    plt.close()

def task_by_task_analysis(trace_root_path, learning_params):
    '''Anaylyze data task by task'''
    root_path, _, files = list(os.walk(trace_root_path))[0]
    files_to_test = sorted([f for f in files if f.endswith(".npy")])

    def _iternal_load_raw_data(_cur_files):
        return load_raw_data(
            [os.path.join(root_path, _f) for _f in _cur_files],
            learning_params, force=True, verbose=False)
    
    ### analysis:
    if False:
        for file in files_to_test:
            xydata = _iternal_load_raw_data([file])
            check_data_distribution_by_file([(file, xydata)], ".")
            x=input("Continue?")

    if True:
        data_tuples = []
        # for file in [files_to_test[0], files_to_test[800], files_to_test[900]]:
        for file in [files_to_test[0]]:
            xydata = _iternal_load_raw_data([file])
            data_tuples.append((file, xydata, "???"))
        check_data_distribution_by_file(data_tuples, "tmp", save_dir=".")

    for file_idx, file in enumerate(files_to_test):
        _cur_files = [file]
        # _cur_files = [files_to_test[0], files_to_test[900]]
        # _cur_files = [files_to_test[0], files_to_test[800]]
        # _cur_files = [files_to_test[800]]
        # _cur_files = ["t4_0.npy"]
        # print(_cur_files)

        xydata = _iternal_load_raw_data(_cur_files)
        check_data_distribution(xydata, ".")

def clustering_analysis(
        trace_root_path,
        learning_params,
        task_sample_num=10,
        save_dir=".",
        cluster_method="di",
        max_sample_num=10000
        ):
    ''' Perform clustering based on X and visualization the results'''
    raw_data, cluster_rst, files_to_test = load_raw_data_w_cluster_id(
        trace_root_path,
        learning_params,
        task_sample_num=task_sample_num,
        cluster_method=cluster_method,
    )

    ### Perform dimension reduction using PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(raw_data.x)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    # centroid_reduced = pca.transform(kmeans.cluster_centers_)

    ### Convert the matrix and vector to a Pandas DataFrame for the ease of plotting
    feat_cols = ['x_'+str(i) for i in range(raw_data.size)]
    df = pd.DataFrame(raw_data.x, columns=feat_cols)
    df['y'] = raw_data.y
    df['cluster'] = cluster_rst
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1]

    # For reproducability of the results
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    cluster_num = len(set(df['cluster'].values))

    ### Plot figure
    fig = plt.figure(figsize=(12, 10))
    _fig_base = fig_base(9, row_first=False)

    ax = fig.add_subplot(_fig_base+1)
    plt.scatter(
        df['pca-one'], 
        df['pca-two'], 
        # zs=pca_result[:, 2], 
        c=df["y"],
        alpha=0.3,
        edgecolors='none',
        cmap=plt.cm.get_cmap('rainbow', 10)
    )
    cbar = plt.colorbar()
    cbar.set_label(label="Execution Time", fontdict=font)
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.title.set_text(f"Execution time\ndistribution with {raw_data.size} samples")

    ax = fig.add_subplot(_fig_base+2)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="cluster",
        palette=sns.color_palette("hls", cluster_num),
        data=df,
        legend="full",
        alpha=0.3
    )
    ax.title.set_text(f"Group {raw_data.size} records from\n{task_sample_num} tasks to {cluster_num} clusters")

    ### Perform dimension reduction using t-SNE
    df_subset = df.loc[rndperm[:max_sample_num],:].copy()
    data_subset = df_subset[feat_cols].values
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_subset)
    df_subset['pca-one'] = pca_result[:,0]
    df_subset['pca-two'] = pca_result[:,1] 
    # df_subset['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    time_start = time.time()
    tsne = TSNE(n_components=2, init='random', verbose=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]

    subset_cluster_num = len(set(df_subset['cluster'].values))
    
    ax = fig.add_subplot(_fig_base+3)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="cluster",
        palette=sns.color_palette("hls", subset_cluster_num),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    ax.title.set_text(f"Dimension Reduction using\nPCA on {len(data_subset)} samples")

    ax = fig.add_subplot(_fig_base+4)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="cluster",
        palette=sns.color_palette("hls", subset_cluster_num),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    ax.title.set_text(f"Dimension Reduction using\nt-SNE on {len(data_subset)} samples")

    ax = fig.add_subplot(_fig_base+5)
    plt.scatter(
        df_subset['tsne-2d-one'], 
        df_subset['tsne-2d-two'], 
        c=df_subset["y"],
        alpha=0.3,
        edgecolors='none',
        cmap=plt.cm.get_cmap('rainbow', 10)
    )
    cbar = plt.colorbar()
    cbar.set_label(label="Execution Time", fontdict=font)
    ax.set_xlabel('tsne-2d-one')
    ax.set_ylabel('tsne-2d-two')
    ax.title.set_text(f"Execution time\ndistribution with {len(data_subset)} samples")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, ".fig", "metalearner", f"clustering_{cluster_method}.png"))

def verify_x2y_mapping(trace_root_path, learning_params, save_dir="."):
    """ Check whether the dataset is learnable by ploting the relationship between x and y"""
    root_path, _, files = list(os.walk(trace_root_path))[0]
    files_to_test = sorted([f for f in files if f.endswith(".npy")])

    def _iternal_load_raw_data(_cur_files):
        print(_cur_files)
        return load_raw_data(
            [os.path.join(root_path, _f) for _f in _cur_files],
            learning_params, force=True, verbose=False)
    
    raw_data = _iternal_load_raw_data(files_to_test[:10])
    check_method = "max"
    # check_method = "l1norm"
    print(raw_data.raw_data.shape)
    ratios = [0.01, 0.05, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    temp, elementwise_rst, classifier = sample_data_around_avg(
        raw_data.raw_data, raw_data.metainfo,
        ratios=ratios, check=check_method)
    cluster_num = len(set(classifier))
    plot_x = np.average(elementwise_rst, axis=1)
    plot_y = np.log(temp[:, 0].flatten())
    
    fig = plt.figure(figsize=(8, 5))
    _fig_base = fig_base(1)

    ax = fig.add_subplot(_fig_base+1)
    sns.scatterplot(
        x=plot_x,
        y=plot_y,
        hue=classifier,
        palette=sns.color_palette("hls", cluster_num),
        legend="full",
        alpha=0.3
        )
    handles, labels = ax.get_legend_handles_labels()
    check_str = "max(norm(x))" if check_method == "max" else "|norm(x)|"
    new_legends = [f"{check_str} <= {labels[0]}"]
    for idx in range(len(labels) - 1):
        new_legends.append(f"{labels[idx]} < {check_str} <= {labels[idx+1]}")
    print(new_legends)
    plt.legend(handles, new_legends)
    plt.xlabel("|norm(x)|", fontsize=16)
    plt.ylabel("Time", fontsize=16)
    # plt.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, ".fig", "metalearner", "verify_x2y_mapping.png"))
    plt.close()

def x2y_sensitivity_analysis(x, y, device):
    ''' Check the sensitivity of each feature entry and find out the following cases

        Case 1: 
            Two samples differ in ONLY one feature entry, but have the same Y 
            --> to find useless entries
        Case 2: 
            The same x and y --> repeated data points
        Case 3: 
            The same x but different y, which indicates the 
            feature vector may not contain enough information.
    '''

    ### If y is normalized, y may < 0, further transform y to make sure y > 0
    y = y - min(y)

    sample_num, entry_num = x.shape
    assert len(y) == sample_num

    avg_per_entry = np.mean(x, axis=0)
    std_per_entry = np.std(x, axis=0)
    
    same_x_y = []
    sample_id_to_same_x_y = {}
    same_x_diff_y = []
    sample_id_to_same_x_diff_y = {}

    log_file = open(f"tmp/x2y_senstivity_ana_log_{device}.txt", 'w')
    def _log(msg: str):
        log_file.write(msg + "\n")
        print(msg)

    valid_entry = np.where(std_per_entry != 0)[0]
    _log(f"Valid entries: {str(valid_entry)}")

    useless_entry_ids = []
    for entry_id in range(entry_num):
        if std_per_entry[entry_id] == 0:
            if avg_per_entry[entry_id] != 0:
                _log(f"All samples share the same non-zero value {avg_per_entry[entry_id]} for Entry {entry_id}")
                useless_entry_ids.append(entry_id)
            continue

    for sample_id in range(sample_num-1):
        for compare_to in range(sample_id+1, sample_num):
            if np.all(x[sample_id] == x[compare_to]):
                ### All feature entries are the same
                if y[sample_id] == y[compare_to]:
                    # _log(f"S{sample_id} and S{compare_to} have the same x and y")
                    if sample_id in sample_id_to_same_x_y:
                        group_id = sample_id_to_same_x_y[sample_id]
                    elif compare_to in sample_id_to_same_x_y:
                        group_id = sample_id_to_same_x_y[compare_to]
                    else:
                        group_id = -1
                    if group_id >= 0:
                        group = same_x_y[group_id]
                        group.add(sample_id)
                        group.add(compare_to)
                    else:
                        group = set([sample_id, compare_to])
                        group_id = len(same_x_y)
                        same_x_y.append(group)
                    sample_id_to_same_x_y[sample_id] = group_id
                    sample_id_to_same_x_y[compare_to] = group_id
                else:
                    # _log(f"S{sample_id} and S{compare_to} have the same x, but different y")
                    if sample_id in sample_id_to_same_x_diff_y:
                        group_id = sample_id_to_same_x_diff_y[sample_id]
                    elif compare_to in sample_id_to_same_x_diff_y:
                        group_id = sample_id_to_same_x_diff_y[compare_to]
                    else:
                        group_id = -1
                    
                    diff2y_ratio = 2 * abs(y[sample_id] - y[compare_to])/(y[sample_id] + y[compare_to])

                    if group_id >= 0:
                        group, norm_diff = same_x_diff_y[group_id]
                        norm_diff.append(diff2y_ratio)
                        group.add(sample_id)
                        group.add(compare_to)
                    else:
                        group = set([sample_id, compare_to])
                        group_id = len(same_x_diff_y)
                        norm_diff = [diff2y_ratio]
                        same_x_diff_y.append((group, norm_diff))
                    sample_id_to_same_x_diff_y[sample_id] = group_id
                    sample_id_to_same_x_diff_y[compare_to] = group_id
                continue

            for entry_id in range(entry_num):
                if entry_id in useless_entry_ids:
                    continue

                ### check if two samples are the same when ignoring curent entry 
                residual = x[sample_id] - x[compare_to]
                residual[entry_id] = 0
                if np.all(residual == 0):
                    ### The remaining entries are the same for the two samples
                    assert x[sample_id][entry_id] != x[compare_to][entry_id]
                        
                    if y[sample_id] == y[compare_to]:
                        _log(f"S{sample_id} and S{compare_to} differ at Entry {entry_id} but have the same y")
                    else:
                        derative = (y[sample_id] - y[compare_to]) / (x[sample_id][entry_id] - x[compare_to][entry_id])
                        _log(f"S{sample_id} and S{compare_to} differ at Entry {entry_id}, with a partial derivative of {derative}")

    _log("The following samples have the same x and y")
    for group in same_x_y:
        _log(f" {str(group)}")

    _log("The following samples have the same x, but different y")
    all_norm_diff = []
    for group, norm_diff in same_x_diff_y:
        _log(f" {str(group)}, avg diff/y = {np.mean(norm_diff):.3f}")
        all_norm_diff += norm_diff
    all_norm_diff = np.array(all_norm_diff)
    _log("Distribution of diff/y for this case:")
    _log(f" Min = {np.min(all_norm_diff):.3f}")
    for percentile in [5, 10, 25, 50, 75, 90, 95]:
        _log(f" {percentile}% = {np.percentile(all_norm_diff, percentile):.3f}")
    _log(f" Max = {np.max(all_norm_diff):.3f}")

    