import os
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from metalearner.feature import ALL_FEATURE_TYPE, is_feature_type

from metalearner.data.rawdata import (
    MIN_PER_TASK_SAMPLE_NUM,
    RawData,
    ASTRawData,
    extract_task_id,
    load_raw_data,
    parse_metainfo
)

def plot_leaf_no_dist(leaf_nos, path_to_rst):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 6))
    bins = np.arange(min(leaf_nos)-0.5, max(leaf_nos) + 1.5, 1)
    n, bins, patches = plt.hist(np.array(leaf_nos), bins=bins, rwidth=0.8)
    plt.xlabel("# of AST Leaf Nodes", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(np.arange(min(leaf_nos), max(leaf_nos) + 1, 1), fontsize=16)
    plt.title(f"AST Leaf Nodes Distribution", fontsize=16)
    plt.tight_layout()
    plt.savefig(path_to_rst)

def plot_leaf_no_dist2(leaf_node_nos, frequencies, path_to_rst, x_label=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4))
    plt.scatter(np.array(leaf_node_nos), np.array(frequencies), s=100)
    if x_label is None:
        x_label = "# of AST Leaf Nodes"
    plt.xlabel(x_label, fontsize=24)
    plt.ylabel("Frequency", fontsize=24)
    # ax.set_yscale("log")
    plt.yscale("log")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.title(f"AST Leaf Nodes Distribution", fontsize=24)
    plt.tight_layout()
    plt.savefig(path_to_rst)

def check_max_leaf_no(raw_data):
    assert is_feature_type(ALL_FEATURE_TYPE.ast_ansor)
    assert isinstance(raw_data, ASTRawData)
    N_max_leaf = raw_data.max_leaf_no()
    print(f"Maximum leaf no for AST-style feature is {N_max_leaf}")

def check_leaf_no_dist(raw_data, path_to_rst=None):
    assert is_feature_type(ALL_FEATURE_TYPE.ast_ansor)
    assert isinstance(raw_data, ASTRawData)
    leaf_nos = raw_data.leaf_no_dist()
    if path_to_rst:
        plot_leaf_no_dist(leaf_nos, path_to_rst)
    from collections import Counter
    counts = Counter(leaf_nos)
    print(f"leaf no dist: {counts}")
    
def check_max_leaf_no_with_files(files, learning_params, file_batch_num=None):
    assert is_feature_type(ALL_FEATURE_TYPE.ast_ansor)
    # load_ts = time.time()
    files_batch = []
    N_max_leaf = 0
    if file_batch_num is None:
        file_batch_num = len(files)
    for _file in files:
        files_batch.append(_file)
        if len(files_batch) < file_batch_num:
            continue

        for _f in files_batch:
            raw_data = load_raw_data(
                [_f], learning_params, verbose=False)
            if raw_data.size < MIN_PER_TASK_SAMPLE_NUM:
                continue
            assert isinstance(raw_data, ASTRawData)
            N_max_leaf = max(N_max_leaf, raw_data.max_leaf_no())
        files_batch = []
    print(f"Maximum leaf no for AST-style feature is {N_max_leaf}")

def check_leaf_no_dist_with_files(files, learning_params, path_to_rst, file_batch_num=None):
    assert is_feature_type(ALL_FEATURE_TYPE.ast_ansor)
    # load_ts = time.time()
    files_batch = []
    leaf_nos = []
    if file_batch_num is None:
        file_batch_num = len(files)
    for _file in files:
        files_batch.append(_file)
        if len(files_batch) < file_batch_num:
            continue

        for _f in files_batch:
            raw_data = load_raw_data(
                [_f], learning_params, verbose=False)
            if raw_data.size < MIN_PER_TASK_SAMPLE_NUM:
                continue
            assert isinstance(raw_data, ASTRawData)
            leaf_nos += raw_data.leaf_no_dist()
        files_batch = []
    plot_leaf_no_dist(leaf_nos, path_to_rst)


def check_ast_node_dist(files, learning_params):
    _dir = ".workspace/ast_dist_ana"
    stat_rst_path = os.path.join(_dir, "ast_node_stat.json")
    if False:
        leaf_no_cnt = {}
        ast_node_cnt = {}
        ast_cnt = {}

        for _f in tqdm(files):
            raw_data = load_raw_data([_f], learning_params, verbose=False)
            for _data in raw_data.raw_data:
                avg, std, flops, ast_features, node_ids, serialized_tree = _data
                # print(ast_features.shape) # (N_seq, N_entry)
                ast_hash = ",".join([str(x) for x in serialized_tree])
                if ast_hash not in ast_cnt:
                    ast_cnt[ast_hash] = 1
                else:
                    ast_cnt[ast_hash] += 1

                leaf_no = str(ast_features.shape[0])
                if leaf_no not in leaf_no_cnt:
                    leaf_no_cnt[leaf_no] = 1
                else:
                    leaf_no_cnt[leaf_no] += 1

                ast_node_no = str(max(serialized_tree))
                if ast_node_no not in ast_node_cnt:
                    ast_node_cnt[ast_node_no] = 1
                else:
                    ast_node_cnt[ast_node_no] += 1

        # save data
        with open(stat_rst_path, 'w') as fp:
            json.dump([leaf_no_cnt, ast_node_cnt, ast_cnt], fp, indent=4)
    else:
        with open(stat_rst_path, 'r') as fp:
            leaf_no_cnt, ast_node_cnt, ast_cnt = json.load(fp)
    
    N_ast_node, Fre_ast_node = zip(*sorted([(int(n_leaf), cnt) for n_leaf, cnt in ast_node_cnt.items()], key=lambda x: x[0]))
    N_leaf, Fre_leaf = zip(*sorted([(int(n_leaf), cnt) for n_leaf, cnt in leaf_no_cnt.items()], key=lambda x: x[0]))

    plot_leaf_no_dist2(N_ast_node, Fre_ast_node, os.path.join(_dir, "ast_node_dist.pdf"), "# of AST nodes")
    plot_leaf_no_dist2(N_leaf, Fre_leaf, os.path.join(_dir, "leaf_node_dist.pdf"), "# of AST leaf nodes")
    


