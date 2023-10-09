import os
import numpy as np
import math

import matplotlib.pyplot as plt

device_paths = [
    ".workspace/ast_ansor/v100",
    ".workspace/ast_ansor/t4",
    ".workspace/ast_ansor/a10",
    ".workspace/ast_ansor/a100",
    ".workspace/ast_ansor/k80"
]
device_names = [os.path.basename(_path).upper() for _path in device_paths]
TASK_SAMPLE_NUM = 200

def _get_device_means():
    device_means = []
    for _device in device_names:
        device_means.append([])

    device2Y = [np.empty((0)) for _ in device_names]
    for task_id in range(TASK_SAMPLE_NUM):
        all_exist = True
        for device_path in device_paths:
            if not os.path.exists(os.path.join(device_path, f"{task_id}.npy")):
                all_exist = False
                break
        
        if not all_exist:
            for device_id in range(len(device_names)):
                device_means[device_id].append(0)
            continue
        
        for device_id, device_path in enumerate(device_paths):
            device_data = np.load(os.path.join(device_path, f"{task_id}.npy"), allow_pickle=True)
            device_means[device_id].append(float(np.mean(device_data[:, 0])))
            device2Y[device_id] = np.concatenate((device2Y[device_id], device_data[:, 0]))
    return device_means, device2Y
    
def plot_y_histogram():
    ''' Plot the Y histogram for each device
    '''
    device_means, device2Y = _get_device_means()

    fig = plt.figure(figsize=(12, 8))
    _figbase = 330

    for device_id, device_y in enumerate(device2Y):
        _figbase += 1
        ax = fig.add_subplot(_figbase)
        _max = np.percentile(device_y, 75)
        _min = 0
        plt.hist(device_y, bins=100, range=(_min, _max))
        plt.xlabel(f"{device_names[device_id]}: Y")
        plt.ylabel("Frequency")

    _figbase += 1
    ax = fig.add_subplot(_figbase)
    x_axis = np.arange(TASK_SAMPLE_NUM)
    for device_id in range(len(device_names)):
        ax.scatter(x_axis[:100], device_means[device_id][:100], label=device_names[device_id])
    plt.xlabel("Task ID")
    plt.ylabel("Mean Measured Cost")
    plt.legend()

    _figbase += 1
    ax = fig.add_subplot(_figbase)
    x_axis = np.arange(TASK_SAMPLE_NUM)
    for device_id in range(len(device_names)):
        ax.scatter(x_axis[100:], device_means[device_id][100:], label=device_names[device_id])
    plt.xlabel("Task ID")
    plt.ylabel("Mean Measured Cost")
    plt.legend()

    plt.tight_layout()
    plt.savefig("tmp2.png")

def plot_cross_device_ratio_histogram():
    ''' Plot cross_device_ratio for each device
    '''
    device_means, device2Y = _get_device_means()

    fig = plt.figure(figsize=(12, 8))
    _figbase = 330

    for device_id, device_y in enumerate(device2Y):
        _figbase += 1
        ax = fig.add_subplot(_figbase)
        _max = np.percentile(device_y, 75)
        _min = 0
        plt.hist(device_y, bins=100, range=(_min, _max))
        plt.xlabel(f"{device_names[device_id]}: Y")
        plt.ylabel("Frequency")

    _figbase += 1
    ax = fig.add_subplot(_figbase)
    x_axis = np.arange(TASK_SAMPLE_NUM)
    for device_id in range(len(device_names)):
        ax.scatter(x_axis[:100], device_means[device_id][:100], label=device_names[device_id])
    plt.xlabel("Task ID")
    plt.ylabel("Mean Measured Cost")
    plt.legend()

    _figbase += 1
    ax = fig.add_subplot(_figbase)
    x_axis = np.arange(TASK_SAMPLE_NUM)
    for device_id in range(len(device_names)):
        ax.scatter(x_axis[100:], device_means[device_id][100:], label=device_names[device_id])
    plt.xlabel("Task ID")
    plt.ylabel("Mean Measured Cost")
    plt.legend()

    _figbase += 1
    ax = fig.add_subplot(_figbase)
    x_axis = np.arange(TASK_SAMPLE_NUM) / 200 * 4
    ratios = []
    for device_id in range(1, len(device_names)):
        ratios.append(np.array(device_means[device_id])/(1e-6 + np.array(device_means[0])))
    
    print(len(ratios), len(device_means))
    ax.boxplot(ratios, labels=device_names[1:])
    ax.plot(x_axis, np.ones_like(x_axis), label="ratio=1")
    plt.xlabel("Devices")
    plt.title(f"Distribution of the ratio of task-mean-cost \n"
        f"compared to {device_names[0]}")
    plt.ylabel(f"Ratio Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig("tmp2.png")

INVALID_ENTRIES = [0, 3, 6, 7, 12, 13, 15, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 94, 102, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146]

def shape_convert(_X):
    if len(_X.shape) == 2:
        return _X
    elif len(_X.shape) == 3:
        return np.sum(_X, axis=1)

def plot_x_device2y(trace_root_path, learning_params):
    ''' Plot the p(y|x, D=d) for each device d
    '''
    from utils.device_info import DEVICE_FEATURE_LEN, get_device_feature
    from utils.util import sample_task_files, fig_base
    from metalearner.data.rawdata import load_raw_data, parse_metainfo

    device2task = sample_task_files(trace_root_path, learning_params["mode"],
        learning_params["gpu_model"], absolute_path=True)
    data_meta_info = parse_metainfo(device2task, learning_params, True)

    device2data = {}
    for device, (_, device_files) in device2task:
        raw_data = load_raw_data(device_files, learning_params, force=False, verbose=False)
        raw_data.metainfo = data_meta_info
        raw_data.freeze()
        raw_data.normalize()
        # if len(raw_data.x.shape) <= 1:
        #     raw_data.x = raw_data.x.reshape((-1, data_meta_info.feature_len))
        raw_data.x = shape_convert(raw_data.x)
        device2data[device] = raw_data
        print(f"Device {device} data shape {raw_data.size}")
    
    all_devices = list(device2data.keys())

    def __plot(entry_id):
        fig = plt.figure(figsize=(12, 8))
        _base = fig_base(len(all_devices))
        for _id, device in enumerate(all_devices):
            ax = fig.add_subplot(_base + _id + 1)
            # print(device2data[device].x[:, entry_id].shape)
            # print(device2data[device].y.shape)
            ax.scatter(device2data[device].x[:, entry_id], device2data[device].y, alpha=0.1, edgecolors='none')
            plt.ylabel("y")
            plt.xlabel(f"x[{entry_id}]")
            plt.title(f"Device {device}")
            
        plt.tight_layout()
        plt.savefig(os.path.join(f"./tmp/x2y/x[{entry_id}]2y.png"))
        plt.close()
    
    for entry_id in range(data_meta_info.feature_len):
        if entry_id in INVALID_ENTRIES:
            continue
        __plot(entry_id)

def cross_device_similarity(learning_params):
    from metalearner.data.rawdata import load_raw_data
    from utils.device_info import get_device_feature

    for task_id in range(TASK_SAMPLE_NUM):
        print(f"\n\nTask {task_id}")
        all_exist = True
        for device_path in device_paths:
            if not os.path.exists(os.path.join(device_path, f"{task_id}.npy")):
                all_exist = False
                break
        if not all_exist:
            continue
        
        device2xy = []
        for device_id, device_path in enumerate(device_paths):
            raw_data = load_raw_data([os.path.join(device_path, f"{task_id}.npy")], learning_params, force=False, verbose=False)
            raw_data.freeze()
            device2xy.append((shape_convert(raw_data.x), raw_data.y))
        
        # device2xy = np.array(device2xy)
        for d1 in range(len(device_paths)-1):
            if len(device2xy[d1][1]) == 0:
                continue
            x_device1 = get_device_feature(device_names[d1])
            for d2 in range(d1+1, len(device_paths)):
                if len(device2xy[d2][1]) == 0:
                    continue
                x_device2 = get_device_feature(device_names[d2])
                device_name_pair = (device_names[d1], device_names[d2])
                print(f"\nDevice {device_name_pair}:")
                for x_d1, y_d1 in zip(*device2xy[d1]):
                    for x_d2, y_d2 in zip(*device2xy[d2]):
                        ### TOOD: change the condition
                        if np.all(x_d1 == x_d2) and y_d1 != y_d2:
                            print(f"({x_d1.tolist()},{y_d1}) vs "
                                f"({x_d2.tolist()},{y_d2})")

def device_data_learnability_compare(trace_root_path, learning_params, sample_num=10000):
    ''' For each device, plot the distribution of (|(y2-y1)/(x2-x1)|)
    '''
    from utils.device_info import DEVICE_FEATURE_LEN, get_device_feature
    from utils.util import sample_task_files, fig_base
    from metalearner.data.rawdata import load_raw_data, parse_metainfo

    device2task = sample_task_files(trace_root_path, learning_params["mode"], 
        learning_params["gpu_model"], absolute_path=True)
    data_meta_info = parse_metainfo(device2task, learning_params, True)

    device2derivative_norms = {}
    device2size = {}
    _device_names = list(device2task.device2info.keys())
    for device, (_, device_files) in device2task:
        print(device)
        raw_data = load_raw_data(device_files, learning_params, force=False, verbose=False)
        raw_data.metainfo = data_meta_info

        if device == "habana" or device.startswith("hl"):
            raw_data.fix_seq_len = [1]
        else:
            raw_data.fix_seq_len = [5]

        raw_data.freeze()
        raw_data.normalize()

        raw_data.x = shape_convert(raw_data.x)
        if sample_num is None:
            sample_num = raw_data.size

        # import pdb
        # pdb.set_trace()
        device2derivative_norms[device] = []
        device2size[device] = raw_data.size
        if raw_data.size <= 0:
            continue
        all_idxs = np.arange(raw_data.size)
        for _ in range(sample_num):
            index1, index2 = np.random.choice(all_idxs, 2, replace=False)
            x1, y1, _ = raw_data.slice_freezed(index1)
            x2, y2, _ = raw_data.slice_freezed(index2)
            derivative = []
            for entry_id in range(data_meta_info.feature_len):
                if entry_id in INVALID_ENTRIES:
                    continue
                derivative.append(abs(y2 - y1) / (abs(x2[entry_id] - x1[entry_id]) + 1e-6))
            device2derivative_norms[device].append(np.linalg.norm(derivative))

    fig = plt.figure(figsize=(12, 6))
    plt.boxplot([device2derivative_norms[device_name] for device_name in _device_names],
        labels=[f"{device_name}({device2size[device_name]})" for device_name in _device_names])
    plt.xlabel("Devices")
    plt.ylabel("Norm(|(y2-y1)/(x2-x1)|)")
    plt.title("Distribution of Norm(|(y2-y1)/(x2-x1)|)")
    plt.tight_layout()
    plt.savefig("tmp/tmp3.png")

def device_data_analyze_via_dim_reduction(trace_root_path, learning_params, sample_num=1024):
    ''' Analyze data through dimension reduction
    '''
    from utils.device_info import DEVICE_FEATURE_LEN, get_device_feature
    from utils.util import sample_task_files, fig_base
    from metalearner.data.rawdata import load_raw_data, parse_metainfo

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    np.random.seed(1024)

    device2task = sample_task_files(trace_root_path, learning_params["mode"],
        learning_params["gpu_model"], absolute_path=True)
    data_meta_info = parse_metainfo(device2task, learning_params, True)

    device2reduct2xy = {}
    for device, (_, device_files) in device2task:
        print(device)

        raw_data = load_raw_data(device_files, learning_params, force=False, verbose=False)
        raw_data.metainfo = data_meta_info

        if "habana" in device or device.startswith("hl"):
            raw_data.fix_seq_len = [1]
        else:
            raw_data.fix_seq_len = [5]

        raw_data.freeze()
        if raw_data.size <= 0:
            continue
        X = shape_convert(raw_data.x)

        from .data_analyze import x2y_sensitivity_analysis
        print(f"Device {device}, Feature len {X.shape}")
        x2y_sensitivity_analysis(X.astype(float)[:2034], raw_data.y[:2034], device)
        raise

        raw_data.normalize()
        X = shape_convert(raw_data.x)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X[:sample_num])
        tsne = TSNE(n_components=2, init='random', verbose=1, perplexity=40, n_iter=300, random_state=0)
        tsne_result = tsne.fit_transform(X[:sample_num])

        device2reduct2xy[device] = {
            "PCA": pca_result,
            "tSNE": tsne_result,
            "Y": raw_data.y[:sample_num]
        }

    def __plot(name, dim_red_method="PCA", angle=None):
        fig = plt.figure(figsize=(12, 12))
        _fig_base = fig_base(len(device2reduct2xy))
        device_id = 0
        for device in device2reduct2xy:
            fit_result = device2reduct2xy[device][dim_red_method]
            Y = device2reduct2xy[device]["Y"]
            ax = fig.add_subplot(_fig_base + device_id + 1, projection="3d")
            ax.scatter(fit_result[:, 0], fit_result[:, 1], Y)
            ax.set_xlabel('Reduced X[0]')
            ax.set_ylabel('Reduced X[1]')
            ax.set_zlabel('Y')
            plt.title(device)
            device_id += 1
            if angle is not None:
                ax.view_init(*angle)
        plt.tight_layout()
        plt.savefig(name)
        plt.close()
    
    __plot("tmp/x2y_3d_pca.png", "PCA", None)
    __plot("tmp/x2y_3d_pca2.png", "PCA", (20, -110))
    __plot("tmp/x2y_3d_tsne.png", "tSNE", None)
    __plot("tmp/x2y_3d_tsne2.png", "tSNE", (20, -110))

if __name__ == "__main__":
    plot_cross_device_ratio_histogram()
