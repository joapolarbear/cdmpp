import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import datetime

from utils.util import sample_task_files
from utils.metrics import metric_cmd

from metalearner.data.rawdata import load_raw_data, parse_metainfo


def verify_selected_tasks(X, sample_idxs_array_list, selected_task_id):
    ''' Verify if the selected tasks have the similar data distribution as the original data set

    Parameters
    -----------
    X: 2-D ndarray
        The original data of all tasks, whose shape is (# of total tasks \times # of feature entries)
    sample_idxs_array_list: list of ndarray
        Each element denotes the sample indices in X of corresponding task
    selected_task_id: 1-D ndarray or set or list
        The ids of selected tasks, corresponding to the first dimension of X

    Return
    --------
    diff: float
        Domain difference between the selected data and the original dataset
    '''
    selected_indices = [sample_idxs_array_list[task_id] for task_id in selected_task_id]
    selected_indices = np.concatenate(selected_indices).astype(int)
    if len(selected_indices) == 0:
        return -1
    selected_data = X[selected_indices]
    cmd = metric_cmd(X, selected_data)
    return cmd

def sample_tasks_implt(X, sample_idxs_array_list, selected_num=10, method="random"):
    ''' Perform task selection

    Parameters
    -----------
    X: 2-D ndarray
        The original data of all tasks, whose shape is (# of total tasks \times # of feature entries)
    sample_idxs_array_list: 2-D ndarray
        The sample indices in X of each tasks

    Return
    --------
    selected_task_id: 1-D ndarray or set or list
        The ids of selected tasks, corresponding to the first dimension of X
    '''
    
    task_num = len(sample_idxs_array_list)

    ### Perform task selection
    print("Perform task selection ... ")
    if method == "kmeans":
        ''' Step 1: Perform KMeans clustering on the entire dataset X first
            Step 2: For each cluster with the cluster center x_c and each task with data X_t, calculate the averate distance D(c, t)
            Step 3: Sort the cluster by cluster size, and select the task with the smallest D(c, t) for each cluster.
        '''
        selected_task_id = set()
        # import code; code.interact(local=locals())
        kmeans = KMeans(n_clusters=selected_num, random_state=0).fit(X)
        cluster_rst = kmeans.labels_
        cluster_ceters = kmeans.cluster_centers_
        cluster_sizes = [len(np.where(cluster_rst == cluster_id)[0]) for cluster_id in range(selected_num)]

        dist_table = np.ones((selected_num, task_num)) * 1e6
        for cluster_id in range(selected_num):
            for task_id in range(task_num):
                X_t = X[sample_idxs_array_list[task_id]]
                if len(X_t) > 0:
                    dist_list2center = [np.linalg.norm(cluster_ceters[cluster_id] - X_t[j]) for j in range(len(X_t))]
                    dist_table[cluster_id, task_id] = np.mean(dist_list2center)

        mapping = {}
        for cluster_id in np.argsort(cluster_sizes)[::-1]:
            for task_id in np.argsort(dist_table[cluster_id]):
                if task_id not in selected_task_id:
                    mapping[task_id] = cluster_id
                    selected_task_id.add(task_id)
                    break

        selected_task_id = list(selected_task_id)
    
    elif method == "kmeans+search":
        kmeans = KMeans(n_clusters=100, random_state=0).fit(X)
        cluster_rst = kmeans.labels_
        cluster_ceters = kmeans.cluster_centers_
        cluster_sizes = [len(np.where(cluster_rst == cluster_id)[0]) for cluster_id in range(selected_num)]
        cluster_size_per_task = []
        for task_id in range(task_num):
            _cluster_rst = cluster_rst[sample_idxs_array_list[task_id]]
            _size = [len(np.where(_cluster_rst == cluster_id)[0]) for cluster_id in range(selected_num)]
            cluster_size_per_task.append(_size)
        cluster_size_per_task = np.array(cluster_size_per_task)

        cluster_sizes_norm = cluster_sizes / np.linalg.norm(cluster_sizes)

        for selected_task_id in [
            [192,133,198,169,170,180,181,182,151,121]]:
            selected_size = np.sum(cluster_size_per_task[selected_task_id], axis=0)
            selected_size_norm = selected_size / np.linalg.norm(selected_size)
            print(np.linalg.norm(cluster_sizes_norm - selected_size_norm))
            
        raise

    elif method == "center":
        data_center = np.mean(X, axis=0)
        task_dist2centers = np.ones(task_num) * 1e6
        for task_id in range(task_num):
            X_t = X[sample_idxs_array_list[task_id]]
            if len(X_t) > 0:
                task_dist2centers[task_id] = np.linalg.norm(
                    np.mean(X_t, axis=0) - data_center)

        selected_task_id = np.argsort(task_dist2centers)[:selected_num]
    
    elif method == "cmd":
        task_dist2centers = np.ones(task_num) * 1e6
        for task_id in range(task_num):
            X_t = X[sample_idxs_array_list[task_id]]
            if len(X_t) > 0:
                task_dist2centers[task_id] = metric_cmd(X, X_t)

        selected_task_id = np.argsort(task_dist2centers)[:selected_num]
    
    elif method == "cmd+rm":
        selected_task_id = []
        task_ids = np.arange(task_num)
        _X = X.copy()
        _sample_idxs_array_list = sample_idxs_array_list.copy()
        for cluster_id in range(selected_num):
            print(f"Cluster {cluster_id}, calculating CMD")
            task_dist2centers = []
            for task_id in tqdm(task_ids):
                X_t = X[_sample_idxs_array_list[task_id]]
                if len(X_t) > 0:
                    task_dist2centers.append(metric_cmd(_X, X_t))
                else:
                    task_dist2centers.append(1e6)

            row = np.argsort(task_dist2centers)[0]
            task_id_selected = task_ids[row]
            selected_task_id.append(task_id_selected)

            ### Update tasks_ids
            task_ids = np.delete(task_ids, row, axis=0)
            ### Update all data _X and _sample_idxs_array_list
            sample_rows = _sample_idxs_array_list[task_id_selected]
            _X = np.delete(_X, sample_rows, axis=0)
            sample_row_st = sample_rows[0]
            sample_row_ed = sample_rows[-1]
            sample_num = len(sample_rows) 
            for task_id in task_ids:
                if len(_sample_idxs_array_list[task_id]) == 0:
                    pass
                elif _sample_idxs_array_list[task_id][-1] < sample_row_st:
                    pass
                elif _sample_idxs_array_list[task_id][0] > sample_row_ed:
                    _sample_idxs_array_list[task_id] -= sample_num
                else:
                    raise ValueError(f"Do not expect to reach this, select task {task_id_selected} among tasks {task_ids}\n"
                        f"Corresponding sample indices: {sample_row_st}~{sample_row_ed}\n"
                        f"Error detected when updating the indices of task {task_id} "
                            f"with {_sample_idxs_array_list[task_id][0]}~{_sample_idxs_array_list[task_id][-1]}")

    elif method == "random":
        selected_task_id = np.random.choice(task_num, selected_num, replace=False)
    else:
        raise ValueError(f"Invalid selection method: {method}")

    return selected_task_id

def sample_tasks2finetune(trace_root_path, learning_params, selected_num=10):
    device2task = sample_task_files(trace_root_path, learning_params["mode"], 
        learning_params["gpu_model"], absolute_path=True)
    assert len(device2task) == 1

    data_meta_info = parse_metainfo(device2task, learning_params, True)

    all_task_files = []
    for device, device_files in device2task:
        all_task_files += device_files
    all_task_files_short = np.array([os.path.basename(_f) for _f in all_task_files])

    # for selected_task_id in [
    #     # [173,144,153,119,165,127,129,133,198,140],
    #     # [91,145,163,194,148,173,185,119,164,105],
    #     # [127,131,100,180,78,143,148,186,23,141],
    #     # [149,110,25,188,121,118,117,189,83,161],
    #     # [125,138,112,155,184,120,65,192,197,88],
    #     # [164,58,177,183,152,161,146,97,135,181],
    #     # [50,102,20,171,118,128,109,105,194,140],
    #     [192,133,198,169,170,180,181,182,151,121]
    #     ]:
	#     print(",".join(_f.split(".npy")[0] for _f in all_task_files_short[selected_task_id]))
    # raise

    print(f"Loading the data of all {len(all_task_files)} tasks'  ... ")
    X = None
    sample_num = 0
    sample_idxs_array_list = []
    for _f in all_task_files:
        raw_data = load_raw_data([_f], learning_params, verbose=False)
        raw_data.metainfo = data_meta_info
        raw_data.freeze()
        raw_data.normalize()
        X_t = raw_data.x
        sample_idxs_array_list.append(np.arange(sample_num, sample_num+len(X_t)))
        if len(X_t) == 0:
            continue
        if X is None:
            X = X_t
        else:
            X = np.concatenate((X, X_t), axis=0)
        sample_num += len(X_t)
    assert sample_num == len(X)
    X = X.reshape((sample_num, -1))

    with open(".workspace/cdpp_finetune/log.txt", 'a') as fp:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        fp.write(f"\n\n\n{now}\n")
        for selected_num in [30, 40]:
            # methods = ["cmd+rm"]
            # methods = ["kmeans"]
            methods = ["kmeans", "center"] + ["random"] * 10
            # methods = ["random"] * 100
            # selected_num = 100
            for method in methods:
                selected_task_id = sample_tasks_implt(X, sample_idxs_array_list,
                    selected_num=selected_num, method=method)
                ### Output and verify the selected tasks
                diff = verify_selected_tasks(X, sample_idxs_array_list, selected_task_id)
                selected_id_str = ",".join([str(_id) for _id in selected_task_id])
                ### NOTE that the selected_task_id is not the task id in the task file name
                selected_task_name_str = ",".join(
                    [_f.split(".npy")[0] for _f in all_task_files_short[selected_task_id]])
                msg = (f"Selected {selected_num} tasks: {selected_id_str} ({selected_task_name_str}) "
                    f"using the '{method}' method, with the diff2origin of {diff}")
                print(msg)
                fp.write(msg + "\n")
