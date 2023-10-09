import os
import sys
import json
import re
from tqdm import tqdm
import numpy as np

def check_task_ids(task_ids, split_stat):
    ''' Check which tasks are missing
    '''
    task_ids = sorted(task_ids)
    missed_task_ids = []
    for i in range(1, len(task_ids)):
        if task_ids[i-1] + 1 != task_ids[i]:
            for task_id in range(task_ids[i-1]+1, task_ids[i]):
                split_name = f"{task_id}"
                if split_name in split_stat and split_stat[split_name]["error"] is not None:
                    pass
                else:
                    missed_task_ids.append(task_id)
    print(f"Miss {len(missed_task_ids)} task ids: {missed_task_ids}")
    print(f"Total task number: {task_ids[-1]+1}")

if True:
    output_dir = os.path.abspath(sys.argv[1])
    assert os.path.isdir(output_dir)
    task_ids = [int(f.split(".npy")[0]) for f in os.listdir(output_dir) if f.endswith(".npy")]

    if os.path.exists(os.path.join(output_dir, 'split_stat.json')):
        with open(os.path.join(output_dir, 'split_stat.json'), 'r') as fp:
            split_stat = json.load(fp)
    else:
        split_stat = {}

    # ### write split_id and corresonding tasks to files, separated with \t
    # with open(os.path.join(output_dir, "split2network.txt"), 'w') as fp:
    #     for split_name in split_stat.keys():
    #         fp.write(f"{split_name}\t{split_stat[split_name]['networks']}\n")
    
    ### save the mapping from tasks to split files
    network2split = {}
    for split_name in split_stat.keys():
        for network_str in split_stat[split_name]["networks"]:
            ### e.g., network_str = "(densenet_121,[(1,3,224,224)]),cuda"
            rst = re.search(r"^\((?P<network_name>\w+),\[\((?P<bs>\d+),.+\)\]\),(?P<device>\w+)$", network_str).groupdict()
            network_name = rst["network_name"]
            bs = rst["bs"]
            device = rst["device"]
            if network_name not in network2split:
                network2split[network_name] = {}
            if bs not in network2split[network_name]:
                network2split[network_name][bs] = set()
            network2split[network_name][bs].add(split_name)

    for network_name in network2split.keys():
        for bs in network2split[network_name]:
            network2split[network_name][bs] = list(network2split[network_name][bs])

    with open(os.path.join(output_dir, "network2split.json"), 'w') as fp:
        json.dump(network2split, fp, indent=4)
    
    check_task_ids(task_ids, split_stat)

# exit(0)


if False:
    import numpy as np
    sample_features = ["avg", "std", "flops", "ast_features", "node_ids", "serialized_tree"]

    def compare_two_split(f1, f2):
        a1 = np.load(f1, allow_pickle=True) # shape = (3709, 6)
        a2 = np.load(f2, allow_pickle=True)
        ### compare feature by feature
        for bias in range(len(sample_features)):
            if isinstance(a1[0][bias], np.ndarray):
                if a1[0][bias].shape != a2[0][bias].shape:
                    print(f"Shape does NOT match over '{sample_features[bias]}': {a1[0][bias].shape} vs {a2[0][bias].shape}")
                    continue
            mean_abs_diff = np.mean(np.abs(np.array([s[bias] for s in a1]) - np.array([s[bias] for s in a2])))
            print(f"ABS MEAN Diff of '{sample_features[bias]}': {mean_abs_diff}")
            abs_mean_diff = np.abs(np.mean([s[bias] for s in a1]) - np.mean([s[bias] for s in a2]))
            print(f"MEAN ABS Diff of '{sample_features[bias]}': {abs_mean_diff}")

    split_id = 0
    f1 = f'ast_ansor/t4_{split_id}.npy'
    f2 = f'ast_ansor2/t4_{split_id}.npy'
    compare_two_split(f1, f2)

    f1 = 'ast_ansor/t4_0.npy'
    f2 = 'ast_ansor/t4_1.npy'


### Generate split stat info
if False:
    _dir = ".workspace/ast_ansor2"
    with open(os.path.join(_dir, 't4_split2task.json'), 'r') as fp:
        split2task = json.load(fp)

    with open(os.path.join(_dir, 't4_invalid_tasks.json'), 'r') as fp:
        invalid_tasks = json.load(fp)

    if os.path.exists(os.path.join(_dir, 'split_stat.json')):
        with open(os.path.join(_dir, 'split_stat.json'), 'r') as fp:
            split_stat = json.load(fp)
    else:
        split_stat = {}

    for split_name in tqdm(split2task.keys()):
        
        if split_name in split_stat and \
                not (split_stat[split_name]["shape"] is None and split_stat[split_name]["error"] is None):
            ### Skip
            continue

        split_path = os.path.join(_dir, f"{split_name}.npy")

        if os.path.exists(split_path):
            data = np.load(split_path, allow_pickle=True)
            data_shape = data.shape
        else:
            data_shape = None
        
        if split_name in invalid_tasks:
            error_msg = invalid_tasks[split_name]["error"]
        else:
            error_msg = None
        
        workload_key, networks = split2task[split_name]

        if split_name not in split_stat:
            split_stat[split_name] = {
                "workload_key": workload_key,
                "networks": networks,
                "shape": data_shape,
                "error": error_msg
            }
        elif (split_stat[split_name]["shape"] is None and split_stat[split_name]["error"] is None):
            assert "workload_key" in split_stat[split_name] and \
                "networks" in split_stat[split_name], (split_name, split_stat[split_name])
            split_stat[split_name]["shape"] = data_shape
            split_stat[split_name]["error"] = error_msg

    with open(os.path.join(_dir, 'split_stat.json'), 'w') as fp:
        json.dump(split_stat, fp, indent=4)

