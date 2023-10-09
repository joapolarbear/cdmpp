'''Unify task ids, making the mapping the same as that in Tenset'''
import os
import sys
import json

import tvm

from common import load_and_register_tasks, get_measure_record_filename

output_dir = os.path.abspath(sys.argv[1])
assert os.path.isdir(output_dir)
fixed_output_dir = os.path.abspath(sys.argv[1]) + "_fixed"
os.makedirs(fixed_output_dir, exist_ok=True)

tenset_tasks = load_and_register_tasks()
tenset_workload_key2id = {}
target = tvm.target.cuda(os.path.basename(output_dir))
for task_id, task in enumerate(tenset_tasks):
    tenset_workload_key2id[os.path.basename(get_measure_record_filename(task, target=target))] = task_id

### Fix split_stat.json
with open(os.path.join(output_dir, 'split_stat.json'), 'r') as fp:
    split_stat = json.load(fp)
new_split_stat = {}
extract_task_id2tenset_task_id = {}
for extract_task_id in split_stat:
    workload_key_device = split_stat[str(extract_task_id)]["workload_key"]
    tenset_task_id = tenset_workload_key2id[workload_key_device]
    new_split_stat[tenset_task_id] = split_stat[extract_task_id]
    extract_task_id2tenset_task_id[extract_task_id] = tenset_task_id
with open(os.path.join(fixed_output_dir, 'split_stat.json'), 'w') as fp:
    json.dump(new_split_stat, fp, indent=4)

### Fix the mapping from tasks to split files
# network2split[network_name][bs] = [task ids]
with open(os.path.join(output_dir, "network2split.json"), 'r') as fp:
    network2split = json.load(fp)
new_network2split = {}
for network_name in network2split:
    new_network2split[network_name] = {}
    for bs in network2split[network_name]:
        new_network2split[network_name][bs] = [extract_task_id2tenset_task_id[_id] 
            for _id in network2split[network_name][bs]]
with open(os.path.join(fixed_output_dir, "network2split.json"), 'w') as fp:
    json.dump(new_network2split, fp, indent=4)

for f in os.listdir(output_dir):
    if not f.endswith(".npy"):
        continue
    fixed_filename = str(extract_task_id2tenset_task_id[f.split(".npy")[0]]) + ".npy"
    # print(f, fixed_filename)
    os.rename(
        os.path.join(output_dir, f),
        os.path.join(fixed_output_dir, fixed_filename)
    )
    