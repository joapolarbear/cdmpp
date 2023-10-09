import numpy as np
t4_0 = np.load("t4/0.npy", allow_pickle=True)
a10_0 = np.load("a10/0.npy", allow_pickle=True)

import os
import json
import sys
sys.path.append("3rdparty/tenset/scripts")
from common import (load_and_register_tasks, get_task_info_filename,
    get_measure_record_filename)
import matplotlib.pyplot as plt
import numpy as np
ALL_TASKS = load_and_register_tasks()
flop_cts = np.array([task.compute_dag.flop_ct for task in ALL_TASKS])
x = np.arange(len(ALL_TASKS))

### Plot task id to FLOPs
fig = plt.figure(figsize=(12, 8))
fig_base = 330
batch_cnt = 9
batch_size = len(x) // batch_cnt
for i in range(batch_cnt):
    fig_base += 1
    ax = fig.add_subplot(fig_base)
    st = i*batch_size
    ed = (i+1)*batch_size if i < batch_cnt - 1 else len(x)
    ax.scatter(x[st:ed], np.log10(np.maximum(1, flop_cts[st:ed])))
    plt.xlabel("TASK IDs")
    plt.ylabel("FLOPs (in log)")
plt.tight_layout()
plt.savefig("tmp.png")

### check networks that have tasks with flop_ct=-1
from dump_network_info import map_task2network, get_network_with_key
task2network = map_task2network()
network2invalid_tasks = {}
invalid_task_ids = set()
for _id, task in enumerate(ALL_TASKS):
    if task.compute_dag.flop_ct == -1:
        for _n in task2network[task.workload_key]:
            ### e.g. _n = '(resnet_50,[(8,3,256,256)]),cuda'
            # _key = _n[1:].split(",")[0]+"_bs"+_n.split(",")[1].split("(")[1]
            _key = _n
            if _key not in network2invalid_tasks:
                network2invalid_tasks[_key] = set()
            network2invalid_tasks[_key].add(_id)
        invalid_task_ids.add(_id)

### e.g. ['(mobilenet_v3,[(1,3,224,224)]),cuda', ...]
networks_to_correct = sorted(list(network2invalid_tasks.keys()), 
    key=lambda _n: len(network2invalid_tasks[_n]), reverse=True)

import re
import tvm
import tvm.auto_scheduler as auto_scheduler

def task2workload_hash_args(_task):
    workload_hash, workload_args = auto_scheduler.utils.decode_workload_key(_task.workload_key)
    return f"{workload_hash}-{workload_args}"

_network = networks_to_correct[0]
rst = re.search(r"\((?P<network_name>\w+),(?P<args>\[.*\])\),(?P<device>\w+)", _network).groupdict()
network_key = (rst["network_name"], eval(rst["args"]))

mod, params, inputs = get_network_with_key(network_key)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, tvm.target.Target(rst["device"]))

print(network2invalid_tasks[_network])
print([ALL_TASKS[task_id_to_correct].workload_key for task_id_to_correct in network2invalid_tasks[_network]])
for task in tasks:
    for task_id_to_correct in network2invalid_tasks[_network]:
        if task2workload_hash_args(task) == task2workload_hash_args(ALL_TASKS[task_id_to_correct]):
            ### Found tasks to correct
            print("Hi", task.compute_dag.flop_ct)
            break

measure_dir = "3rdparty/tenset/scripts/dataset/test_measure_records/a10/"
measure_stat_path = os.path.join(measure_dir, "measure_stat.json")
with open(measure_stat_path, 'r') as fp:
     measure_stat = json.load(fp)
for _stat in measure_stat:
    if _stat is None:
        continue
    task_idx = _stat["task_idx"]


