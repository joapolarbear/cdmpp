from collections import defaultdict, namedtuple
import pickle
import os

import tvm
from tvm import relay, auto_scheduler
from tvm.auto_scheduler.utils import to_str_round

####################################
##### Network Utilities
####################################
def convert_to_nhwc(mod):
    """Convert to NHWC layout"""
    desired_layouts = {
        "nn.conv2d": ["NHWC", "default"],
        "nn.conv3d": ["NDHWC", "default"],
    }
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod

# The format for a line in the results file
BenchmarkRecord = namedtuple("BenchmarkRecord",
                             ['device', 'backend', 'workload_type', 'workload_name',
                              'library', 'algorithm', 'value', 'time_stamp'])

def log_line(record, out_file):
    with open(out_file, 'a') as fout:
        fout.write("\t".join([to_str_round(x) for x in record]) + '\n')


####################################
##### Dataset Utilities
####################################
SCRIPT_DIR = os.path.dirname(__file__)
NETWORK_INFO_FOLDER = os.path.join(SCRIPT_DIR, 'dataset/network_info')
TO_MEASURE_PROGRAM_FOLDER = os.path.join(SCRIPT_DIR, 'dataset/to_measure_programs')
MEASURE_RECORD_FOLDER = os.path.join(SCRIPT_DIR, 'dataset/measure_records')
TEST_MEASURE_RECORD_FOLDER = os.path.join(SCRIPT_DIR, 'dataset/test_measure_records')
COMPARE_MEASURE_RECORD_FOLDER = os.path.join(SCRIPT_DIR, 'dataset/compare_measure_records')

def clean_name(x):
    x = str(x)
    x = x.replace(" ", "")
    x = x.replace('"', '')
    x = x.replace("'", '')
    return x

def get_relay_ir_filename(network_key):
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_key)}.relay.pkl"

def get_task_info_filename(network_key, target):
    network_task_key = (network_key,) + (str(target.kind),)
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_task_key)}.task.pkl"

def get_to_measure_filename(task):
    task_key = (task.workload_key, str(task.target.kind))
    return f"{TO_MEASURE_PROGRAM_FOLDER}/{clean_name(task_key)}.json"

def get_measure_record_filename(task, target=None):
    target = target or task.target
    task_key = (task.workload_key, str(target.kind))
    return f"{MEASURE_RECORD_FOLDER}/{target.model}/{clean_name(task_key)}.json"

def get_test_measure_record_filename(task, target=None):
    target = target or task.target
    task_key = (task.workload_key, str(target.kind))
    return f"{TEST_MEASURE_RECORD_FOLDER}/{target.model}/{clean_name(task_key)}.json"

def get_compare_measure_record_filename(task, target=None):
    target = target or task.target
    task_key = (task.workload_key, str(target.kind))
    return f"{COMPARE_MEASURE_RECORD_FOLDER}/{target.model}/{clean_name(task_key)}.json"

def load_and_register_tasks():
    tasks = pickle.load(open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "rb"))

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)

    return tasks


####################################
##### Other Utilities
####################################

def dtype2torch(x):
    import torch

    return {
        'float32': torch.float32
    }[x]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

