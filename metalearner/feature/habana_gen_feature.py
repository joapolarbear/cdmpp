import json
import os, sys
import re
import numpy as np
from yaml import serialize

RESERVED_TENSOR_DIM_NUM = 6
PER_TENSOR_FEATURE_LEN = 4 + 2 + RESERVED_TENSOR_DIM_NUM
MAX_INP_NUM = 3
FEATURE_LEN = 164
# FEATURE_LEN = -1
FIX_LEAF_NODE_NUM = 1
MEASURED_RECORD_FILENAME = "default_profiling.json"
META_SUFFIX = ".recipe-PostGraph-symbol.json"
TRAIN2ALL_RATIO = 0.8

ALL_ATTRS = [
    r"(?P<name>[\w:\/\-]+)",
    r"(?P<shape>\[[\d,]+\])",
    r"(?P<dtype>(float|int)(8|16|32))",
    r"zp = (?P<zp>-?\d+), scale = (?P<scale>[\d.\-e]+)",
    r"ModelParam = (?P<model_param>(0|1))"
]
ATTR_STR = r"^"
for _attr in ALL_ATTRS:
    ATTR_STR += (_attr + r"  \|  ")
ATTR_STR += r"$"

ALL_OP = []

def op2id(op):
    if op not in ALL_OP:
        ALL_OP.append(op)
    return ALL_OP.index(op)

with open(os.path.join(os.path.dirname(__file__), "ansor_feature_name.txt"), 'r') as fp:
    ANSOR_FEATURE_NAME = fp.readlines()

def match_attr(attr_str):
    ''' Match an attribute string, 
        e.g., "input_0": "input_tensor:0  |  [8,224,224,3]  |  int8  |  zp = -14, scale = 0.0187234  |  ModelParam = 0  |  ",
    '''
    try:
        rst = re.search(ATTR_STR, attr_str).groupdict()
    except:
        for attr_regex in ALL_ATTRS:
            print(re.search(attr_regex, attr_str))
        import code; code.interact(local=locals())
    rst["shape"] = eval(rst["shape"])
    rst["zp"] = eval(rst["zp"])
    rst["scale"] = eval(rst["scale"])
    rst["model_param"] = eval(rst["model_param"])
    return rst

def shape2size(_shape: list):
    ret = 1
    for _s in _shape:
        ret *= _s
    return ret

def _gen_feature_for_one_trial(trial_dir):
    node2meta = {}
    meta_filename = None
    for _file in os.listdir(trial_dir):
        if _file.endswith(META_SUFFIX):
            meta_filename = _file
            break
    assert meta_filename is not None
    with open(os.path.join(trial_dir, meta_filename), 'r') as fp:
        metadata = json.load(fp)
    for node in metadata["nodes"]:
        node2meta[node["name"]] = node

    ### Parse the measured record file
    with open(os.path.join(trial_dir, MEASURED_RECORD_FILENAME), 'r') as fp:
        traces = json.load(fp)
    pid_tid2process = {}
    pid_tid2last_trace = {}
    trace_stat = {}
    target_processes = ["*MME", "*TPC"]
    target_pids = [None] * len(target_processes)
    for trace in traces["traceEvents"]:
        pid = trace["pid"]
        tid = trace["tid"]
        if trace["ph"] == "M":
            assert trace["cat"] == "__metadata", trace
            if trace["name"] == "process_name":
                assert pid not in pid_tid2process
                pid_tid2process[pid] = {"name": trace["args"]["name"], "tids": {}}
                if trace["args"]["name"] in target_processes:
                    target_pids[target_processes.index(trace["args"]["name"])] = pid
            elif trace["name"] == "thread_name":
                assert pid in pid_tid2process
                pid_tid2process[pid]["tids"][tid] = trace["args"]["name"]
            else:
                raise ValueError(trace)
        elif trace["pid"] not in target_pids:
            ### Ignore
            continue
        elif trace["name"].startswith("MME") or trace["name"].startswith("TPC"):
            ### TODO
            continue
        elif trace["ph"] == "B":
            if pid not in pid_tid2last_trace:
                pid_tid2last_trace[pid] = {}
            pid_tid2last_trace[pid][tid] = trace
        elif trace["ph"] == "E":
            ### Get the last event of the same pid and tid
            try:
                last_trace = pid_tid2last_trace[pid][tid]
                pid_tid2last_trace[pid].pop(tid)
            except:
                print(trace)
                import code; code.interact(local=locals())
            if trace["name"] != last_trace["name"]:
                if last_trace["name"] == "MME context":
                    last_trace = None
                    continue
                import code; code.interact(local=locals())

            _dur = (trace["ts"] - last_trace["ts"]) * 1e-3
            if _dur < 0:
                import pdb; pdb.set_trace()
            core_type = target_pids.index(pid)
            if trace["name"] not in trace_stat:
                trace_stat[trace["name"]] = {
                    "dur": _dur,
                    # "op": trace["args"]["op"]
                    "core_type": core_type
                }
            else:
                trace_stat[trace["name"]]["dur"] += _dur
            last_trace = None
        else:
            raise ValueError(trace)

    # print(f"Collect traces for {len(trace_stat)} ops")
    # print(trace_stat["resnet_model/batch_normalization/FusedBatchNorm"])

    def __parse_tensor_feature(_tensor: dict):
        if _tensor["dtype"].startswith("float"):
            isfloat = 1
            byte_per_unit = int(_tensor["dtype"].split("float")[1]) / 8
        else:
            isfloat = 0
            byte_per_unit = int(_tensor["dtype"].split("int")[1]) / 8
        size_in_unit = shape2size(_tensor["shape"])
        _to_add = [size_in_unit, _tensor["zp"], _tensor["scale"], _tensor["model_param"], isfloat, byte_per_unit]

        assert len(_tensor["shape"]) <= RESERVED_TENSOR_DIM_NUM
        _to_add += _tensor["shape"] + [0] * (RESERVED_TENSOR_DIM_NUM - len(_tensor["shape"]))
        
        assert len(_to_add) == PER_TENSOR_FEATURE_LEN
        return _to_add
        
    yx_feature = []
    for node, _stat in trace_stat.items():
        if node not in node2meta:
            print(node)
        sample_y = _stat["dur"]
        sample_x = []

        _meta = node2meta[node]
        _op = _meta["op"]
        # assert _op == _stat["op"], (_stat, _meta)
        _inputs = []
        _outputs = []
        for _attr, _attr_str in _meta["attrs"].items():
            if _attr.startswith("input"):
                _inputs.append(match_attr(_attr_str))
            elif _attr.startswith("output"):
                _outputs.append(match_attr(_attr_str))

        ### Add input related features
        _inp_id = 0
        for _inp in _inputs[:MAX_INP_NUM]:
            sample_x += __parse_tensor_feature(_inp)
            _inp_id += 1
        while _inp_id < MAX_INP_NUM:
            _to_add = [0] * PER_TENSOR_FEATURE_LEN
            sample_x += _to_add
            _inp_id += 1
        
        ### Add output related features
        assert len(_outputs) == 1
        sample_x += __parse_tensor_feature(_outputs[0])
        
        ### Other information
        sample_x.append(_stat["core_type"])
        sample_x.append(op2id(_op))

        ### Pad to match the fixed feature lenghth
        if FEATURE_LEN > 0:
            if len(sample_x) < FEATURE_LEN:
                sample_x += [0] * (FEATURE_LEN - len(sample_x))
            elif len(sample_x) > FEATURE_LEN:
                sample_x = sample_x[:FEATURE_LEN]

        ### Mimic the structure of compact AST
        _std = 0.0
        _flop_ct = -1
        ast_features = np.array([sample_x] + [[0] * FEATURE_LEN] * (FIX_LEAF_NODE_NUM - 1))
        node_ids = np.arange(FIX_LEAF_NODE_NUM)
        serialized_tree = np.concatenate((
            node_ids, np.ones_like(node_ids) * -1
        ))
        yx_feature.append([sample_y, _std, _flop_ct,
            ast_features, node_ids, serialized_tree])

    print(len(yx_feature))
    return yx_feature

def traverse_gen_features(root_dir):
    all_yx_data = []
    for network_name in os.listdir(root_dir):
        if network_name.startswith("."):
            continue
        network_dir = os.path.join(root_dir, network_name)
        for device in os.listdir(network_dir):
            if device.startswith("."):
                continue
            device_dir = os.path.join(network_dir, device)
            for dtype_name in os.listdir(device_dir):
                if dtype_name.startswith("."):
                    continue
                dtype_dir = os.path.join(device_dir, dtype_name)
                for bs_name in os.listdir(dtype_dir):
                    if bs_name.startswith("."):
                        continue
                    rst = re.search(r"BS_(?P<bs>\d+)", bs_name).groupdict()
                    bs = eval(rst["bs"])
                    profile_rst_dir = os.path.join(dtype_dir, bs_name, "habana_profiler")
                    if MEASURED_RECORD_FILENAME not in os.listdir(profile_rst_dir):
                        print(f"[Warning] Profiling results under {profile_rst_dir} is missing")
                    else:
                        yx_feature = _gen_feature_for_one_trial(profile_rst_dir)
                        all_yx_data += yx_feature

    all_yx_data = np.array(all_yx_data, dtype=object)
    print(len(all_yx_data))

    return all_yx_data

def gen_habana_x_device(_len):
    DEVICE_FEATURE_HEAD = ["device_id", "clock_MHz", "memory_gb", "shm_kb", "L2cache_mb", "L1cache_kb", "BW_gbps", "fp32_tflops"]
    X_device = [-1, -1, 16, -1, -1, -1, 40, -1]
    assert len(X_device) == len(DEVICE_FEATURE_HEAD), (len(X_device), len(DEVICE_FEATURE_HEAD))
    return np.array([X_device] * _len)

root_dir = sys.argv[1]
output_dir = sys.argv[2]
all_yx_data = traverse_gen_features(root_dir)
print("Shape: ", all_yx_data[0, 3].astype(float).shape)

### Train and test split

os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "0.npy"), all_yx_data, 
    allow_pickle=True)
