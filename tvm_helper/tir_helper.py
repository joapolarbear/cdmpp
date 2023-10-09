import os, sys
import re
import numpy as np

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler.measure import recover_measure_input

### Tenset
from measure_programs import parse_cost, INVALID_TIME_UPPER

from utils.device_info import short_device_name


VECTOR_RE = "[a-zA-Z\d\._]+\[[^\[\]=]+\]"
ALL_TIR_OPS = [
    "add",
    "negative",
    "multiply",
    "divide",
    "sqrt",
    "expand_dims",
    "relu",
    "reshape",
    "allocate",
    "assign",
    "subtract",
    "exp",
    "concat",
    "batch_matmul_NT",
]
TIR_OP_ALIAS = {
    "+": "add",
    "-": "negative",
    "*": "multiply",
    "/": "divide"
}
ANNOTATION_DIM_LEN = 12
FEATURE_PER_TIR_OP = [
    "loop_steps",
    "loop_cnt",
]

class IteratorAnnotation:
    ### This iterator has no annotation. */
    kNone = 0
    ### This iterator has been unrolled. */
    kUnroll = 1
    ### This iterator has been vectorized. */
    kVectorize = 2
    ### This iterator has been paralleld. */
    kParallel = 3
    ### This iterator has been bind to vthread. */
    kVThread = 4
    ### This iterator has been bind to blockIdx.x. */
    kBlockX = 5
    ### This iterator has been bind to threadIdx.x. */
    kThreadX = 6
    ### This iterator has been bind to blockIdx.y. */
    kBlockY = 7
    ### This iterator has been bind to threadIdx.y. */
    kThreadY = 8
    ### This iterator has been bind to blockIdx.y. */
    kBlockZ = 9
    ### This iterator has been bind to threadIdx.y. */
    kThreadZ = 10
    ### This iterator has been mapped with a tensorize intrinsic. */
    kTensorize = 11

### pre additional dimensions
pre_add_dims = ["avg", "std", "FLOPs"]
# pre_add_dims = ["avg"]
def feature_lable_len(_pre_add_dims):
    return len(ALL_TIR_OPS) * (len(FEATURE_PER_TIR_OP)+ANNOTATION_DIM_LEN) + len(_pre_add_dims)

def feature_len():
    return len(ALL_TIR_OPS) * (len(FEATURE_PER_TIR_OP)+ANNOTATION_DIM_LEN)

def feature_shape2d():
    return (len(ALL_TIR_OPS), len(FEATURE_PER_TIR_OP)+ANNOTATION_DIM_LEN)

def init_features():
    # return np.zeros((len(ALL_TIR_OPS), len(FEATURE_PER_TIR_OP)), dtype=float)
    return np.zeros(feature_shape2d(), dtype=float)

class loopInfo:
    def __init__(self):
        self.clear()

    def clear(self):
        self.loop_steps = []
        self.cached_iter = None
        self.cache_used = False

    def step_into_loop(self, steps, annotate_idx=None):
        iter_info = {
            "steps": steps,
        }
        if self.cached_iter is None:
            iter_info["iter_annot"] = [1.] * ANNOTATION_DIM_LEN
        else:
            ### If there is some cached iteration annotations, they are used to describe the loop
            iter_info["iter_annot"] = self.cached_iter
        self.cached_iter = None
        self.cache_used = True

        if annotate_idx:
            iter_info["iter_annot"][annotate_idx] *= steps

        self.loop_steps.append(iter_info)

    def step_out_loop(self):
        self.cached_iter = None
        self.cache_used = False
        self.loop_steps.pop(-1)
    
    def cache_iter_annot(self, steps, annotate_idx):
        if self.cached_iter is None or self.cache_used:
            self.cached_iter = [1.] * ANNOTATION_DIM_LEN
            self.cache_used = False
        self.cached_iter[annotate_idx] *= steps

    def searialize_to_feature(self, tir_feature, tir_op_idx, ramp_lane):
        '''
        tir_feature: 2d-array, shape=(# of tir ops, # of feature per tir op)
        '''
        total_steps = 1
        total_annotation_steps = [1.] * ANNOTATION_DIM_LEN
        for _loop in self.loop_steps:
            total_steps *= _loop["steps"]
            for idx in range(ANNOTATION_DIM_LEN):
                total_annotation_steps[idx] *= _loop["iter_annot"][idx]
        if self.cached_iter is not None:
            for idx in range(ANNOTATION_DIM_LEN):
                total_annotation_steps[idx] *= self.cached_iter[idx]
            self.cache_used = True
        tir_feature[tir_op_idx, FEATURE_PER_TIR_OP.index("loop_steps")] += total_steps * ramp_lane
        tir_feature[tir_op_idx, FEATURE_PER_TIR_OP.index("loop_cnt")] += len(self.loop_steps)
        for idx in range(ANNOTATION_DIM_LEN):
            tir_feature[tir_op_idx, len(FEATURE_PER_TIR_OP) + idx] += total_annotation_steps[idx]

def match_loop(_str):
    match = re.search(r"\([a-zA-Z\d\._]+, (?P<start>\d+), (?P<end>\d+)\)", _str)
    rst = match.groupdict()
    return float(rst["end"]) - int(rst["start"]) + 1.

def match_cuda_iter_annot(_str):
    ''' Match iteration annotations related to cuda
    Samples:
    // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 2
    // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 1024
    '''
    match = re.search(r"// attr \[iter_var\((?P<cuda_para>(blockIdx|threadIdx))\.(?P<dim>(x|y|z)).* thread_extent = (?P<extent>[\d]+)", _str)
    rst = match.groupdict()
    
    if rst["cuda_para"] == "blockIdx":
        if rst["dim"] == "x":
            iter_annot = IteratorAnnotation.kBlockX
        elif rst["dim"] == "y":
            iter_annot = IteratorAnnotation.kBlockY
        elif rst["dim"] == "z":
            iter_annot = IteratorAnnotation.kBlockZ
        else:
            raise ValueError(_str, rst)
    elif rst["cuda_para"] == "threadIdx":
        if rst["dim"] == "x":
            iter_annot = IteratorAnnotation.kThreadX
        elif rst["dim"] == "y":
            iter_annot = IteratorAnnotation.kThreadY
        elif rst["dim"] == "z":
            iter_annot = IteratorAnnotation.kThreadZ
        else:
            raise ValueError(_str, rst)
    else:
        raise ValueError(_str, rst)
    return float(rst["extent"]), iter_annot

def _parse_tir_feature_internal(var_tir_by_lines):
    state = 0 ### out of a loop
    loop_info = loopInfo()
    tir_feature = init_features()
    brackets = []
    for var_line in var_tir_by_lines:
        # print(var_line)
        var_line = var_line.strip()
        if var_line.startswith("parallel"):
            brackets.append("iter")
            _steps = match_loop(var_line)
            loop_info.step_into_loop(_steps, annotate_idx=IteratorAnnotation.kParallel)
        elif var_line.startswith("unrolled"):
            brackets.append("iter")
            _steps = match_loop(var_line)
            loop_info.step_into_loop(_steps, annotate_idx=IteratorAnnotation.kUnroll)
        elif var_line.startswith("for"):
            brackets.append("iter")
            _steps = match_loop(var_line)
            loop_info.step_into_loop(_steps, annotate_idx=None)
        elif var_line.startswith("//"):
            if var_line.startswith("// attr [iter_var"):
                ### e.g. // attr [iter_var(blockIdx.z, , blockIdx.z)] thread_extent = 8
                extent, iter_annot = match_cuda_iter_annot(var_line)
                loop_info.cache_iter_annot(extent, iter_annot)
            elif var_line.startswith("// attr [comm_reducer"):
                ### TODO (huhanpeng) ho to handle if reduce iteration
                pass
            else:
                raise ValueError(var_line)
        elif var_line.startswith("if"):
            ### TODO (huhanpeng) ho to handle if condition
            loop_info.cached_iter = None
            brackets.append("if")
            pass
        elif var_line.startswith("tir.tvm_thread_allreduce"):
            ### TODO (huhanpeng) ho to handle if tir.tvm_thread_allreduce
            loop_info.cached_iter = None
            pass
        elif var_line.startswith("}"):
            if brackets.pop(-1) == "iter":
                loop_info.step_out_loop()
            state -= 1
            assert state >= 0, (state, var_line, var_tir_by_lines)
            if state == 0:
                ### end of the scope of one var
                ### clear loop info
                loop_info.clear()
        else:
            ### TIR OPs
            ### Parse ramp expr first
            match = re.search(r"ramp\([\(\)a-zA-Z\d\*\+\. ]+, (?P<stride>\d+), (?P<lane>\d+)\)", var_line)
            if match is not None:
                ramp_lane = int(match.groupdict()["lane"])
            else:
                ramp_lane = 1
            if var_line.startswith("T_") and not var_line.startswith("T_matmul"):
                match = re.search(r"T_(?P<tir_op>[a-zA-Z_\d]+)", var_line)
                tir_ops = [match.groupdict()["tir_op"]]
            elif var_line.startswith("allocate"):
                tir_ops = ["allocate"]   
            elif re.match("^" + VECTOR_RE + " = " + VECTOR_RE, var_line):
                tir_ops = ["assign"]
            else:
                tmp = var_line.split(" = ")[1]
                match = re.findall(VECTOR_RE, tmp)
                for sub_str in match:
                    tmp = tmp.replace(sub_str, "")
                tir_ops = []
                for char in tmp:
                    if char in ['+', '-', '*', '/']:
                        tir_ops.append(TIR_OP_ALIAS[char])     
            for tir_op in tir_ops:
                # assert tir_op in ALL_TIR_OPS, (tir_op, var_line, var_tir_by_lines)
                if tir_op not in ALL_TIR_OPS:
                    print(f"Warning: {tir_op} not in ALL_TIR_OPS")
                    continue
                tir_op_idx = ALL_TIR_OPS.index(tir_op)
                loop_info.searialize_to_feature(tir_feature, tir_op_idx, ramp_lane)
        if var_line.endswith("{"):
            state += 1
    return tir_feature

def parse_operator_module(op_mod_str):
    ### The input ir is the subgraph generated in the process of AutoTVM
    var_tir_by_line = op_mod_str.split("\n")
    return _parse_tir_feature_internal(var_tir_by_line[2:-2])

def parse_tir_feature_impl(var_tir_str):
    var_tir_by_line = var_tir_str.split("\n")
    op_name = var_tir_by_line[0].split("):")[0]
    tir_feature = _parse_tir_feature_internal(var_tir_by_line[1:-2])
    return op_name, tir_feature

def parse_tir_feature(file):
    with open(file, 'r') as fp:
        tir_all = fp.read()

    tir_by_var = tir_all.split("GlobalVar(")[1:]
    tir_op_features = []
    for var_tir in tir_by_var:
        ### for one var or op
        op_name, tir_feature = parse_tir_feature_impl(var_tir)
        tir_op_features.append((op_name, tir_feature))
        # if op_name == "tvmgen_default_fused_nn_conv2d_2":
        #     print(op_name, tir_feature)
        #     raise
    return tir_op_features

###########################################################################

def reshape_tir_feature(old_tir_op_num, old_feature_per_tir_op_num, trace_dir, add_dim_len):
    ''' Due to lack of scalability of the feature design, 
        reconstuct the features after tunning # of TIR OP or
        # of FEATURE_PER_TIR_OP  
    '''
    root_path, _, files = list(os.walk(trace_dir))[0]
    xydata = None
    feature_type="cus-feature"

    loop_info = loopInfo()

    for file in files:
        if not file.endswith(".npy"):
            continue
        if "norm-max" in file:
            continue
        if feature_type is not None and feature_type not in file:
            continue
        data = np.load(os.path.join(root_path, file))
        print(f"Read np.array of shape {data.shape} from {file}")
        split_data = np.split(data, [1+add_dim_len], axis=1)
        y_train = split_data[0]
        try:
            x_train = split_data[1].reshape((len(y_train), old_tir_op_num, old_feature_per_tir_op_num))
        except:
            print("[Warning] shape doesn't match. Skip the file")
            continue
        if old_tir_op_num != len(ALL_TIR_OPS):
            res = np.zeros((len(y_train), len(ALL_TIR_OPS)-old_tir_op_num, old_feature_per_tir_op_num), dtype=float)
            x_train = np.concatenate((x_train, res), axis=1)
        if old_feature_per_tir_op_num != len(FEATURE_PER_TIR_OP):
            raise NotImplementedError()
        all_data = np.concatenate(
            (y_train, x_train.reshape(len(y_train), -1)),
            axis=1)
        print(f"Result data shape {all_data.shape}")
        save_path = os.path.join(root_path, "v2_" + file)
        np.save(save_path, all_data)

def wrap_get_per_store_features_from_states(states, task, parse_ast=True, delete_invalid=True):
    std_flop_ct_list = [(0, task.compute_dag.flop_ct)] * len(states)

    ### shape=(# of states, 3)
    features_ = auto_scheduler.feature.get_per_store_features_from_states(
        states, task, parse_ast=parse_ast)
    
    ### Flatten features
    flatten_feature = []
    failed_id = []
    for i in range(len(features_)):
        ast_features, node_ids, serialized_tree = features_[i]
        if len(serialized_tree) == 0:
            ### Failed during lowering
            ### TODO: We have filter out invalid inputs from `inputs_to_test`, but some times lowering specific schedules fail
            failed_id.append(i)
            flatten_feature.append((None, None, None))
        else:
            assert ast_features.shape[1] == 1, (ast_features.shape, i, std_flop_ct_list[i], node_ids, serialized_tree)
            ast_features = np.sum(ast_features, 1)
            flatten_feature.append((ast_features, node_ids, serialized_tree))
    flatten_feature = np.array(flatten_feature, dtype=object)
    ret = np.concatenate((np.array(std_flop_ct_list), flatten_feature), axis=1)
    if delete_invalid:
        return np.delete(ret, failed_id, axis=0)
    else:
        return ret
        
def wrap_get_per_store_features_from_measure_pairs(
        inputs, results,
        get_workload_embedding=False,
        verbose=False,
        parse_ast=False,
        delete_invalid=True,
        lock=None):
    ''' A wrapper of auto_scheduler.feature.get_per_store_features_from_measure_pairs
        Return X, Y pair parsed from the inputs and results pair
        For inputs that fail to be lowered, they are removed from the returned X, Y pairs
    '''
    # ### Debug
    # for i in range(len(inputs)):
    #     print(i)
    #     features_, normalized_throughputs, task_ids, min_latency =\
    #         auto_scheduler.feature.get_per_store_features_from_measure_pairs([inputs[i]], [results[i]], parse_ast=True)
    #     res = results[i]
    #     dur, std = parse_cost(res, verbose)
    #     ast_features, node_ids, serialized_tree = features_[0]
    #     print(np.array(ast_features).shape, len(node_ids), len(serialized_tree), dur, std)
    # raise

    ### Check average time, standard deviation and flops, 
    # then decide if to extract features from a record
    std_flop_ct_list = []
    dur_list = []
    inputs_to_test = []
    results_to_test = []
    for i in range(len(inputs)): 
        res = results[i]
        dur, std = parse_cost(res, verbose)
        if dur is None:
            continue

        inputs_to_test.append(inputs[i])
        results_to_test.append(results[i])

        inp = inputs[i]
        inp = recover_measure_input(inp, True)
        flop_ct = inp.task.compute_dag.flop_ct
        std_flop_ct_list.append([std, flop_ct])
        dur_list.append(dur)

    ### Exctract features
    # For TVM 0.9.0, min_latency is removed from the returned resutls
    if lock is not None:
        lock.acquire()
    ret = auto_scheduler.feature.get_per_store_features_from_measure_pairs(inputs_to_test, results_to_test, parse_ast=parse_ast)
    if lock is not None:
        lock.release()
    features_, normalized_throughputs, task_ids = ret[:3]
            
    assert not np.any(task_ids)   # all task ids should be zero
    if get_workload_embedding:
        ## Add task embedding into the feature, See XGB_Model
        task_embedding = auto_scheduler.cost_model.xgb_model.get_workload_embedding(inp.task.workload_key)

    ### Flatten features
    flatten_feature = []
    failed_id = []
    if parse_ast:
        for i in range(len(features_)):
            ast_features, node_ids, serialized_tree = features_[i]
            if len(serialized_tree) == 0:
                ### Failed during lowering
                ### TODO: We have filter out invalid inputs from `inputs_to_test`, but some times lowering specific schedules fail
                failed_id.append(i)
                flatten_feature.append((None, None, None))
            else:
                assert ast_features.shape[1] == 1, (ast_features.shape, i, dur_list[i], std_flop_ct_list[i], node_ids, serialized_tree)
                ast_features = np.sum(ast_features, 1)
                flatten_feature.append((ast_features, node_ids, serialized_tree))
        flatten_feature = np.array(flatten_feature, dtype=object)
        assert len(flatten_feature) == len(dur_list)
    else:
        ### features_'s datatype = object, and each element's shape = (?, 164)
        for i in range(len(features_)):
            if get_workload_embedding:
                tmp = np.tile(task_embedding, (len(features_[i]), 1))
                _feature = np.concatenate([features_[i], tmp], axis=1)
            else:
                _feature = features_[i]
            flatten_feature.append(np.sum(_feature, 0))
        flatten_feature = np.array(flatten_feature)
        assert len(flatten_feature) == len(dur_list)
    
    X = np.concatenate((np.array(std_flop_ct_list), flatten_feature), axis=1)
    Y = np.array(dur_list)
    if delete_invalid:
        return np.delete(X, failed_id, axis=0), np.delete(Y, failed_id, axis=0)
    else:
        return X, Y

STR2TARGET4CPU = {
    "e5-2673": "llvm -mcpu=core-avx2 -model=e5-2673",
    "epyc-7452": "llvm -mcpu=core-avx2 -model=epyc-7452",
    "platinum-8272": "llvm -mcpu=skylake-avx512 -model=platinum-8272",
    "graviton2": "llvm -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod -model=graviton2",
}

def device_str2target(device: str):
    if device in STR2TARGET4CPU:
        is_cuda_device = False
        return tvm.target.Target(STR2TARGET4CPU[device])
    is_cuda_device = True
    short_name = device
    try:
        short_name = short_device_name(device.upper()).lower() 
    except:
        is_cuda_device = False
    if is_cuda_device:
        return tvm.target.cuda(short_name)
    else:
        raise ValueError(device)

if __name__ == "__main__":
    tir_op_features = parse_tir_feature(sys.argv[1])
    tir_ops, _ = zip(*tir_op_features)
    rst = [op for op in tir_ops if "conv2d" in op]
    print(len(rst))
    print(rst)
