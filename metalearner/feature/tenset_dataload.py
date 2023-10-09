
''' Make sure to use old version of tvm

Load data from Tenset

E.g,

python3 metalearner/feature/tenset_dataload.py \
    -o .workspace/ast_ansor2/t4 \
    -i 3rdparty/tenset/scripts/dataset/measure_records/t4/ \
    -s 0 \
    --reload \
    -j 8 \

'''
import argparse
import enum
from tqdm import tqdm
import numpy as np
import re
import os
import json
import pickle
import threading
from typing import Any, List, NamedTuple

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler import ComputeDAG, LayoutRewriteOption
from tvm.auto_scheduler.measure import recover_measure_input
from tvm.auto_scheduler.workload_registry import workload_key_to_tensors

from common import (load_and_register_tasks, get_task_info_filename,
    get_measure_record_filename)

from tvm_helper.tir_helper import (
    ALL_TIR_OPS,
    loopInfo,
    init_features,
    wrap_get_per_store_features_from_measure_pairs,
    parse_cost,
    device_str2target
)

from utils.util import notify, read_yes, warning

print("Load tasks...")
ALL_TASKS = load_and_register_tasks()

'''
enum class IteratorKind : int {
  /*! \brief Spatial iterator. */
  kSpatial = 0,
  /*! \brief Reduction iterator. */
  kReduction = 1,
  /*! \brief Fused spatial and reduction iterator. */
  kMixed = 2,
  /*! \brief Special iterator. (e.g. virtual root iterator) */
  kSpecial = 3
};

/*! \brief The type of an iterator's annotation. */
enum class IteratorAnnotation : int {
  /*! \brief This iterator has no annotation. */
  kNone = 0,
  /*! \brief This iterator has been unrolled. */
  kUnroll = 1,
  /*! \brief This iterator has been vectorized. */
  kVectorize = 2,
  /*! \brief This iterator has been paralleld. */
  kParallel = 3,
  /*! \brief This iterator has been bind to vthread. */
  kVThread = 4,
  /*! \brief This iterator has been bind to blockIdx.x. */
  kBlockX = 5,
  /*! \brief This iterator has been bind to threadIdx.x. */
  kThreadX = 6,
  /*! \brief This iterator has been bind to blockIdx.y. */
  kBlockY = 7,
  /*! \brief This iterator has been bind to threadIdx.y. */
  kThreadY = 8,
  /*! \brief This iterator has been bind to blockIdx.y. */
  kBlockZ = 9,
  /*! \brief This iterator has been bind to threadIdx.y. */
  kThreadZ = 10,
  /*! \brief This iterator has been mapped with a tensorize intrinsic. */
  kTensorize = 11
};
'''

def parse_tenset_per_inp(inp, res, verbose=False):
    dur, std = parse_cost(res, verbose)
    if dur is None:
        return None, None, "Timeout during run"

    inp = recover_measure_input(inp, True)
    flop_ct = inp.task.compute_dag.flop_ct
    tir_feature = init_features()
    
    # sch, args = inp.task.compute_dag.apply_steps_from_state(
    #     inp.state, layout_rewrite=None
    # )
    # print("Lowered TIR:")
    # print(tvm.lower(sch, args, simple_mode=True))
    # raise

    for idx, stage in enumerate(inp.state.stages):
        if stage.op_type == 0:
            ### Placeholder
            continue

        # stage.op.input_tensors
        # stage.op.num_outputs   
        # stage.op.tag
        # stage.op.attrs
        # stage.op.reduce_axis
        # stage.op.handle
        # stage.op.output()
        # stage.op.same_as()

        # print(f"\nStage {idx}: op_type={stage.op_type}")
        # print(f"    op_name={stage.op.name}")
        # print(f"    op_body={stage.op.body}")
        # print(f"    op_axis={stage.op.axis}")
        # for iter in stage.iters:
        #     print(f"        name={iter.name}, kind={iter.iter_kind}, anot={iter.annotation}, range={iter.range}, extent={iter.range.extent}")
        

        if stage.op.name.startswith("T_"):
            match = re.search(r"T_(?P<tir_op>[a-zA-Z_\d]+)", stage.op.name)
            tir_ops = [match.groupdict()["tir_op"]]
        else:
            ### TODO (huhanpeng)
            print(stage.op.name, stage.op.body)
            tir_ops = ["assign"]
        
        loop_info = loopInfo()

        for iter in stage.iters:
            try:
                _steps = iter.range.extent.value
            except:
                ### TODO (huhanpeng)
                # isinstance(iter.range.extent, tvm.tir.expr.Add)
                return None, None, "Fail to convert extent to int"
            loop_info.step_into_loop(_steps, iter.annotation)
        
        for tir_op in tir_ops:
            if tir_op == "multiply_red" or tir_op == "dense":
                tir_op = "multiply"
            # assert tir_op in ALL_TIR_OPS, (tir_op, var_line, var_tir_by_lines)
            if tir_op not in ALL_TIR_OPS:
                warning(f"{tir_op} not in ALL_TIR_OPS")
                return None, None, f"Undefined TIR OP {tir_op}"
            tir_op_idx = ALL_TIR_OPS.index(tir_op)
            loop_info.searialize_to_feature(tir_feature, tir_op_idx, 1)

    customize_feature = tir_feature.flatten()
    addition_dim = np.array([std, flop_ct])

    return np.concatenate((addition_dim, customize_feature)), dur, None

def parse_sparse_feature_per_task(task_file, tvm_lock=None):
    inputs, results = auto_scheduler.RecordReader(task_file).read_lines()
    tmp_xs = []
    tmp_ys = []
    in_valide_inp_id = []
    for i in tqdm(range(len(inputs))):
        _x, _y, msg = parse_tenset_per_inp(inputs[i], results[i])
        if _x is None:
            in_valide_inp_id.append((i, msg))
        else:
            tmp_xs.append(_x)
            tmp_ys.append(_y)
    return np.array(tmp_xs), np.array(tmp_ys), in_valide_inp_id, inputs[0].task.workload_key

def parse_ansor_feature_per_task(task_file, min_sample_size=48, 
        verbose=False, get_workload_embedding=False, tvm_lock=None):
    """Make a dataset file from raw log files
       Modified from tvm.autoscheduler.make_dataset_from_log_file
    """
    assert os.path.exists(task_file), f"{task_file} does not exist."

    # Read measure records
    task_workload_key = None
    measure_records = {}
    for inp, res in auto_scheduler.RecordReader(task_file):
        task = auto_scheduler.dataset.input_to_learning_task(inp)
        if task not in measure_records:
            measure_records[task] = [[], []]
        if task_workload_key is None:
            task_workload_key = inp.task.workload_key
        measure_records[task][0].append(inp)
        measure_records[task][1].append(res)

    # Featurize
    features = None
    tmp_ys = None
    in_valide_inp_id = []
    workload_embed_dict = {}
    assert len(measure_records) == 1
    for task, (inputs, results) in measure_records.items():
        features, tmp_ys = wrap_get_per_store_features_from_measure_pairs(
            inputs, results, task,
            get_workload_embedding=False,
            verbose=False, lock=tvm_lock)
        if verbose:
            print("Task: %s\tSize: %d" % (task, len(features)))
    if features is None:
        in_valide_inp_id = ["Feature is None"]
        return None, None, in_valide_inp_id, task_workload_key
    elif len(features) < min_sample_size:
        # Delete task with too few samples
        if verbose:
            print(f"Deleted, min_sample_size={min_sample_size}")
        in_valide_inp_id = ["Too Less Samples", len(features), min_sample_size]
        return None, None, in_valide_inp_id, task_workload_key
    else:
        return features, tmp_ys, [], task_workload_key

def parse_ast_feature_per_task(task_file, min_sample_size=48, 
        verbose=False, get_workload_embedding=False, tvm_lock=None):
    """Make a dataset file with AST from raw log files
       Modified from tvm.autoscheduler.make_dataset_from_log_file
    """
    assert os.path.exists(task_file), f"{task_file} does not exist."

    # Read measure records
    measure_records = {}
    task_workload_key = None
    for inp, res in auto_scheduler.RecordReader(task_file):
        # task = auto_scheduler.dataset.input_to_learning_task(inp)

        # task = auto_scheduler.SearchTask(
        #     workload_key=inp.task.workload_key,
        #     target=inp.task.target,
        # )
        task_key = inp.task.workload_key
        if task_workload_key is None:
            task_workload_key = inp.task.workload_key

        if task_key not in measure_records:
            measure_records[task_key] = [[], []]
        measure_records[task_key][0].append(inp)
        measure_records[task_key][1].append(res)

    # Featurize
    features = None
    tmp_ys = None
    in_valide_inp_id = []
    workload_embed_dict = {}
    if len(measure_records) != 1:
        import code
        code.interact(local=locals())
    for task_key, (inputs, results) in measure_records.items():
        features, tmp_ys = wrap_get_per_store_features_from_measure_pairs(
            inputs, results,
            get_workload_embedding=False,
            verbose=False,
            parse_ast=True,
            lock=tvm_lock)
        if verbose:
            print("Task: %s\tSize: %d" % (inputs[0].task, len(features)))
    if features is None:
        in_valide_inp_id = ["Feature is None"]
        return None, None, in_valide_inp_id, task_workload_key
    elif len(features) < min_sample_size:
        # Delete task with too few samples
        if verbose:
            print(f"Deleted, min_sample_size={min_sample_size}")
        in_valide_inp_id = ["Too Less Samples", len(features), min_sample_size]
        return None, None, in_valide_inp_id, task_workload_key
    else:
        return features, tmp_ys, [], task_workload_key

def check_same_task(task_file):
    ''' Check whether all records in one task file belong to the same task
    '''
    inputs, results = auto_scheduler.RecordReader(task_file).read_lines()
    task_id = None
    for i in range(len(inputs)):
        inp = recover_measure_input(inputs[i], True)
        if task_id is None:
            task_id = inp.task.workload_key
        else:
            if task_id != inp.task.workload_key:
                return False
    
    return True

def gen_split_path(output_dir, split=None):
    if split is None:
        split = 0
    return os.path.join(output_dir, f"{split}.npy")

def save_rst(x_train, y_train, split_path):
    assert x_train.shape[0] == y_train.shape[0]
    if x_train is None or x_train.shape[0] == 0:
        warning("No data to save")
        return
    _data = np.concatenate((np.expand_dims(y_train, 1), x_train), axis=1)
    np.save(split_path, _data)
    return _data.shape

def get_hold_out_task(target, network, bs):
    network_keys = []

    if network == "resnet_50":
        batch_size = bs
        network_keys.append((f'resnet_{50}',
                                [(batch_size, 3, 224, 224)]))
    else:
        # resnet_18 and resnet_50
        for layer in [18, 50]:
            network_keys.append((f'resnet_{layer}', [(1, 3, 224, 224)]))

        # mobilenet_v2
        network_keys.append(('mobilenet_v2', [(1, 3, 224, 224)]))

        # resnext
        network_keys.append(('resnext_50', [(1, 3, 224, 224)]))

        # bert
        for scale in ['tiny', 'base']:
            network_keys.append((f'bert_{scale}', [(1, 128)]))

    filenames = set()
    for network_key in tqdm(network_keys):
        # Read tasks of the network
        task_info_filename = get_task_info_filename(network_key, target)
        tasks, _ = pickle.load(open(task_info_filename, "rb"))
        for task in tasks:
            filename = get_measure_record_filename(task, target)
            filenames.add(filename)

    return filenames

class ThreadSafeData():
    # If the buffer holds constants, the values will contain that otherwise None
    def __init__(self, data: Any, store_path: str, lock:threading.Lock):
        self.data = data
        self.store_path = store_path
        self.lock = lock

    def save_json(self):
        with self.lock:
            with open(self.store_path, 'w') as fp:
                json.dump(self.data, fp, indent=4)
    
    def load_json(self):
        with open(self.store_path, 'r') as fp:
            self.data = json.load(fp)

def _parse_one_task(thread_id, output_dir, task_idx, task_file, split_stat, _fake, tvm_lock):
    ### Save one split for each task, check whether corresponding split exists
    split_path = gen_split_path(output_dir, split=task_idx)
    split_name = os.path.basename(split_path).split(".")[0]
    if split_name in split_stat.data and \
            not (split_stat.data[split_name]["shape"] is None and split_stat.data[split_name]["error"] is None):
        return

    print(f"\n[{thread_id}] Parsing the file {task_idx}/{len(task_files)} ... ")
    
    timeout_sec = 300

    data_shape = None
    error_msg = None

    x_train, y_train = None, None
    task_workload_key = None

    if not _fake:

        resutls = [None, None, "time_out", None]
        _t = threading.Thread(target=_parse_feature_func, args=(resutls, [task_file], {"tvm_lock": tvm_lock}))
        _t.start()
        _t.join(timeout=timeout_sec)
        _xs, _ys, in_valide_inp_id, task_workload_key = resutls

        # check_same_task(task_file)

        if in_valide_inp_id == "time_out":
            error_msg = f"Time out {timeout_sec}"
            print(f"[{thread_id}] {error_msg}")
        elif _xs is None or _xs.shape[0] == 0:
            error_msg = "No valid data"
            print(f"[{thread_id}] {error_msg}")
        else:
            x_train, y_train = _xs, _ys
    else:
        ### Only see if there are valid data
        valid_cnt = 0
        for inp, res in auto_scheduler.RecordReader(task_file):
            if task_workload_key is None:
                task_workload_key = inp.task.workload_key
            if res.error_no != 0:
                assert len(res.costs) == 1
                # print(costs, "Timeout during run")
            else:
                valid_cnt += 1
        if valid_cnt == 0:
            return
    
    ### Save one split for each task
    if not _fake and error_msg is None:
        data_shape = save_rst(x_train, y_train, split_path)
        print(f"[{thread_id}] - Save data of task {idx} to {split_path}, "
            f"x_shape={x_train.shape}, y_shape={y_train.shape}")  
    
    ### Record split_stat
    with split_stat.lock:
        if split_name not in split_stat.data:
            split_stat.data[split_name] = {
                "workload_key": os.path.basename(task_file),
                "networks": task2network[task_workload_key],
                "shape": data_shape,
                "error": error_msg
            }
        elif not _fake and split_stat.data[split_name]["shape"] is None and split_stat.data[split_name]["error"] is None:
            assert "workload_key" in split_stat.data[split_name] and \
                "networks" in split_stat.data[split_name], (split_name, split_stat.data[split_name])
            split_stat.data[split_name]["shape"] = data_shape
            split_stat.data[split_name]["error"] = error_msg
        else:
            print("tenset_dataload 421")
            import code
            code.interact(local=locals())
            raise ValueError("Not to expected")
    split_stat.save_json()

def parse_tenset_thread(thread_id, args, task_files, output_dir,
        split_stat: ThreadSafeData, tvm_lock: threading.Lock, measure_stat=None):
    if measure_stat is None:
        notify("No measure_stat found")
        task_idx = thread_id
        for task_idx, _task in enumerate(ALL_TASKS):
            if task_idx < args.st or task_idx >= args.ed:
                continue
            if task_idx % (thread_id + 1) != 0:
                continue
            target = device_str2target(os.path.basename(os.path.abspath(args.input)))
            task_file = get_measure_record_filename(_task, target)
            _parse_one_task(thread_id, output_dir, task_idx, task_file, split_stat, args.fake, tvm_lock)
    else:
        ''' measure_stat is given, fix the mapping between task ids and tasks
        `measure_stat` is a list of dict, an example of that stat of one task
            {
                "task_idx": 0,
                "filename": "([0523205b02615ce027c9bdeaa6a9392d,1,14,14,480,1,1,1,480],cuda).json",
                "total_num": 607,
                "workload_key": "[\"0523205b02615ce027c9bdeaa6a9392d\", 1, 14, 14, 480, 1, 1, 1, 480]",
                "counter": [1, 0, 0, 0, 0, 0, 320, 0, 0]
            }
        '''
        _input_dir = os.path.dirname(task_files[0])
        for _stat in measure_stat:
            if _stat is None:
                continue
            task_idx = _stat["task_idx"]
            if task_idx < args.st or task_idx >= args.ed:
                continue
            if task_idx % (thread_id + 1) != 0:
                continue
            valid_measure_num = _stat["counter"][auto_scheduler.measure.MeasureErrorNo.NO_ERROR]
            if not (valid_measure_num / _stat["total_num"] >= 0.25 and \
                (valid_measure_num > 100 or _stat["total_num"] < 100)):
                ### Not enough meausured data, skip
                print(f"[{thread_id}] Skip task {task_idx}")
                continue
            task_file = os.path.join(_input_dir, _stat["filename"])
            _parse_one_task(thread_id, output_dir, task_idx, task_file, split_stat, args.fake, tvm_lock)

    print(f"[{thread_id}] Thread ends")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Profile TIR")
    parser.add_argument('-i', '--input', type=str, default=None, help="Path to store the inputs")
    # parser.add_argument('-w', '--workspace', type=str, default=".", help="Path to store the results")
    parser.add_argument('-o', '--output', type=str, default=".", help="Path to store the results")
    parser.add_argument('-j', '--job', type=int, default=1, help="Job number to parse data cocurrently")
    parser.add_argument('--st', type=int, default=0, help="Start task idx")
    parser.add_argument('--ed', type=int, default=None, help="End task idx")
    parser.add_argument('--reload', action='store_true', help="load the progress of generating features")
    parser.add_argument('--feature', type=str, choices=["default", "ansor", "ast"], default="ast", help="Feature extractor used")
    parser.add_argument('--fake', action='store_true', help="Fake parser")
    args = parser.parse_args()
    if args.ed is None:
        args.ed = len(ALL_TASKS)

    def _parse_feature_func(resutls, _args, _kwargs):
        if args.feature == "default":
            _xs, _ys, in_valide_inp_id, task_workload_key = parse_sparse_feature_per_task(*_args, **_kwargs)
        elif args.feature == "ansor":
            _xs, _ys, in_valide_inp_id, task_workload_key = parse_ansor_feature_per_task(*_args, **_kwargs)
        elif args.feature == "ast":
            _xs, _ys, in_valide_inp_id, task_workload_key = parse_ast_feature_per_task(*_args, **_kwargs)
        else:
            raise ValueError(f"Invalide feature extractor {args.feature}")
        resutls[0:] = _xs, _ys, in_valide_inp_id, task_workload_key

    ### Inputs
    measure_stat = None
    if not os.path.exists(args.input):
        compatible_input = args.input.replace("measure_records", "test_measure_records")
        if not os.path.exists(compatible_input):
            raise ValueError(f"Input record dir {compatible_input} doesn't exist!")
        else:
            warning(f"{args.input} doesn't exists, use {compatible_input} instead.")
            args.input = compatible_input
    if os.path.isdir(args.input):
        root, _, files = list(os.walk(args.input))[0]
        task_files = [os.path.join(root, _file) for _file in files if not "measure_stat" in _file]
        measure_stat_path = os.path.join(root, "measure_stat.json")
        if os.path.exists(measure_stat_path):
            with open(measure_stat_path, 'r') as fp:
                measure_stat = json.load(fp)
    else:
        task_files = [os.path.abspath(args.input)]

    ### Outputs
    output_dir = os.path.abspath(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not args.reload:
        # raise ValueError(f"Dir {output_dir} exists")
        pass

    split_stat = ThreadSafeData(
        data = {},
        store_path = os.path.join(output_dir, f"split_stat.json"),
        lock = threading.Lock()
    )

    ### Task to network info
    from dump_network_info import map_task2network
    task2network = map_task2network()

    if False:
        target = "cuda"
        target = tvm.target.cuda("t4")
        filenames = get_hold_out_task(target, "resnet_50", 8)
        print(len(filenames))
        sum = 0
        min_sum = 0
        max_sum = 0
        for task_file in filenames:
            _xs, _ys, in_valide_inp_id = _parse_feature_func(task_file)
            print(np.mean(_ys), np.std(_ys))
            sum += np.mean(_ys)
            min_sum += min(_ys)
            max_sum += max(_ys)
        print(sum, min_sum, max_sum)
        raise
    
    ### Reload preprogress
    if args.reload and not args.fake:
        # if os.path.exists(progress_split.store_path):
        #     with open(progress_split.store_path, 'r') as fp:
        #         progress_split.data = [int(x) for x in fp.read().split(",")]
        #     notify(f"Reload the progress from {progress_split.store_path}: "
        #             f"starting from file_id={progress_split.data[0]}, split_id={progress_split.data[1]}")

        # if os.path.exists(invalid_tasks.store_path):
        #     invalid_tasks.load_json()
        #     notify(f"Reload the error msg from {invalid_tasks.store_path}")
        
        if os.path.exists(split_stat.store_path):
            split_stat.load_json()

    ### Start parse each task file
    idx = 0
    all_threads = []
    tvm_lock = threading.Lock()
    for j in range(args.job):
        t = threading.Thread(target=parse_tenset_thread, args=(j, args, task_files, output_dir,
                split_stat, tvm_lock, measure_stat))
        t.start()
        all_threads.append(t)

    for t in all_threads:
        t.join()

    split_stat.save_json()


### Correct the format of split_stat.json
# import os
# import json

# output_dir = os.getcwd()
# files = os.listdir(output_dir)
# for _file in files:
#     if _file.startswith("t4_"):
#         os.rename(_file, _file.split("t4_")[1])



# with open(os.path.join(output_dir, "network2split.json"), 'r') as fp:
#         network2split = json.load(fp)
# new = {}
# for network in network2split:
#     new[network] = {}
#     for bs in network2split[network]:
#         splits = [_file.split("t4_")[1] for _file in network2split[network][bs]]
#         new[network][bs] = splits
# with open(os.path.join(output_dir, "_network2split.json"), 'w') as fp:
#     json.dump(new, fp, indent=4)


# with open(os.path.join(output_dir, "split_stat.json"), 'r') as fp:
#         split_stat = json.load(fp)

# new = {}
# for _file in split_stat:
#     new[_file.split("t4_")[1]] = split_stat[_file]

# with open(os.path.join(output_dir, "_split_stat.json"), 'w') as fp:
#     json.dump(new, fp, indent=4)