from typing import Union, List
import os
import pickle
from collections import Counter

import tvm
from tvm import auto_scheduler, relay
from tvm.auto_scheduler.cost_model.cost_model import PythonBasedModel
from tvm.auto_scheduler.feature import get_per_store_features_from_states

from common import load_and_register_tasks

TENSET_STATE = True
load_and_register_tasks()

def get_workload_entry(records, target_key, workload_key):
    """Get the entry of the target key and workload key hash in the given record map.

    Parameters
    ----------
    records: Dict[str, Dict[str, Dict[str, Any]]]
        The best record map.
    target_key: str
        The first key to the records.
    workload_key: str
        The workload key that can be decoded to workload hash and args.

    Returns
    -------
    entry: Dict[str, Any]
        The entry in records with target key and workload hash.
    workload_hash: str
        The workload hash decoded from workload_key.
    workload_args: str converted from Tuple[Any, ...]
        The hashable tuple of workload args decoded from workload_key.
    """
    workload_hash, workload_args = auto_scheduler.utils.decode_workload_key(workload_key)
    if target_key not in records:
        records[target_key] = {}
    if workload_hash not in records[target_key]:
        records[target_key][workload_hash] = {}
    return records[target_key][workload_hash], workload_hash, str(workload_args)

def task2workload_hash_args(_task):
    workload_hash, workload_args = auto_scheduler.utils.decode_workload_key(_task.workload_key)
    return f"{workload_hash}-{workload_args}"

def check_network_tasks(tasks1, tasks2):
    ''' Check if two sets of tasks intersect with each other
    '''
    hash_args1: set = set([task2workload_hash_args(_task) for _task in tasks1])
    hash_args2: set = set([task2workload_hash_args(_task) for _task in tasks2])
    print(f"Network 1 and 2 with {len(hash_args1)} and {len(hash_args2)} tasks respectively")
    print(f"Two networks have {len(hash_args1.intersection(hash_args2))} tasks in common")

def wrap_relay_build(mod, target, params):
    with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)
    return lib

def save_relay_ir(mod, params, inputs, relay_cache_path):
    os.makedirs(os.path.dirname(relay_cache_path), exist_ok=True)
    mod_json = tvm.ir.save_json(mod)
    params_bytes = relay.save_param_dict(params)
    pickle.dump((mod_json, params_bytes, inputs), open(relay_cache_path, "wb"))

def load_relay_ir(relay_cache_path):
    mod_json, params_bytes, inputs = pickle.load(open(relay_cache_path, "rb"))
    params = relay.load_param_dict(params_bytes)
    mod = tvm.ir.load_json(mod_json)
    return mod, params, inputs

def get_network_key(network_args):
    name, batch_size = network_args['network'], network_args['batch_size']
    if name in ['resnet_18', 'resnet_50', 'mobilenet_v2', 'mobilenet_v3',
                'wide_resnet_50', 'resnext_50', 'densenet_121']:
        network_key = (name, [(batch_size, 3, 224, 224)])
    elif name in ['inception_v3']:
        network_key = (name, [(batch_size, 3, 299, 299)])
    elif name in ['bert_tiny', 'bert_base', 'bert_medium', 'bert_large']:
        network_key = (name, [(batch_size, 128)])
    elif name == 'dcgan':
        network_key = (name, [(batch_size, 3, 64, 64)])
    else:
        raise ValueError("Invalid network: " + name)
    return network_key

def _relay_ir_filename(network_args):
    return f"{network_args['network']}-{network_args['batch_size']}/relay.pkl"

def _network_target_dir(root, network_args, target):
    _dir = os.path.join(root, f"{network_args['network']}-{network_args['batch_size']}", f"{target.kind}_{target.model}")
    os.makedirs(_dir, exist_ok=True)
    return _dir

def _workload_key2func_name_filename(network_args, target):
    return f"{network_args['network']}-{network_args['batch_size']}/{target.kind}_{target.model}/funcname.json"

def _graph_filename():
    return f"graph.json"

def _measure_result_filename(network_args, target):
    return f"{network_args['network']}-{network_args['batch_size']}/{target.kind}_{target.model}/measure.txt"

def _replay_rst_filename(via=None):
    if via == "profile":
        via_str = "_via_profile"
    elif via == "tenset":
        via_str = "_via_tenset"
    else:
        via_str = ""
    is_tenset_state = "_tenset" if TENSET_STATE else ""
    return f"replay{via_str}{is_tenset_state}.txt"

class MockCostModel(PythonBasedModel):
    """A mock cost model that rates 1 only for the states with tile_k=2."""

    def predict(self, task, states):
        features = get_per_store_features_from_states(states, task)
        scores = [1.] * len(features)
        for idx, feature in enumerate(features):
            if feature.min() == feature.max() == 0:
                scores[idx] = float('-inf')
        return scores


class MyDispatchContext(auto_scheduler.DispatchContext):
    """
    Apply specific schedules and record workload_key2func_name mapping
    """
    def __init__(self, task2state={}, workload_key2func_name=None):
        super(MyDispatchContext, self).__init__()
        self.task2state = task2state
        self.workload_key2func_name = workload_key2func_name

        self.include_compatible = True
        self.verbose = 0
        self._old_verbose = 0

    def _query_inside(self, target, workload_key, func_name):

        ### Record func_name
        ### TODO: Depecrated, since func_name has been recorded in task
        if isinstance(self.workload_key2func_name, dict):
            entry, workload_hash, workload_args = get_workload_entry(
                    self.workload_key2func_name, target.model, workload_key)
            if workload_args in entry:
                assert func_name == entry[workload_args], (
                    func_name, workload_hash, workload_args, entry[workload_args])
            else:
                entry[workload_args] = func_name

        if len(self.task2state) == 0:
            return None

        if target is None:
            raise RuntimeError(
                "Need a target context to find the history best. "
                "Hint: If your target is llvm, use `with tvm.target.create('llvm'):`"
                " above the dispatcher call. So does other target. "
            )

        """The helper function to match the record in the given map
        and return the matched state, or None if no match.
        """
        entry, workload_hash, workload_args = get_workload_entry(
            self.task2state, target.model, workload_key
        )

        ret = None
        if workload_args in entry:
            ret = entry[workload_args][0]
            # print(f"[MyDispatchContext] Found states for {workload_key}:{func_name}")
        else:
            # print(f"[MyDispatchContext] Warning: fail to find valid states for {workload_key}:{func_name}")
            pass

        # elif self.include_compatible:
        #     best_cost = float("inf")
        #     for args, val in entry.items():
        #         dis_f = auto_scheduler.utils.calc_workload_dis_factor(
        #             (workload_hash, workload_args), (workload_hash, args)
        #         )
        #         if dis_f == float("inf"):
        #             continue
        #         state, cost = val
        #         cost *= dis_f
        #         if ret is None or cost < best_cost:
        #             best_cost = cost
        #             ret = state


        # ret = match_record(self.task2state, target.model, workload_key)

        return ret

    # def update(self, target, workload_key, state):
    #     entry, workload_hash, workload_args = get_workload_entry(
    #         self.task2state, target.model, workload_key
    #     )
    #     # print(f"Register {target.model} {workload_key}")
    #     entry[workload_args] = (state, 1)
    
    # def __enter__(self, *args, **kwargs):
    #     super(MyDispatchContext, self).__enter__(*args, **kwargs)
    #     self._old_verbose = self._old_ctx.verbose
    #     self._old_ctx.verbose = 0

    # def __exit__(self, *args, **kwargs):
    #     self._old_ctx.verbose = self._old_verbose
    #     super(MyDispatchContext, self).__exit__(*args, **kwargs)


class TaskInfo:
    def __init__(self, tasks, task_weights, target):
        self.tasks = tasks
        self.task_weights = task_weights
        self.target = target

        ### Used for query performance with func_name
        self.func_name2worload_key = {}
        for task in self.tasks:
            workload_hash, workload_args = auto_scheduler.utils.decode_workload_key(task.workload_key)
            for func_name in task.desc.split(","):
                func_name = "_".join(func_name.split("_")[2:])
                self.func_name2worload_key[func_name] = (workload_hash, str(workload_args))

        self.task2state: Union[dict, None] = {}
        self.task_state_src = ["unknow"] * len(self.tasks)
        self.runtime_info = {}

        ### TODO: xydata' size is not equal to the task number, since there are some task whose measured results are in valid
        self.xydata = []
    
    def get_func_name_by_task(self, task):
        func_names = task.desc.split(",")
        assert len(func_names) == 1, (task.workload_key, func_names)
        return func_names[0]

    def record_task_runtime_info(self, task, info_dict: dict):
        entry, _, workload_args = get_workload_entry(self.runtime_info,
            self.target.model, task.workload_key)
        entry[workload_args] = info_dict
    
    def query_runtime_info_by_task(self, task):
        entry, _, workload_args = get_workload_entry(self.runtime_info,
            self.target.model, task.workload_key)
        return entry[workload_args]
    
    def query_func_cost(self, func_name):
        workload_hash, workload_args = self.func_name2worload_key[func_name]
        return self.runtime_info[workload_hash][workload_args]["cost"]

    def record_state_by_task(self, task, state, dur=None):
        entry, _, workload_args = get_workload_entry(self.task2state,
            self.target.model, task.workload_key)
        entry[workload_args] = (state, dur)
    
    def get_measure_inps(self):
        measure_inp_state = []
        for task in self.tasks:
            state, dur = self.query_state_by_task(task)
            inp = auto_scheduler.MeasureInput(task, state)
            measure_inp_state.append((inp, dur))
        return measure_inp_state
    
    def query_state_by_task(self, task):
        entry, worklaod_hash, workload_args = get_workload_entry(self.task2state,
                self.target.model, task.workload_key)
        return entry[workload_args]
