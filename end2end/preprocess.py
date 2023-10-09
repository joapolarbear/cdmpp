import json
import numpy as np
from typing import Union, List
from tqdm import tqdm
import os
import pickle
from collections import Counter
import time

import tvm
from tvm import auto_scheduler

### from tenset
from tune_network import get_network
from common import (get_task_info_filename,
    get_measure_record_filename, get_test_measure_record_filename)
from measure_programs import parse_cost, INVALID_TIME_UPPER

from metalearner.data.rawdata import remove_outliers
from utils.env import PROJECT_CFG
from utils.util import STD2AVG_UPPER_BOUND, LARGEST_COST

from .util import (_network_target_dir, _relay_ir_filename, _graph_filename,
    wrap_relay_build, save_relay_ir, load_relay_ir, get_network_key)
from .util import task2workload_hash_args, check_network_tasks
from .util import MyDispatchContext, TaskInfo, MockCostModel, TENSET_STATE


def wrap_get_network(network_args, cache_root):
    relay_cache_path = os.path.join(cache_root, _relay_ir_filename(network_args))
    if os.path.exists(relay_cache_path):
        mod, params, inputs = load_relay_ir(relay_cache_path)
    else:
        mod, params, inputs = get_network(network_args)
        save_relay_ir(mod, params, inputs, relay_cache_path)
    return mod, params, inputs

def wrap_extract_task_info(cache_dir, mod, params, target):
    ''' Load or extract tasks and metainfo from `mod`
    '''
    tasks_path = os.path.join(cache_dir, "tasks.pkl")
    if os.path.exists(tasks_path):
        tasks, task_weights = pickle.load(open(tasks_path, "rb"))
        print(f"Load tasks from {tasks_path}")
    else:
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        pickle.dump((tasks, task_weights), open(tasks_path, "wb"))
        print(f"Store tasks at {tasks_path}")

    task_info = TaskInfo(tasks, task_weights, target)
    funcname2task_path = os.path.join(cache_dir, "func_name.json")
    if not os.path.exists(funcname2task_path):
        with open(funcname2task_path, 'w') as fp:
            json.dump(task_info.func_name2worload_key, fp, indent=4)
    return task_info

def _is_valid_state(task, target, state):
    ''' Compile with a schedule for correctness check
    '''
    try:
        sch, args = task.compute_dag.apply_steps_from_state(state)
        mod_ref = tvm.build(sch, args, target)
    except KeyboardInterrupt:
        exit(-1)
    except RuntimeError:
        return False
    return True

def _select_valid_state(task, target, states):
    valid_state = None
    for state_idx, state in enumerate(states):
        if _is_valid_state(task, target, state):
            valid_state = state
            # print(f"Found valid state after {state_idx} failure")
            break
        else:
            continue
    if valid_state is None:
        print("[Error] no valid states found. Sample a state to show the error")
        sch, args = task.compute_dag.apply_steps_from_state(states[0])
        mod_ref = tvm.build(sch, args, target)
    return valid_state

def _sample_valid_state_for_task(task, target):
    policy = auto_scheduler.SketchPolicy(task, verbose=0, 
        # program_cost_model=PythonBasedModel(),
        program_cost_model=MockCostModel()
    )
    # state = task.compute_dag.init_state
    states = policy.sample_initial_population()
    return _select_valid_state(task, target, states)

def _select_valid_inp(task, target, inputs):
    valid_inp = None
    for inp_idx, inp in enumerate(inputs):
        if _is_valid_state(task, target, inp.state):
            valid_inp = inp
            # print(f"Found valid input after {inp_idx} failure")
            break
        else:
            continue
    return valid_inp

def check_compare_valid(task, target, inputs, results):
    ''' We have two methods to state/schedule correctness check
        1. Compile with a schedule, i.e. _is_valid_state(task, target, state)
        2. See if corresponding measurement record in Tenset is valid

        This method is used to check whether the two methods return the same resutls
    '''
    dur_std_list = [parse_cost(rst) for rst in results]
    is_valid_list = [_is_valid_state(task, target, inp.state) for inp in inputs[:10]]
    is_valid_list_from_measure = [dur is not None for dur, std in dur_std_list[:10]]

def get_valid_inp_results_from_tenset(task, target):
    measure_record_filename = get_measure_record_filename(task, target)
    if not os.path.exists(measure_record_filename):
        measure_record_filename = get_test_measure_record_filename(task, target)
    inputs, results = auto_scheduler.RecordReader(measure_record_filename).read_lines()
    # check_compare_valid(task, target, inputs, results)
    min_cost, max_cost = PROJECT_CFG.get("cost_range", (0, LARGEST_COST))
    filter_input_result_list = []
    for inp_id in range(len(inputs)):
        dur, std = parse_cost(results[inp_id])
        if dur is not None and dur >= min_cost \
                and dur <= max_cost and std/dur <= STD2AVG_UPPER_BOUND:
            filter_input_result_list.append((inputs[inp_id], results[inp_id], dur))

    ### Sort input and output pairs according to the cost
    # filter_input_result_list = sorted(filter_input_result_list, key=lambda x: x[2])

    return filter_input_result_list

def get_network_tasks(network_args, target, cache_root):
    mod, params, inputs = wrap_get_network(network_args, cache_root)

    # Extract search tasks
    cache_dir = _network_target_dir(cache_root, network_args, target)
    task_info = wrap_extract_task_info(cache_dir, mod, params, target)

    return mod, params, inputs, task_info

def get_network_with_states(network_args, target, cache_root):
    
    mod, params, inputs = wrap_get_network(network_args, cache_root)

    # Extract search tasks
    cache_dir = _network_target_dir(cache_root, network_args, target)
    task_info = wrap_extract_task_info(cache_dir, mod, params, target)

    measure_inputs = None
    filename = os.path.join(cache_dir, "task2state.records")
    if os.path.exists(filename):
        # measure_inputs = [inp for inp, _ in auto_scheduler.load_records(filename)]
        measure_inputs, measure_results = auto_scheduler.RecordReader(filename).read_lines()

        check_network_tasks(task_info.tasks, [inp.task for inp in measure_inputs])
        task_info.task_state_src = pickle.load(open(filename+".src.pkl", 'rb'))
        print(f"Load cached states from {filename}, src: {Counter(task_info.task_state_src)}")
    elif TENSET_STATE:
        print(f"[Get states] Sample states for tasks ... ")
        ### Select a state from the Tenset dataset

        # Read tasks of the network
        network_key = get_network_key(network_args)
        task_info_filename = get_task_info_filename(network_key, target)

        if not os.path.exists(task_info_filename):
            tasks = task_info.tasks
        else:
            print(f"[Get states] Load tasks from Tenset: {task_info_filename} for {network_key} {target}")
            tasks, task_weights = pickle.load(open(task_info_filename, "rb"))
            check_network_tasks(task_info.tasks, tasks)

        # np.random.seed(1)
        np.random.seed(int(time.time()))

        print(f"[Get states] Load measured records from Tenset ... ")
        workload_hash2inps = {}
        for task in tasks:
            filter_input_result_list = get_valid_inp_results_from_tenset(task, target)
            if len(filter_input_result_list) > 0:
                _key = task2workload_hash_args(task)
                workload_hash2inps[_key] = filter_input_result_list
        
        measure_inputs = []
        measure_results = []
        fail_cnt = 0
        print(f"[Get states] Decides states for each extracted tasks ... ")
        pbar = tqdm(total=len(task_info.tasks))
        for task_idx, task in tqdm(enumerate(task_info.tasks)):
            _key = task2workload_hash_args(task)
            if _key in workload_hash2inps:
                input_result_list = workload_hash2inps[_key]
                # inp = _select_valid_inp(task, target, inputs)
                # inp, result, _ = input_result_list[1]
                inp, result, _ = input_result_list[np.random.choice(len(input_result_list))]

                # dur_list = [x[-1] for x in input_result_list]
                # avg_percentile = np.percentile(dur_list, 26)
                # for _id in range(len(input_result_list)):
                #     inp, result, _dur = input_result_list[_id]
                #     if _dur >= avg_percentile:
                #         break

                inp.task = task
                task_info.task_state_src[task_idx] = 'tenset'
            else:
                fail_cnt += 1
                ### Fail to find tasks with the same _key in Tenset,
                # print(f"[Warning] Fail to find tasks with the same _key in Tenset")
                valid_state = _sample_valid_state_for_task(task, target)
                if valid_state is None:
                    raise ValueError(f"Fail to find a valid state for task {task_idx} {task.workload_key}")
                inp = auto_scheduler.MeasureInput(task, valid_state)
                task_info.task_state_src[task_idx] = 'random'
                # MeasureResult(self, costs, error_no, error_msg, all_cost, timestamp)
                result = auto_scheduler.measure.MeasureResult([INVALID_TIME_UPPER], 0, "", 0.2, 1)
            measure_inputs.append(inp)
            measure_results.append(result)
            pbar.update(1)
        pbar.close()
        print(f"[Get states] Load {len(measure_inputs)-fail_cnt} states from Tenset and randomly gen {fail_cnt} states")
        auto_scheduler.save_records(filename, measure_inputs, measure_results)
        pickle.dump(task_info.task_state_src, open(filename+".src.pkl", 'wb'))
    else:
        ### Randomly sample states
        measure_inputs = []
        measure_results = []
        for task_idx, task in tqdm(enumerate(task_info.tasks), total=len(task_info.tasks)):
            print(f"Generate valid state for task {task_idx}/{len(task_info.tasks)} ... ")
            valid_state = _sample_valid_state_for_task(task, target)
            if valid_state is None:
                raise ValueError(f"Fail to find a valid state for task {task_idx} {task.workload_key}")
            task_info.task_state_src[task_idx] = 'random'
            measure_inputs.append(auto_scheduler.MeasureInput(task, valid_state))
            # MeasureResult(self, costs, error_no, error_msg, all_cost, timestamp)
            measure_results.append(auto_scheduler.measure.MeasureResult([INVALID_TIME_UPPER], 0, "", 0.2, 1))
        print(f"[Get states] Randomly gen {len(measure_inputs)} states")
        auto_scheduler.save_records(filename, measure_inputs, measure_results)
        pickle.dump(task_info.task_state_src, open(filename+".src.pkl", 'wb'))

    for inp, result in zip(measure_inputs, measure_results):
        dur, std = parse_cost(result)
        task_info.record_state_by_task(inp.task, inp.state, dur)
    return mod, params, inputs, task_info


def offline_preprocess(network_args, target, cache_root="tmp/end2end"):

    ''' 
    1. Load and cache Relay IR
    2. Load and cache tasks, task weights
    3. Load and cache task2state mapping
    ### For dpro replay
    4. Load and cache task2funcname mapping
    '''
    mod, params, inputs, task_info = get_network_with_states(network_args, target, cache_root)
    
    ### For dRPO replay
    cache_dir = _network_target_dir(cache_root, network_args, target)
    graph_path = os.path.join(cache_dir, _graph_filename())
    if not os.path.exists(graph_path):
        print(f"Extract DAG for {network_args}")
        with MyDispatchContext(task2state=task_info.task2state):
            lib = wrap_relay_build(mod, target, params)
        with open(graph_path, 'w') as fp:
            json.dump(json.loads(lib.graph_json), fp, indent=4)