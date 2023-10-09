import argparse
import numpy as np
from typing import Union, List
import os

import tvm
from tvm import auto_scheduler
from tvm.contrib import graph_executor

from .preprocess import get_network_with_states
from .util import MyDispatchContext, wrap_relay_build, _network_target_dir

def measure_network(network_args, target, cache_root="tmp/end2end", breakdown=False):
    mod, params, inputs, task_info = get_network_with_states(network_args, target, cache_root)
    
    ### Apply states and lowering
    print("[Measure] compiling ... ")
    with MyDispatchContext(task2state=task_info.task2state):
        lib = wrap_relay_build(mod, target, params)
        
    # Create graph executor
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    if breakdown:
        print("Start to profile the TIR module")
        from tvm.contrib.debugger import debug_executor
        graph, _lib, params = lib.get_executor_config(), lib.lib, lib.params
        module = debug_executor.create(graph, _lib, dev,
            dump_root=".workspace/tvmdbg")
    else:
        module = graph_executor.GraphModule(lib["default"](dev))
    
    ### Set input
    # inputs = [(input_name, input_shape, dtype)]
    for input_name, input_shape, dtype in inputs:  
        input_data = np.ones(input_shape).astype(dtype)
        print(input_name, input_shape, dtype)
        module.set_input(input_name, input_data)

    if breakdown:
        ### Warm up
        module.run()
        return

    # Evaluate
    print("[Measure] Evaluate inference time cost...")
    measure_rst = module.benchmark(dev, repeat=3, min_repeat_ms=500)
    print(measure_rst)
    cache_dir = _network_target_dir(cache_root, network_args, target)
    with open(os.path.join(cache_dir, "measure.txt"), 'w') as fp:
        fp.write(str(measure_rst))

def do_measure_inputs(measure_inputs, target):
    # Make measure
    # Set measurement arguments
    from measure_programs import make_measurer, parse_cost
    measurer_kwargs = {
        "build_timeout": 60,
        "run_timeout": 60,
        "number": 2,
        "enable_cpu_cache_flush": (target.kind == "llvm"),
        "verbose": 1,
        "repeat": 2,
    }
    measurer_kwargs['log_filename'] = "tmp/measure_by_task.txt"
    measurer = make_measurer(**measurer_kwargs)

    dur_std_list = []
    for measure_input in measure_inputs:
        empty_policy = auto_scheduler.search_policy.EmptyPolicy(measure_input.task)
        inp_batch = [measure_input]
        res_batch = measurer.measure(measure_input.task, empty_policy, inp_batch)
        tenset_dur_std = parse_cost(res_batch[0])
        dur_std_list.append(tenset_dur_std)
    
    return dur_std_list

def measure_by_task(network_args, target, cache_root="tmp/end2end"):
    
    mod, params, inputs, task_info = get_network_with_states(network_args, target, cache_root)
    
    measure_inputs = [inp for inp, dur in task_info.get_measure_inps()]
    dur_std_list = do_measure_inputs(measure_inputs, target)

    for i, measure_input in enumerate(measure_inputs):
        func_name = task_info.get_func_name_by_task(measure_input.task)
        tenset_dur_std = dur_std_list[i]
        print(f"{func_name}: {tenset_dur_std[0] * 1e6} \u00B1 {tenset_dur_std[1] * 1e6} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, default="replay", help="[replay|measure]")
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    # parser.add_argument("--log-file", type=str, required=True)
    args = parser.parse_args()

    network_args = {
        "network": args.network,
        "batch_size": args.batch_size,
    }

    if args.option == "measure":
        measure_network(network_args, args.target)
    elif args.option == "breakdown":
        measure_network(network_args, args.target, breakdown=True)
    else:
        raise ValueError
