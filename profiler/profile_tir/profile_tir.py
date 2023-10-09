import os
from re import X
import time
import numpy as np
from tqdm import tqdm
import argparse
import json
import timeit

import tvm
from tvm import autotvm
from tvm import auto_scheduler
import tvm.relay as relay
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor

from tvm_helper.tir_helper import (
    parse_operator_module,
    wrap_get_per_store_features_from_measure_pairs
)
import onnx_utils
import utils.gpu_utils as gpu_utils
import tvm_transformers

tvm._ffi._init_api("ir", __name__)

CUSTOMIZED_FEATURE=True

DEBUG_ON = True
STD2DUR_UPPER = 1e5 if DEBUG_ON else 1
ADDITIONAL_DIM_LEN = 2

# python3 profiler/profile_tir/profile_tir.py -o profile -m .workspace/onnx_models -w .workspace/profile --per_task
# python3 profiler/profile_tir/profile_tir.py -o tir_info -m .workspace/onnx_models -w .workspace/tir_info
parser = argparse.ArgumentParser(prog="Profile TIR")
parser.add_argument('-o', '--option', type=str, default="profile", help="One of [profile|tir_info]")
parser.add_argument('-w', '--workspace', type=str, default=".", help="Path to store the results")
parser.add_argument('-m', '--model_dir', type=str, default=".", help="Path to store the ONNX models")
parser.add_argument('-b', '--batch_size', type=int, default=8, help="Batch Size")
parser.add_argument('--trial_num', type=int, default=10, help="Trial num per TIR task")
parser.add_argument('--per_task', action="store_true", help="Save data per_task")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
PROFILE_TRIAL_NUM = args.trial_num

class XYPair:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
    
    def concatenate(self, others):
        if self.x is None:
            self.x = others.x
            self.y = others.y
        else:
            self.x = np.concatenate((self.x, others.x), axis=0)
            self.y = np.concatenate((self.y, others.y), axis=0)
    
    def __len__(self):
        if self.x:
            return len(self.x)
        else:
            return 0
    
    def check(self):
        if self.x is None:
            return
        assert len(self.y) == len(self.x)
    
    def tvm_inner_feature(self, tuner_obj):
        tuner_obj.cost_model._reset_pool(tuner_obj.cost_model.space, tuner_obj.cost_model.target, tuner_obj.cost_model.task)
        self.x = tuner_obj.cost_model._get_feature(self.x)
    
    def finalize(self):
        if self.x is None:
            return None
        return np.concatenate((np.expand_dims(self.y, 1), self.x), axis=1)
    

###############################################################################
# Profile TIR Time Cost
# -------------------
# Collect time cost for each tir kernel, which will be used to train the
# cross-domain cost model
def customized_feature(inputs, results, config_frontiers, residual_configs):
    tmp_xs = []
    tmp_ys = []
    # for each pair of input and result
    for idx in range(len(inputs)):
        inp = inputs[idx]
        res, ir_by_target = results[idx]
        '''
            """Error type for MeasureResult"""
            NO_ERROR = 0              # no error
            INSTANTIATION_ERROR = 1   # actively detected error in instantiating a template with a config
            COMPILE_HOST = 2          # error when compiling code on host (e.g. tvm.build)
            COMPILE_DEVICE = 3        # error when compiling code on device (e.g. OpenCL JIT on the device)
            RUNTIME_DEVICE = 4        # error when run program on device
            WRONG_ANSWER = 5          # answer is wrong when compared to a golden output
            BUILD_TIMEOUT = 6         # timeout during compilation
            RUN_TIMEOUT = 7           # timeout during run
            UNKNOWN_ERROR = 8         # unknown error
        '''
        if res.error_no == 0:
            ### Successfully run
            dur = np.mean(res.costs[10:])
            std = np.std(res.costs[10:])
            if std / dur > STD2DUR_UPPER:
                ### Filter out samples with large std.
                # Use a large value as the upperbound to make a less strict constraint
                if config_frontiers[idx][1] > 1:
                    residual_configs.append(
                        (config_frontiers[idx][0], config_frontiers[idx][1]-1))
            else:
                flops = inp.task.flop / dur
                index = inp.config.index
                if CUSTOMIZED_FEATURE:
                    tir_text = AsText(ir_by_target, False, None)
                    customize_feature = np.array(
                        parse_operator_module(tir_text)).flatten()
                    addition_dim = np.array([std, inp.task.flop])
                    assert addition_dim.shape[0] == ADDITIONAL_DIM_LEN
                    tmp_xs.append(np.concatenate((addition_dim, customize_feature)))
                    # xs.append(customize_feature)
                else:
                    tmp_xs.append(index)
                tmp_ys.append(dur)
        elif res.error_no == 1:
            ### INSTANTIATION_ERROR
            pass
        else:
            ### error
            # print(f"Error {res.error_no}, {inp}")
            # print(f"{res.costs[0]}")
            pass
        return XYPair(x=np.array(tmp_xs), y=np.array(tmp_ys))

def profile_configs(configs, tuner_obj, measure_batch):
    ### Each config is tested for at most 10 times in case of unstable time cost
    config_frontiers = [(config, 10) for config in configs]
    config_xypair = XYPair()
    while len(config_frontiers) > 0:
        inputs = [autotvm.measure.MeasureInput(tuner_obj.task.target, tuner_obj.task, config) for config, _ in config_frontiers]
        results = measure_batch(inputs)
        
        ### Empty the config to be re-measured first
        residual_configs = []
        _tmp_xypair = customized_feature(inputs, results, config_frontiers, residual_configs)
        config_xypair.concatenate(_tmp_xypair)
        
        ### Updated configs to be evaluated
        config_frontiers = residual_configs
        
    return config_xypair

def profiler_tir_per_task(tuner_obj, measure_option, n_trial):
    ### tuner_obj.tune(...)
    measure_batch = autotvm.measure.create_measure_batch(tuner_obj.task, measure_option)
    n_parallel = getattr(measure_batch, "n_parallel", 1)
    assert n_parallel == 1

    task_xypair = XYPair()
    ### for each trial
    trial_id = 0
    with tqdm(total=n_trial) as pbar:
        while trial_id < n_trial:
            if not tuner_obj.has_next():
                break
            configs = tuner_obj.next_batch(10)
            config_xypair = profile_configs(configs, tuner_obj, measure_batch)
            if len(config_xypair) > 0:
                pbar.update(len(config_xypair))
                trial_id += len(config_xypair)
                task_xypair.concatenate(config_xypair)
            else:
                pass

    if len(task_xypair) == 0:
        return task_xypair
    
    if not CUSTOMIZED_FEATURE:
        task_xypair.tvm_inner_feature(tuner_obj)
  
    task_xypair.check()
    
    return task_xypair.finalize()

def profile_tir_per_mod(mod, params, target, save_path,
        feature_type="curve",
        n_trial = 10):
    
    runner = autotvm.LocalRunner(
            number=1,
            repeat=20,
            timeout=10,
            min_repeat_ms=0,
            # enable_cpu_cache_flush=True,
    )

    # test(mod, params, target)
    # raise

    # tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
    # measure_option = autotvm.measure_option(
    #     builder=autotvm.LocalBuilder(build_func="default", n_parallel=1),
    #     runner=runner
    # )
    # names = auto_scheduler.feature.get_per_store_feature_names()
    # print(names)
    # print(len(names))

    from end2end.preprocess import _sample_valid_state_for_task
    from metalearner.data.rawdata import ASTRawData
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    for task_id, task in enumerate(tasks):
        valid_state = _sample_valid_state_for_task(task, target)
        inp = auto_scheduler.MeasureInput(task, valid_state)
        # MeasureResult(self, costs, error_no, error_msg, all_cost, timestamp)
        result = auto_scheduler.measure.MeasureResult([1.], 0, "", 0.2, 1)
        ### features.shape = [sample_num, std, flop, ast_features, node_ids, serialized_tree>]
        features, tmp_ys = wrap_get_per_store_features_from_measure_pairs(
            [inp], [result],
            get_workload_embedding=False,
            verbose=False,
            parse_ast=True)
                
        xydata = np.concatenate((np.expand_dims(tmp_ys, 1), features), axis=1)
        leaf_no = ASTRawData.leaf_no(xydata[0])
        print(task_id, leaf_no, xydata.shape, task.desc)
    
    raise
    
    all_data = None
    end2end_time = 0
    for task_id, task in enumerate(tasks):

        print(f" - task_id {task_id}/{len(tasks)}, {task.name}")
        tuner_obj = autotvm.tuner.XGBTuner(task, loss_type="rank", feature_type=feature_type)
        data_per_task = profiler_tir_per_task(tuner_obj, measure_option, n_trial)
        print(f" - _data shape {data_per_task.shape}")
        
        if args.per_task:
            np.save(f"{save_path}_task-{task_id}", data_per_task)
        else:
            if all_data is None:
                all_data = data_per_task
            else:
                all_data = np.concatenate((all_data, data_per_task), axis=0)
    
    if not args.per_task:
        np.save(save_path, all_data)
        ### np.load(save_path)

def gen_save_path(model, target, n_trial, feature_type, workspace="."):
    return os.path.join(workspace,
        f"{model}_{target}_{'cus-feature' if CUSTOMIZED_FEATURE else feature_type}_bs-{BATCH_SIZE}_trial-num-{n_trial}")

def profile_tir_per_onnx(model_name, n_trial=10, target = "llvm", feature_type="curve_sum", workspace='.', model_dir="."):
    save_path = gen_save_path(model_name, target, n_trial, feature_type, workspace=workspace)
    
    onnx_model = onnx_utils.load_onnx_model(model_name, model_dir=model_dir)
    onnx_utils.change_onnx_input_dim(onnx_model, batch_size=BATCH_SIZE)
    input_dict = onnx_utils.parse_onnx_model_input(onnx_model, batch_size=BATCH_SIZE)
    
    print(f"Sample TIR features based on {model_name}")
    mod, params = relay.frontend.from_onnx(onnx_model, input_dict)

    profile_tir_per_mod(mod, params, target, save_path,
        feature_type=feature_type, n_trial = n_trial)

def profile_tir_per_transformer(model_name, n_trial=10, target = "llvm", feature_type="curve_sum", workspace='.', model_dir="."):
    save_path = gen_save_path(model_name, target, n_trial, feature_type, workspace=workspace)

    print(f"Sample TIR features based on {model_name}")
    mod, params, input_dict = tvm_transformers.wrap_import_graphdef(model_name, BATCH_SIZE, seq_len=128)
    profile_tir_per_mod(mod, params, target, save_path,
        feature_type=feature_type, n_trial = n_trial)

###############################################################################
# Dump TIR Info of a Module
# -------------------
# Including graph info, tir text and end2end performance    
def dump_tir_info(
        model_name,
        mod,
        params,
        tir_dump_dir,
        input_dict,
        target = "cuda",
        execute=False):
    ### Set the path to dump TIR text, <tir_dump_dir>/tir.txt
    os.environ["TVM_DUMP_TIR_DIR"] = tir_dump_dir
    ### Convert Relay IR to Tensor IR and perform codegen
    with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.FuseOps.max_depth": 0},
        ):
        lib = relay.build(mod, target=target, params=params)

    graph, _lib, _params = lib.get_executor_config(), lib.lib, lib.params
    with open(os.path.join(tir_dump_dir, 
            "{}_tir_graph.json".format(model_name)), 'w') as fp:
        json.dump(json.loads(lib.graph_json), fp, indent=4)    
    os.system("mv {} {}".format(
        os.path.join(tir_dump_dir, "tir.txt"),
        os.path.join(tir_dump_dir, "{}_tir_text.txt".format(model_name))
    ))

    if execute:
        dev = tvm.device(str(target), 0)
        if DEBUG_ON:
            module = debug_executor.create(graph, _lib, dev, dump_root=".workspace/tvmdbg")
        else:
            module = graph_executor.GraphModule(lib["default"](dev))

        ### Set input
        for input_name, input_shape in input_dict.items():  
            input_data = np.ones(input_shape)
            module.set_input(input_name, input_data)

        module.run()
        if DEBUG_ON:
            return
        # tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
        timing_number = 10
        timing_repeat = 10
        unoptimized = (
            np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
            * 1000
            / timing_number
        )
        print(f"Run time performance {unoptimized}")

        with open(os.path.join(tir_dump_dir, 
                "{}_tir_perf.txt".format(model_name)), 'w') as fp:
            fp.write(f"Performance of {model_name} on target {target} = {np.average(unoptimized)} ms\n({unoptimized})")

def dump_tir_info_via_onnx(
        model_name,
        tir_dump_dir,
        target = "cuda",
        execute=False):
    onnx_model = onnx_utils.load_onnx_model(model_name, model_dir=args.model_dir)
    onnx_utils.change_onnx_input_dim(onnx_model, batch_size=BATCH_SIZE)
    input_dict = onnx_utils.parse_onnx_model_input(onnx_model, batch_size=BATCH_SIZE)

    print(input_dict)
    ### Convert ONNX model to Relay IR Module
    mod, params = relay.frontend.from_onnx(onnx_model, input_dict)
    # mod = relay.transform.DynamicToStatic()(mod)
    dump_tir_info(model_name, mod, params, tir_dump_dir, input_dict,
        target=target,
        execute=execute)

def make_measurer(run_timeout, repeat, number, enable_cpu_cache_flush,
                  verbose):
    ''' Make a measure, refer to tenset's script: measure_programs.py
    '''
    
    builder = auto_scheduler.measure.LocalBuilder()
    # runner = auto_scheduler.measure.LocalRunner(
    #     timeout=run_timeout, repeat=repeat, number=number,
    #     enable_cpu_cache_flush=enable_cpu_cache_flush)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(timeout=run_timeout)
    runner = measure_ctx.runner

    measurer = auto_scheduler.measure.ProgramMeasurer(
        builder,
        runner,
        # [auto_scheduler.RecordToFile(log_filename)],
        [],
	    verbose=verbose,
    )
    return measurer

def test(mod, params, target, num_measures_per_round=32):
    ''' Ref to auto_scheduler.TaskScheduler
    '''
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    # Run measurement
    target = tvm.target.Target(target)

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)
    
    for task_idx, task in enumerate(tasks):
        
        # Set measurement arguments
        measurer_kwargs = {
            "run_timeout": 5,
            "number": 1,
            "enable_cpu_cache_flush": True,
            "verbose": 1,
        }
        if task.compute_dag.flop_ct >= 2416443392.0:
            measurer_kwargs['repeat'] = 4
        elif task.compute_dag.flop_ct >= 834928640.0:
            measurer_kwargs['repeat'] = 6
        elif task.compute_dag.flop_ct <= 2097152.0:
            measurer_kwargs['repeat'] = 10
        else:
            measurer_kwargs['repeat'] = 8

        # Make measuer
        # measurer_kwargs['log_filename'] = log_filename
        measurer = make_measurer(**measurer_kwargs)
        
        # policy = auto_scheduler.SketchPolicy(task, verbose=0)
        policy = auto_scheduler.EmptyPolicy(task)
        print(task.compute_dag.init_state)

        _inputs = []
        _results = []
        target_len = 2
        while len(_inputs) < target_len:
            measure_inputs, measure_results = policy.continue_search_one_round(
                num_measures_per_round, measurer
            )
            for idx, res in enumerate(measure_results):
                print(res)
                raise
                if res.error_no == auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
                    _inputs.append(measure_inputs[idx])
                    _results.append(res)

        features, tmp_ys = wrap_get_per_store_features_from_measure_pairs(_inputs, _results, task)
        print(features.shape)
        raise

if __name__ == "__main__":
    # logging config (for printing tuning log to screen)
    # logging.getLogger("autotvm").setLevel(logging.DEBUG)
    # logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    onnx_model_list = [
        # "resnet50-habana",
        "resnet18-habana",
        # "resnet50",
        # "densenet-121",
        # "vgg16",
        # "alexnet"
        # "caffenet"
        # "inceptionv2"
        # "mobilenet"
        # "gpt2"
        ]
    transformer_model_list = [
        # "bert_base",
    ]

    taget_list = [
        "llvm",
        "cuda",
    ]

    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    gpu_utils.check_gpu_util()

    if args.option == "profile":
        for model in onnx_model_list:
            profile_tir_per_onnx(
                model,
                n_trial = PROFILE_TRIAL_NUM,
                target = taget_list[1],
                feature_type="curve_sum",
                workspace=args.workspace,
                model_dir=args.model_dir
            )
        for model in transformer_model_list:
            profile_tir_per_transformer(
                model,
                n_trial = PROFILE_TRIAL_NUM,
                target = taget_list[1],
                feature_type="curve_sum",
                workspace=args.workspace,
                model_dir=args.model_dir
            )
    elif args.option == "tir_info":
        for model_name in onnx_model_list:
            print("Profile TIR info for {}".format(model_name))
            dump_tir_info_via_onnx(
                model_name,
                args.workspace,
                target = "cuda",
                execute = True
            )
        for model_name in transformer_model_list:
            print("Profile TIR info for {}".format(model_name))
            mod, params, input_dict = tvm_transformers.wrap_import_graphdef(model_name, BATCH_SIZE, seq_len=128)
            print(input_dict)
            dump_tir_info(model_name, mod, params, args.workspace, input_dict,
                target="cuda",
                execute=True)
    elif args.option == "tir_reshape":
        from tvm_helper.tir_helper import reshape_tir_feature
        reshape_tir_feature(13, 4, args.workspace, ADDITIONAL_DIM_LEN)
    else:
        raise ValueError(args.option)
