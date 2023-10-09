import os
import numpy as np

from argparser import parse_args
args = parse_args()

from utils.env import PROJECT_CFG
from utils.device_info import short_device_name


if __name__ == '__main__':
    learning_params = vars(args)
    assert 'ave_lb' in learning_params
    learning_params.update({
        # "gpu_model": GPU_MODEL.lower(),
        "feature_type": None
    })
    if args.source_data == 'op':
        from metalearner.op_based_entry import metric_learning_w_op_level_data
        metric_learning_w_op_level_data(
            PROJECT_CFG,
            args.metric_learner,
            args.force_load_data,
            args.opt,
            args.op_type)
        exit(0)
    elif args.source_data == 'tir':
        from metalearner.tir_based_entry import learn_entry, entry_init
        
        if args.option.startswith("train") or args.option.startswith("run"):
            learn_entry(learning_params)
            exit(0) 
        elif args.option.startswith("tune"):
            learn_entry(learning_params, tune=True)
            exit(0)
        elif args.option.startswith('replay'):
            from metalearner.tir_based_entry import end2end_replay
            end2end_replay(learning_params)
        elif args.option.startswith("analyze"):
            from metalearner.tir_based_entry import analyze_entry
            analyze_entry(learning_params)
        elif args.option.startswith("test"):
            from metalearner.tir_based_entry import test_entry
            test_entry(learning_params)
        elif args.option.startswith("reshape_feature"):
            from tvm_helper.tir_helper import reshape_tir_feature
            from profiler.profile_tir.profile_tir import ADDITIONAL_DIM_LEN
            trace_root_path = os.path.join(PROJECT_CFG["source_data_root"], "tir_traces/T4")
            reshape_tir_feature(13, 4, trace_root_path, ADDITIONAL_DIM_LEN)
        elif args.option.startswith("tvmdbg"):
            import json
            tvm_dbg_dir = ".workspace/tvmdbg/_tvmdbg_device_CUDA_0/"
            graph_file = "_tvmdbg_graph_dump.json"
            trace_file = "_tvmdbg_execution_trace.json"
            
            with open(os.path.join(tvm_dbg_dir, trace_file), 'r') as fp:
                traces = json.load(fp)["traceEvents"]
            node_time_dict = {}
            for idx in range(len(traces)):
                trace = traces[idx]
                if ("dense" in trace["name"] or "conv2d" in trace["name"]) and trace["ph"] == "B":
                    next_trace = traces[idx+1]
                    assert trace["name"] == next_trace["name"] and next_trace["ph"] == "E"
                    node_time_dict[trace["name"]] = (next_trace["ts"] - trace["ts"]) / 1000.
                    print(trace["name"], node_time_dict[trace["name"]])

            with open(os.path.join(tvm_dbg_dir, graph_file), 'r') as fp:
                graph = json.load(fp)
            sum = 0
            nodes = graph["nodes"]
            func_time_dict = {}
            for node in nodes:
                if not ("conv2d" in node["name"] or "dense" in trace["name"]):
                    continue
                func_name = node["attrs"]["func_name"]
                # print(func_name)
                if func_name not in func_time_dict:
                    func_time_dict[func_name] = []
                func_time_dict[func_name].append(
                    node_time_dict[node["name"]]
                )
            for func_name, func_times in func_time_dict.items():
                sum += np.mean(func_times)
            print(f"Sum of the average TIR function time: {sum} ms, over {len(func_time_dict)} TIR functions")
        elif args.option.startswith("make_dataset"):
            from metalearner.data.preprocess import test_data_index, make_dataset
            # test_data_index(learning_params)
            trace_root_path = entry_init(learning_params)
            make_dataset(learning_params)
        elif args.option.startswith("metadata"):
            from utils.util import sample_task_files, MODE_DEL
            trace_root_path = entry_init(learning_params)
            mode = ""
            root, dirs, files = list(os.walk(learning_params["input"]))[0]
            for _device in dirs:
                if _device.startswith("metainfo"):
                    continue
                print(f"< Device {_device}")
                if len(mode) == 0:
                    mode += f"{_device}:cross"
                else:
                    mode += f"{MODE_DEL.inter_device}{_device}:cross"

            device2task = sample_task_files(learning_params["input"], mode, learning_params["gpu_model"])
            print(device2task)

            from metalearner.data.rawdata import parse_metainfo
            parse_metainfo(device2task, learning_params, use_default_path=True)
 
        elif args.option.startswith("verify"):
            ### TODO(huhanpeng): integrate this part into --option=analyze
            trace_root_path = entry_init(learning_params)
            from metalearner.cross_task_similarity import cross_task_analyze_entry
            from metalearner.feature import ALL_FEATURE_TYPE, is_feature_type
            if learning_params["tiramisu"]:
                assert is_feature_type(ALL_FEATURE_TYPE.ast_ansor)
                raise NotImplementedError()
            else:
                cross_task_analyze_entry(trace_root_path, learning_params)
        elif args.option.startswith("sche_search"):
            from metalearner.tir_based_entry import sche_search_entry
            sche_search_entry(learning_params)
        else:
            raise ValueError(args.source_data, args.option)
    else:
        raise ValueError(args.source_data)
        from analytical.learner import test_cost_model
        root_dir = "/home/tiger/ws/CrossDevicePredict/cross_model_traces/V100"
        test_cost_model(root_dir, gpu_model)