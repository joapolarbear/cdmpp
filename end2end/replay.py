"""Estimate the total latency of a network using the trained cost model
"""
import argparse
import networkx as nx
import json
import numpy as np
from typing import Union, List
import os
import pandas as pd

from tvm import auto_scheduler
    
from metalearner.learner import InferBuffer, BaseLearner
from metalearner.data.rawdata import ASTRawData
from metalearner.data.dataloader import MyDataSet
from metalearner.clustering import LearnerClusterHub
from tvm_helper.tir_helper import (
    wrap_get_per_store_features_from_measure_pairs,
    parse_tir_feature
)

from .preprocess import get_network_with_states
from .util import _network_target_dir, _replay_rst_filename
from .measure import do_measure_inputs

# Performance Prediciton

def default_dpro_node_name(name):
    return "default->FW.{}".format(name)

'''
_dag, _ = parse_tvm_graph(
            os.path.join(target_model_path, "{}_tir_graph.json".format(target_model_name)),
            os.path.join(target_model_path, "{}_tir_text.txt".format(target_model_name)),
            learner)
'''
def parse_tvm_graph(graph_path, tir_text_path,
        learner=None,
        batch_size=32):
    ''' Parse tvm graph from the dumped JSON file
    Sample nodes in the JSON file:
        {
            "op": "null",
            "name": "p3",
            "inputs": []
        },
        {
            "op": "tvm_op",
            "name": "tvmgen_default_fused_multiply",
            "attrs": {
                "num_outputs": "1",
                "num_inputs": "2",
                "flatten_data": "0",
                "hash": "eb8e1349d0f16905",
                "func_name": "tvmgen_default_fused_multiply"
            },
            "inputs": [
                [
                    6,
                    0,
                    0
                ],
                [
                    7,
                    0,
                    0
                ]
            ]
        },
    
    Parameters
    ----------
    graph_path:
        Path to store the tir graph of the target model
    tir_text_path:
        Path to store the tir text
    learner:
        Metric learner used to estimate the execution time of each tir kernel

    Return
    -------
    G: networkx.DiGraph
        Dependency graph
    tir_func2nodes: str-set dict
        Map from TIR functions to graph nodes
    '''

    with open(graph_path, 'r') as fp:
        graph = json.load(fp)

    ### Read TIR for TIR kernels
    tir2feature = dict(parse_tir_feature(tir_text_path))

    G = nx.DiGraph()
    nodes = graph["nodes"]
    edges_to_add = set()
    tir_func2nodes = {}

    def assign_pred_rst(_key, _value):
        # if _value < 0 or _value > 10:
        #     G.nodes[_key]["avg"] = 0.01
        G.nodes[_key]["avg"] = max(float(_value), 0.0)
        if "__nop" not in _key and "conv2d" in _key:
            print("Node {}, avg = {:.6f}".format(
                _key, G.nodes[_key]["avg"]))

    infer_buffer = InferBuffer(learner, batch_size=batch_size, assign_pred_rst=assign_pred_rst) 
        
    for node in nodes:
        if node["op"] != "tvm_op":
            continue
        node_name = default_dpro_node_name(node["name"])
        G.add_node(node_name, **node["attrs"])
        
        ### Statistic the mapping between tir nodes and tir functions
        func_name = node["attrs"]["func_name"]
        if func_name not in tir_func2nodes:
            tir_func2nodes[func_name] = set()
        tir_func2nodes[func_name].add(node_name)

        assert learner is not None
        if func_name not in tir2feature:
            if func_name != "__nop":
                print(f"Can not find tir kernel {func_name} in the dumped TIR")
            G.nodes[node_name]["avg"] = 0
            # import code
            # code.interact(local=locals())
        else:
            tir_feature = np.array(tir2feature[func_name]).flatten()
            infer_buffer.add(node_name, tir_feature)
        
        ### Add edges to the DFG
        for _input in node["inputs"]:
            assert len(_input) == 3 and _input[1] == 0 and _input[2] == 0, node
            _input_node_id = _input[0]
            if nodes[_input_node_id]["op"] == "tvm_op":
                edges_to_add.add(
                    (default_dpro_node_name(nodes[_input_node_id]["name"]),
                    node_name))

    infer_buffer.finalize()
    G.add_edges_from(edges_to_add)

    # for func_name, nodes in tir_func2nodes.items():
    #     if len(nodes) > 1:
    #         print(func_name, nodes)

    conv2d_funcs = [func for func in tir_func2nodes.keys() if "conv2d" in func]
    conv2d_nodes = [node for node in G.nodes if "conv2d" in node]
    print("Graph Information")
    print(f" - # of tir funcs: {len(tir_func2nodes)}")
    print(f" - # of tir conv2d funcs: {len(conv2d_funcs)}")
    print(f" - # of nodes in the DFG: {len(G.nodes)}")
    print(f" - # of conv2d nodes: {len(conv2d_nodes)}")
    return G, tir_func2nodes

######################## Estimation ##############################################
def get_tir_func_stat(network_args, target, is_ast_feature=True,
        learner_hub: Union[None, LearnerClusterHub]=None,
        cache_dir="tmp/end2end", use_tenset_cost=False):
    ''' Infer the cost model for each of the TIR kernels/tasks
    '''

    tir_func2stat = {}
    mod, params, inputs, task_info = get_network_with_states(network_args, target, cache_dir)
    
    for idx, task in enumerate(task_info.tasks):
        state, _dur = task_info.query_state_by_task(task)
        leaf_no = -1
        if is_ast_feature:
            if learner_hub is None:
                estimated_time = 0
            else:

                inp = auto_scheduler.MeasureInput(task, state)
                # MeasureResult(self, costs, error_no, error_msg, all_cost, timestamp)
                if _dur is None:
                    result = auto_scheduler.measure.MeasureResult([1.], 0, "", 0.2, 1)
                else:
                    result = auto_scheduler.measure.MeasureResult([_dur], 0, "", 0.2, 1)

                ### features.shape = [sample_num, <std, flop, ast_features, node_ids, serialized_tree>]
                features, tmp_ys = wrap_get_per_store_features_from_measure_pairs(
                        [inp], [result],
                        get_workload_embedding=False,
                        verbose=False,
                        parse_ast=True)
                        
                xydata = np.concatenate((np.expand_dims(tmp_ys, 1), features), axis=1)
                leaf_no = ASTRawData.leaf_no(xydata[0])
                if _dur is not None:
                    task_info.xydata.append(xydata[0])

                if features[0][1] < 0:
                    ### TODO (huhanpeng): how to handle samples with flops = -1
                    estimated_time = 0
                else:
                    learner = learner_hub.switch({"leaf_node_no": leaf_no, "FLOPs": features[0][1]})
                    if learner is None:
                        estimated_time = 0
                    else:
                        raw_data = ASTRawData(xydata, learner.data_meta_info, disable_filter=True)
                        dataset = MyDataSet(raw_data)
                        ### We have encode filters into the LearnerClusterHub:learner_hub, 
                        # in valid samples should have been filtered out at learner_hub.switch
                        assert len(dataset) > 0
                        input_data, sample_cnt = learner.dataset2xylist(dataset)
                        assert len(input_data) == 1
                        X, Y = input_data[0]
                        assert len(X) == 1
                        X = learner.data_to_train_device(X)[0]
                        estimated_time = float(learner.predict(X)[0][0])
        else:
            raise NotImplementedError()
        
        if use_tenset_cost:
            estimated_time = _dur if _dur is not None else 0
            cost_src = "T_tenset"
        else:
            cost_src = "T_predict" 
        
        task_info.record_task_runtime_info(task, {
            "cost": estimated_time,
            "cost_src": cost_src,
            "leaf_no": leaf_no,
            "workload_key": task.workload_key,
            "FLOPs": task.compute_dag.flop_ct,
            "T_tenset": _dur if _dur is not None else -1
        })

    return task_info

def get_tir_func_stat_via_profile(network_args, target, cache_dir="tmp/end2end"):

    cost_src = "T_profile"

    mod, params, inputs, task_info = get_network_with_states(network_args, target, cache_dir)

    measure_inp_state = task_info.get_measure_inps()

    to_measure_pair = []
    measured_task_dur_pair = []
    use_tenset_profiled = False
    for task_id in range(len(task_info.tasks)):
        inp, dur = measure_inp_state[task_id]
        if use_tenset_profiled and dur is not None:
            measured_task_dur_pair.append((task_info.tasks[task_id], dur))
        else:
            to_measure_pair.append((task_info.tasks[task_id], inp))
    
    ### Newly measured tasks
    dur_std_list = do_measure_inputs([inp for task, inp in to_measure_pair], target)
    for idx in range(len(to_measure_pair)):
        estimated_time = dur_std_list[idx][0]
        task, _ = to_measure_pair[idx]
        task_info.record_task_runtime_info(task, {
            "cost": estimated_time,
            "cost_src": cost_src,
            "leaf_no": -1,
            "workload_key": task.workload_key,
            "FLOPs": task.compute_dag.flop_ct,
            "T_tenset": -1
        })

    ### For tenset measured tasks
    for task, dur in measured_task_dur_pair:
        task_info.record_task_runtime_info(task, {
            "cost": dur,
            "cost_src": cost_src,
            "leaf_no": -1,
            "workload_key": task.workload_key,
            "FLOPs": task.compute_dag.flop_ct,
            "T_tenset": -1
        })
    
    return task_info

def replay_simple_sum(task_info, _replay_log, stat_pd: pd.DataFrame):
    total_latency = 0
    for task_idx, task in enumerate(task_info.tasks):
        stat = task_info.query_runtime_info_by_task(task)
        # _replay_log(f"Task {task_idx} (workload key: {stat['workload_key'][:20]}...): "
        #     f"T'={stat['cost']:.6f}, T={stat['T_tenset']:.6f}, W={task_info.task_weights[task_idx]}, "
        #     f"N_leaf={stat['leaf_no']}, src={task_info.task_state_src[task_idx]}, FLOPs={stat['FLOPs']}")
        # # print(task.compute_dag)
        if task_idx not in stat_pd.index:
            # "T_profile", "T_tenset", "T_predict", "weight", "N_leaf", "src", "flop_ct", "workload_key":
            stat_pd.loc[task_idx] = [None, None, None, task_info.task_weights[task_idx], stat['leaf_no'],
                task_info.task_state_src[task_idx], stat['FLOPs'], f"{stat['workload_key'][:20]}..."]

        stat_pd.loc[task_idx, stat["cost_src"]] = stat['cost']
        if stat["T_tenset"] > 0:
            stat_pd.loc[task_idx, "T_tenset"] = stat['T_tenset']

        if stat_pd.loc[task_idx, "N_leaf"] < 0 and stat['leaf_no'] > 0:
            stat_pd.loc[task_idx, "N_leaf"] = stat['leaf_no']

        _cost = stat["cost"]
        if stat["cost_src"] == "T_predict":
            T_tenset = stat_pd.loc[task_idx, "T_tenset"]
            T_profile = stat_pd.loc[task_idx, "T_profile"]
            if T_tenset > 0 and T_profile > 0:
                _cost = _cost * T_profile / T_tenset

        # if stat['cost'] is not None and stat['cost'] > 0:
        #     _cost = stat["cost"]
        # elif stat_pd.loc[task_idx, "T_tenset"] > 0:
        #     _cost = stat_pd.loc[task_idx, "T_tenset"]
        # elif stat_pd.loc[task_idx, "T_profile"] > 0:
        #     _cost = stat_pd.loc[task_idx, "T_profile"]
        # else:
        #     _cost = stat["cost"]

        total_latency += _cost * task_info.task_weights[task_idx]
    return total_latency

def replay_via_dpro(task_info, graph, _replay_log):
    import dpro
    
    G = nx.DiGraph()
    nodes = graph["nodes"]
    edges_to_add = set()
    tir_func2nodes = {}

    for node in nodes:
        if node["op"] != "tvm_op":
            continue
        node_name = default_dpro_node_name(node["name"])
        G.add_node(node_name, **node["attrs"])
        
        ### Statistic the mapping between tir nodes and tir functions
        func_name = node["attrs"]["func_name"]
        if func_name not in tir_func2nodes:
            tir_func2nodes[func_name] = set()
        tir_func2nodes[func_name].add(node_name)

        if func_name not in task_info.func_name2worload_key:
            if func_name != "__nop":
                _replay_log(f"Can not find tir kernel {func_name} in the dumped TIR")
            G.nodes[node_name]["avg"] = 0
        else:
            G.nodes[node_name]["avg"] = task_info.query_func_cost(func_name)
        
        ### Add edges to the DFG
        for _input in node["inputs"]:
            assert len(_input) == 3 and _input[1] == 0 and _input[2] == 0, node
            _input_node_id = _input[0]
            if nodes[_input_node_id]["op"] == "tvm_op":
                edges_to_add.add(
                    (default_dpro_node_name(nodes[_input_node_id]["name"]),
                    node_name))

    G.add_edges_from(edges_to_add)

    # for func_name, nodes in tir_func2nodes.items():
    #     if len(nodes) > 1:
    #         print(func_name, nodes)

    conv2d_funcs = [func for func in tir_func2nodes.keys() if "conv2d" in func]
    conv2d_nodes = [node for node in G.nodes if "conv2d" in node]
    _replay_log("Graph Information")
    _replay_log(f" - # of tir funcs: {len(tir_func2nodes)}")
    _replay_log(f" - # of tir conv2d funcs: {len(conv2d_funcs)}")
    _replay_log(f" - # of nodes in the DFG: {len(G.nodes)}")
    _replay_log(f" - # of conv2d nodes: {len(conv2d_nodes)}")

    replayer = dpro.replay.Replayer(dag=G, _step_num=1, leaf_dirs=None, 
            dump_path=".", comm_backend="NONE", byteps_graph=None)
    step_end_time = [t for t in replayer.replay(verbose=False).values()]
    iter_time = max(step_end_time)
    return iter_time

def estimaet_network(network_args, target, is_ast_feature=True,
        learner_hub: Union[None, LearnerClusterHub]=None,
        cache_root="tmp/end2end",
        via=""):

    cache_dir = _network_target_dir(cache_root, network_args, target)

    if via == "profile":
        task_info = get_tir_func_stat_via_profile(network_args, target)
    else:
        task_info = get_tir_func_stat(network_args, target, is_ast_feature=is_ast_feature,
            learner_hub=learner_hub, use_tenset_cost=(via=="tenset"))

    pd_path = os.path.join(cache_dir, "statistic.csv")
    if os.path.exists(pd_path):
        stat_pd = pd.read_csv(pd_path)
    else:
        stat_pd = pd.DataFrame(data={
            "task_id": [], "T_profile": [], "T_tenset": [], "T_predict": [],
            "weight": [], "N_leaf": [], "src": [], "flop_ct": [], "workload_key": []})
    stat_pd.set_index("task_id", inplace = True)

    with open(os.path.join(cache_dir, _replay_rst_filename(via=via)), 'w') as replay_fp:

        def replay_log(msg):
            print(msg)
            replay_fp.write(msg+"\n")

        ### Via simple summation
        total_latency = replay_simple_sum(task_info, replay_log, stat_pd) * 1000
        msg = f"[Simple] Estimated total latency of {network_args['network']}: {total_latency:.2f} ms"
        replay_log(msg)

        ### Via dPRO's replayer
        # with open(os.path.join(cache_dir, _graph_filename()), 'r') as fp:
        #     graph = json.load(fp)
        # total_latency = replay_via_dpro(task_info, graph, replay_log)
        # msg = f"[dPRO] Estimated total latency of {network_args['network']}: {total_latency:.2f} ms"
        # replay_log(msg)
    
    ### Save statistic results
    stat_pd.to_csv(pd_path)
    if len(task_info.xydata) > 0:
        xydata = np.array(task_info.xydata, dtype=object)
        np.save(os.path.join(cache_dir, "xydata.npy"), xydata, allow_pickle=True)

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

    estimaet_network(network_args, args.target)

