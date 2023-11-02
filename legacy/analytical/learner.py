import os
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
from datetime import datetime
import re

from utils.base import DIMENSION_NAME
from utils.util import FULL_HEADERS, idw_average, Filters, notify
from utils.env import PROJECT_CFG
from utils.op_info import op_type2kernel_type, feature_encoder, raw_feature_index
from utils.dtype_info import convert2std_dtype
from utils.device_info import short_device_name

from analytical.cost_model.cutlass_cm import CUTLASS_CM, find_evaluate_cutlass_kernel_iplmt
from dataloader import collect_data

from dpro import collect
from dpro import trace_utils
from dpro import replay

def feature_style_convert(dpro_feature, avg, gpu_model, dtype, model, op_type):
    ### convert feature from dPRO style to standard style defined in op_info.py
    cdpp_feature = [
        avg,
        feature_encoder(DIMENSION_NAME.gpu_model)(gpu_model),
        feature_encoder(DIMENSION_NAME.dtype)(dtype),
        feature_encoder(DIMENSION_NAME.model)(model),
        feature_encoder(DIMENSION_NAME.op_type)(op_type)]

    if op_type == "Conv2D":
        bs =  dpro_feature[-2]
        remain = dpro_feature[:-2]
        cdpp_feature += [bs] + remain

        H = raw_feature_index(cdpp_feature, "H", op_type=op_type)
        P = raw_feature_index(cdpp_feature, "P", op_type=op_type)
        stride = round(H / P)
        conv_kind = feature_encoder("conv_kind")("fprop")
        cdpp_feature += [stride, conv_kind]

    elif op_type == "MatMul":
        bs = dpro_feature[-1]
        remain = dpro_feature[:-1]
        cdpp_feature += [bs] + remain + [1]
    else:
        cdpp_feature += [-1 if elem is None else elem for elem in dpro_feature]
    return cdpp_feature

def parse_cdpp_feature(_clct, op, model_name, gpu_model):
    op_name = trace_utils.parse_op_name(op)
    avg = _clct.trail_dag.nodes[op]["avg"]
    op_type = _clct.para_dict.parse_op_type(op_name)
    dtype = convert2std_dtype(_clct.para_dict.ret_op_precision(op_name))
    metadata_dpro_style = _clct.para_dict.ret_rawmeta(op_name)
    if metadata_dpro_style is None:
        metadata_dpro_style = _clct.para_dict.ret_metadata(op_name)
    return feature_style_convert(
                metadata_dpro_style, avg, gpu_model, dtype,
                model=model_name, op_type=op_type)

def traverse_all_op(_clct, model_name, gpu_model):
    print("\n#################\nTraverse {}".format(model_name))
    for op in _clct.trail_dag.nodes:
        op_name = trace_utils.parse_op_name(op)
        avg = _clct.trail_dag.nodes[op]["avg"]
        op_type = _clct.para_dict.parse_op_type(op_name)
        dtype = convert2std_dtype(_clct.para_dict.ret_op_precision(op_name))
        # print("\n")
        # print(op_name, avg, op_type)
        if op_type in ["Conv2D", "MatMul"]:
            print("\n")
            print(op_name, avg, op_type)
            metadata_dpro_style = _clct.para_dict.ret_rawmeta(op_name)
            feature_style_convert(
                metadata_dpro_style, avg, gpu_model, dtype,
                model=model_name, op_type=op_type)
            

def pred_via_cutalss_kernel_cm(op_type, op, _clct, model, _dag, gpu_model, cutlass_cm, verbose):
    if op_type in [
        "MatMul",
        "Conv2D"
        ]:
        header = FULL_HEADERS[op_type]
        feature = parse_cdpp_feature(_clct, op, model, gpu_model)
        target_kernel_type = op_type2kernel_type(op_type)
        try:
            kernel, predY, error = find_evaluate_cutlass_kernel_iplmt(
                cutlass_cm,
                feature,
                target_kernel_type,
                gpu_model,
                gpu_model,
                header,
                verbose=verbose)
            _dag.nodes[op]["avg"] = predY
            print("Update op {}'s time to {:.3f} ms, from {:.3f}".format(op, predY, _clct.trail_dag.nodes[op]["avg"]))
        except KeyError:
            raise
            print("Failed to query cost model for {}".format(op))
        
    else:
        ### for other op types, seek insights from the source model
        pass

def pred_via_knn(op_type, optype2features, op, _clct, model, _dag, gpu_model, n_neighbors = 2):
    if op_type not in optype2features:
        print("Warning: {} deos not exist in source model".format(op_type))
        return
    if n_neighbors > len(optype2features[op_type]):
        n_neighbors = len(optype2features[op_type])
    feature = parse_cdpp_feature(_clct, op, model, gpu_model)
    nbr_features = np.array(optype2features[op_type])[:, 1:]
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(nbr_features)
    distance, nbr_indxs = neigh.kneighbors([feature[1:]], return_distance=True)
    selected_avgs = np.array(optype2features[op_type])[nbr_indxs[0], 0].reshape((n_neighbors, 1))
    distance = distance[0]
    predY = idw_average(selected_avgs, distance)
    _dag.nodes[op]["avg"] = predY[0]
    # print("Update op {}'s time to {:.3f} ms, from {:.3f}".format(op, predY[0], clct_target.trail_dag.nodes[op]["avg"]))


def cross_model_predict(
        cutlass_cm,
        clct_target,
        model_target,
        clct_source,
        gpu_model,
        verbose=False,
        learn_method=2):
    optype2features = {}
    source_op_type_set = set()
    for op in clct_source.trail_dag.nodes:
        op_name = trace_utils.parse_op_name(op)
        op_type = clct_source.para_dict.parse_op_type(op_name)
        source_op_type_set.add(op_type)
        feature = parse_cdpp_feature(clct_source, op, model_target, gpu_model)
        if feature[0] == 0:
            continue
        if op_type not in optype2features:
            optype2features[op_type] = [] 
        optype2features[op_type].append(feature)

    ### decide the op time of the target model
    target_op_type_set = set()
    _dag = clct_target.trail_dag.copy()
    for op in clct_target.trail_dag.nodes:
        op_name = trace_utils.parse_op_name(op)
        op_type = clct_target.para_dict.parse_op_type(op_name)
        target_op_type_set.add(op_type)

        if learn_method == 0:
            pred_via_cutalss_kernel_cm(op_type, op, clct_target, model_target, _dag, gpu_model, cutlass_cm, verbose)
        elif learn_method == 1:
            pred_via_knn(op_type, optype2features, op, clct_target, model_target, _dag, gpu_model)
        elif learn_method == 2:
            # pred_via_metalearning(op_type, optype2features, op, clct_target, model_target, _dag)
            raise NotImplementedError()

    print("# of op types in the source model: {}".format(len(source_op_type_set)))
    print("# of op types in the target model: {}".format(len(target_op_type_set)))
    print("# of op types in the target model but not in source: {}".format(len(target_op_type_set.difference(source_op_type_set))))
    print("# of common op types in the target and source: {}".format(len(target_op_type_set.intersection(source_op_type_set))))

    return _dag

def wrap_replay(_clct, dag=None):
    _dag = _clct.trail_dag if dag is None else dag
    replayer = replay.Replayer(
        dag=_dag, 
        _step_num=1, 
        leaf_dirs=_clct.all_prefix_list(), 
        dump_path=_clct.pm.path,
        comm_backend=_clct.comm_backend,
        byteps_graph=_clct.byteps_graph)
    step_end_time_ms = [t / 1000 for t in replayer.replay(verbose=False).values()]
    iter_time = max(step_end_time_ms)
    return iter_time

def test_cost_model(root_dir, gpu_model):
    clct_resnet50 = collect.Collector(
            os.path.join(root_dir, "tf_resnet50"), 
            comm_backend = "NONE",
            platform = "TENSORFLOW"
        )
    iter_time = clct_resnet50.init(force_=True)
    replay_time = wrap_replay(clct_resnet50)
    print(iter_time, replay_time)
    # traverse_all_op(clct_resnet50, "ResNet50", GPU_MODEL)

    clct_vgg16 = collect.Collector(
            os.path.join(root_dir, "tf_vgg16"), 
            comm_backend = "NONE",
            platform = "TENSORFLOW"
        )
    iter_time = clct_vgg16.init(force_=True)
    replay_time = wrap_replay(clct_vgg16)
    print(iter_time, replay_time)

    clct_icptv3 = collect.Collector(
            os.path.join(root_dir, "tf_icptv3"), 
            comm_backend = "NONE",
            platform = "TENSORFLOW"
        )
    iter_time = clct_icptv3.init(force_=True)
    replay_time = wrap_replay(clct_icptv3)
    print(iter_time, replay_time)

    cutlass_cm = CUTLASS_CM()
    _dag = cross_model_predict(
        cutlass_cm,
        clct_target=clct_icptv3,
        model_target="InceptionV3",
        clct_source=clct_resnet50,
        gpu_model=gpu_model,
        verbose=False)

    iter_time = wrap_replay(clct_vgg16, dag=_dag)
    print(iter_time)