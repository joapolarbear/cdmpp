import os
import sys
import json
import numpy as np
import pickle
import time
import csv
import math

from utils.base import DIMENSION_NAME
from utils.util import FULL_HEADERS, Scaler, line2list
from utils.util import ret_model_name, ret_dtype, ret_bs, bs2dir
from utils.device_info import ALL_GPU_MODEL, DEVICE_INFO, short_device_name, query_core_num
from utils.cutlass_api import parse_kernel_name
from utils.op_info import eq_gemm_size, feature_encoder, feature_decode
from utils.op_info import raw_feature_index, enriched_feature_index, enriched_header

def parse_metadata(metadata_path, bs):
    with open(metadata_path, 'r') as fp:
        metadata = json.load(fp)

    cache_hyper_para = {}
    for op_name in metadata.keys():
        inputs = metadata[op_name]["input"]
        outputs = metadata[op_name]["output"]
        op_type = metadata[op_name]["op"]
        if op_type == "Conv2D":
            assert len(outputs) == 1
            shape_ = outputs[0]["shape"]
            assert len(shape_) == 4, (outputs[0]["shape"], metadata[op_name])
            N = shape_[0]
            assert N == bs
            if N is None:
                raise ValueError(metadata[op_name])

            ### TODO (huhanpeng), assume the width=height
            P = Q = shape_[2]
            ### Consider different layouts
            if shape_[1] == P:
                is_NCHW = False
                K = shape_[3]
            else:
                is_NCHW = True
                K = shape_[1]

            assert len(inputs) == 2
            C = None         # input channel
            H = W = None     # Input height/weight
            R = S = None     # filter size

            ### The first 'input' is input, the second 'input' is the filter
            ### TODO (huhanpeng): need to verify
            H = W = inputs[0]["shape"][2]
            if is_NCHW:
                C = inputs[0]["shape"][1]
            else:
                ### NHWC
                C = inputs[0]["shape"][3]
            R, S = inputs[1]["shape"][0], inputs[1]["shape"][1]
            assert inputs[1]["shape"][2] == C   # input channel
            assert inputs[1]["shape"][3] == K   # output channel
            stride = round(H / P)
            cache_hyper_para[op_name] = [H, W, C, R, S, P, Q, K, stride, feature_encoder("conv_kind")("fprop")]
        elif op_type == "MatMul":
            assert len(outputs) == 1, metadata[op_name]
            assert len(inputs) == 2, metadata[op_name]
            output_shape_ = outputs[0]["shape"]
            input_shape0 = inputs[0]["shape"]
            input_shape1 = inputs[1]["shape"]
            if output_shape_[0] == input_shape0[0]:
                ### NOTE, bs_prime may not be equal to bs
                bs_prime, C_out = output_shape_
                C_in = input_shape0[1]
            elif output_shape_[0] == input_shape1[0]:
                ### NOTE, bs_prime may not be equal to bs
                bs_prime, C_out = output_shape_
                C_in = input_shape1[1]
            elif input_shape0[0] == input_shape1[0]:
                ### NOTE, bs_prime may not be equal to bs
                C_in, C_out = output_shape_
                bs_prime = input_shape0[0]
            else:
                raise ValueError(op_name, metadata[op_name])

            cache_hyper_para[op_name] = [C_in, C_out, bs_prime / bs]
        elif op_type == "MatMul":
            pass

    return cache_hyper_para

def check_metadata_equal(metadatas):
    equal = True
    for i in range(1, len(metadatas)):
        if metadatas[i] != metadatas[i-1]:
            print("Two different metadata files: {} VS {}".format(
                metadata_paths[i-1], metadata_paths[i]))
            equal = False
    if equal:
        print("All metadata files are the same.")

home_path = "/home/tiger"
metadata_paths = [
    ### different GPU models
    # "{}/tos_traces/traces/TF_2_4_ResNet50/Tesla_V100-SXM2-32GB/dtype_fp32/BS_1/tf/metadata.json".format(home_path),
    # "{}/tos_traces/traces/TF_2_4_ResNet50/A100-SXM4-40GB/dtype_fp32/BS_1/tf/metadata.json".format(home_path),
    # "{}/tos_traces/traces/TF_2_4_ResNet50/Tesla_T4/dtype_fp32/BS_1/tf/metadata.json".format(home_path),

    "{}/tos_traces/traces/TF_2_4_ResNet50/Tesla_V100-SXM2-32GB/dtype_fp32/BS_1/tf/metadata.json".format(home_path),
    "{}/tos_traces/traces/TF_2_4_ResNet50/Tesla_V100-SXM2-32GB/dtype_fp32/BS_2/tf/metadata.json".format(home_path),
    "{}/tos_traces/traces/TF_2_4_ResNet50/Tesla_V100-SXM2-32GB/dtype_fp32/BS_4/tf/metadata.json".format(home_path),
]
# metadatas = [parse_metadata(_path) for _path in metadata_paths]
# check_metadata_equal(metadatas)


def parse_traces(trace_path, filters=None):
    if trace_path.endswith(".gz"):
        os.system("gzip -fdk {}".format(trace_path))
        json_path = trace_path.replace(".gz", "")
        with open(json_path, 'r') as fp:
            traces = json.load(fp)["traceEvents"]
        os.system("rm {}".format(json_path))
    else:
        with open(trace_path, 'r') as fp:
            traces = json.load(fp)["traceEvents"]

    pid_dict = {}
    name2stat = {}

    cur_step = None
    step_start = None
    step_end = None
    iter_time = []

    for trace in traces:
        if "ph" not in trace:
            continue
        if trace["ph"] == "M":
            if trace["name"] == "process_name" and trace["pid"] not in pid_dict:
                pid_dict[trace["pid"]] = {"name": trace["args"]["name"]}
            if trace["name"] == "thread_name" and trace["tid"] not in pid_dict[trace["pid"]]:
                pid_dict[trace["pid"]][trace["tid"]] = trace["args"]["name"]

        elif trace["ph"] == "X":
            pid_name = pid_dict[trace["pid"]]["name"]
            tid_name = pid_dict[trace["pid"]][trace["tid"]]
            if "/device:GPU" in pid_name:
                if "TensorFlow Ops" in tid_name:
                    try:
                        step_id = int(trace["args"]['group_id'])
                    except:
                        step_id = None
                    if step_id is not None:
                        if cur_step is None:
                            cur_step = step_id
                            step_start = trace["ts"]
                        elif step_id != cur_step:
                            ### New iteration
                            assert step_id == cur_step + 1
                            iter_time.append((step_end - step_start) / 1000.)
                            cur_step = step_id
                            step_start = trace["ts"]
                            step_end = trace["ts"] + trace["dur"]
                        else:
                            ### In one iteration
                            step_end = trace["ts"] + trace["dur"]

                    try:
                        trace_name = trace["args"]["long_name"].split(":")[0]
                        trace_name = trace_name.split(
                            "StatefulPartitionedCall/")[1] if trace_name.startswith("StatefulPartitionedCall") else trace_name
                    except IndexError:
                        continue
                    except:
                        print(trace)
                        raise
                    op_type = trace["name"]
                    ### Apply filters
                    if filters and filters.not_in_filter(DIMENSION_NAME.op_type, op_type):
                        continue

                    # step_id = trace["args"]["group_id"]
                    if trace_name not in name2stat:
                        name2stat[trace_name] = {
                            "cnt": 1,
                            "time": [trace["dur"] / 1000.0],
                            # "min_t": trace["dur"] / 1000.0,
                            # "max_t": trace["dur"] / 1000.0,
                            DIMENSION_NAME.op_type: op_type,
                            # "id": len(self.name2sta)
                        }
                    else:
                        name2stat[trace_name]["cnt"] += 1
                        name2stat[trace_name]["time"].append(
                            trace["dur"] / 1000.0)
                        # name2stat[trace_name]["min_t"] = min(
                        #     name2stat[trace_name]["min_t"], trace["dur"] / 1000.0)
                        # name2stat[trace_name]["max_t"] = max(
                        #     name2stat[trace_name]["max_t"], trace["dur"] / 1000.0)

    for _, statistic in name2stat.items():
        statistic["time"] = np.array(statistic["time"])
        statistic["avg"] = np.average(statistic["time"])
        statistic["median"] = np.median(statistic["time"])
        statistic["var"] = np.std(statistic["time"])
        if statistic["var"] / statistic["avg"] > 0.1:
            statistic["avg"] = np.average(statistic["time"][1:])
            statistic["var"] = np.std(statistic["time"][1:])
        statistic["time"] = None

    name2stat["iter_time"] = np.average(iter_time) if len(iter_time) >0 else None
    return name2stat

def _collect_dnn_traces(root_path, root_dirs, filters=None, per_dnn_trace_hook=None):
    xydata = {}
    for root_dir in sorted(root_dirs):
        model_name, tf_version = ret_model_name(root_dir)
        if filters and filters.not_in_filter(DIMENSION_NAME.model, model_name):
            continue
        print(model_name, tf_version)
        model_path, model_dirs, _ = list(
            os.walk(os.path.join(root_path, root_dir)))[0]
        for model_dir in sorted(model_dirs):
            gpu_model = short_device_name(model_dir)
            if filters and filters.not_in_filter(DIMENSION_NAME.gpu_model, gpu_model):
                continue
            print("- GPU model: ", gpu_model)
            gpu_model_path, gpu_model_dirs, _ = list(
                os.walk(os.path.join(model_path, model_dir)))[0]
            for gpu_model_dir in sorted(gpu_model_dirs):
                dtype = ret_dtype(gpu_model_dir)
                if filters and filters.not_in_filter(DIMENSION_NAME.dtype, dtype):
                    continue
                print(" - dtype = ", dtype)
                dtype_path, dtype_dirs, _ = list(
                    os.walk(os.path.join(gpu_model_path, gpu_model_dir)))[0]
                for dtype_dir in sorted(dtype_dirs):
                    bs = ret_bs(dtype_dir)
                    if filters and filters.not_in_filter(DIMENSION_NAME.bs, bs):
                        continue
                    leaf_dir = os.path.join(
                        os.path.join(dtype_path, dtype_dir), "tf")
                    metadata_path = os.path.join(leaf_dir, "metadata.json")
                    trace_path = os.path.join(leaf_dir, "trace.json.gz")
                    if not os.path.exists(trace_path):
                        continue

                    ### To save the space, we only save on piece of metadata
                    ### Under the directory of bs=1
                    real_meta_bs = None
                    if not os.path.exists(metadata_path):
                        METADATE_PATH_BS = 1
                        metadata_path = os.path.join(os.path.join(
                            dtype_path, bs2dir(METADATE_PATH_BS)), "tf/metadata.json")
                        metadata_path = metadata_path.replace(
                            model_dir, "Tesla_V100-SXM2-32GB")
                        assert os.path.exists(metadata_path), (metadata_path)
                        print(
                            "  - BS = {}, use {} as the metadata path".format(bs, metadata_path))
                        real_meta_bs = METADATE_PATH_BS
                    else:
                        print("  - BS = ", bs)
                        real_meta_bs = bs
                    name2stat = parse_traces(trace_path, filters=filters)
                    meta_data = parse_metadata(metadata_path, real_meta_bs)
                    if per_dnn_trace_hook is not None:
                        per_dnn_trace_hook(name2stat)

                    # from util import ALL_OP_TYPE
                    # print(set([stat["op_type"] for stat in name2stat.values()]).difference(set(ALL_OP_TYPE)))

                    ### for each op in name2stat, form a data point, combining with metadata and base parameters
                    cnt = 0
                    for op_name, stat in name2stat.items():
                        ### TODO (huhanpeng): currently only support conv2d
                        if op_name == "iter_time":
                            continue
                        if stat["op_type"] not in xydata:
                            xydata[stat["op_type"]] = []

                        if model_name.lower() in op_name:
                            op_split = op_name.split("/")
                            op_name = "/".join(op_split[op_split.index(model_name.lower()):])
                        
                        if op_name not in meta_data:
                            print(op_name)
                        else:
                            xydata[stat["op_type"]].append([
                                stat["avg"],
                                feature_encoder(DIMENSION_NAME.gpu_model)(gpu_model),
                                feature_encoder(DIMENSION_NAME.dtype)(dtype),
                                feature_encoder(DIMENSION_NAME.model)(model_name),
                                feature_encoder(DIMENSION_NAME.op_type)(stat["op_type"]),
                                bs] + meta_data.get(op_name, []))
                            cnt += 1
                    print("   - Add {} data points.".format(cnt))
    return xydata

def gen_raw_feature(ret_list, raw_data, gpu_model):
    ''' Generate raw_feature and append the result to ret_list
    '''
    if raw_data[0] > 100:
        return
    ret_list.append([
        raw_data[0],
        feature_encoder(DIMENSION_NAME.gpu_model)(gpu_model),
        feature_encoder(DIMENSION_NAME.dtype)(raw_data[1]),
        feature_encoder(DIMENSION_NAME.model)("None"),
        feature_encoder(DIMENSION_NAME.op_type)(raw_data[2])] + raw_data[3:])

def _wrap_gen_raw_feature(xydata_dict, raw_data, gpu_model, filters=None):
    _op_type = raw_data[2]
    if filters and filters.not_in_filter(DIMENSION_NAME.op_type, _op_type):
        return
    if filters and filters.not_in_filter(DIMENSION_NAME.dtype, raw_data[1]):
        return
    if _op_type not in xydata_dict:
        xydata_dict[_op_type] = []
    if _op_type == "Conv2D":
        raw_data.append(feature_encoder("conv_kind")("fprop"))
    gen_raw_feature(xydata_dict[_op_type], raw_data, gpu_model)

def _collect_per_op_traces(root_path, root_dirs, filters=None):
    xydata = {}
    for root_dir in sorted(root_dirs):
        if root_dir.startswith("."):
            continue
        gpu_model = short_device_name(root_dir)
        if filters and filters.not_in_filter(DIMENSION_NAME.gpu_model, gpu_model):
            continue
        gpu_model_root, _, gpu_model_files = list(
            os.walk(os.path.join(root_path, root_dir)))[0]
        for _file in gpu_model_files:
            if _file.endswith(".json"):
                with open(os.path.join(gpu_model_root, _file), "r") as f:
                    op_type_datas = json.load(f)
                op_type = _file.split(".json")[0]
                ### Apply filters
                if filters and filters.not_in_filter(DIMENSION_NAME.op_type, op_type):
                    continue
                for raw_data in op_type_datas:
                    _wrap_gen_raw_feature(xydata, raw_data, gpu_model, filters=filters)
            elif _file.endswith("_op.txt"):
                op_type = _file.split("_op.txt")[0]
                ### Apply filters
                if filters and filters.not_in_filter(DIMENSION_NAME.op_type, op_type):
                    continue
                with open(os.path.join(gpu_model_root, _file), 'r') as fp:
                    op_trace = fp.readlines()
                for idx in range(len(op_trace)):
                    all_raw_data = line2list(op_trace[idx])
                    for per_op_raw_data in all_raw_data:
                        wrap_gen_raw_feature(xydata, per_op_raw_data, gpu_model, filters=filters)
            else:
                pass
    return xydata

def _collect_cutlass_gemm(header, row, kernel_data_dict, gpu_model):
    kernel = row[3]
    Bytes, Flops, Runtime, GB_per_s, GFLOPs = row[-5], row[-4], row[-3], row[-2], row[-1]
    op_type = row[header.index("OperationKind")]
    beta = row[header.index("beta")]
    if int(beta) != 1:
        return
    bs = row[header.index("m")]
    c_in = row[header.index("k")]
    c_out = row[header.index("n")]
    dtype = row[header.index('accum')]
    dtype = "fp" + dtype.strip("f")
    # print(kernel, Bytes, Flops, Runtime, GB_per_s, GFLOPs)
    if kernel not in kernel_data_dict:
        kernel_data_dict[kernel] = []
    kernel_data_dict[kernel].append([
        float(Runtime),
        feature_encoder(DIMENSION_NAME.gpu_model)(gpu_model),
        feature_encoder(DIMENSION_NAME.dtype)(dtype),
        feature_encoder(DIMENSION_NAME.model)("None"),
        feature_encoder(DIMENSION_NAME.op_type)("MatMul"),
        int(bs), int(c_in), int(c_out), 1])

def _collect_cutlass_conv2d(header, row, kernel_data_dict, gpu_model):
    kernel = row[3]
    Bytes, Flops, Runtime, GB_per_s, GFLOPs = row[-5], row[-4], row[-3], row[-2], row[-1]
    op_type = row[header.index("OperationKind")]
    conv_kind = row[header.index("conv_kind")] # fprop, wgrad or dgrad
    n, h, w, c, k, r, s, p, q = row[header.index("n"):(header.index("q")+1)]
    # assert h == w and r == s and p == q, (header, row)
    stride_h = row[header.index("stride_h")]
    stride_w = row[header.index("stride_w")]
    assert stride_w == stride_h

    dtype = row[header.index('accum')]
    dtype = "fp" + dtype.strip("f")
    # ["H", "W", "C", "R", "S", "P", "Q", "K", "stride"]
    # print(kernel, Bytes, Flops, Runtime, GB_per_s, GFLOPs)
    if kernel not in kernel_data_dict:
        kernel_data_dict[kernel] = []
    kernel_data_dict[kernel].append([
        float(Runtime),
        feature_encoder(DIMENSION_NAME.gpu_model)(gpu_model),
        feature_encoder(DIMENSION_NAME.dtype)(dtype),
        feature_encoder(DIMENSION_NAME.model)("None"),
        feature_encoder(DIMENSION_NAME.op_type)("Conv2D"),
        int(n), int(h), int(w), int(c), int(r),
        int(s), int(p), int(q), int(k), int(stride_h), feature_encoder("conv_kind")(conv_kind)])

def _collect_by_kernel_type(
        csvfile_path,
        cache_dir,
        gpu_model,
        kernel_type,
        cutlass_feature_handler,
        force=False):
    cache_path = os.path.join(cache_dir, "{}-{}".format(gpu_model, kernel_type))
    if not force and os.path.exists(cache_path):
        st = time.time()
        with open(cache_path, "rb") as f:
            kernel_data_dict = pickle.load(f)
        print("Load data at {} using {:.3f}s".format(cache_path, time.time() - st))
        return kernel_data_dict
    
    kernel_data_dict = {}
    header = None
    st = time.time()
    with open(csvfile_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        line_cnt = 0 
        for row in csv_reader:
            if line_cnt == 0:
                header = row
            else:
                cutlass_feature_handler(header, row, kernel_data_dict, gpu_model)
            line_cnt += 1
    with open(cache_path, "wb") as f:
        pickle.dump(kernel_data_dict, f)
    print("Take {} s to collect {} kernels for {}".format(time.time() - st, line_cnt-1, kernel_type))
    return kernel_data_dict

def combine_kernel_dict(dict0, dict1):
    for kernel in dict1.keys():
        if kernel not in dict0:
            dict0[kernel] = dict1[kernel]
        else:
            dict0[kernel] += dict1[kernel]
    
def _collect_data_cutlass(trace_root_path, cache_dir, filters, force=False):
    kernel_data_dict = {}
    for _gpu_model in os.listdir(trace_root_path):
        if not _gpu_model.startswith("cutlass_test_"):
            continue
        gpu_model = short_device_name(_gpu_model.split("cutlass_test_")[1])
        if filters.not_in_filter(DIMENSION_NAME.gpu_model, gpu_model):
            continue
        gpu_model_path = os.path.join(trace_root_path, _gpu_model)
        print(gpu_model)
        for file in os.listdir(gpu_model_path):
            print(" - ", file)
        
            if filters.not_in_filter("kernel", "gemm"):
                pass
            elif file.endswith(".gemm.csv"):
                tmp_kernel_dict = _collect_by_kernel_type(
                    os.path.join(gpu_model_path, file),
                    cache_dir,
                    gpu_model,
                    "gemm",
                    _collect_cutlass_gemm,
                    force=force)
                combine_kernel_dict(kernel_data_dict, tmp_kernel_dict)

            if filters.not_in_filter("kernel", "conv2d"):
                pass
            elif file.endswith(".conv2d.csv"):
                tmp_kernel_dict = _collect_by_kernel_type(
                    os.path.join(gpu_model_path, file),
                    cache_dir,
                    gpu_model,
                    "conv2d_fprop",
                    _collect_cutlass_conv2d,
                    force=force)
                combine_kernel_dict(kernel_data_dict, tmp_kernel_dict)
        
        # for kernel in kernel_data_dict.keys():
        #     kernel_data = kernel_data_dict[kernel]
        #     print(kernel, len(kernel_data))
        #     print(header)
        #     print(kernel_data[0])
        #     raise
    return kernel_data_dict

def collect_data(root_dir, cache_dir, filters=None, cutlass=False, force=False, per_dnn_trace_hook=None):
    ''' Return alldata, shape=(num_samples, ndims)
    '''
    if cutlass:
        kernel2xydata = _collect_data_cutlass(root_dir, cache_dir, filters, force=force)
        return kernel2xydata

    if filters is None:
        cache_path = os.path.join(cache_dir, "data-all.pickle")
    else:
        cache_path = os.path.join(
            cache_dir, "data-{}.pickle".format(filters.serialize_filter()))
    if not force and os.path.exists(cache_path):
        st = time.time()
        with open(cache_path, "rb") as f:
            op2xydata = pickle.load(f)
        print("Load data at {} using {:.3f}s".format(cache_path, time.time() - st))
        return op2xydata

    print("Generate data at {}...".format(cache_path))
    _, root_dirs, _ = list(os.walk(root_dir))[0]
    if root_dirs[0] in ALL_GPU_MODEL:
        print("Collect per-op traces")
        op2xydata = _collect_per_op_traces(root_dir, root_dirs, filters)
    else:
        print("Collect DNN traces")
        op2xydata = _collect_dnn_traces(root_dir, root_dirs, filters, per_dnn_trace_hook=per_dnn_trace_hook)

    with open(cache_path, "wb") as f:
        pickle.dump(op2xydata, f)
    return op2xydata

def enrich_raw_feature(raw_feature, op_type, 
        ave=None,
        source_gpu=None,
        target_gpu=None,
        kernel=None):

    dtype = feature_decode(DIMENSION_NAME.dtype,
        raw_feature_index(raw_feature, DIMENSION_NAME.dtype, op_type))
    
    flops = raw_feature_index(raw_feature, "flops", op_type)
    size = raw_feature_index(raw_feature, "size", op_type)
    transform = raw_feature_index(raw_feature, "transform", op_type)
    ai = raw_feature_index(raw_feature, "ai", op_type)
    perf = raw_feature_index(raw_feature, "perf", op_type)

    if ave is None:
        new_feature = [raw_feature[0]]
    else:
        new_feature = [ave]
    
    new_feature = new_feature + [flops, size, transform, ai, perf]

    assert kernel is not None and \
            source_gpu is not None and target_gpu is not None
    source_sm_num = DEVICE_INFO[source_gpu]["SM_count"]
    target_sm_num = DEVICE_INFO[target_gpu]["SM_count"]
    source_bw = DEVICE_INFO[source_gpu]["BW_GB_per_s"]
    target_bw = DEVICE_INFO[target_gpu]["BW_GB_per_s"]
    source_clock = DEVICE_INFO[source_gpu]["clock_MHz"]
    target_clock = DEVICE_INFO[target_gpu]["clock_MHz"]
    
    kernel_info = parse_kernel_name(kernel)
    Mtile = int(kernel_info["threadblock_m"])
    Mtile = int(kernel_info["threadblock_n"])
    Ktile = int(kernel_info["threadblock_k"])
    pipeline_stage = int(kernel_info["pipeline_stage"])
    align = int(kernel_info["align"]) if kernel_info["align"] is not None else -1

    tensor_core_or_not = (kernel_info["tensor_core_or_not"] == "tensorop")
    source_core_num = query_core_num(source_gpu, dtype, tensor_core_or_not)
    target_core_num = query_core_num(target_gpu, dtype, tensor_core_or_not)

    if op_type == "Conv2D":

        GEMM_M, GEMM_N, GEMM_K = eq_gemm_size(raw_feature)

    elif op_type == "MatMul" and kernel is not None and \
            source_gpu is not None and target_gpu is not None:

        GEMM_M, GEMM_K, GEMM_N = raw_feature[5:8]
        GEMM_M *= raw_feature[8]
        
    new_feature = new_feature + [
        source_sm_num,
        target_sm_num,
        source_bw,
        target_bw,
        source_clock,
        target_clock,
        source_core_num,
        target_core_num,
        Mtile, Mtile, Ktile,
        pipeline_stage, align,
        GEMM_M, GEMM_N, GEMM_K]
    
    B = int(np.ceil(GEMM_M/Mtile) * np.ceil(GEMM_N/Mtile))
    W = int(np.ceil(B/target_sm_num))
    pipeline_num = W * np.ceil(GEMM_K/Ktile)
    tile_size_in = Mtile * Ktile + Ktile * Mtile
    tile_size_out = Mtile * Mtile
    tile_flop = Mtile * Ktile * Mtile

    new_feature = new_feature + [
        B, # number of thread blocks of a kernel
        W, # number of waves
        pipeline_num,   # pipeline stage number
        tile_size_in,   # input size for each tile
        tile_size_out,  # output size for each tile
        tile_flop       # the number of float operations per tile
    ]
        
    return new_feature + raw_feature[1:]

# (n_sample, n_dim) /cdot (n_dim, 1) = (n_dim, 1)
    # data on current gpu       matrix      data on the target gpu

def _split_train_test_data(X, Y):
    n_samples = X.shape[0]
    TRAIN_PERCENT = 0.9
    mask = np.zeros(n_samples, dtype=bool)
    train_idx = np.random.choice(n_samples, math.ceil(TRAIN_PERCENT * n_samples), replace=False)
    mask[train_idx] = True

    # print(kernel, X.shape, Y.shape)

    trainX, trainY = X[mask, :], Y[mask]
    testX, testY = X[~mask, :], Y[~mask]
    if len(testX) == 0:
        testX, testY = trainX, trainY
    return trainX, trainY, testX, testY

def _customize_feature_data(gpu1, gpu2,
        raw_features,
        ave_lower_bound,
        feature_dict,
        target_op_type,
        kernel=None):
    gpu = feature_decode("gpu_model",
        raw_feature_index(raw_features, DIMENSION_NAME.gpu_model, target_op_type))
    if gpu != gpu1 and gpu != gpu2:
        return
    
    # ave_lower_bound = 0.1
    if ave_lower_bound and raw_features[0] < ave_lower_bound:
        return

    feature_id = ",".join([str(v) for v in raw_features[2:]])
    if feature_id not in feature_dict:
        feature_dict[feature_id] = {}
    # assert len(raw_features) == len(header), (raw_features, header)
    if gpu1 == gpu2:
        feature_dict[feature_id]["X"] = enrich_raw_feature(raw_features, target_op_type,
            ave=1,
            source_gpu=gpu1,
            target_gpu=gpu2,
            kernel=kernel)

        # from utils.op_info import parse_raw_feature
        # print(raw_features)
        # print(feature_dict[feature_id]["X"])
        # print(parse_raw_feature(feature_dict[feature_id]["X"]))
        # assert parse_raw_feature(feature_dict[feature_id]["X"])[1:] == raw_features[1:]
        # raise

        feature_dict[feature_id]["Y"] = raw_features[0]
    else:
        if (gpu == gpu1):
            feature_dict[feature_id]["X"] = enrich_raw_feature(raw_features, target_op_type,
                source_gpu=gpu1,
                target_gpu=gpu2,
                kernel=kernel)
        if (gpu == gpu2):
            feature_dict[feature_id]["Y"] = raw_features[0]

def _record_xydata_upper_bound(X, Y, cus_feature_data, scaler, new_header):
    if "X" in cus_feature_data and "Y" in cus_feature_data:
        ### Record the maximum value
        scaler.record_dims(new_header, cus_feature_data["X"])
        scaler.record(DIMENSION_NAME.ave, cus_feature_data["Y"])

        ### Append to training/test data list
        X.append(cus_feature_data["X"])
        Y.append(cus_feature_data["Y"])

def collect_data_two_gpu(gpu1, gpu2,
    op2xydata, 
    ave_lower_bound=None):
    
    if gpu1 == gpu2:
        print("[Warning] source and target device are the same: {}".format(gpu1))
    
    assert len(op2xydata) == 1, "Currently only one op type is support, but {} is given: {}".format(
        len(op2xydata), op2xydata.keys())

    optype2feature = {}
    for op_type, _data in op2xydata.items():
        if op_type not in optype2feature:
            optype2feature[op_type] = {}
        feature_dict = optype2feature[op_type]
        # header = FULL_HEADERS[op_type]
        for features in _data:
            _customize_feature_data(gpu1, gpu2,
                features,
                ave_lower_bound,
                feature_dict,
                op_type)

    xy_by_op_type = {}
    scaler = Scaler()
    for op_type, feature_dict in optype2feature.items():
        new_header = enriched_header(op_type)
        for cus_feature_data in feature_dict.values():
            if "X" in cus_feature_data and "Y" in cus_feature_data:
                if op_type not in xy_by_op_type:
                        xy_by_op_type[op_type] = {"X": [], "Y": []}
                _record_xydata_upper_bound(
                    xy_by_op_type[op_type]["X"],
                    xy_by_op_type[op_type]["Y"],
                    cus_feature_data,
                    scaler,
                    new_header)
    
    ### Generate training and test data
    for op_type in xy_by_op_type.keys():
        X = np.array(xy_by_op_type[op_type]["X"])
        Y = np.array(xy_by_op_type[op_type]["Y"])
        xy_by_op_type[op_type] = _split_train_test_data(X, Y)

    return xy_by_op_type, scaler

def fine_grain_kernel(kernel, ave, cus_feature, target_op_type, isRawFeature=False):
    _feature_index_func = raw_feature_index if isRawFeature else enriched_feature_index
    kernel_id_list = [kernel]
    op_scale = ">=1ms" if ave >= 1 else "<1ms"
    if target_op_type == "Conv2D":
        H = _feature_index_func(cus_feature, "H", target_op_type)
        P = _feature_index_func(cus_feature, "P", target_op_type)
        C = _feature_index_func(cus_feature, "C", target_op_type)
        K = _feature_index_func(cus_feature, "K", target_op_type)
        R = _feature_index_func(cus_feature, "R", target_op_type)
        S = _feature_index_func(cus_feature, "S", target_op_type)
        stride = _feature_index_func(cus_feature, "stride", target_op_type)

        # _kernel = "{}_c_tile={}_k_tile={}".format(kernel, C, K)
        # _kernel = "{}_R={}_S={}_stride={}".format(kernel, R, S, stride)
        # _kernel = "{}_c_tile={}_k_tile={}_R={}_S={}".format(kernel, C, K, R, S)
        kernel_id_list.append("H={}".format(H))
        kernel_id_list.append("scale{}".format(op_scale))
        kernel_id_list.append("R={}".format(R))
    else:
        # kernel_id_list.append("scale{}".format(op_scale))
        pass

    return "_".join(kernel_id_list)

def collect_data_two_gpu_cutlass(gpu1, gpu2,
    kernel_data_dict,
    target_op_type,
    target_kernel_type=None,
    ave_lower_bound=None):
    
    if gpu1 == gpu2:
        print("[Warning] source and target device are the same: {}".format(gpu1))
    
    kernel2feature = {}
    for kernel, _data in kernel_data_dict.items():
        if kernel not in kernel2feature:
            kernel2feature[kernel] = {}
        feature_dict = kernel2feature[kernel]
        # print(kernel)
        for features in _data:
            op_type = feature_decode(DIMENSION_NAME.op_type,
                raw_feature_index(features, DIMENSION_NAME.op_type, target_op_type))
            if op_type != target_op_type:
                continue
            if target_op_type == "Conv2D":
                conv_kind = feature_decode("conv_kind",
                    raw_feature_index(features, "conv_kind", target_op_type))
                if target_kernel_type and not target_kernel_type.endswith(conv_kind):
                    continue
            _customize_feature_data(gpu1, gpu2,
                features,
                ave_lower_bound,
                feature_dict,
                target_op_type, kernel=kernel)

    # print(len(list(kernel_data_dict.keys())), len(feature_dict))
    xy_by_kernel = {}
    scaler = Scaler()
    new_header = enriched_header(target_op_type)
    for kernel, feature_dict in kernel2feature.items():
        for cus_feature_data in feature_dict.values():
            if "X" in cus_feature_data and "Y" in cus_feature_data:
                _kernel = fine_grain_kernel(kernel, cus_feature_data["Y"], cus_feature_data["X"], target_op_type)
                if _kernel not in xy_by_kernel:
                    xy_by_kernel[_kernel] = {"X": [], "Y": []}
                _record_xydata_upper_bound(
                    xy_by_kernel[_kernel]["X"],
                    xy_by_kernel[_kernel]["Y"],
                    cus_feature_data,
                    scaler,
                    new_header)

    # print(len(xy_by_kernel))
    ### Generate training and test data
    for kernel in xy_by_kernel.keys():
        X = np.array(xy_by_kernel[kernel]["X"])
        Y = np.array(xy_by_kernel[kernel]["Y"])
        xy_by_kernel[kernel] = _split_train_test_data(X, Y)

    return xy_by_kernel, scaler

def group_data(xydata, group_dims, index_only=False):
    ''' list, a list of dimensions, we will divide data into several groups
        according to the values
    Parameters
    -----------
    index_only: if set true, this function returns a dict, where the keys are still
        the group id, the value is a mapping telling which indexes of features are
        kept for each op type, e.g.
        {
            "model_0": {
                "Conv2D": [False, True, True, ....]
            }
        }
    '''
    ret = {}
    for op_type, data in xydata.items():
        header = FULL_HEADERS[op_type]
        for idx, feature in enumerate(data):
            grp_id = "/".join(["{}={}".format(dim, feature_decode(dim, feature[header.index(dim)]))
                              for dim in group_dims if dim in header])
            if grp_id not in ret:
                if index_only:
                    ret[grp_id] = {}
                    for op_type in xydata.keys():
                        ret[grp_id][op_type] = np.zeros(xydata[op_type].shape[0], dtype=bool)
                else:  
                    ret[grp_id] = []
            if index_only:
                ret[grp_id][op_type][idx] = True
            else:
                ret[grp_id].append(feature)
    return ret
             
