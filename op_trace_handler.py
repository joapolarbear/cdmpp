import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from utils.util import DIMENSION_NAME
from utils.util import Setting, Filters, PROJECT_DIR
from utils.util import gen_feature_id, axis_label
from utils.op_info import FULL_HEADERS, BASE_HEADER_LEN
from utils.op_info import feature_decode, raw_feature_index, kernel_type2op_type
from utils.env import PROJECT_CFG
from utils.device_info import DEVICE_INFO, short_device_name

from dataloader import collect_data, group_data
from analytical.cost_model.cost_model import OP_STAND_ALONE_CM

from profiler.profile_op.common import ConfigGenThread
from profiler.profile_op.ops import op_search_space

parser = argparse.ArgumentParser(prog="OP trace handler")
parser.add_argument('--force_load_data', action='store_true', help="If specified, force to load raw data")
parser.add_argument('--force_fit', action='store_true', help="If specified, force to fit the cm")
parser.add_argument('--check', action='store_true', help="If specified, debug the cost model")
parser.add_argument('--compare', action='store_true', help="If specified, check the slow"
    " down when an op runs in a DNN, compared to runs in a standalone manner.")
parser.add_argument('--gen_cfg_dir', type=str, default=None, help="If not None, generate op"
    "configs under the specifed path for profiling each op individually")
# parser.add_argument('--kernel', type=str, default=None, help="specify the kernel to fit")
args = parser.parse_args()

SHOW_ROOFLINE = False

### Configuration
trace_root_path = os.path.join(PROJECT_CFG["source_data_root"], "op_level_trace")
source_gpu = short_device_name(PROJECT_CFG.get("source_gpu", "Tesla_V100-SXM2-32GB"))
target_gpu = short_device_name(PROJECT_CFG.get("source_gpu", "Tesla_V100-SXM2-32GB"))
AVE_LOWER_BOUND_MS = float(PROJECT_CFG.get("AVE_LOWER_BOUND_MS", 0.1))
target_model = PROJECT_CFG.get("target_model", "ResNet50")
target_kernel_type = PROJECT_CFG.get("target_kernel_type", "gemm")
target_op_type = kernel_type2op_type(target_kernel_type)
target_dtype = PROJECT_CFG.get("dtype", "fp32")

filters_pre_process = Filters({
    DIMENSION_NAME.op_type: [target_op_type],
    DIMENSION_NAME.dtype: [target_dtype],
    DIMENSION_NAME.gpu_model: [source_gpu, target_gpu],
    DIMENSION_NAME.model: [target_model]
})

filters = Filters({
    DIMENSION_NAME.op_type: [target_op_type],
    DIMENSION_NAME.dtype: [target_dtype],
    DIMENSION_NAME.gpu_model: [source_gpu, target_gpu],
    # "R": [1],  # filter size
    # "H": [14], # input shape
    # "P": [14], # output shape
    # "C": [1024], # input channel
    # "K": [256], # output channel
    # "bs": [4]
})

font = {"color": "darkred",
        "size": 16,
        "family": "serif"}

def dim_reduction_comp(xydata):
    xdata = []
    ydata = []
    gpu_model = []
    for op_type, _data in xydata.items():
        for features in _data:
            xdata.append([features[0]] + features[2:])
            ydata.append(features[0])
            gpu_model.append(features[1])

    from sklearn.manifold import TSNE
    _data = np.array(xdata)
    print(_data.shape)
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(_data)
    print(X_tsne.shape)
    
    plt.style.use("dark_background")
    plt.figure(figsize=(8.5, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=ydata, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title("t-SNE", fontdict=font)
    cbar = plt.colorbar()
    cbar.set_label(label='Ave', fontdict=font)
    plt.clim(0, 5)
    plt.subplot(1, 2, 2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=gpu_model, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title("t-SNE", fontdict=font)
    cbar = plt.colorbar()
    cbar.set_label(label='GPU model', fontdict=font)
    plt.tight_layout()
    plt.savefig("_fig/dim_reduct.png")

def plot_scatter_2d(xydata, xaxis_name, yaxis_name, group_dims):
    ''' 
    group_dims: list, a list of dimensions, we will divide data into several groups
        according to the values of those dimensions and plot the scatter respectively
    '''
    data_in_group = group_data(xydata, group_dims)
    op_type = list(xydata.keys())[0]
    fig = plt.figure(figsize=(12, 6))
    for grp_id in sorted(data_in_group.keys()):
        grp_data = np.array(data_in_group[grp_id]).T
        xdata = raw_feature_index(grp_data, xaxis_name, op_type)
        ydata = raw_feature_index(grp_data, yaxis_name, op_type)
        plt.scatter(xdata, ydata, label=grp_id)
    
    if SHOW_ROOFLINE and xaxis_name == "ai" and yaxis_name == "perf" and "gpu_model" in group_dims:
        for gpu_model in DEVICE_INFO.keys():
            if filters.not_in_filter(DIMENSION_NAME.gpu_model, gpu_model):
                continue
            if "BW_GB_per_s" in DEVICE_INFO[gpu_model]:
                plt.plot(xdata, DEVICE_INFO[gpu_model]["BW_GB_per_s"] * 1e9 * xdata, label="{}: Bx".format(gpu_model))
            if filters.not_in_filter(DIMENSION_NAME.dtype, "fp16"):
                if "fp16_tflops" in DEVICE_INFO[gpu_model]:
                    plt.plot(
                        xdata, DEVICE_INFO[gpu_model]["fp16_tflops"] * 1e12 * np.ones_like(xdata), label="{} fp16: P".format(gpu_model))
            if not filters.not_in_filter(DIMENSION_NAME.dtype, "fp32"):
                if "fp32_tflops" in DEVICE_INFO[gpu_model]:
                    plt.plot(
                        xdata, DEVICE_INFO[gpu_model]["fp32_tflops"] * 1e12 * np.ones_like(xdata), label="{} fp32: P".format(gpu_model))
    # elif SHOW_ROOFLINE and yaxis_name == "ave"

    _text = ("filters={}".format(filters.serialize_filter(feature_decode=True)))
    plt.text(50, 0, _text, wrap=True, alpha=0.6)
    plt.xlabel(axis_label(xaxis_name), fontsize=16)
    plt.ylabel(axis_label(yaxis_name), fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "_fig/op_stand_alone/{}_mape.png".format(setting)))

def compare_inDnn2standalone(dnn_trace_path, per_op_trace_path):
    ''' Compare the performance of an operator when it is profiled
        in a DNN model or standalone
    '''
    dnn_op2data = filters.apply_filters_to_op(collect_data(dnn_trace_path, dnn_trace_path, filters_pre_process, force=args.force_load_data))
    per_op_op2data = filters.apply_filters_to_op(collect_data(per_op_trace_path, per_op_trace_path, filters_pre_process, force=args.force_load_data))
    name0 = "In a Dnn"
    name1 = "Standalone"

    print(len(dnn_op2data[target_op_type]))
    print(len(per_op_op2data[target_op_type]))

    _dict = {}
    for op_type, data in dnn_op2data.items():
        header = FULL_HEADERS[op_type]
        for idx, feature in enumerate(data):
            feature_id = gen_feature_id(feature)
            if feature_id not in _dict:
                _dict[feature_id] = {}
            assert len(header) - len(feature) <=2, (header, feature, len(header), len(feature))
            feature[3] = 0
            _dict[feature_id][name0] = feature
            print(feature_id)
    
    _dict2 = set()
    for op_type, data in per_op_op2data.items():
        header = FULL_HEADERS[op_type]
        for idx, feature in enumerate(data):
            feature[3] = 0
            feature_id = gen_feature_id(feature)
            _dict2.add(feature_id)
            if feature_id not in _dict:
                # print(len(feature[:14]))
                # print(feature_id)
                # raise
                continue
            assert len(header) - len(feature) <=2, (header, feature, len(header), len(feature))
            _dict[feature_id][name1] = feature[0]
    
    X, Y = [], []
    for _data in _dict.values():
        if name0 in _data and name1 in _data:
            X.append(_data[name0])
            Y.append(_data[name1])
    X, Y = np.array(X), np.array(Y)
    print("X.shape={}, Y.shape={}".format(X.shape, Y.shape))
    # print(list(_dict.keys()))
    # print(_dict2)
    
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121)
    third_dim = "flops"
    cbar_value = raw_feature_index(X.T, third_dim, target_op_type)
    ave_X = raw_feature_index(X.T, "ave", target_op_type)
    plt.scatter(ave_X, Y, c=cbar_value, alpha=0.5, edgecolors='none',
                cmap=plt.cm.get_cmap('rainbow', 100))
    plt.plot(ave_X, ave_X, label="y=x")
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.5)
    plt.xlabel("Ave of {} (ms)".format(name0), fontsize=font["size"])
    plt.ylabel("Ave of {} (ms)".format(name1), fontsize=font["size"])
    plt.legend(fontsize=font["size"])
    cbar = plt.colorbar()
    cbar.set_label(label=axis_label(third_dim), fontsize=font["size"])

    error = np.average((Y - ave_X) / ave_X)
    ax = fig.add_subplot(122)
    plt.scatter(cbar_value, ave_X, label=name0)
    plt.scatter(cbar_value, Y, label=name1)
    plt.xlabel(axis_label(third_dim), fontsize=font["size"])
    plt.ylabel("Real Execution Time (ms)", fontsize=font["size"])
    plt.text(1e10, 8, "MPE Error: {:.3f} %".format(error * 100))
    plt.legend(fontsize=font["size"])
    plt.tight_layout()
    plt.savefig("_fig/inDnnVSStandAlone.png")

def gen_cfg_from_dnn_trace(dnn_trace_path):
    ### We only focus ResNet50 first
    _filters = Filters({
        DIMENSION_NAME.model: ["ResNet50"],
        DIMENSION_NAME.op_type: ["Conv2D", "MatMul"]})
    dnn_op2data = collect_data(dnn_trace_path, dnn_trace_path, _filters, force=args.force_load_data)
    for op_type, raw_features in dnn_op2data.items():
        op_cfg_path = os.path.join(args.gen_cfg_dir, "{}_cfgs.txt".format(op_type))
        range_dict, op_fn, input_shape_fn,\
            feature_fn, check_valid_fn = op_search_space(op_type)
        config_t = ConfigGenThread(range_dict, check_valid_fn=check_valid_fn)

        header = FULL_HEADERS[op_type]
        dtype_idx = header.index(DIMENSION_NAME.dtype)
        bs_idx = header.index(DIMENSION_NAME.bs)

        for idx, raw_feature in enumerate(raw_features):
            config = {}
            for idx in range(len(raw_feature)):
                if idx == dtype_idx:
                    config["dtype"] = feature_decode(DIMENSION_NAME.dtype, raw_feature[idx])
                elif idx == bs_idx:
                    config["N"] = raw_feature[idx]
                elif idx >= BASE_HEADER_LEN:
                    config[header[idx]] = raw_feature[idx]
            if op_type == "Conv2D":
                config["stride"] = round(config["H"] / config["P"])
                padding = round(((config["P"] - 1) * config["stride"] + config["R"] - config["H"]) / 2)
                if padding == 0:
                    config["padding"] = "valid"
                elif padding == config["R"] // 2:
                    config["padding"] = "same"
                else:
                    raise ValueError(padding)
            config_t.add_cfg(config)
        config_t.dump(cache_path=op_cfg_path)


# root_dir = "{}/dnn_traces".format(trace_root_path)
root_dir = "{}/op_database".format(trace_root_path)

if __name__ == "__main__":
    if args.compare:
        if args.gen_cfg_dir is not None:
            gen_cfg_from_dnn_trace(dnn_trace_path="{}/dnn_traces".format(trace_root_path))
        else:
            per_op_trace_dir = os.path.join(PROJECT_CFG["source_data_root"], "contention")
            # per_op_trace_dir = "{}/op_database".format(trace_root_path)
            compare_inDnn2standalone("{}/dnn_traces".format(trace_root_path),
                                per_op_trace_dir)

                                
        exit(0)

    op2xydata = collect_data(root_dir, root_dir, filters_pre_process, force=args.force_load_data)
    
    ### Use two filters to avoid redundant cache
    # If we only use one filter for both rawdata loading and filtering
    # it will maintain a cache for each version of the filter
    # filters.check_filters()
    # op2xydata = filters.apply_filters_to_op(op2xydata)

    ### Train and evaluate the cost model
    setting = Setting(source_gpu, target_gpu,
        ave_lower_bound_ms=AVE_LOWER_BOUND_MS,
        target_op_type=target_op_type,
        target_dtype=target_dtype)
    op_stand_alone_cm = OP_STAND_ALONE_CM()
    mape_rst_path = os.path.join(PROJECT_DIR, "_cache/op_stand_alone_mape/{}_mape.pickle".format(setting))
    mape_list, method_list = op_stand_alone_cm.train_evaluate_all_ops(
        op2xydata,
        setting,
        mape_rst_path,
        force_fit=args.force_fit,
        check_one_op=args.check)

    # dim_reduction_comp(op2xydata)
    plot_scatter_2d(op2xydata, 
        xaxis_name = "ai",
        yaxis_name = "perf", 
        group_dims=["gpu_model", 
                    # "dtype",
                    # "C", 
                    # "K",
                    # "C_in", "C_out"
                    ])

