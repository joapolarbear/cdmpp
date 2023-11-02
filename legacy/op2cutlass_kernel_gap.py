import os, sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.base import DIMENSION_NAME
from utils.util import Filters, Setting, axis_label
from utils.util import PROJECT_DIR, FULL_HEADERS
from utils.env import PROJECT_CFG
from utils.op_info import kernel_type2op_type, raw_feature_index
from utils.device_info import short_device_name

from dataloader import collect_data
from analytical.cost_model.cutlass_cm import CUTLASS_CM, find_evaluate_cutlass_kernel_iplmt

parser = argparse.ArgumentParser(prog="Test")
parser.add_argument('--force_load_data', action='store_true', help="force to load raw data")
parser.add_argument('--force_pred', action='store_true', help="force to inference")
parser.add_argument('--verbose', action='store_true', help="show CUTLASS commands and raw outputs")
args = parser.parse_args()

target_kernel_type = PROJECT_CFG.get("target_kernel_type", "gemm")
target_op_type = kernel_type2op_type(target_kernel_type)
source_gpu = short_device_name(PROJECT_CFG.get("source_gpu", "Tesla_V100-SXM2-32GB"))
target_gpu = short_device_name(PROJECT_CFG.get("source_gpu", "Tesla_V100-SXM2-32GB"))
AVE_LOWER_BOUND_MS = float(PROJECT_CFG.get("AVE_LOWER_BOUND_MS", 0.1))
trace_root_path = os.path.join(PROJECT_CFG["source_data_root"], "op_level_trace")
target_dtype = PROJECT_CFG.get("dtype", "fp32")
setting = Setting(
    source_gpu,
    target_gpu,
    None,
    AVE_LOWER_BOUND_MS,
    None,
    target_op_type=target_op_type,
    target_dtype=target_dtype)

filters = Filters({
    DIMENSION_NAME.op_type: [target_op_type],
    DIMENSION_NAME.dtype: [target_dtype],
    DIMENSION_NAME.gpu_model: [source_gpu, target_gpu],
})

STR2OP = {
    "/": int.__truediv__,
    "//": int.__floordiv__,
    "*": int.__mul__,
    "+": int.__add__,
}

def find_evaluate_cutlass_kernel(optype2xydata):
    ### Use the CUTLASS API to find the corresponding kernel
    # then use the CUTLASS cost model to predict the execution time of this kernel
    mapping_rst_path = os.path.join(PROJECT_DIR, "_cache/op2cutlass_{}.pickle".format(setting))
    if os.path.exists(mapping_rst_path) and not args.force_pred:
        with open(mapping_rst_path, 'rb') as fp:
            testY_list, predY_list, error_list, kernel_list, raw_feature_list = pickle.load(fp)
    else:
        header = FULL_HEADERS[target_op_type]
        cutlass_cm = CUTLASS_CM()

        testY_list = []
        predY_list = []
        error_list = []
        kernel_list = []
        raw_feature_list = []

        for feature in tqdm(optype2xydata[target_op_type]):
            if feature[0] < AVE_LOWER_BOUND_MS:
                continue
        
            kernel, predY, error = find_evaluate_cutlass_kernel_iplmt(
                cutlass_cm,
                feature,
                target_kernel_type,
                source_gpu,
                target_gpu,
                header,
                verbose=args.verbose)

            testY_list.append(feature[0])
            predY_list.append(predY)
            error_list.append(error)
            kernel_list.append(kernel)
            raw_feature_list.append(feature)

        with open(mapping_rst_path, 'wb') as fp:
            pickle.dump([testY_list, predY_list, error_list, kernel_list, raw_feature_list], fp)
    return testY_list, predY_list, error_list, kernel_list, raw_feature_list

def _grp_id_per_dim(raw_feature, dim, op_type=None):
    if isinstance(dim, str):
        return "{}={}".format(dim, raw_feature_index(raw_feature, dim, op_type=op_type))
    else:
        _dim, op, second = dim
        if isinstance(second, str):
            second_value = raw_feature_index(raw_feature, second, op_type=op_type)
        else:
            second_value = second
        _dim_value = raw_feature_index(raw_feature, _dim, op_type=op_type)
        return "{}{}{}={}".format(_dim, op, second, STR2OP[op](int(_dim_value), int(second_value)))
        
def _gen_group_id(kernel, raw_feature, group_dims, op_type):
    spliters = [_grp_id_per_dim(raw_feature, dim, op_type=op_type)
                              for dim in group_dims if dim != "kernel"]
    if "kernel" in group_dims:
        spliters.append("{}={}".format("kernel", kernel))
    return "-".join(spliters)

def group_data(kernel_list, raw_feature_list, testY_list, predY_list, error_list, group_dims, cbar_dim=None):
    grouped_data = {}
    group_ids = []
    for idx, kernel in enumerate(kernel_list):
        if cbar_dim is not None:
            grp_id = "default"
        else:
            grp_id = _gen_group_id(kernel, raw_feature_list[idx], group_dims, target_op_type)
        if grp_id not in grouped_data:
            grouped_data[grp_id] = [[], [], [], []]
            group_ids.append(grp_id)
        grouped_data[grp_id][0].append(testY_list[idx])
        grouped_data[grp_id][1].append(predY_list[idx])
        grouped_data[grp_id][2].append(error_list[idx])
        if cbar_dim is not None:
            grouped_data[grp_id][3].append(
                raw_feature_index(
                    raw_feature_list[idx],
                    cbar_dim,
                    op_type=target_op_type))
    return grouped_data, group_ids

def visual_rst(testY_list, kernel_list, grouped_data, group_ids, cbar_dim=None):
    ### Visualize the predicted kernel time and compare it with the op time
    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(221)
    ax.plot(testY_list, testY_list, "--", label="y=x")

    from scipy.optimize import curve_fit
    def func(x, a, b):
        return a * x + b

    if cbar_dim is None:
        for idx, grp_id in enumerate(group_ids):
            ax.scatter(grouped_data[grp_id][0], grouped_data[grp_id][1], label="{}-{}".format(idx, grp_id))
    else:
        for idx, grp_id in enumerate(group_ids):
            plt.scatter(
                grouped_data[grp_id][0],
                grouped_data[grp_id][1],
                c=grouped_data[grp_id][3],
                alpha=0.5,
                edgecolors='none',
                cmap=plt.cm.get_cmap('rainbow', 100))
            
            popt, pcov = curve_fit(func, grouped_data[grp_id][1], grouped_data[grp_id][0])
            print(popt)
            
        cbar = plt.colorbar()
        cbar.set_label(label=axis_label(cbar_dim), fontsize=16)

    plt.xlabel("Real Time (ms)", fontsize=16)
    plt.ylabel("Predicted Time (ms)", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=12)

    plt.title("Op2CUTLASS {} kernels ".format(len(set(kernel_list))) + str(setting), fontsize=16)

    ax = fig.add_subplot(222)
    for idx, grp_id in enumerate(group_ids):
        ax.scatter(grouped_data[grp_id][3], grouped_data[grp_id][0], label="{}-{}".format(idx, grp_id))
    plt.xlabel("{}, op-level".format(axis_label(cbar_dim)), fontsize=16)
    plt.ylabel(axis_label("ave"), fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.title("Op2CUTLASS {} kernels ".format(len(set(kernel_list))) + str(setting), fontsize=16)


    ax2 = fig.add_subplot(223)
    error_list_by_grp_id = []
    for grp_id in group_ids:
        error_list_by_grp_id.append(grouped_data[grp_id][2])
    ax2.boxplot(error_list_by_grp_id[::-1], 
        labels=["kernel-{}".format(idx) for idx, _ in enumerate(group_ids)][::-1], vert=False, showmeans=True)
    ref_line = [10] * 6
    ax2.plot(ref_line, list(range(len(ref_line))), '--', label="10%")
    plt.xlabel("Error (%)", fontsize=16)
    plt.xlim(0, 200)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=16)

    if cbar_dim is not None:
        ax = fig.add_subplot(224)
        # ax.plot(testY_list, testY_list, "--", label="y=x")
        for idx, grp_id in enumerate(group_ids):
            plt.scatter(grouped_data[grp_id][3], grouped_data[grp_id][0], label="Real Op Time (ms)")
            plt.scatter(grouped_data[grp_id][3], grouped_data[grp_id][1], label="Predicted CUTLASS-Kernel Time (ms)")
        
            plt.text(1e10, 1-0.5*idx, "{}: {} nodes".format(grp_id, len(grouped_data[grp_id][3])))

        plt.xlabel(axis_label(cbar_dim), fontsize=16)
        plt.ylabel("Time (ms)", fontsize=16)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.title("Op2CUTLASS {} kernels ".format(len(set(kernel_list))) + str(setting), fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "_fig/op2cutlass/op2cutlass_{}.png".format(setting)))

if __name__ == "__main__":
    ### Collect Op-level trace which is profiled in a stand-alone mode
    # root_dir = "{}/dnn_traces".format(trace_root_path) # in dnn
    root_dir = "{}/op_database".format(trace_root_path) # stand-alone
    optype2xydata = collect_data(root_dir, root_dir, filters, force=args.force_load_data, per_dnn_trace_hook=None)

    group_dims = [
        # "kernel",
        # DIMENSION_NAME.dtype,
        # DIMENSION_NAME.bs,
        # "C_in",
        # "C_out",
        # "flops",
        # "size",
        # ("ai", "//", 1*32),
        # ("C_out", "//", "bs"),
    ]

    cbar_dim = None
    cbar_dim = "flops"

    testY_list, predY_list, error_list, kernel_list, raw_feature_list = find_evaluate_cutlass_kernel(optype2xydata)

    grouped_data, group_ids = group_data(kernel_list, raw_feature_list, testY_list, predY_list, error_list, group_dims, cbar_dim=cbar_dim)

    visual_rst(testY_list, kernel_list, grouped_data, group_ids, cbar_dim=cbar_dim)