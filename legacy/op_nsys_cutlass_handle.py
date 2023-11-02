import os
import numpy as np
import matplotlib.pyplot as plt
import re

from dataloader import gen_raw_feature
from utils.util import PROJECT_DIR, axis_label
from utils.util import line2dict, line2list
from utils.device_info import short_device_name
from utils.op_info import raw_feature_index

fontsize = 18
fontdict = {"color": "darkred",
        "size": fontsize,
        "family": "serif"}

def compare_op_nsys_cutlass(gpu_dir, gpu_model, target_op_type, axis_dim):
    with open(os.path.join(gpu_dir, "{}_op.txt".format(target_op_type)), 'r') as fp:
        op_trace = fp.readlines()

    with open(os.path.join(gpu_dir, "{}_nsys.txt".format(target_op_type)), 'r') as fp:
        nsys_trace = fp.readlines()

    with open(os.path.join(gpu_dir, "{}_cutlass.txt".format(target_op_type)), 'r') as fp:
        cutlass_trace = fp.readlines()

    assert len(op_trace) == len(nsys_trace), (len(op_trace), len(nsys_trace))
    # assert len(op_trace) == len(cutlass_trace)

    raw_op_data = []
    nsys_data = []
    nsys_kernels = []
    cutlass_data = []
    cutlass_kernels = []
    for idx in range(len(op_trace)):
        if len(nsys_trace[idx]) < 4:
            continue

        ### OP
        all_raw_data = line2list(op_trace[idx])
        for per_op_raw_data in all_raw_data:
            try:
                if per_op_raw_data[2] == target_op_type:
                    # print(per_op_raw_data)
                    gen_raw_feature(raw_op_data, per_op_raw_data, gpu_model)
            except:
                print(per_op_raw_data)
                print(op_trace[idx])
                raise
        
        ### Nsys
        # print(idx, nsys_trace[idx], len(nsys_trace[idx]))
        nsys_feature_dict = line2dict(nsys_trace[idx])
        nsys_data.append(float(nsys_feature_dict['Average']) / (1000.**2))
        nsys_kernels.append(nsys_feature_dict['Operation'])

        ### CUTLASS
        features = cutlass_trace[idx].split(",")
        cutlass_data.append(float(features[3]))
        cutlass_kernels.append(features[0])

    raw_op_data = np.array(raw_op_data)
    nsys_data = np.array(nsys_data)
    cutlass_data = np.array(cutlass_data)
    axis_data = raw_feature_index(
        raw_op_data.T, axis_dim, op_type=target_op_type)

    print(raw_op_data.shape, nsys_data.shape, cutlass_data.shape, axis_data.shape)
    log_str = "# of Nsys kernels={}, # of CUTLASS kernels={}".format(
        len(set(nsys_kernels)), len(set(cutlass_kernels)))
    print(log_str)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.scatter(axis_data, raw_op_data.T[0], s=300, alpha=0.3, label="Op-level")
    ax.scatter(axis_data, nsys_data, label="Nsys-kernel-level")
    
    # ax.scatter(axis_data, cutlass_data, label="CUTLASS-kernel-level")

    cutlass_kernel_dict = {}
    for cutlass_kernel in set(cutlass_kernels):
        cutlass_kernel_dict[cutlass_kernel] = np.array([False] * len(cutlass_kernels))
    for idx, _kernel in enumerate(cutlass_kernels):
        cutlass_kernel_dict[_kernel][idx] = True
    log_list = []
    for kernel_id, cutlass_kernel in enumerate(set(cutlass_kernels)):
        cutlass_kernel_dict[cutlass_kernel]
        ax.scatter(
            axis_data[cutlass_kernel_dict[cutlass_kernel]],
            cutlass_data[cutlass_kernel_dict[cutlass_kernel]],
            label="CUTLASS-kernel_{}-level".format(kernel_id))
        log_list.append(f'{kernel_id}:{cutlass_kernel}')
    ax.text(4e9, 0.6, "\n".join(log_list), fontsize=12)
    
    ax.text(1e9, 3, log_str, fontdict=fontdict)
    plt.xlabel(axis_label(axis_dim), fontsize=fontsize)
    plt.ylabel("Real Tested Time (ms)", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.title("Real Tested Time on {} for {}".format(gpu_model, target_op_type), fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(
        PROJECT_DIR, "_fig/op_nsys_cutlass/{}_{}_op_nsys_cutlass.png".format(gpu_model, target_op_type)))


if __name__ == "__main__":
    # _gpu_model = "Tesla_V100-SXM2-32GB"
    # root_dir = "/home/tiger/ws/CrossDevicePredict/op_nsys_cutlass"

    _gpu_model = "Tesla_T4"
    root_dir = "/home/tiger/op_nsys_cutlass"

    gpu_model = short_device_name(_gpu_model)
    gpu_dir = os.path.join(root_dir, _gpu_model)
    target_op_type = "MatMul"
    axis_dim = "flops"
    compare_op_nsys_cutlass(gpu_dir, gpu_model, target_op_type, axis_dim)