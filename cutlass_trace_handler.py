import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

from utils.util import DIMENSION_NAME
from utils.util import Setting, PROJECT_DIR, Filters
from utils.device_info import short_device_name
from utils.env import PROJECT_CFG
from dataloader import collect_data
from analytical.cost_model.cutlass_cm import CUTLASS_CM
from analytical.cost_model.mlp import DNNPredictor


parser = argparse.ArgumentParser(prog="CUTLASS data handler")
parser.add_argument('--force_load_data', action='store_true', help="force to load raw data")
parser.add_argument('--force_fit', action='store_true', help="force to fit the cm")
parser.add_argument('--check', action='store_true', help="check the cm")
parser.add_argument('--kernel', type=str, default=None, help="specify the kernel to fit")
parser.add_argument('--similarity_check', action='store_true', help="check the similarity")
args = parser.parse_args()

if args.kernel:
    debug_kernel_list = args.kernel.split(',')
else:
    debug_kernel_list = None


### Configuration
trace_root_path = os.path.join(PROJECT_CFG["source_data_root"], "cutlass_test")
source_gpu = short_device_name(PROJECT_CFG.get("source_gpu", "Tesla_V100-SXM2-32GB"))
target_gpu = short_device_name(PROJECT_CFG.get("target_gpu", "Tesla_V100-SXM2-32GB"))
target_kernel_type = PROJECT_CFG.get("target_kernel_type", "gemm")
AVE_LOWER_BOUND_MS = float(PROJECT_CFG.get("AVE_LOWER_BOUND_MS", 0.1))

if __name__ == '__main__':
    if args.similarity_check:
        cutlass_cm = CUTLASS_CM()
        cutlass_cm.cross_kernel_similarity(target_kernel_type, source_gpu, target_gpu)
        exit(0)

    ### Load raw features
    filters = Filters({
        "kernel": [
            "gemm",
            "conv2d"
        ],
        DIMENSION_NAME.gpu_model: [
            source_gpu,
            target_gpu,
        ]
    })
    kernel_data_dict = collect_data(
        trace_root_path,
        os.path.join(PROJECT_DIR, "_cache/cutlass_data"),
        filters,
        cutlass=True,
        force=args.force_load_data)
    print("# of kernels = {}".format(len(kernel_data_dict)))

    ### Train and evaluate the cost model
    setting = Setting(source_gpu, target_gpu,
        target_kernel_type, 
        AVE_LOWER_BOUND_MS,
        debug_kernel_list)

    use_dnn = False
    if use_dnn:
        cutlass_cm = DNNPredictor()
        mape_rst_path = os.path.join(PROJECT_DIR, "_cache/cutlass_mape/{}_mape_dnn.pickle".format(setting))
    else:
        cutlass_cm = CUTLASS_CM()
        mape_rst_path = os.path.join(PROJECT_DIR, "_cache/cutlass_mape/{}_mape.pickle".format(setting))
    
    mape_array, method_list = cutlass_cm.train_evaluate_all_kernels(
        kernel_data_dict,
        setting,
        mape_rst_path,
        force_fit=args.force_fit,
        check_one_kernel=args.check)

    if debug_kernel_list:
        exit(0)

    print("\n\n{}, {} kernels, {} methods".format(
        setting, mape_array.shape[0], mape_array.shape[1]))

    method_list = np.array([method.replace("pipeline_num", "tile_num") for method in method_list])
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(211)
    plt.title(str(setting) + ", {} kernels".format(mape_array.shape[0]), fontsize=16)
    plt.boxplot(mape_array[:, ::-1], labels=method_list[::-1], vert=False, showmeans=True)
    ref_line = [10] * round(mape_array.shape[1]+1)
    plt.plot(ref_line, list(range(len(ref_line))), '--', label="10%")
    plt.xticks(fontsize=16)
    plt.xlim(0, 100)
    plt.xlabel("Prediction Error (%)", fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)

    ax = fig.add_subplot(212)
    n_kernels = mape_array.shape[0]
    SHOW_PERCENT = 0.05
    mask = np.zeros(n_kernels, dtype=bool)
    train_idx = np.random.choice(n_kernels, math.ceil(SHOW_PERCENT * n_kernels), replace=False)
    mask[train_idx] = True
    plt.title(str(setting) + ", {} kernels -- {}% data".format(mape_array.shape[0], SHOW_PERCENT * 100), fontsize=16)
    for idx in range(len(method_list)):
        if idx <= 3:
            continue
        method = method_list[idx]
        ax.plot(np.array(range(n_kernels))[mask], mape_array.T[idx][mask], label=method)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Kernels", fontsize=16)
    plt.ylim(0, 100)
    plt.ylabel("Prediction Error (%)", fontsize=16)
    plt.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "_fig/cutlass/{}_mape.png".format(setting)))