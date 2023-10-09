''' Plot the time distribution of operators in one DNN model
'''
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils.base import DIMENSION_NAME
from utils.util import Filters, PROJECT_DIR
from utils.env import PROJECT_CFG
from dataloader import collect_data

parser = argparse.ArgumentParser(prog="Test")
parser.add_argument('--force_load_data', action='store_true', help="force to load raw data")
args = parser.parse_args()

fontdict = {"color": "darkred",
        "size": 18,
        "family": "serif"}

filters = Filters({
    DIMENSION_NAME.op_type: [
        # "MatMul",
        "Conv2D",
        # "Relu",
        # "Tanh",
        ],
    DIMENSION_NAME.dtype: [
        "fp32",
        # "fp16",
        ],
    "model": [
        "ResNet50"
        ],
    "gpu_model": [
        # "A100-SXM4-40GB", 
        # "Tesla_T4", 
        # "A30", 
        "Tesla_V100-SXM2-32GB",
        # "A100-SXM4-40GB_1g_5gb", 
        # "A100-SXM4-40GB_2g_10gb", "A100-SXM4-40GB_3g_20gb",
        # "A100-SXM4-40GB_4g_20gb", 
        ],
    DIMENSION_NAME.bs: [
        64
    ]
})

def wrap_str(_str):
    LEN = 50
    ret = ""
    idx = 0
    while idx < len(_str):
        if idx > 0:
            ret += "\n"
        ret += _str[(idx):(idx+LEN)]
        idx += LEN
    return ret

def per_dnn_trace_hook(name2stat):
    ave_upper_bould_list = [0.01, 0.05, 0.06, 0.1, 0.5, 1, 5, 10, 100, 1000] ### In ms level
    op_cnt_list = np.array([0] * len(ave_upper_bould_list))
    accu_ave_list = np.array([0.0] * len(ave_upper_bould_list))
    x_axis = np.log10(np.array(ave_upper_bould_list))
    # x_axis = ave_upper_bould_list

    for name, stat in name2stat.items():
        if name == "iter_time":
            continue
        for idx, ave_upper_bount in enumerate(ave_upper_bould_list):
            if stat["avg"] <= ave_upper_bount:
                op_cnt_list[idx] += 1
                accu_ave_list[idx] += stat["avg"]
                break
    print(op_cnt_list, accu_ave_list, name2stat["iter_time"])
    ### Normalize
    pdf_cnt = op_cnt_list / np.sum(op_cnt_list)
    pdf_ave = accu_ave_list / np.sum(accu_ave_list)

    cdf_cnt = np.cumsum(pdf_cnt)
    cdf_ave = np.cumsum(pdf_ave)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(x_axis, cdf_cnt, marker=".", markersize=10, label="CDF of # of Ops")
    plt.plot(x_axis, cdf_ave, marker="^", markersize=10, label="CDF of Average Time")
    plt.text(1, 0.5, "Iteration Time = {:.3f} ms".format(name2stat["iter_time"]), fontdict=fontdict)
    # plt.plot(x_axis, pdf_cnt, label="PDF of # of Ops")
    # plt.plot(x_axis, pdf_ave, label="PDF of Average Time")
    fontsize = 22
    plt.legend(fontsize=fontsize)
    plt.xlabel("Average Execution Time (ms, in log_10 scale)", fontsize=fontsize)
    plt.ylabel("CDF", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(wrap_str(filters.serialize_filter(decode=True)), fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "_fig/time_ratio.png"))
    exit(0)

trace_root_path = os.path.join(PROJECT_CFG["source_data_root"], "op_level_trace")
root_dir = "{}/dnn_traces".format(trace_root_path)
xydata = collect_data(root_dir, root_dir, filters, force=args.force_load_data, per_dnn_trace_hook=per_dnn_trace_hook)

