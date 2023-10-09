import os, sys
import subprocess
import argparse

from profiler.profile_op.op_cfg import config2str, str2config
from utils.cutlass_api import OperationArgs, find_cutlass_kernel
from utils.device_info import DEVICE_INFO, short_device_name, query_cc

import tensorflow as tf

parser = argparse.ArgumentParser(prog="Test")
parser.add_argument('--cfg_str', type=str, default=None, help="If set, run one specific config")
parser.add_argument('--ops', type=str, default=None, help="OPs to run")
parser.add_argument('--target_gpu', type=str, default=None, help="target gpu")
parser.add_argument('--cache_path', type=str, default=None, help="Path to cache profile results")
args = parser.parse_args()


op2cutlass_kernel_type = {
    "MatMul": "gemm",
    # "MatMul": "sparsgemm",
    "Conv2D": "conv2d_fprop",
}

DTYPE_NAME_CONVERTER = {
    tf.dtypes.float16: "f16",
    tf.dtypes.float32: "f32"
}

if __name__ == "__main__":
    op_args = OperationArgs()

    op_args.kernel_type = op2cutlass_kernel_type[args.ops]

    capability = query_cc(args.target_gpu)
    op_args.cc_major = capability // 10
    op_args.cc_minor = capability % 10

    config = str2config(args.cfg_str)
    op_args.dtype = DTYPE_NAME_CONVERTER[config["dtype"]]
    op_args.m = config["N"]
    op_args.k = config["C_in"]
    op_args.n = config["C_out"]
    kernel = find_cutlass_kernel(op_args, verbose=False)

    cmd = ["cutlass_profiler"]
    cmd.append("--kernels={}".format(kernel))
    cmd.append("--m={}".format(op_args.m))
    cmd.append("--n={}".format(op_args.n))
    cmd.append("--k={}".format(op_args.k))
    cmd.append("--beta=1")
    cmd.append("--A={}".format(op_args.dtype))
    cmd.append("--B={}".format(op_args.dtype))
    cmd.append("--C={}".format(op_args.dtype))
    cmd.append("--accum={}".format(op_args.dtype))
    lines = subprocess.check_output(cmd, shell=False).decode('utf-8').split("\n")
    header = lines[-3].split(",")
    data = lines[-2].split(",")

    if args.cache_path is not None:
        with open(args.cache_path, 'a') as fp:
            fp.write("{},{},{},{}\n".format(
                data[header.index("Operation")],
                data[header.index("Bytes")],
                data[header.index("Flops")],
                data[header.index("Runtime")]
                ))

# cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8