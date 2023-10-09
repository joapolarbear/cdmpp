import os
import re
import subprocess
import hashlib

from utils.env import PROJECT_CFG
from utils.util import Scaler

cutlass_mapping_handler = os.path.join(PROJECT_CFG["cutlass_path"], "build/examples/20_find_op_kernels/20_find_op_kernels")

def safe_str2int(_str):
    return -1 if _str is None else int(_str)

def suffix_hash(_str):
    _str = "" if _str is None else _str
    return int(hashlib.sha1(_str.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
        
tensor_core_or_not = ["tensorop", "simt"]
dtype_out = ["f16", "f32"]
dtype_accum = ["s", "h", "d", "i", "c", "z", "gc", "gz"]
kernel_kind = ["gemm", "spgemm", "fprop", "dgrad", "wgrad", "fprop3d", "dgrad3d", "wgrad3d"]
iter_algo = ["analytic", "optimized"]
layout = ["tt", "nn", "tn", "nt", "nhwc", "nchw"]

class KernelFeature:
    def __init__(self, name, regrex, to_int_func, weight=1):
        self.name = name
        self.regrex = regrex
        self.to_int_func = to_int_func
        self.weight = weight

KERNEL_FEATURES = [
    ### Feature name, Regrex String, toIntFunc,
    KernelFeature("provider", "(?P<provider>cutlass)", lambda x: 0, 1000),
    KernelFeature("tensor_core_or_not", "_(?P<tensor_core_or_not>tensorop|simt)", lambda x: tensor_core_or_not.index(x), 1000),
    KernelFeature("dtype_out", "(_(?P<dtype_out>f16|f32))?", lambda x: -1 if x is None else dtype_out.index(x), 1000),
    KernelFeature("dtype_accum", "_(?P<dtype_accum>s|h|d|i|c|z|gc|gz)", lambda x: dtype_accum.index(x), 1000),
    KernelFeature("tensor_core_size", "(?P<tensor_core_size>\d+)?", safe_str2int, 1),
    KernelFeature("kernel_kind", "(?P<kernel_kind>gemm|spgemm|fprop|dgrad|wgrad|fprop3d|dgrad3d|wgrad3d)", lambda x: kernel_kind.index(x), 1000),
    KernelFeature("iter_algo", "(_(?P<iter_algo>analytic|optimized))?", lambda x: -1 if x is None else iter_algo.index(x), 1000),
    KernelFeature("dtype_in", "(_(?P<dtype_in>f16|f32))?", lambda x: -1 if x is None else dtype_out.index(x), 1000),
    KernelFeature("threadblock_m", "_(?P<threadblock_m>\d+)", safe_str2int, 1),
    KernelFeature("threadblock_n", "x(?P<threadblock_n>\d+)", safe_str2int, 10),
    KernelFeature("threadblock_k", "_(?P<threadblock_k>\d+)", safe_str2int, 1),
    KernelFeature("pipeline_stage", "x(?P<pipeline_stage>\d+)", safe_str2int, 1),
    KernelFeature("layout", "_(?P<layout>((t|n){2})|(nhwc|nchw))", lambda x: layout.index(x), 0.001),
    KernelFeature("align", "(_(align(?P<align>\d)))?", safe_str2int, 10),
    KernelFeature("suffix", "(_(?P<suffix>.+))?", suffix_hash, 1000),
]

KernelFeatureNames = [kf.name for kf in KERNEL_FEATURES]
KernelFeatureScaler = Scaler()

### Cross-kernel prediction only supports two kernels 
# that the features corresponding to the feature id 
# in `FIXED_KERNEL_FEATURES` are the same
FIXED_KERNEL_FEATURES = [0, 1, 2, 3, 5, 6, 7]

KERNEL_RE_STR = ""
for kernel_feature in KERNEL_FEATURES:
    KERNEL_RE_STR += kernel_feature.regrex

def parse_kernel_name(kernel_name):
    rst = re.match(KERNEL_RE_STR, kernel_name)
    if rst is None:
        print(kernel_name)
        raise ValueError(kernel_name)
    return rst.groupdict()

def gen_kernel_feature(kernel):
    rst = parse_kernel_name(kernel)
    ret = []
    for kernel_feature in KERNEL_FEATURES:
        _raw = kernel_feature.to_int_func(rst[kernel_feature.name])
        KernelFeatureScaler.record(kernel_feature.name, _raw)
        ret.append(_raw * kernel_feature.weight)
    return ret

def norm_kernel_feature(raw_feature):
    return KernelFeatureScaler.normalize(KernelFeatureNames, raw_feature)

def allow_cross_kernel_pred(feature1, feature2):
    for idx in FIXED_KERNEL_FEATURES:
        if feature1[idx] != feature2[idx]:
            return False
    return True

class OperationArgs:
    def __init__(self):
        self.cc_major = 7
        self.cc_minor = 0
        self.kernel_type = "gemm" # or "conv2d_fprop"
        self.dtype = "f16" # or f32
        # dtype = "f16,nhwc" # or f32
        self.beta = 1

        self.m = 1024
        self.n = 1024
        self.k = 1024

        self.alignment = 16

        self.iter_algo = "analytic"


def find_cutlass_kernel(_op_args, verbose=False, all_kernel=False):
    '''
    Parameters
    ----------
    all_kernel: bool
        If specified, ret all possible kernels meeting the capacity, alignment requirements
    '''
    command = [cutlass_mapping_handler]
    command.append("--operation={}".format(_op_args.kernel_type))
    command.append("--A={}".format(_op_args.dtype))
    command.append("--B={}".format(_op_args.dtype))
    command.append("--C={}".format(_op_args.dtype))
    command.append("--accum={}".format(_op_args.dtype))
    command.append("--cc_major={}".format(_op_args.cc_major))
    command.append("--cc_minor={}".format(_op_args.cc_minor))

    if _op_args.kernel_type == "gemm":
        command.append("--m={}".format(_op_args.m))
        command.append("--n={}".format(_op_args.n))
        command.append("--k={}".format(_op_args.k))
        command.append("--beta={}".format(_op_args.beta))
        command.append("--alignment={}".format(_op_args.alignment))
    elif _op_args.kernel_type == "sparsegemm":
        command.append("--m={}".format(_op_args.m))
        command.append("--n={}".format(_op_args.n))
        command.append("--k={}".format(_op_args.k))
        command.append("--beta={}".format(_op_args.beta))
    elif _op_args.kernel_type == "conv2d_fprop":
        command.append("--conv_kind=fprop")
        command.append("--iter_algo={}".format(_op_args.iter_algo))
    else:
        raise ValueError(_op_args.kernel_type)
    
    if all_kernel:
        command.append("--all")

    ret = subprocess.check_output(command, env=None, stderr=subprocess.STDOUT, shell=False).decode('utf-8')
    match = re.findall(r"Recently executed '(.*)'", ret)

    if verbose:
        print(" ".join(command))
        print(ret)

    if len(match) == 0:
        print(ret)
        return None
    if all_kernel:
        return match[0].split(",")
    else:
        return match[0]