import numpy as np

from utils.base import DIMENSION_NAME, ALL_DNN_MODEL, model2int
from utils.dtype_info import DTYPE_SIZE_IN_BYTE, ALL_DTYPE, dtype2int
from utils.device_info import gpu_model2int, ALL_GPU_MODEL

ALL_CONV_KIND = ['fprop', 'dgrad', 'wgrad']
def conv_kind2int(conv_kind):
    return ALL_CONV_KIND.index(conv_kind)

def eq_gemm_size(a):
    ### refer to CUTLASS, only for Conv2D
    ### a is the raw feature
    conv_kind_elem = a[FULL_HEADERS["Conv2D"].index("conv_kind")]
    if isinstance(conv_kind_elem, int):
        conv_kind = ALL_CONV_KIND[conv_kind_elem]
    else:
        try:
            conv_kind = ALL_CONV_KIND[int(conv_kind_elem[0])]
        except:
            print(conv_kind_elem)
            import code
            code.interact(local=locals())
            raise

    if conv_kind == "fprop":
        m = a[5] * a[11] * a[12]
        n = a[13]
        k = a[9] * a[10] * a[8]
    elif conv_kind == "dgrad":
        m = a[5] * a[6] * a[7]
        n = a[8]
        k = a[9] * a[10] * a[13]
    elif conv_kind == "wgrad":
        m = a[13]
        n = a[9] * a[10] * a[8]
        k = a[5] * a[11] * a[12]
    else:
        raise ValueError("Invalid Conv Operator (fprop, dgrad, wgrad): {}".format(conv_kind))
    return m, n, k

def conv2d_flops(feature):
    ### Cutlass's method
    m, n, k = eq_gemm_size(feature)
    flops_mainloop_ = m * n * k * 2
    flops_epilogue_ = m * n * 2
    return flops_mainloop_ + flops_epilogue_

    ### previous method
    a = feature
    return a[5] * a[13] * a[12] * a[11] * a[8] * a[9] * a[10]

def conv2d_size_in_float(feature):
    ### Cutlass's method
    m, n, k = eq_gemm_size(feature)
    return m * n + n * k + m * n

    ### Previous method
    a = feature
    return a[5] * a[6] * a[7] * a[8] + a[8] * a[9] * a[10] * a[13] + a[5] * a[11] * a[12] * a[13]
    
OP_HYPER_PARAMETERS = {
    "Conv2D": {
            "para": ["H", "W", "C", "R", "S", "P", "Q", "K", "stride", "conv_kind"],
            ###       6    7    8    9    10   11   12   13      14,      15
            "fun_flops": conv2d_flops,
            "fun_size": conv2d_size_in_float,
            "fun_transform": lambda a: a[5] * a[12] * a[11] * a[9] * a[10] * a[8]
        },
    "MatMul": {
            "para": ["C_in", "C_out", "bs_mult"],
            ###        6        7        8   
            "fun_flops": lambda a: a[8] * a[5] * a[6] * a[7],
            "fun_size": lambda a: a[8] * a[5] * a[6] + a[6] * a[7] + a[8] * a[5] * a[7]
        },
    "Relu": {
        "para": ["R"],
        "fun_flops": lambda a: a[5] * a[6],
        "fun_size": lambda a: 2 * a[5] * a[6],
    },
    "Tanh": {
        "para": ["R"],
        "fun_flops": lambda a: a[5] * a[6],
        "fun_size": lambda a: 2 * a[5] * a[6],
    },
    "CastToFp16": {},
    "CastToFp32": {}
}

FULL_HEADERS = {
    "base": ["ave", "gpu_model", "dtype", "model", "op_type", "bs"],
    "enriched": [
        "flops", "size", "transform", "ai", "perf",
        "source_sm_num",
        "target_sm_num",
        "source_bw",
        "target_bw",
        "source_clock",
        "target_clock",
        "source_core_num",
        "target_core_num",
        "Mtile", "Ntile", "Ktile",
        "pipeline_stage", "align",
        "GEMM_M", "GEMM_N", "GEMM_K",
        "tb_num",
        "wave_num",
        "pipeline_num",
        "tile_size_in",
        "tile_size_out",
        "tile_flop"
    ]
}
BASE_HEADER_LEN = len(FULL_HEADERS['base'])
for key in OP_HYPER_PARAMETERS:
    FULL_HEADERS[key] = FULL_HEADERS["base"] + OP_HYPER_PARAMETERS[key].get("para", [])

ALL_OP_TYPE = ['Mul', 'AddV2', 'Cast', 'Transpose', 'Pad', 'while/body/_1/while/StatefulPartitionedCall/gradient_tape/resnet50/conv1/Conv2D/Conv2DBackpropFilter-0-TransposeNCHWToNHWC-LayoutOptimizer:Transpose', 'Conv2D', 'MaxPool', 'Mean', 'MatMul', 'BiasAdd', 'Square', 'SoftmaxCrossEntropyWithLogits', 'Sum', 'Sqrt', 'Greater',
               'BiasAddGrad', 'Tile', 'while/body/_1/while/StatefulPartitionedCall/gradient_tape/resnet50/activation_48/ReluGrad-0-TransposeNCHWToNHWC-LayoutOptimizer:Transpose', 'Sub', 'ReluGrad', 'Conv2DBackpropFilter', 'Conv2DBackpropInput', 'AddN', 'RealDiv', 'Select', 'MaxPoolGrad', 'FusedBatchNormV3', 'Relu', 'FusedBatchNormGradV3',
               ### BERT-Large
               "RandomUniform", "GreaterEqual", "OneHot", "Reciprocal", 'RsqrtGrad', 'Einsum', 'L2Loss', 'StridedSliceGrad', 'Minimum', 'Softmax', 'Pow', 'StridedSlice', 'SquaredDifference', 'Rsqrt', 'Tanh', 'Neg', 'BatchMatMulV2', 'LogSoftmax', 'UnsortedSegmentSum', 'ArgMax', 'Pack', 'GatherV2', 'TanhGrad', 'Fill', 'Exp',
               "Add", 'AssignAddVariableOp', 'LessEqual', 'SelectV2',
               "Placeholder", "ShapeN", "Const", "StatelessIf",
               "ReadVariableOp", "ResourceApplyGradientDescent", "AssignVariableOp",
               "ZerosLike", "Slice", "Reshape", "SparseSoftmaxCrossEntropyWithLogits",
               "ExpandDims", "Identity", "Maximum", "FloorDiv"
               ]

def op_type2int(op_type):
    return ALL_OP_TYPE.index(op_type)


def enriched_base_index(dim):
    new_base = [FULL_HEADERS["base"][0]] + FULL_HEADERS['enriched'] + FULL_HEADERS["base"][1:]
    return new_base.index(dim)

DIM_CONVERT2INT = [lambda x: x, gpu_model2int, dtype2int, model2int, op_type2int, lambda x: x]

def feature_encoder(dim):
    if dim in FULL_HEADERS["base"]:
        return DIM_CONVERT2INT[FULL_HEADERS["base"].index(dim)]
    elif dim == "conv_kind":
        return conv_kind2int
    else:
        return lambda x: x

def feature_decode(dim, value):
    if dim == DIMENSION_NAME.gpu_model:
        return ALL_GPU_MODEL[int(value)]
    elif dim == DIMENSION_NAME.model:
        return ALL_DNN_MODEL[int(value)]
    elif dim == DIMENSION_NAME.dtype:
        return ALL_DTYPE[int(value)]
    elif dim == DIMENSION_NAME.op_type:
        return ALL_OP_TYPE[int(value)]
    elif dim == "conv_kind":
        return ALL_CONV_KIND[int(value)]
    else:
        return value

def raw_feature_index(features, dim, op_type=None):
    ''' Given features, parse values of the specific dim
    ---
    features: may be a 1-D list or 2-D array, 
        when it is a 2-D array, its shape is (n_dims, n_samples)
    
    Return:
        a value (if features is a 1-D list) or an 1-D array (if features is a 2-D array)
    '''
    headers = FULL_HEADERS[op_type]
    if dim in headers:
        return features[headers.index(dim)]
    elif dim.lower() == "transform":
        if isinstance(features, list):
            return OP_HYPER_PARAMETERS[op_type].get("fun_transform", lambda a: 1)(features)
        else:
            return OP_HYPER_PARAMETERS[op_type].get("fun_transform", lambda a: np.ones(a.shape[1]))(features)
    elif dim.lower() == "flops":
        return OP_HYPER_PARAMETERS[op_type]["fun_flops"](features)
    elif dim.lower() == "size":
        size_in_float = OP_HYPER_PARAMETERS[op_type]["fun_size"](features)
        byte_per_float = []
        dtypes = raw_feature_index(features, "dtype", op_type)
        if isinstance(dtypes, int) or isinstance(dtypes, float):
            return size_in_float * DTYPE_SIZE_IN_BYTE[feature_decode("dtype", int(dtypes))]
        else:
            for dtype in dtypes:
                byte_per_float.append(DTYPE_SIZE_IN_BYTE[feature_decode("dtype", int(dtype))])
            return size_in_float * np.array(byte_per_float)
    elif dim.lower() == "ai" or dim.lower() == "arithmetic_intensity":
        flops = raw_feature_index(features, "flops", op_type)
        size_in_byte = raw_feature_index(features, "size", op_type)
        return flops / size_in_byte
    elif dim.lower() == "p" or dim.lower() == "performance" or dim.lower().startswith("perf"):
        flops = raw_feature_index(features, "flops", op_type)
        ave_in_ms = raw_feature_index(features, "ave", op_type)
        return flops / (ave_in_ms * 1e-3) if ave_in_ms != 0 else -1

def enriched_header(op_type):
    return [FULL_HEADERS[op_type][0]] + FULL_HEADERS['enriched'] + FULL_HEADERS[op_type][1:]

def enriched_feature_index(enriched_feature, dim, op_type):
    new_header = enriched_header(op_type)
    return enriched_feature[new_header.index(dim)]

def parse_raw_feature(enriched_feature):
    return [enriched_feature[0]] + \
        list(enriched_feature[1+len(FULL_HEADERS['enriched']):])

OP2KERNL_TYPE = {
    "Conv2D": "conv2d_fprop",
    "Conv2DBackpropFilter": "conv2d_wprop",
    "Conv2DBackpropInput": "conv2d_dprop",
    "MatMul": "gemm"
}

def op_type2kernel_type(op_type):
    return OP2KERNL_TYPE[op_type]

def kernel_type2op_type(target_kernel_type):
    for op_type, kernel_type in OP2KERNL_TYPE.items():
        if kernel_type == target_kernel_type:
            return op_type
    raise ValueError(kernel_type)
