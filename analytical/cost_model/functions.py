
import numpy as np

from utils.op_info import enriched_base_index

flops_idx = enriched_base_index("flops")
size_idx = enriched_base_index("size")
transform_idx = enriched_base_index("transform")
ai_idx = enriched_base_index("ai")
perf_idx = enriched_base_index("perf")

sm_num_idx = enriched_base_index("target_sm_num")
bw_idx = enriched_base_index("target_bw")
clock_idx = enriched_base_index("target_clock")
core_num_idx = enriched_base_index("target_core_num")

pipeline_num_idx = enriched_base_index("pipeline_num")
tile_size_in_idx = enriched_base_index("tile_size_in")
tile_size_out_idx = enriched_base_index("tile_size_out")
tile_flop_idx = enriched_base_index("tile_flop")

### Method 1
def pred_func_1(X, A):
    return X[:, 0] * A[0] + A[1]

def init_func_1(X):
    return [1, 1]

def bound_func_1(X):
    return [(0, None), (0, None)]

# def grad_func(A):
#     tmp = (pred_func(trainX, A) - trainY)
#     delta = trainX[:, 0]
#     return 2 * np.matmul(delta.T, tmp)

### Method 2
def pred_func_2(X, A):
    return np.matmul(X, A)

def init_func_2(X):
    return np.ones([X.shape[1]])

def bound_func_2(X):
    return None

# def grad_func(A):
#     tmp = (pred_func(trainX, A) - trainY)
#     delta = trainX
#     return 2 * np.matmul(delta.T, tmp)

### Method 3
def pred_func_3(X, A):
    co = np.matmul(X[:, 1:], A)
    return X[:, 0] * co

def init_func_3(X):
    return np.ones([X.shape[1]-1])

def bound_func_3(X):
    return None

# def grad_func(A):
#     tmp = (pred_func(trainX, A) - trainY)
#     delta = trainX[:, 0].reshape(trainX.shape[0], 1) * trainX[:, 1:]
#     return 2 * np.matmul(delta.T, tmp)

### Method 4
def pred_func_4(X, a):
    ai = X.T[ai_idx]
    return X[:, 0] * np.log(a[0] * ai + a[1]) * a[2] + a[3]

def init_func_4(X):
    return [1, 1, 1, 1]

def bound_func_4(X):
    return [(0, None), (None, None), (0, None), (0, None)]

# def grad_func(a):
#     tmp0 = (pred_func(trainX, a) - trainY)
#     _flops = trainX.T[1]
#     delta = [None, None, None]
#     delta[0] = trainX[:, 0] * a[2] * (_flops / (a[0] * _flops - a[1]))
#     delta[1] = trainX[:, 0] * a[2] * (- 1 / (a[0] * _flops - a[1]))
#     delta[2] = trainX[:, 0] * np.log(a[0] * _flops - a[1])
#     delta = np.array(delta).T
#     print(delta.T.shape)
#     print(tmp0.shape)
#     return 2 * np.matmul(delta.T, tmp0)

def pred_func_5(X, a):
    _flops = X.T[flops_idx]
    _size = X.T[size_idx]
    _transform = X.T[transform_idx]
    return (a[0] * _flops + a[1] * _transform) + a[2] * _size + a[3]

def init_func_5(X):
    return [1, 1, 1, 1]

def bound_func_5(X):
    return [(0, None), (0, None), (0, None), (0, None)]

def _wrap_func_678(X, A):
    pipeline_num = X.T[pipeline_num_idx]
    tile_size_in = X.T[tile_size_in_idx]
    tile_size_out = X.T[tile_size_out_idx]
    tile_flop = X.T[tile_flop_idx]
    _transform = X.T[transform_idx]

    bw_per_sm = X.T[bw_idx]
    clock = X.T[clock_idx]
    core_num = X.T[core_num_idx]

    T_read_tile = tile_size_in * A[0] / bw_per_sm + A[5]
    T_comp_tile = tile_flop * A[1] / (clock * core_num) + A[6]
    T_write_tile = tile_size_out * A[2] / bw_per_sm + A[7]
    T_transform = _transform * A[3]+ A[8]

    return T_read_tile, T_comp_tile, T_write_tile, T_transform, pipeline_num

def init_func_678(X):
    return [1, 1, 1, 1, 1, 1, 1, 1, 1]

def bound_func_678(X):
    return [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]

def pred_func_6(X, A):
    T_read_tile, T_comp_tile, T_write_tile, T_transform, pipeline_num = _wrap_func_678(X, A)
    return T_read_tile * pipeline_num + T_comp_tile + T_write_tile + T_transform + A[4]


def pred_func_7(X, A):
    T_read_tile, T_comp_tile, T_write_tile, T_transform, pipeline_num = _wrap_func_678(X, A)
    return T_read_tile + T_comp_tile * pipeline_num + T_write_tile + T_transform + A[4]

def pred_func_8(X, A):
    T_read_tile, T_comp_tile, T_write_tile, T_transform, pipeline_num = _wrap_func_678(X, A)
    return T_read_tile + T_comp_tile + T_write_tile * pipeline_num + T_transform + A[4]

COST_FUNC_LIST = [
    ### Method Name, fun, x0
    ("f(X) = X[0] * bandwidth_ratio", None, None, None),
    ("f(X) = X[0] * scalar", pred_func_1, init_func_1, bound_func_1),
    ("f(X) = XA", pred_func_2, init_func_2, bound_func_2),
    ("f(X) = (X[1:]A) * X[0]", pred_func_3, init_func_3, bound_func_3),
    ("f(X) = (log(I)) * X[0]", pred_func_4, init_func_4, bound_func_4),
    ("f(X) = flops + size", pred_func_5, init_func_5, bound_func_5),
    ("f(X) = r * tile_num + c + w ", pred_func_6, init_func_678, bound_func_678),
    ("f(X) = r + c * tile_num + w", pred_func_7, init_func_678, bound_func_678),
    ("f(X) = r + c + w * tile_num", pred_func_8, init_func_678, bound_func_678),
]