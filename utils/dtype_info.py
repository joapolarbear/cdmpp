
ALL_DTYPE = [
    "fp16", "fp32",
    "int32", "int64",
    "resource",
    "bool",
    None]
DTYPE_SIZE_IN_BYTE = {
    "fp16": 2, 
    "fp32": 4
    }

### convert to cutlass style
CUTLASS_DTYPE_ALIAS = {
    "f16": ["f16", "fp16", "float16"],
    "f32": ["f32", "fp32", "float32"]
}
def convert2cutlass_dtype(dtype_str):
    for cutlass_dtype, alias in CUTLASS_DTYPE_ALIAS.items():
        if dtype_str in alias:
            return cutlass_dtype
    raise ValueError(dtype_str)

DTYPE_ALIAS = {
    "fp16": ["f16", "fp16", "float16"],
    "fp32": ["f32", "fp32", "float32"],
    "int32": ["int32"],
    "int64": ["int64"],
    "resource": ["resource"],
    "bool": ["bool"]
}
def convert2std_dtype(dtype_str):
    if dtype_str is None:
        return None
    for std_dtype, alias in DTYPE_ALIAS.items():
        if dtype_str in alias:
            return std_dtype
    raise ValueError(dtype_str)

def dtype2int(dtype):
    return ALL_DTYPE.index(dtype)
