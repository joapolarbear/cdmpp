import sys
import os
import numpy as np
import pandas as pd

DEVICE_INFO = pd.read_csv(os.path.join(os.path.dirname(__file__), "device_info.csv"))
DEVICE_INFO.set_index("device_name", inplace = True)

# DEVICE_INFO = {
#         "A100-SXM4-40GB_1g_5gb":{
#             "BW_gbps": 3 * 1600 / 8
#         }, 
#         "A100-SXM4-40GB_2g_10gb": {
#             "BW_gbps": 2 * 1600 / 8
#         }, 
#         "A100-SXM4-40GB_3g_20gb": {
#             "BW_gbps": 4 * 1600 / 8
#         },
#         "A100-SXM4-40GB_4g_20gb": {
#             "BW_gbps": 8 * 1600 / 8
#         },
#         ### TODO (huhanpeng) MIG devices
#     }

ALIAS_DICT = {
    "A100": ["A100-SXM4-40GB", "A100_Graphics_Device", "A100-SXM-80GB"],
    "V100": ["Tesla_V100-SXM2-32GB", "Tesla_V100-PCIE-32GB"],
    "T4": ["Tesla_T4"],
    "A100-SXM4-40GB_1g_5gb": [],
    "A100-SXM4-40GB_2g_10gb": [],
    "A100-SXM4-40GB_3g_20gb": [],
    "A100-SXM4-40GB_4g_20gb": [],
    "A30": [],
    "A10": ["NVIDIA_A10"],
    "K80": [],
    "P100": ["Tesla_P100-PCIE-12GB"],
    "HL-100": [],
    "E5-2673": [],
    "EPYC-7452": [],
    "GRAVITON2": [],
    "PLATINUM-8272": []
}

ALIAS2SHORT = {}

for gpu_model, alias in ALIAS_DICT.items():
    ALIAS2SHORT[gpu_model] = gpu_model
    for alia in alias:
        ALIAS2SHORT[alia] = gpu_model

### **NOTE**: ALL_GPU_MODEL must be modified manually, instead of being generated from ALIAS2SHORT
# Be careful to change the order of gpu names, because changing the order of a gpu model 
# name may affect the current cached profiled data
ALL_GPU_MODEL = [
    "A100-SXM4-40GB_1g_5gb",
    "A100-SXM4-40GB_2g_10gb",
    "A100-SXM4-40GB_3g_20gb",
    "A100_Graphics_Device",
    "A100-SXM4-40GB_4g_20gb",
    "A100-SXM4-40GB",
    "Tesla_T4",
    "A30",
    "Tesla_V100-SXM2-32GB",
    "A100",
    "T4",
    "V100",
    "A10",
    "K80",
    "P100",
    "HL-100",
    "E5-2673",
    "EPYC-7452",
    "GRAVITON2",
    "PLATINUM-8272"
]

def gpu_model2int(gpu_model):
    return ALL_GPU_MODEL.index(gpu_model)

def short_device_name(gpu_model):
    return ALIAS2SHORT[gpu_model]

def query_cc(gpu_model):
    ''' Query the compute capability '''
    return int(DEVICE_INFO.loc[short_device_name(gpu_model.upper())].capability)

def query_core_num(gpu_model, dtype, tensor_core_or_not):
    ''' Return the number of arithmetic units corresponding to `dtype` '''
    if dtype == "fp16":
        if tensor_core_or_not:
            return int(DEVICE_INFO.loc[short_device_name(gpu_model)].tensor_core_au_per_sm)
        elif "fp16_au_per_sm" in DEVICE_INFO[short_device_name(gpu_model)]:
            return int(DEVICE_INFO.loc[short_device_name(gpu_model)].fp16_au_per_sm)
        else:
            return int(DEVICE_INFO.loc[short_device_name(gpu_model)].fp32_au_per_sm)
    elif dtype == "fp32":
        return int(DEVICE_INFO.loc[short_device_name(gpu_model)].fp32_au_per_sm)

DEVICE_FEATURE_HEAD = [
    "device_id", 
    "clock_MHz", "memory_gb", "shm_L3_kb", "L2cache_mb", "L1cache_kb", "BW_gbps", "fp32_tflops"]
DEVICE_FEATURE_LEN = len(DEVICE_FEATURE_HEAD)

def get_device_feature(device: str):
    ''' Return device-dependent features'''
    device = short_device_name(device.upper())
    _device_info = DEVICE_INFO.loc[short_device_name(device)]
    MAX = DEVICE_INFO.loc["MAX"]
    device_feature = np.array([
        # gpu_model2int(device),
        1.,
        float(_device_info.clock_MHz) / float(MAX.clock_MHz),
        # int(_device_info.SM_count),
        np.log2(int(_device_info.memory_gb)) / np.log2(int(MAX.memory_gb)),
        np.log2(int(_device_info.shm_L3_kb)) / np.log2(int(MAX.shm_L3_kb)),
        np.log2(int(_device_info.L2cache_mb)) / np.log2(int(MAX.L2cache_mb)),
        np.log2(int(_device_info.L1cache_kb)) / np.log2(int(MAX.L1cache_kb)),
        np.log2(float(_device_info.BW_gbps)) / np.log2(float(MAX.BW_gbps)),
        float(_device_info.fp32_tflops) / float(MAX.fp32_tflops)
    ])
    assert len(device_feature) == DEVICE_FEATURE_LEN
    return device_feature

if __name__ == "__main__":
    cc = query_cc(sys.argv[1])
    print(cc)