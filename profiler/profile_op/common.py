import ctypes
import os, sys
import queue
import json
import numpy as np
from tqdm import tqdm

from profiler.profile_op.op_cfg import config2str, str2config, ConfigGenThread

upper_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(upper_path, "process"))

from dataloader import parse_traces

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # default level，display all information
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # only display warning and Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # only display Error
import tensorflow as tf
from tensorflow.python.client import device_lib

SAMPLE_SIZE = 1000
TMP_TRACE_DIR = os.environ.get("TMP_TRACE_DIR", "/home/tiger/traces")
FINAL_TRACE_DIR = os.environ.get("FINAL_TRACE_DIR", "/home/tiger/op_database/data")
ENABLE_NSYS = True
STEP_NUM = 10
WARMUP_STEP = 10

_cudart = ctypes.CDLL('libcudart.so')

def cu_prof_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception('cudaProfilerStart() returned %d' % ret)


def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)


def trace_start(model_dir, enable_nsys=False):
    if enable_nsys:
        cu_prof_start()
    else:
        tf.profiler.experimental.start(model_dir)


def trace_end(enable_nsys=False):
    if enable_nsys:
        cu_prof_stop()
    else:
        tf.profiler.experimental.stop()


def check_input_size(input_shape, dtype):
    float_byte = 4 if dtype == tf.dtypes.float32 else 2
    ### Extremly large tensors lead to OOM, size upper bound: 512 MB
    if float_byte * np.prod(input_shape) / (1024 ** 2) > 512:
        return False
    return True

def parse_gpu_model():
    all_devices = device_lib.list_local_devices()
    gpu_model = None
    for device in all_devices:
        if device.device_type == "GPU":
            gpu_model = device.physical_device_desc.split("name: ")[1].split(",")[0]
            gpu_model = gpu_model.replace(" ", "_")
            break
    assert gpu_model is not None, all_devices
    return gpu_model

def dump_rst(op_type, rst):
    gpu_model = parse_gpu_model()
    gpu_dir = os.path.join(FINAL_TRACE_DIR, gpu_model)
    if not os.path.exists(gpu_dir):
        os.makedirs(gpu_dir)
    final_path = os.path.join(gpu_dir, "{}.json".format(op_type))
    with open(final_path, "w") as f:
        json.dump(rst, f)
    print("Dump {} sample points at {}".format(len(rst), final_path))


def handle_tf_traces(op_type, config, y, feature_fn, raw_features=None):
    dirs = os.listdir(os.path.join(TMP_TRACE_DIR, "plugins/profile"))
    os.system(
        "cp {}/plugins/profile/{}/*.trace.json.gz {}/trace.json.gz && ".format(TMP_TRACE_DIR, dirs[0], TMP_TRACE_DIR) +
        "rm -rf {}/plugins {}/events*".format(
            TMP_TRACE_DIR, TMP_TRACE_DIR)
    )

    trace_path = "{}/trace.json.gz".format(TMP_TRACE_DIR)
    name2stat = parse_traces(trace_path, filters=None)
    new_sample_cnt = 0
    for _key, stat in name2stat.items():
        if _key == "iter_time":
            continue
        if stat["op_type"] == op_type:
            new_sample_cnt += 1
            # print(" - new point, ave: {} ms, stdev: {}, median: {}".format(stat["avg"], stat["var"], stat["median"]))
        ### For the trival op types, collect traces first, may be useful
        # if stat["avg"] > 100:
        #     raise
        feature = [
            float(stat["avg"]),
            "fp16" if config["dtype"] == tf.dtypes.float16 else "fp32",
            stat["op_type"]] + [int(x) for x in feature_fn(config, y.shape)]
        if raw_features is not None:
            raw_features.append(feature)

    if new_sample_cnt == 0:
        # raise ValueError("No new {} points found. {} ==> {}".format(
        #     op_type, config_str, prev_config_str))
        print("[Warning] No new {} points found. {}".format(op_type, config2str(config)))
        return
    elif new_sample_cnt > 1:
        raise ValueError("Multiple {} points found. {}".format(op_type, config2str(config)))
    os.system("rm -rf {}/*".format(TMP_TRACE_DIR))

def run_one_cfg(
        op_type,
        config,
        op_fn,
        input_shape_fn,
        feature_fn,
        raw_features=None,
        enable_nsys=False):
    input_shape = input_shape_fn(config)

    ### Start running the OP and profiling
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[-1], 'GPU')
    x = tf.random.normal(input_shape, dtype=config["dtype"])
    with tf.device("GPU"):
        try:
            for _ in range(WARMUP_STEP):
                y = op_fn(config)(x)
        except Exception as e:
            raise
        trace_start(TMP_TRACE_DIR, enable_nsys=enable_nsys)
        ### NOTE: no activation, no bias
        for _ in range(STEP_NUM):
            y = op_fn(config)(x)
        trace_end(enable_nsys=enable_nsys)
        
    ### Collect traces
    if enable_nsys:
        pass
    else:
        handle_tf_traces(op_type, config, y, feature_fn, raw_features=raw_features)

def run_one_cfg_str(
        op_type,
        config_str,
        op_fn,
        input_shape_fn,
        feature_fn,
        raw_features=None,
        enable_nsys=False):
    run_one_cfg(
            op_type,
            str2config(config_str),
            op_fn,
            input_shape_fn,
            feature_fn,
            raw_features=raw_features,
            enable_nsys=enable_nsys)

def run_all_cfg(op_type, range_dict, op_fn, input_shape_fn, feature_fn, check_valid_fn=None):
    print("Generate data points for {} ...".format(op_type))
    
    config_t = ConfigGenThread(range_dict, check_valid_fn=check_valid_fn)
    config_t.start()
    config_t.join()

    pgsbar = tqdm(total=config_t.size())
    raw_features = []
    while not config_t.empty():
        try:
            config = config_t.get()
        except queue.Empty:
            raise ValueError("Waiting for new sample points for {} sec".format(60))
               
        # input_shape = (33, 93, 93, 549)
        # config["K"] = 656
        # config["R"] = 1
        # config["stride"] = 1
        # config["dtype"] = tf.dtypes.float16

        pgsbar.update()

        run_one_cfg(
            op_type,
            config,
            op_fn,
            input_shape_fn,
            feature_fn,
            raw_features=raw_features,
            enable_nsys=ENABLE_NSYS)

    pgsbar.close()
    dump_rst(op_type, raw_features)
