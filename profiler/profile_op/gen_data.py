import common
import argparse
import tensorflow as tf
import numpy as np

from profiler.profile_op.ops import op_search_space
from profiler.profile_op.op_cfg import ConfigGenThread
from profiler.profile_op.common import run_one_cfg_str

parser = argparse.ArgumentParser(prog="Test")
parser.add_argument('--op', type=str, default=None,
    choices=[None, "Conv2D", "MatMul", "Relu", "Tanh"], help="Generate configs only")
parser.add_argument('--cfg_cache_path', type=str, default=None, help="Path to cache op configs")
parser.add_argument('--cache_path', type=str, default=None, help="Path to cache profile results")
parser.add_argument('--cfg_str', type=str, default=None, help="If specified, run one specific config")
parser.add_argument('--nsys', action="store_true", default=False, help="If specified, profile with nsys")
args = parser.parse_args()

def wrap_profile_op(op_type):
    range_dict, op_fn, input_shape_fn,\
        feature_fn, check_valid_fn = op_search_space(op_type)
    common.run_all_cfg(op_type, range_dict,
                    op_fn, input_shape_fn, feature_fn, check_valid_fn=check_valid_fn)

def wrap_profile_one_cfg(op_type, config_str, enable_nsys=False, cache_path=None):
    range_dict, op_fn, input_shape_fn,\
        feature_fn, check_valid_fn = op_search_space(op_type)
    raw_features = []
    run_one_cfg_str(
        op_type,
        config_str,
        op_fn,
        input_shape_fn,
        feature_fn,
        raw_features=raw_features,
        enable_nsys=enable_nsys)
    if cache_path is not None:
        with open(cache_path, 'a') as fp:
            fp.write(str(raw_features)+"\n")

def wrap_gen_cfg(op_type, cache_path):
    range_dict, op_fn, input_shape_fn,\
        feature_fn, check_valid_fn = op_search_space(op_type)
    config_t = ConfigGenThread(range_dict, check_valid_fn=check_valid_fn)
    config_t.start()
    config_t.join()
    config_t.dump(cache_path=cache_path)

if __name__ == "__main__":
    if args.cfg_cache_path is not None:
        wrap_gen_cfg(args.op, args.cfg_cache_path)
        exit(0)
    if args.cfg_str is not None:
        wrap_profile_one_cfg(args.op, args.cfg_str, enable_nsys=args.nsys, cache_path=args.cache_path)
        exit(0)
    wrap_profile_op("Conv2D")
    wrap_profile_op("MatMul")
    # wrap_profile_op("Relu")
    # wrap_profile_op("Tanh")
