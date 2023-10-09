
import tensorflow as tf
import numpy as np

from profiler.profile_op.common import check_input_size

def op_search_space(op_type):
    if op_type == "Conv2D":
        range_dict = {
            "dtype": [tf.dtypes.float32, tf.dtypes.float16],
            "N": [1, 8, 16, 32, 64, 128, 256, 512, 1024],
            "H": [7, 14, 28, 56, 112, 224],
            # "W":,   W = H
            "C": [1, 8, 16, 32, 64, 128, 256, 512, 1024],
            "R": [1, 3],
            # "S":,   R = S,
            # "P":,   derived from stride
            # "Q":,   Q = P
            "K": [1, 8, 16, 32, 64, 128, 256, 512, 1024],
            # No activation
            # No bias
            "stride": [1, 2]
        }

        def input_shape_fn(config):
            return (config["N"], config["H"], config["H"], config["C"])

        def op_fn(config):
            ### if padding = valid,  P = (H - R) / stride + 1
            ### if padding = same, P = ceil(H / stride)
            return tf.keras.layers.Conv2D(
                config["K"], config["R"], strides=config["stride"], padding=config["padding"], use_bias=False)

        def feature_fn(config, output_shape):
            return [
                config["N"],
                config["H"], config["H"], config["C"],
                config["R"], config["R"],
                output_shape[-2], output_shape[-2], config["K"],
                config["stride"]]

        def output_shape_fn(config):
            ### NOTE: estimate, used to check output size
            return (config["N"], round(config["H"] / config["stride"]), round(config["H"] / config["stride"]), config["K"])

        def check_valid_fn(config):
            if config["R"] >= config["H"]:
                return False
            if config["H"] * config["C"] > 2000:
                return False
            if max(config["C"], config["K"]) / min(config["C"], config["K"]) > 4:
                return False
            if not check_input_size(input_shape_fn(config), config["dtype"]):
                return False
            if not check_input_size(output_shape_fn(config), config["dtype"]):
                return False
            return True

    elif op_type == "MatMul":

        ### MatMul and BiasAdd
        range_dict = {
            "dtype": [tf.dtypes.float32, tf.dtypes.float16],
            "N": [1, 8, 16, 32, 64, 128, 256, 512, 1024],
            "C_in": [1, 32, 64, 128, 256, 512, 768, 1024, 2048, 3072, 4096],
            "C_out": [1, 32, 64, 128, 256, 512, 768, 1024, 2048, 3072, 4096],
        }

        def input_shape_fn(config):
            return (config["N"], config["C_in"])

        def op_fn(config):
            return tf.keras.layers.Dense(config["C_out"], use_bias=True)

        def feature_fn(config, output_shape):
            return [config["N"], config["C_in"], config["C_out"], 1]

        check_valid_fn = None

    elif op_type == "Relu":
        ### Relu

        range_dict = {
            "dtype": [tf.dtypes.float32, tf.dtypes.float16],
            "dim0": [1, 8, 16, 32, 64, 128, 256, 512, 1024],
            "dim1": [1, 32, 64, 128, 256, 512, 768, 1024, 2048, 
                     1024*3, 1024*4, 1024*5, 1024*6, 1024*7, 1024*8,
                     1024*16, 1024*32, 1024*64]
        }

        def input_shape_fn(config):
            return (config["dim0"], config["dim1"])

         # def input_shape_fn(config):
        #     return (config["dim0"])

        # def input_shape_fn(config):
        #     return (config["dim0"], config["dim1"], config["dim2"], config["dim3"])

        def op_fn(config):
            return tf.keras.activations.relu

        def feature_fn(config, output_shape):
            return list(input_shape_fn(config))

        def check_valid_fn(config):
            if not check_input_size(input_shape_fn(config), config["dtype"]):
                return False
            return True

    elif op_type == "Tanh":
        '''
        One of transcendental element-wise operations
            // We treat transcendental operations separately since one transcendental
            // operation can correspond to several floating point ops.
            // kLogistic is included in "trascendental" as it is implemented using
            // trascendental ops (tanh or exp).   -- from tensorflow comments
        
        It's used in BERT
        '''

        range_dict = {
            "dtype": [tf.dtypes.float32, tf.dtypes.float16],
            "dim0": [1, 8, 16, 32, 64, 128, 256, 512, 1024],
            "dim1": [1, 32, 64, 128, 256, 512, 768, 1024, 2048,
                     1024*3, 1024*4, 1024*5, 1024*6, 1024*7, 1024*8,
                     1024*16, 1024*32, 1024*64]
        }

        def input_shape_fn(config):
            return (config["dim0"], config["dim1"])

        def op_fn(config):
            return tf.keras.activations.tanh
        
        def feature_fn(config, output_shape):
            return list(input_shape_fn(config))

        def check_valid_fn(config):
            if not check_input_size(input_shape_fn(config), config["dtype"]):
                return False
            return True

    return range_dict, op_fn, input_shape_fn, feature_fn, check_valid_fn
