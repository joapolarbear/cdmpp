import onnx
import os
from tvm.contrib.download import download_testdata

class ModelInfo:
    def __init__(self, url, input_dict):
        self.url = url
        self.input_dict = input_dict
        self.path = url.split("/")[-1]

onnx_url = "https://github.com/onnx/models/raw/main/"
model_dict = {
    "resnet50": ModelInfo(
        onnx_url + "vision/classification/resnet/model/resnet50-v2-7.onnx",
        {"data": [None, 3, 224, 224]},
    ),
    "resnet50-habana": ModelInfo(
        "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.onnx",
        {"data": [None, 3, 224, 224]},
    ),
    "resnet18-habana": ModelInfo(
        "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx",
        {"data": [None, 3, 224, 224]},
    ),
    "densenet-121": ModelInfo(
        onnx_url + "vision/classification/densenet-121/model/densenet-7.onnx",
        {"data_0": [None, 3, 224, 224]},
    ),
    "vgg16": ModelInfo(
        onnx_url + "vision/classification/vgg/model/vgg16-12.onnx",
        {"data": [None, 3, 224, 224]},
    ),
    "bert": ModelInfo(
        onnx_url + "text/machine_comprehension/bert-squad/model/bertsquad-10.onnx",
        {
            "input_ids": [],
            "input_mask": [],
            "segment_ids": [],
            "label_ids": []
        },
    ),
    "alexnet": ModelInfo(
        onnx_url + "vision/classification/alexnet/model/bvlcalexnet-9.onnx",
        None
    ),
    "caffenet": ModelInfo(
        onnx_url + "vision/classification/caffenet/model/caffenet-12.onnx",
        None
    ),
    "inceptionv2": ModelInfo(
        onnx_url + "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx",
        None
    ),
    "mobilenet": ModelInfo(
        onnx_url + "vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        None
    ),
    "gpt2": ModelInfo(
        onnx_url + "text/machine_comprehension/gpt-2/model/gpt2-10.onnx",
        None
    )
}

def load_onnx_model(model, model_dir="."):
    if model in model_dict:
        model_path = os.path.join(model_dir, model_dict[model].path)
        if not os.path.exists(model_path):
            model_url = model_dict[model].url
            print(f"{model} does not exist at {model_dir},\n\tdownload it from {model_url}")
            path = download_testdata(model_url, model, module=None)
            os.system(f"mv {path} {model_path}")
    else:
        model_path = model
    return onnx.load(model_path)
    # return onnx.load("test_model.onnx")

def is_onnx_input(graph, _freeze_params=False):
    ''' refer to relay.frontend.from_onnx '''
    _nodes = set()
    _params = set()
    _input_names = set()
    for init_tensor in graph.initializer:
        if not init_tensor.name.strip():
            raise ValueError("Tensor's name is required.")
        if _freeze_params:
            _nodes.add(init_tensor.name)
        else:
            _nodes.add(init_tensor.name)
            _params.add(init_tensor.name)
    # parse inputs
    for i in graph.input:
        if i.name in _params:
            # i is a param instead of input
            continue
        elif i.name in _nodes:
            continue
        else:
            # Real input
            _input_names.add(i.name)
    return _input_names

def parse_onnx_model_input(onnx_model, batch_size=1):
    ''' Refer to https://stackoverflow.com/questions/56734576/find-input-shape-from-onnx-file
    '''
    # The model is represented as a protobuf structure and it can be accessed
    # using the standard python-for-protobuf methods

    ### used to judge wheher it's a real input or parameter
    # refer to relay.frontend.from_onnx
    nodes_or_params = set()
    for init_tensor in onnx_model.graph.initializer:
        if not init_tensor.name.strip():
            raise ValueError("Tensor's name is required.")
        nodes_or_params.add(init_tensor.name)

    input_dict = {}
    # iterate through inputs of the graph
    for input in onnx_model.graph.input:
        if input.name in nodes_or_params:
            continue
        print (input.name, end=": ")
        # get type of input tensor
        tensor_type = input.type.tensor_type
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            # iterate through dimensions of the shape:
            input_shape = []
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if (d.HasField("dim_value")):
                    print (d.dim_value, end=", ")  # known dimension
                    input_shape.append(int(d.dim_value))
                elif (d.HasField("dim_param")):
                    print (d.dim_param, end=", ")  # unknown dimension with symbolic name
                    input_shape.append(batch_size)
                else:
                    print ("?", end=", ")  # unknown dimension with no name
            input_dict[input.name] = input_shape
        else:
            print ("unknown rank", end="")
        print()
    return input_dict

def change_onnx_input_dim(model, batch_size=1):
    ''' Change the input dimension of on ONNX model in an
    in-place manner, refer to https://github.com/onnx/onnx/issues/2182
    '''
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "N"

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    # if len(inputs) > 1:
    #     ### TODO (huhanpeng): handle bert later
    #     return
    for input in inputs:
        ### TODO (huhanpeng)
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        # dim1.dim_param = sym_batch_dim
        # or update it to be an actual value:
        dim1.dim_value = batch_size
    
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')

def partition_onnx_model(model, start=0, end=-1, str_filter=None):
    ### https://github.com/onnx/onnx/issues/2078
    if str_filter:
        oldnodes = [n for n in model.graph.node if str_filter in n.name]
    else:
        oldnodes = [n for n in model.graph.node]
    start = start or 0
    end = end or -1
    newnodes = oldnodes[start:end] # or whatever
    print([n.name for n in newnodes])
    del model.graph.node[:] # clear old nodes
    model.graph.node.extend(newnodes)