import torch
import torch.nn as nn
import torch.nn.init as init

BATCH_SIZE = 32

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

        self.conv = nn.Conv2d(3, 64, (7, 7), 2, bias=False)
        # self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        return x
torch_model = TestModel()

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
# set the model to inference mode
torch_model.eval()

x = torch.randn(BATCH_SIZE, 3, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
    x,                         # model input (or a tuple for multiple inputs)
    "test_model.onnx",   # where to save the model (can be a file or file-like object)
    export_params=True,        # store the trained parameter weights inside the model file
    opset_version=10,          # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names = ['input'],   # the model's input names
    output_names = ['output'], # the model's output names
    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                'output' : {0 : 'batch_size'}})