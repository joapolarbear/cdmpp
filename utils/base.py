import os

class DIMENSION_NAME:
    ave = "ave"
    gpu_model = "gpu_model"
    dtype = "dtype"
    model = "model"
    op_type = "op_type"
    bs = "bs"

ALL_DNN_MODEL = ["ResNet50", "BERT-Large", "None", "VGG16", "InceptionV3"]
def model2int(model):
    return ALL_DNN_MODEL.index(model)

ALL_DIMENSION = [DIMENSION_NAME.ave, DIMENSION_NAME.gpu_model,
                 DIMENSION_NAME.dtype, DIMENSION_NAME.model, DIMENSION_NAME.op_type,
                 DIMENSION_NAME.bs]

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
