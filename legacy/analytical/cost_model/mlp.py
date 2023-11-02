import os
import numpy as np
import pickle
import time

from utils.base import DIMENSION_NAME
from utils.op_info import enriched_header, kernel_type2op_type
from utils.util import Scaler, PROJECT_DIR
from dataloader import collect_data_two_gpu_cutlass

from analytical.cost_model.cutlass_cm import CUTLASS_CM

import tensorflow as tf
import tensorflow.keras as keras

def build_model(input_shape, layer_num=8, layer_size=1024):
    layers = [keras.layers.Dense(layer_size, activation='relu', input_shape=input_shape)]
    for _ in range(layer_num):
        layers.append(keras.layers.Dense(layer_size, activation='relu'))
    layers.append(keras.layers.Dense(1))
    model = keras.Sequential(layers)

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

class DNNPredictor(CUTLASS_CM):
    def __init__(self, norm=True, batch_size=8, one_model=False):

        super(CUTLASS_CM, self).__init__()
        self.cm_name = "DNN Predictor" 

        self.cm_dir = os.path.join(PROJECT_DIR, "_cache/cutlass_cms/dnn")
        if not os.path.exists(self.cm_dir):
            os.makedirs(self.cm_dir)

        self.normalize = norm
        self.average_output = True
        
        self.batch_size = batch_size

        self.model = None
        self.one_model = one_model

        self.load()

    def init_cm(self, file_key, cm_key, scaler=None):
        ### Store the normalize_upper, which can be used to normalize data during inference.
        self.scaler.combine(scaler)

    def inference_iplmt(self, testX, op_type, _model):
        if self.normalize:
            new_header = enriched_header(op_type)
            norm_testX = self.scaler.normalize(new_header, testX)
        else:
            norm_testX = np.array(testX)

        predicted = _model.predict(norm_testX).flatten()

        if self.normalize:
            return self.scaler.denormalize(DIMENSION_NAME.ave, predicted)
        else:
            return predicted

    def ret_model(self, source_gpu, target_gpu, op_type, kernel):
        new_header = enriched_header(op_type)
        if self.one_model:
            if self.model is None:
                self.model = build_model(input_shape=[len(new_header)])
            return self.model
        else:
            # Build a DNNRegressor
            if op_type not in self.CM_DICT:
                self.CM_DICT[op_type] = build_model(input_shape=[len(new_header)])
            return self.CM_DICT[op_type]
    
    def dump(self):
        self.scaler.dump()

        for op_type, _model in self.CM_DICT.items():
            _model.save(os.path.join(self.cm_dir, op_type))
        if self.model is not None:
            self.model.save(os.path.join(self.cm_dir, "one_model"))

    def load(self):
        self.scaler = Scaler(dump_path=os.path.join(self.cm_dir, "../norm_upper.json"))
        self.scaler.load()

        if os.path.exists(self.cm_dir):
            for _dir in os.listdir(self.cm_dir):
                if _dir == "one_model":
                    self.model = keras.models.load_model(os.path.join(self.cm_dir, _dir))
                else:
                    self.CM_DICT[_dir] = keras.models.load_model(os.path.join(self.cm_dir, _dir))
                print("Successfully load DNN at {} for {}".format(self.cm_dir, _dir))

    def training(self, trainX, trainY, testX, testY, 
            source_gpu, target_gpu, 
            op_type, kernel):

        _model = self.ret_model(source_gpu, target_gpu, op_type, kernel)
        if self.normalize:
            new_header = enriched_header(op_type)
            norm_trainX = self.scaler.normalize(new_header, trainX)
            norm_trainY = self.scaler.normalize(DIMENSION_NAME.ave, trainY)
        else:
            norm_trainX = trainX
            norm_trainY = trainY

        ### do training
        history = _model.fit(
            norm_trainX, norm_trainY,
            epochs=10, validation_split=0.2, verbose=0,
            callbacks=[])
        
        ### validation
        predicted = self.inference_iplmt(testX, op_type, _model)
        mape = self.prediction_error(testY, predicted)

        print("{}, mape: {:.3f} %".format("DNNPredictor", mape * 100))

        self.dump()

        return [mape * 100], ["DNNPredictor"]

    def inference(self, testX, source_gpu, target_gpu, kernel, op_type, method=-1, verbose=False):
        _model = self.ret_model(source_gpu, target_gpu, op_type, kernel)
        return self.inference_iplmt(testX, op_type, _model)
