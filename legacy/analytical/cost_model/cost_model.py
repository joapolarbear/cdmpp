import os
import pickle
import time
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import subprocess
import PyInquirer

from utils.base import DIMENSION_NAME
from utils.util import Scaler, axis_label, PROJECT_DIR
from utils.op_info import enriched_header
from dataloader import collect_data_two_gpu
from analytical.cost_model.functions import COST_FUNC_LIST, flops_idx, size_idx, ai_idx, perf_idx
from analytical.cost_model.functions import pipeline_num_idx, tile_size_in_idx, tile_size_out_idx, tile_flop_idx

font = {"color": "darkred",
        "size": 13,
        "family": "serif"}

class BASE_CM:
    def __init__(self):
        ''' self.CM_DICT is a two-level dict
            * The first level is keyed by op_type
            * The second level is keyed by cm_key, 
                which can be customized by successors
        '''
        self.CM_DICT = {}
        self.cm_dir = None
        self.cm_name = None

        self.scaler = None

        self.normalize = True
        self.average_output = True
    
    def gen_cm_key(self):
        ''' The secondary key in the cost model'''
        raise NotImplementedError()

    def decode_cm_key(self, cm_key):
        return cm_key.split("-")

    def gen_file_key(self, op_type, source_gpu, target_gpu):
        ''' The cost model for each file_key is cached as a file
        '''
        return "-".join([op_type, "norm" if self.normalize else "no_norm", source_gpu, target_gpu])
    
    def decode_file_key(self, file_key):
        sp = file_key.split("-")
        op_type = sp[0]
        norm = True if sp[1] == "norm" else False
        source_gpu = sp[2]
        target_gpu = sp[3]
        return op_type, norm, source_gpu, target_gpu

    def load(self):
        if os.path.exists(self.cm_dir):
            for file in os.listdir(self.cm_dir):
                if not file.endswith(".pickle"):
                    continue
                file_key = file.split(".")[0]
                _, norm, source_gpu, target_gpu = self.decode_file_key(file_key)
                if self.normalize == norm:
                    with open(os.path.join(self.cm_dir, file), 'rb') as fp:
                        self.CM_DICT[file_key] = pickle.load(fp)
                    print("Successfully load {} from {} {}".format(self.cm_name, self.cm_dir, file_key))
        
        self.scaler = Scaler(dump_path=os.path.join(self.cm_dir, "norm_upper.json"))
        self.scaler.load()

    def dump(self):
        for file_key in self.CM_DICT.keys():
            dump_path = os.path.join(self.cm_dir, file_key+".pickle")
            ### Pickle dump will delete existing file first then dump data
            # to avoid file lossing due to some interupption during the dump process
            # dump data to a temp file first
            with open(dump_path+".swap", 'wb') as fp:
                pickle.dump(self.CM_DICT[file_key], fp)
            subprocess.check_call("mv {} {}".format(dump_path+".swap", dump_path), stderr=subprocess.STDOUT, shell=True)
        
        self.scaler.dump()

    def init_cm(self, file_key, cm_key, scaler=None):
        ### The cost model is keyed by op_type+gpu type, so that we can access them as necessary
        if file_key not in self.CM_DICT:
            self.CM_DICT[file_key] = {}
        cm_dict = self.CM_DICT[file_key]
        if cm_key not in cm_dict:
            cm_dict[cm_key] = dict([(idx, {}) for idx in range(1, len(COST_FUNC_LIST))])
        
        ### Store the normalize_upper, which can be used to normalize data during inference.
        self.scaler.combine(scaler)

    def prediction_error(self, real, pred):
        # mape = abs(np.average(pred) - np.average(real)) / np.average(real)
        mape = np.average(np.abs((pred - real) / real))
        # mape = np.average(np.arctanh(((pred - real) / real)))
        return mape

    def training_iplt(self, trainX, trainY, testX, testY,
            op_type, file_key, cm_key):
        ''' 
        trainX, trainY, testX, testY: numpy.ndarray
            Training and test data. The shape of trainX = (n_sample, n_dim)
        '''

        cm_dict = self.CM_DICT[file_key]

        if self.normalize:
            new_header = enriched_header(op_type)
            norm_trainX = self.scaler.normalize(new_header, trainX)
            # norm_trainY = self.scaler.normalize(DIMENSION_NAME.ave, trainY)
        else:
            norm_trainX = trainX
        norm_trainY = trainY

        mape_list = [0]
        method_list = [COST_FUNC_LIST[0][0]]

        def test_rst(method_id):
            predict = self.inference_iplt(testX, op_type, file_key, cm_key, method=method_id)
            mape = self.prediction_error(testY, predict)
            print("{}, mape: {:.3f} %".format(COST_FUNC_LIST[method_id][0], mape * 100))
            mape_list.append(mape * 100)
            method_list.append(COST_FUNC_LIST[method_id][0])
            return mape

        def do_train(pred_func, init_value, bounds):    
            def loss_func(x):
                return (np.linalg.norm(pred_func(norm_trainX, x) - norm_trainY)) ** 2
            res = minimize(loss_func, init_value,
                method='L-BFGS-B',
                bounds=bounds)

            # res = curve_fit(pred_func, trainX, trainY)
            return res
        
        opt_error = None
        opt_method_id = None
        for method_id in range(len(COST_FUNC_LIST)):
            if method_id == 0:
                continue

            if method_id not in cm_dict[cm_key]:
                cm_dict[cm_key][method_id] = {}

            res = do_train(
                COST_FUNC_LIST[method_id][1],
                COST_FUNC_LIST[method_id][2](trainX),
                COST_FUNC_LIST[method_id][3](trainX))
            cm_dict[cm_key][method_id]["para"] = res
            error = test_rst(method_id)

            ### TODO (huhanpeng): currently, we manually limited the method id among 6-8
            if method_id in [6, 7, 8]:
                if opt_error is None or error < opt_error:
                    opt_method_id = method_id
                    opt_error = error

        cm_dict[cm_key]["opt_method_id"] = opt_method_id

        self.dump()

        return mape_list, method_list
    
    def inference_iplt(self, testX, op_type, file_key, cm_key, method=-1, verbose=False):
        cm_dict = self.CM_DICT[file_key]
        if method < 0:
            method = cm_dict[cm_key]["opt_method_id"]
            if verbose:
                print("!!! Use the optimal tile-based fitting function by default")
        model_para = cm_dict[cm_key][method]["para"].x
        if verbose:
            print("Use {} to perform inference".format(COST_FUNC_LIST[method][0]))
            print(" ** Para: {}".format(model_para)) 
        
        return self.inference_internal(testX, method, model_para, op_type)
        
    def inference_internal(self, testX, method_id, model_para, op_type):
        if self.normalize:
            new_header = enriched_header(op_type)
            norm_testX = self.scaler.normalize(new_header, testX)
            # return self.scaler.denormalize(DIMENSION_NAME.ave, COST_FUNC_LIST[method_id][1](norm_testX, model_para))
        else:
            norm_testX = np.array(testX)
        return COST_FUNC_LIST[method_id][1](norm_testX, model_para)
            
    def check_exist_in_cm_iplt(self, file_key, cm_key):
        if file_key not in self.CM_DICT:
            return False
        cm_dict = self.CM_DICT[file_key]
        if cm_key not in cm_dict:
            return False
        return True

    def case_study(self, trainX, trainY, testX, testY, op_type,
            source_gpu, target_gpu, file_key, cm_key, method_list, mape_list):
        ''' Input shape: trainX=(n_sample, n_dim)
        '''
        method_list = ["{}: {:30s}, Error: {:.3f} %".format(idx, method, mape_list[idx]) 
            for idx, method in enumerate(method_list)]
        questions = [
            {
                "type": 'list',
                "message": "Select one option",
                "name": "option",
                "choices": method_list + ["Check next case", "Exit"]
            }
        ]
        while True:
            print("\n")
            answers = PyInquirer.prompt(questions)
            if "option" not in answers:
                continue
            elif answers["option"] == "Exit":
                exit(0)
            elif answers["option"] == "Check next case":
                return
            else:
                policy = int(answers["option"].split(":")[0])

            pred_test_Y = self.inference_iplt(
                testX, op_type, file_key, cm_key,
                method=policy, verbose=True)

            ave_train_X = trainX.T[0]
            ratio = ave_train_X / trainY
            ratio = 2 * ratio / max(ratio)

            flops = trainX.T[flops_idx]
            size = trainX.T[size_idx]
            ai = trainX.T[ai_idx]
            perf = trainX.T[perf_idx]

            pipeline_num = trainX.T[pipeline_num_idx]
            tile_size_in = trainX.T[tile_size_in_idx]
            tile_size_out = trainX.T[tile_size_out_idx]
            tile_flop = trainX.T[tile_flop_idx]


            ### Plot figures

            fig = plt.figure(figsize=(16, 12))
            fig_base = 330
            fig_idx = 0

            ### Observation
            fig_idx += 1
            ax = fig.add_subplot(fig_base+fig_idx)
            ax.scatter(flops, ave_train_X, label=source_gpu, alpha=0.5, edgecolors='none')
            ax.scatter(flops, trainY, label=target_gpu, alpha=0.5, edgecolors='none')
            plt.ylabel(axis_label("ave"))
            plt.xlabel(axis_label("flops"))
            plt.legend()

            fig_idx += 1
            ax = fig.add_subplot(fig_base+fig_idx)
            ax.scatter(size, ave_train_X, label=source_gpu, alpha=0.5, edgecolors='none')
            ax.scatter(size, trainY, label=target_gpu, alpha=0.5, edgecolors='none')
            plt.ylabel(axis_label("ave"))
            plt.xlabel(axis_label("size"))
            plt.legend()

            if False:
                ### 3d visualization
                fig_idx += 1
                ax = fig.add_subplot(fig_base+fig_idx, projection="3d")
                # ax.scatter(flops, size, ave_train_X, label="V100", alpha=0.5, edgecolors='none')
                # ax.scatter(flops, size, trainY, label=target_gpu, alpha=0.5, edgecolors='none')
                ax.plot_trisurf(flops, size, ave_train_X, label=source_gpu, linewidth=0.2, antialiased=True)
                ax.plot_trisurf(flops, size, trainY, label=target_gpu, linewidth=0.2, antialiased=True)
                plt.ylabel(axis_label("size"))
                plt.xlabel(axis_label("flops"))
                ax.set_zlabel(axis_label("ave"))
                ax.view_init(20, -110)

                fig_idx += 1
                ax = fig.add_subplot(fig_base+fig_idx)
                ax.scatter(flops, perf, label=source_gpu, alpha=0.5, edgecolors='none')
                ax.scatter(flops, flops / (trainY * 1e-3), label=target_gpu, alpha=0.5, edgecolors='none')
                plt.ylabel(axis_label("perf"))
                plt.xlabel(axis_label("flops"))
                plt.legend()

                ### flops to size
                # ax = fig.add_subplot(335, projection='3d')
                fig_idx += 1
                ax = fig.add_subplot(fig_base+fig_idx)
                plt.scatter(flops, size, c=ave_train_X, alpha=0.5, edgecolors='none',
                            cmap=plt.cm.get_cmap('rainbow', 100))
                # plt.scatter(flops, size, ave_train_X, alpha=0.5)
                plt.xlabel(axis_label("flops"))
                plt.ylabel(axis_label("size"))
                cbar = plt.colorbar()
                cbar.set_label(label=axis_label("ave"), fontdict=font)

                fig_idx += 1
                ax = fig.add_subplot(fig_base+fig_idx)
                ax.scatter(ai, ave_train_X, label=source_gpu, alpha=0.5, edgecolors='none')
                ax.scatter(ai, trainY, label=target_gpu, alpha=0.5, edgecolors='none')
                plt.ylabel(axis_label("ave"))
                plt.xlabel(axis_label("ai"))
                plt.legend()

            ### Roofline model
            fig_idx += 1
            ax = fig.add_subplot(fig_base+fig_idx)
            ax.scatter(ai, perf, label=source_gpu, alpha=0.5, edgecolors='none')
            ax.scatter(ai, flops / (trainY * 1e-3), label=target_gpu, alpha=0.5, edgecolors='none')
            plt.ylabel(axis_label("perf"))
            plt.xlabel(axis_label("ai"))
            plt.legend()
            plt.title("Roofline Model")

            ### !!!
            fig_idx += 1
            ax = fig.add_subplot(fig_base+fig_idx)
            cbar_value = trainX.T[4]
            plt.scatter(ave_train_X, trainY, c=cbar_value, alpha=0.5, edgecolors='none',
                        cmap=plt.cm.get_cmap('rainbow', 100))
            plt.xlabel("Ave of {} (ms)".format(source_gpu))
            plt.ylabel("Ave of {} (ms)".format(target_gpu))
            cbar = plt.colorbar()
            cbar.set_label(label=axis_label("ai"), fontdict=font)

            ### Predict results
            fig_idx += 1
            ax = fig.add_subplot(fig_base+fig_idx)
            test_flops = testX.T[1]
            plt.scatter(test_flops, testY, alpha=0.5,
                        label="real")

            plt.scatter(test_flops, pred_test_Y, alpha=0.5,
                        label="pred")

            plt.xlabel(axis_label("flops"))
            plt.ylabel("Ave of {} (ms)".format(target_gpu))
            plt.legend()

            fig_idx += 1
            ax = fig.add_subplot(fig_base+fig_idx)
            # ax.scatter(flops, B, label="B", alpha=0.5, edgecolors='none')
            # ax.scatter(flops, W, label="W", alpha=0.5, edgecolors='none')
            # ax.scatter(flops, pipeline_num, label="pipeline_num", alpha=0.5, edgecolors='none')
            # ax.scatter(pipeline_num, ave_train_X, label=source_gpu, alpha=0.5, edgecolors='none')
            ax.scatter(pipeline_num, trainY, label=target_gpu, alpha=0.5, edgecolors='none')
            plt.ylabel("pipeline num")
            # plt.xlabel(axis_label("flops"))
            plt.legend()

            fig_idx += 1
            ax = fig.add_subplot(fig_base+fig_idx)
            ax.scatter(flops, pipeline_num, label=target_gpu, alpha=0.5, edgecolors='none')
            plt.ylabel("pipeline num")
            plt.xlabel(axis_label("flops"))
            plt.legend()

            ### save
            plt.tight_layout()
            # plt.show()
            
            plt.savefig(os.path.join(PROJECT_DIR, "_fig/cross_gpu_comp.png"))
            plt.close(fig)

class OP_STAND_ALONE_CM(BASE_CM):
    def __init__(self,):
        super(OP_STAND_ALONE_CM, self).__init__()    
        self.cm_name = "OP_STAND_ALONE COST Model"
        self.cm_dir = os.path.join(os.path.join(PROJECT_DIR, "_cache"), "op_stand_alone_cms")
        if not os.path.exists(self.cm_dir):
            os.makedirs(self.cm_dir)
        self.load()

    def gen_cm_key(self):
        return "default"
    
    def inference(self, testX, source_gpu, target_gpu, op_type, method=-1, verbose=False):
        '''Test the cost model for a specific kernel and method'''
        ### Input shape: testX=(n_sample, n_dim)
        cm_key = self.gen_cm_key()
        file_key = self.gen_file_key(op_type, source_gpu, target_gpu)
        return self.inference_iplt(testX, op_type, file_key, cm_key, method=method, verbose=verbose)

    def training(self, trainX, trainY, testX, testY, 
            source_gpu, target_gpu, 
            op_type):
        ''' Train the cost model using different method for one kernel
            Input shape: trainX=(n_sample, n_dim)
        '''
        ### X A = Y ==> A = inverse(X) Y
        print(trainX.shape, testX.shape)
        cm_key = self.gen_cm_key()
        file_key = self.gen_file_key(op_type, source_gpu, target_gpu)
        return self.training_iplt(trainX, trainY, testX, testY, op_type, file_key, cm_key)
    
    def train_evaluate_one_op(self, trainX, trainY, testX, testY,
            op_type,
            source_gpu="Tesla_V100-SXM2-32GB",
            target_gpu="A100-SXM4-40GB",
            group_dims=None,
            scaler=None,
            check_one_op=False
            ):
        ''' Train and evaluate the Cost Model for one specific op type

        Parameters
        -----------
        trainX, trainY, testX, testY: numpy.ndarray
            Training and test data. The shape of trainX = (n_sample, n_dim)
        op_type: str
            Op type, MatMul or Conv2D
        normalize_upper: dict
            Store the maximum value of each dimension, used to normalize features
        check_one_op: boolean
            If it's True, only test one op, used for debug
        '''

        print("\n{}: {} training sampels with feature size of {}, {} test samples".format(
            op_type, trainX.shape[0], trainX.shape[1], testX.shape[0]))
         
        cm_key = self.gen_cm_key()
        file_key = self.gen_file_key(op_type, source_gpu, target_gpu)
        self.init_cm(file_key, cm_key, scaler=scaler)

        ### Train the cost model with different fitting functions
        mape, method_list = self.training(
            trainX, trainY, testX, testY,
            source_gpu, target_gpu,
            op_type)

        if check_one_op:
            self.case_study(trainX, trainY, testX, testY, op_type,
                source_gpu, target_gpu, file_key, cm_key, method_list, mape)
    
        return mape, method_list

    def train_evaluate_all_ops(self,
            op2xydata, 
            setting,
            mape_rst_path,
            force_fit=False,
            check_one_op=False):
        ''' Train and evaluate the Cost Model for each op

        Parameters
        -----------
        op2xydata: dict
            Raw features grouped by op type
        setting: class Setting
            Arguments
        mape_rst_path: str
            Path to cache the tested MAPE results
        force_fit: boolean
            If it's True, force to training the cost models no matter whether cached results exist
        check_one_op: boolean
            If it's True, only test one op_debug, used for debug
        '''
        if os.path.exists(mape_rst_path) and not force_fit:
            with open(mape_rst_path, 'rb') as fp:
                mape_list, method_list = pickle.load(fp)
        else:
            ### Generate training and test data for current setting 
            # (for a specific pair of source gpu and target gpu)
            # NOTE: the result features are enriched with flops, size, 
            # transform, ai and perf and can not be indexed using FULL_HEADERS
            st = time.time()
            op2xydata, scaler = collect_data_two_gpu(
                    setting.source_gpu,
                    setting.target_gpu, 
                    op2xydata,
                    ave_lower_bound=setting.ave_lower_bound)
            print("Take {:.3f} s to generate training and test data.".format(time.time() - st))

            ### Train and evaluate the cost model for current op
            mape_list = []
            method_list = None

            for op_type in op2xydata.keys():
                trainX, trainY, testX, testY = op2xydata[op_type]
                mape, method_list = self.train_evaluate_one_op(
                    trainX, trainY, testX, testY, 
                    setting.target_op_type,
                    setting.source_gpu, setting.target_gpu,
                    scaler=scaler,
                    check_one_op=check_one_op)
                mape_list.append(mape)

            if len(mape_list) == 0:
                raise ValueError("Empty mape result")
            ### shape = (# of kernels, # of methods)
            mape_list = np.array(mape_list)
            if not setting.debug_kernel_list:
                with open(mape_rst_path, 'wb') as fp:
                    pickle.dump([mape_list, method_list], fp)
            
        return mape_list, method_list