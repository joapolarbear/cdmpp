import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

from utils.base import DIMENSION_NAME
from utils.util import PROJECT_DIR, idw_average
from utils.op_info import kernel_type2op_type, feature_decode
from dataloader import collect_data_two_gpu_cutlass, enrich_raw_feature
from analytical.cost_model.functions import COST_FUNC_LIST
from utils.cutlass_api import gen_kernel_feature, allow_cross_kernel_pred, norm_kernel_feature

from utils.cutlass_api import OperationArgs, find_cutlass_kernel
from utils.device_info import query_cc
from utils.dtype_info import convert2cutlass_dtype
from dataloader import fine_grain_kernel

from .cost_model import BASE_CM

def pick_cutlass_kernel_from_candidates(possible_kernels, possible_predYs, possible_errors):
    pick_idx = np.argmin(possible_predYs)
    return possible_kernels[pick_idx], possible_predYs[pick_idx], possible_errors[pick_idx]

def find_evaluate_cutlass_kernel_iplmt(
        cutlass_cm,
        raw_feature,
        target_kernel_type,
        source_gpu,
        target_gpu,
        header,
        verbose=False):

    target_op_type = kernel_type2op_type(target_kernel_type)

    op_args = OperationArgs()
    op_args.kernel_type = target_kernel_type

    capability = query_cc(target_gpu)
    op_args.cc_major = capability // 10
    op_args.cc_minor = capability % 10

    dtype_idx = header.index(DIMENSION_NAME.dtype)
    op_args.dtype = convert2cutlass_dtype(
        feature_decode(DIMENSION_NAME.dtype, raw_feature[dtype_idx]))

    bs_idx = header.index(DIMENSION_NAME.bs)

    if target_kernel_type == "gemm":
        op_args.m = raw_feature[bs_idx]
        op_args.k = raw_feature[bs_idx+1]
        op_args.n = raw_feature[bs_idx+2]

        possible_kernels = []
        for _alignment in [1, 2, 4, 8, 16]:
            op_args.alignment = _alignment
            possible_kernels += find_cutlass_kernel(op_args, verbose=verbose, all_kernel=True)

    elif target_kernel_type == "conv2d_fprop":
        possible_kernels = []
        for iter_algo in ["analytic", "optimized"]:
            op_args.iter_algo = iter_algo
            possible_kernels += find_cutlass_kernel(op_args, verbose=verbose, all_kernel=True)

    possible_predYs = []
    possible_errors = []
    for _kernel in possible_kernels:
        __kernel = fine_grain_kernel(_kernel, raw_feature[0], raw_feature, target_op_type, isRawFeature=True)
        testX = enrich_raw_feature(
            raw_feature, target_op_type,
            ave=1,
            source_gpu=source_gpu,
            target_gpu=target_gpu,
            kernel=__kernel)
        try:
            if cutlass_cm.check_exist_in_cm(
                    source_gpu, target_gpu,
                    target_op_type, __kernel):
                _predY = cutlass_cm.inference(
                    np.array([testX]),
                    source_gpu,
                    target_gpu,
                    __kernel,
                    target_op_type,
                    method=-1,
                    verbose=verbose)[0]
            else:
                _predY, mape_per_kernel = cutlass_cm.cross_kernel_prediction(
                    np.array([testX]), 
                    None, __kernel,
                    source_gpu, target_gpu,
                    target_op_type,
                    n_neighbors=1,
                    verbose=verbose)
                _predY = _predY[0]

        except:
            print(f"Fail to query cm for {__kernel}")
            raise
        if raw_feature[0] == 0:
            _error = 1000
        else:
            _error = 100* np.abs(_predY - raw_feature[0]) / raw_feature[0]
        if verbose:
            print("True={:.3f}, Pred={:.3f}, Error={:.3f}%, Kernel={}".format(
                raw_feature[0], _predY,
                _error, __kernel))
        possible_predYs.append(_predY)
        possible_errors.append(_error)
    
    kernel, predY, error = pick_cutlass_kernel_from_candidates(possible_kernels, possible_predYs, possible_errors)
    return kernel, predY, error

class CUTLASS_CM(BASE_CM):
    def __init__(self,):
        super(CUTLASS_CM, self).__init__()    
        self.cm_name = "CUTLASS COST Model"
        self.cm_dir = os.path.join(os.path.join(PROJECT_DIR, "_cache"), "cutlass_cms")
        if not os.path.exists(self.cm_dir):
            os.makedirs(self.cm_dir)
        self.load()
        
    def gen_cm_key(self, kernel):
        return "-".join([kernel])

    def inference(self, testX, source_gpu, target_gpu, kernel, op_type, method=-1, verbose=False):
        '''Test the cost model for a specific kernel and method'''
        ### Input shape: testX=(n_sample, n_dim)
        cm_key = self.gen_cm_key(kernel)
        file_key = self.gen_file_key(op_type, source_gpu, target_gpu)
        return self.inference_iplt(testX, op_type, file_key, cm_key, method=method, verbose=verbose)

    def training(self, trainX, trainY, testX, testY, 
            source_gpu, target_gpu, 
            op_type, kernel):
        ''' Train the cost model using different method for one kernel
            Input shape: trainX=(n_sample, n_dim)
        '''
        ### X A = Y ==> A = inverse(X) Y
        print(trainX.shape, testX.shape)
        cm_key = self.gen_cm_key(kernel)
        file_key = self.gen_file_key(op_type, source_gpu, target_gpu)
        return self.training_iplt(trainX, trainY, testX, testY, op_type, file_key, cm_key)
    
    def train_evaluate_one_kernel(self, trainX, trainY, testX, testY,
            kernel, op_type,
            source_gpu="Tesla_V100-SXM2-32GB",
            target_gpu="A100-SXM4-40GB",
            group_dims=None,
            scaler=None,
            check_one_kernel=False
            ):
        ''' Train and evaluate the Cost Model for one specific kernel

        Parameters
        -----------
        trainX, trainY, testX, testY: numpy.ndarray
            Training and test data. The shape of trainX = (n_sample, n_dim)
        op_type: str
            Op type, MatMul or Conv2D
        scaler: class Scaler
            Store the maximum value of each dimension, used to normalize features
        check_one_kernel: boolean
            If it's True, only test one CUTLASS kernel, used for debug
        '''

        print("\n{}: {} training sampels with feature size of {}, {} test samples".format(
            kernel, trainX.shape[0], trainX.shape[1], testX.shape[0]))
         
        cm_key = self.gen_cm_key(kernel)
        file_key = self.gen_file_key(op_type, source_gpu, target_gpu)
        self.init_cm(file_key, cm_key, scaler=scaler)

        ### Train the cost model with different fitting functions
        mape, method_list = self.training(
            trainX, trainY, testX, testY,
            source_gpu, target_gpu,
            op_type, kernel)

        if check_one_kernel:
            self.case_study(trainX, trainY, testX, testY, op_type,
                source_gpu, target_gpu, file_key, cm_key, method_list, mape)
            
        return mape, method_list

    def train_evaluate_all_kernels(self,
            kernel_data_dict, 
            setting,
            mape_rst_path,
            force_fit=False,
            check_one_kernel=False):
        ''' Train and evaluate the Cost Model for each kernel

        Parameters
        -----------
        kernel_data_dict: dict
            Raw features grouped by CUTLASS kernels
        setting: class Setting
            Arguments
        mape_rst_path: str
            Path to cache the tested MAPE results
        force_fit: boolean
            If it's True, force to training the cost models no matter whether cached results exist
        check_one_kernel: boolean
            If it's True, only test one CUTLASS kernel, used for debug
        '''
        if os.path.exists(mape_rst_path) and not force_fit:
            with open(mape_rst_path, 'rb') as fp:
                mape_list, method_list, all_kernels = pickle.load(fp)
            xy_by_kernel = None
        else:
            target_op_type = kernel_type2op_type(setting.kernel_type)
            ### Generate training and test data for current setting 
            # (for a specific pair of source gpu and target gpu)
            # NOTE: the result features are enriched with flops, size, 
            # transform, ai and perf and can not be indexed using FULL_HEADERS
            st = time.time()
            xy_by_kernel, scaler = collect_data_two_gpu_cutlass(
                    setting.source_gpu,
                    setting.target_gpu, 
                    kernel_data_dict,
                    target_op_type=target_op_type,
                    target_kernel_type=setting.kernel_type,
                    ave_lower_bound=setting.ave_lower_bound)
            print("Take {:.3f} s to generate training and test data.".format(time.time() - st))

            ### Train and evaluate the cost model kernel-by-kernel
            mape_list = []
            method_list = None
            all_kernels = list(xy_by_kernel.keys())
            for kernel in all_kernels:
                
                ### The debug_kernel_list is used to check some specific kernels
                if setting.debug_kernel_list is not None:
                    is_ignore = True
                    for debug_kernel in setting.debug_kernel_list:
                        if kernel.startswith(debug_kernel):
                            is_ignore = False
                            break
                    if is_ignore:
                        continue
            
                trainX, trainY, testX, testY = xy_by_kernel[kernel]
                mape, method_list = self.train_evaluate_one_kernel(
                    trainX, trainY, testX, testY, 
                    kernel, target_op_type,
                    setting.source_gpu, setting.target_gpu,
                    scaler=scaler,
                    check_one_kernel=check_one_kernel)
                mape_list.append(mape)

            if setting.debug_kernel_list is not None:
                exit(0)

            if len(mape_list) == 0:
                raise ValueError("Empty mape result")
            ### shape = (# of kernels, # of methods)
            mape_list = np.array(mape_list)
            if not setting.debug_kernel_list:
                with open(mape_rst_path, 'wb') as fp:
                    pickle.dump([mape_list, method_list, all_kernels], fp)
        
        if xy_by_kernel is None:
            target_op_type = kernel_type2op_type(setting.kernel_type)
            xy_by_kernel, scaler = collect_data_two_gpu_cutlass(
                    setting.source_gpu,
                    setting.target_gpu, 
                    kernel_data_dict,
                    target_op_type=target_op_type,
                    target_kernel_type=setting.kernel_type,
                    ave_lower_bound=setting.ave_lower_bound)
        
        method_list.append("tile_based")
        NEIGHBOR_NUM_LIST = [2, 3, 4]
        # NEIGHBOR_NUM_LIST = [2]
        for n_neighbors in NEIGHBOR_NUM_LIST:
            method_list.append(f"Cross-kernel, {n_neighbors} nbr")
    
        mape_cross_kernel_list = []
        for kernel in all_kernels:
            trainX, trainY, testX, testY = xy_by_kernel[kernel]
            origin_pred_time = self.inference(
                testX, setting.source_gpu, setting.target_gpu,
                kernel, target_op_type, method=-1)
            origin_mape = 100 * self.prediction_error(testY, origin_pred_time)
            mape_cross_kernel = [origin_mape]
            for n_neighbors in NEIGHBOR_NUM_LIST:
                try:
                    _, mape_per_kernel = self.cross_kernel_prediction(
                        testX, testY, kernel,
                        setting.source_gpu, setting.target_gpu,
                        target_op_type,
                        n_neighbors=n_neighbors)
                except:
                    raise
                print("Prediction Error: {:.3f} % -> KNN: {:.3f} %".format(
                    origin_mape, mape_per_kernel))
                mape_cross_kernel.append(mape_per_kernel)
            mape_cross_kernel_list.append(mape_cross_kernel)
        mape_list = np.concatenate((mape_list, np.array(mape_cross_kernel_list)), axis=1)
               
        return mape_list, method_list
    
    def ret_kernel_features(self, file_key, source_gpu, target_gpu, target_kernel_feature=None):
        cm_dict = self.CM_DICT[file_key]
        kernel_features = []
        fitlered_kernels = []
        for cm_key in cm_dict.keys():

            _kernel = self.decode_cm_key(cm_key)[0]
            _, _, _source_gpu, _target_gpu = self.decode_file_key(file_key)
            if source_gpu != _source_gpu or target_gpu != _target_gpu:
                continue

            _feature = gen_kernel_feature(_kernel)
            ### filter some kernels here
            if target_kernel_feature is not None and \
                    allow_cross_kernel_pred(target_kernel_feature, _feature):
                kernel_features.append(_feature)
                fitlered_kernels.append(_kernel)
        
        kernel_features = np.array(kernel_features)
        fitlered_kernels = np.array(fitlered_kernels)
        return fitlered_kernels, kernel_features

    def gen_knn_kernels(self,
            target_kernel,
            file_key,
            source_gpu,
            target_gpu,
            n_neighbors=3):

        target_kernel_feature = gen_kernel_feature(target_kernel)
        fitlered_kernels, kernel_features = self.ret_kernel_features(
            file_key, source_gpu, target_gpu, target_kernel_feature=target_kernel_feature)
        
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(norm_kernel_feature(kernel_features))
        distance, nbr_indxs = neigh.kneighbors([norm_kernel_feature(target_kernel_feature)], return_distance=True)
        ### The shape of output = (# of queries, # of neighbors)
        # since we query one kernel each time, so we directly use
        # the index 0
        return list(fitlered_kernels[nbr_indxs[0]]), list(distance[0])

    def check_exist_in_cm(self, source_gpu, target_gpu, op_type, kernel):
        cm_key = self.gen_cm_key(kernel)
        file_key = self.gen_file_key(op_type, source_gpu, target_gpu)
        return self.check_exist_in_cm_iplt(file_key, cm_key)

    def cross_kernel_prediction(self,
            testX,
            testY,
            target_kernel,
            source_gpu,
            target_gpu,
            target_op_type,
            n_neighbors=3,
            allow_same_kernel=False,
            verbose=False):

        ### Step 1: neighbor selection
        file_key = self.gen_file_key(target_op_type, source_gpu, target_gpu)
        try:
            knn_kernels, distance = self.gen_knn_kernels(target_kernel, file_key, source_gpu, target_gpu, n_neighbors=n_neighbors)
        except:
            return None, 100 * 100

        ### TODO (huhanpeng): for debugging
        if not allow_same_kernel and target_kernel in knn_kernels:
            idx = knn_kernels.index(target_kernel)
            knn_kernels.pop(idx)
            distance.pop(idx)
        
        if verbose:
            print(f"\nKernel: {target_kernel}, {n_neighbors} neighbors:")
            for idx, _kernel in enumerate(knn_kernels):
                print("\t* {}, d={:.3f}".format(_kernel, distance[idx]))

        ### Step 2: cross-kernel prediction
        if self.average_output:
            ### average the model output of neighbouring kernels
            pred_times_by_refer = []
            # raw_testX = [parse_raw_feature(elem) for elem in testX]
            for ref_kernel in knn_kernels:
                
                # _testX = [enrich_raw_feature(elem,
                #     target_op_type,
                #     ave=elem[0],
                #     source_gpu=source_gpu,
                #     target_gpu=target_gpu,
                #     kernel=ref_kernel) for elem in raw_testX]
                _testX = testX
                pred_times_by_refer.append(
                    self.inference(
                        _testX,
                        source_gpu,
                        target_gpu,
                        ref_kernel,
                        target_op_type,
                        method=-1,
                        verbose=verbose))

            pred_time_by_refer_ = idw_average(pred_times_by_refer, distance=distance)
        else:
            ### average the model parameters first then do inference
            cm_dict = self.CM_DICT[file_key]
        
            method_list = [6, 7, 8]
            max_time = None
            for _method in method_list:
                ### Average model parameters
                para_list = []
                for _kernel in knn_kernels:
                    cm_key = self.gen_cm_key(_kernel)
                    para = cm_dict[cm_key][_method]["para"].x
                    para_list.append(para)
                para_array = idw_average(para_list, distance=distance)

                ### Do inference
                pred_time = self.inference_internal(testX, _method, para_array, target_op_type)
                
                # if max_time is not None:
                #     print(max_time - pred_time)
                if max_time is None or np.average(pred_time) > np.average(max_time):
                    max_time = pred_time
            pred_time_by_refer_ = max_time

        if testY is None:
            mape_refer = 1000
        else:
            mape_refer = self.prediction_error(testY, pred_time_by_refer_)
        
        return pred_time_by_refer_, mape_refer * 100

    def cross_kernel_similarity(self, kernel_type, source_gpu, target_gpu):
        target_op_type = kernel_type2op_type(kernel_type)
        file_key = self.gen_file_key(target_op_type, source_gpu, target_gpu)
        cm_dict = self.CM_DICT[file_key]
        method_list = [6, 7, 8]
        model_para_list = [[], [], []]
        for kernel_idx, cm_key in enumerate(cm_dict.keys()):
            if not ("cutlass_tensorop_s884gemm_f16_64x64" in cm_key):
                continue
            
            print(cm_key)
            
            for idx, _method in enumerate(method_list):

                para = cm_dict[cm_key][_method]["para"].x
                if kernel_idx == 320 or kernel_idx == 321:
                    print(para)
                model_para_list[idx].append(para)
                
        fig = plt.figure(figsize=(16, 12))
        fig_base = 220

        for idx in range(len(method_list)):
            fig_idx = idx + 1
            ax = fig.add_subplot(fig_base+fig_idx)
            ax = sns.heatmap(model_para_list[idx], 
                # cmap="RdBu_r"
                )
            plt.xlabel("Theta of \'{}\'".format(COST_FUNC_LIST[method_list[idx]][0]), fontsize=16)
            plt.ylabel("Kernels", fontsize=16)
            plt.title("Model Parameters Comparison", fontsize=16)
            # ax = sns.heatmap(model_para_list[0], mask=mask, cmap="YlGnBu")
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_DIR, "_fig/similarity.png"))