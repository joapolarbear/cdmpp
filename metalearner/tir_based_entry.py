import os
import numpy as np

from utils.util import notify, task_repr, sample_task_files, LARGEST_COST, TrainTestPair
from utils.env import PROJECT_CFG

import metalearner.learner.recur_lstm_learner as ast_lstm
from metalearner.data.dataloader import (
    load_iter_dataset,
    load_raw_data_w_cluster_id,
    group_raw_data,
    rawdata2dataset_pair,
)
from metalearner.feature import (
    ALL_FEATURE_TYPE,
    init_fea_info_via_data,
    init_fea_info,
    is_feature_type
)
from .learner import _metric_learning_w_tir_data_impl, TaskRuntimeCache, ret_learner

def dataset_cost_range(learner, files_to_test, learning_params, verbose, data_meta_info):
    ds_pair, data_meta_info = load_iter_dataset(learning_params)

    import torch
    min_cost = LARGEST_COST
    max_cost = 0
    xy_list, sample_cnt = learner.dataset2xylist(ds_pair.train)
    for _, batch_y in xy_list:
        de_std_batch_y = learner.data_meta_info.de_standardize_output(batch_y)
        min_cost = min(min_cost, float(torch.min(de_std_batch_y)))
        max_cost = max(max_cost, float(torch.max(de_std_batch_y)))
    xy_list, sample_cnt = learner.dataset2xylist(ds_pair.val)
    for _, batch_y in xy_list:
        de_std_batch_y = learner.data_meta_info.de_standardize_output(batch_y)
        min_cost = min(min_cost, float(torch.min(de_std_batch_y)))
        max_cost = max(max_cost, float(torch.max(de_std_batch_y)))
    return min_cost, max_cost

def cross_task_learning(trace_root_path, learning_params, tune, verbose=True):

    # from utils.util import test_sample_task_files
    # test_sample_task_files(trace_root_path, learning_params)
    # raise

    ds_pair, data_meta_info = load_iter_dataset(learning_params)

    if ds_pair is None:
        return -1
    if tune:
        from metalearner.auto_tune import auto_tune
        return auto_tune(ds_pair, data_meta_info, learning_params)
    else:
        return _metric_learning_w_tir_data_impl(ds_pair, data_meta_info, learning_params, verbose=verbose)

def learn_by_task(trace_root_path, learning_params, tune, sample_file_num=None):
    ''' Train a cost model for each task file
    '''
    ### Debug: set true to debug file by file
    debug_files = None
    
    if debug_files is None:
        files_to_test = sample_task_files(trace_root_path, learning_params["mode"], 
                learning_params["gpu_model"], absolute_path=True)[learning_params["gpu_model"]]["tasks"]
        task_runtime_cache = TaskRuntimeCache(".workspace/cm", PROJECT_CFG["cfg_name"])
    else:
        files_to_test = debug_files
        task_runtime_cache = None
        
    assert not tune
    total_files_num = len(files_to_test)
    for file_idx, _file in enumerate(files_to_test):
        _cur_files = [_file]
        print(f"Evaluating {_file} ...")

        file_grp_id = task_runtime_cache.get_file_group_id(_cur_files) if debug_files is None else None
        if debug_files is not None or not task_runtime_cache.contain(file_grp_id):
            raise DeprecationWarning("Not adapted to latest cross_task_learning yet")
            ret = cross_task_learning(_cur_files, learning_params, tune,
                verbose=(debug_files is not None))
            if ret == -1:
                print(f"Skip {_file}")
                if task_runtime_cache:
                    task_runtime_cache.record(file_grp_id, {"mape_val": None, "mape_train": None})
                    task_runtime_cache.save()
                continue
            else:
                assert ret is not None
                learner, (metrics_val, metrics_train) = ret

                if task_runtime_cache:
                    mape_val = float(metrics_val["mape"]) if metrics_val else -1
                    mape_train = float(metrics_train["mape"]) if metrics_train else -1
                    task_runtime_cache.record(file_grp_id, {"mape_val": float(mape_val),
                                                                "mape_train": float(mape_train)})
                    task_runtime_cache.save()

        if task_runtime_cache:
            notify(f"file {_cur_files} ({file_idx}/{total_files_num}), "
                f"mape_val={task_runtime_cache[file_grp_id]['mape_val']:.12f}, "
                f"mape_train={task_runtime_cache[file_grp_id]['mape_train']:.12f}\n")
        
        if debug_files:
            input("Continue?")

def group_task_learning(trace_root_path, learning_params, task_sample_num=10):
    ''' Perform clustering-based cross-task learning based on a subset of task files
    '''
    config_name = f"{PROJECT_CFG['cfg_name']}"
    max_step = 160e3
    cluster_method = "kmeans"
    # max_step = 3

    raw_data, cluster_rst, files_to_test = load_raw_data_w_cluster_id(
        trace_root_path,
        learning_params,
        task_sample_num=task_sample_num,
        cluster_method=cluster_method,
    )

    def __inner_learning_func(_raw_data, _cm_save_dir=None):
        _learning_params = learning_params.copy()
        _learning_params.update({
            "cache_dir": _cm_save_dir,
            "test_data": None,
            "max_epoch": None,
            "max_step": max_step
        })

        ds_pair = rawdata2dataset_pair(_raw_data, verbose=False)
        learner, (metrics_val, metrics_train) = _metric_learning_w_tir_data_impl(
            ds_pair,
            _raw_data.metainfo,
            _learning_params,
            verbose=False)
        return metrics_val["mape"], metrics_train["mape"]

    ### Single task learning

    ### Load the cached learning rst
    task_runtime_cache = TaskRuntimeCache(".workspace/cm", cfg_name=PROJECT_CFG["cfg_name"])

    '''
    ### First, perform single-task learning
    for _file in files_to_test:
        _repr = task_runtime_cache.get_file_group_id([_file])
        if not task_runtime_cache.contain(_repr):
            ds_pair, data_meta_info = load_iter_dataset(learning_params)
            if ds_pair is None:
                error = None
                continue
            else:
                learner, (metrics_val, metrics_train) = _metric_learning_w_tir_data_impl(
                        ds_pair,
                        data_meta_info,
                        learning_params,
                        verbose=False)
            mape_val, mape_train = float(metrics_val["mape"]), float(metrics_train["mape"]) if metrics_train is not None else None
            task_runtime_cache.record(_repr,
                {"mape_val": float(mape_val),
                    "mape_train": float(mape_train)})
            
        __inner_nofify("_file {}, mape_val={}, mape_train={}\n".format(
            _file,
            task_runtime_cache[_repr]["mape_val"],
            task_runtime_cache[_repr]["mape_train"]))

    task_runtime_cache.save()

    ### Then, perform cross-task learning
    _repr = task_runtime_cache.get_file_group_id(files_to_test)
    if not task_runtime_cache.contain(_repr):
        mape_val, mape_train = __inner_learning_func(raw_data)
        task_runtime_cache.record(_repr,
            {"mape_val": float(mape_val),
                "mape_train": float(mape_train)})
        task_runtime_cache.save()
    __inner_nofify("Cross-task, mape_val={:.12f}, mape_train={:.12f}".format(
        task_runtime_cache[_repr]["mape_val"],
        task_runtime_cache[_repr]["mape_train"]))
    '''
    ### Fiannly, perform grouped-task learning
    group_idx_dict = group_raw_data(cluster_rst)
    _cross_repr = task_repr(files_to_test)
    _cluster_task_save_dir = os.path.join(".workspace", "verify", config_name, "cross_task", _cross_repr, f"{cluster_method}-{len(group_idx_dict)}")
    os.makedirs(_cluster_task_save_dir, exist_ok=True)
    print(f"store the results at {_cluster_task_save_dir}")

    group_log_path = os.path.join(_cluster_task_save_dir, "group_log.txt")
    group_log_fp = open(group_log_path, "a")
    def __inner_nofify(_str):
        notify(_str)
        group_log_fp.write(_str + "\n")
        group_log_fp.flush()

    for cluster_id in sorted(group_idx_dict):
        _save_dir = os.path.join(_cluster_task_save_dir, f"{cluster_id}")
        _cm_save_dir = os.path.join(_save_dir, "cm")
        _rst_save_dir = os.path.join(_save_dir, "rst")
        os.makedirs(_cm_save_dir, exist_ok=True)
        os.makedirs(_rst_save_dir, exist_ok=True)

        group_xydata = raw_data.subset(np.array(group_idx_dict[cluster_id]))
        mape_val, mape_train = __inner_learning_func(group_xydata, _cm_save_dir)
        __inner_nofify("Cluster {}, mape_val={}, mape_train={}".format(
            cluster_id, mape_val, mape_train))
    
    group_log_fp.close()

def entry_init(learning_params):
    if learning_params["input"] is not None:
        trace_root_path = os.path.abspath(learning_params["input"])
    else:
        raise ValueError("No input file is specified")
 
    init_fea_info_via_data(trace_root_path)

    return trace_root_path

def learn_entry(learning_params:dict, tune=False):
    '''
    Sample:

    # Training
    bash scripts/train.sh run --mode sample10 -i .workspace/ast_ansor --tb_logdir .workspace/runs/test5
    
    '''
    # import pdb;pdb.set_trace()
    trace_root_path = entry_init(learning_params)
    if learning_params["mode"] == "single":
        if learning_params["tiramisu"]:
            assert is_feature_type(ALL_FEATURE_TYPE.ast_ansor)
            raise NotImplementedError()
        else:
            learn_by_task(trace_root_path, learning_params, tune)
    elif learning_params["mode"] == "group":
        if learning_params["tiramisu"]:
            assert is_feature_type(ALL_FEATURE_TYPE.ast_ansor)
            raise NotImplementedError()
        else:
            group_task_learning(trace_root_path, learning_params, task_sample_num = 10)
    else:
        if learning_params["tiramisu"]:
            assert is_feature_type(ALL_FEATURE_TYPE.ast_ansor)
            files_to_test = sample_task_files(trace_root_path, learning_params["mode"], 
                learning_params["gpu_model"], absolute_path=True)[learning_params["gpu_model"]]["tasks"]
            ast_lstm.test_tiramisu(files_to_test, learning_params)
        elif learning_params["pipeline"]:
            from .pipeline import pipeline_learning
            pipeline_learning(learning_params)
        else:
            cross_task_learning(trace_root_path, learning_params, tune)

def analyze_entry(learning_params):
    '''
    Sample:
        bash scripts/train.sh analyze --mode sample10 -i .workspace/ast_ansor --tb_logdir .workspace/runs/test5
        bash scripts/train.sh analyze --mode [t4_0, t4_1007.npy] -i .workspace/ast_ansor --tb_logdir .workspace/runs/test5
    '''
    
    trace_root_path = entry_init(learning_params)
    
    options = learning_params["option"].split(",")
    print(options)
    if len(options) > 1:
        if options[1] == "clustering":
            from metalearner.analyze.data_analyze import clustering_analysis
            clustering_analysis(trace_root_path, learning_params)
        elif options[1] == "x2y_mapping":
            from metalearner.analyze.data_analyze import verify_x2y_mapping
            verify_x2y_mapping(trace_root_path, learning_params)
        elif options[1] == "x_device2y":
            from metalearner.analyze import plot_x_device2y
            plot_x_device2y(trace_root_path, learning_params)
        elif options[1] == "cross_device_similarity":
            from metalearner.analyze import cross_device_similarity
            cross_device_similarity(learning_params)
        elif options[1] == "device_data":
            from metalearner.analyze import (device_data_learnability_compare,
                device_data_analyze_via_dim_reduction) 
            device_data_learnability_compare(trace_root_path, learning_params)
            device_data_analyze_via_dim_reduction(trace_root_path, learning_params)
        elif options[1] == "sample_tasks":
            ### sample task analyze
            from metalearner.analyze.sample4finetune import sample_tasks2finetune
            sample_tasks2finetune(trace_root_path, learning_params)
        elif options[1] == "network_data":
            ### Network analyze
            import metalearner.analyze.network2files as net_analyze
            net_analyze.entry(trace_root_path, learning_params)
        elif options[1] == "leaf_no":
            from metalearner.analyze.ast_data_analysis import check_ast_node_dist
            device2task = sample_task_files(learning_params["input"], learning_params["mode"],
                    learning_params["gpu_model"], absolute_path=True)
            check_ast_node_dist(device2task[learning_params["gpu_model"]]["tasks"], learning_params)
        else:
            raise ValueError(options)
        exit(0)

    ### Analyze the training and test results
    if learning_params["tiramisu"]:
        assert is_feature_type(ALL_FEATURE_TYPE.ast_ansor)
        raise NotImplementedError()
    else:
        from metalearner.clustering import LearnerClusterHub
        from metalearner.learner import parse_metric_learner
        from metalearner.analyze import (
            analyze_train_test_rst,
            multitask_train_rst,
        )
        learner_cls = parse_metric_learner(learning_params["metric_learner"])
        learner = learner_cls.load_init(learning_params["cache_dir"])
        print("[Analyze] Load training and test data ..")
        ds_pair, data_meta_info = load_iter_dataset(learning_params)

        ana_train_rst = True
        if ana_train_rst:
            rst_path_pair = TrainTestPair(
                os.path.join(learner.cache_path, "training.pickle"), os.path.join(learner.cache_path, "test.pickle"))
            latent_path_pair = TrainTestPair(
                os.path.join(learner.cache_path, "training_latent.pickle"), os.path.join(learner.cache_path, "test_latent.pickle"))
            if not all([
                os.path.exists(rst_path_pair.train), os.path.exists(rst_path_pair.val),
                os.path.exists(latent_path_pair.train), os.path.exists(latent_path_pair.val)
                ]):
                
                metrics_train, train_rst_path = learner.test_on_dataset(ds_pair.train, "training")
                metrics_val, test_rst_path = learner.test_on_dataset(ds_pair.val, "test")

            raise NotImplementedError("Please refer to script/cmd2error.sh")
            analyze_train_test_rst(rst_path_pair, latent_path_pair)
        else:
            device2task = sample_task_files(trace_root_path, learning_params["mode"], 
                learning_params["gpu_model"], absolute_path=True)
            # random.shuffle(files_to_test)
            multitask_train_rst(device2task[learning_params["gpu_model"]]["tasks"],
                learner, learning_params)

def end2end_replay(learning_params):
    from metalearner.learner import parse_metric_learner
    from metalearner.clustering import LearnerClusterHub

    assert learning_params["cache_dir"] is not None, \
        f"Cost Model path must be given, but {learning_params['cache_dir']} is provided"

    ### TODO (huhanpeng): for simplicity, use ast_ansor as the feature type by default 
    init_fea_info(ALL_FEATURE_TYPE.ast_ansor)

    ### Load the trained cost model
    learner_cls = parse_metric_learner(learning_params["metric_learner"])
    learner_hub = LearnerClusterHub(learning_params["cache_dir"], learner_cls)

    from end2end import estimaet_network, measure_network, measure_by_task, offline_preprocess

    if learning_params["networks"] is None:
        networks = [
            # 'resnet_18',
            'resnet_50',
            # 'resnet3d_18',
            # 'mobilenet_v2', 'mobilenet_v3',
            # 'wide_resnet_50', 'resnext_50',
            # 'densenet_121',
            # 'inception_v3',
            # 'bert_tiny',
            # 'bert_base',
            # 'bert_medium', 'bert_large',
            # 'dcgan',
        ]
    else:
        networks = learning_params["networks"].split(",")

    if learning_params["batch_sizes"] is None:
        batch_sizes = [1, 4, 8]
    else:
        batch_sizes = [eval(bs) for bs in learning_params["batch_sizes"].split(",")]

    import tvm
    from utils.gpu_utils import get_gpu_name
    from utils.device_info import query_cc
    target_gpu = learning_params['gpu_model']
    capability = query_cc(target_gpu)
    tvm_target = tvm.target.cuda(model=target_gpu.lower(), arch=f"sm_{capability}")

    print(f"\n\nEnd2end experiments on {networks} with bs = {batch_sizes} on {tvm_target}")

    for _replay_mode in learning_params["replay_mode"].split(","):
        if _replay_mode in ["measure", "breakdown", "measure_by_task", "replay_via_profile"]:
            gpu_model = get_gpu_name()[0].lower()
            assert gpu_model == tvm_target.model, \
                f"{tvm_target.model} is required, but {gpu_model} is used"
            break

    if False:
        ### Use the min_cost and max_cost as a reference to 
        # select schedules for the target network
        raise NotImplementedError()
        trace_root_path = os.path.abspath(learning_params["input"])
        root_path, files_to_test = sample_task_files(trace_root_path, learning_params, 
                    abs=True)[learning_params["gpu_model"]]
        min_cost, max_cost = dataset_cost_range(learner, files_to_test,
                learning_params, verbose=False, data_meta_info=learner.data_meta_info)
        print(min_cost, max_cost)
        PROJECT_CFG["cost_range"] = (min_cost, max_cost)

    for _replay_mode in learning_params["replay_mode"].split(","):
        for network in networks:
            for batch_size in batch_sizes:
                network_args = {"network": network, "batch_size": batch_size}
                if _replay_mode == "measure":
                    print(f"\nMeasure the perfomrance of {network}, BS={batch_size}")
                    measure_network(network_args, tvm_target)
                elif _replay_mode == "breakdown":
                    print(f"\nBreakdown the perfomrance of {network}, BS={batch_size} and plot the timeline")
                    measure_network(network_args, tvm_target, breakdown=True)
                elif _replay_mode == "measure_by_task":
                    raise ValueError("[Deprecated] please use --replay_mode=replay_via_profile instead")
                    print(f"\nMeasure the perfomrance of {network} by TASK, BS={batch_size}")
                    measure_by_task(network_args, tvm_target)
                elif _replay_mode == "prepare":
                    ### Prepare task, task_weights and states for estimation
                    offline_preprocess(network_args, tvm_target)
                elif _replay_mode == "test_by_task":
                    print(f"\nTest learner by task of {network}, BS={batch_size}")
                    from end2end import test_learner_on_network
                    test_learner_on_network(
                        network_args, tvm_target,
                        is_ast_feature=is_feature_type(ALL_FEATURE_TYPE.ast_ansor),
                        learner_hub=learner_hub)
                elif _replay_mode == "replay_via_profile":
                    ### Estimation via profile
                    print(f"\nEvaluate the perfomrance of {network}, BS={batch_size} via profile")
                    estimaet_network(
                        network_args, tvm_target,
                        is_ast_feature=is_feature_type(ALL_FEATURE_TYPE.ast_ansor),
                        learner_hub=learner_hub, via="profile")
                elif _replay_mode == "replay_via_tenset":
                    ### Estimation via profile
                    print(f"\nEvaluate the perfomrance of {network}, BS={batch_size} via tenset cost")
                    estimaet_network(
                        network_args, tvm_target,
                        is_ast_feature=is_feature_type(ALL_FEATURE_TYPE.ast_ansor),
                        learner_hub=learner_hub, via="tenset")
                elif _replay_mode == "replay":
                    ### Estimation
                    print(f"\nEvaluate the perfomrance of {network}, BS={batch_size}")
                    estimaet_network(
                        network_args, tvm_target,
                        is_ast_feature=is_feature_type(ALL_FEATURE_TYPE.ast_ansor),
                        learner_hub=learner_hub)
                else:
                    raise ValueError("learning_params['replay_mode']")

def test_entry(learning_params):

    trace_root_path = entry_init(learning_params)

    files_to_test = sample_task_files(trace_root_path, learning_params["mode"], 
                learning_params["gpu_model"], absolute_path=True)[learning_params["gpu_model"]]["tasks"]

    from metalearner.test.test_dataset import (
        test_dataset_and_iterable_dataset,
        test_load_raw_data_cost,
        test_load_raw_data_freeze_cost)
    test_dataset_and_iterable_dataset(files_to_test, learning_params)
    # test_load_raw_data_cost(files_to_test, learning_params)
    # test_load_raw_data_freeze_cost(files_to_test, learning_params)

def sche_search_entry(learning_params):
    ### TODO (huhanpeng): for simplicity, use ast_ansor as the feature type by default 
    init_fea_info(ALL_FEATURE_TYPE.ast_ansor)

    from metalearner.to_ansor import (
        test_sketch_search_policy_cdmppmodel,
        sche_search
    )
    # test_sketch_search_policy_cdmppmodel(learning_params)
    sche_search(learning_params)