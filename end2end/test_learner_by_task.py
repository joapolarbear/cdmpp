import numpy as np
import pickle

from tvm_helper.tir_helper import (
    wrap_get_per_store_features_from_measure_pairs,
    parse_cost
)
from metalearner.data.rawdata import ASTRawData
from metalearner.data.dataloader import MyDataSet
from metalearner.learner import BaseLearner

# Tenset
from common import get_task_info_filename

from .preprocess import get_valid_inp_results_from_tenset
from .util import get_network_key

def test_learner_on_one_task(task_id, learner, task, target):
    filter_input_result_list = get_valid_inp_results_from_tenset(task, target)

    if len(filter_input_result_list) == 0:
        print(f"\nVerify the learner on task {task_id} ({task.workload_key[:20]}...): (no state in the required cost range)")
        return

    inputs, results, dur_list = zip(*filter_input_result_list)

    ### features_.shape = [std, flops, sample_num, <std, flop, ast_features, node_ids, serialized_tree>]
    features, tmp_ys = wrap_get_per_store_features_from_measure_pairs(
            inputs, results,
            get_workload_embedding=False,
            verbose=False,
            parse_ast=True)

    xydata = np.concatenate((np.expand_dims(tmp_ys, 1), features), axis=1)
    raw_data = ASTRawData(xydata, learner.data_meta_info)
    dataset = MyDataSet(raw_data)
    if len(dataset) == 0:
        print(f"\nVerify the learner on task {task_id} ({task.workload_key[:20]}...): (no data for required N_leaf)")
        return
    _metrics, _ = learner.test_on_dataset(dataset)
    print(f"\nVerify the learner on task {task_id} ({task.workload_key[:20]}...): {_metrics['mape']}")
    
def test_learner_on_network(network_args, target, learner: BaseLearner, is_ast_feature=True):
    if not is_ast_feature:
        raise NotImplementedError()
    # Parse tasks of the network
    network_key = get_network_key(network_args)
    task_info_filename = get_task_info_filename(network_key, target)
    print(f"Load tasks from Tenset: {task_info_filename} for {network_key} {target}")
    tasks, task_weights = pickle.load(open(task_info_filename, "rb"))

    for task_id, task in enumerate(tasks):
        test_learner_on_one_task(task_id, learner, task, target)