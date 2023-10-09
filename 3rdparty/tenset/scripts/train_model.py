import argparse
import logging
import pickle
import random

import torch
import numpy as np

import tvm
from common import load_and_register_tasks, get_task_info_filename

from tvm.auto_scheduler.utils import to_str_round
from tenset_cost_model.metric import (
    metric_rmse,
    metric_r_squared,
    metric_pairwise_comp_accuracy,
    metric_top_k_recall,
    metric_peak_score,
    metric_mape,
    random_mix,
)
from tenset_cost_model.cost_model import XGBModelInternal, RandomModelInternal


def evaluate_model(model, test_set):
    # make prediction
    prediction = model.predict(test_set)

    # compute weighted average of metrics over all tasks
    tasks = list(test_set.tasks())
    weights = [len(test_set.throughputs[t]) for t in tasks]
    print("Test set sizes:", weights)

    
    rmse_list = []
    r_sqaured_list = []
    pair_acc_list = []
    peak_score1_list = []
    peak_score5_list = []
    # add mape
    mape_list = []
    mape_avg_list = []

    for task in tasks:

        ### Calculate flop_cnt
        file_name = get_task_info_filename(task.workload_key, tvm.target.Target(task.target))
        file_name = file_name.replace("network_info", "to_measure_programs").replace("task.pkl", "json")
        inputs, _ = tvm.auto_scheduler.RecordReader(file_name).read_lines()
        search_task = tvm.auto_scheduler.measure.recover_measure_input(inputs[0]).task
        flop_ct = search_task.compute_dag.flop_ct

        preds = prediction[task]
        labels = test_set.throughputs[task]

        rmse_list.append(np.square(metric_rmse(preds, labels)))
        r_sqaured_list.append(metric_r_squared(preds, labels))
        pair_acc_list.append(metric_pairwise_comp_accuracy(preds, labels))
        peak_score1_list.append(metric_peak_score(preds, labels, 1))
        peak_score5_list.append(metric_peak_score(preds, labels, 5))
        # add mape
        mape_list.append(metric_mape(preds, labels))
        mape_avg_list.append(metric_mape(flop_ct/preds, flop_ct/labels))

    rmse = np.sqrt(np.average(rmse_list, weights=weights))
    r_sqaured = np.average(r_sqaured_list, weights=weights)
    pair_acc = np.average(pair_acc_list, weights=weights)
    peak_score1 = np.average(peak_score1_list, weights=weights)
    peak_score5 = np.average(peak_score5_list, weights=weights)
    # add mape
    mape = np.average(mape_list, weights=weights)
    mape_avg = np.average(mape_avg_list, weights=weights)
    
    eval_res = {
        "RMSE": rmse,
        "R^2": r_sqaured,
        # "mape": mape,
        "mape_avg": mape_avg,
        "pairwise comparision accuracy": pair_acc,
        "average peak score@1": peak_score1,
        "average peak score@5": peak_score5,
    }
    return eval_res


def make_model(name):
    """Make model according to a name"""
    if name == "xgb":
        return XGBModelInternal()
    # elif name == "mlp":
    #     return MLPModelInternal()
    elif name == "random":
        return RandomModelInternal()
    else:
        raise ValueError("Invalid model: " + name)
 

def train_zero_shot(dataset, train_ratio, model_names, split_scheme):
    # Split dataset
    if split_scheme == "within_task":
        train_set, test_set = dataset.random_split_within_task(train_ratio)
    elif split_scheme == "by_task":
        train_set, test_set = dataset.random_split_by_task(train_ratio)
    elif split_scheme == "by_target":
        train_set, test_set = dataset.random_split_by_target(train_ratio)
    else:
        raise ValueError("Invalid split scheme: " + split_scheme)

    print("Train set: %d. Task 0 = %s" % (len(train_set), train_set.tasks()[0]))
    if len(test_set) == 0:
        test_set = train_set
    print("Test set:  %d. Task 0 = %s" % (len(test_set), test_set.tasks()[0]))

    # Make models
    names = model_names.split("@")
    models = []
    for name in names:
        models.append(make_model(name))

    eval_results = []
    for name, model in zip(names, models):
        # Train the model
        filename = name + ".pkl"
        model.fit_base(train_set, valid_set=test_set)
        print("Save model to %s" % filename)
        model.save(filename)

        # Evaluate the model
        eval_res = evaluate_model(model, test_set)
        print(name, to_str_round(eval_res))
        eval_results.append(eval_res)

    # Print evaluation results
    for i in range(len(models)):
        print("-" * 60)
        print("Model: %s" % names[i])
        for key, val in eval_results[i].items():
            print("%s: %.4f" % (key, val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset.pkl")
    parser.add_argument("--models", type=str, default="xgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-scheme",
        type=str,
        choices=["by_task", "within_task", "by_target"],
        default="within_task",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()
    print("Arguments: %s" % str(args))

    # Setup random seed and logging
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    logging.basicConfig()
    logging.getLogger("auto_scheduler").setLevel(logging.DEBUG)

    print("Load all tasks...")
    load_and_register_tasks()

    print("Load dataset...")
    dataset = pickle.load(open(args.dataset, "rb"))

    train_zero_shot(dataset, args.train_ratio, args.models, args.split_scheme)

