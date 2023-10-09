import os
import sys
import numpy as np
from tvm import auto_scheduler
from tvm.auto_scheduler.measure_record import RecordReader
from tvm.auto_scheduler import XGBModel

from utils.util import warning
from utils.base import PROJECT_DIR
from tvm_helper.tir_helper import wrap_get_per_store_features_from_states
from metalearner.learner import parse_metric_learner
from metalearner.clustering import LearnerClusterHub
from metalearner.data.rawdata import ASTRawData
from metalearner.data.dataloader import MyDataSet

from end2end.preprocess import get_network_tasks

### Required to for cost model loading using pickle.load
sys.path.append(os.path.join(
    PROJECT_DIR, "3rdparty", "tlp", "scripts"))
from tlp_train import AttentionModule

NUM_MEASURE_PER_ROUND = 2
class CDMPPModel(XGBModel):
    """A model that returns random estimation for all inputs"""

    def __init__(self, cache_dir, *args, **kwargs):
        super(CDMPPModel, self).__init__(*args, **kwargs)

        assert cache_dir is not None, \
            f"Cost Model path must be given, but {cache_dir} is provided"

        ### Load the trained cost model
        learner_cls = parse_metric_learner(4)
        self.learner_hub = LearnerClusterHub(cache_dir, learner_cls)

    def update(self, inputs, results):
        """Update the cost model according to new measurement results (training data).

        Parameters
        ----------
        inputs : List[auto_scheduler.measure.MeasureInput]
            The measurement inputs
        results : List[auto_scheduler.measure.MeasureResult]
            The measurement results
        """
        pass

    def predict(self, task, states):

        ### Lowering and extract features
        all_features = wrap_get_per_store_features_from_states(states, 
            task, parse_ast=True, delete_invalid=False)

        ### Group state ids according to the leaf node number
        leaf_no2state_ids = {}
        for state_id, (std, flop_ct, ast_features, node_ids, serialized_tree) in enumerate(all_features):
            if ast_features is None:
                continue
            leaf_no = len(ast_features)
            if leaf_no not in leaf_no2state_ids:
                leaf_no2state_ids[leaf_no] = []
            leaf_no2state_ids[leaf_no].append(state_id)
        
        # Predict -inf for invalid states that failed to be lowered.
        ret = np.ones(len(states), dtype=float) * float("-inf")

        for leaf_no, state_ids in leaf_no2state_ids.items():
            features = all_features[state_ids]

            valid_ret = []
            tmp_ys = np.ones(len(features), dtype=float) * 1
            xydata = np.concatenate((np.expand_dims(tmp_ys, 1), features), axis=1)
            if features[0][1] < 0:
                ### TODO (huhanpeng): how to handle samples with flops = -1
                # estimated_time = 0
                warning(f" - flops = -1, use {super(CDMPPModel, self).__class__.__name__} instead")
                return super(CDMPPModel, self).predict(task, states)
            else:
                learner = self.learner_hub.switch({"leaf_node_no": leaf_no, "FLOPs": features[0][1]})
                if learner is None:
                    ### Fall back to parent cost model for the cases currently not supported
                    warning(f" - Use {super(CDMPPModel, self).__class__.__name__} instead")
                    return super(CDMPPModel, self).predict(task, states)
                else:
                    raw_data = ASTRawData(xydata, learner.data_meta_info, disable_filter=True)
                    dataset = MyDataSet(raw_data)
                    ### We have encode filters into the LearnerClusterHub:learner_hub, 
                    # in valid samples should have been filtered out at learner_hub.switch
                    assert len(dataset) > 0
                    try:
                        input_data, sample_cnt = learner.dataset2xylist(dataset)
                    except:
                        import code; code.interact(local=locals())
                    for X, Y in input_data:
                        X = learner.data_to_train_device(X)[0]
                        estimated_time = learner.predict(X)
                        valid_ret.extend(estimated_time.flatten())
        
            # Convert the predicted time to throughput as scores
            try:
                ret[state_ids] = 1. / np.array(valid_ret)
            except:
                import pdb; pdb.set_trace()

        return ret

    def save(self, file_name: str):
        """Save the model to a file

        Parameters
        ----------
        file_name: str
            The filename
        """
        pass

    def load(self, file_name: str):
        """Load the model from a file

        Parameters
        ----------
        file_name: str
            The filename
        """
        pass



import random
import multiprocessing
import tempfile

import tvm
import tvm.testing
from tvm import auto_scheduler
from tvm import te
from tvm.auto_scheduler.utils import get_const_tuple

def matmul_auto_scheduler_test(N, M, K):
    A = te.placeholder((N, K), name="A")
    B = te.placeholder((K, M), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]),
        name="C",
        attrs={"layout_free_placeholders": [B]},
    )
    return [A, B, C]


class CustomMeasureCallback(auto_scheduler.measure.PythonBasedMeasureCallback):
    """A simple Python-based callback for testing."""

    def callback(self, policy, inputs, results):
        assert isinstance(policy, auto_scheduler.search_policy.SearchPolicy)
        for inp, res in zip(inputs, results):
            assert isinstance(inp, auto_scheduler.MeasureInput)
            assert isinstance(res, auto_scheduler.MeasureResult)

def search_common(
    task=None,
    target="llvm",
    search_policy="sketch",
    runner="local",
    num_measure_trials=100,
    cost_model=auto_scheduler.RandomModel(),
    init_search_callbacks=None,
):
    if task is None:
        task = auto_scheduler.SearchTask(
            func=matmul_auto_scheduler_test, args=(64, 64, 64), target=target
        )
    target = task.target

    print("Test search policy '%s' for '%s'" % (search_policy, target))

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        init_search_callbacks = init_search_callbacks or []
        init_search_callbacks.append(auto_scheduler.PreloadMeasuredStates(log_file))

        if search_policy == "empty":
            search_policy = auto_scheduler.EmptyPolicy(task)
        elif search_policy == "sketch":
            search_policy = auto_scheduler.SketchPolicy(
                task, program_cost_model=cost_model, init_search_callbacks=init_search_callbacks
            )
        else:
            raise ValueError("Invalid policy: " + search_policy)

        # Tune
        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=num_measure_trials,
            num_measures_per_round=NUM_MEASURE_PER_ROUND,
            early_stopping=1,
            runner=runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file), CustomMeasureCallback()],
        )
        task.tune(tuning_options=tuning_options, search_policy=search_policy)

        # Compile with the best schedule
        sch, args = task.apply_best(log_file)
        mod = tvm.build(sch, args, target)

        # Compile with naive schedule for correctness check
        sch, args = task.compute_dag.apply_steps_from_state(task.compute_dag.init_state)
        mod_ref = tvm.build(sch, args, "llvm")

        ctx = tvm.device(str(target), 0)
        np_arrays = [np.random.uniform(size=get_const_tuple(x.shape)).astype(x.dtype) for x in args]

        tvm_arrays = [tvm.nd.array(x, ctx) for x in np_arrays]
        mod(*tvm_arrays)
        actual = [x.asnumpy() for x in tvm_arrays]

        tvm_arrays = [tvm.nd.array(x) for x in np_arrays]
        mod_ref(*tvm_arrays)
        expected = [x.asnumpy() for x in tvm_arrays]

        for x, y in zip(actual, expected):
            tvm.testing.assert_allclose(x, y, rtol=1e-5)

def test_sketch_search_policy_cdmppmodel(learning_params):
    search_common(target="cuda", cost_model=CDMPPModel(learning_params["cache_dir"]))

def sche_search_for_network(network_args, target, cost_model):
    cache_root="tmp/end2end"
    mod, params, inputs, task_info = get_network_tasks(
        network_args, target, cache_root)
    for task in task_info.tasks:
        search_common(task=task, target=target, cost_model=cost_model)


from tune_network import tune_and_evaluate

def sche_search(learning_params):
    from utils.gpu_utils import get_gpu_name
    from utils.device_info import query_cc
    target_gpu = learning_params['gpu_model']
    capability = query_cc(target_gpu)
    tvm_target = tvm.target.cuda(model=target_gpu.lower(), arch=f"sm_{capability}")

    networks = [
        'resnet_18',
        # 'resnet_50',
        # 'resnet3d_18',
        'mobilenet_v2', 
        # 'mobilenet_v3',
        # 'wide_resnet_50', 'resnext_50',
        # 'densenet_121',
        # 'inception_v3',
        # 'bert_tiny',
        # 'bert_base',
        # 'bert_medium', 'bert_large',
        # 'dcgan',
    ]
    # cm = CDMPPModel(learning_params["cache_dir"])
    # cm = auto_scheduler.XGBModel()

    batch_sizes = [1]
    for network in networks:
        for batch_size in batch_sizes:
            network_args = {
                "network": network,
                "batch_size": batch_size,
            }
                                              
            # sche_search_for_network(network_args, tvm_target, cm)
            tuning_args = {
                "eval_only": False,
                "continue_tuning": False,
                "n_trials": 5000,
                "run_timeout": 25,
                "cost_model": "tlp-no-update", # ['xgb', 'random', 'xgb-no-update', 'cdmpp', "tlp-no-update"]
                # ['xgb', 'xgb-no-update', 'mlp', 'mlp-no-update', 'tab', 'tab-no-update', 'tlp-no-update']:
                "load_model": None,
            }

            if tvm_target.model == "unknown":
                log_file = "%s-B%d-%s-%s.json" % (network, batch_size, tvm_target.kind,
                    tuning_args["cost_model"])
            else:
                log_file = "%s-B%d-%s-%s-%s.json" % (network, batch_size, tvm_target.kind,
                    tvm_target.model, tuning_args["cost_model"])
            tuning_args["log_file"] = log_file

            if tuning_args["cost_model"] == "cdmpp":
                tuning_args["cost_model"] = CDMPPModel(learning_params["cache_dir"])
            elif tuning_args["cost_model"].startswith("tlp"):
                tuning_args["load_model"] = ".workspace/sche_search/cost_models/tlp_model_49.pkl"
            elif tuning_args["cost_model"] == "xgb-no-update":
                tuning_args["load_model"] = ".workspace/sche_search/cost_models/xgb_t4_1_200.pkl"

            tune_and_evaluate(network_args, tuning_args, 
                tvm_target, None, ".workspace/sche_search/result.tsv")
                            
    