import argparse
import torch
import numpy as np
import random
import os

import utils.env as cdpp_env
from utils.util import read_yes, warn_once, range2str

def parse_args():
    parser = argparse.ArgumentParser(prog="Cross Model Performance Prediction")
    parser.add_argument('-o', '--option', type=str, default="profile", help="One of [profile|tir_info]")
    parser.add_argument('-i', '--input', type=str, default=None, help="Path to store the inputs")
    parser.add_argument('-w', '--workspace', type=str, default=None, help="Path to store the results")
    parser.add_argument('-c', '--config', type=str, default=None, help="Path to the config file")
    parser.add_argument('-y', '--yes', action="store_true", help="Answer yes for any confirmation")
    parser.add_argument('-g', '--gpu_model', type=str, default='t4', help="GPU model name")

    parser.add_argument('--log_level', type=str, default="error", help="logging level")
    parser.add_argument('-p', '--pipeline', action="store_true", help="Enable pipeline if this argument is set")

    parser.add_argument('-E', '--epoch', type=int, default=None, help="Training epochs")
    parser.add_argument('-S', '--step', type=int, default=None, help="Training steps")

    ########################### About Dataset
    group_data = parser.add_argument_group('Dataset')
    group_data.add_argument('--mode', type=str, default="cross", help="How to learn task files, e.g., single|cross|group|samplek")
    group_data.add_argument('--force_load_data', action='store_true', help="If specified, force to load raw data")
    group_data.add_argument('--source_data', type=str, default='tir', help="One of op, tir, or not specified")
    group_data.add_argument('--op_type', type=str, default=None, help="If set, focus on one op_type")

    group_data.add_argument('-f', '--filters', type=int, default=110, help="Set 111 to enable three filters respectively, "
        "1) remove small samples; 2) remove unstable samples; 3) remove outliers")
    group_data.add_argument('--flop_bound', type=str, default=None, help="FLOP bount to perform flop-based clustering")
    group_data.add_argument('--ave_lb', type=float, default=0., help='Time cost lower bound (in second) used to filter out samples')

    ### TODO (huhanpeng): see if this can be deprecated, since we have --output_norm_method and --input_norm_method
    group_data.add_argument('--disable_norm', action="store_true", help="Diable normalization")

    ### If args.output_norm_method.startswith("cls_"), Convert the regression problem to classification probelm,"
    # i.e., y < class_base ==> 0, y >= class_base ==> 1")
    group_data.add_argument('--output_norm_method', type=str, default="0",
        # choices=["std", "none", "log", "cls_<class_base>"],
        help="Methods to normalize outputs")
    group_data.add_argument('--input_norm_method', type=str, default="min-max",
        # choices=["std", "none", "log", "cls_<class_base>"],
        help="Methods to normalize outputs")

    group_data.add_argument('--leaf_node_no', type=str, default='5', 
        help="Leaf node numbers we focus on, if there are mutiple valuses, separated with ,")
    group_data.add_argument('--synthetic', action="store_true", help="Use synthetic if set True")

    ########################### About cost model
    group_cm = parser.add_argument_group('Cost Model')
    group_cm.add_argument('--debug', action="store_true", help="debug step by step")
    group_cm.add_argument('--metric_learner', type=int, default=2, help="Specify the Metric Learner, use -1 to list all")
    group_cm.add_argument('--load_cache', action="store_true", help="Set to ture to load the cached model")
    group_cm.add_argument('--cache_dir', type=str, default=None, help="The dir load the cached model")
    group_cm.add_argument('-t', '--tb_logdir', type=str, default=None, help="Tensorboard logdir")

    ## Although we have YAML config file to specify the hyper-parameters, 
    # they can be overridden by the command arguments
    group_cm.add_argument('-B', '--batch_size', type=int, default=None, help="Batch size for training")
    # group_cm.add_argument('--opt', type=str, default="adam", help="optimizer")
    # group_cm.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    # group_cm.add_argument('--wd', type=float, default=1e-4, help="weight")
    group_cm.add_argument('--disable_pe', action="store_true", help="Set true to disable positional encoding")
    group_cm.add_argument('--loss_func', type=str, default=None, help="The loss function")
    group_cm.add_argument('--residual', action="store_true", help="Use residual connection")
    group_cm.add_argument('--domain_diff_metric', type=str, default=None, help="Metric used to evaluate domain difference")

    ### The following arguments task effect only when --load_cache is True
    group_cm.add_argument('--finetune_cache_dir', type=str, default=None, help="Take effect when `load_cache` is True. "
        "The directory to save fine-tune resutls, not the same as where the model is loaded.")
    group_cm.add_argument('--finetune_cfg', type=str, default=None, help="Path of the configuration used in fine-tune")
    group_cm.add_argument('--finetune_datapath', type=str, default=None, help="Path to dataset for finetune")
    
    ########################### About auto-tuning
    group_tune = parser.add_argument_group('Auto Tune')
    group_tune.add_argument('--tune_method', type=str, default="file_grid", help="Tunning method, one of 'file_grid', 'bo', 'grid'")
    
    ########################### About baselines
    group_base = parser.add_argument_group('Baselines')
    group_base.add_argument('--tiramisu', action="store_true", help="Evaluate Tiramisu")

    ########################### End2end performance prediction
    group_end2end = parser.add_argument_group("End2End")
    group_end2end.add_argument('--replay_mode', type=str, default="", help="Replay mode")
    group_end2end.add_argument('--networks', type=str, default=None, help="Networks to evaluate")
    group_end2end.add_argument('--batch_sizes', type=str, default=None, help="Batch size(s) to evaluate")

    args = parser.parse_args()

    ################## process arguments ######################

    if args.config is None:
        warn_once(f"Project config file is not specified, use the default configuration")
        args.config = 'configs/model/mlp-standard.yaml'
    cdpp_env.PROJECT_CFG = cdpp_env.read_yaml_cfg(args.config)

    # enable tensorboard output
    _tensorboard_output = os.environ.get("TENSORBOARD_OUTPUT", None)
    if _tensorboard_output is not None:
        assert args.tb_log_dir is None, (f"Environmen variable 'TENSORBOARD_OUTPUT' ({_tensorboard_output})"
            " and --tb_log_dir ({args.tb_log_dir}) can not be given at the same time")
        args.tb_log_dir = _tensorboard_output
        
    if not args.tiramisu:
        if cdpp_env.PROJECT_CFG["cost_model"]["USE_CMD_REGULAR"]:
            if not read_yes("USE_CMD_REGULAR?", yes=args.yes):
                exit(0)

        if cdpp_env.PROJECT_CFG["cost_model"]["use_residual"]:
            if not read_yes("residual?", yes=args.yes):
                exit(0)

        ### Apply positional encodings
        cdpp_env.PROJECT_CFG["USE_PE"] = not args.disable_pe
        # read_yes("[AST Rawdata] Use positional encoding", yes=args.yes)
        if cdpp_env.PROJECT_CFG["USE_PE"]:
            pass
        else:
            cdpp_env.PROJECT_CFG["USE_PE"] = False
            warn_once("[AST Rawdata] PE is NOT used")
        cdpp_env.PROJECT_CFG["MAX_SEQ_LEN"] = 16

        if args.leaf_node_no is not None:
            cdpp_env.PROJECT_CFG["FIX_SEQ_LEN"] = [int(v) for v in args.leaf_node_no.split(",")]
            if not read_yes(f"Do not use padding and focus on samples of "
                    f"leaf no = {cdpp_env.PROJECT_CFG['FIX_SEQ_LEN']} ?", yes=args.yes):
                exit(0)
    
        if args.synthetic:
            if not read_yes(f"Use synthetic data?", yes=args.yes):
                exit(0)
        
        if args.output_norm_method != 'std':
            if not read_yes(f"Are you sure to use \'{args.output_norm_method}\' to norm outputs?", yes=args.yes):
                    exit(0)
        cdpp_env.PROJECT_CFG['OUTPUT_NORM_METHOD'] = args.output_norm_method

        if args.input_norm_method != 'min-max':
            if not read_yes(f"Are you sure to use \'{args.input_norm_method}\' to norm inputs?", yes=args.yes):
                    exit(0)
        cdpp_env.PROJECT_CFG['INPUT_NORM_METHOD'] = args.input_norm_method

        cdpp_env.PROJECT_CFG["FILTERS_VIA_FEATURE_ENTRY"] = bool(int(os.environ.get("CDPP_FILTERS_VIA_FEATURE_ENTRY", '0')))
        if cdpp_env.PROJECT_CFG["FILTERS_VIA_FEATURE_ENTRY"]:
            if not read_yes("Are you sure to filter test samples based on feature entry distribution?"):
                exit(0)
        
        if args.flop_bound is not None:
            # cdpp_env.PROJECT_CFG["FLOP_BOUND"] = (1e9, None)
            cdpp_env.PROJECT_CFG["FLOP_BOUND"] = eval(args.flop_bound)
            if not read_yes(f"Use flops to partition dataset ? {range2str(cdpp_env.PROJECT_CFG['FLOP_BOUND'])} is given"):
                exit(0)
    
    elif args.tiramisu:
        cdpp_env.PROJECT_CFG['OUTPUT_NORM_METHOD'] = 'std' # fix: by default
        cdpp_env.PROJECT_CFG['INPUT_NORM_METHOD'] = "min-max"

    cdpp_env.PROJECT_CFG["FILTERS"] = [args.filters // 100, args.filters % 100 // 10, args.filters % 100 % 10]
    if cdpp_env.PROJECT_CFG["FILTERS"] != [1, 1, 1]:
        filter_prompt = [
            "Do not remove small samples?",
            "Do not remove unstable samples?",
            "Do not remove outliers?"
        ]
        for filter_id, _filter in enumerate(cdpp_env.PROJECT_CFG["FILTERS"]):
            if not _filter and not read_yes(filter_prompt[filter_id], yes=args.yes):
                exit(0)
    
    cdpp_env.PROJECT_CFG["AUTOTUNE_TENSORBOARD"] = bool(int(os.environ.get("CDPP_AUTOTUNE_TENSORBOARD", "0")))
    if cdpp_env.PROJECT_CFG["AUTOTUNE_TENSORBOARD"]:
        if not read_yes("[Autotune] Are you sure to dump traces for each search trial?", yes=args.yes):
            exit(0)

    ### Fix the random seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    ### causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
    torch.backends.cudnn.benchmark = False
    ### Avoiding nondeterministic algorithms
    torch.use_deterministic_algorithms(True)

    ### Config device
    os.environ['train_device'] = f'cuda:{0}' # training device: 'cpu' or 'cuda:X'
    os.environ['store_device'] = f'cuda:{0}' # Data storing device:  'cpu' or 'cuda:X'

    return args