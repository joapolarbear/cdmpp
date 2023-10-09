import json
import re
import os

from utils.util import good_new, prompt, notify, INFINITE_ERROR, warning

from metalearner.learner import ALL_METRIC_LERNER
from .tuner import TuneState

def fine_tune_config(_config, dataset, data_meta_info, learning_params, verbose=True):
    learner = ALL_METRIC_LERNER[learning_params["metric_learner"]](
        data_meta_info, 
        cache_path=os.path.join(learning_params["workspace"], "cm"),
        opt_type=_config["opt_type"],
        lr=_config["lr"],
        wd=_config["wd"],
        tb_log_dir= os.path.join(
            learning_params["tb_logdir"],
            os.path.basename(learning_params["workspace"])),
        log_level="debug",
        use_cuda=_config["cuda_id"]
    )
    if learning_params["load_cache"]:
        learner.load()
    
    learner.assign_config(_config, verbose=verbose)
    learner.register_hooks_for_grads()
    notify(f"Parameter count: {learner.para_cnt(_config)}")
    learner.train(dataset, verbose=verbose)

def fine_tune(dataset, data_meta_info, learning_params, top_k=1):
    tune_state = TuneState(learning_params["workspace"])
    if os.path.exists(tune_state.log_filename):
        with open(tune_state.log_filename, 'r') as fp:
            lines = fp.readlines()
        prompt(f"Tuning log exists at {tune_state.log_filename}, finetune the first {top_k} best ones")
        for line in lines[::-1]:
            if len(line) == 0:
                continue
            try:
                dump_info = json.loads(line)
                tune_state.best_target = dump_info["best_target"]
            except (json.decoder.JSONDecodeError, KeyError):
                continue
            if dump_info["state"] == 1:
                fine_tune_config(
                    dump_info["config"],
                    dataset,
                    data_meta_info,
                    learning_params)
                top_k -= 1
                if top_k == 0:
                    break
    else:
        warning(f"There is not tuning log at {tune_state.log_filename}")