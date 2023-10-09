import os

from .tune_space import GridAutotuneSpace
from .grid_search import GridAutoTuner, FileGridAutoTuner
from .bo_search import BOAutoTuner, XGBBOTuner
from .finetune import fine_tune
from .optuna_search import OptunaTuner
from .tuner import AutoTuner

from utils.util import TrainTestPair


def auto_tune(ds_pair: TrainTestPair, data_meta_info, learning_params):
    assert learning_params["metric_learner"] in [2, 3], "Not ready for other learning methods"
    if learning_params["workspace"] is None:
        learning_params["workspace"] = ".workspace/autotune"
    if not os.path.exists(learning_params["workspace"]):
        os.makedirs(learning_params["workspace"])

    if learning_params["tune_method"] == "grid":
        grid_tuner = GridAutoTuner(
            process_num = 1,
            data_meta_info=data_meta_info,
            ds_pair=ds_pair,
            learning_params=learning_params
        )
        grid_tuner.grid_tune()
    elif learning_params["tune_method"] == "gen_cfg":
        search_space = GridAutotuneSpace()
        def check_func(_config):
            if "embed_layer_unit" in _config and "mlp_layer_unit" in _config and "embed_feature_len" in _config:
                if _config["embed_layer_unit"] == _config["mlp_layer_unit"] and \
                    _config["embed_layer_unit"] == _config["embed_feature_len"]:
                    return True
                else:
                    return False
            else:
                return True
        search_space.register_config_filter(check_func)
        search_space.dump_all_configs(os.path.join(learning_params["workspace"], "all_cfgs"), split=8)
    elif learning_params["tune_method"] == "file_grid":
        all_cfg_dir = os.path.join(learning_params["workspace"], "all_cfgs")
        if not os.path.exists(all_cfg_dir):
            raise ValueError(f"{all_cfg_dir} does not exsit")
        root, _, files = list(os.walk(all_cfg_dir))[0]
        files = [os.path.join(root, f) for f in files]
        grid_tuner = FileGridAutoTuner(
            cfg_files=files,
            data_meta_info=data_meta_info,
            ds_pair=ds_pair,
            learning_params=learning_params
        )
        grid_tuner.grid_tune()
    elif learning_params["tune_method"] == "bo":
        if learning_params["metric_learner"] == 3:
            BOTuner = XGBBOTuner
        elif learning_params["metric_learner"] == 2:
            BOTuner = BOAutoTuner
        else:
            raise
        bo_tuner = BOTuner(
            data_meta_info=data_meta_info,
            ds_pair=ds_pair,
            learning_params=learning_params
        )
        bo_tuner.tune(init_points=20, n_iter=10000)
    elif learning_params["tune_method"] == "fine_tune":
        fine_tune(ds_pair, data_meta_info, learning_params, top_k=1)
    elif learning_params["tune_method"] == "reproduce":
        from .config_to_test import config
        tuner = AutoTuner(
            data_meta_info=data_meta_info,
            ds_pair=ds_pair,
            learning_params=learning_params
        )
        tuner.target_func(config, reproduce=True)
    elif learning_params["tune_method"].startswith("optuna"):
        tune_options = learning_params["tune_method"].split(",")
        if len(tune_options) > 1 and tune_options[1] == "monitor":
            assert len(tune_options) > 2, tune_options
            OptunaTuner.monitor(storage_path=",".join(tune_options[2:]))
            exit(0)
        tuner = OptunaTuner(
            data_meta_info=data_meta_info,
            ds_pair=ds_pair,
            learning_params=learning_params
        )
        tuner.tune()
    else:
        raise ValueError(f"Invalide tune option {learning_params['tune_method']}")
        
if __name__ == '__main__':
    _space = GridAutotuneSpace()
    print(_space[0])
    print(_space[1])
    print(_space[4])