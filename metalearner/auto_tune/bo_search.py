import os

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from utils.util import prompt
from .tune_space import BOAutotuneSpace, XGBAutotuneSpace
from .tuner import AutoTuner

from metalearner.learner.xgb import XGBLearner


class BOAutoTuner(AutoTuner):
    def __init__(self, *args, **kwargs):
        super(BOAutoTuner, self).__init__(*args, **kwargs)
        self.search_space = BOAutotuneSpace()
        msg = f"BOAutoTuner is used"
        prompt(msg)
        self.tune_state.log(msg)

    def wrap_target_func(self, 
            hidden_layer_unit,
            embed_layer_num,
            mlp_layer_num,
            opt_type,
            wd,
            lr,
            batch_size,
        ):
        _config = {
            "embed_layer_unit": hidden_layer_unit,
            "embed_layer_num": embed_layer_num,
            "mlp_layer_unit": hidden_layer_unit,
            "mlp_layer_num": mlp_layer_num,
            "embed_feature_len": hidden_layer_unit,
            "opt_type": opt_type,
            "wd": wd,
            "lr": lr,
            "batch_size": batch_size,
            "cuda_id": 0,
        }
        for _config_key in _config.keys():
            _config[_config_key] = self.search_space.convert(_config_key, _config[_config_key])
        return self.target_func(_config)

    def tune(self, init_points=10, n_iter=100):
        # Bounded region of parameter space
        pbounds = {}
        for dim in self.search_space.pbound_dict.keys():
            if self.search_space.pbound_dict[dim][0] == self.search_space.pbound_dict[dim][1]:
                pbounds[dim] = (
                    self.search_space.pbound_dict[dim][0],
                    self.search_space.pbound_dict[dim][1]+1e-6)
            else:
                pbounds[dim] = (
                    self.search_space.pbound_dict[dim][0],
                    self.search_space.pbound_dict[dim][1])

        optimizer = BayesianOptimization(
            f=self.wrap_target_func,
            pbounds=pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)
        save_path = os.path.join(self.workspace, "bo_logs.json")
        if os.path.exists(save_path):
            # New optimizer is loaded with previously seen points
            load_logs(optimizer, logs=[save_path])

        logger = JSONLogger(path=save_path)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        # import pdb 
        # pdb.set_trace()
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )


class XGBBOTuner(BOAutoTuner):
    def __init__(self, *args, **kwargs):
        super(XGBBOTuner, self).__init__(*args, **kwargs)
        self.search_space = XGBAutotuneSpace()

    def wrap_target_func(self, 
            max_depth,
            gamma,
            min_child_weight,
            subsample,
            eta
        ):
        _config = {
            "max_depth": max_depth,
            "gamma": gamma,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "eta": eta
        }
        for _config_key in _config.keys():
            _config[_config_key] = self.search_space.convert(_config_key, _config[_config_key])
        return self.target_func(_config)
    
    def target_func(self, _config):
        learner = XGBLearner(
            None, self.data_meta_info, 
            cache_path=None,
            tb_log_dir= None,
            log_level="debug"
        )

        verbose = False
        learner.assign_config(_config, verbose=verbose)

        metrics_val, metrics_train= learner.train(self.dataset, verbose=verbose)
        mape_val, rmse_val, mape_train = float(metrics_val["mape"]), float(metrics_val["rmse"]), float(metrics_train["mape"]) if metrics_train is not None else None
        learner.exit()

        self.tune_state.check_best(mape_val, learner, _config,
            mape_val=mape_val, mape_train=mape_train)
        return mape_val
