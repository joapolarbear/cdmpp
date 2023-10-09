''' Hyper-paramteters using Optuna
    see https://optuna.readthedocs.io/en/stable/tutorial
'''
import logging
import sys, os
import optuna

from .tuner import AutoTuner
from utils.env import PROJECT_CFG
from utils.util import read_yes

class OptunaTuner(AutoTuner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tune_state = None

        ''' Create Study
            
            Sampling algorithm selection
            * Tree-structured Parzen Estimator algorithm implemented in optuna.samplers.TPESampler
            * CMA-ES based algorithm implemented in optuna.samplers.CmaEsSampler
            * Grid Search implemented in optuna.samplers.GridSampler
            * Random Search implemented in optuna.samplers.RandomSampler
            The default sampler is optuna.samplers.TPESampler.
            E.g.
                study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
                study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())

        '''
        self.device_name = os.environ.get("CDPP_DEVICE_NAME", "none")

        # Add stream handler of stdout to show the messages
        # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = f"{os.path.basename(self.learning_params['input'])}_{self.learning_params['mode']}"  # Unique identifier of the study.
        if self.multi_object:
            study_name = f"{study_name}_mo"
        study_name = f"{study_name}.log"
        storage_dir = os.environ.get("CDPP_AUTOTUNE_DIR", None)
        self.storage_path = os.path.join(storage_dir, study_name) if storage_dir else study_name

        ### Use SQLite
        # storage_name = f"sqlite:///{self.storage_path}.db"

        ### Use file-based
        storage_name = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(self.storage_path),
        )

        ### Optimize direction for each objective
        directions = ["minimize", "minimize"] if self.multi_object else ["minimize"]

        optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(f".workspace/autotune/{study_name}.log"))
        self.study = optuna.create_study(
            study_name=study_name, 
            storage=storage_name,
            load_if_exists=True,
            directions=directions)
        print(f"Sampler is {self.study.sampler.__class__.__name__}")

    def objective(self, trial):
        opt_type = trial.suggest_categorical("opt_type", ["adam", "sgd"])
        loss_type = trial.suggest_categorical("loss_type", ["MSE", "MSE+MAPE"])
        wd = trial.suggest_float("wd", 1e-5, 0.9, log=True)
        lr = trial.suggest_float("lr", 1e-7, 0.9, log=True)
        lr_scheduler = trial.suggest_categorical("lr_scheduler", ["one_cycle", "cycle", "exp", "none"])
        # batch_size = trial.suggest_int("batch_size", 128, 1024, log=True)
        batch_size = trial.suggest_int("batch_size", 4, 512, log=True)
        # batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

        tsfm_layer_num = trial.suggest_int("tsfm_layer_num", 1, 12)

        input_layer_unit = trial.suggest_int("input_layer_unit", 164, 1024, log=True)
        embed_layer_unit = trial.suggest_int("embed_layer_unit", 64, 1024, log=True)
        embed_layer_num = trial.suggest_int("embed_layer_num", 1, 9, step=2)
        embed_feature_len = trial.suggest_int("embed_feature_len", 64, 1024, log=True)
        mlp_layer_unit = trial.suggest_int("mlp_layer_unit", 64, 1024, log=True)
        mlp_layer_num = trial.suggest_int("mlp_layer_num", 1, 9, step=2)

        _config = {
            "batch_size": batch_size, 
            "lr": lr,
            "wd": wd,
            "opt_type": opt_type,
            "loss_type": loss_type,
            "lr_scheduler": lr_scheduler,

            "input_layer_unit": input_layer_unit,
            "embed_layer_unit": embed_layer_unit,
            "embed_layer_num": embed_layer_num,
            "embed_feature_len": embed_feature_len,
            "mlp_layer_unit": mlp_layer_unit,
            "mlp_layer_num": mlp_layer_num,

            "tsfm_layer_num": tsfm_layer_num,

            "USE_CMD_REGULAR": False,
            "enable_up_sampling": False,
            "use_clip": None

        }
        
        if PROJECT_CFG["AUTOTUNE_TENSORBOARD"]:
            _config["trial_id"] = trial.number
        
        trial.set_user_attr("device", self.device_name)

        # if _config["loss_type"] == "MAPE" or _config["op_type"] == "rms" \
        #     or _config["lr_scheduler"] == "cycle":
        #     return [101, 101] ### Pruning
            
        return self.target_func(_config)
    
    def tune(self):
        
        def print_best_callback(study, trial):
            if len(trial.values) > 1:
                print("\n[Best Trials]")
                for best_trial in study.best_trials:
                    print(f" * Trial {best_trial.number} with value: {best_trial.values}, "
                        f"Best params: {best_trial.params}, Device: {best_trial.user_attrs.get('device', 'none')}")
                print("\n")
            elif len(trial.values) == 1:
                best_trial = study.best_trial
                print(f"\n[Best Trial] {best_trial.number} with value: {best_trial.values[0]}, "
                    f"Best params: {best_trial.params}, Device: {best_trial.user_attrs.get('device', 'none')}\n")
            else:
                assert False, "Should not reach."

        def _print_trial_hook(study, trial):
            print(f" * Trial {trial.number} with value: {trial.values}, params: {trial.params}, "
                f"duration: {trial.duration} s")
            exit(0)
        
        hooks = [
            print_best_callback,
            # _print_trial_hook
        ]

        self.study.optimize(self.objective, n_trials=1000, callbacks=hooks)
        print(f"\nBest params {self.study.best_params}, best value {self.study.best_value}")
    
    @staticmethod
    def monitor(storage_path):
        ### Use file-based
        storage_name = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(storage_path),
        )
        loaded_study = optuna.load_study(study_name = None, storage=storage_name)

        print(f"Optuna study {loaded_study.study_name} information")
        print(f"Study directions: {loaded_study.directions}")

        print("\nBest Trials:")
        for best_trial in loaded_study.best_trials:
            print(f" * Trial {best_trial.number} with value: {best_trial.values}, Best params: {best_trial.params}, "
                f"Device: {best_trial.user_attrs.get('device', 'none')}, "
                f"Start time: {best_trial.datetime_start}")
        
        if read_yes("Show all trials?"):
            print("\nAll Trials:")
            for trial in loaded_study.trials:
                if not trial.state.is_finished():
                    continue
                print(f" * Trial {trial.number} with value: {trial.values}, params: {trial.params}, "
                    f"Start time: {trial.datetime_start}, duration: {trial.duration} s")

        df = loaded_study.trials_dataframe()
        df.to_csv(storage_path + ".csv")



