import copy
import utils.env as cdpp_env

from .cluster_hub import BaseClusterHub

class LearnerClusterHub(BaseClusterHub):
    def __init__(self, path, learner_cls):
        self.learner_cls = learner_cls
        super(LearnerClusterHub, self).__init__()

        if path.endswith(".json"):
            self.load_from_json(path)
        else:
            self.add_entry_fron_config(path, cdpp_env.PROJECT_CFG)
        print(f"[{type(self).__name__}] ", self.to_json())

    def finalize_entry(self, target_entry):
        ''' The function to further process the entry, e.g., parse a learner from the entry
        '''
        _origin_env = copy.deepcopy(cdpp_env.PROJECT_CFG)
        if ":" in target_entry:
            ### e.g. dir/to/cm/:cost_sensitive,fix_seq_len=1
            args = target_entry.split(":")[1].split(",")
            target_entry = target_entry.split(":")[0]
            for _arg in args:
                if "=" in _arg:
                    _key, _value = _arg.split("=")
                    cdpp_env.PROJECT_CFG[_key] = _value
        else:
            args = []
        _learner = self.learner_cls.load_init(target_entry)
        cdpp_env.PROJECT_CFG = _origin_env
        for _arg in args:
            if _arg == "cost_sensitive":
                _learner.cost_sensitive_loss = True
        _learner.data_meta_info.tsfm_hub.print()
        return _learner

