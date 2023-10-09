import os
from typing import Union

from tvm_helper.tir_helper import pre_add_dims

from utils.env import read_yaml_cfg


class ALL_FEATURE_TYPE:
    debug = "debug"
    cus = "cus"
    ansor = "ansor"
    ast_ansor = "ast_ansor"


class FeatureInfo:
    def __init__(self, feature_type):
        self.update_feature(feature_type)
        
    def update_feature(self, feature_type):
        self.feature_type = feature_type
        if self.feature_type == ALL_FEATURE_TYPE.debug:
            self._pre_add_dims = ["avg"]
            self.task_id_idx = None
        elif self.feature_type == ALL_FEATURE_TYPE.ansor or self.feature_type == ALL_FEATURE_TYPE.ast_ansor:
            self._pre_add_dims = pre_add_dims
            self.task_id_idx = None
        elif self.feature_type == ALL_FEATURE_TYPE.cus:
            self._pre_add_dims = pre_add_dims
            self._pre_add_dims.append("task_id")
            self.task_id_idx = len(self._pre_add_dims)-1
        else:
            raise ValueError(self.feature_type)
    
    def __str__(self):
        return f"feature_type={self.feature_type}, PRE_Additional_Entries={self._pre_add_dims}, task_id_idx={self.task_id_idx}"

FEATURE_INFO: Union[None, FeatureInfo] = None

def init_fea_info_via_data(input_dir):
    input_dir = os.path.abspath(input_dir)
    cfg = read_yaml_cfg(os.path.join(input_dir, "cfg.yaml"))
    global FEATURE_INFO
    assert FEATURE_INFO is None, (input_dir, str(FEATURE_INFO))
    FEATURE_INFO = FeatureInfo(cfg["feature_type"])

def init_fea_info(fea_type):
    global FEATURE_INFO
    assert FEATURE_INFO is None, str(FEATURE_INFO)
    FEATURE_INFO = FeatureInfo(fea_type)

def is_feature_type(target_type):
    return FEATURE_INFO.feature_type == target_type