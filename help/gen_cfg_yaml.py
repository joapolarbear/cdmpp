import re
import os
import sys
import json
import yaml
import copy

from utils.env import read_yaml_cfg, PROJECT_DIR

sample_config = read_yaml_cfg(os.path.join(PROJECT_DIR, "configs", "model", "mlp-standard.yaml"))
with open(sys.argv[1], 'r') as fp:
    lines = fp.readlines()

for l in lines:
    _sample_config = copy.deepcopy(sample_config)
    match = re.search(r"Trial (?P<trial_id>[\d]+)", l)
    _sample_config["cfg_name"] = f"search_trial_{match['trial_id']}"
    ss = l.split("{")[1].split("}")[0]
    sl = [pair.split(": ") for pair in ss.split(", ")]
    config = _sample_config["cost_model"]
    for k, v in sl:
        key = k.strip("'")
        if re.match(r"^[\d]+$", v):
            v = int(v)
        elif re.match(r"^[-\d.e]+$", v):
            v = float(v)
        else:
            v = v.strip("'")
        config[key] = v
    config['embedded_layers'] = [config['embed_layer_unit']] * config['embed_layer_num']
    config['regression_layers'] = [config['mlp_layer_unit']] * config['mlp_layer_num']
    config.pop("embed_layer_unit")
    config.pop('embed_layer_num')
    config.pop('mlp_layer_unit')
    config.pop('mlp_layer_num')
    with open(os.path.join(PROJECT_DIR, "tmp", f"{_sample_config['cfg_name']}.yaml"), 'w') as yaml_fp:
        yaml_fp.write(yaml.dump(_sample_config, default_flow_style=False))

'''
TRIAL_NAME=search_trial_26
TRIAL_NAME=search_trial_79
TRIAL_NAME=search_trial_84
TRIAL_NAME=search_trial_86
launch -- bash scripts/train.sh run \
    --mode sample200 \
    -i .workspace/ansor -c tmp/${TRIAL_NAME}.yaml \
    --tb_logdir .workspace/runs/${TRIAL_NAME}
'''
        
