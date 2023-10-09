#!/usr/bin/env python3
import os
import yaml
import json
import re
from jinja2 import Environment, FileSystemLoader

from utils.base import PROJECT_DIR

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def read_yaml_cfg(cfg_path, **kwargs):
    ''' Read YAML config file
        Use `kwargs` to render template
    '''
    if cfg_path is None or not os.path.exists(cfg_path):
        raise ValueError(f"Config path {cfg_path} does NOT exist")
        # return {}
    cfg_path = os.path.abspath(cfg_path)
    cfg_dir = os.path.dirname(cfg_path)
    env = Environment(loader=FileSystemLoader(cfg_dir))
    template = env.get_template(os.path.basename(cfg_path))
    c = template.render(**kwargs)
    cfg = yaml.load(c, Loader=yaml.Loader)
    if cfg is not None:
        print("Use the config file: {}".format(cfg_path))
    return cfg

def read_yaml_from_dir(target_dir):
    if target_dir is None or not os.path.exists(target_dir):
        raise ValueError("Valid path that consists the config yaml file must be given")
    cfg_file = None
    for file in os.listdir(target_dir):
        if file.endswith(".yaml"):
            if cfg_file is not None:
                raise ValueError("Two cfg file {} and {}".format(cfg_file, file))
            cfg_file = os.path.join(target_dir, file)
    assert cfg_file is not None
    return read_yaml_cfg(cfg_file, path=PROJECT_DIR)

def read_json_cfg(json_path=None):
    with open(json_path, 'r') as fp:
        cfg = json.load(fp)
    return cfg

PROJECT_CFG = None



