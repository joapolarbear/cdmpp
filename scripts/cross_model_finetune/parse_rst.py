import os
import sys
import re
import json
import pandas as pd

line = "Time 4322.830 s - Epoch 492 step 92000 bs 600 - loss_train=0.711212098598, {'mape': 0.00027807533778090874, 'rmse': 3.0450093463500007e-07, '20%accuracy': 1.0, '10%accuracy': 1.0, '5%accuracy': 0.9999282639885222}"
SAMPLE_NUM = 200

LINE_PATTERN = r"Time\s+(?P<time>\d+\.\d+|\d+) s - "
LINE_PATTERN += r"Epoch\s+(?P<epoch>\d+) "
LINE_PATTERN += r"step (?P<step>\d+) "
LINE_PATTERN += r"bs (?P<bs>\d+) - "
LINE_PATTERN += r"loss_train=(?P<loss_train>\d+\.\d+|\d+), "
LINE_PATTERN += r"(?P<test_metric>\{.*\})"

def parse_log_file(path):
    with open(path, 'r') as fp:
        lines = fp.readlines()
    
    best_mape = 1e6
    best_match = None
    for line in lines:
        match = re.search(LINE_PATTERN, line)
        if match is None:
            continue
        line_match = match.groupdict()
        line_match["test_metric"] = json.loads(line_match["test_metric"].replace(
            "'", "\"").replace("nan", "1e6").replace("inf", "1e6"))
        line_match["test_metric"]["mape"] *= 100
        line_match["test_metric"]["rmse"] *= 1000
        if line_match["test_metric"]["mape"] < best_mape:
            best_mape = line_match["test_metric"]["mape"]
            best_match = line_match
    return best_match

def exp_cross_model_finetune(exp_dir):
    log_dir = os.path.join(exp_dir, "logs")
    for _file in os.listdir(log_dir):
        file_match = re.search(r"(?P<device>t4|v100|p100|a100|k80)_"
            r"(?P<try_id>\d+)_(?P<sample_num>\d+)_(?P<network>\w+).txt", 
            os.path.basename(_file)).groupdict()
        
        sample_num = int(file_match["sample_num"])
        if sample_num != SAMPLE_NUM:
            continue
        
        device = file_match["device"]
        best_rst = parse_log_file(os.path.join(log_dir, _file))
        if best_rst is None:
            print(f"No data for {_file}")
        else:
            print(_file, best_rst['test_metric'])

    print("\n\n")

print(sys.argv[1])
exp_cross_model_finetune(sys.argv[1])