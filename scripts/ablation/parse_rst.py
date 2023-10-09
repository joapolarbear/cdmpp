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
 
index_dict = {"device": []}
index_dict.update(dict([(norm_method_id, []) for norm_method_id in range(8)]))
df_mape = pd.DataFrame(index_dict)
df_mape.set_index("device", inplace=True)

def add_data(df, device, norm_method_id, mape):
    norm_method_id = int(norm_method_id)
    if device not in df.index:
        df.loc[device] = dict([(norm_method_id, 0) for norm_method_id in range(8)])
    # print(device, norm_method_id, mape)
    df.loc[device, norm_method_id] = mape

def exp_norm_method(ablation_dir):
    index_dict = {"device": []}
    index_dict.update(dict([(norm_method_id, []) for norm_method_id in range(8)]))
    df_mape = pd.DataFrame(index_dict)
    df_mape.set_index("device", inplace=True)
    
    def _add_data(df, device, norm_method_id, mape):
        norm_method_id = int(norm_method_id)
        if device not in df.index:
            df.loc[device] = dict([(norm_method_id, 0) for norm_method_id in range(8)])
        # print(device, norm_method_id, mape)
        df.loc[device, norm_method_id] = mape

    log_dir = os.path.join(ablation_dir, "norm_method", "logs")
    for _file in os.listdir(log_dir):
        file_match = re.search(r"(?P<device>t4|v100|p100|a100|k80)_"
            r"(?P<try_id>\d+)_(?P<sample_num>\d+)_(?P<norm_method_id>\d+).txt", 
            os.path.basename(_file)).groupdict()
        
        sample_num = int(file_match["sample_num"])
        if sample_num != SAMPLE_NUM:
            continue
        
        device = file_match["device"]
        norm_method_id = int(file_match["norm_method_id"])
        best_rst = parse_log_file(os.path.join(log_dir, _file))
        if best_rst is None:
            print(f"No data for {_file}")
        else:
            _add_data(df_mape, device, norm_method_id, best_rst["test_metric"]["mape"])
    
    print("\n### Effect of norm methods")
    print(df_mape)
    df_mape.to_csv(os.path.join(ablation_dir, "rst_norm_method.csv"))
    print("\n\n")

def exp_loss_func(ablation_dir):
    LOSS_FUNCS = ["mse", "mape", "mspe", "mse+mape"]

    index_dict = {"device": []}
    index_dict.update(dict([(loss_func, []) for loss_func in LOSS_FUNCS]))
    df_mape = pd.DataFrame(index_dict)
    df_mape.set_index("device", inplace=True)

    index_dict = {"device": []}
    index_dict.update(dict([(loss_func, []) for loss_func in LOSS_FUNCS]))
    df_rmse = pd.DataFrame(index_dict)
    df_rmse.set_index("device", inplace=True)

    def _add_data(df, device, _loss_func, data):
        if device not in df.index:
            df.loc[device] = dict([(loss_func, 0) for loss_func in LOSS_FUNCS])
        # print(device, norm_method_id, data)
        df.loc[device, _loss_func] = data

    log_dir = os.path.join(ablation_dir, "norm_method", "logs")
    OURPUT_NORM_METHOD_ID = 0
    for _file in os.listdir(log_dir):
        file_match = re.search(r"(?P<device>t4|v100|p100|a100|k80)_"
            r"(?P<try_id>\d+)_(?P<sample_num>\d+)_(?P<norm_method_id>\d+).txt", 
            os.path.basename(_file)).groupdict()
        sample_num = int(file_match["sample_num"])
        norm_method_id = int(file_match["norm_method_id"])
        if sample_num != SAMPLE_NUM:
            continue
        if norm_method_id != OURPUT_NORM_METHOD_ID:
            continue
        
        device = file_match["device"]
        best_rst = parse_log_file(os.path.join(log_dir, _file))
        if best_rst is None:
            print(f"No data for {_file}")
        else:
            _add_data(df_mape, device, "mse+mape", best_rst["test_metric"]["mape"])
            _add_data(df_rmse, device, "mse+mape", best_rst["test_metric"]["rmse"])
    
    log_dir = os.path.join(ablation_dir, "loss_func", "logs")
    for _file in os.listdir(log_dir):
        file_match = re.search(r"(?P<device>t4|v100|p100|a100|k80)_"
            r"(?P<try_id>\d+)_(?P<sample_num>\d+)_(?P<loss_func>mape|mse|mspe).txt", 
            os.path.basename(_file)).groupdict()
        sample_num = int(file_match["sample_num"])
        
        if sample_num != SAMPLE_NUM:
            continue
        
        loss_func = file_match["loss_func"]
        device = file_match["device"]
        best_rst = parse_log_file(os.path.join(log_dir, _file))
        if best_rst is None:
            print(f"No data for {_file}")
        else:
            _add_data(df_mape, device, loss_func, best_rst["test_metric"]["mape"])
            _add_data(df_rmse, device, loss_func, best_rst["test_metric"]["rmse"])
    
    print("\n### Effect of loss funcs")
    print("MAPE")
    print(df_mape)
    print("RMSE")
    print(df_rmse)
    df_mape.to_csv(os.path.join(ablation_dir, "rst_loss_func_mape.csv"))
    df_rmse.to_csv(os.path.join(ablation_dir, "rst_loss_func_rmse.csv"))
    print("\n\n")

def exp_pe(ablation_dir):
    COLUMN_INDEXS = ["w/ PE", "w/o PE"]

    index_dict = {"device": []}
    index_dict.update(dict([(idx, []) for idx in COLUMN_INDEXS]))
    df_mape = pd.DataFrame(index_dict)
    df_mape.set_index("device", inplace=True)

    def _add_data(df, device, column, mape):
        if device not in df.index:
            df.loc[device] = dict([(idx, 0) for idx in COLUMN_INDEXS])
        # print(device, norm_method_id, mape)
        df.loc[device, column] = mape

    log_dir = os.path.join(ablation_dir, "norm_method", "logs")
    OURPUT_NORM_METHOD_ID = 0
    for _file in os.listdir(log_dir):
        file_match = re.search(r"(?P<device>t4|v100|p100|a100|k80)_"
            r"(?P<try_id>\d+)_(?P<sample_num>\d+)_(?P<norm_method_id>\d+).txt", 
            os.path.basename(_file)).groupdict()
        sample_num = int(file_match["sample_num"])
        norm_method_id = int(file_match["norm_method_id"])
        if sample_num != SAMPLE_NUM:
            continue
        if norm_method_id != OURPUT_NORM_METHOD_ID:
            continue
        
        device = file_match["device"]
        best_rst = parse_log_file(os.path.join(log_dir, _file))
        if best_rst is None:
            print(f"No data for {_file}")
        else:
            _add_data(df_mape, device, "w/ PE", best_rst["test_metric"]["mape"])
    
    log_dir = os.path.join(ablation_dir, "pe", "logs")
    for _file in os.listdir(log_dir):
        file_match = re.search(r"(?P<device>t4|v100|p100|a100|k80)_"
            r"(?P<try_id>\d+)_(?P<sample_num>\d+)_disable_pe.txt", 
            os.path.basename(_file)).groupdict()
        sample_num = int(file_match["sample_num"])
        
        if sample_num != SAMPLE_NUM:
            continue
        
        device = file_match["device"]
        best_rst = parse_log_file(os.path.join(log_dir, _file))
        if best_rst is None:
            print(f"No data for {_file}")
        else:
            _add_data(df_mape, device, "w/o PE", best_rst["test_metric"]["mape"])
    
    print("\n### Effect of PE")
    print(df_mape)
    df_mape.to_csv(os.path.join(ablation_dir, "rst_pe.csv"))
    print("\n\n")

def exp_domain_adaption(ablation_dir):
    COLUMN_INDEXS = ["none", "cmd", "mmd"]

    index_dict = {"device": []}
    index_dict.update(dict([(idx, []) for idx in COLUMN_INDEXS]))
    df_mape = pd.DataFrame(index_dict)
    df_mape.set_index("device", inplace=True)

    def _add_data(df, device, column, mape):
        if device not in df.index:
            df.loc[device] = dict([(idx, 0) for idx in COLUMN_INDEXS])
        # print(device, norm_method_id, mape)
        df.loc[device, column] = mape

    log_dir = os.path.join(ablation_dir, "norm_method", "logs")
    OURPUT_NORM_METHOD_ID = 0
    for _file in os.listdir(log_dir):
        file_match = re.search(r"(?P<device>t4|v100|p100|a100|k80)_"
            r"(?P<try_id>\d+)_(?P<sample_num>\d+)_(?P<norm_method_id>\d+).txt", 
            os.path.basename(_file)).groupdict()
        sample_num = int(file_match["sample_num"])
        norm_method_id = int(file_match["norm_method_id"])
        if sample_num != SAMPLE_NUM:
            continue
        if norm_method_id != OURPUT_NORM_METHOD_ID:
            continue
        
        device = file_match["device"]
        best_rst = parse_log_file(os.path.join(log_dir, _file))
        if best_rst is None:
            print(f"No data for {_file}")
        else:
            _add_data(df_mape, device, "none", best_rst["test_metric"]["mape"])
    
    log_dir = os.path.join(ablation_dir, "domain_adaption", "logs")
    for _file in os.listdir(log_dir):
        file_match = re.search(r"(?P<device>t4|v100|p100|a100|k80)_"
            r"(?P<try_id>\d+)_(?P<sample_num>\d+)_(?P<domain_diff_metric>((0_)?\d+)?cmd|mmd).txt", 
            os.path.basename(_file)).groupdict()
        sample_num = int(file_match["sample_num"])
        
        if sample_num != SAMPLE_NUM:
            continue
        
        device = file_match["device"]
        domain_diff_metric = file_match["domain_diff_metric"]
        best_rst = parse_log_file(os.path.join(log_dir, _file))
        if best_rst is None:
            print(f"No data for {_file}")
        else:
            _add_data(df_mape, device, domain_diff_metric, best_rst["test_metric"]["mape"])
    
    print("\n### Effect of Domain Adaption")
    print(df_mape)
    df_mape.to_csv(os.path.join(ablation_dir, "rst_domain_adaption.csv"))
    print("\n\n")

print(sys.argv[1])
exp_norm_method(sys.argv[1])
exp_loss_func(sys.argv[1])
exp_pe(sys.argv[1])
exp_domain_adaption(sys.argv[1])