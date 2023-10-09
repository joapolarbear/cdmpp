import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

import seaborn as sns

from utils.util import fig_base


font = {"color": "darkred",
        "size": 13,
        "family": "serif"}
def visual_percentile(path):
    with open(path, 'r') as fp:
        task2cache = json.load(fp)

    rst = {
        "mape_train": [],
        "mape_val": []
    }
    for task, _cache in task2cache.items():
        if _cache is None or "mape_val" not in _cache \
            or "mape_train" not in _cache:
            continue
        rst["mape_train"].append(_cache["mape_train"])
        rst["mape_val"].append(_cache["mape_val"])
    # df = pd.DataFrame({
    #     "Dataset": ["Training", "Test"],
    #     "MAPE": [rst["mape_train"], rst["mape_val"]]})
    df = pd.DataFrame(rst)
    print(df.head(6))

    fig = plt.figure(figsize=(12, 6))
    _fig_base = fig_base(2, row_first=True)
    ax = fig.add_subplot(_fig_base+1)    
    sns.boxplot(
        x = df["mape_val"],
    )
    plt.xlabel("Test MAPE ", fontsize=16)
    ax = fig.add_subplot(_fig_base+2)    
    sns.boxplot(
        x = df["mape_train"],
    )
    plt.xlabel("Training MAPE ", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(".", ".fig", "metalearner", "single_task_error_stat.png"))
    plt.close()

paths = [
    # ".workspace/cm/task_runtime_cache_xgb.json",
    ".workspace/cm/task_runtime_cache_xgb-ansor.json"
    ]
for path in paths:
    visual_percentile(path)
