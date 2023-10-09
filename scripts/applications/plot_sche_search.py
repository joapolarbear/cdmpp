from dis import code_info
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

rst_root = ".workspace/sche_search/rst"
_, date_dirs, _ = list((os.walk(rst_root)))[0]
align = True
all_data = {}
def add_data(model, bs, device, method, data):
    # print("add", model, bs, device, method, data)
    if model not in all_data:
        all_data[model] = {}
    if bs not in all_data[model]:
        all_data[model][bs] = {}
    if device not in all_data[model][bs]:
        all_data[model][bs][device] = {}
    all_data[model][bs][device][method] = data

for date_dir in date_dirs:
    _root, _, files = list((os.walk(os.path.join(rst_root, date_dir))))[0]
    for file in files:
        if not file.endswith(".tsv"):
            continue
        _path = os.path.join(_root, file)
        try:
            match = re.search(r'(?P<model>\w+)-B(?P<bs>\d+)-cuda-(?P<device>\w+)-(?P<method>cdmpp|xgb|tlp-no-update).tsv', file).groupdict()
        except:
            import pdb; pdb.set_trace()

        with open(_path, "r") as fp:
            lines = fp.readlines()
        
        trajectory = []
        for line in lines:
            try:
                _match = re.search(f'ElapsedTime\(s\)\t(?P<time>\d+)\tEstimatedLatency\(ms\)\t(?P<latency>(N/A|[+-]?([0-9]*[.])?[0-9]+))\t', line).groupdict()
            except:
                import pdb; pdb.set_trace()
            try:
                trajectory.append([int(_match["time"]), float(_match["latency"])])
            except:
                continue
        
        if len(trajectory) > 0:
            add_data(match["model"], match["bs"], match["device"], 
                match["method"], np.array(trajectory))

fig_dir = os.path.join(".workspace", "fig", "sche_search")
os.makedirs(fig_dir, exist_ok=True)

mpl.rcParams['hatch.linewidth'] = 0.5
# Set the palette using the name of a palette:
sns.set_theme(style="whitegrid", color_codes=True)
tips = sns.load_dataset("tips")
marks = ["/", "-", "\\", "x", "+", "."]
barwidth = 0.25
fontsize = 36

alias = {
    "cdmpp": "CDMPP",
    "xgb": "XGB",
    "tlp-no-update": "TLP"
}

for model in all_data:
    for bs in all_data[model]:
        for device in all_data[model][bs]:
            print(model, bs, device)
            fig = plt.figure(figsize=(8, 5))
            for method in all_data[model][bs][device]:
                _data = all_data[model][bs][device][method]
                if align:
                    _data[:, 1] -= min(_data[:, 1])
                plt.plot(_data[:, 0]/1e3, _data[:, 1], linewidth=3, label=alias[method])
            plt.xlabel(r"Search Time ($10^3$s)", fontsize=fontsize)
            plt.ylabel("Latency (ms)", fontsize=fontsize)
            plt.xticks(fontsize=fontsize-2)
            plt.yticks(fontsize=fontsize-2)
            plt.locator_params(nbins=5)
            plt.legend(fontsize=fontsize)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 
                f"{model}_{bs}_{device}.pdf"))


