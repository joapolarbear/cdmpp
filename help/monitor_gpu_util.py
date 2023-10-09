'''
Example
    python3 xxx.py [monitor_duration] [name]
'''
import os, sys
import subprocess
import re
import time
import numpy as np
from tqdm import tqdm

from utils.gpu_utils import get_gpu_util

name = sys.argv[2]

match = re.search(r'((?P<hour>\d+)h)?((?P<minute>\d+)m)?(?P<second>\d+)s?', sys.argv[1])
rst = match.groupdict()

hour = int(rst["hour"]) if rst["hour"] else 0
minute = int(rst["minute"]) if rst["minute"] else 0
to_monitor_sec = hour * 3600 + minute * 60 + int(rst.get("second", 0))

st = time.time()
ed = st + to_monitor_sec

print(f"Monitor the GPU utilization for {to_monitor_sec} s ...")

util_list = []
with tqdm(total=100) as pbar:
    update_interval = to_monitor_sec / 100.
    last_update_t = st
    while True:
        t = time.time()
        if t > ed:
            break
        util = get_gpu_util()
        util_list.append((t - st, np.mean(util)))
        if t - last_update_t > update_interval:
            pbar.update(1)
            last_update_t = t
        time.sleep(0.1)
relative_times, utils = zip(*util_list)
print(f"Average GPU util: {np.mean(utils)}")

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(111)
plt.title(f"GPU Utilization", fontsize=16)

plt.plot(relative_times, utils)
plt.xticks(fontsize=16)
plt.xlabel("Relative time (s)", fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("GPU Utilization", fontsize=16)
# plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig(f"tmp/{name}.png")