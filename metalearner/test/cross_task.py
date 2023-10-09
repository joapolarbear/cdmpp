import os
import sys
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir, os.pardir))
target_path = os.path.abspath(
    os.path.join(project_root, ".workspace", "group_summary"))
print(target_path)
sys.path.append(project_root)
from utils.util import fig_base
sns.set_theme(style="whitegrid")
fontdict = {"color": "darkred",
        "size": 18,
        "family": "serif"}

class Result:
    def __init__(self):
        self.cluster_ids = []
        self.cluster_errors = []
    @property
    def cluster_num(self):
        return len(self.cluster_ids)
    @property
    def average(self):
        return np.average(self.cluster_errors)
    @property
    def max(self):
        return max(self.cluster_errors)
    def record(self, cluster_id, error):
        self.cluster_ids.append(cluster_id)
        self.cluster_errors.append(error)
    def to_df(self):
        return pd.DataFrame({
            "cluster_id": self.cluster_ids,
            "error": self.cluster_errors
        })

rst_dict = {
    1: Result(),
    3: Result(),
    5: Result(),
    10: Result(),
    50: Result()
}
task_files = []
def read_log(file_name, cluster_num, first=False):
    with open(file_name, 'r') as fp:
        for line in fp.readlines():
            if len(line) == 0:
                continue
            if first and line.startswith("file"):
                match = re.search(r"file .workspace/tenset/(?P<task_name>t4_(?P<task_id>[\d]+).npy), error=(?P<error>[\d.]+)", line)
                # print(match["task_id"], match["error"])
                task_files.append(match["task_name"])
                rst_dict[50].record(match["task_id"], float(match["error"]))
            elif first and line.startswith("Cross-task"):
                match = re.search(r"Cross-task, error=(?P<error>[\d.]+)", line)
                # print("All", match["error"])
                rst_dict[1].record(0, float(match["error"]))
            elif line.startswith("Cluster"):
                match = re.search(r"Cluster (?P<cluster_id>[\d]+), error=(?P<error>[\d.]+)", line)
                # print(match["cluster_id"], match["error"])
                rst_dict[cluster_num].record(match["cluster_id"], float(match["error"]))
            else:
                continue

file1 = os.path.join(target_path, "log_50task_to_10cluster.txt")
file2 = os.path.join(target_path, "log_50task_to_5cluster.txt")
file3 = os.path.join(target_path, "log_50task_to_3cluster.txt")
read_log(file1, 10, first=True)
read_log(file2, 5)
read_log(file3, 3)


### Visualize the results
fig = plt.figure(figsize=(12, 10))
_fig_base = fig_base(4, row_first=False)

ax = fig.add_subplot(_fig_base+1)
sns.barplot(x="cluster_id", y="error", data=rst_dict[50].to_df())
plt.text(0, 0.9*rst_dict[50].max, s=f"Average Error = {rst_dict[50].average:.3f}", fontdict=fontdict)
plt.text(0, 0.8*rst_dict[50].max, s=f"Cross-Task Training Error = {rst_dict[1].average:.3f}", fontdict=fontdict)
ax.set_xlabel('Task ID')
ax.set_ylabel('Error')
ax.title.set_text(f"Single task learning error with {len(task_files)} tasks")

ax = fig.add_subplot(_fig_base+2)
sns.barplot(x="cluster_id", y="error", data=rst_dict[10].to_df())
plt.text(0, 0.9*rst_dict[10].max, s=f"Average Error = {rst_dict[10].average:.3f}", fontdict=fontdict)
ax.set_xlabel('Cluster ID')
ax.set_ylabel('Error')
ax.title.set_text(f"Divide samples from {len(task_files)} tasks into {rst_dict[10].cluster_num} clusters")

ax = fig.add_subplot(_fig_base+3)
sns.barplot(x="cluster_id", y="error", data=rst_dict[5].to_df())
plt.text(0, 0.9*rst_dict[5].max, s=f"Average Error = {rst_dict[5].average:.3f}", fontdict=fontdict)
ax.set_xlabel('Cluster ID')
ax.set_ylabel('Error')
ax.title.set_text(f"Divide samples from {len(task_files)} tasks into {rst_dict[5].cluster_num} clusters")

ax = fig.add_subplot(_fig_base+4)
sns.barplot(x="cluster_id", y="error", data=rst_dict[3].to_df())
plt.text(0, 0.9*rst_dict[3].max, s=f"Average Error = {rst_dict[3].average:.3f}", fontdict=fontdict)
ax.set_xlabel('Cluster ID')
ax.set_ylabel('Error')
ax.title.set_text(f"Divide samples from {len(task_files)} tasks into {rst_dict[3].cluster_num} clusters")

plt.tight_layout()
plt.savefig(os.path.join(project_root, ".fig", "metalearner", f"cross_task.png"))
