''' Analyze the results of replay

Example:
python3 end2end/analyze.py \
    tmp/end2end/resnet_50-8/cuda_t4/replay_tenset.txt \
    tmp/end2end/resnet_50-8/cuda_t4/replay_via_profile_tenset.txt

'''
import sys
import re
import numpy as np
import pandas as pd

replay_rst_path = sys.argv[1]
replay_via_profile_rst_path = sys.argv[2]
print(f"Analyze the replay result, replay rst:{replay_rst_path}, via_profile: {replay_via_profile_rst_path}")

def read_replay_rst_df(_path, trial_name):
    df = pd.DataFrame({"task_id": [], "workload_key": [], "cost": [],
        "weight": [], "N_leaf": [], "src": []})
    iter_time = None
    with open(_path, 'r') as fp:
        for line in fp.readlines():
            if len(line) == 0:
                continue
            match = re.search(r"Estimated total latency of (?P<network_name>\w+): (?P<iter_time>[.\d]+) ms", line)
            if match:
                rst = match.groupdict()
                print(f"{trial_name} total latency of {rst['network_name']} is {float(rst['iter_time'])} ms")
                iter_time = float(rst['iter_time'])
                continue
            match = re.search(r"Task (?P<task_id>\d+) \(workload key: \[\"(?P<workload_key>\w+)\.{3}\): "
                r"T'=(?P<cost>[\d.]+), W=(?P<weight>\d+), N_leaf=(?P<N_leaf>(-1|[.\d]+))(, src=(?P<src>(random|tenset)))?", line)
            if match is None:
                import code
                code.interact(local=locals())
            rst = match.groupdict()
            if "src" not in rst:
                rst["src"] = "old_random"
            df.loc[len(df.index)] = [int(rst["task_id"]), rst["workload_key"], float(rst["cost"]), 
                int(rst["weight"]), int(rst["N_leaf"]), rst["src"]]
    return df, iter_time

df_replay, iter_time_replay = read_replay_rst_df(replay_rst_path, "Estimated")
df_profile, iter_time_profile = read_replay_rst_df(replay_via_profile_rst_path, "Profiled")

assert (df_replay["workload_key"] == df_profile["workload_key"]).all()
assert (df_replay["weight"] == df_profile["weight"]).all()

leaf_node_5_index = df_replay["N_leaf"] == 5
if "src" in df_replay:
    assert (df_replay["src"] == df_profile["src"]).all()
    tenset_index = df_replay["src"] != "random"
    valid_index = np.logical_and(tenset_index, leaf_node_5_index)
else:
    valid_index = leaf_node_5_index

T_prim_tenset = df_replay["cost"][valid_index]
T_tenset = df_profile["cost"][valid_index]
print(list(zip(T_tenset, T_prim_tenset)))
error = np.mean(np.abs(T_prim_tenset - T_tenset) / T_tenset) * 100
print(f"MAPE={error:.3f}%")
print(f"End2End Error={100 * (abs(iter_time_profile-iter_time_replay)/iter_time_profile):.3f}%")