import os
import numpy as np
import sys

DEVICE = sys.argv[1]

all_y = np.empty(0)
for task_id in range(100):
    task_feature_path = os.path.join(".workspace", "ast_ansor", DEVICE.lower(), f"{task_id}.npy")
    if not os.path.exists(task_feature_path):
        print(f"Path {task_feature_path} does NOT exist")
        continue
    data = np.load(task_feature_path, allow_pickle=True)
    all_y = np.concatenate((all_y, data[:, 0]), axis=0)

print("Average cost: ", np.average(all_y))
print("Min cost: ", np.min(all_y))
print("Max cost: ", np.max(all_y))