import os, sys
import json
import numpy as np


dir_ = ".workspace/verify/2022-07-27-12:16:00/"
files = os.listdir(dir_)
for _file in files:
    if not _file.endswith("json"):
        continue
    with open(os.path.join(dir_, _file), 'r') as fp:
        # {
        #         "single_task": single_task_error.tolist(),
        #         "cross_task": cross_task_error.tolist()
        #     },
        rst = json.load(fp)
    single_task_error = rst["single_task"]
    cross_task_error = rst["cross_task"]
    total_sample_num = len(single_task_error)
    _str = f"\n{_file}: {total_sample_num} test samples"
    # _str += "\nFormat: each line is in the following format: B C: a (r) b (r), c(r), which means that, "
    # _str += "For a test set, there are `c` test samples that has an MAPE > B"
    # _str += " in single-task learning and an MAPE < C in cross-task learning, "
    # _str += "The ratio is r. Tensor a and b denotes the cnt <C, C to B and >B"
    for lower_before, upper_after in [(0.2, 0.1)]:
        cnt4single = [0, 0, 0]
        cnt4cross = [0, 0, 0]
        flip_cnt = 0.
        for i in range(total_sample_num):
            if single_task_error[i] >= lower_before:
                cnt4single[2] += 1
            elif single_task_error[i] >= upper_after:
                cnt4single[1] += 1
            else:
                cnt4single[0] += 0
            
            if cross_task_error[i] >= lower_before:
                cnt4cross[2] += 1
            elif cross_task_error[i] >= upper_after:
                cnt4cross[1] += 1
            else:
                cnt4cross[0] += 0

            if single_task_error[i] >= lower_before and cross_task_error[i] < upper_after:
                flip_cnt += 1
        cnt4single = np.array(cnt4single)
        cnt4cross = np.array(cnt4cross)
        _str += f"\n{lower_before} {upper_after}: "
        _str += f"{cnt4single}({cnt4single/total_sample_num}) -> "
        _str += f"{cnt4cross}({cnt4cross/total_sample_num}), "
        _str += f"{flip_cnt} ({flip_cnt/total_sample_num:.6f}%)"
        _str += f" ==> {flip_cnt/cnt4single[2]:.6f} %"
    with open(os.path.join(dir_, f"updated_improvent_rate.txt"), 'w') as fp:
        fp.write(_str)
    print(_str)