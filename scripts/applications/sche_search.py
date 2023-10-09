import os
import re
import subprocess
import shutil
import sys

import dpro

# CM_PATH = "/mnt/bn/hphu-tenset/tmp/20221119_autotune_trial_1575-y_norm_0/cm/BaseLearner"
DEVICE = sys.argv[1]
CM_PATH = sys.argv[2]

def wrap_shell(command):
    if isinstance(command, str):
        command = command.split(" ")
    try:
        ret = subprocess.run(command, check=True, shell=False)
    except:
        return -1

    return 0

_command = f"bash scripts/train.sh sche_search -y --cache_dir {CM_PATH} "
wrap_shell(_command)

