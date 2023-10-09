import os
import re
import subprocess
import shutil
import sys

import dpro

# CM_PATH = "/mnt/bn/hphu-tenset/tmp/20221119_autotune_trial_1575-y_norm_0/cm/BaseLearner"
# DEVICE = sys.argv[1]
# CM_PATH = sys.argv[2]

def mape_error(Y_p, Y):
    return abs(Y_p - Y) / Y

class FILENAME:
    measure_rst = "measure.txt"
    replay_tenset = "replay_tenset.txt"
    replay_via_tenset = "replay_via_tenset_tenset.txt"
    replay_via_profile = "replay_via_profile_tenset.txt"
    task2state = "task2state.records"

def parse_rst(rst_dir):
    with open(os.path.join(rst_dir, FILENAME.measure_rst), 'r') as fp:
        lines = fp.readlines()
        # rst = re.search(r" *(?P<mean>[\d.]+) *(?P<median>[\d.]+) *(?P<max>[\d.]+) *(?P<min>[\d.]+) *(?P<std>[\d.]+)", lines[3]).groupdict()
        rst = re.search(r"(?P<mean>[\d.]+)", lines[2]).groupdict()
        T_measure = eval(rst["mean"])
        # print(T_measure)

    with open(os.path.join(rst_dir, FILENAME.replay_tenset), 'r') as fp:
        lines = fp.readlines()
        rst = re.search(r": (?P<time>[\d.]+) ms", lines[0]).groupdict()
        T_predict = eval(rst["time"])
        # print(T_predict)

    with open(os.path.join(rst_dir, FILENAME.replay_via_tenset), 'r') as fp:
        lines = fp.readlines()
        rst = re.search(r": (?P<time>[\d.]+) ms", lines[0]).groupdict()
        T_tenset = eval(rst["time"])

    with open(os.path.join(rst_dir, FILENAME.replay_via_profile), 'r') as fp:
        lines = fp.readlines()
        rst = re.search(r": (?P<time>[\d.]+) ms", lines[0]).groupdict()
        T_profile = eval(rst["time"])
        # print(T_profile)
    
    return T_measure, T_profile, T_tenset, T_predict


RST_DIR = "tmp/end2end"
BEST_RST_DIR = "tmp/end2end_best"; os.makedirs(BEST_RST_DIR, exist_ok=True)
networks = [
    'resnet_18',
    'resnet_50',
    'resnet3d_18',
    'mobilenet_v2', 'mobilenet_v3',
    'wide_resnet_50', 'resnext_50',
    'densenet_121',
    'inception_v3',
    'bert_tiny',
    'bert_base',
    'bert_medium', 'bert_large',
    'dcgan',
]
batch_sizes = [1, 4, 8]

def wrap_shell(command):
    if isinstance(command, str):
        command = command.split(" ")
    try:
        ret = subprocess.run(command, check=True, shell=False)
    except:
        return -1

    return 0

def search_proper_schedule(_network, _bs):
    network_dir = f"{RST_DIR}/{_network}-{_bs}/cuda_{DEVICE.lower()}"
    replay_command = (f"bash scripts/replay.sh -y --cache_dir {CM_PATH} "
        f"--networks {_network} --batch_sizes {_bs} -g {DEVICE}")

    for trial in range(100):
        if os.path.exists(os.path.join(network_dir, FILENAME.task2state)):
            os.remove(os.path.join(network_dir, FILENAME.task2state))

        ret = wrap_shell(f"{replay_command} --replay_mode measure,replay_via_profile,replay_via_tenset,replay")
        if ret != 0:
            continue
        # wrap_shell(f"{replay_command} --replay_mode replay_via_tenset,replay")
        
        T_measure, T_profile, T_tenset, T_predict = parse_rst(network_dir)
        E_profile = mape_error(T_profile, T_measure)
        E_tenset = mape_error(T_tenset, T_measure)
        E_predict = mape_error(T_predict, T_measure)

        print(dpro.base.bcolors.CYELLOWBG + f"Trial {trial}, T_measure={T_measure:.3f}, "
            f"T_profile={T_profile: .3f}({100*E_profile:.1f}%), "
            f"T_tenset={T_tenset: .3f}({100*E_tenset:.1f}%), "
            f"T_predict={T_predict: .3f}({100*E_predict:.1f}%)" + dpro.base.bcolors.ENDC)
        print()

        if E_predict < 0.1:
            best_dir = f"{BEST_RST_DIR}/{_network}-{_bs}/cuda_{DEVICE.lower()}"
            shutil.copytree(os.path.dirname(network_dir), os.path.join(BEST_RST_DIR, f"{_network}-{_bs}"))
            break


def sample_and_measure_only(_network, _bs):
    network_dir = f"{RST_DIR}/{_network}-{_bs}/cuda_{DEVICE.lower()}"
    replay_command = (f"bash scripts/replay.sh -y --cache_dir {CM_PATH} "
        f"--networks {_network} --batch_sizes {_bs} -g {DEVICE}")

    if os.path.exists(os.path.join(network_dir, FILENAME.task2state)):
        os.remove(os.path.join(network_dir, FILENAME.task2state))

    ret = wrap_shell(f"{replay_command} --replay_mode measure")

# search_proper_schedule("resnet_50", 1)
# search_proper_schedule("resnet_50", 4)
# search_proper_schedule("resnet_50", 8)

# sample_and_measure_only("inception_v3", 1)
# sample_and_measure_only("inception_v3", 4)
# sample_and_measure_only("inception_v3", 8)

# sample_and_measure_only("bert_base", 1)
# sample_and_measure_only("bert_base", 4)
# sample_and_measure_only("bert_base", 8)

if __name__ == '__main__':
    _network = sys.argv[1]
    _bs = int(sys.argv[2])
    DEVICE = sys.argv[3]
    CM_PATH = sys.argv[4]
    search_proper_schedule(_network, _bs)
