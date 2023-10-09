import os
import re
import json
import time
from multiprocessing import Process

from utils.util import prompt
from .tune_space import GridAutotuneSpace
from .tuner import AutoTuner


class GridAutoTuner(AutoTuner):
    def __init__(self, process_num, *args, **kwargs):
        super(GridAutoTuner, self).__init__(*args, **kwargs)
        self.search_space = GridAutotuneSpace()
        self.process_num = process_num
        msg = f"GridAutoTuner is used, with {self.process_num} processes"
        prompt(msg)
        self.tune_state.log(msg)

    def _tuning_process(self, process_id):
        config_id = process_id
        while config_id < self.search_space.len:
            _config = self.search_space[config_id]
            if _config is None:
                continue
            _config["cuda_id"] = process_id
            _ = self.target_func(_config)
            config_id += self.process_num
        
    def grid_tune(self):
        processes = []
        for num in range(self.process_num):
            p = Process(target=self._tuning_process, args=(num,))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()


class FileGridAutoTuner(AutoTuner):
    ''' Grid tuning based on configuration files
    '''
    def __init__(self, cfg_files, *args, **kwargs):
        super(FileGridAutoTuner, self).__init__(*args, **kwargs)
        self.search_space = GridAutotuneSpace()
        self.process_num = 1
        self.all_cfgs = []
        for cfg_file in cfg_files:
            with open(cfg_file, 'r') as fp:
                self.all_cfgs += json.load(fp)

        ### read the current progress of search
        self.start_cfg_id = 0
        if os.path.exists(self.tune_state.log_filename):
            with open(self.tune_state.log_filename, 'r') as fp:
                lines = fp.readlines()
                for line in lines[::-1]:
                    if len(line) == 0:
                        continue
                    try:
                        dump_info = json.loads(line)
                        self.tune_state.best_target = dump_info["best_target"]
                        match = re.search(r"\[[\d]+/[\d]+\] (?P<config_id>[\d]+)/[\d]+", dump_info["config"]["progress"])
                        self.start_cfg_id = int(match["config_id"]) + 1
                    except (json.decoder.JSONDecodeError, KeyError):
                        continue
                    break
            
        msg = f"{self.workspace}: FileGridAutoTuner is used, start from the {self.start_cfg_id}th cfgs among {len(self.all_cfgs)} cfgs from {len(cfg_files)} files, current best error: {self.tune_state.best_target}"
        prompt(msg)
        dump_info = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "msg": msg
        }
        self.tune_state.log(json.dumps(dump_info))

    def _tuning_process(self, process_id):
        config_id = process_id
        while config_id < len(self.all_cfgs):
            if config_id < self.start_cfg_id:
                config_id += self.process_num
                continue
            _config = self.all_cfgs[config_id]
            _config["cuda_id"] = process_id
            _config["use_mse_loss"] = True
            _config["progress"] = f"[{process_id}/{self.process_num}] {config_id}/{len(self.all_cfgs)}"
            error = self.target_func(_config)
            config_id += self.process_num
        
    def grid_tune(self):
        processes = []
        assert self.process_num == 1
        self._tuning_process(0)
