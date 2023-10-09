from multiprocessing import Process, Lock, Manager
import time
from typing import List
import os
import dpro
import shutil

def CHECK_EXIST(path):
    _dir = os.path.dirname(path) if os.path.isfile(path) else path
    os.makedirs(_dir, exist_ok=True)
    return _dir

def wrap_remove(path):
    if os.path.islink(path):
        os.remove(path)
    else:
        shutil.rmtree(path)

def sampler_process(learning_params, pipe_cnt, permanent_vars: dict):
    ''' Sample raw features from raw data
        * If data is not in the local machine, download it from the remote server
    '''
    # remote_raw_data_path = xxx
    remote_raw_reature_path = "mnt:/mnt/bn/hphu-tenset/cdpp/data/raw_features/ast_ansor"
    pipeline_tmp_dir = CHECK_EXIST(os.path.join(permanent_vars["ws"], f"{pipe_cnt}"))
    local_raw_feature_path = os.path.join(pipeline_tmp_dir, "raw_feature")

    if os.path.exists(local_raw_feature_path):
        return

    if remote_raw_reature_path.startswith("mnt:"):
        ### Mount raw features to local
        mounted_path = remote_raw_reature_path.split(":")[1]
        os.system(f"ln -sf {mounted_path} {local_raw_feature_path}")
    else:
        raise ValueError(remote_raw_reature_path)
    
def preprocess_process(learning_params, pipe_cnt, permanent_vars: dict):
    ''' Preprocess raw features and cache them
        Remove raw features
    '''
    pipeline_tmp_dir = CHECK_EXIST(os.path.join(permanent_vars["ws"], f"{pipe_cnt}"))
    local_raw_feature_path = os.path.join(pipeline_tmp_dir, "raw_feature")
    local_preprocessed_path = os.path.join(pipeline_tmp_dir, "preprocessed")

    if os.path.exists(local_preprocessed_path):
        return
    
    ### Preprocess raw features and cache them
    from metalearner.data.preprocess import make_dataset
    from tvm_helper.metadata import ASTMetaInfo
    learning_params["input"] = local_raw_feature_path
    os.environ["CDPP_DATASET_PATH"] = local_preprocessed_path
    data_meta_info = ASTMetaInfo.load_init(
        os.path.join(learning_params["input"], "metainfo.pickle"))
    make_dataset(learning_params, data_meta_info=data_meta_info, verbose=False)

    ### Remove raw features
    wrap_remove(local_raw_feature_path)

def learning_process(learning_params, pipe_cnt, permanent_vars: dict):
    ''' Load cached pre-processed data and train the cost model
    '''
    pipeline_tmp_dir = CHECK_EXIST(os.path.join(permanent_vars["ws"], f"{pipe_cnt}"))
    local_preprocessed_path = os.path.join(pipeline_tmp_dir, "preprocessed")

    ### Training
    from metalearner.data.dataloader import load_iter_dataset
    from metalearner.learner import _metric_learning_w_tir_data_impl

    os.environ["CDPP_DATASET_PATH"] = local_preprocessed_path
    learning_params["finetune_epoch"] = 32
    verbose = True
    ds_pair, data_meta_info = load_iter_dataset(learning_params)
    if "learner" in permanent_vars:
        learning_params["load_cache"] = False
        permanent_vars["learner"].train(ds_pair, 
            verbose=verbose, learning_params=learning_params)
    else:
        learner, rst = _metric_learning_w_tir_data_impl(ds_pair, 
            data_meta_info, learning_params, verbose=verbose)
        permanent_vars["learner"] = learner

    ### Remove cached preprocessed data
    wrap_remove(pipeline_tmp_dir)

IDLE_SLEEP = 0.1

class PipelineNode:
    def __init__(self, name, target, pipeline_ws):
        self.prev = None
        self.succ = None
        
        self.name = name
        self.target = target
        self.pipeline_ws = pipeline_ws

        self.global_state = None
        self.state = None
        
        self.proc = None
    
    def pipeline_process(self, prev_state, succ_state):
        pipe_cnt = 0
        permanent_vars = {"ws": self.pipeline_ws}
        process_state = self.state
        while not self.global_state["shutdown"]:
            if (prev_state is None or process_state["prev_done"]) and (succ_state is None or process_state["succ_start"]):
                ### Start phase
                if prev_state:
                    process_state["prev_done"] = False
                    prev_state["succ_start"] = True
                if succ_state:
                    process_state["succ_start"] = False
                print(dpro.base.bcolors.CGREENBG + f"<<<{self.name}{pipe_cnt}>>>" + dpro.base.bcolors.ENDC)

                ### Running
                self.target(pipe_cnt, permanent_vars)

                ### End phase
                pipe_cnt += 1
                if succ_state:
                    succ_state["prev_done"] = True
            else:
                time.sleep(IDLE_SLEEP)
        print(dpro.base.bcolors.CYELLOWBG + f"{self.name} Shutdown>>>" + dpro.base.bcolors.ENDC)
    
    def start(self):
        assert self.state is not None and self.global_state is not None
        prev_state = None if self.prev is None else self.prev.state
        succ_state = None if self.succ is None else self.succ.state
        self.proc = Process(target=self.pipeline_process, args=(prev_state, succ_state))
        self.proc.start()
    
    def join(self):
        self.proc.join()

class Pipeline:
    def __init__(self, pipeline_ws="tmp/pipeline"):
        self.pipeline_ws = pipeline_ws
        self.manager = Manager()
        self.global_state = self.manager.dict({'shutdown': False})

        self.nodes: List[PipelineNode] = []
        self.fixed = False
    
    def start(self):
        assert not self.fixed
        print(dpro.base.bcolors.CYELLOWBG + f"<<<Start pipeline at {self.pipeline_ws}" + dpro.base.bcolors.ENDC)
        for node in self.nodes:
            node.start()
        self.fixed = True
    
    def shutdown(self):
        self.global_state["shutdown"] = True
        for node in self.nodes:
            node.join()
        self.manager.shutdown()
        wrap_remove(self.pipeline_ws)
    
    def add_node(self, name, target):
        node = PipelineNode(name, target, self.pipeline_ws)
        node.global_state = self.global_state
        node.state = self.manager.dict({"prev_done": False, "succ_start": True})

        if len(self.nodes) == 0:
            node.state["prev_done"] = True
        else:
            self.nodes[-1].succ = node
            node.prev = self.nodes[-1]
        self.nodes.append(node)
        return self


def test_pipeline():
    pipe = Pipeline()
    pipe.add_node("S", lambda pipe_cnt, permanent_vars: time.sleep(1))
    pipe.add_node("P", lambda pipe_cnt, permanent_vars: time.sleep(2))
    pipe.add_node("L", lambda pipe_cnt, permanent_vars: time.sleep(3))

    pipe.start()
    time.sleep(30)
    pipe.shutdown()


def pipeline_learning(learning_params):
    pipeline_ws = os.path.join(learning_params["tb_logdir"], "pipeline")
    if os.path.exists(pipeline_ws):
        wrap_remove(pipeline_ws)
    pipe = Pipeline(pipeline_ws)
    pipe.add_node("Sampler", lambda pipe_cnt, permanent_vars: sampler_process(learning_params, pipe_cnt, permanent_vars))
    pipe.add_node("Pre-processor", lambda pipe_cnt, permanent_vars: preprocess_process(learning_params, pipe_cnt, permanent_vars))
    pipe.add_node("Learner", lambda pipe_cnt, permanent_vars: learning_process(learning_params, pipe_cnt, permanent_vars))

    pipe.start()
    time.sleep(3600 * 24 * 7)
    pipe.shutdown()


if __name__ == "__main__":
    test_pipeline()