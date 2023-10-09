import os

from utils.base import DIMENSION_NAME
from utils.util import Filters
from utils.env import PROJECT_CFG
from utils.device_info import short_device_name

from metalearner.learner import parse_metric_learner

from dataloader import collect_data

def metric_learning_w_op_level_data(project_cfg, metric_learner, force_load_data, opt, op_type):
    trace_root_path = os.path.join(project_cfg["source_data_root"], "op_level_trace")
    root_dir = "{}/op_database".format(trace_root_path)
    source_gpu = short_device_name(project_cfg.get("source_gpu", "Tesla_V100-SXM2-32GB"))
    target_gpu = short_device_name(project_cfg.get("source_gpu", "Tesla_V100-SXM2-32GB"))
    AVE_LOWER_BOUND_MS = float(project_cfg.get("AVE_LOWER_BOUND_MS", 0.1))
    target_dtype = project_cfg.get("dtype", "fp32")
    filters_pre_process = Filters({
        DIMENSION_NAME.dtype: [target_dtype],
        DIMENSION_NAME.gpu_model: [source_gpu, target_gpu]
    })
    op2xydata = collect_data(
        root_dir, root_dir, filters_pre_process, 
        force=force_load_data)

    learner = parse_metric_learner(metric_learner)(opt_type=opt)
    learner.train(
        op2xydata,
        ave_lower_bound_ms=AVE_LOWER_BOUND_MS,
        one_op_type=op_type
    )