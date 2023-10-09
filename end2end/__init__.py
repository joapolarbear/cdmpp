''' End2end Perfomrance Measurement and Estimation

# Prepare tasks, task_weights, task2state mapping
launch --type t4 -- bash scripts/train.sh replay -y \
    --cache_dir .workspace/runs/20221120_autotune_trial_603/cm/BaseLearner \
    -i .workspace/ast_ansor2 \
    --mode sample200,network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8 \
    --ave_lb 0 \
    --replay_mode preprocess

# Measure end2end iteration time
launch --type t4 -- bash scripts/train.sh replay -y \
    --cache_dir .workspace/runs/20221120_autotune_trial_603/cm/BaseLearner \
    -i .workspace/ast_ansor2 \
    --mode sample200,network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8 \
    --ave_lb 0 \
    --replay_mode measure

... --replay_mode measure_by_task
... --replay_mode breakdown

# Estimate end2end iteration time
launch --type t4 -- bash scripts/train.sh replay -y \
    --cache_dir .workspace/runs/20221120_autotune_trial_603/cm/BaseLearner \
    -i .workspace/ast_ansor2 \
    --mode sample200,network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8 \
    --ave_lb 0 \
    --replay_mode replay

... --replay_mode replay_via_profile

# Verify learner by task
launch --type t4 -- bash scripts/train.sh replay -y \
    --cache_dir .workspace/runs/20221120_autotune_trial_603/cm/BaseLearner \
    -i .workspace/ast_ansor2 \
    --mode sample200,network-resnet_50_bs1-resnet_50_bs4-resnet_50_bs8 \
    --ave_lb 0 \
    --replay_mode test_by_task
    
'''

from .replay import estimaet_network
from .preprocess import offline_preprocess
from .measure import measure_network, measure_by_task

from .test_learner_by_task import test_learner_on_network