#!/bin/bash

set -x

# bash scripts/train.sh run -y \
#     --mode '(t4:random32),(v100:random32),(p100:random32),(a100:random32),(k80:random32)' \
#     -c tmp/search_trial_20221119_1575.yaml \
#     -t .workspace/runs/cdpp-pipeline-add_z_d-fix_batch_first_bug \
#     --pipeline \
#     $@
    
### Apply CMD-based regularizer
bash scripts/train.sh run -y \
    --mode '(t4:random32),(v100:random32),(p100:random32),(a100:random32),(k80:random32)' \
    -c tmp/search_trial_20221119_1575-use_cmd.yaml \
    -t .workspace/runs/cdpp-pipeline-add_z_d-fix_batch_first_bug-cmd \
    --pipeline \
    $@