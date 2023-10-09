#!/bin/bash
set -x

export PATH=/usr/local/cuda/bin:$PATH

python3 metalearner/feature/habana_gen_feature.py \
    ~/habana_tf_1_5 \
    ~/habana_feature