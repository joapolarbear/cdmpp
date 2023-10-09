#!/bin/bash

cd
if [ -d "cross-device-perf-predictor" ]; then
    cd cross-device-perf-predictor && git checkout . && git pull
else
    git clone https://github.com/joapolarbear/cross-device-perf-predictor
    cd cross-device-perf-predictor
fi
bash ./tools/cutlass/cutlass_test.sh $@