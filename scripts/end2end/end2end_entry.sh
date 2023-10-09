#!/bin/bash

PROJECT_PATH=$(realpath ./)

cd $PROJECT_PATH/scripts/end2end/
echo $PWD
python3 recur_test.py $@
cd -