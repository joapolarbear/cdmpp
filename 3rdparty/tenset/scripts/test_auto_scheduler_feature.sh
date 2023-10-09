#!/bin/bash
usrname=$(whoami)
if [ $usrname != "root" ]; then
    sudo_prefix="sudo"
else
    sudo_prefix=""
fi

cd build && make -j8 && cd ../python && $sudo_prefix python3 setup.py install
python3 ../tests/python/unittest/test_auto_scheduler_feature.py

