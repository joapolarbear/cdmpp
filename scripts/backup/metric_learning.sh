#!/bin/bash

python3 main.py --log_level=info --metric_learner=0
python3 main.py --log_level=info --metric_learner=0 --op_type="Conv2D"
python3 main.py --log_level=info --metric_learner=0 --source_data="op"
python3 main.py --log_level=info --metric_learner=0 --op_type="Conv2D" --source_data="op"

python3 main.py --log_level=info --metric_learner=1
python3 main.py --log_level=info --metric_learner=1 --op_type="Conv2D"
python3 main.py --log_level=info --metric_learner=1 --source_data="op"
python3 main.py --log_level=info --metric_learner=1 --op_type="Conv2D" --source_data="op"

python3 main.py --log_level=info --metric_learner=2
python3 main.py --log_level=info --metric_learner=2 --op_type="Conv2D"
python3 main.py --log_level=info --metric_learner=2 --source_data="op"
python3 main.py --log_level=info --metric_learner=2 --op_type="Conv2D" --source_data="op"
