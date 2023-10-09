## Dataset organization

The dataset is stored under `<ProjectDir>/scripts/dataset` folder.

- dataset
  - `dataset/network_info`: The metadata for networks
     - `*.relay.pkl`: The relay IR of a network. One network per file.
         - For example, `('resnet_50', [(1, 3, 224, 224)]).relay.pkl` contains the relay IR of resnet_50 with input shape (1, 3, 224, 224).
     - `*.task.pkl`: The tasks and their weights in a network. One (network, targte) pair per file.
         - For example, `(('resnet_50', [(1, 3, 224, 224)]), 'llvm').task.pkl` contains the all tasks of resnet_50 on llvm backend.
  - `dataset/to_measure_programs`: The generated random programs for measurement
     - `all_tasks.pkl`: A file containing all tasks. It is used an an index for all tasks.
     - `*.json`: The randomly generated programs (schedules) for measurement. One file per task.
  - `dataset/measure_records`:
     - `e5-2666/*.json`: measurement records collected on an Intel e5-2666.
     - ...: 

## Data Collection Procedure

1. (about 30 mins) Dump metadata of all networks. The metadata includes all tasks and relay IR of a network.
```
python3 dump_network_info.py
```
The relay IR is stored at `dataset/network_info/{clean_name(network_key)}.relay.pkl` and the task info is stored at `dataset/network_info/{clean_name(network_task_key)}.task.pkl` by task. File `dataset/network_info/all_tasks.pkl` stores all metadata of all tasks

2. (about 30 mins) Dump all programs for measurement
```
python3 dump_programs.py
```
For each task, the results of different 'states/schedulings' are stored at
`dataset/to_measure_programs/{clean_name(task_key)}.json`

3. Measure all programs
```
python3 measure_programs.py
```
For each task, the measurement results are stored at `dataset/measure_records/{target.model}/{clean_name(task_key)}.json`

## Cost Model Training

1. Create training dataset for cost model training
```
python3 make_dataset.py
```

2. Train the cost model
```
python3 train_model.py
```
