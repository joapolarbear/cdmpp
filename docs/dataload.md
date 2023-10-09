# Dataloader API

This doc will introduce how to specify the location to load data and how to perform partition to get training and test datasets.

## Requirement for data loading

### Specify the source of data
A list of device (optional) + sample200 | cross | list-1,2,3| networks-a-b

### Specify the method for data partition, i.e., how to decide the training and test dataset
1. Shuffle all samples and directly perform partition
2. Partition by device
3. Partition by DNN models
4. Fix the test dataset, change the training dataset


## Design

### API/argument design:
- `--mode` = `<source_mode>.<split_mode>`
- `<source_mode>` = `sample_mode1+sample_mode2+...+sample_mode3`
- `sample_mode` could be one of `samplek, cross, list-A-B-C` or `network-A-B`
- `split_mode` is optional, 
    - if not set, shuffle all loaded dataset and perform partition
    - Split by device, e.g., `by_device:train-t4-v100,test-a100`
    - Split by DNN models, e.g., `by_net:train-A-B,test-C`
    - [Fuzzy match] If `train-` is not specified, the remaining devices or nets belong to the training set, otherwise, `train-` is given explicitly, but without concrete devices or nets, it means the training set is empty

### Data Structure to store mode info

```
class Device2Task:
    def __init__(self):
        ''' Example
            "t4": {
                "tasks": [task1, task2],
                "root_path": root_path
                "attr": { 
                    "split": 0 # no specified split
                }
            }
        '''
        self.device2info: Dict[str, Dict] = {}
        self.data_split_mode = "default"
```

### Adapt data preprocess to this modification