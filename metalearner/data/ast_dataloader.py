import numpy as np
from typing import Union
import random

import torch

from .rawdata import TRAIN2ALL_RATIO, RawData

class ASTNode:
    ''' Node in an AST

    Members
    -------
    node_id: int
        Node ID
    childs: List(ASTNode)
        Child nodes of this node
    parent: Union[ASTNode, None]
        Paraent node
    is_leaf: bool
        Set True if the node is a leaf node, which needs computation
    indice_in_featuers: np.array
        Since the feature extractor Return
            * ast_features: an N_leaf x N_entry array
            * node_ids: an N_leaf array denotes node_ids of leaf nodes corresponding to ast_features
            * serialized_tree: an 1D array
        So indice_in_featuers is used to index ast_features, node_ids[indice_in_featuers] = self.node_id
    '''
    def __init__(self, node_id):
        self.node_id = int(node_id)
        self.childs = []
        self.parent = None
        self.is_leaf = False
        self.indice_in_featuers: Union[torch.Tensor, None] = None
    
    def add_child(self, child):
        self.childs.append(child)
        child.parent = self
    
    def __str__(self):
        return f"{self.node_id}"

    def get_tree_footprint(self):
        footprint = '<BL' + str(self.node_id)
        if len(self.childs) == 0:
            footprint += '['
            if hasattr(self, 'indice_in_featuers') and self.indice_in_featuers is not None:
                for idx in self.indice_in_featuers:
                    footprint+='CI'+str(int(idx))
            footprint += ']'
        for child in self.childs:
            footprint += child.get_tree_footprint()
        footprint += 'EL' + str(self.node_id)+'>'
        return footprint


class AST(torch.Tensor):
    '''https://discuss.pytorch.org/t/subclassing-torch-tensor/23754'''
    @staticmethod
    def __new__(cls, serialized_tree, idx_node_ids, train_device, *args, **kwargs):
        return super().__new__(cls, serialized_tree, *args, **kwargs)

    def __init__(self, serialized_tree, idx_node_ids, train_device):
        self.serialized_tree = serialized_tree
        self.root = None
        self.pre_order_ptr = None
        self.idx_node_ids = list(idx_node_ids)
        self.train_device = train_device

    def pre_order_add_node(self, node_id):
        if not isinstance(node_id, ASTNode):
            node = ASTNode(node_id)
        else:
            node = node_id
        
        if node.node_id in self.idx_node_ids:
            assert node.indice_in_featuers is None
            node.indice_in_featuers = torch.tensor([self.idx_node_ids.index(node.node_id)]).to(self.train_device)
            node.is_leaf = True

        if self.root is None:
            assert node.node_id != -1
            self.root = node
            self.pre_order_ptr = node
        else:
            if node.node_id == -1:
                self.pre_order_ptr = self.pre_order_ptr.parent
            else:
                self.pre_order_ptr.add_child(node)
                self.pre_order_ptr = node
    
    def draw_ast(self, prefix: str, node: Union[ASTNode, None], isLast: bool):
        ret = ""
        if node is not None:
            ret += prefix
            ret += ("└──" if isLast else "├──")
            ret += str(node) + "\n"
            for idx, child in enumerate(node.childs):
                ret += self.draw_ast(prefix+("     " if isLast else "|    "), child, idx == (len(node.childs)-1))
        return ret

    def __str__(self):
        ret = "********** AST **********\n"
        ret += self.draw_ast("", self.root, True)
        return ret
    
    def get_tree_footprint(self):
        return self.root.get_tree_footprint()
    
    @staticmethod
    def deserialize_tree(serialized_tree, node_ids, train_device):
        ast = AST(serialized_tree, node_ids, train_device)
        ast.serialized_tree = "_".join([str(x) for x in serialized_tree])
        for node_id in serialized_tree:
            ast.pre_order_add_node(node_id)
        assert ast.pre_order_ptr is None
        return ast


class ASTDataset(torch.utils.data.Dataset):
    ''' Implement the similar dataset as test_tiramisu
        Note: only samples with the same tree architecture are batched
    '''
    def __init__(self, raw_data):
        ### The raw_data has been preprocessed
        self.raw_data = raw_data

        self.X = []
        self.Y = []
        self.batched_exec_time = []
        self.batches_dict = dict()

        self.deserialized = False
    
    def deserialize(self, max_batch_size, train_device, store_device):
        '''
            1. Deserialize the AST and features
            2. Normalize input features and standardize outputs
            3. Batch samples with the same tree architecture
        '''
        assert not self.deserialized
        for _data in self.raw_data.raw_data:
            avg, std, flops, ast_features, node_ids, serialized_tree = _data
            tree = AST.deserialize_tree(serialized_tree, node_ids, train_device)
            # tree_footprint = tree.get_tree_footprint()
            tree_footprint = str(tree)
            self.batches_dict[tree_footprint] = self.batches_dict.get(
                tree_footprint, 
                {
                    'tree': tree,
                    'ast_features': [],
                    'exec_time_list': []
                })
            self.batches_dict[tree_footprint]['ast_features'].append(np.array(ast_features))
            self.batches_dict[tree_footprint]['exec_time_list'].append(avg)
        
        storing_device = store_device
        for tree_footprint in self.batches_dict:
             
            ### Perform normalization and standardization
            self.batches_dict[tree_footprint]['ast_features'] = [self.raw_data.metainfo.norm_input(tensor) for tensor in self.batches_dict[tree_footprint]['ast_features']]
            self.batches_dict[tree_footprint]['exec_time_list'] = self.raw_data.metainfo.standardize_output(
                self.batches_dict[tree_footprint]['exec_time_list'])

            ### Group data as batches
            for chunk in range(0, len(self.batches_dict[tree_footprint]['exec_time_list']), max_batch_size):
                # Check GPU memory in order to avoid Out of memory error
                try:
                    if storing_device.type=='cuda': 
                        if ((torch.cuda.memory_allocated(storing_device.index)/torch.cuda.get_device_properties(storing_device.index).total_memory)>0.80):
                            print('GPU memory on '+str(storing_device)+' nearly full, switching to CPU memory')
                            storing_device = torch.device('cpu')

                    self.batched_exec_time.append(
                        self.batches_dict[tree_footprint]['exec_time_list'][chunk:chunk+max_batch_size])
                    
                    
                    self.X.append((
                        self.batches_dict[tree_footprint]['tree'],
                        torch.FloatTensor(np.array(self.batches_dict[tree_footprint]['ast_features'][chunk:chunk+max_batch_size])).to(storing_device)))
                    self.Y.append(
                        torch.FloatTensor(self.batches_dict[tree_footprint]['exec_time_list'][chunk:chunk+max_batch_size]).to(storing_device))
                except:
                    import pdb; pdb.set_trace()
        self.deserialized = True
        print(f'Number of batches {len(self.Y)}, batch size <= {max_batch_size}')  
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            ### x, y, di
            return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)


def load_ast_dataset(raw_data, train_device, store_device, max_batch_size=2048):
    dataset = ASTDataset(raw_data)
    dataset.deserialize(max_batch_size, train_device, store_device)

    split_ratio = 1 - TRAIN2ALL_RATIO
    validation_size = int(split_ratio * len(dataset))
    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    val_batches_indices, train_batches_indices = indices[:validation_size],\
                                               indices[validation_size:]
    val_batches_list = []
    train_batches_list = []
    for i in val_batches_indices:
        val_batches_list.append(dataset[i])
    for i in train_batches_indices:
        train_batches_list.append(dataset[i])
    print("Data loaded. Sizes: " + str((len(val_batches_list), len(train_batches_list))) + " (val, train) batches")
    return dataset, val_batches_list, val_batches_indices, train_batches_list, train_batches_indices
