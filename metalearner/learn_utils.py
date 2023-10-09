
import os
from typing import Union, List
import torch
import pickle
import numpy as np

class CMOutput:
    def __init__(self, preds, embedding=None):
        self.preds: torch.Tensor = preds
        self.embedding = embedding
    
    def hard_clip(self, lower_bound):
        self.preds = torch.maximum(self.preds, lower_bound)
    
    @staticmethod
    def concat(output_list, dim: int):
        _preds = torch.cat([e.preds for e in output_list], dim)
        if output_list[0].embedding:
            _embedding = torch.cat([e.embedding for e in output_list], dim)
        else:
            _embedding = None
        return CMOutput(preds=_preds, embedding=_embedding)
    
    def concat_to(self, others):
        ### concat to another CMOutput in a in-place manner
        self.preds = torch.cat((self.preds, others.preds), axis=0)
        if self.embedding is not None and others.embedding is not None:
            self.embedding = torch.cat((self.embedding, others.embedding), axis=0)

    @property
    def device(self):
        return self.preds.device

    @property
    def shape(self):
        return self.preds.shape
    
    def mean(self, *args, **kwargs):
        return self.preds.mean(*args, **kwargs)
    
    def __getitem__(self, index):
        return self.preds[index]
    
    def de_standardize(self, metainfo):
        return metainfo.de_standardize_output(self.preds)

class CMFeature:
    def __init__(self, x_tir, x_device=None):
        self.x_tir = x_tir
        self.x_device = x_device
    
    def to(self, *args, **kwargs):
        self.x_tir = self.x_tir.to(*args, **kwargs)
        if self.x_device is not None:
            self.x_device = self.x_device.to(*args, **kwargs)
        return self
    
    def detach(self):
        self.x_tir = self.x_tir.detach()
        if self.x_device is not None:
            self.x_device = self.x_device.detach()
        return self
    
    def __len__(self):
        return len(self.x_tir)
