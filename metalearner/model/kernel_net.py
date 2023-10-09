import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .base_net import MLP

class Kernel(nn.Module):
    def __init__(self, kernel_size):
        super(Kernel, self).__init__()

        self.L = Variable(
            torch.rand(kernel_size, kernel_size),
                        requires_grad=True).cuda()
        self.kernel_size = kernel_size

    def forward(self, support_x, support_y):
        '''
        Parameters:
        ----
        support_x: shape = (n_sample1, n_feature)
        support_y: shape = (n_sample2, n_feature)

        Returns:
        ----
        distance: shape = (n_sample2, n_sample1)
        '''
        distance = []
        for sample in support_y:
            _tmp = []
            for nei in support_x:
                d = torch.matmul((sample - nei), self.L).norm()
                k = torch.exp(-d/self.kernel_size) / np.sqrt(2 * torch.pi)
                _tmp.append(k)
            _tmp = torch.stack(_tmp)
            _tmp = _tmp / (torch.sum(_tmp) + 1e-6)
            distance.append(_tmp)
        distance = torch.stack(distance)
        return distance

class KernelRegression(nn.Module):
    def __init__(self, input_len):
        super(KernelRegression, self).__init__()

        self.Embedding = MLP([input_len, 1024, 1024, 1024])
        self.bn = nn.BatchNorm1d(1024, affine=False)
        self.kernel = Kernel(1024)
    
    def forward(self, support_x, support_y, query_x):
        embedded_support_x = self.bn(self.Embedding(support_x))
        embedded_query_x = self.bn(self.Embedding(query_x))

        kernel = self.kernel(embedded_support_x, embedded_query_x)
        preds = torch.matmul(kernel, support_y)
        return preds

    
