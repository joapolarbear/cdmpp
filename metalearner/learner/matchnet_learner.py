import os
import numpy as np

import torch

from metalearner.model import MatchNet
from metalearner.learner.base_learner import BaseLearner

class MatchNetLearner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super(MatchNetLearner, self).__init__(*args, **kwargs)
        
        self.match_net = MatchNet(input_len=self.input_len)
        self.init_device()

    def init_device(self):
        if self.use_cuda != -1:
            self.match_net.cuda(self.use_cuda)

    def _inference(self, support_x, support_y, query_x, query_y):
        raise NotImplementedError("Output format should be CMOutput")
        preds = self.match_net(support_x, support_y, query_x)
        return preds
    
    def _loss(self, outputs, support_x, support_y, query_x, query_y, debug=False):
        loss = F.mse_loss(outputs.preds, query_y)
        return loss
    
    def compute_metrics(self, preds, support_x, support_y, query_x, query_y):
        error = np.average(np.abs(((preds - query_y) / query_y).cpu().detach().numpy()))
        return error
    
    def save(self, path=None):
        _path = path if path is not None else self.cache_path
        if _path is None:
            return
        super(MatchNetLearner, self).save(_path)
        torch.save(self.match_net, os.path.join(_path, "match_net.torch"))

    def load(self, path=None):
        _path = path if path is not None else self.cache_path
        super(MatchNetLearner, self).load(_path)

        self.match_net = torch.load(os.path.join(_path, "match_net.torch"))
        self.init_device()
