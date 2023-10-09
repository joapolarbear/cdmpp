import os
import numpy as np

from metalearner.model import KernelRegression
from metalearner.learner.base_learner import BaseLearner

class KernelRegressionLearner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super(KernelRegressionLearner, self).__init__(*args, **kwargs)
        
        self.match_net = KernelRegression(input_len=self.input_len)
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
