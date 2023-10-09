'''
Refer to 'Matching networks for one shot learning'

'''
import torch
import torch.nn as nn
from torch.autograd import Variable

from .base_net import MLP


class CosDistance(nn.Module):
    def __init__(self):
        super(CosDistance, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, support_set, query_set): 
        similarities = []
        for query_image in query_set:
            cosine_similarity = self.cos(support_set, query_image)
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities


class AttentionalRegressor(nn.Module):
    def __init__(self):
        super(AttentionalRegressor, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, similarities, support_set_y):
        """
        Produces pdfs over the support set classes for the target set image.

        Parameters
        ---------
        similarities:
            A tensor with cosine similarities of size 
            [bs_of_query_set, bs_of_support_set]
        support_set_y:
            A tensor with the one hot vectors of the targets for each support set image
        
        Return
        ------
            Softmax pdf, [sequence_length,  batch_size, num_classes]
        """
        softmax_similarities = self.softmax(similarities)
        preds = torch.matmul(softmax_similarities, support_set_y)
        return preds

class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer 
                            e.g. [100, 100, 100] returns a 3 layer, 100
        :param batch_size: The experiments batch size
        """
        self.batch_size = batch_size
        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)

        '''
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        bidirectional: If True, becomes a bidirectional RNN. Default: False
        '''
        self.lstm = nn.LSTM(input_size=self.vector_dim,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)

    def forward(self, inputs):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param x: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        c0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),
                      requires_grad=False).cuda()
        h0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),
                      requires_grad=False).cuda()
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        return output, hn, cn

class MatchNet(nn.Module):
    ''' Refer to https://github.com/gitabcworld/MatchingNetworks.git
    '''
    def __init__(self, input_len):
        super(MatchNet, self).__init__()
        self.Embedding = MLP([input_len, 1024, 1024, 1024])
        self.Distance = CosDistance()
        self.Regression = AttentionalRegressor()
        self.bn = nn.BatchNorm1d(1024, affine=False)

    def forward(self, support_x, support_y, query_x):
        ### Embedding
        embedded_support_x = self.bn(self.Embedding(support_x))
        embedded_query_x = self.bn(self.Embedding(query_x))
        attention = self.Distance(embedded_support_x, embedded_query_x)
        preds = self.Regression(attention, support_y)
        return preds
