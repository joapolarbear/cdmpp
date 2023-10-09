import torch.nn as nn

class TrainInfinityError(ValueError):
    pass

class MLP(nn.Module):
    def __init__(self, shapes, activation="relu", bn=True, dropout_rate=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for idx in range(len(shapes)-1):
            self.layers.append(nn.Linear(shapes[idx], shapes[idx+1]))
        
        self.activation = activation
        self.bn = bn
        self.bn_layers = nn.ModuleList()
        if self.bn:
            for idx in range(len(shapes)-1):
                self.bn_layers.append(nn.BatchNorm1d(shapes[idx+1]))
        # self.activation_func = F.relu if activation else (lambda x: x)

        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = None
        self.dropout = None if dropout_rate is None else nn.Dropout(dropout_rate)

    def forward(self, input):
        x = input
        for idx, layer in enumerate(self.layers):
            x = layer(x)

            if self.bn:
                x = self.bn_layers[idx](x)
                
            if self.activation:
                x = self.activation(x)

            if self.dropout:
                x = self.dropout(x)
                
        return x
    