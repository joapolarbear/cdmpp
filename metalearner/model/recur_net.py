''' A recursive LSTM cost model, refer to 
    https://proceedings.mlsys.org/paper/2021/file/3def184ad8f4755ff269862ea77393dd-Paper.pdf
'''
import numpy as np
import torch
import torch.nn as nn

from utils.util import warn_once


class Model_Recursive_LSTM_v2(nn.Module):
    ''' Refer to https://github.com/Tiramisu-Compiler/tiramisu/blob/master/utils/CostModels/Recursive_LSTM_v2_MAPE/utils.py#L294
    '''
    def __init__(self, input_size, disable_norm, comp_embed_layer_sizes=[600, 350, 200, 180], drops=[0.225, 0.225, 0.225, 0.225], output_size=1, residual=False):
        super().__init__()
        comp_embed_layer_sizes = [input_size] + comp_embed_layer_sizes
        embedding_size = comp_embed_layer_sizes[-1]
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [embedding_size*2] + comp_embed_layer_sizes[-2:]
        
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts= nn.ModuleList()
        ''' When batch_first=True, 
            By default, D=1 (not biderectional), N_layer=1
            * the input shape= (B, L, H_in), h_0's shape = (D*N_layer, B, H_out), c_0 = (D * N_layer, B, H_cell)
            * The output shape = (B, L, D*H_out), h_n'shape = (D*N_layer, B, H_out), c_n's shape = (D*N_layer, B, H_cell)
        '''
        self.comps_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.nodes_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts= nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts= nn.ModuleList()

        for i in range(len(comp_embed_layer_sizes)-1):
            self.comp_embedding_layers.append(nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i+1], bias=True))
            nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        
        for i in range(len(regression_layer_sizes)-1):
            self.regression_layers.append(nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i+1], bias=True))
            nn.init.xavier_uniform_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))

        for i in range(len(concat_layer_sizes)-1):
            self.concat_layers.append(nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i+1], bias=True))
            # nn.init.xavier_uniform_(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))

        self.activation = nn.ELU()
        self.no_comps_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))
        self.no_nodes_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))

        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        nn.init.xavier_uniform_(self.predict.weight)
        
        ### Whether to use a residual connection
        self.residual = residual
        if self.residual:
            warn_once("Use residual connection")
        
        self.disable_norm = disable_norm
        
    def get_hidden_state(self, node, ast_feature_embed):
        # print(f"\n<{node.node_id}")
        nodes_list = []
        for n in node.childs:
            nodes_list.append(self.get_hidden_state(n, ast_feature_embed))

        if (nodes_list != []):
            nodes_tensor = torch.cat(nodes_list, 1) 
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1, 0, 2)
        else:       
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(ast_feature_embed.shape[0], -1, -1)

        if node.is_leaf:
            selected_comps_tensor = torch.index_select(ast_feature_embed, 1, node.indice_in_featuers)
            # print(f"selected_comps_tensor: {selected_comps_tensor.shape}")
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(selected_comps_tensor)
            # print(f"comps_h_n: {comps_h_n.shape}")
            comps_h_n = comps_h_n.permute(1, 0, 2)
            # print(f"comps_h_n permute: {comps_h_n.shape}")
        else:
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(ast_feature_embed.shape[0], -1, -1)

        x = torch.cat((nodes_h_n, comps_h_n), 2)

        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.activation(x))
        
        # print(f"{node.node_id}>")
        return x  

    def forward(self, tree_tensors):
        try:
            tree, ast_features = tree_tensors
        except:
            import pdb; pdb.set_trace()

        ### Computation embbedding layer
        # print("input feature shape: ", ast_features.shape)
        # print("# of leaf nodes of this batch: ", len(tree.idx_node_ids))

        x = ast_features
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.activation(x))  
        ast_feature_embed = x
        
        ### Recursive loop embbeding layer
        prog_embedding = self.get_hidden_state(tree.root, ast_feature_embed)

        if self.residual:
            ### `ast_feature_embed`'s shape = (B, N_leaf, N_embed_entry)
            #   `prog_embedding`'s shape = (B, 1, N_embed_entry)
            x = prog_embedding + torch.sum(ast_feature_embed, 1, keepdim=True)
        else:
            x = prog_embedding

        ### regression layer
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.activation(x))
        out = self.predict(x)
        
        if self.disable_norm:
            return self.activation(out[:, 0, 0])
        else:
            return out[:, 0, 0]
    
    def register_hooks_for_grads_weights(self, monitor):
        ### BW Hooks
        monitor.register_bw_hook(self.comp_embedding_layers[-1], "Embedding/Last")
        monitor.register_bw_hook(self.comp_embedding_layers[0], "Embedding/First")
        
        # monitor.register_bw_hook(self.comps_lstm, "LSTM/Comp", compatible=True)
        # monitor.register_bw_hook(self.nodes_lstm, "LSTM/Nodes", compatible=True)
        # monitor.register_bw_hook(self.concat_layers[-1], "Concat/Last")
        # monitor.register_bw_hook(self.concat_layers[0], "Concat/First")
        # monitor.register_bw_hook(self.regression_layers[-1], "Regression/Last")
        # monitor.register_bw_hook(self.regression_layers[0], "Regression/First")

        monitor.register_bw_hook(self.predict, "Predict")
        
        ### FW hooks
        # monitor.register_fw_hook(self.comp_embedding_layers[-1], "Embedding/Last")
        # monitor.register_fw_hook(self.concat_layers[-1], "Concat/Last")
        # monitor.register_fw_hook(self.regression_layers[-1], "Regression/Last")
        # monitor.register_fw_hook(self.predict, "Predict")


    def para_cnt(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        para_cnt = sum([np.prod(p.size()) for p in model_parameters])
        return para_cnt