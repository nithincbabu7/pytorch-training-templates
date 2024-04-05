import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dimension_list, mlp_layers=['linear', 'bn', 'relu', 'dropout', 'linear'], drp_p=0.1):
        super().__init__()
        self.drp_p = drp_p
        layer_list = []
        dimension_pair_list = list(zip(dimension_list, dimension_list[1:]+[dimension_list[-1]]))
        j=0
        for i, layer in enumerate(mlp_layers):
            if layer == 'linear':
                layer_list.append(self.load_layer(layer, dimension_pair_list[j]))
                j+=1
            elif layer in ['bn', 'ln']:
                layer_list.append(self.load_layer(layer, [dimension_pair_list[j][0]]))
            else:
                layer_list.append(self.load_layer(layer))
        self.out_network = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.out_network(x)

    def load_layer(self, layer_name, dimensions=None, args=None):
        if layer_name == 'linear':
            return nn.Linear(in_features=dimensions[0], out_features=dimensions[1])
        elif layer_name == 'dropout':
            return nn.Dropout(self.drp_p)
        elif layer_name == 'bn':
            return nn.BatchNorm1d(num_features=dimensions[0])
        elif layer_name == 'ln':
            return nn.LayerNorm(dimensions[0])
        elif layer_name == 'gelu':
            return nn.GELU()
        elif layer_name == 'relu':
            return nn.ReLU()
        elif layer_name == 'tanh':
            return nn.Tanh()
        elif layer_name == 'elu':
            return nn.ELU()
        elif layer_name == 'gelu':
            return nn.GELU()
        elif layer_name == 'swish':
            return nn.SiLU()
        elif layer_name == 'sigmoid':
            return nn.Sigmoid()


# Self attention based aggregator to mix arbitrary number of features into one. Adds CLS token to aggregate features.
class AttentionAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=8, add_mlp=True, add_ln=True, residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.add_mlp = add_mlp
        self.add_ln = add_ln
        self.residual = residual
        self.self_attention_aggregators = nn.ModuleList([nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True) for _ in range(num_layers)])
        if add_mlp:
            self.mlps = nn.ModuleList([MLP([embed_dim, embed_dim], mlp_layers=['linear', 'dropout'], drp_p=0.1) for _ in range(num_layers)])
        if add_ln:
            self.lns = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)

    def forward(self, x):   # B x L x F
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)     # B x (L+1) x F
        
        attn_weights_list = []
        hidden_layers = []
        for i in range(self.num_layers):
            x_out, attn_weights = self.self_attention_aggregators[i](x, x, x, need_weights=True, average_attn_weights=False)
            attn_weights_list.append(attn_weights)
            if self.add_mlp:
                x_out = self.mlps[i](x_out)
            x = x + x_out if self.residual else x_out
            if self.add_ln:
                x = self.lns[i](x)      # LayerNorm after adding residual - similar to Bert
            hidden_layers.append(x)
        return {'output': x[:, :1, :],                  # B x 1 x F (CLS token output)
                'hidden_layers': hidden_layers,         # List of B x (L+1) x F
                'attn_weights': attn_weights_list,      # List of B x H x (L+1) x (L+1)
                }


if __name__ == '__main__':

    # Example usage of MLP model
    # Linear(1024 -> 512) -> (BatchNorm) -> (ReLU) -> (Dropout) -> Linear(512 -> 10)
    mlp_model = MLP(dimension_list=[1024, 512, 10], mlp_layers=['linear', 'bn', 'relu', 'dropout', 'linear']).cuda()
    x = torch.randn(32, 1024).cuda() # 32 samples, 1024 features
    out = mlp_model(x)  # (32, 10)

    # Example usage of AttentionAggregator model
    aggregate_model = AttentionAggregator(embed_dim=1024).cuda()
    x = torch.randn(32, 10, 1024).cuda()  # 32 samples, 10 features, 1024 dimensions
    out = aggregate_model(x)
    # out['output'] -> (32, 1, 1024) - 32 samples, 1024 dimensions (the 10 features are aggregated adaptively. CLS token output is taken as the final result)
    # out['hidden_layers'] -> List of (32, 11, 1024) - 32 samples, 11 features, 1024 dimensions (hidden layers of the model)
    #                      -> out['hidden_layers'][-1] - the output layer including all the other features.
    # out['attn_weights'] -> List of (32, 8, 11, 11) - 32 samples, 8 heads, 11 features, 11 features (attention weights of each layer)
