import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dimension_list, mlp_layers=['linear', 'bn', 'relu', 'dropout', 'linear']):
        super().__init__()

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
            return nn.Dropout(0.5)
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
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.self_attention_aggregator = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)

    def forward(self, x):
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x_out, attn_weights = self.self_attention_aggregator(x, x, x, need_weights=True, average_attn_weights=False)
        return {'output': x_out[:, 0, :].unsqueeze(0),
                'x_out': x_out,
                'attn_weights': attn_weights,
                }


if __name__ == '__main__':

    # Example usage of MLP model
    # Linear(1024 -> 512) -> (BatchNorm) -> (ReLU) -> (Dropout) -> Linear(512 -> 10)
    mlp_model = MLP(dimension_list=[1024, 512, 10], mlp_layers=['linear', 'bn', 'relu', 'dropout', 'linear'])
    x = torch.randn(32, 1024) # 32 samples, 1024 features
    out = mlp_model(x)  # (32, 10)

    # Example usage of AttentionAggregator model
    aggregate_model = AttentionAggregator(embed_dim=1024, num_heads=1)
    x = torch.randn(32, 10, 1024)  # 32 samples, 10 features, 1024 dimensions
    out = aggregate_model(x)
    # out['output'] -> (32, 1024) - 32 samples, 1024 dimensions (the 10 features are aggregated adaptively)
    # out['x_out'] -> (32, 11, 1024) - 32 samples, 11 features, 1024 dimensions (added CLS token)
    # out['attn_weights'] -> (32, 32, 11) - 32 samples, 32x11 attention matrix