import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

'''
class GraphEmbedder(nn.Module):
    def __init__(self, node_dims: dict, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_mlps = nn.ModuleDict()

        for node_type, input_dim in node_dims.items():
            self.node_mlps[node_type] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, batch):
        device = next(self.parameters()).device

        # Infer batch size from one of the node types
        any_node_type = next(iter(batch.node_types))
        batch_size = int(batch[any_node_type].batch.max()) + 1

        embed = torch.zeros(batch_size, self.hidden_dim, device=device)

        for node_type, mlp in self.node_mlps.items():
            if not hasattr(batch[node_type], 'x'):
                continue

            x = mlp(batch[node_type].x)
            pooled = global_mean_pool(x, batch[node_type].batch)  # [num_graphs_present, hidden]
            present_batch_ids = batch[node_type].batch.unique()
            embed.index_copy_(0, present_batch_ids, pooled)

        return embed  # [num_graphs_in_batch, hidden_dim]
'''

class GraphEmbedder(nn.Module):
    def __init__(self, node_dims: dict, hidden_dims=[128]):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1] if hidden_dims else None
        self.node_mlps = nn.ModuleDict()

        for node_type, input_dim in node_dims.items():
            layers = []
            dims = [input_dim] + hidden_dims

            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())

            # Remove last activation
            if layers:
                layers = layers[:-1]

            self.node_mlps[node_type] = nn.Sequential(*layers)

    def forward(self, batch):
        device = next(self.parameters()).device

        any_node_type = next(iter(batch.node_types))
        batch_size = int(batch[any_node_type].batch.max()) + 1

        embed = torch.zeros(batch_size, self.output_dim, device=device)

        for node_type, mlp in self.node_mlps.items():
            if not hasattr(batch[node_type], 'x'):
                continue

            x = mlp(batch[node_type].x)
            pooled = global_mean_pool(x, batch[node_type].batch)
            present_batch_ids = batch[node_type].batch.unique()
            embed.index_copy_(0, present_batch_ids, pooled)

        return embed  # [num_graphs_in_batch, output_dim]
    
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, proj_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.proj(x)

def kl_divergence_between_gaussians(z1, z2, eps=1e-6):
    mu1, mu2 = z1.mean(0), z2.mean(0)
    var1 = z1.var(0) + eps
    var2 = z2.var(0) + eps

    kl = 0.5 * torch.sum(
        torch.log(var2 / var1)
        + (var1 + (mu1 - mu2).pow(2)) / var2
        - 1
    )
    return kl
