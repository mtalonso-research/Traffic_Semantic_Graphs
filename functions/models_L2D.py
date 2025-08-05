import json
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from torch_geometric.utils import to_dense_batch
import torch.optim as optim

sys.path.append(os.path.abspath(".."))
from functions.graphs_L2D import create_dataloaders, combined_graph_viewer

class NodeTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4):
        super(NodeTransformer, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=2*in_dim,
            batch_first=True   # (batch, seq_len, features)
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # Optional projection to desired output dim
        self.proj = nn.Linear(in_dim, out_dim) if out_dim != in_dim else nn.Identity()

    def forward(self, x, batch):
        """
        x: [num_nodes, in_dim]
        batch: [num_nodes] -> graph ids for each node
        """
        # Convert sparse to dense (pad graphs to max node count in batch)
        x_padded, mask = to_dense_batch(x, batch)  # [batch_size, max_nodes, in_dim]

        # Apply transformer
        out = self.transformer(x_padded, src_key_padding_mask=~mask)

        # Project to output dim
        out = self.proj(out)  # [batch_size, max_nodes, out_dim]

        # Flatten back to node list
        out = out[mask]  # remove padding

        return out
    
class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level pooling
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        x = self.lin(x)
        return x
    
class GraphPipeline(nn.Module):
    def __init__(self, in_dim, trans_dim, gnn_hidden, graph_emb_dim):
        super(GraphPipeline, self).__init__()
        self.transformer = NodeTransformer(in_dim=in_dim, out_dim=trans_dim, num_heads=2)
        self.gnn = SimpleGNN(in_dim=trans_dim, hidden_dim=gnn_hidden, out_dim=2)#graph_emb_dim)

    def forward(self, x, edge_index, batch):
        # Step 1: Update node features via transformer
        x_new = self.transformer(x, batch)

        # Step 2: Get graph embedding via GNN
        graph_emb = self.gnn(x_new, edge_index, batch)
        return graph_emb
    
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)

        # Placeholder loss (to be replaced later)
        # Assume batch.y is compatible with out
        loss = loss_fn(out, batch.y)

        # Backward + Optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        total_loss += loss.item()

    return total_loss / len(loader)