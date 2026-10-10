import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from tqdm import tqdm


class GCN(nn.Module):
    """
    Standard Full-Batch Graph Convolutional Network (Kipf & Welling, 2017).
    Includes Batch Normalization, Dropout, and ReLU activations.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 3, dropout: float = 0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels, cached=True))
        else:
            # First layer
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Output layer
            self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index_or_adj):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index_or_adj)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index_or_adj)
        return x.log_softmax(dim=-1)


class GraphSAGE(nn.Module):
    """
    GraphSAGE (Hamilton et al., 2017) architecture.
    Supports both mini-batch training with sampled bipartite subgraphs
    and full-graph layer-by-layer evaluation for inference.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 3, dropout: float = 0.5):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        """
        Forward pass on sampled subgraph from NeighborLoader.
        x: features of all sampled nodes in the batch
        edge_index: sampled subgraph edges
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            if len(self.bns) > i:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)

