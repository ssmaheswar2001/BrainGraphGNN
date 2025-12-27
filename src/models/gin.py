"""
Graph Isomorphism Network (GIN) classifier
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm


class GIN_Classifier(nn.Module):
    """
    Graph Isomorphism Network with:
      - Expressive MLPs
      - Sum pooling (standard for GIN)
      - Additional pooling strategies
    """
    
    def __init__(self, num_node_features, num_classes=2, hidden_channels=128,
                 num_layers=4, dropout=0.3):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GIN layers with MLPs
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        nn1 = nn.Sequential(
            Linear(num_node_features, hidden_channels),
            BatchNorm(hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
            BatchNorm(hidden_channels),
            nn.ReLU()
        )
        self.convs.append(GINConv(nn1, train_eps=True))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers-1):
            nn_hidden = nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                nn.ReLU()
            )
            self.convs.append(GINConv(nn_hidden, train_eps=True))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Multi-scale pooling
        pooled_dim = hidden_channels * 3
        
        # Classification head
        self.fc1 = Linear(pooled_dim, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = Linear(hidden_channels // 2, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GIN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Multi-scale pooling (GIN uses sum, but we add others)
        x_sum = global_add_pool(x, batch)  # GIN typically uses sum pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_sum, x_mean, x_max], dim=1)
        
        # Classification
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        
        return x