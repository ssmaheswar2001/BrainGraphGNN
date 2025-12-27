"""
Graph Attention Network (GAT) classifier
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm


class GAT_Classifier(nn.Module):
    """
    Graph Attention Network with improvements:
      - Multi-head attention
      - Edge attention
      - Multi-scale pooling
    """
    
    def __init__(self, num_node_features, num_classes=2, hidden_channels=128,
                 num_layers=4, heads=4, dropout=0.3):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer with multi-head attention
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_channels * heads)
        
        # Hidden layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.bns.append(BatchNorm(hidden_channels * heads))
        
        # Output layer (single head)
        self.conv_out = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.bn_out = BatchNorm(hidden_channels)
        
        # Multi-scale pooling
        pooled_dim = hidden_channels * 3
        
        # Classification
        self.fc1 = Linear(pooled_dim, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = Linear(hidden_channels // 2, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_out(x, edge_index)
        x = self.bn_out(x)
        x = F.elu(x)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # Classification
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        
        return x