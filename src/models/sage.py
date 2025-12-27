"""
GraphSAGE classifier
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm


class GraphSAGE_Classifier(nn.Module):
    """
    GraphSAGE with:
      - Neighborhood aggregation
      - Multi-scale pooling
      - Deeper architecture
    """
    
    def __init__(self, num_node_features, num_classes=2, hidden_channels=128,
                 num_layers=4, dropout=0.3, aggr='mean'):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # SAGE layers
        self.conv1 = SAGEConv(num_node_features, hidden_channels, aggr=aggr)
        self.bn1 = BatchNorm(hidden_channels)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Output layer
        self.conv_out = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
        self.bn_out = BatchNorm(hidden_channels)
        
        # Multi-scale pooling
        pooled_dim = hidden_channels * 3
        
        # Classification head
        self.fc1 = Linear(pooled_dim, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = Linear(hidden_channels // 2, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_out(x, edge_index)
        x = self.bn_out(x)
        x = F.relu(x)
        
        # Multi-scale pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
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