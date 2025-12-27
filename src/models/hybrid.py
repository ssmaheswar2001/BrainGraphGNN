"""
Hybrid GNN classifier combining multiple architectures
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm


class HGNN_Classifier(nn.Module):
    """
    Hybrid architecture combining different GNN types:
      - Initial GCN layer for basic graph structure
      - GAT layer for attention-based aggregation
      - GIN layer for expressive power
      - GraphSAGE layer for neighborhood aggregation
      - Multi-scale pooling
    """
    
    def __init__(self, num_node_features, num_classes=2, hidden_channels=128,
                 dropout=0.3, heads=4):
        super().__init__()
        
        self.dropout = dropout
        
        # Layer 1: GCN for basic structure
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        
        # Layer 2: GAT for attention
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_channels * heads)
        
        # Layer 3: GIN for expressive power
        ginn_nn = nn.Sequential(
            Linear(hidden_channels * heads, hidden_channels),
            BatchNorm(hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.conv3 = GINConv(ginn_nn, train_eps=True)
        self.bn3 = BatchNorm(hidden_channels)
        
        # Layer 4: GraphSAGE for neighborhood aggregation
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.bn4 = BatchNorm(hidden_channels)
        
        # Multiple pooling strategies
        pooled_dim = hidden_channels * 3  # mean + max + sum
        
        # Classification head
        self.fc1 = Linear(pooled_dim, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = Linear(hidden_channels // 2, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1: GCN
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2: GAT
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3: GIN
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 4: GraphSAGE
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Multiple pooling strategies
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        # Concatenate pooled representations
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