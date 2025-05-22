import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_classes: int,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        """
        Graph Convolutional Network.

        Parameters:
        -- in_channels: input feature size (dataset.num_node_features)
        -- hidden_channels: number of neurons in hidden layers
        -- num_classes: number of output tasks (dataset.num_classes)
        -- num_layers: number of convolutional layers (>=2)
        -- dropout: dropout probability between layers
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):
        """
        Forward pass through the GCN model.
        
        Parameters:
        -- data: PyG Data object containing graph information
        
        Returns:
        -- output: model predictions for each task
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply graph convolutions and nonlinearities
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global mean pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Apply final linear layer
        return self.lin(x)