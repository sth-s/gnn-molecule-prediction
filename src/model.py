import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.nn.norm import GraphNorm

class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_graph_norm: bool = True
    ):
        """
        Graph Convolutional Network with GraphNorm applied before activation.

        Parameters:
        - in_channels: size of input node features
        - hidden_channels: size of hidden embeddings
        - num_classes: number of output tasks
        - num_layers: number of GCN layers (>=1)
        - dropout: dropout probability
        - use_graph_norm: whether to apply GraphNorm after each convolution
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_graph_norm = use_graph_norm

        # Build convolutional + normalization layers
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(
            GraphNorm(hidden_channels) if use_graph_norm else nn.Identity()
        )

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(
                GraphNorm(hidden_channels) if use_graph_norm else nn.Identity()
            )

        self.dropout = dropout
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        """
        Forward pass through the GCN model.

        Parameters:
        -- data: PyG Data object containing graph information
        
        Returns:
        -- output: model predictions for each task
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            # Apply GraphNorm before activation
            x = norm(x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool node embeddings to graph-level
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GIN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_graph_norm: bool = True
    ):
        """
        Graph Isomorphism Network with GraphNorm applied before activation.
        GIN is generally more powerful than GCN for graph representation learning.

        Parameters:
        - in_channels: size of input node features
        - hidden_channels: size of hidden embeddings
        - num_classes: number of output tasks
        - num_layers: number of GIN layers (>=1)
        - dropout: dropout probability
        - use_graph_norm: whether to apply GraphNorm after each convolution
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_graph_norm = use_graph_norm

        # Build convolutional + normalization layers
        # For GINConv, we need to define the MLP inside each GIN layer
        
        # First layer
        self.convs.append(GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
        ))
        self.norms.append(
            GraphNorm(hidden_channels) if use_graph_norm else nn.Identity()
        )

        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
            ))
            self.norms.append(
                GraphNorm(hidden_channels) if use_graph_norm else nn.Identity()
            )

        self.dropout = dropout
        
        # Enhanced 2-layer MLP "tail" for final predictions
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, data):
        """
        Forward pass through the GIN model.

        Parameters:
        -- data: PyG Data object containing graph information
        
        Returns:
        -- output: model predictions for each task
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            # Apply GraphNorm before activation
            x = norm(x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool node embeddings to graph-level
        x = global_mean_pool(x, batch)
        
        # Apply improved MLP tail
        return self.mlp(x)