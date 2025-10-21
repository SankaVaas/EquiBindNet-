"""
EquiBindNet: SE(3)-Equivariant GNN for Binding Affinity Prediction
"""
import torch
import torch.nn as nn
from e3nn import o3
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class EquiBindNet(nn.Module):
    """
    SE(3)-Equivariant Graph Neural Network for protein-drug binding prediction
    
    Architecture:
    1. Input embedding
    2. Multiple equivariant convolution layers
    3. Global pooling (graph → scalar)
    4. MLP prediction head
    """
    
    def __init__(
        self,
        num_atom_types=10,
        hidden_dim=64,
        num_layers=3,
        output_dim=1,
        max_radius=5.0,
        use_simple=True  # Use SimpleEquivariantLayer for CPU
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Input embedding: atom types → features
        self.atom_embedding = nn.Linear(num_atom_types, hidden_dim)
        
        # Equivariant layers
        if use_simple:
            print("Using SimpleEquivariantLayer (CPU-optimized)")
            from models.geometric_layers import SimpleEquivariantLayer
            
            self.conv_layers = nn.ModuleList([
                SimpleEquivariantLayer(
                    hidden_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    hidden_features=hidden_dim
                )
                for i in range(num_layers)
            ])
        else:
            print("Using SE3ConvLayer (full equivariance)")
            from geometric_layers import SE3ConvLayer
            
            self.conv_layers = nn.ModuleList([
                SE3ConvLayer(
                    hidden_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    hidden_features=hidden_dim // 2,
                    max_radius=max_radius
                )
                for i in range(num_layers)
            ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Prediction head (invariant to rotations)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, data):
        """
        data: PyG Data object with x, edge_index, pos, batch
        """
        x = data.x  # Node features [N, num_atom_types]
        edge_index = data.edge_index  # [2, E]
        pos = data.pos  # 3D coordinates [N, 3]
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        
        # Embed atom types
        x = self.atom_embedding(x)  # [N, hidden_dim]
        
        # Equivariant convolutions
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            # Message passing
            x_new = conv(x, edge_index, pos)
            
            # Layer norm
            x_new = norm(x_new)
            
            # Residual connection (if dimensions match)
            if x_new.shape == x.shape:
                x = x + x_new
            else:
                x = x_new
            
            # Non-linearity
            x = F.relu(x)
        
        # Global pooling (graph → single vector)
        graph_repr = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Predict binding affinity
        pred = self.pred_head(graph_repr)  # [batch_size, 1]
        
        return pred


class BaselineCNN(nn.Module):
    """
    Baseline model WITHOUT geometric equivariance
    For comparison - enhanced with better regularization
    """
    
    def __init__(self, num_atom_types=10, hidden_dim=64, num_layers=3):
        super().__init__()
        
        from torch_geometric.nn import GCNConv, BatchNorm
        
        # Input embedding with normalization
        self.atom_embedding = nn.Sequential(
            nn.Linear(num_atom_types, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # GCN layers with batch norm and residual connections
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # More sophisticated prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, data):
        x = self.atom_embedding(data.x)
        
        # Convolution layers with residual connections
        for conv, norm in zip(self.conv_layers, self.batch_norms):
            x_new = norm(F.relu(conv(x, data.edge_index)))
            # Residual connection
            if x_new.shape == x.shape:
                x = x + x_new
            else:
                x = x_new
        
        # Global pooling
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        graph_repr = global_mean_pool(x, batch)
        
        return self.pred_head(graph_repr)


# Test instantiation
if __name__ == "__main__":
    print("="*60)
    print("Testing EquiBindNet Model")
    print("="*60)
    
    model = EquiBindNet(
        num_atom_types=10,
        hidden_dim=64,
        num_layers=3,
        use_simple=True
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model created!")
    print(f"   Parameters: {num_params:,}")
    print(f"   Layers: {model.num_layers}")
    print(f"   Hidden dim: {model.hidden_dim}")
    
    # Test forward pass
    from torch_geometric.data import Data
    
    # Dummy data
    dummy_data = Data(
        x=torch.randn(100, 10),  # 100 atoms, 10 features
        edge_index=torch.randint(0, 100, (2, 500)),  # 500 edges
        pos=torch.randn(100, 3),  # 3D coordinates
        batch=torch.zeros(100, dtype=torch.long)
    )
    
    with torch.no_grad():
        pred = model(dummy_data)
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Input: {dummy_data.num_nodes} nodes")
    print(f"   Output: {pred.shape}")
    print(f"   Prediction: {pred.item():.3f}")