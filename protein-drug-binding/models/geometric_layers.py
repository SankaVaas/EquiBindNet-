"""
SE(3)-Equivariant Graph Neural Network Layers
Using e3nn library for geometric deep learning
"""
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

class SE3ConvLayer(MessagePassing):
    """
    SE(3)-Equivariant Convolution Layer
    
    This layer respects 3D rotations and translations:
    - If you rotate the input, the output rotates the same way
    - Features are represented using spherical harmonics
    """
    
    def __init__(self, in_features, out_features, hidden_features=32, max_radius=5.0):
        super().__init__(aggr='add', node_dim=0)
        
        self.max_radius = max_radius
        
        # Irreps (irreducible representations) define the geometric structure
        # l=0: scalars (invariant), l=1: vectors (equivariant), l=2: tensors
        self.irreps_in = o3.Irreps(f"{in_features}x0e")  # Start with scalars
        self.irreps_hidden = o3.Irreps(f"{hidden_features}x0e + {hidden_features}x1o")  # Scalars + vectors
        self.irreps_out = o3.Irreps(f"{out_features}x0e")
        
        # Spherical harmonics for geometric encoding
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)  # Up to l=2
        
        # Tensor product: combines node features with geometric information
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_hidden,
            shared_weights=False
        )
        
        # Non-linearity that preserves equivariance
        self.gate = Gate(
            f"{hidden_features}x0e",  # Scalars (pass through gate)
            [torch.relu],              # Activation for scalars
            f"{hidden_features}x1o",  # Vectors (gated)
            [torch.tanh],              # Activation for gates
            f"{hidden_features}x1o"   # Output vectors
        )
        
        # Output projection
        self.linear = o3.Linear(self.irreps_hidden, self.irreps_out)
        
    def forward(self, x, edge_index, pos, edge_attr):
        """
        x: node features [N, F]
        edge_index: connectivity [2, E]
        pos: 3D coordinates [N, 3]
        edge_attr: edge features [E, D]
        """
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)
    
    def message(self, x_j, pos_i, pos_j, edge_attr):
        """
        Compute messages from neighbors
        x_j: neighbor features
        pos_i, pos_j: 3D positions
        edge_attr: edge attributes
        """
        # Relative position vector
        rel_pos = pos_j - pos_i  # [E, 3]
        distance = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        
        # Normalize and compute spherical harmonics
        rel_pos_normalized = rel_pos / (distance + 1e-8)
        sh = o3.spherical_harmonics(
            self.irreps_sh, 
            rel_pos_normalized, 
            normalize=True
        )  # [E, num_sh]
        
        # Distance-based edge weight (smooth cutoff)
        edge_weight = self._smooth_cutoff(distance)
        
        # Tensor product: combine features with geometry
        msg = self.tp(x_j, sh)  # [E, hidden]
        
        # Apply gate non-linearity
        msg = self.gate(msg)
        
        # Weight by distance
        msg = msg * edge_weight
        
        return msg
    
    def update(self, aggr_out, x):
        """Update node features after aggregation"""
        # Residual connection
        out = self.linear(aggr_out)
        return out + x if out.shape == x.shape else out
    
    def _smooth_cutoff(self, distance):
        """Smooth cutoff function for distance weighting"""
        x = distance / self.max_radius
        cutoff = torch.where(
            x < 1,
            torch.exp(-x / (1 - x**2 + 1e-8)),
            torch.zeros_like(x)
        )
        return cutoff


class SimpleEquivariantLayer(nn.Module):
    """
    Simplified SE(3)-equivariant layer for faster training
    Good for CPU training
    """
    
    def __init__(self, in_features, out_features, hidden_features=64):
        super().__init__()
        
        # Scalar features only (l=0) for speed
        self.irreps_in = o3.Irreps(f"{in_features}x0e")
        self.irreps_hidden = o3.Irreps(f"{hidden_features}x0e")
        self.irreps_out = o3.Irreps(f"{out_features}x0e")
        
        # Spherical harmonics (l=0,1 only)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=1)
        
        # Tensor product
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_hidden,
            shared_weights=True  # Share weights for speed
        )
        
        # Simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )
        
    def forward(self, x, edge_index, pos):
        """Forward pass"""
        row, col = edge_index
        
        # Compute spherical harmonics for edges
        rel_pos = pos[col] - pos[row]
        distance = torch.norm(rel_pos, dim=-1, keepdim=True)
        rel_pos_normalized = rel_pos / (distance + 1e-8)
        
        sh = o3.spherical_harmonics(
            self.irreps_sh,
            rel_pos_normalized,
            normalize=True
        )
        
        # Tensor product
        edge_features = self.tp(x[col], sh)
        
        # Aggregate
        aggr = scatter(edge_features, row, dim=0, dim_size=x.size(0), reduce='mean')
        
        # MLP
        out = self.mlp(aggr)
        
        return out


print("✅ Geometric layers defined!")
print("   - SE3ConvLayer: Full SE(3)-equivariance with l=0,1,2")
print("   - SimpleEquivariantLayer: Fast version with l=0,1")