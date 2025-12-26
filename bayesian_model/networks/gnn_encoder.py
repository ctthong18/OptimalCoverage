from typing import Optional, Sequence
import torch
import torch.nn as nn
import numpy as np

try:
    from torch_geometric.nn import GCNConv
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False
    
class DependencyGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = [128, 128], activate=nn.ReLU):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self._use_pyg = _HAS_PYG
        self.layers = nn.ModuleList()
        if self._use_pyg:
            for i in range(len(dims) - 1):
                self.layers.append(GCNConv(dims[i], dims[i + 1]))
        else:
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        
        self.act = activate()
        
    def forward(self, node_features: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = node_features
        if self._use_pyg:
            assert edge_index is not None, "edge_index required for torch_geometric GCNConv"
            for idx, layer in enumerate(self.layers):
                x = layer(x, edge_index)
                if idx < len(self.layers) - 1:
                    x = self.act(x)
            
            return x
        else:
            N = x.shape[0]
            if edge_index is None:
                A = torch.eye(N, device=x.device)
            else:
                if isinstance(edge_index, torch.Tensor):
                    ei = edge_index.cpu().numpy()
                else:
                    ei = np.asarray(edge_index)
                    
                A = np.zeros((N, N), dtype=float)
                for u, v in zip(ei[0], ei[1]):
                    A[int(u), int(v)] = 1.0
                
                A = A + A.T
                for i in range(N):
                    A[i, i] = 1.0
                
                deg = A.sum(axis=1, keepdims=True)
                deg[deg == 0] = 1.0
                A = A / deg
                A = torch.tensor(A, dtype=x.dtype, device=x.device)
            for idx, lin in enumerate(self.layers):
                x = lin(x)
                x = A.matmul(x)
                if idx < len(self.layers) - 1:
                    x = self.act(x)
            
            return x