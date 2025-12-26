from turtle import forward
from typing import Optional, Sequence, Union, List
import torch
import torch.nn as nn
from .gnn_encoder import DependencyGNN

class GNNCritic(nn.Module):
    def __init__(self, node_features_dim: int, gnn_hidden: Sequence[int] = (128, 128), mlp_hidden: Sequence[int] = (128, )):
        super().__init__()
        self.gnn = DependencyGNN(node_features_dim, hidden_dims=gnn_hidden)
        g_out = gnn_hidden[-1] if len(gnn_hidden) > 0 else node_features_dim
        
        mlp_layers = []
        
        last = g_out
        for h in mlp_hidden:
            mlp_layers.append(nn.Linear(last, h))
            mlp_layers.append(nn.ReLU())
            last = h
        mlp_layers.append(nn.Linear(last, 1))
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, node_features: torch.Tensor, edge_index: Optional[torch.Tensor], local_index: int) -> torch.Tensor:
        H = self.gnn(node_features, edge_index)
        h_i = H[local_index]
        q = self.mlp(h_i)
        return q
    
    def forward_batch(self, batch_node_feats: List[torch.Tensor], batch_edge_idx: List[Optional[torch.Tensor]], local_indices: List[int]) -> torch.Tensor:
        qs = []
        for nf, ei, li in zip(batch_node_feats, batch_edge_idx, local_indices):
            qs.append(self.forward(nf, ei, li))
        return torch.stack(qs, dim=0).squeeze(-1)
    