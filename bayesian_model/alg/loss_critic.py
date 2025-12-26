from typing import List, Optional
import torch
import torch.nn as nn

mse_loss = nn.MSELoss()

def compute_critic_targets(critics_target, batch_node_feats_next, batch_edge_idx_next, local_indices, rewards, gamma, device):
    qs = []
    with torch.no_grad():
        for  nf, ei, li in zip(batch_node_feats_next, batch_edge_idx_next, local_indices):
            nf = nf.to(device)
            if ei is not None:
                ei = ei.to(device)
            q = critics_target(nf, ei, li)
            if q.dim() == 0:
                q = q.squeeze(0)
            
            qs.append(q)
            
        qs_tensor = torch.stack(qs, dim=0).float().to(device)
        y = rewards.to(device) + gamma * qs_tensor * (1.0)
    
    return y

def critic_mse_loss(q_values, targets):
    return mse_loss(q_values, targets)