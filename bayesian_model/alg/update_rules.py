from typing import Iterable
import torch

def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float):
    for p_src, p_tgt in zip(source.parameters(), target.parameters()):
        p_tgt.data.copy_(tau * p_src.data + (1.0 - tau) * p_tgt.data)

def hard_update(source: torch.nn.Module, target: torch.nn.Module):
    for p_src, p_tgt in zip(source.parameters(), target.parameters()):
        p_tgt.data.copy_(p_src.data)