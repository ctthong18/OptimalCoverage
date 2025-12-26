from typing import Sequence, Optional, Tuple, Type
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: nn.Module = nn.ReLU,
        output_activation: Optional[nn.Module] = nn.Tanh,
        action_scale: Optional[float] = None,
        action_bias: Optional[float] = 0,
    ):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        
        layers.append(nn.Linear(last, act_dim))
        
        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)
        
        self.register_buffer("_action_scale", torch.tensor(action_scale) if action_scale is not None else torch.tensor(1.0))
        self.register_buffer("_action_bias", torch.tensor(action_bias) if action_bias is not None else torch.tensor(0.0))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)
        return out * self._action_scale + self._action_bias
    
    def act(self, obs: torch.Tensor, noise_std: Optional[float] = None) -> torch.Tensor:
        with torch.no_grad():
            a = self.forward(obs)
            if noise_std is not None and noise_std > 0.0:
                a = a + torch.randn_like(a) * noise_std
        
        return a
    