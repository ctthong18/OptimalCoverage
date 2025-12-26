import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple

from new_network.base_network import QNetwork
from new_network.rnn_network import RNNQNetwork, AttentionRNNQNetwork, BiRNNQNetwork, HierarchicalRNNQNetwork
from .mixer import (
    QPLEXMixingNetwork, AttentionMixingNetwork,
    AdaptiveMixingNetwork, HierarchicalMixingNetwork
)


class EnhancedQPLEXNetwork(nn.Module):
    """Enhanced QPLEX network kết hợp mạng mới và ý tưởng từ deeprec.
    
    Debug Mode:
        To enable debug mode for shape tracking, pass debug=True in the config:
        config = {'debug': True, ...}
        
        When debug mode is enabled, the network will print intermediate tensor shapes
        during forward pass to help diagnose shape mismatches and transformation issues.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int,
                 config: Dict[str, Any], debug: bool = False):
        super(EnhancedQPLEXNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.config = config
        # Enable debug mode for shape tracking (can also be set via config)
        self.debug = debug or config.get('debug', False)
        
        network_config = config.get('network', {})
        q_net_config = network_config.get('q_network', {})
        mix_net_config = network_config.get('mixing_network', {})
        
        # Individual Q-networks với các kiến trúc mới
        self.q_networks = nn.ModuleList()
        q_net_type = q_net_config.get('type', 'attention_rnn')
        for _ in range(n_agents):
            if q_net_type == 'attention_rnn':
                q_net = AttentionRNNQNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=q_net_config.get('rnn_hidden_dim', 128),
                    num_layers=q_net_config.get('rnn_layers', 2),
                    rnn_type=q_net_config.get('rnn_type', 'lstm'),
                    num_heads=q_net_config.get('num_attention_heads', 4),
                    dropout=q_net_config.get('dropout', 0.1)
                )
            elif q_net_type == 'hierarchical_rnn':
                q_net = HierarchicalRNNQNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=q_net_config.get('rnn_hidden_dim', 128),
                    num_layers=q_net_config.get('rnn_layers', 2),
                    rnn_type=q_net_config.get('rnn_type', 'lstm'),
                    dropout=q_net_config.get('dropout', 0.1)
                )
            elif q_net_type == 'bi_rnn':
                q_net = BiRNNQNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=q_net_config.get('rnn_hidden_dim', 128),
                    num_layers=q_net_config.get('rnn_layers', 2),
                    rnn_type=q_net_config.get('rnn_type', 'lstm'),
                    dropout=q_net_config.get('dropout', 0.1)
                )
            elif q_net_type == 'rnn':
                q_net = RNNQNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=q_net_config.get('rnn_hidden_dim', 128),
                    num_layers=q_net_config.get('rnn_layers', 2),
                    rnn_type=q_net_config.get('rnn_type', 'lstm'),
                    dropout=q_net_config.get('dropout', 0.1)
                )
            else:
                q_net = QNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dims=q_net_config.get('hidden_dims', [256, 128]),
                    use_rnn=q_net_config.get('use_rnn', False),
                    rnn_hidden_dim=q_net_config.get('rnn_hidden_dim', 128),
                    rnn_layers=q_net_config.get('rnn_layers', 1),
                    use_attention=q_net_config.get('use_attention', False),
                    num_attention_heads=q_net_config.get('num_attention_heads', 4)
                )
            self.q_networks.append(q_net)
        
        # Mixing network
        mixer_type = mix_net_config.get('type', 'adaptive')
        hidden_dims = mix_net_config.get('hidden_dims', [256, 256])
        if mixer_type == 'adaptive':
            self.mixing_network = AdaptiveMixingNetwork(
                state_dim=state_dim,
                n_agents=n_agents,
                hidden_dims=hidden_dims,
                complexity_threshold=mix_net_config.get('complexity_threshold', 0.6),
                debug=self.debug
            )
        elif mixer_type == 'hierarchical':
            self.mixing_network = HierarchicalMixingNetwork(
                state_dim=state_dim,
                n_agents=n_agents,
                hidden_dims=hidden_dims,
                num_levels=mix_net_config.get('num_levels', 2),
                debug=self.debug
            )
        elif mixer_type == 'attention':
            self.mixing_network = AttentionMixingNetwork(
                state_dim=state_dim,
                n_agents=n_agents,
                hidden_dim=hidden_dims[0],
                num_heads=mix_net_config.get('num_attention_heads', 4),
                dropout=mix_net_config.get('dropout', 0.0),
                debug=self.debug
            )
        else:
            self.mixing_network = QPLEXMixingNetwork(
                state_dim=state_dim,
                n_agents=n_agents,
                hidden_dims=hidden_dims,
                use_hypernet=mix_net_config.get('use_hypernet', True),
                dueling=mix_net_config.get('dueling', True),
                debug=self.debug
            )
        
        # State encoder
        if network_config.get('use_state_encoder', True):
            encoder_hidden = network_config.get('state_encoder_hidden', 256)
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, encoder_hidden),
                nn.ReLU(),
                nn.Linear(encoder_hidden, encoder_hidden // 2),
                nn.ReLU(),
                nn.Linear(encoder_hidden // 2, state_dim)
            )
        else:
            self.state_encoder = None
    
    def forward(self, obs: torch.Tensor, state: torch.Tensor,
                hidden: Optional[List[Tuple]] = None):
        """
        Forward pass through the network.
        
        Args:
            obs: Observations tensor (batch, n_agents, obs_dim)
            state: Global state tensor (batch, state_dim)
            hidden: Optional list of hidden states for each agent
            
        Returns:
            q_values: Individual Q-values (batch, n_agents, action_dim)
            q_total: Mixed total Q-value (batch, 1)
            new_hidden: Updated hidden states
        """
        if self.debug:
            print(f"[EnhancedQPLEXNetwork.forward] Input shapes: obs={tuple(obs.shape)}, state={tuple(state.shape)}")
        
        if obs.dim() < 2:
            raise RuntimeError(f"obs must have at least 2 dims (batch, n_agents, ...), got {tuple(obs.shape)}")
        
        batch_size = obs.shape[0]
        if hidden is None:
            hidden = [None] * self.n_agents

        encoded_state = self.state_encoder(state) if self.state_encoder is not None else state
        
        if self.debug and self.state_encoder is not None:
            print(f"[EnhancedQPLEXNetwork.forward] Encoded state shape: {tuple(encoded_state.shape)}")

        # Collect Q-values from individual agent networks
        q_values_list = []
        new_hidden = []
        for i in range(self.n_agents):
            q_val, h = self.q_networks[i](obs[:, i], hidden[i])
            if not torch.is_tensor(q_val):
                raise RuntimeError(f"q_network[{i}] returned non-tensor: {type(q_val)}")
            if self.debug:
                print(f"[EnhancedQPLEXNetwork.forward] Agent {i} Q-values shape: {tuple(q_val.shape)}")
            q_values_list.append(q_val)
            new_hidden.append(h)

        # Stack along agent dimension
        try:
            q_values = torch.stack(q_values_list, dim=1)
            if self.debug:
                print(f"[EnhancedQPLEXNetwork.forward] After stack: q_values shape = {tuple(q_values.shape)}")
        except Exception as e:
            shapes = [tuple(t.shape) for t in q_values_list]
            raise RuntimeError(f"Failed to stack q_values; per-agent shapes = {shapes}") from e

        # Normalize shape to (batch, n_agents, action_dim)
        if q_values.dim() == 2:
            # Case: (batch, n_agents) - each agent returned scalar Q-value
            if q_values.size(1) == self.n_agents:
                q_individual = q_values
                q_total = self.mixing_network(q_individual, encoded_state)
                return q_values.unsqueeze(-1), q_total, new_hidden
            # Case: flattened (batch, n_agents * action_dim)
            elif q_values.size(1) % self.n_agents == 0:
                action_dim = q_values.size(1) // self.n_agents
                q_values = q_values.view(batch_size, self.n_agents, action_dim)
            else:
                raise RuntimeError(f"Cannot reshape q_values {tuple(q_values.shape)} with n_agents={self.n_agents}")
        
        elif q_values.dim() == 3:
            b, d1, d2 = q_values.shape
            # Case: (batch, action_dim, n_agents) - need to permute
            if d1 == self.action_dim and d2 == self.n_agents:
                q_values = q_values.permute(0, 2, 1)
            # Case: (batch, n_agents, action_dim) - already correct
            elif d1 == self.n_agents:
                pass
            else:
                raise RuntimeError(f"Unexpected q_values shape {tuple(q_values.shape)}, expected (batch, {self.n_agents}, {self.action_dim})")
        else:
            raise RuntimeError(f"Unexpected q_values dimensions: {q_values.dim()}, shape: {tuple(q_values.shape)}")

        # Validate final shape before max operation
        if q_values.size(1) != self.n_agents:
            shapes = [tuple(t.shape) for t in q_values_list]
            raise RuntimeError(f"q_values has wrong agent dimension: {tuple(q_values.shape)}, per-agent shapes: {shapes}")

        if self.debug:
            print(f"[EnhancedQPLEXNetwork.forward] Normalized q_values shape: {tuple(q_values.shape)}")

        # Extract individual Q-values (max over actions per agent)
        if q_values.size(2) == 1:
            q_individual = q_values.squeeze(-1)
        else:
            q_individual = q_values.max(dim=-1)[0]

        if self.debug:
            print(f"[EnhancedQPLEXNetwork.forward] q_individual shape (after max): {tuple(q_individual.shape)}")

        # Mix individual Q-values into total Q-value
        q_total = self.mixing_network(q_individual, encoded_state)
        
        if self.debug:
            print(f"[EnhancedQPLEXNetwork.forward] q_total shape: {tuple(q_total.shape)}")
        
        return q_values, q_total, new_hidden

    
    def get_q_values(self, obs: torch.Tensor, hidden: Optional[List[Tuple]] = None):
        batch_size, n_agents = obs.shape[:2]
        if hidden is None:
            hidden = [None] * n_agents
        q_values = []
        new_hidden = []
        for i in range(n_agents):
            q_val, h = self.q_networks[i](obs[:, i], hidden[i])
            q_values.append(q_val)
            new_hidden.append(h)
        q_values = torch.stack(q_values, dim=1)
        return q_values, new_hidden

