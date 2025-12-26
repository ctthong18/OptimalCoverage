"""QPLEX HRL Model - Hierarchical network architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Import advanced networks from qplex_dev
from new_network.base_network import QPLEXNetwork, MixingNetwork
from new_network.rnn_network import AttentionRNNQNetwork, HierarchicalRNNQNetwork


class HierarchicalTargetSelector(nn.Module):
    """
    Hierarchical target selection network inspired by MATE HRL approach.
    
    This network learns to select which targets to focus on at different time scales:
    - Fast selection: Frame-by-frame target selection
    - Slow selection: Strategic target prioritization
    """
    
    def __init__(self, obs_dim: int, n_targets: int, hidden_dim: int = 128,
                 frame_skip: int = 5, use_attention: bool = True):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_targets = n_targets
        self.hidden_dim = hidden_dim
        self.frame_skip = frame_skip
        self.use_attention = use_attention
        
        # Fast selector (every frame)
        self.fast_selector = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_targets)
        )
        
        # Slow selector (every frame_skip frames)
        self.slow_selector = nn.Sequential(
            nn.Linear(obs_dim + n_targets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_targets)
        )
        
        # Attention mechanism for target importance
        if use_attention:
            self.target_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True
            )
            self.target_proj = nn.Linear(obs_dim, hidden_dim)
        
        # Frame counter for hierarchical timing
        self.frame_counter = 0
        self.last_slow_selection = None
    
    def forward(self, obs: torch.Tensor, target_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for hierarchical target selection.
        
        Args:
            obs: Camera observation [batch_size, obs_dim]
            target_mask: Mask for visible targets [batch_size, n_targets]
        
        Returns:
            target_selection: Binary selection for each target [batch_size, n_targets]
            info: Additional information about selection process
        """
        batch_size = obs.size(0)
        
        # Fast selection (every frame)
        fast_logits = self.fast_selector(obs)
        
        # Slow selection (every frame_skip frames)
        if self.frame_counter % self.frame_skip == 0 or self.last_slow_selection is None:
            if self.last_slow_selection is not None:
                slow_input = torch.cat([obs, self.last_slow_selection], dim=-1)
            else:
                slow_input = torch.cat([obs, torch.zeros(batch_size, self.n_targets, device=obs.device)], dim=-1)
            
            slow_logits = self.slow_selector(slow_input)
            self.last_slow_selection = torch.sigmoid(slow_logits).detach()
        else:
            slow_logits = torch.zeros_like(fast_logits)
        
        # Combine fast and slow selections
        if self.last_slow_selection is not None:
            # Weight fast selection by slow selection (strategic guidance)
            combined_logits = fast_logits + 0.5 * torch.log(self.last_slow_selection + 1e-8)
        else:
            combined_logits = fast_logits
        
        # Apply attention if enabled
        if self.use_attention and hasattr(self, 'target_attention'):
            target_features = self.target_proj(obs).unsqueeze(1)  # [batch, 1, hidden]
            target_features = target_features.repeat(1, self.n_targets, 1)  # [batch, n_targets, hidden]
            
            attended_features, attention_weights = self.target_attention(
                target_features, target_features, target_features
            )
            
            # Use attention to modulate selection
            attention_logits = torch.mean(attended_features, dim=-1)  # [batch, n_targets]
            combined_logits = combined_logits + 0.3 * attention_logits
        
        # Apply target visibility mask
        if target_mask is not None:
            combined_logits = combined_logits.masked_fill(~target_mask.bool(), -float('inf'))
        
        # Convert to binary selection (multi-hot encoding)
        target_probs = torch.sigmoid(combined_logits)
        
        # Sample binary selection during training, use threshold during evaluation
        if self.training:
            target_selection = torch.bernoulli(target_probs)
        else:
            target_selection = (target_probs > 0.5).float()
        
        # Ensure at least one target is selected if any are visible
        if target_mask is not None:
            no_selection = (target_selection.sum(dim=-1) == 0) & (target_mask.sum(dim=-1) > 0)
            if no_selection.any():
                # Select the most probable visible target
                masked_probs = target_probs.clone()
                masked_probs[~target_mask.bool()] = -1
                best_targets = torch.argmax(masked_probs, dim=-1)
                target_selection[no_selection, best_targets[no_selection]] = 1.0
        
        self.frame_counter += 1
        
        info = {
            'fast_logits': fast_logits,
            'slow_logits': slow_logits,
            'target_probs': target_probs,
            'frame_counter': self.frame_counter,
            'slow_selection_active': self.frame_counter % self.frame_skip == 0
        }
        
        return target_selection, info
    
    def reset(self):
        """Reset hierarchical state for new episode."""
        self.frame_counter = 0
        self.last_slow_selection = None


class QPLEXHRLModel(nn.Module):
    """
    QPLEX HRL Model combining hierarchical target selection with advanced Q-networks.
    
    This model integrates:
    1. Hierarchical target selection (inspired by MATE HRL)
    2. Advanced Q-networks from qplex_dev (AttentionRNN, HierarchicalRNN)
    3. Adaptive mixing network for multi-agent coordination
    """
    
    def __init__(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int,
                 n_targets: int, config: Dict[str, Any]):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.config = config
        
        # Network configuration
        network_config = config.get('network', {})
        q_net_config = network_config.get('q_network', {})
        mix_net_config = network_config.get('mixing_network', {})
        hrl_config = network_config.get('hrl', {})
        
        # Hierarchical target selector for each agent
        self.target_selectors = nn.ModuleList([
            HierarchicalTargetSelector(
                obs_dim=obs_dim,
                n_targets=n_targets,
                hidden_dim=hrl_config.get('selector_hidden_dim', 128),
                frame_skip=hrl_config.get('frame_skip', 5),
                use_attention=hrl_config.get('use_attention', True)
            )
            for _ in range(n_agents)
        ])
        
        # Enhanced Q-networks (from qplex_dev)
        q_net_type = q_net_config.get('type', 'attention_rnn')
        self.q_networks = nn.ModuleList()
        
        for i in range(n_agents):
            if q_net_type == 'attention_rnn':
                q_net = AttentionRNNQNetwork(
                    obs_dim=obs_dim + n_targets,  # obs + target selection
                    action_dim=action_dim,
                    hidden_dim=q_net_config.get('rnn_hidden_dim', 128),
                    num_layers=q_net_config.get('rnn_layers', 2),
                    rnn_type=q_net_config.get('rnn_type', 'lstm'),
                    num_heads=q_net_config.get('num_attention_heads', 4),
                    dropout=q_net_config.get('dropout', 0.1)
                )
            elif q_net_type == 'hierarchical_rnn':
                q_net = HierarchicalRNNQNetwork(
                    obs_dim=obs_dim + n_targets,
                    action_dim=action_dim,
                    hidden_dim=q_net_config.get('rnn_hidden_dim', 128),
                    num_layers=q_net_config.get('rnn_layers', 2),
                    rnn_type=q_net_config.get('rnn_type', 'lstm'),
                    dropout=q_net_config.get('dropout', 0.1)
                )
            else:
                # Fallback to standard Q-network with RNN
                from new_network.base_network import QNetwork
                q_net = QNetwork(
                    obs_dim=obs_dim + n_targets,
                    action_dim=action_dim,
                    hidden_dims=q_net_config.get('hidden_dims', [256, 256]),
                    use_rnn=q_net_config.get('use_rnn', True),
                    rnn_hidden_dim=q_net_config.get('rnn_hidden_dim', 128),
                    rnn_layers=q_net_config.get('rnn_layers', 2),
                    use_attention=q_net_config.get('use_attention', True),
                    num_attention_heads=q_net_config.get('num_attention_heads', 4)
                )
            
            self.q_networks.append(q_net)
        
        # Enhanced mixing network (from qplex_dev)
        mix_type = mix_net_config.get('type', 'adaptive')
        if mix_type == 'adaptive':
            # Adaptive mixing network
            self.mixing_net = AdaptiveMixingNetwork(
                state_dim=state_dim + n_agents * n_targets,  # state + all target selections
                n_agents=n_agents,
                hidden_dims=mix_net_config.get('hidden_dims', [256, 256]),
                complexity_threshold=mix_net_config.get('complexity_threshold', 0.6),
                use_hypernet=mix_net_config.get('use_hypernet', True)
            )
        else:
            # Standard mixing network
            self.mixing_net = MixingNetwork(
                state_dim=state_dim + n_agents * n_targets,
                n_agents=n_agents,
                hidden_dims=mix_net_config.get('hidden_dims', [256, 256]),
                use_hypernet=mix_net_config.get('use_hypernet', True)
            )
        
        # State encoder for global state processing
        if network_config.get('use_state_encoder', True):
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, network_config.get('state_encoder_hidden', 256)),
                nn.ReLU(),
                nn.Linear(network_config.get('state_encoder_hidden', 256), state_dim)
            )
        else:
            self.state_encoder = None
    
    def forward(self, obs: torch.Tensor, state: torch.Tensor, 
                target_masks: Optional[torch.Tensor] = None,
                hidden_states: Optional[List] = None) -> Tuple[torch.Tensor, torch.Tensor, List, Dict[str, Any]]:
        """
        Forward pass for QPLEX HRL model.
        
        Args:
            obs: Agent observations [batch_size, n_agents, obs_dim]
            state: Global state [batch_size, state_dim]
            target_masks: Target visibility masks [batch_size, n_agents, n_targets]
            hidden_states: RNN hidden states for each agent
        
        Returns:
            q_values: Individual Q-values [batch_size, n_agents, action_dim]
            q_total: Mixed Q-value [batch_size, 1]
            new_hidden_states: Updated hidden states
            info: Additional information
        """
        batch_size, n_agents = obs.shape[:2]
        
        if hidden_states is None:
            hidden_states = [None] * n_agents
        
        # Hierarchical target selection for each agent
        target_selections = []
        selection_info = []
        
        for i in range(n_agents):
            agent_obs = obs[:, i]  # [batch_size, obs_dim]
            agent_mask = target_masks[:, i] if target_masks is not None else None
            
            target_selection, sel_info = self.target_selectors[i](agent_obs, agent_mask)
            target_selections.append(target_selection)
            selection_info.append(sel_info)
        
        target_selections = torch.stack(target_selections, dim=1)  # [batch_size, n_agents, n_targets]
        
        # Augment observations with target selections
        augmented_obs = torch.cat([obs, target_selections], dim=-1)  # [batch_size, n_agents, obs_dim + n_targets]
        
        # Individual Q-values with enhanced networks
        q_values = []
        new_hidden_states = []
        
        for i in range(n_agents):
            agent_aug_obs = augmented_obs[:, i]  # [batch_size, obs_dim + n_targets]
            q_val, new_hidden = self.q_networks[i](agent_aug_obs, hidden_states[i])
            q_values.append(q_val)
            new_hidden_states.append(new_hidden)
        
        q_values = torch.stack(q_values, dim=1)  # [batch_size, n_agents, action_dim]
        
        # Encode global state
        if self.state_encoder is not None:
            encoded_state = self.state_encoder(state)
        else:
            encoded_state = state
        
        # Augment state with target selections
        flat_selections = target_selections.view(batch_size, -1)  # [batch_size, n_agents * n_targets]
        augmented_state = torch.cat([encoded_state, flat_selections], dim=-1)
        
        # Mixed Q-value
        q_input = q_values.max(dim=-1)[0]  # [batch_size, n_agents]
        q_total = self.mixing_net(q_input, augmented_state)
        
        # Compile information
        info = {
            'target_selections': target_selections,
            'selection_info': selection_info,
            'augmented_state': augmented_state,
            'q_individual_max': q_input
        }
        
        return q_values, q_total, new_hidden_states, info
    
    def reset_hierarchical_states(self):
        """Reset hierarchical states for all agents."""
        for selector in self.target_selectors:
            selector.reset()
    
    def get_target_selections(self, obs: torch.Tensor, target_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get target selections without Q-value computation."""
        batch_size, n_agents = obs.shape[:2]
        
        target_selections = []
        for i in range(n_agents):
            agent_obs = obs[:, i]
            agent_mask = target_masks[:, i] if target_masks is not None else None
            target_selection, _ = self.target_selectors[i](agent_obs, agent_mask)
            target_selections.append(target_selection)
        
        return torch.stack(target_selections, dim=1)


class AdaptiveMixingNetwork(nn.Module):
    """
    Adaptive mixing network that adjusts complexity based on task difficulty.
    Inspired by deeprec ideas and enhanced for HRL.
    """
    
    def __init__(self, state_dim: int, n_agents: int, hidden_dims: List[int] = [256, 256],
                 complexity_threshold: float = 0.6, use_hypernet: bool = True):
        super().__init__()
        
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.complexity_threshold = complexity_threshold
        self.use_hypernet = use_hypernet
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 1),
            nn.Sigmoid()
        )
        
        # Simple mixing network (for low complexity)
        self.simple_mixer = MixingNetwork(
            state_dim=state_dim,
            n_agents=n_agents,
            hidden_dims=[hidden_dims[0]],
            use_hypernet=False
        )
        
        # Complex mixing network (for high complexity)
        self.complex_mixer = MixingNetwork(
            state_dim=state_dim,
            n_agents=n_agents,
            hidden_dims=hidden_dims,
            use_hypernet=use_hypernet
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 1),
            nn.Sigmoid()
        )
    
    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Adaptive mixing based on estimated task complexity.
        
        Args:
            q_values: Individual Q-values [batch_size, n_agents]
            state: Global state [batch_size, state_dim]
        
        Returns:
            q_total: Mixed Q-value [batch_size, 1]
        """
        # Estimate task complexity
        complexity = self.complexity_estimator(state)  # [batch_size, 1]
        
        # Compute both simple and complex mixing
        q_simple = self.simple_mixer(q_values, state)
        q_complex = self.complex_mixer(q_values, state)
        
        # Adaptive gating
        gate_weight = self.gate(state)  # [batch_size, 1]
        
        # Use complexity to determine mixing strategy
        use_complex = (complexity > self.complexity_threshold).float()
        adaptive_weight = use_complex * gate_weight + (1 - use_complex) * (1 - gate_weight)
        
        # Weighted combination
        q_total = adaptive_weight * q_complex + (1 - adaptive_weight) * q_simple
        
        return q_total