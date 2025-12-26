"""Mixing network implementations for QPLEX algorithm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math

from networks.base_networks import MixingNetwork, Attention


def _ensure_q_individual(q_values: torch.Tensor, n_agents: int, caller: str = "") -> torch.Tensor:
    """
    Ensure q_values is (batch, n_agents) where each entry is the scalar
    individual Q (e.g. max over actions). Supports these input shapes:
      - (batch, n_agents) -> returned unchanged
      - (batch, n_agents, action_dim) -> returns max over last dim
      - (batch, M) where M is divisible by n_agents -> reshape to (batch, n_agents, k) and max over last dim
      - (batch, n_agents, 1) -> squeeze last dim
    Otherwise raise informative RuntimeError.
    """
    if not torch.is_tensor(q_values):
        raise RuntimeError(f"[{caller}] q_values must be a torch.Tensor, got {type(q_values)}")

    if q_values.dim() == 2:
        # (batch, n_agents) OR (batch, something_else)
        if q_values.size(1) == n_agents:
            return q_values
        else:
            # try to see if it's flattened (batch, n_agents * k)
            total = q_values.size(1)
            if total % n_agents == 0:
                k = total // n_agents
                # reshape and reduce
                q_reshaped = q_values.view(q_values.size(0), n_agents, k)
                return q_reshaped.max(dim=-1)[0]
            else:
                raise RuntimeError(f"[{caller}] q_values has shape {tuple(q_values.shape)}: second dimension "
                                   f"is {total} but expected {n_agents} (or divisible by {n_agents})")
    elif q_values.dim() == 3:
        # (batch, n_agents, action_dim) or (batch, n_agents, 1)
        if q_values.size(1) != n_agents:
            # maybe transposed or wrong ordering
            # try to detect if first dim after batch equals action_dim and second equals n_agents
            if q_values.size(2) == n_agents and q_values.size(1) != n_agents:
                # swap dims
                q_swapped = q_values.transpose(1, 2)  # (batch, n_agents, action_dim)
                return q_swapped.max(dim=-1)[0]
            else:
                raise RuntimeError(f"[{caller}] q_values has shape {tuple(q_values.shape)}: expected second dim == {n_agents}")
        # normal case
        if q_values.size(2) == 1:
            return q_values.squeeze(-1)
        return q_values.max(dim=-1)[0]
    else:
        raise RuntimeError(f"[{caller}] q_values has unsupported dim {q_values.dim()}, expected 2 or 3")


def _validate_bmm_shapes(a: torch.Tensor, b: torch.Tensor, operation: str = "BMM") -> None:
    """
    Validate tensor shapes for batch matrix multiplication (torch.bmm).
    
    Args:
        a: First tensor for BMM operation
        b: Second tensor for BMM operation
        operation: Name/context of the operation for error messages
        
    Raises:
        RuntimeError: If tensors don't meet BMM requirements:
            - Both tensors must be 3D
            - Batch sizes must match (dim 0)
            - Inner dimensions must be compatible: a.size(2) == b.size(1)
    """
    # Check that both tensors are 3D
    if a.dim() != 3 or b.dim() != 3:
        raise RuntimeError(
            f"[{operation}] BMM requires 3D tensors, got {a.dim()}D and {b.dim()}D. "
            f"Tensor shapes: a={tuple(a.shape)}, b={tuple(b.shape)}"
        )
    
    # Check that batch sizes match
    if a.size(0) != b.size(0):
        raise RuntimeError(
            f"[{operation}] Batch sizes must match for BMM. "
            f"Got a.size(0)={a.size(0)} vs b.size(0)={b.size(0)}. "
            f"Tensor shapes: a={tuple(a.shape)}, b={tuple(b.shape)}"
        )
    
    # Check that inner dimensions are compatible for matrix multiplication
    if a.size(2) != b.size(1):
        raise RuntimeError(
            f"[{operation}] Inner dimensions must be compatible for BMM: "
            f"a.size(2) must equal b.size(1). "
            f"Got a.size(2)={a.size(2)} vs b.size(1)={b.size(1)}. "
            f"Tensor shapes: a={tuple(a.shape)}, b={tuple(b.shape)}"
        )


class QPLEXMixingNetwork(nn.Module):
    """QPLEX-specific mixing network with dueling architecture.
    
    Debug Mode:
        Pass debug=True to enable shape tracking during forward pass.
        This will print intermediate tensor shapes to help diagnose dimension issues.
    """

    def __init__(self, state_dim: int, n_agents: int, hidden_dims: List[int] = [256, 256],
                 use_hypernet: bool = True, dueling: bool = True, debug: bool = False):
        super(QPLEXMixingNetwork, self).__init__()

        self.state_dim = state_dim
        self.n_agents = n_agents
        self.use_hypernet = use_hypernet
        self.dueling = dueling
        self.debug = debug

        if use_hypernet:
            # Hypernetwork approach for QPLEX
            self.hyper_w1 = nn.Linear(state_dim, hidden_dims[0] * n_agents)
            self.hyper_w2 = nn.Linear(state_dim, hidden_dims[1] * hidden_dims[0])
            self.hyper_b1 = nn.Linear(state_dim, hidden_dims[0])
            self.hyper_b2 = nn.Linear(state_dim, hidden_dims[1])

            if dueling:
                # Dueling architecture: separate value and advantage streams
                self.hyper_w_value = nn.Linear(state_dim, hidden_dims[1])
                self.hyper_w_advantage = nn.Linear(state_dim, hidden_dims[1] * n_agents)
                self.hyper_b_value = nn.Linear(state_dim, 1)
                self.hyper_b_advantage = nn.Linear(state_dim, n_agents)
            else:
                self.hyper_w_final = nn.Linear(state_dim, hidden_dims[1])
        else:
            # Standard mixing network
            self.mixing_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], 1)
            )

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_values: Individual Q-values of shape (batch_size, n_agents) or (batch_size, n_agents, action_dim)
            state: Global state of shape (batch_size, state_dim)

        Returns:
            q_total: Total Q-value of shape (batch_size, 1)
        """
        if self.debug:
            print(f"[QPLEXMixingNetwork.forward] Input: q_values={tuple(q_values.shape)}, state={tuple(state.shape)}")
        
        # Defensive preprocessing: ensure q_values is (batch, n_agents)
        q_values = _ensure_q_individual(q_values, self.n_agents, caller="QPLEXMixingNetwork")
        
        if self.debug:
            print(f"[QPLEXMixingNetwork.forward] After _ensure_q_individual: q_values={tuple(q_values.shape)}")

        if self.use_hypernet:
            batch_size = q_values.size(0)

            if self.dueling:
                # Dueling architecture
                # Value stream: computes a scalar baseline value per batch
                w_value = torch.abs(self.hyper_w_value(state))           # (batch, hidden_dim)
                b_value = self.hyper_b_value(state)                      # (batch, 1)

                # Advantage stream: computes agent-specific advantages
                w_advantage = torch.abs(self.hyper_w_advantage(state))   # (batch, hidden_dim * n_agents)
                b_advantage = self.hyper_b_advantage(state)              # (batch, n_agents)
                
                if self.debug:
                    print(f"[QPLEXMixingNetwork.dueling] w_advantage before reshape: {tuple(w_advantage.shape)}")
                
                # Reshape w_advantage to (batch, n_agents, hidden_dim)
                # This allows bmm: (batch, 1, n_agents) @ (batch, n_agents, hidden_dim) -> (batch, 1, hidden_dim)
                w_advantage = w_advantage.view(batch_size, self.n_agents, -1)
                
                if self.debug:
                    print(f"[QPLEXMixingNetwork.dueling] w_advantage after reshape: {tuple(w_advantage.shape)}")

                # Compute advantage: q_values weighted by w_advantage
                # q_values.unsqueeze(1): (batch, 1, n_agents)
                # w_advantage: (batch, n_agents, hidden_dim)
                # bmm result: (batch, 1, hidden_dim)
                q_unsqueezed = q_values.unsqueeze(1)
                
                if self.debug:
                    print(f"[QPLEXMixingNetwork.dueling] q_unsqueezed: {tuple(q_unsqueezed.shape)}")
                
                _validate_bmm_shapes(q_unsqueezed, w_advantage, "QPLEXMixingNetwork.dueling.advantage")
                advantage = torch.bmm(q_unsqueezed, w_advantage).squeeze(1)  # (batch, hidden_dim)
                
                if self.debug:
                    print(f"[QPLEXMixingNetwork.dueling] advantage after bmm: {tuple(advantage.shape)}")
                
                # Add advantage bias: weight each agent's bias by their Q-value contribution
                # Normalize q_values to get agent importance weights
                q_weights = F.softmax(q_values, dim=1)  # (batch, n_agents)
                # Weighted sum of biases
                advantage_bias = (q_weights * b_advantage).sum(dim=1, keepdim=True)  # (batch, 1)
                advantage = advantage + advantage_bias  # (batch, hidden_dim)

                # Compute value: aggregate Q-values weighted by learned value weights
                # w_value represents importance of each hidden dimension for value computation
                # Compute mean Q-value as baseline, then weight it
                q_mean = q_values.mean(dim=1, keepdim=True)  # (batch, 1)
                # Use w_value to compute a weighted scalar: dot product with a learned projection
                value_weight = w_value.mean(dim=1, keepdim=True)  # (batch, 1) - aggregate hidden dims
                value = value_weight * q_mean + b_value  # (batch, 1)

                # Combine value and advantage using dueling formula: Q_tot = V + (A - mean(A))
                # Reduce advantage to scalar by taking mean across hidden dimensions
                advantage_scalar = advantage.mean(dim=1, keepdim=True)  # (batch, 1)
                q_total = value + advantage_scalar
                
                if self.debug:
                    print(f"[QPLEXMixingNetwork.dueling] value: {tuple(value.shape)}, advantage_scalar: {tuple(advantage_scalar.shape)}, q_total: {tuple(q_total.shape)}")
            else:
                # Standard hypernetwork mixing
                # First layer
                w1 = torch.abs(self.hyper_w1(state))
                b1 = self.hyper_b1(state)
                w1 = w1.view(batch_size, self.n_agents, -1)
                b1 = b1.view(batch_size, 1, -1)

                q_unsqueezed = q_values.unsqueeze(1)
                _validate_bmm_shapes(q_unsqueezed, w1, "QPLEXMixingNetwork.standard.layer1")
                hidden = F.elu(torch.bmm(q_unsqueezed, w1) + b1)

                # Second layer
                w2 = torch.abs(self.hyper_w2(state))
                b2 = self.hyper_b2(state)
                w2 = w2.view(batch_size, -1, 1)
                b2 = b2.view(batch_size, 1, 1)

                _validate_bmm_shapes(hidden, w2, "QPLEXMixingNetwork.standard.layer2")
                q_total = torch.bmm(hidden, w2) + b2

                # Final scaling (optional)
                w_final = torch.abs(self.hyper_w_final(state))
                q_total = q_total * w_final
        else:
            # Standard mixing
            q_total = self.mixing_net(state) * q_values.sum(dim=1, keepdim=True)

        return q_total


class AttentionMixingNetwork(nn.Module):
    """Mixing network with attention mechanism.
    
    Debug Mode:
        Pass debug=True to enable shape tracking during forward pass.
    """

    def __init__(self, state_dim: int, n_agents: int, hidden_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.0, debug: bool = False):
        super(AttentionMixingNetwork, self).__init__()

        self.state_dim = state_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.debug = debug

        # State embedding
        self.state_embedding = nn.Linear(state_dim, hidden_dim)

        # Agent embedding
        self.agent_embedding = nn.Linear(1, hidden_dim)  # Q-values as input

        # Attention mechanism
        self.attention = Attention(hidden_dim, num_heads, dropout)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_values: Individual Q-values of shape (batch_size, n_agents) or (batch, n_agents, action_dim)
            state: Global state of shape (batch_size, state_dim)

        Returns:
            q_total: Total Q-value of shape (batch_size, 1)
        """
        # Ensure q_values is (batch, n_agents)
        q_values = _ensure_q_individual(q_values, self.n_agents, caller="AttentionMixingNetwork")

        batch_size = q_values.size(0)

        # Embed state
        state_emb = self.state_embedding(state)  # (batch_size, hidden_dim)
        state_emb = state_emb.unsqueeze(1).repeat(1, self.n_agents, 1)  # (batch_size, n_agents, hidden_dim)

        # Embed agent Q-values
        agent_emb = self.agent_embedding(q_values.unsqueeze(-1))  # (batch_size, n_agents, hidden_dim)

        # Combine state and agent embeddings
        combined_emb = state_emb + agent_emb
        combined_emb = self.dropout(combined_emb)

        # Apply attention
        attn_out = self.attention(combined_emb, combined_emb, combined_emb)

        # Residual connection
        attn_out = attn_out + combined_emb

        # Global pooling
        pooled = attn_out.mean(dim=1)  # (batch_size, hidden_dim)

        # Output
        q_total = self.output_layers(pooled)

        return q_total


class MonotonicMixingNetwork(nn.Module):
    """Monotonic mixing network ensuring monotonicity in Q-values.
    
    Debug Mode:
        Pass debug=True to enable shape tracking during forward pass.
    """

    def __init__(self, state_dim: int, n_agents: int, hidden_dims: List[int] = [256, 256], debug: bool = False):
        super(MonotonicMixingNetwork, self).__init__()

        self.state_dim = state_dim
        self.n_agents = n_agents
        self.debug = debug

        # Hypernetworks with monotonicity constraints
        self.hyper_w1 = nn.Linear(state_dim, hidden_dims[0] * n_agents)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dims[1] * hidden_dims[0])
        self.hyper_b1 = nn.Linear(state_dim, hidden_dims[0])
        self.hyper_b2 = nn.Linear(state_dim, hidden_dims[1])
        self.hyper_w_final = nn.Linear(state_dim, hidden_dims[1])

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_values: Individual Q-values of shape (batch_size, n_agents) or (batch, n_agents, action_dim)
            state: Global state of shape (batch_size, state_dim)

        Returns:
            q_total: Total Q-value of shape (batch_size, 1)
        """
        q_values = _ensure_q_individual(q_values, self.n_agents, caller="MonotonicMixingNetwork")

        batch_size = q_values.size(0)

        # First layer with monotonicity (positive weights)
        w1 = torch.abs(self.hyper_w1(state))
        b1 = self.hyper_b1(state)
        w1 = w1.view(batch_size, self.n_agents, -1)
        b1 = b1.view(batch_size, 1, -1)

        q_unsqueezed = q_values.unsqueeze(1)
        _validate_bmm_shapes(q_unsqueezed, w1, "MonotonicMixingNetwork.layer1")
        hidden = F.elu(torch.bmm(q_unsqueezed, w1) + b1)

        # Second layer with monotonicity (positive weights)
        w2 = torch.abs(self.hyper_w2(state))
        b2 = self.hyper_b2(state)
        w2 = w2.view(batch_size, -1, 1)
        b2 = b2.view(batch_size, 1, 1)

        _validate_bmm_shapes(hidden, w2, "MonotonicMixingNetwork.layer2")
        q_total = torch.bmm(hidden, w2) + b2

        # Final layer with monotonicity (positive weights)
        w_final = torch.abs(self.hyper_w_final(state))
        q_total = q_total * w_final

        return q_total


class HierarchicalMixingNetwork(nn.Module):
    """Hierarchical mixing network with multiple levels of abstraction.
    
    Debug Mode:
        Pass debug=True to enable shape tracking during forward pass.
    """

    def __init__(self, state_dim: int, n_agents: int, hidden_dims: List[int] = [256, 256],
                 num_levels: int = 2, debug: bool = False):
        super(HierarchicalMixingNetwork, self).__init__()

        self.state_dim = state_dim
        self.n_agents = n_agents
        self.num_levels = num_levels
        self.debug = debug

        # Multiple mixing networks for different levels
        self.mixing_networks = nn.ModuleList([
            QPLEXMixingNetwork(state_dim, n_agents, hidden_dims, use_hypernet=True, dueling=False, debug=debug)
            for _ in range(num_levels)
        ])

        # Level weights
        self.level_weights = nn.Linear(state_dim, num_levels)

        # Final combination
        self.final_combine = nn.Linear(num_levels, 1)

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_values: Individual Q-values of shape (batch_size, n_agents) or (batch, n_agents, action_dim)
            state: Global state of shape (batch_size, state_dim)

        Returns:
            q_total: Total Q-value of shape (batch_size, 1)
        """
        q_values = _ensure_q_individual(q_values, self.n_agents, caller="HierarchicalMixingNetwork")

        batch_size = q_values.size(0)

        # Get outputs from different levels
        level_outputs = []
        for mixing_net in self.mixing_networks:
            level_output = mixing_net(q_values, state)
            level_outputs.append(level_output)

        # Stack level outputs
        level_outputs = torch.cat(level_outputs, dim=1)  # (batch_size, num_levels)

        # Compute level weights
        level_weights = F.softmax(self.level_weights(state), dim=1)  # (batch_size, num_levels)

        # Weighted combination
        weighted_outputs = level_outputs * level_weights

        # Final combination
        q_total = self.final_combine(weighted_outputs)

        return q_total


class AdaptiveMixingNetwork(nn.Module):
    """Adaptive mixing network that adjusts based on state complexity.
    
    Debug Mode:
        Pass debug=True to enable shape tracking during forward pass.
    """

    def __init__(self, state_dim: int, n_agents: int, hidden_dims: List[int] = [256, 256],
                 complexity_threshold: float = 0.5, debug: bool = False):
        super(AdaptiveMixingNetwork, self).__init__()

        self.state_dim = state_dim
        self.n_agents = n_agents
        self.complexity_threshold = complexity_threshold
        self.debug = debug

        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Simple mixing network (for low complexity)
        self.simple_mixer = QPLEXMixingNetwork(state_dim, n_agents, hidden_dims,
                                              use_hypernet=False, dueling=False, debug=debug)

        # Complex mixing network (for high complexity)
        self.complex_mixer = QPLEXMixingNetwork(state_dim, n_agents, hidden_dims,
                                               use_hypernet=True, dueling=True, debug=debug)

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_values: Individual Q-values of shape (batch_size, n_agents) or (batch, n_agents, action_dim)
            state: Global state of shape (batch_size, state_dim)

        Returns:
            q_total: Total Q-value of shape (batch_size, 1)
        """
        if self.debug:
            print(f"[AdaptiveMixingNetwork.forward] Input: q_values={tuple(q_values.shape)}, state={tuple(state.shape)}")
        
        # Defensive: ensure q_values is individual scalars per agent
        q_values = _ensure_q_individual(q_values, self.n_agents, caller="AdaptiveMixingNetwork")

        # Estimate state complexity
        complexity = self.complexity_estimator(state)  # (batch_size, 1)
        
        if self.debug:
            print(f"[AdaptiveMixingNetwork.forward] Complexity: {complexity.mean().item():.4f}")

        # Get outputs from both mixers
        simple_output = self.simple_mixer(q_values, state)
        complex_output = self.complex_mixer(q_values, state)

        # Adaptive combination based on complexity
        alpha = torch.sigmoid((complexity - self.complexity_threshold) * 10)
        q_total = alpha * complex_output + (1 - alpha) * simple_output
        
        if self.debug:
            print(f"[AdaptiveMixingNetwork.forward] Alpha (complexity weight): {alpha.mean().item():.4f}, q_total: {tuple(q_total.shape)}")

        return q_total
