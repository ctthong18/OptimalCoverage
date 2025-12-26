"""
Unit tests for QPLEX tensor shape transformations.

Tests cover:
- Individual Q-network output shapes with various batch sizes
- Stacking operation produces (batch, n_agents, action_dim)
- _ensure_q_individual with various input shapes (2D, 3D, edge cases)
- Mixing network forward pass with different batch sizes and n_agents values
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Tuple

from algorithms.qplex_dev.enhanced_model import EnhancedQPLEXNetwork
from algorithms.qplex_dev.mixer import (
    _ensure_q_individual,
    _validate_bmm_shapes,
    QPLEXMixingNetwork,
    AttentionMixingNetwork,
    MonotonicMixingNetwork,
    HierarchicalMixingNetwork,
    AdaptiveMixingNetwork
)
from new_network.rnn_network import AttentionRNNQNetwork


class TestIndividualQNetworkShapes:
    """Test individual Q-network output shapes with various batch sizes."""
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    @pytest.mark.parametrize("obs_dim", [10, 20])
    @pytest.mark.parametrize("action_dim", [2, 5])
    def test_attention_rnn_q_network_output_shape(self, batch_size, obs_dim, action_dim):
        """Test AttentionRNNQNetwork outputs correct shape for various batch sizes."""
        hidden_dim = 64
        num_layers = 1
        num_heads = 2
        
        q_net = AttentionRNNQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.0
        )
        
        # Create input
        obs = torch.randn(batch_size, obs_dim)
        
        # Forward pass
        q_values, hidden = q_net(obs, None)
        
        # Verify output shape
        assert q_values.shape == (batch_size, action_dim), \
            f"Expected shape ({batch_size}, {action_dim}), got {tuple(q_values.shape)}"
        assert torch.is_tensor(q_values), "Q-values must be a tensor"
        assert not torch.isnan(q_values).any(), "Q-values contain NaN"
        assert not torch.isinf(q_values).any(), "Q-values contain Inf"


class TestStackingOperation:
    """Test stacking operation produces (batch, n_agents, action_dim)."""
    
    @pytest.mark.parametrize("batch_size,n_agents,action_dim", [
        (1, 4, 2),
        (8, 4, 2),
        (16, 8, 5),
        (32, 3, 10)
    ])
    def test_stack_q_values_from_multiple_agents(self, batch_size, n_agents, action_dim):
        """Test stacking Q-values from multiple agents produces correct shape."""
        # Simulate Q-values from individual agents
        q_values_list = []
        for _ in range(n_agents):
            q_val = torch.randn(batch_size, action_dim)
            q_values_list.append(q_val)
        
        # Stack along agent dimension
        q_values_stacked = torch.stack(q_values_list, dim=1)
        
        # Verify shape
        expected_shape = (batch_size, n_agents, action_dim)
        assert q_values_stacked.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {tuple(q_values_stacked.shape)}"
    
    @pytest.mark.parametrize("batch_size,n_agents,action_dim", [
        (1, 4, 2),
        (16, 8, 5)
    ])
    def test_max_over_actions_produces_individual_q(self, batch_size, n_agents, action_dim):
        """Test max operation over actions produces (batch, n_agents)."""
        q_values = torch.randn(batch_size, n_agents, action_dim)
        
        # Max over actions
        q_individual = q_values.max(dim=-1)[0]
        
        # Verify shape
        expected_shape = (batch_size, n_agents)
        assert q_individual.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {tuple(q_individual.shape)}"


class TestEnsureQIndividual:
    """Test _ensure_q_individual with various input shapes."""
    
    @pytest.mark.parametrize("batch_size,n_agents", [(1, 4), (8, 4), (16, 8)])
    def test_2d_correct_shape(self, batch_size, n_agents):
        """Test 2D input with correct shape (batch, n_agents)."""
        q_values = torch.randn(batch_size, n_agents)
        result = _ensure_q_individual(q_values, n_agents, caller="test")
        
        assert result.shape == (batch_size, n_agents)
        assert torch.equal(result, q_values), "Should return unchanged for correct 2D shape"
    
    @pytest.mark.parametrize("batch_size,n_agents,action_dim", [
        (1, 4, 2),
        (8, 4, 5),
        (16, 8, 3)
    ])
    def test_3d_with_actions(self, batch_size, n_agents, action_dim):
        """Test 3D input (batch, n_agents, action_dim) returns max over actions."""
        q_values = torch.randn(batch_size, n_agents, action_dim)
        result = _ensure_q_individual(q_values, n_agents, caller="test")
        
        expected_shape = (batch_size, n_agents)
        assert result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {tuple(result.shape)}"
        
        # Verify it's actually the max
        expected_max = q_values.max(dim=-1)[0]
        assert torch.allclose(result, expected_max), "Should return max over action dimension"
    
    @pytest.mark.parametrize("batch_size,n_agents", [(1, 4), (8, 8)])
    def test_3d_with_singleton_action(self, batch_size, n_agents):
        """Test 3D input (batch, n_agents, 1) squeezes last dimension."""
        q_values = torch.randn(batch_size, n_agents, 1)
        result = _ensure_q_individual(q_values, n_agents, caller="test")
        
        expected_shape = (batch_size, n_agents)
        assert result.shape == expected_shape
        assert torch.allclose(result, q_values.squeeze(-1))
    
    @pytest.mark.parametrize("batch_size,n_agents,action_dim", [
        (1, 4, 2),
        (8, 4, 5)
    ])
    def test_2d_flattened_reshapes_correctly(self, batch_size, n_agents, action_dim):
        """Test 2D flattened input (batch, n_agents * action_dim) reshapes and reduces."""
        q_values = torch.randn(batch_size, n_agents * action_dim)
        result = _ensure_q_individual(q_values, n_agents, caller="test")
        
        expected_shape = (batch_size, n_agents)
        assert result.shape == expected_shape
    
    def test_invalid_2d_shape_raises_error(self):
        """Test invalid 2D shape raises informative error."""
        n_agents = 4
        q_values = torch.randn(2, 7)  # 7 is not divisible by 4
        
        with pytest.raises(RuntimeError) as exc_info:
            _ensure_q_individual(q_values, n_agents, caller="test_caller")
        
        error_msg = str(exc_info.value)
        assert "test_caller" in error_msg
        assert "7" in error_msg
        assert "4" in error_msg
    
    def test_transposed_3d_shape_corrects(self):
        """Test transposed 3D shape (batch, action_dim, n_agents) gets corrected."""
        batch_size, n_agents, action_dim = 2, 4, 5
        # Create transposed shape
        q_values = torch.randn(batch_size, action_dim, n_agents)
        result = _ensure_q_individual(q_values, n_agents, caller="test")
        
        expected_shape = (batch_size, n_agents)
        assert result.shape == expected_shape
    
    def test_non_tensor_raises_error(self):
        """Test non-tensor input raises error."""
        with pytest.raises(RuntimeError) as exc_info:
            _ensure_q_individual([1, 2, 3], 4, caller="test")
        
        assert "must be a torch.Tensor" in str(exc_info.value)
    
    def test_wrong_dimensions_raises_error(self):
        """Test wrong number of dimensions raises error."""
        q_values = torch.randn(2, 4, 5, 3)  # 4D tensor
        
        with pytest.raises(RuntimeError) as exc_info:
            _ensure_q_individual(q_values, 4, caller="test")
        
        assert "unsupported dim" in str(exc_info.value)


class TestValidateBMMShapes:
    """Test _validate_bmm_shapes helper function."""
    
    def test_valid_bmm_shapes(self):
        """Test valid BMM shapes pass validation."""
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        
        # Should not raise
        _validate_bmm_shapes(a, b, "test_operation")
    
    def test_invalid_dimensions_raises_error(self):
        """Test non-3D tensors raise error."""
        a = torch.randn(2, 3)  # 2D
        b = torch.randn(2, 3, 4)  # 3D
        
        with pytest.raises(RuntimeError) as exc_info:
            _validate_bmm_shapes(a, b, "test_op")
        
        error_msg = str(exc_info.value)
        assert "test_op" in error_msg
        assert "3D tensors" in error_msg
    
    def test_mismatched_batch_sizes_raises_error(self):
        """Test mismatched batch sizes raise error."""
        a = torch.randn(2, 3, 4)
        b = torch.randn(5, 4, 5)  # Different batch size
        
        with pytest.raises(RuntimeError) as exc_info:
            _validate_bmm_shapes(a, b, "test_op")
        
        error_msg = str(exc_info.value)
        assert "Batch sizes must match" in error_msg
        assert "2" in error_msg and "5" in error_msg
    
    def test_incompatible_inner_dimensions_raises_error(self):
        """Test incompatible inner dimensions raise error."""
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 7, 5)  # a.size(2)=4 != b.size(1)=7
        
        with pytest.raises(RuntimeError) as exc_info:
            _validate_bmm_shapes(a, b, "test_op")
        
        error_msg = str(exc_info.value)
        assert "Inner dimensions must be compatible" in error_msg
        assert "4" in error_msg and "7" in error_msg


class TestMixingNetworkShapes:
    """Test mixing network forward pass with different batch sizes and n_agents."""
    
    @pytest.mark.parametrize("batch_size,n_agents,state_dim", [
        (1, 4, 20),
        (8, 4, 20),
        (16, 8, 30),
        (32, 3, 15)
    ])
    def test_qplex_mixing_network_shapes(self, batch_size, n_agents, state_dim):
        """Test QPLEXMixingNetwork with various batch sizes and n_agents."""
        hidden_dims = [128, 128]
        
        mixer = QPLEXMixingNetwork(
            state_dim=state_dim,
            n_agents=n_agents,
            hidden_dims=hidden_dims,
            use_hypernet=True,
            dueling=True,
            debug=False
        )
        
        # Create inputs
        q_values = torch.randn(batch_size, n_agents)
        state = torch.randn(batch_size, state_dim)
        
        # Forward pass
        q_total = mixer(q_values, state)
        
        # Verify output shape
        expected_shape = (batch_size, 1)
        assert q_total.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {tuple(q_total.shape)}"
        assert not torch.isnan(q_total).any(), "Q_total contains NaN"
        assert not torch.isinf(q_total).any(), "Q_total contains Inf"
    
    @pytest.mark.parametrize("batch_size,n_agents,state_dim", [
        (1, 4, 20),
        (16, 8, 30)
    ])
    def test_qplex_mixing_with_3d_input(self, batch_size, n_agents, state_dim):
        """Test QPLEXMixingNetwork handles 3D input (batch, n_agents, action_dim)."""
        action_dim = 5
        hidden_dims = [128, 128]
        
        mixer = QPLEXMixingNetwork(
            state_dim=state_dim,
            n_agents=n_agents,
            hidden_dims=hidden_dims,
            use_hypernet=True,
            dueling=True,
            debug=False
        )
        
        # Create 3D input
        q_values = torch.randn(batch_size, n_agents, action_dim)
        state = torch.randn(batch_size, state_dim)
        
        # Forward pass - should handle 3D input via _ensure_q_individual
        q_total = mixer(q_values, state)
        
        # Verify output shape
        expected_shape = (batch_size, 1)
        assert q_total.shape == expected_shape
    
    @pytest.mark.parametrize("batch_size,n_agents,state_dim", [
        (1, 4, 20),
        (16, 8, 30)
    ])
    def test_attention_mixing_network_shapes(self, batch_size, n_agents, state_dim):
        """Test AttentionMixingNetwork with various configurations."""
        hidden_dim = 128
        num_heads = 4
        
        mixer = AttentionMixingNetwork(
            state_dim=state_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            debug=False
        )
        
        q_values = torch.randn(batch_size, n_agents)
        state = torch.randn(batch_size, state_dim)
        
        q_total = mixer(q_values, state)
        
        expected_shape = (batch_size, 1)
        assert q_total.shape == expected_shape
    
    @pytest.mark.parametrize("batch_size,n_agents,state_dim", [
        (1, 4, 20),
        (16, 8, 30)
    ])
    def test_monotonic_mixing_network_shapes(self, batch_size, n_agents, state_dim):
        """Test MonotonicMixingNetwork with various configurations."""
        hidden_dims = [128, 128]
        
        mixer = MonotonicMixingNetwork(
            state_dim=state_dim,
            n_agents=n_agents,
            hidden_dims=hidden_dims,
            debug=False
        )
        
        q_values = torch.randn(batch_size, n_agents)
        state = torch.randn(batch_size, state_dim)
        
        q_total = mixer(q_values, state)
        
        expected_shape = (batch_size, 1)
        assert q_total.shape == expected_shape
    
    @pytest.mark.parametrize("batch_size,n_agents,state_dim", [
        (1, 4, 20),
        (16, 8, 30)
    ])
    def test_hierarchical_mixing_network_shapes(self, batch_size, n_agents, state_dim):
        """Test HierarchicalMixingNetwork with various configurations."""
        hidden_dims = [128, 128]
        num_levels = 2
        
        mixer = HierarchicalMixingNetwork(
            state_dim=state_dim,
            n_agents=n_agents,
            hidden_dims=hidden_dims,
            num_levels=num_levels,
            debug=False
        )
        
        q_values = torch.randn(batch_size, n_agents)
        state = torch.randn(batch_size, state_dim)
        
        q_total = mixer(q_values, state)
        
        expected_shape = (batch_size, 1)
        assert q_total.shape == expected_shape
    
    @pytest.mark.parametrize("batch_size,n_agents,state_dim", [
        (1, 4, 20),
        (16, 8, 30)
    ])
    def test_adaptive_mixing_network_shapes(self, batch_size, n_agents, state_dim):
        """Test AdaptiveMixingNetwork with various configurations."""
        hidden_dims = [128, 128]
        
        mixer = AdaptiveMixingNetwork(
            state_dim=state_dim,
            n_agents=n_agents,
            hidden_dims=hidden_dims,
            complexity_threshold=0.5,
            debug=False
        )
        
        q_values = torch.randn(batch_size, n_agents)
        state = torch.randn(batch_size, state_dim)
        
        q_total = mixer(q_values, state)
        
        expected_shape = (batch_size, 1)
        assert q_total.shape == expected_shape


class TestEnhancedQPLEXNetworkShapes:
    """Test EnhancedQPLEXNetwork end-to-end shape transformations."""
    
    @pytest.mark.parametrize("batch_size,n_agents", [(1, 4), (8, 4), (16, 8)])
    def test_forward_pass_output_shapes(self, batch_size, n_agents):
        """Test full forward pass produces correct output shapes."""
        obs_dim = 10
        action_dim = 2
        state_dim = 20
        
        config = {
            'debug': False,
            'network': {
                'q_network': {
                    'type': 'attention_rnn',
                    'rnn_hidden_dim': 64,
                    'rnn_layers': 1,
                    'num_attention_heads': 2,
                    'dropout': 0.0
                },
                'mixing_network': {
                    'type': 'qplex',
                    'hidden_dims': [128, 128],
                    'use_hypernet': True,
                    'dueling': True
                },
                'use_state_encoder': False
            }
        }
        
        network = EnhancedQPLEXNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            config=config
        )
        
        # Create inputs
        obs = torch.randn(batch_size, n_agents, obs_dim)
        state = torch.randn(batch_size, state_dim)
        
        # Forward pass
        q_values, q_total, hidden = network(obs, state)
        
        # Verify shapes
        assert q_values.shape == (batch_size, n_agents, action_dim), \
            f"Expected q_values shape ({batch_size}, {n_agents}, {action_dim}), got {tuple(q_values.shape)}"
        assert q_total.shape == (batch_size, 1), \
            f"Expected q_total shape ({batch_size}, 1), got {tuple(q_total.shape)}"
        assert len(hidden) == n_agents, \
            f"Expected {n_agents} hidden states, got {len(hidden)}"
    
    @pytest.mark.parametrize("mixer_type", [
        'qplex', 'adaptive', 'hierarchical', 'attention'
    ])
    def test_different_mixer_types(self, mixer_type):
        """Test forward pass with different mixing network types."""
        batch_size = 4
        n_agents = 4
        obs_dim = 10
        action_dim = 2
        state_dim = 20
        
        config = {
            'debug': False,
            'network': {
                'q_network': {
                    'type': 'attention_rnn',
                    'rnn_hidden_dim': 64,
                    'rnn_layers': 1,
                    'num_attention_heads': 2,
                    'dropout': 0.0
                },
                'mixing_network': {
                    'type': mixer_type,
                    'hidden_dims': [128, 128]
                },
                'use_state_encoder': False
            }
        }
        
        network = EnhancedQPLEXNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            config=config
        )
        
        obs = torch.randn(batch_size, n_agents, obs_dim)
        state = torch.randn(batch_size, state_dim)
        
        q_values, q_total, hidden = network(obs, state)
        
        assert q_values.shape == (batch_size, n_agents, action_dim)
        assert q_total.shape == (batch_size, 1)


class TestIntegrationFullForwardPass:
    """Integration tests for full EnhancedQPLEXNetwork forward pass.
    
    Tests verify:
    - No shape errors occur during full forward pass
    - Output has correct shape (batch, 1) for q_total
    - Works with different batch sizes (1 and 32)
    - Works with different n_agents (4 and 8)
    - Works with different mixing network types
    """
    
    @pytest.mark.parametrize("batch_size", [1, 32])
    @pytest.mark.parametrize("n_agents", [4, 8])
    def test_full_forward_pass_various_batch_and_agents(self, batch_size, n_agents):
        """Test full forward pass with batch_size=1,32 and n_agents=4,8."""
        obs_dim = 15
        action_dim = 3
        state_dim = 25
        
        config = {
            'debug': False,
            'network': {
                'q_network': {
                    'type': 'attention_rnn',
                    'rnn_hidden_dim': 64,
                    'rnn_layers': 1,
                    'num_attention_heads': 2,
                    'dropout': 0.0
                },
                'mixing_network': {
                    'type': 'qplex',
                    'hidden_dims': [128, 128],
                    'use_hypernet': True,
                    'dueling': True
                },
                'use_state_encoder': False
            }
        }
        
        network = EnhancedQPLEXNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            config=config
        )
        
        # Create inputs
        obs = torch.randn(batch_size, n_agents, obs_dim)
        state = torch.randn(batch_size, state_dim)
        
        # Forward pass - should not raise any shape errors
        q_values, q_total, hidden = network(obs, state)
        
        # Verify output shapes
        assert q_values.shape == (batch_size, n_agents, action_dim), \
            f"Expected q_values shape ({batch_size}, {n_agents}, {action_dim}), got {tuple(q_values.shape)}"
        assert q_total.shape == (batch_size, 1), \
            f"Expected q_total shape ({batch_size}, 1), got {tuple(q_total.shape)}"
        
        # Verify no NaN or Inf values
        assert not torch.isnan(q_values).any(), "q_values contains NaN"
        assert not torch.isinf(q_values).any(), "q_values contains Inf"
        assert not torch.isnan(q_total).any(), "q_total contains NaN"
        assert not torch.isinf(q_total).any(), "q_total contains Inf"
        
        # Verify hidden states
        assert len(hidden) == n_agents, f"Expected {n_agents} hidden states, got {len(hidden)}"
    
    @pytest.mark.parametrize("mixer_type", ['adaptive', 'hierarchical', 'attention'])
    @pytest.mark.parametrize("batch_size", [1, 32])
    def test_full_forward_pass_different_mixers(self, mixer_type, batch_size):
        """Test full forward pass with different mixing network types."""
        n_agents = 4
        obs_dim = 12
        action_dim = 2
        state_dim = 20
        
        config = {
            'debug': False,
            'network': {
                'q_network': {
                    'type': 'attention_rnn',
                    'rnn_hidden_dim': 64,
                    'rnn_layers': 1,
                    'num_attention_heads': 2,
                    'dropout': 0.0
                },
                'mixing_network': {
                    'type': mixer_type,
                    'hidden_dims': [128, 128]
                },
                'use_state_encoder': False
            }
        }
        
        network = EnhancedQPLEXNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            config=config
        )
        
        # Create inputs
        obs = torch.randn(batch_size, n_agents, obs_dim)
        state = torch.randn(batch_size, state_dim)
        
        # Forward pass - should not raise any shape errors
        q_values, q_total, hidden = network(obs, state)
        
        # Verify output shapes
        assert q_values.shape == (batch_size, n_agents, action_dim), \
            f"[{mixer_type}] Expected q_values shape ({batch_size}, {n_agents}, {action_dim}), got {tuple(q_values.shape)}"
        assert q_total.shape == (batch_size, 1), \
            f"[{mixer_type}] Expected q_total shape ({batch_size}, 1), got {tuple(q_total.shape)}"
        
        # Verify no NaN or Inf values
        assert not torch.isnan(q_values).any(), f"[{mixer_type}] q_values contains NaN"
        assert not torch.isinf(q_values).any(), f"[{mixer_type}] q_values contains Inf"
        assert not torch.isnan(q_total).any(), f"[{mixer_type}] q_total contains NaN"
        assert not torch.isinf(q_total).any(), f"[{mixer_type}] q_total contains Inf"
    
    @pytest.mark.parametrize("n_agents", [4, 8])
    def test_full_forward_pass_with_state_encoder(self, n_agents):
        """Test full forward pass with state encoder enabled."""
        batch_size = 16
        obs_dim = 10
        action_dim = 2
        state_dim = 20
        
        config = {
            'debug': False,
            'network': {
                'q_network': {
                    'type': 'attention_rnn',
                    'rnn_hidden_dim': 64,
                    'rnn_layers': 1,
                    'num_attention_heads': 2,
                    'dropout': 0.0
                },
                'mixing_network': {
                    'type': 'qplex',
                    'hidden_dims': [128, 128],
                    'use_hypernet': True,
                    'dueling': True
                },
                'use_state_encoder': True,
                'state_encoder_hidden': 256
            }
        }
        
        network = EnhancedQPLEXNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            config=config
        )
        
        # Create inputs
        obs = torch.randn(batch_size, n_agents, obs_dim)
        state = torch.randn(batch_size, state_dim)
        
        # Forward pass
        q_values, q_total, hidden = network(obs, state)
        
        # Verify output shapes
        assert q_values.shape == (batch_size, n_agents, action_dim)
        assert q_total.shape == (batch_size, 1)
        
        # Verify no errors
        assert not torch.isnan(q_values).any()
        assert not torch.isnan(q_total).any()
    
    def test_full_forward_pass_multiple_iterations(self):
        """Test multiple forward passes to ensure consistency."""
        batch_size = 8
        n_agents = 4
        obs_dim = 10
        action_dim = 2
        state_dim = 20
        num_iterations = 5
        
        config = {
            'debug': False,
            'network': {
                'q_network': {
                    'type': 'attention_rnn',
                    'rnn_hidden_dim': 64,
                    'rnn_layers': 1,
                    'num_attention_heads': 2,
                    'dropout': 0.0
                },
                'mixing_network': {
                    'type': 'adaptive',
                    'hidden_dims': [128, 128]
                },
                'use_state_encoder': False
            }
        }
        
        network = EnhancedQPLEXNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            config=config
        )
        
        # Run multiple forward passes
        for i in range(num_iterations):
            obs = torch.randn(batch_size, n_agents, obs_dim)
            state = torch.randn(batch_size, state_dim)
            
            # Forward pass
            q_values, q_total, hidden = network(obs, state)
            
            # Verify shapes remain consistent
            assert q_values.shape == (batch_size, n_agents, action_dim), \
                f"Iteration {i}: q_values shape mismatch"
            assert q_total.shape == (batch_size, 1), \
                f"Iteration {i}: q_total shape mismatch"
            
            # Verify no errors
            assert not torch.isnan(q_values).any(), f"Iteration {i}: q_values contains NaN"
            assert not torch.isnan(q_total).any(), f"Iteration {i}: q_total contains NaN"
    
    def test_full_forward_pass_with_hidden_state_persistence(self):
        """Test forward pass with hidden state persistence across steps."""
        batch_size = 4
        n_agents = 4
        obs_dim = 10
        action_dim = 2
        state_dim = 20
        num_steps = 3
        
        config = {
            'debug': False,
            'network': {
                'q_network': {
                    'type': 'attention_rnn',
                    'rnn_hidden_dim': 64,
                    'rnn_layers': 1,
                    'num_attention_heads': 2,
                    'dropout': 0.0
                },
                'mixing_network': {
                    'type': 'qplex',
                    'hidden_dims': [128, 128],
                    'use_hypernet': True,
                    'dueling': True
                },
                'use_state_encoder': False
            }
        }
        
        network = EnhancedQPLEXNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            config=config
        )
        
        # Initialize hidden state
        hidden = None
        
        # Run multiple steps with hidden state persistence
        for step in range(num_steps):
            obs = torch.randn(batch_size, n_agents, obs_dim)
            state = torch.randn(batch_size, state_dim)
            
            # Forward pass with previous hidden state
            q_values, q_total, hidden = network(obs, state, hidden)
            
            # Verify shapes
            assert q_values.shape == (batch_size, n_agents, action_dim), \
                f"Step {step}: q_values shape mismatch"
            assert q_total.shape == (batch_size, 1), \
                f"Step {step}: q_total shape mismatch"
            
            # Verify hidden state structure
            assert len(hidden) == n_agents, \
                f"Step {step}: Expected {n_agents} hidden states, got {len(hidden)}"
            
            # Verify no errors
            assert not torch.isnan(q_values).any(), f"Step {step}: q_values contains NaN"
            assert not torch.isnan(q_total).any(), f"Step {step}: q_total contains NaN"
