"""QPLEX HRL RLlib Policy - Ray RLlib integration for QPLEX HRL."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import copy

from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.modelv2 import ModelV2

from .model import QPLEXHRLModel
from .agent import QPLEXHRLAgent


class QPLEXHRLTorchModel(TorchModelV2, nn.Module):
    """
    QPLEX HRL model wrapper for RLlib.
    
    This class wraps our QPLEXHRLModel to be compatible with RLlib's TorchModelV2 interface.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        
        # Extract dimensions from observation space
        if hasattr(obs_space, 'original_space'):
            # Multi-agent case
            single_obs_space = obs_space.original_space.spaces[0]
            self.obs_dim = single_obs_space.shape[0]
            self.n_agents = len(obs_space.original_space.spaces)
        else:
            # Single agent case
            self.obs_dim = obs_space.shape[0]
            self.n_agents = 1
        
        # Extract action dimension
        if hasattr(action_space, 'n'):
            self.action_dim = action_space.n
        else:
            self.action_dim = action_space.shape[0]
        
        # Get state dimension and number of targets from model config
        self.state_dim = model_config.get('state_dim', 156)
        self.n_targets = model_config.get('n_targets', 4)
        
        # Create QPLEX HRL model
        hrl_config = model_config.get('custom_model_config', {})
        self.qplex_hrl_model = QPLEXHRLModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            n_agents=self.n_agents,
            n_targets=self.n_targets,
            config=hrl_config
        )
        
        # RNN state tracking
        self._hidden_states = None
        self._seq_lens = None
    
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        """
        Forward pass for RLlib.
        
        Args:
            input_dict: Dictionary containing 'obs' and optionally 'state_in'
            state: RNN hidden states
            seq_lens: Sequence lengths for RNN
        
        Returns:
            logits: Action logits [batch_size * n_agents, action_dim]
            new_state: Updated RNN hidden states
        """
        obs = input_dict['obs']  # [batch_size, obs_dim] or [batch_size, n_agents, obs_dim]
        
        # Handle different observation formats
        if len(obs.shape) == 2:
            # Single agent or flattened multi-agent
            batch_size = obs.shape[0]
            if self.n_agents > 1:
                # Reshape to multi-agent format
                obs = obs.view(batch_size, self.n_agents, self.obs_dim)
            else:
                obs = obs.unsqueeze(1)  # Add agent dimension
        else:
            batch_size = obs.shape[0]
        
        # Get global state (if available)
        if 'state_in' in input_dict:
            global_state = input_dict['state_in']
        else:
            # Create dummy global state
            global_state = torch.zeros(batch_size, self.state_dim, device=obs.device)
        
        # Extract target masks (if available)
        target_masks = None
        if 'target_masks' in input_dict:
            target_masks = input_dict['target_masks']
        
        # Convert RLlib state format to our format
        hidden_states = None
        if state and len(state) > 0:
            # Convert from RLlib format to our format
            hidden_states = self._convert_rllib_state_to_internal(state)
        
        # Forward pass through QPLEX HRL model
        q_values, q_total, new_hidden_states, info = self.qplex_hrl_model(
            obs, global_state, target_masks, hidden_states
        )
        
        # Use individual Q-values as logits for action selection
        # Reshape to [batch_size * n_agents, action_dim] for RLlib
        logits = q_values.view(batch_size * self.n_agents, self.action_dim)
        
        # Convert internal state format back to RLlib format
        new_state = self._convert_internal_state_to_rllib(new_hidden_states)
        
        # Store additional info for value function
        self._last_q_total = q_total
        self._last_info = info
        
        return logits, new_state
    
    def value_function(self) -> TensorType:
        """Return the value function (Q_total from mixing network)."""
        if hasattr(self, '_last_q_total') and self._last_q_total is not None:
            return self._last_q_total.squeeze(-1)  # Remove last dimension
        else:
            return torch.zeros(1)
    
    def _convert_rllib_state_to_internal(self, rllib_state: List[TensorType]) -> List[Tuple]:
        """Convert RLlib state format to our internal format."""
        # This is a simplified conversion - may need adjustment based on actual RNN structure
        if not rllib_state:
            return None
        
        # Assume LSTM states: [h_0, c_0] for each agent
        hidden_states = []
        state_per_agent = len(rllib_state) // self.n_agents
        
        for i in range(self.n_agents):
            if state_per_agent == 2:  # LSTM
                h = rllib_state[i * 2]
                c = rllib_state[i * 2 + 1]
                hidden_states.append((h, c))
            else:  # GRU or other
                h = rllib_state[i]
                hidden_states.append((h,))
        
        return hidden_states
    
    def _convert_internal_state_to_rllib(self, internal_state: List[Tuple]) -> List[TensorType]:
        """Convert our internal state format to RLlib format."""
        if not internal_state:
            return []
        
        rllib_state = []
        for agent_state in internal_state:
            if agent_state is not None:
                for state_tensor in agent_state:
                    rllib_state.append(state_tensor)
        
        return rllib_state
    
    def get_initial_state(self) -> List[TensorType]:
        """Get initial RNN state."""
        # Initialize hidden states for all agents
        initial_states = []
        
        # Assume LSTM with hidden_dim from model config
        hidden_dim = self.model_config.get('lstm_cell_size', 128)
        
        for _ in range(self.n_agents):
            # LSTM: (h_0, c_0)
            h_0 = torch.zeros(1, hidden_dim)
            c_0 = torch.zeros(1, hidden_dim)
            initial_states.extend([h_0, c_0])
        
        return initial_states


class QPLEXHRLTorchPolicy(TorchPolicy):
    """
    QPLEX HRL Torch Policy for RLlib.
    
    This policy integrates our QPLEX HRL implementation with RLlib's distributed training framework.
    """
    
    def __init__(self, observation_space, action_space, config):
        # Set up model configuration
        model_config = config.get('model', {})
        model_config['custom_model_config'] = config.get('custom_model_config', {})
        
        # Extract environment dimensions from config
        env_config = config.get('env_config', {})
        model_config['state_dim'] = env_config.get('state_dim', 156)
        model_config['n_targets'] = env_config.get('n_targets', 4)
        
        config['model'] = model_config
        
        # Initialize parent policy
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            config=config,
            model_class=QPLEXHRLTorchModel,
            loss_fn=self._qplex_hrl_loss,
            action_distribution_class=None  # Will use default
        )
        
        # QPLEX HRL specific configuration
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.mixer_type = config.get('mixer', 'qplex')
        self.mixing_embed_dim = config.get('mixing_embed_dim', 128)
        
        # Target network for QPLEX
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        
        # Freeze target network parameters
        for param in self.target_model.parameters():
            param.requires_grad = False
        
        # Training step counter for target network updates
        self._training_step = 0
        self.target_update_freq = config.get('target_network_update_freq', 500)
    
    def _qplex_hrl_loss(self, policy, model, dist_class, train_batch):
        """
        QPLEX HRL loss function.
        
        This implements the QPLEX loss with hierarchical enhancements.
        """
        # Extract batch data
        obs = train_batch['obs']
        actions = train_batch['actions']
        rewards = train_batch['rewards']
        next_obs = train_batch['new_obs']
        dones = train_batch['dones']
        
        # Get current Q-values
        logits, _ = model(train_batch, [], None)
        
        # Reshape logits back to [batch_size, n_agents, action_dim]
        batch_size = obs.shape[0]
        n_agents = model.n_agents
        action_dim = model.action_dim
        
        q_values = logits.view(batch_size, n_agents, action_dim)
        
        # Gather Q-values for taken actions
        actions_long = actions.long().unsqueeze(-1)
        current_q_values = q_values.gather(-1, actions_long).squeeze(-1)
        
        # Get target Q-values
        with torch.no_grad():
            # Create next_obs input dict
            next_input_dict = {'obs': next_obs}
            
            # Double Q-learning: use main network to select actions
            next_logits, _ = model(next_input_dict, [], None)
            next_q_values = next_logits.view(batch_size, n_agents, action_dim)
            next_actions = torch.argmax(next_q_values, dim=-1)
            
            # Use target network to evaluate
            target_next_logits, _ = self.target_model(next_input_dict, [], None)
            target_next_q_values = target_next_logits.view(batch_size, n_agents, action_dim)
            
            next_actions_long = next_actions.unsqueeze(-1)
            target_q_values = target_next_q_values.gather(-1, next_actions_long).squeeze(-1)
            
            # Compute target values
            target_values = rewards + self.gamma * target_q_values * (1 - dones.unsqueeze(-1))
        
        # TD error
        td_error = nn.functional.mse_loss(current_q_values, target_values, reduction='none')
        
        # Mean over agents and batch
        loss = td_error.mean()
        
        # Additional losses for HRL (if available)
        if hasattr(model, '_last_info') and model._last_info is not None:
            info = model._last_info
            
            # Selection entropy loss (encourage diverse target selection)
            if 'target_selections' in info:
                target_selections = info['target_selections']
                selection_probs = torch.sigmoid(target_selections)
                
                entropy = -torch.sum(
                    selection_probs * torch.log(selection_probs + 1e-8) +
                    (1 - selection_probs) * torch.log(1 - selection_probs + 1e-8),
                    dim=-1
                ).mean()
                
                # Add entropy regularization
                entropy_weight = policy.config.get('selection_entropy_weight', 0.01)
                loss = loss - entropy_weight * entropy
        
        # Store statistics
        policy._loss_stats = {
            'td_error': td_error.mean().item(),
            'q_values': current_q_values.mean().item(),
            'target_values': target_values.mean().item(),
            'loss': loss.item()
        }
        
        return loss
    
    def learn_on_batch(self, samples):
        """Learn on a batch of samples."""
        # Increment training step
        self._training_step += 1
        
        # Call parent learn_on_batch
        result = super().learn_on_batch(samples)
        
        # Update target network
        if self._training_step % self.target_update_freq == 0:
            self._update_target_network()
            result['target_network_updated'] = True
        
        # Add HRL-specific statistics
        if hasattr(self, '_loss_stats'):
            result.update(self._loss_stats)
        
        return result
    
    def _update_target_network(self):
        """Update target network using soft update."""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def get_weights(self):
        """Get model weights."""
        weights = super().get_weights()
        weights['target_model'] = self.target_model.state_dict()
        weights['training_step'] = self._training_step
        return weights
    
    def set_weights(self, weights):
        """Set model weights."""
        super().set_weights(weights)
        if 'target_model' in weights:
            self.target_model.load_state_dict(weights['target_model'])
        if 'training_step' in weights:
            self._training_step = weights['training_step']
    
    def export_model(self, export_dir):
        """Export model for deployment."""
        # Export main model
        super().export_model(export_dir)
        
        # Export target model
        target_path = f"{export_dir}/target_model.pt"
        torch.save(self.target_model.state_dict(), target_path)
        
        return export_dir


# Register the policy with RLlib
def get_policy_class(config):
    """Get the QPLEX HRL policy class."""
    return QPLEXHRLTorchPolicy


# Policy mapping function for multi-agent
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map agents to policies."""
    return "qplex_hrl_policy"