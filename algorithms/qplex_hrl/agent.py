"""QPLEX HRL Agent - Hierarchical agent with target selection capabilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import copy

from .model import QPLEXHRLModel


class QPLEXHRLAgent:
    """
    QPLEX HRL Agent with hierarchical target selection and advanced Q-networks.
    
    This agent combines:
    1. Hierarchical target selection (fast + slow timescales)
    2. Enhanced Q-networks from qplex_dev
    3. Adaptive exploration strategies
    4. Coverage-aware action selection
    """
    
    def __init__(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int,
                 n_targets: int, config: Dict[str, Any], device: torch.device):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.config = config
        self.device = device
        
        # Algorithm configuration
        algo_config = config.get('algorithm', {})
        self.gamma = algo_config.get('gamma', 0.99)
        self.tau = algo_config.get('tau', 0.005)
        self.epsilon = algo_config.get('epsilon_start', 1.0)
        self.epsilon_end = algo_config.get('epsilon_end', 0.05)
        self.epsilon_decay = algo_config.get('epsilon_decay', 0.9995)
        self.double_q = algo_config.get('double_q', True)
        self.dueling = algo_config.get('dueling', True)
        
        # HRL specific configuration
        hrl_config = config.get('network', {}).get('hrl', {})
        self.frame_skip = hrl_config.get('frame_skip', 5)
        self.coverage_weight = hrl_config.get('coverage_weight', 1.0)
        self.exploration_bonus = hrl_config.get('exploration_bonus', 0.1)
        
        # Create main network
        self.q_network = QPLEXHRLModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            n_targets=n_targets,
            config=config
        ).to(device)
        
        # Create target network
        self.target_q_network = copy.deepcopy(self.q_network).to(device)
        self.target_q_network.eval()
        
        # Freeze target network
        for param in self.target_q_network.parameters():
            param.requires_grad = False
        
        # Hidden states for RNN networks
        self.hidden_states = None
        self.target_hidden_states = None
        
        # Action mapping for discrete to continuous conversion
        self.action_mapping = self._create_action_mapping()
        
        # Coverage tracking for exploration bonus
        self.coverage_history = []
        self.episode_step = 0
    
    def _create_action_mapping(self) -> np.ndarray:
        """Create mapping from discrete actions to continuous camera actions."""
        # Simple mapping: discrete actions to continuous camera movements
        # Action 0: rotate left, Action 1: rotate right, Action 2: zoom in, Action 3: zoom out
        if self.action_dim == 2:
            # Continuous action space - no mapping needed
            return None
        else:
            # Discrete action space - create mapping
            mapping = np.array([
                [-1.0, 0.0],  # Action 0: rotate left
                [1.0, 0.0],   # Action 1: rotate right
                [0.0, -1.0],  # Action 2: zoom out
                [0.0, 1.0],   # Action 3: zoom in
            ])
            return mapping
    
    def select_action(self, obs: np.ndarray, state: np.ndarray, 
                     target_masks: Optional[np.ndarray] = None,
                     evaluate: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select actions using hierarchical target selection and Q-networks.
        
        Args:
            obs: Agent observations [n_agents, obs_dim]
            state: Global state [state_dim]
            target_masks: Target visibility masks [n_agents, n_targets]
            evaluate: Whether in evaluation mode
        
        Returns:
            actions: Selected actions [n_agents, action_dim]
            info: Additional information about action selection
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # [1, n_agents, obs_dim]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, state_dim]
        
        if target_masks is not None:
            target_masks_tensor = torch.FloatTensor(target_masks).unsqueeze(0).to(self.device)
        else:
            target_masks_tensor = None
        
        with torch.no_grad():
            # Forward pass through HRL model
            q_values, q_total, new_hidden_states, model_info = self.q_network(
                obs_tensor, state_tensor, target_masks_tensor, self.hidden_states
            )
            
            # Update hidden states
            self.hidden_states = new_hidden_states
            
            # Extract target selections
            target_selections = model_info['target_selections'].squeeze(0).cpu().numpy()  # [n_agents, n_targets]
            
            # Action selection with epsilon-greedy
            if evaluate or np.random.random() > self.epsilon:
                # Greedy action selection
                actions_discrete = torch.argmax(q_values.squeeze(0), dim=-1).cpu().numpy()  # [n_agents]
            else:
                # Random action selection
                actions_discrete = np.random.randint(0, self.action_dim, size=self.n_agents)
            
            # Convert discrete actions to continuous if needed
            if self.action_mapping is not None:
                actions_continuous = self.action_mapping[actions_discrete]  # [n_agents, 2]
            else:
                # Already continuous or handle differently
                actions_continuous = actions_discrete.reshape(self.n_agents, -1)
                if actions_continuous.shape[1] == 1:
                    # Expand to 2D if needed
                    actions_continuous = np.column_stack([actions_continuous, np.zeros(self.n_agents)])
            
            # Apply hierarchical target selection to actions
            # Modulate actions based on target selections
            enhanced_actions = self._apply_target_selection_to_actions(
                actions_continuous, target_selections, obs, target_masks
            )
            
            # Coverage-aware exploration bonus
            if not evaluate:
                enhanced_actions = self._apply_coverage_exploration(
                    enhanced_actions, target_selections, obs
                )
        
        # Update epsilon
        if not evaluate:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.episode_step += 1
        
        # Compile action info
        action_info = {
            'target_selections': target_selections,
            'q_values': q_values.squeeze(0).cpu().numpy(),
            'q_total': q_total.squeeze(0).cpu().numpy(),
            'epsilon': self.epsilon,
            'episode_step': self.episode_step,
            'model_info': model_info
        }
        
        return enhanced_actions, action_info
    
    def _apply_target_selection_to_actions(self, actions: np.ndarray, target_selections: np.ndarray,
                                         obs: np.ndarray, target_masks: Optional[np.ndarray]) -> np.ndarray:
        """
        Apply hierarchical target selection to modulate camera actions.
        
        This function implements the core HRL idea: use target selection to guide camera movements
        towards selected targets, similar to MATE HRL approach.
        """
        enhanced_actions = actions.copy()
        
        for agent_idx in range(self.n_agents):
            agent_selection = target_selections[agent_idx]  # [n_targets]
            selected_targets = np.where(agent_selection > 0.5)[0]
            
            if len(selected_targets) > 0:
                # Calculate centroid of selected targets (simplified)
                # In practice, this would use actual target positions from observation
                target_weight = np.mean(agent_selection[selected_targets])
                
                # Modulate action magnitude based on target selection confidence
                action_magnitude = np.linalg.norm(enhanced_actions[agent_idx])
                if action_magnitude > 0:
                    # Scale action by target selection confidence
                    enhanced_actions[agent_idx] *= (0.5 + 0.5 * target_weight)
                
                # Add small bias towards target-rich areas (simplified heuristic)
                if target_masks is not None:
                    visible_selected = agent_selection * target_masks[agent_idx]
                    if np.sum(visible_selected) > 0:
                        # Bias towards areas with more selected visible targets
                        bias_strength = 0.1 * np.sum(visible_selected) / len(selected_targets)
                        # Simple bias: if more targets on right, bias right
                        target_bias = np.array([bias_strength, 0.0])
                        enhanced_actions[agent_idx] += target_bias
        
        # Clip actions to valid range
        enhanced_actions = np.clip(enhanced_actions, -1.0, 1.0)
        
        return enhanced_actions
    
    def _apply_coverage_exploration(self, actions: np.ndarray, target_selections: np.ndarray,
                                  obs: np.ndarray) -> np.ndarray:
        """
        Apply coverage-aware exploration bonus to encourage better coverage.
        """
        if self.exploration_bonus <= 0:
            return actions
        
        enhanced_actions = actions.copy()
        
        # Calculate coverage score (simplified)
        total_coverage = np.sum(target_selections) / (self.n_agents * self.n_targets)
        self.coverage_history.append(total_coverage)
        
        # Keep only recent history
        if len(self.coverage_history) > 100:
            self.coverage_history = self.coverage_history[-100:]
        
        # If coverage is low, add exploration noise
        if len(self.coverage_history) > 10:
            recent_coverage = np.mean(self.coverage_history[-10:])
            if recent_coverage < 0.3:  # Low coverage threshold
                # Add exploration noise
                exploration_noise = np.random.normal(0, self.exploration_bonus, enhanced_actions.shape)
                enhanced_actions += exploration_noise
                enhanced_actions = np.clip(enhanced_actions, -1.0, 1.0)
        
        return enhanced_actions
    
    def update_target_network(self):
        """Update target network using soft update."""
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def reset_hidden_states(self):
        """Reset hidden states for new episode."""
        self.hidden_states = None
        self.target_hidden_states = None
        self.q_network.reset_hierarchical_states()
        self.target_q_network.reset_hierarchical_states()
        self.episode_step = 0
        self.coverage_history = []
    
    def get_target_selections(self, obs: np.ndarray, state: np.ndarray,
                            target_masks: Optional[np.ndarray] = None) -> np.ndarray:
        """Get target selections without action selection."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if target_masks is not None:
            target_masks_tensor = torch.FloatTensor(target_masks).unsqueeze(0).to(self.device)
        else:
            target_masks_tensor = None
        
        with torch.no_grad():
            target_selections = self.q_network.get_target_selections(obs_tensor, target_masks_tensor)
            return target_selections.squeeze(0).cpu().numpy()
    
    def compute_td_error(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                        next_obs: torch.Tensor, dones: torch.Tensor, state: torch.Tensor,
                        next_state: torch.Tensor, target_masks: Optional[torch.Tensor] = None,
                        next_target_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute TD error for training.
        
        Args:
            obs: Current observations [batch_size, n_agents, obs_dim]
            actions: Actions taken [batch_size, n_agents]
            rewards: Rewards received [batch_size, n_agents]
            next_obs: Next observations [batch_size, n_agents, obs_dim]
            dones: Done flags [batch_size]
            state: Current global state [batch_size, state_dim]
            next_state: Next global state [batch_size, state_dim]
            target_masks: Current target masks [batch_size, n_agents, n_targets]
            next_target_masks: Next target masks [batch_size, n_agents, n_targets]
        
        Returns:
            td_error: TD error for training [batch_size]
        """
        batch_size = obs.size(0)
        
        # Current Q-values
        q_values, q_total, _, _ = self.q_network(obs, state, target_masks, None)
        
        # Gather Q-values for taken actions
        actions_long = actions.long().unsqueeze(-1)  # [batch_size, n_agents, 1]
        current_q_values = q_values.gather(-1, actions_long).squeeze(-1)  # [batch_size, n_agents]
        
        # Target Q-values
        with torch.no_grad():
            if self.double_q:
                # Double Q-learning: use main network to select actions, target network to evaluate
                next_q_values, _, _, _ = self.q_network(next_obs, next_state, next_target_masks, None)
                next_actions = torch.argmax(next_q_values, dim=-1)  # [batch_size, n_agents]
                
                target_next_q_values, target_next_q_total, _, _ = self.target_q_network(
                    next_obs, next_state, next_target_masks, None
                )
                next_actions_long = next_actions.unsqueeze(-1)
                target_q_values = target_next_q_values.gather(-1, next_actions_long).squeeze(-1)
            else:
                # Standard Q-learning
                target_next_q_values, target_next_q_total, _, _ = self.target_q_network(
                    next_obs, next_state, next_target_masks, None
                )
                target_q_values = torch.max(target_next_q_values, dim=-1)[0]  # [batch_size, n_agents]
            
            # Compute target values
            target_values = rewards + self.gamma * target_q_values * (1 - dones.unsqueeze(-1))
        
        # TD error
        td_error = F.mse_loss(current_q_values, target_values, reduction='none')
        td_error = td_error.mean(dim=-1)  # Average over agents
        
        return td_error
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'epsilon': self.epsilon,
            'episode_step': self.episode_step,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.episode_step = checkpoint.get('episode_step', 0)
        
        print(f"Loaded QPLEX HRL agent from {filepath}")
        print(f"  Epsilon: {self.epsilon:.4f}")
        print(f"  Episode step: {self.episode_step}")
    
    def train(self):
        """Set agent to training mode."""
        self.q_network.train()
    
    def eval(self):
        """Set agent to evaluation mode."""
        self.q_network.eval()