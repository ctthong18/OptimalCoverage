"""QPLEX HRL Learner - Training orchestrator for hierarchical QPLEX."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import copy
from collections import deque
import random

from .agent import QPLEXHRLAgent
from algorithms.qplex_dev.learner import ReplayBuffer  # Reuse buffer from qplex_dev


class HRLReplayBuffer(ReplayBuffer):
    """
    Enhanced replay buffer for HRL with target selection information.
    Extends the qplex_dev ReplayBuffer to store hierarchical information.
    """
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, state_dim: int, 
                 n_agents: int, n_targets: int, device: torch.device):
        super().__init__(capacity, obs_dim, action_dim, state_dim, n_agents, device)
        
        self.n_targets = n_targets
        
        # Additional storage for HRL information
        self.target_selections = np.zeros((capacity, n_agents, n_targets), dtype=np.float32)
        self.target_masks = np.zeros((capacity, n_agents, n_targets), dtype=np.float32)
        self.next_target_masks = np.zeros((capacity, n_agents, n_targets), dtype=np.float32)
        self.coverage_rates = np.zeros(capacity, dtype=np.float32)
        self.selection_info = [None] * capacity
    
    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            next_obs: np.ndarray, done: bool, state: np.ndarray, next_state: np.ndarray,
            target_selections: Optional[np.ndarray] = None,
            target_masks: Optional[np.ndarray] = None,
            next_target_masks: Optional[np.ndarray] = None,
            coverage_rate: float = 0.0,
            selection_info: Optional[Dict] = None):
        """Add experience with HRL information."""
        
        # Call parent add method
        super().add(obs, actions, rewards, next_obs, done, state, next_state)
        
        # Store HRL-specific information at the same index
        idx = (self.position - 1) % self.capacity
        
        if target_selections is not None:
            self.target_selections[idx] = target_selections
        if target_masks is not None:
            self.target_masks[idx] = target_masks
        if next_target_masks is not None:
            self.next_target_masks[idx] = next_target_masks
        
        self.coverage_rates[idx] = coverage_rate
        self.selection_info[idx] = selection_info
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch with HRL information."""
        
        # Get base sample
        batch = super().sample(batch_size)
        
        # Add HRL information
        indices = batch['indices'] if 'indices' in batch else np.random.choice(self.size, batch_size, replace=False)
        
        batch.update({
            'target_selections': torch.FloatTensor(self.target_selections[indices]).to(self.device),
            'target_masks': torch.FloatTensor(self.target_masks[indices]).to(self.device),
            'next_target_masks': torch.FloatTensor(self.next_target_masks[indices]).to(self.device),
            'coverage_rates': torch.FloatTensor(self.coverage_rates[indices]).to(self.device)
        })
        
        return batch


class QPLEXHRLLearner:
    """
    QPLEX HRL Learner combining hierarchical target selection with advanced Q-learning.
    
    This learner integrates:
    1. Hierarchical target selection training
    2. Coverage-aware reward shaping
    3. Advanced optimization techniques from qplex_dev
    4. Multi-timescale learning for HRL
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        # Training configuration
        training_config = config.get('training', {})
        self.batch_size = training_config.get('batch_size', 64)
        self.learning_rate = config.get('algorithm', {}).get('learning_rate', 0.0005)
        self.target_update_interval = training_config.get('target_update_interval', 500)
        self.gradient_steps = training_config.get('gradient_steps', 1)
        self.train_freq = training_config.get('train_freq', 1)
        self.learning_starts = training_config.get('learning_starts', 1000)
        
        # HRL specific configuration
        hrl_config = config.get('network', {}).get('hrl', {})
        self.coverage_reward_weight = hrl_config.get('coverage_reward_weight', 1.0)
        self.selection_entropy_weight = hrl_config.get('selection_entropy_weight', 0.01)
        self.hierarchical_loss_weight = hrl_config.get('hierarchical_loss_weight', 0.1)
        
        # Initialize components (will be set in setup)
        self.agent = None
        self.buffer = None
        self.optimizer = None
        self.scheduler = None
        
        # Training statistics
        self.timestep = 0
        self.episode_count = 0
        self.training_stats = {
            'mean_loss': 0.0,
            'mean_q_values': 0.0,
            'mean_td_error': 0.0,
            'mean_coverage_rate': 0.0,
            'mean_selection_entropy': 0.0,
            'mean_episode_reward': 0.0,
            'buffer_size': 0
        }
        
        # Coverage tracking for reward shaping
        self.coverage_history = deque(maxlen=1000)
        self.best_coverage = 0.0
    
    def setup(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int, n_targets: int):
        """Setup learner components."""
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_targets = n_targets
        
        # Create agent
        self.agent = QPLEXHRLAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            n_targets=n_targets,
            config=self.config,
            device=self.device
        )
        
        # Create enhanced replay buffer
        buffer_size = self.config.get('training', {}).get('buffer_size', 200000)
        self.buffer = HRLReplayBuffer(
            capacity=buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            n_targets=n_targets,
            device=self.device
        )
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.agent.q_network.parameters(),
            lr=self.learning_rate,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10000,
            gamma=0.9
        )
        
        print(f"QPLEX HRL Learner setup complete:")
        print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}, State dim: {state_dim}")
        print(f"  Agents: {n_agents}, Targets: {n_targets}")
        print(f"  Buffer size: {buffer_size}, Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
    
    def select_action(self, obs: np.ndarray, state: np.ndarray,
                     target_masks: Optional[np.ndarray] = None,
                     evaluate: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using HRL agent."""
        return self.agent.select_action(obs, state, target_masks, evaluate)
    
    def learn(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
              next_obs: np.ndarray, done: bool, state: np.ndarray, next_state: np.ndarray,
              info: Optional[Dict] = None, env: Optional[Any] = None) -> Dict[str, Any]:
        """
        Learn from experience with HRL enhancements.
        
        Args:
            obs: Current observations
            actions: Actions taken
            rewards: Rewards received
            next_obs: Next observations
            done: Episode done flag
            state: Current global state
            next_state: Next global state
            info: Additional information from environment
            env: Environment instance for extracting coverage information
        
        Returns:
            learning_info: Information about learning step
        """
        # Extract HRL information
        target_masks = None
        next_target_masks = None
        coverage_rate = 0.0
        
        # Extract target masks and coverage from environment or info
        if env is not None:
            # Try to extract target visibility masks
            if hasattr(env, 'get_target_masks'):
                target_masks = env.get_target_masks()
                next_target_masks = target_masks  # Assume same for next step
            
            # Extract coverage rate
            if hasattr(env, 'coverage_rate'):
                coverage_rate = env.coverage_rate
            elif hasattr(env, 'get_coverage_rate'):
                coverage_rate = env.get_coverage_rate()
        
        # Extract from info if available
        if info is not None:
            if 'target_masks' in info:
                target_masks = info['target_masks']
            if 'next_target_masks' in info:
                next_target_masks = info['next_target_masks']
            if 'coverage_rate' in info:
                coverage_rate = info['coverage_rate']
            
            # Extract from env_info if nested
            if 'env_info' in info and info['env_info'] is not None:
                env_info = info['env_info']
                if isinstance(env_info, (list, tuple)) and len(env_info) > 0:
                    camera_infos, _ = env_info
                    if camera_infos and len(camera_infos) > 0:
                        coverage_rate = camera_infos[0].get('coverage_rate', coverage_rate)
        
        # Get target selections from agent
        target_selections = None
        if target_masks is not None:
            target_selections = self.agent.get_target_selections(obs, state, target_masks)
        
        # Coverage-aware reward shaping
        shaped_rewards = self._apply_coverage_reward_shaping(rewards, coverage_rate)
        
        # Store experience in buffer
        self.buffer.add(
            obs=obs,
            actions=actions,
            rewards=shaped_rewards,
            next_obs=next_obs,
            done=done,
            state=state,
            next_state=next_state,
            target_selections=target_selections,
            target_masks=target_masks,
            next_target_masks=next_target_masks,
            coverage_rate=coverage_rate
        )
        
        # Update timestep
        self.timestep += 1
        
        # Update coverage tracking
        self.coverage_history.append(coverage_rate)
        if coverage_rate > self.best_coverage:
            self.best_coverage = coverage_rate
        
        learning_info = {'timestep': self.timestep}
        
        # Training step
        if (self.timestep >= self.learning_starts and 
            self.timestep % self.train_freq == 0 and 
            self.buffer.size >= self.batch_size):
            
            # Perform gradient steps
            total_loss = 0.0
            total_td_error = 0.0
            total_q_values = 0.0
            total_selection_entropy = 0.0
            
            for _ in range(self.gradient_steps):
                loss_info = self._training_step()
                total_loss += loss_info['loss']
                total_td_error += loss_info['td_error']
                total_q_values += loss_info['q_values']
                total_selection_entropy += loss_info.get('selection_entropy', 0.0)
            
            # Average over gradient steps
            avg_loss = total_loss / self.gradient_steps
            avg_td_error = total_td_error / self.gradient_steps
            avg_q_values = total_q_values / self.gradient_steps
            avg_selection_entropy = total_selection_entropy / self.gradient_steps
            
            # Update training statistics
            self.training_stats.update({
                'mean_loss': avg_loss,
                'mean_td_error': avg_td_error,
                'mean_q_values': avg_q_values,
                'mean_selection_entropy': avg_selection_entropy,
                'mean_coverage_rate': np.mean(list(self.coverage_history)) if self.coverage_history else 0.0,
                'buffer_size': self.buffer.size
            })
            
            learning_info.update({
                'loss': avg_loss,
                'td_error': avg_td_error,
                'q_values': avg_q_values,
                'selection_entropy': avg_selection_entropy,
                'coverage_rate': coverage_rate
            })
        
        # Update target network
        if self.timestep % self.target_update_interval == 0:
            self.agent.update_target_network()
            learning_info['target_updated'] = True
        
        return learning_info
    
    def _apply_coverage_reward_shaping(self, rewards: np.ndarray, coverage_rate: float) -> np.ndarray:
        """Apply coverage-aware reward shaping."""
        if self.coverage_reward_weight <= 0:
            return rewards
        
        shaped_rewards = rewards.copy()
        
        # Coverage improvement bonus
        if len(self.coverage_history) > 0:
            recent_coverage = np.mean(list(self.coverage_history)[-10:])
            if coverage_rate > recent_coverage:
                coverage_bonus = self.coverage_reward_weight * (coverage_rate - recent_coverage)
                shaped_rewards += coverage_bonus
        
        # Best coverage bonus
        if coverage_rate > self.best_coverage * 0.95:  # Within 5% of best
            shaped_rewards += self.coverage_reward_weight * 0.1
        
        return shaped_rewards
    
    def _training_step(self) -> Dict[str, float]:
        """Perform one training step."""
        # Sample batch
        batch = self.buffer.sample(self.batch_size)
        
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        state = batch['state']
        next_state = batch['next_state']
        target_masks = batch.get('target_masks')
        next_target_masks = batch.get('next_target_masks')
        
        # Compute TD error
        td_error = self.agent.compute_td_error(
            obs, actions, rewards, next_obs, dones, state, next_state,
            target_masks, next_target_masks
        )
        
        # Main Q-learning loss
        q_loss = td_error.mean()
        
        # Hierarchical selection loss (encourage diverse target selection)
        selection_loss = 0.0
        selection_entropy = 0.0
        
        if target_masks is not None and self.selection_entropy_weight > 0:
            # Get current target selections
            with torch.no_grad():
                current_selections = self.agent.q_network.get_target_selections(obs, target_masks)
            
            # Encourage entropy in target selection (diversity)
            selection_probs = torch.sigmoid(current_selections)
            selection_entropy = -torch.sum(
                selection_probs * torch.log(selection_probs + 1e-8) +
                (1 - selection_probs) * torch.log(1 - selection_probs + 1e-8),
                dim=-1
            ).mean()
            
            # Selection loss encourages entropy
            selection_loss = -self.selection_entropy_weight * selection_entropy
        
        # Total loss
        total_loss = q_loss + selection_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute additional statistics
        with torch.no_grad():
            q_values, _, _, _ = self.agent.q_network(obs, state, target_masks, None)
            mean_q_values = q_values.mean().item()
        
        return {
            'loss': total_loss.item(),
            'q_loss': q_loss.item(),
            'selection_loss': selection_loss.item() if isinstance(selection_loss, torch.Tensor) else selection_loss,
            'td_error': td_error.mean().item(),
            'q_values': mean_q_values,
            'selection_entropy': selection_entropy.item() if isinstance(selection_entropy, torch.Tensor) else selection_entropy
        }
    
    def reset_hidden_states(self):
        """Reset hidden states for new episode."""
        if self.agent is not None:
            self.agent.reset_hidden_states()
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics."""
        stats = self.training_stats.copy()
        stats.update({
            'timestep': self.timestep,
            'episode_count': self.episode_count,
            'epsilon': self.agent.epsilon if self.agent else 0.0,
            'best_coverage': self.best_coverage,
            'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        })
        return stats
    
    def save(self, filepath: str):
        """Save learner state."""
        if self.agent is None:
            raise ValueError("Agent not initialized. Call setup() first.")
        
        # Save agent
        agent_filepath = filepath.replace('.pth', '_agent.pth')
        self.agent.save(agent_filepath)
        
        # Save buffer
        buffer_filepath = filepath.replace('.pth', '_buffer.pkl')
        self.buffer.save(buffer_filepath)
        
        # Save learner state
        learner_state = {
            'timestep': self.timestep,
            'episode_count': self.episode_count,
            'training_stats': self.training_stats,
            'best_coverage': self.best_coverage,
            'coverage_history': list(self.coverage_history),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config
        }
        
        torch.save(learner_state, filepath)
        
        print(f"Saved QPLEX HRL learner to {filepath}")
        print(f"  Agent saved to: {agent_filepath}")
        print(f"  Buffer saved to: {buffer_filepath}")
    
    def load(self, filepath: str):
        """Load learner state."""
        # Load learner state
        learner_state = torch.load(filepath, map_location=self.device)
        
        self.timestep = learner_state.get('timestep', 0)
        self.episode_count = learner_state.get('episode_count', 0)
        self.training_stats = learner_state.get('training_stats', self.training_stats)
        self.best_coverage = learner_state.get('best_coverage', 0.0)
        
        coverage_history = learner_state.get('coverage_history', [])
        self.coverage_history = deque(coverage_history, maxlen=1000)
        
        # Load optimizer and scheduler states
        if self.optimizer and 'optimizer_state_dict' in learner_state:
            self.optimizer.load_state_dict(learner_state['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in learner_state:
            self.scheduler.load_state_dict(learner_state['scheduler_state_dict'])
        
        # Load agent
        agent_filepath = filepath.replace('.pth', '_agent.pth')
        if self.agent:
            self.agent.load(agent_filepath)
        
        # Load buffer
        buffer_filepath = filepath.replace('.pth', '_buffer.pkl')
        if self.buffer:
            self.buffer.load(buffer_filepath)
        
        print(f"Loaded QPLEX HRL learner from {filepath}")
        print(f"  Timestep: {self.timestep}, Episode: {self.episode_count}")
        print(f"  Best coverage: {self.best_coverage:.4f}")
        print(f"  Buffer size: {self.buffer.size if self.buffer else 0}")
    
    def train(self):
        """Set learner to training mode."""
        if self.agent:
            self.agent.train()
    
    def eval(self):
        """Set learner to evaluation mode."""
        if self.agent:
            self.agent.eval()