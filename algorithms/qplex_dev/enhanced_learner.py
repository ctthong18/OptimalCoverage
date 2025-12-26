import os
from typing import Dict, Any

import numpy as np
import torch

from .learner import ReplayBuffer
from .enhanced_agent import EnhancedQPLEXAgent
from training_dev_qplex.curriculum import CurriculumExploration
from training_dev_qplex.advanced_training import AdvancedTraining
from training_dev_qplex.reward_shaping import OptimizedTensorReward


class EnhancedQPLEXLearner:
    """Enhanced QPLEX Learner sử dụng Enhanced Agent và training utilities."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        # Training parameters
        tr = config['training']
        self.total_timesteps = tr['total_timesteps']
        self.learning_starts = tr['learning_starts']
        self.train_freq = tr['train_freq']
        self.target_update_interval = tr['target_update_interval']
        self.gradient_steps = tr['gradient_steps']
        self.batch_size = tr['batch_size']
        self.buffer_size = tr['buffer_size']
        
        # State
        self.agent = None
        self.buffer = None
        self.timestep = 0
        self.episode_count = 0
        self.last_log_time = 0
        self.last_save_time = 0
        
        # Utilities
        self.curriculum = CurriculumExploration(config) if config.get('use_curriculum', True) else None
        self.adv_train = AdvancedTraining(self, config)
        self.reward_shaper = OptimizedTensorReward() if tr.get('use_reward_shaping', False) else None
        
        # Stats
        from collections import deque
        self.training_stats: Dict[str, any] = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'losses': deque(maxlen=1000),
            'q_values': deque(maxlen=1000),
            'td_errors': deque(maxlen=1000),
            'epsilon': deque(maxlen=1000)
        }
    
    def setup(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int):
        self.agent = EnhancedQPLEXAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            config=self.config,
            device=self.device
        )
        self.buffer = ReplayBuffer(
            capacity=self.buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents
        )
    
    def _maybe_shape_rewards(self, env, rewards: np.ndarray, info=None) -> np.ndarray:
        """Shape rewards with focus on mean_coverage_rate optimization."""
        if self.reward_shaper is None:
            return rewards
        
        n_agents = len(rewards)
        
        # Extract coverage_state from environment and info
        coverage_state = {
            'n_tensors': n_agents,
            'coverage_scores': np.ones(n_agents) * 0.5,  # Default
            'target_coverage_rate': 0.0,  # PRIMARY METRIC - will be updated
            'target_distances': np.ones(n_agents) * 5.0,
            'rotation_costs': np.zeros(n_agents),
            'obstacle_distances': np.ones(n_agents) * 10.0,
            'target_velocity': np.zeros(2),
            'reliable_radius': 1.0,
            'max_radius': 5.0,
            'obstacle_violations': 0
        }
        
        # Try to get coverage_rate from environment (PRIMARY METRIC)
        if hasattr(env, 'coverage_rate'):
            coverage_state['target_coverage_rate'] = env.coverage_rate
        
        # Try to get from info if available
        if info is not None and len(info) > 0:
            camera_infos, target_infos = info
            if camera_infos and len(camera_infos) > 0:
                # PRIMARY: Get mean_coverage_rate from info
                coverage_rate = camera_infos[0].get('coverage_rate', 0.0)
                if coverage_rate > 0:
                    coverage_state['target_coverage_rate'] = coverage_rate
                
                # Try to extract other metrics if available
                # (These might not be directly available, but we try)
        
        # If we have a coverage_state attribute, use it
        if hasattr(env, 'coverage_state') and isinstance(env.coverage_state, dict):
            coverage_state.update(env.coverage_state)
        
        shaping = self.reward_shaper.compute(coverage_state)
        return rewards + shaping
    
    def learn(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
              next_obs: np.ndarray, done: bool, state: np.ndarray, next_state: np.ndarray,
              info=None, env=None):
        # Optional reward shaping with mean_coverage optimization
        if env is not None:
            rewards = self._maybe_shape_rewards(env, rewards, info)
        
        self.buffer.add(obs, actions, rewards, next_obs, done, state, next_state)
        self.timestep += 1
        
        learning_info: Dict[str, float] = {}
        if (self.timestep >= self.learning_starts and 
            self.timestep % self.train_freq == 0 and 
            self.buffer.size >= self.batch_size):
            for _ in range(self.gradient_steps):
                batch = self.buffer.sample(self.batch_size)
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                train_info = self.agent.train(batch)
                learning_info.update(train_info)
                self.training_stats['losses'].append(train_info['loss'])
                self.training_stats['q_values'].append(train_info['q_values'])
                self.training_stats['td_errors'].append(train_info['td_error'])
                self.training_stats['epsilon'].append(train_info['epsilon'])
        if done:
            self.episode_count += 1
            if info is not None:
                ep_rew = info.get('episode_reward', float(np.sum(rewards)))
                ep_len = info.get('episode_length', self.timestep - self.last_log_time)
                self.training_stats['episode_rewards'].append(ep_rew)
                self.training_stats['episode_lengths'].append(ep_len)
                self.last_log_time = self.timestep
        return learning_info
    
    def select_action(self, obs: np.ndarray, state: np.ndarray, evaluate: bool = False):
        return self.agent.select_action(obs, state, evaluate)
    
    def reset_hidden_states(self):
        if self.agent is not None:
            self.agent.reset_hidden_states()
    
    def get_training_stats(self) -> Dict[str, float]:
        stats = {}
        import numpy as _np
        if len(self.training_stats['episode_rewards']) > 0:
            stats['mean_episode_reward'] = _np.mean(self.training_stats['episode_rewards'])
            stats['std_episode_reward'] = _np.std(self.training_stats['episode_rewards'])
            stats['mean_episode_length'] = _np.mean(self.training_stats['episode_lengths'])
        if len(self.training_stats['losses']) > 0:
            stats['mean_loss'] = _np.mean(self.training_stats['losses'])
            stats['mean_q_values'] = _np.mean(self.training_stats['q_values'])
            stats['mean_td_error'] = _np.mean(self.training_stats['td_errors'])
            stats['mean_epsilon'] = _np.mean(self.training_stats['epsilon'])
        stats['timestep'] = self.timestep
        stats['episode_count'] = self.episode_count
        stats['buffer_size'] = self.buffer.size if self.buffer else 0
        return stats
    
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        agent_path = filepath.replace('.pth', '_agent.pth')
        self.agent.save(agent_path)
        buffer_path = filepath.replace('.pth', '_buffer.pkl')
        self.buffer.save(buffer_path)
        learner_state = {
            'timestep': self.timestep,
            'episode_count': self.episode_count,
            'training_stats': dict(self.training_stats),
            'config': self.config
        }
        torch.save(learner_state, filepath)
    
    def load(self, filepath: str):
        learner_state = torch.load(filepath, map_location=self.device)
        self.timestep = learner_state['timestep']
        self.episode_count = learner_state['episode_count']
        self.training_stats = learner_state['training_stats']
        agent_path = filepath.replace('.pth', '_agent.pth')
        self.agent.load(agent_path)
        buffer_path = filepath.replace('.pth', '_buffer.pkl')
        if os.path.exists(buffer_path):
            self.buffer.load(buffer_path)

