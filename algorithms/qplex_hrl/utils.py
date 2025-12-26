"""Utility functions for QPLEX HRL - copied from MATE examples to avoid import issues."""

import copy
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import gym
import numpy as np
from gym import spaces

try:
    import ray
    from ray.rllib.env import MultiAgentEnv
    from ray.rllib.utils.typing import MultiAgentDict
    from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
    from ray.rllib.agents.callbacks import DefaultCallbacks
    from ray.rllib.env.base_env import BaseEnv
    from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
    from ray.rllib.policy import Policy
    from ray.rllib.policy.sample_batch import SampleBatch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    MultiAgentEnv = object
    MultiAgentDict = dict


__all__ = [
    'MetricCollector',
    'CustomMetricCallback', 
    'RLlibMultiAgentAPI',
    'RLlibMultiAgentCentralizedTraining',
    'SymlinkCheckpointCallback',
    'WandbLoggerCallback',
]


class MetricCollector:
    """Collect metrics from info dictionaries."""
    
    def __init__(self, info_keys):
        self.info_keys = info_keys
        self.metrics = defaultdict(list)
    
    def add(self, info):
        """Add info dictionary to metrics."""
        for key, value in info.items():
            if key in self.info_keys:
                self.metrics[key].append(value)
    
    def collect(self):
        """Collect and aggregate metrics."""
        result = {}
        for key, values in self.metrics.items():
            if not values:
                continue
                
            reduction = self.info_keys[key]
            if reduction == 'sum':
                result[key] = sum(values)
            elif reduction == 'mean':
                result[key] = np.mean(values)
            elif reduction == 'last':
                result[key] = values[-1]
            else:
                result[key] = values
                
        return result


if RAY_AVAILABLE:
    class CustomMetricCallback(DefaultCallbacks):
        """Custom callback for collecting metrics in Ray RLlib."""
        
        DEFAULT_CUSTOM_METRICS = {
            'raw_reward': 'sum',
            'normalized_raw_reward': 'sum',
            'coverage_rate': 'mean',
            'real_coverage_rate': 'mean',
            'mean_transport_rate': 'last',
            'num_delivered_cargoes': 'last',
            'num_tracked': 'mean',
        }

        def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            episode: MultiAgentEpisode,
            env_index: int,
            **kwargs,
        ):
            # Collect custom metrics from episode info
            for agent_id, info in episode.last_info_for().items():
                if info is None:
                    continue
                    
                for key, value in info.items():
                    if isinstance(value, (int, float, np.number)):
                        episode.custom_metrics[f"{key}_{agent_id}"] = value

        def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: MultiAgentEpisode,
            env_index: int,
            **kwargs,
        ):
            # Aggregate metrics across agents
            metric_sums = defaultdict(list)
            
            for key, value in episode.custom_metrics.items():
                if '_' in key:
                    metric_name = '_'.join(key.split('_')[:-1])
                    metric_sums[metric_name].append(value)
            
            # Calculate aggregated metrics
            for metric_name, values in metric_sums.items():
                if values:
                    episode.custom_metrics[f"{metric_name}_mean"] = np.mean(values)
                    episode.custom_metrics[f"{metric_name}_sum"] = np.sum(values)
                    episode.custom_metrics[f"{metric_name}_max"] = np.max(values)
                    episode.custom_metrics[f"{metric_name}_min"] = np.min(values)

else:
    class CustomMetricCallback:
        """Dummy callback when Ray is not available."""
        DEFAULT_CUSTOM_METRICS = {}


class RLlibMultiAgentAPI(gym.Wrapper):
    """Convert MATE environment to RLlib MultiAgent API."""
    
    def __init__(self, env):
        super().__init__(env)
        self.agent_ids = [f"agent_{i}" for i in range(self.num_cameras)]
        self._agent_ids = set(self.agent_ids)
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return {agent_id: obs[i] for i, agent_id in enumerate(self.agent_ids)}
    
    def step(self, action_dict):
        actions = [action_dict[agent_id] for agent_id in self.agent_ids]
        obs, rewards, dones, infos = self.env.step(actions)
        
        obs_dict = {agent_id: obs[i] for i, agent_id in enumerate(self.agent_ids)}
        reward_dict = {agent_id: rewards[i] for i, agent_id in enumerate(self.agent_ids)}
        done_dict = {agent_id: dones[i] for i, agent_id in enumerate(self.agent_ids)}
        done_dict["__all__"] = all(dones)
        info_dict = {agent_id: infos[i] for i, agent_id in enumerate(self.agent_ids)}
        
        return obs_dict, reward_dict, done_dict, info_dict


class RLlibMultiAgentCentralizedTraining(gym.Wrapper):
    """Wrapper for centralized training with RLlib."""
    
    def __init__(self, env):
        super().__init__(env)
        
    def with_agent_groups(self, groups, obs_space, act_space):
        """Group agents for centralized training."""
        self.agent_groups = groups
        self.grouped_obs_space = obs_space
        self.grouped_act_space = act_space
        return self


class SymlinkCheckpointCallback:
    """Callback to create symlinks to best checkpoints."""
    
    def __init__(self):
        pass
        
    def __call__(self, trial):
        """Create symlink to best checkpoint."""
        pass


class WandbLoggerCallback:
    """Weights & Biases logger callback."""
    
    def __init__(self, project=None, group=None, **kwargs):
        self.project = project
        self.group = group
        self.kwargs = kwargs
        
    @staticmethod
    def is_available():
        """Check if wandb is available."""
        try:
            import wandb
            return True
        except ImportError:
            return False
            
    def __call__(self, trial):
        """Log to wandb."""
        if self.is_available():
            import wandb
            wandb.init(project=self.project, group=self.group, **self.kwargs)