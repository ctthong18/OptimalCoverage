#!/usr/bin/env python3

"""Training script for QPLEX HRL using Ray RLlib (exactly like MATE HRL examples)."""

# Run: python -m runners.train_qplex_hrl_ray

import argparse
import copy
import os
import sys
from math import ceil
from pathlib import Path
import yaml
import torch
import numpy as np
import random

# Prefer the installed Ray; the bundled mate/ray wheels lack _raylet on Windows.
from mate import ray
from mate.ray import tune
from mate.ray.rllib.agents.trainer_template import build_trainer

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MATE environment and HRL wrappers
import mate
from algorithms.qplex_hrl.wrappers import HierarchicalCamera, DiscreteMultiSelection
from algorithms.qplex_hrl.utils import (
    CustomMetricCallback, 
    RLlibMultiAgentAPI, 
    RLlibMultiAgentCentralizedTraining,
    SymlinkCheckpointCallback,
    WandbLoggerCallback
)

# Import QPLEX HRL RLlib components
from algorithms.qplex_hrl.rllib_policy import QPLEXHRLTorchPolicy


DEBUG = getattr(sys, 'gettrace', lambda: None)() is not None

HERE = Path(__file__).absolute().parent.parent
LOCAL_DIR = HERE / 'logs' / 'qplex_hrl_ray' / 'ray_results'
if DEBUG:
    print(f'DEBUG MODE: {DEBUG}')
    LOCAL_DIR = LOCAL_DIR / 'debug'


# Node resources (exactly like MATE HRL)
SLURM_CPUS_ON_NODE = int(os.getenv('SLURM_CPUS_ON_NODE', str(os.cpu_count() or 4)))
NUM_NODE_CPUS = max(1, min(os.cpu_count() or 4, SLURM_CPUS_ON_NODE))
assert NUM_NODE_CPUS >= 2
NUM_NODE_GPUS = torch.cuda.device_count()

# Training resources (exactly like MATE HRL)
PRESERVED_NUM_CPUS = 1  # for raylet
NUM_CPUS_FOR_TRAINER = 1
NUM_GPUS_FOR_TRAINER = min(NUM_NODE_GPUS, 0.25)  # can be overridden by command line arguments

MAX_NUM_CPUS_FOR_WORKER = max(0, NUM_NODE_CPUS - PRESERVED_NUM_CPUS - NUM_CPUS_FOR_TRAINER)
MAX_NUM_WORKERS = min(32, MAX_NUM_CPUS_FOR_WORKER)  # use at most 32 workers
NUM_WORKERS = MAX_NUM_WORKERS if not DEBUG else 0  # can be overridden by command line arguments


def target_agent_factory():
    """Create target agent factory (exactly like MATE HRL)."""
    return mate.agents.GreedyTargetAgent(seed=0)


def make_hrl_env(env_config):
    """
    Create HRL environment (exactly like MATE HRL examples).
    """
    env_config = env_config or {}
    env_id = env_config.get('env_id', 'MultiAgentTracking-v0')
    
    # Create base MATE environment
    base_env = mate.make(
        env_id, 
        config=env_config.get('config'), 
        **env_config.get('config_overrides', {})
    )
    
    # Enhanced observation (if specified)
    if str(env_config.get('enhanced_observation', None)).lower() != 'none':
        base_env = mate.EnhancedObservation(base_env, team=env_config['enhanced_observation'])
    
    # Create target agent
    target_agent = env_config.get('opponent_agent_factory', target_agent_factory())()
    env = mate.MultiCamera(base_env, target_agent=target_agent)
    
    # Apply coordinate transformations (exactly like MATE HRL)
    env = mate.RelativeCoordinates(env)
    env = mate.RescaledObservation(env)
    env = mate.RepeatedRewardIndividualDone(env)
    
    # Apply auxiliary rewards for coverage optimization
    if 'reward_coefficients' in env_config:
        env = mate.AuxiliaryCameraRewards(
            env,
            coefficients=env_config['reward_coefficients'],
            reduction=env_config.get('reward_reduction', 'none'),
        )
    
    # Apply hierarchical camera wrapper (key component from MATE HRL)
    multi_selection = env_config.get('multi_selection', False)
    frame_skip = env_config.get('frame_skip', 1)
    
    env = HierarchicalCamera(
        env, 
        multi_selection=multi_selection, 
        frame_skip=frame_skip
    )
    
    # Convert multi-discrete to discrete if needed
    if multi_selection:
        env = DiscreteMultiSelection(env)
    
    # Apply RLlib wrappers (exactly like MATE HRL)
    env = RLlibMultiAgentAPI(env)
    env = RLlibMultiAgentCentralizedTraining(env)
    
    # Set up action and observation spaces for grouped agents
    from gym import spaces
    action_space = spaces.Tuple((env.action_space,) * len(env.agent_ids))
    observation_space = spaces.Tuple((env.observation_space,) * len(env.agent_ids))
    setattr(observation_space, 'original_space', copy.deepcopy(observation_space))
    
    # Group agents (exactly like MATE HRL)
    env = env.with_agent_groups(
        groups={'camera': env.agent_ids}, 
        obs_space=observation_space, 
        act_space=action_space
    )
    
    return env


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Register environment (exactly like MATE HRL)
tune.register_env('mate-hrl.qplex.camera', make_hrl_env)

# Create QPLEX HRL Trainer (exactly like MATE HRL)
QPLEXHRLTrainer = build_trainer(
    name="QPLEX_HRL",
    default_policy=QPLEXHRLTorchPolicy,
)

# Create config based on YAML (exactly like MATE HRL structure)
def create_rllib_config(base_config: dict) -> dict:
    """Create RLlib config from YAML config (exactly like MATE HRL)."""
    
    config = {
        # Framework (exactly like MATE HRL)
        'framework': 'torch',
        'seed': base_config.get('seed', 0),
        
        # Environment (exactly like MATE HRL)
        'env': 'mate-hrl.qplex.camera',
        'env_config': {
            'env_id': 'MultiAgentTracking-v0',
            'config': base_config['env']['config_file'],
            'config_overrides': {'reward_type': base_config['env'].get('reward_type', 'dense')},
            'reward_coefficients': base_config.get('network', {}).get('hrl', {}).get('reward_coefficients', {'coverage_rate': 1.0}),
            'reward_reduction': base_config.get('network', {}).get('hrl', {}).get('reward_reduction', 'mean'),
            'multi_selection': base_config.get('network', {}).get('hrl', {}).get('multi_selection', True),
            'frame_skip': base_config.get('network', {}).get('hrl', {}).get('frame_skip', 5),
            'enhanced_observation': 'none',
            'opponent_agent_factory': target_agent_factory,
            'state_dim': 156,  # MATE 4v4-9 state dimension
            'n_targets': 4,    # Number of targets
        },
        'disable_env_checking': True,
        'horizon': base_config['env'].get('max_episode_steps', 500),
        'callbacks': CustomMetricCallback,
        
        # Model (exactly like MATE HRL)
        'normalize_actions': True,
        'model': {
            'fcnet_hiddens': [512, 256],  # not used
            'fcnet_activation': 'tanh',
            'lstm_cell_size': base_config.get('network', {}).get('q_network', {}).get('rnn_hidden_dim', 256),
            'max_seq_len': 10000,  # for complete episode
            'custom_model_config': base_config.get('network', {}),
        },
        
        # QPLEX specific (exactly like MATE HRL)
        'mixer': 'qplex',
        'mixing_embed_dim': 128,
        'selection_entropy_weight': base_config.get('network', {}).get('hrl', {}).get('selection_entropy_weight', 0.01),
        
        # Policy (exactly like MATE HRL)
        'gamma': base_config.get('algorithm', {}).get('gamma', 0.99),
        
        # Exploration (exactly like MATE HRL)
        'explore': True,
        'exploration_config': {
            'type': 'EpsilonGreedy',
            'initial_epsilon': base_config.get('algorithm', {}).get('epsilon_start', 1.0),
            'final_epsilon': base_config.get('algorithm', {}).get('epsilon_end', 0.02),
            'epsilon_timesteps': 50000,  # trained environment steps
        },
        
        # Replay Buffer & Optimization (exactly like MATE HRL)
        'batch_mode': 'complete_episodes',
        'rollout_fragment_length': 0,  # send sampled episodes to the replay buffer immediately
        'buffer_size': 2000,  # each item contains `num_workers` episodes (will be updated in train.py)
        'timesteps_per_iteration': base_config.get('training', {}).get('train_freq', 5120),  # environment steps
        'learning_starts': base_config.get('training', {}).get('learning_starts', 5000),  # environment steps
        'train_batch_size': base_config.get('training', {}).get('batch_size', 1024),  # environment steps
        'target_network_update_freq': base_config.get('training', {}).get('target_update_interval', 500),  # environment steps
        'metrics_num_episodes_for_smoothing': 25,
        'grad_norm_clipping': 1000.0,
        'lr': base_config.get('algorithm', {}).get('learning_rate', 1e-4),
    }
    
    return config


# Create experiment (exactly like MATE HRL)
def create_experiment(config_path: str) -> tune.Experiment:
    """Create experiment from config (exactly like MATE HRL)."""
    base_config = load_config(config_path)
    rllib_config = create_rllib_config(base_config)
    
    experiment = tune.Experiment(
        name='HRL-QPLEX',
        run=QPLEXHRLTrainer,
        config=copy.deepcopy(rllib_config),
        local_dir=LOCAL_DIR,
        stop={'timesteps_total': base_config.get('training', {}).get('total_timesteps', 10e6)},
        checkpoint_score_attr='episode_reward_mean',
        checkpoint_freq=20,
        checkpoint_at_end=True,
        max_failures=-1,
    )
    
    return experiment


def train(
    experiment,
    project=None,
    group=None,
    local_dir=None,
    num_gpus=NUM_GPUS_FOR_TRAINER,
    num_workers=NUM_WORKERS,
    num_envs_per_worker=8,
    seed=None,
    timesteps_total=None,
    buffer_capacity=2000,
):
    """Train function (exactly like MATE HRL)."""
    tune_callbacks = [SymlinkCheckpointCallback()]
    if WandbLoggerCallback.is_available():
        project = project or 'qplex-hrl-mate'
        group = group or f'hrl.qplex.camera.{experiment.name}'
        tune_callbacks.append(WandbLoggerCallback(project=project, group=group))

    if not ray.is_initialized():
        ray.init(num_cpus=NUM_NODE_CPUS, local_mode=DEBUG)

    num_ray_cpus = round(ray.cluster_resources()['CPU'])
    num_ray_gpus = ray.cluster_resources().get('GPU', 0.0)
    num_gpus = min(num_gpus, num_ray_gpus)
    num_workers = max(0, min(num_workers, num_ray_cpus - NUM_CPUS_FOR_TRAINER))

    experiment.spec['config'].update(
        num_cpus_for_driver=NUM_CPUS_FOR_TRAINER,
        num_gpus=num_gpus,
        num_gpus_per_worker=0,
        num_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
    )
    if seed is not None:
        seed = tune.grid_search(seed) if isinstance(seed, (list, tuple)) else seed
        experiment.spec['config'].update(seed=seed)
    if timesteps_total is not None:
        experiment.spec['stop'].update(timesteps_total=timesteps_total)
    if local_dir is not None:
        experiment.spec['local_dir'] = local_dir

    # Update replay buffer capacity (exactly like MATE HRL)
    experiment.spec['config'].update(buffer_size=ceil(buffer_capacity / max(num_workers, 1)))

    print(f"Starting QPLEX HRL training...")
    print(f"Workers: {num_workers}, GPUs: {num_gpus}")
    print(f"Total timesteps: {experiment.spec['stop']['timesteps_total']}")
    print(f"[Optimizing for: coverage_rate as PRIMARY METRIC]")

    analysis = tune.run(
        experiment, metric='episode_reward_mean', mode='max', callbacks=tune_callbacks, verbose=3
    )
    return analysis


def main():
    """Main function (exactly like MATE HRL)."""
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument(
        '--config', type=str, default='configs/qplex_hrl_4v4_9_ray.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--project', type=str, metavar='PROJECT', default=None, help='W&B project name'
    )
    parser.add_argument('--group', type=str, metavar='GROUP', default=None, help='W&B group name')
    parser.add_argument(
        '--local-dir',
        type=str,
        metavar='DIR',
        default=LOCAL_DIR,
        help='Local directory for the experiment (default: %(default)s)',
    )
    parser.add_argument(
        '--num-gpus',
        type=float,
        metavar='GPU',
        default=NUM_GPUS_FOR_TRAINER,
        help='number of GPUs for trainer (default: %(default)g)',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        metavar='WORKER',
        default=NUM_WORKERS,
        help='number of rollout workers (default: %(default)d)',
    )
    parser.add_argument(
        '--num-envs-per-worker',
        type=int,
        metavar='ENV',
        default=8,
        help='number of environments per rollout worker (default: %(default)d)',
    )
    parser.add_argument(
        '--timesteps-total',
        type=float,
        metavar='STEP',
        default=10e6,
        help='number of environment steps (default: %(default).1e)',
    )
    parser.add_argument(
        '--seed', type=int, metavar='SEED', nargs='*', default=None, help='the global seed(s)'
    )
    parser.add_argument(
        '--buffer-capacity',
        type=float,
        metavar='EPISODE',
        default=2000,
        help='capacity for the replay buffer (default: %(default).1e episodes)',
    )

    args = parser.parse_args()
    
    # Create experiment from config
    experiment = create_experiment(args.config)

    try:
        analysis = train(experiment, **vars(args))
        
        print("QPLEX HRL training completed!")
        print(f"Best trial: {analysis.best_trial}")
        print(f"Best result: {analysis.best_result}")
        
        return analysis
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()