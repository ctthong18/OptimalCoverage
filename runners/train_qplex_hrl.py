"""Training script for QPLEX HRL - Hierarchical Reinforcement Learning with QPLEX."""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import time
import logging
from pathlib import Path
import json
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MATE environment and HRL wrappers
import mate
from mate.environment import MultiAgentTracking
from mate.agents import GreedyTargetAgent, RandomTargetAgent

# Import HRL wrappers from MATE examples
from mate.examples.hrl.wrappers import HierarchicalCamera, MultiDiscrete2DiscreteActionMapper

# Import QPLEX HRL components
from algorithms.qplex_hrl.learner import QPLEXHRLLearner
from algorithms.qplex_hrl.agent import QPLEXHRLAgent

# Import evaluation utilities (similar to MATE HRL approach)
from evaluation_utils import (
    ImprovedEvaluator,
    EvaluationConfig,
    EvaluationLogger
)


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("QPLEX_HRL")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_hrl_environment(config: Dict[str, Any]) -> MultiAgentTracking:
    """
    Create MATE environment with HRL wrappers (similar to MATE HRL examples).
    
    This function creates the environment with hierarchical wrappers that enable
    multi-level target selection and frame skipping, similar to the approach
    used in mate/examples/hrl/qplex.
    """
    env_config = config['env']
    hrl_config = config.get('network', {}).get('hrl', {})
    
    # Create base MATE environment
    base_env = MultiAgentTracking(
        config=env_config['config_file'],
        render_mode=env_config.get('render_mode', 'human'),
        window_size=env_config.get('window_size', 800)
    )
    
    # Create target agent (similar to MATE HRL examples)
    target_agent_type = hrl_config.get('target_agent_type', 'greedy')
    if target_agent_type == 'greedy':
        target_agent = GreedyTargetAgent(seed=config.get('seed', 42))
    else:
        target_agent = RandomTargetAgent(seed=config.get('seed', 42))
    
    # Wrap with MultiCamera (similar to MATE HRL)
    env = mate.MultiCamera(base_env, target_agent=target_agent)
    
    # Apply coordinate transformations (similar to MATE HRL)
    env = mate.RelativeCoordinates(env)
    env = mate.RescaledObservation(env)
    env = mate.RepeatedRewardIndividualDone(env)
    
    # Apply auxiliary rewards for coverage optimization
    reward_coefficients = hrl_config.get('reward_coefficients', {'coverage_rate': 1.0})
    if reward_coefficients:
        env = mate.AuxiliaryCameraRewards(
            env,
            coefficients=reward_coefficients,
            reduction=hrl_config.get('reward_reduction', 'mean')
        )
    
    # Apply hierarchical camera wrapper (key component from MATE HRL)
    multi_selection = hrl_config.get('multi_selection', True)
    frame_skip = hrl_config.get('frame_skip', 5)
    
    env = HierarchicalCamera(
        env,
        multi_selection=multi_selection,
        frame_skip=frame_skip
    )
    
    # Convert multi-discrete to discrete if needed
    if multi_selection:
        env = mate.examples.hrl.wrappers.DiscreteMultiSelection(env)
    
    return env


def extract_target_masks(env, obs) -> Optional[np.ndarray]:
    """
    Extract target visibility masks from environment observation.
    
    This function extracts which targets are visible to each camera,
    which is crucial for hierarchical target selection.
    """
    try:
        if hasattr(env, 'observation_slices'):
            # Use MATE's observation slices to extract target masks
            target_mask_slice = env.observation_slices.get('opponent_mask')
            if target_mask_slice is not None:
                camera_obs, _ = obs
                n_cameras = camera_obs.shape[0]
                target_masks = []
                
                for i in range(n_cameras):
                    mask = camera_obs[i, target_mask_slice].astype(bool)
                    target_masks.append(mask)
                
                return np.array(target_masks)
        
        # Fallback: assume all targets are visible
        if hasattr(env, 'num_cameras') and hasattr(env, 'num_targets'):
            return np.ones((env.num_cameras, env.num_targets), dtype=bool)
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not extract target masks: {e}")
        return None


def evaluate_hrl_agent(learner: QPLEXHRLLearner, env, n_episodes: int = 10, 
                      render: bool = False) -> Dict[str, float]:
    """
    Evaluate HRL agent with hierarchical metrics (similar to MATE HRL evaluation).
    
    This evaluation focuses on hierarchical metrics including:
    - Coverage rates (primary metric for HRL)
    - Target selection efficiency
    - Transport rates
    - Episode rewards and lengths
    """
    episode_rewards = []
    episode_lengths = []
    coverage_rates = []
    transport_rates = []
    selection_efficiencies = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        camera_obs, target_obs = obs
        state = env.state()
        
        # Reset hierarchical states
        learner.reset_hidden_states()
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Track hierarchical metrics
        episode_selections = []
        episode_valid_selections = []
        
        while not done:
            # Extract target masks
            target_masks = extract_target_masks(env, obs)
            
            # Select actions using HRL agent
            camera_actions, action_info = learner.select_action(
                camera_obs, state, target_masks, evaluate=True
            )
            
            # Track target selections for efficiency calculation
            if 'target_selections' in action_info:
                selections = action_info['target_selections']
                episode_selections.append(np.sum(selections))
                
                if target_masks is not None:
                    valid_selections = np.sum(selections * target_masks)
                    episode_valid_selections.append(valid_selections)
            
            # Random target actions (targets are controlled by target agent in env)
            target_actions = np.zeros((env.num_targets, 2))
            
            # Combine actions
            actions = (camera_actions, target_actions)
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            camera_obs, target_obs = obs
            camera_rewards, target_rewards = rewards
            state = env.state()
            
            episode_reward += np.sum(camera_rewards)
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Extract hierarchical metrics from final info
        coverage_rate = 0.0
        transport_rate = 0.0
        
        if info and len(info) > 0:
            camera_infos, target_infos = info
            if camera_infos and len(camera_infos) > 0:
                coverage_rate = camera_infos[0].get('coverage_rate', 0.0)
                transport_rate = camera_infos[0].get('mean_transport_rate', 0.0)
        
        # Try to get from environment directly
        if coverage_rate == 0.0 and hasattr(env, 'coverage_rate'):
            coverage_rate = env.coverage_rate
        
        coverage_rates.append(coverage_rate)
        transport_rates.append(transport_rate)
        
        # Calculate selection efficiency
        if episode_selections and episode_valid_selections:
            total_selections = np.sum(episode_selections)
            total_valid = np.sum(episode_valid_selections)
            efficiency = total_valid / max(total_selections, 1)
            selection_efficiencies.append(efficiency)
    
    return {
        'mean_episode_reward': np.mean(episode_rewards),
        'std_episode_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'mean_coverage_rate': np.mean(coverage_rates) if coverage_rates else 0.0,
        'std_coverage_rate': np.std(coverage_rates) if coverage_rates else 0.0,
        'mean_transport_rate': np.mean(transport_rates) if transport_rates else 0.0,
        'std_transport_rate': np.std(transport_rates) if transport_rates else 0.0,
        'mean_selection_efficiency': np.mean(selection_efficiencies) if selection_efficiencies else 0.0,
        'std_selection_efficiency': np.std(selection_efficiencies) if selection_efficiencies else 0.0
    }


def create_evaluation_config(config: Dict[str, Any]) -> EvaluationConfig:
    """Create EvaluationConfig for comprehensive evaluation."""
    eval_config_dict = config.get('evaluation', {})
    
    eval_config = EvaluationConfig(
        n_eval_runs=eval_config_dict.get('n_eval_runs', 5),
        n_episodes_per_run=eval_config_dict.get('n_episodes_per_run', 200),  # Smaller for HRL
        n_warmup_episodes=eval_config_dict.get('n_warmup_episodes', 10),
        batch_size=eval_config_dict.get('batch_size', 50),
        remove_outliers=eval_config_dict.get('remove_outliers', True),
        outlier_method=eval_config_dict.get('outlier_method', 'iqr'),
        outlier_threshold=eval_config_dict.get('outlier_threshold', 1.5),
        confidence_level=eval_config_dict.get('confidence_level', 0.95),
        seeds=eval_config_dict.get('seeds', None)
    )
    
    return eval_config


def clone_hrl_learner(learner: QPLEXHRLLearner) -> QPLEXHRLLearner:
    """Clone HRL learner for multi-group evaluation."""
    import copy
    
    # Create new learner with same config and device
    cloned_learner = QPLEXHRLLearner(learner.config, learner.device)
    
    # Setup with same dimensions
    if learner.agent is not None:
        cloned_learner.setup(
            learner.obs_dim,
            learner.action_dim,
            learner.state_dim,
            learner.n_agents,
            learner.n_targets
        )
        
        # Copy model parameters
        cloned_learner.agent.q_network.load_state_dict(
            copy.deepcopy(learner.agent.q_network.state_dict())
        )
        cloned_learner.agent.target_q_network.load_state_dict(
            copy.deepcopy(learner.agent.target_q_network.state_dict())
        )
        
        # Copy epsilon and other agent state
        cloned_learner.agent.epsilon = learner.agent.epsilon
        cloned_learner.agent.episode_step = learner.agent.episode_step
    
    return cloned_learner


def train_qplex_hrl(config: Dict[str, Any], logger: logging.Logger, 
                   resume_path: Optional[str] = None):
    """Main training function for QPLEX HRL."""
    # Set random seeds
    seed = config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Setup device
    device = torch.device("cuda" if config['device']['use_cuda'] and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create HRL environment
    env = create_hrl_environment(config)
    logger.info(f"HRL Environment created: {env}")
    
    # Get environment dimensions
    obs_dim = env.camera_observation_dim
    action_dim = env.camera_action_space.n if hasattr(env.camera_action_space, 'n') else 2
    state_dim = env.state_space.shape[0]
    n_agents = env.num_cameras
    n_targets = env.num_targets
    
    logger.info(f"Environment dimensions:")
    logger.info(f"  Observation dim: {obs_dim}")
    logger.info(f"  Action dim: {action_dim}")
    logger.info(f"  State dim: {state_dim}")
    logger.info(f"  Number of agents: {n_agents}")
    logger.info(f"  Number of targets: {n_targets}")
    
    # Create HRL learner
    learner = QPLEXHRLLearner(config, device)
    learner.setup(obs_dim, action_dim, state_dim, n_agents, n_targets)
    logger.info("QPLEX HRL learner created and setup")
    
    # Resume from checkpoint if provided
    start_timestep = 0
    if resume_path is not None:
        try:
            logger.info(f"Loading checkpoint from {resume_path}...")
            learner.load(resume_path)
            start_timestep = learner.timestep
            logger.info(f"Resumed successfully! Starting from timestep {start_timestep}")
            logger.info(f"  Episode count: {learner.episode_count}")
            logger.info(f"  Best coverage: {learner.best_coverage:.4f}")
            logger.info(f"  Buffer size: {learner.buffer.size if learner.buffer else 0}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            start_timestep = 0
    
    # Training parameters
    training_config = config['training']
    total_timesteps = training_config['total_timesteps']
    learning_starts = training_config['learning_starts']
    
    # Logging parameters
    logging_config = config['logging']
    log_interval = logging_config['log_interval']
    eval_interval = logging_config['eval_interval']
    save_interval = logging_config['save_interval']
    log_dir = logging_config['log_dir']
    model_dir = logging_config['model_dir']
    
    # Evaluation parameters
    eval_config = config['evaluation']
    n_eval_episodes = eval_config['n_eval_episodes']
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Training loop
    logger.info(f"Starting QPLEX HRL training from timestep {start_timestep}/{total_timesteps}...")
    logger.info(f"[Optimizing for: mean_coverage_rate as PRIMARY METRIC]")
    start_time = time.time()
    
    episode_count = learner.episode_count if resume_path else 0
    episode_reward = 0
    episode_length = 0
    
    # Reset environment and hierarchical states
    obs, info = env.reset()
    camera_obs, target_obs = obs
    state = env.state()
    learner.reset_hidden_states()
    
    for timestep in range(start_timestep, total_timesteps):
        # Extract target masks for hierarchical selection
        target_masks = extract_target_masks(env, obs)
        
        # Select actions using HRL agent
        camera_actions, action_info = learner.select_action(camera_obs, state, target_masks)
        
        # Random target actions (handled by environment's target agent)
        target_actions = np.zeros((env.num_targets, 2))
        
        # Combine actions
        actions = (camera_actions, target_actions)
        
        # Step environment
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        next_camera_obs, next_target_obs = next_obs
        camera_rewards, target_rewards = rewards
        next_state = env.state()
        done = terminated or truncated
        
        # Learn from experience with HRL enhancements
        learning_info = learner.learn(
            obs=camera_obs,
            actions=camera_actions,
            rewards=camera_rewards,
            next_obs=next_camera_obs,
            done=done,
            state=state,
            next_state=next_state,
            info=info,
            env=env
        )
        
        # Update episode statistics
        episode_reward += np.sum(camera_rewards)
        episode_length += 1
        
        # Update observations and state
        camera_obs = next_camera_obs
        state = next_state
        obs = next_obs
        
        # Episode finished
        if done:
            episode_count += 1
            learner.episode_count = episode_count
            
            # Extract final coverage rate for logging
            coverage_rate = 0.0
            if info and len(info) > 0:
                camera_infos, _ = info
                if camera_infos and len(camera_infos) > 0:
                    coverage_rate = camera_infos[0].get('coverage_rate', 0.0)
            
            if hasattr(env, 'coverage_rate'):
                coverage_rate = max(coverage_rate, env.coverage_rate)
            
            # Log episode statistics with HRL focus
            if episode_count % 10 == 0:
                target_selections = action_info.get('target_selections', np.zeros((n_agents, n_targets)))
                selection_rate = np.mean(target_selections) if target_selections is not None else 0.0
                
                logger.info(f"Episode {episode_count}: Reward = {episode_reward:.2f}, "
                          f"Length = {episode_length}, Coverage = {coverage_rate:.4f} [PRIMARY], "
                          f"Selection Rate = {selection_rate:.3f}")
            
            # Reset environment
            obs, info = env.reset()
            camera_obs, target_obs = obs
            state = env.state()
            
            # Reset hierarchical states
            learner.reset_hidden_states()
            
            # Reset episode statistics
            episode_reward = 0
            episode_length = 0
        
        # Logging
        if timestep % log_interval == 0 and timestep > 0:
            stats = learner.get_training_stats()
            elapsed_time = time.time() - start_time
            fps = timestep / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"Timestep {timestep}/{total_timesteps}")
            logger.info(f"  Elapsed time: {elapsed_time:.2f}s")
            logger.info(f"  Episode count: {episode_count}")
            logger.info(f"  Mean episode reward: {stats.get('mean_episode_reward', 0.0):.2f}")
            logger.info(f"  Mean loss: {stats.get('mean_loss', 0.0):.4f}")
            logger.info(f"  Mean Q-values: {stats.get('mean_q_values', 0.0):.4f}")
            logger.info(f"  Mean TD error: {stats.get('mean_td_error', 0.0):.4f}")
            logger.info(f"  Mean epsilon: {stats.get('epsilon', 0.0):.4f}")
            logger.info(f"  Mean coverage rate: {stats.get('mean_coverage_rate', 0.0):.4f} [PRIMARY METRIC]")
            logger.info(f"  Best coverage: {stats.get('best_coverage', 0.0):.4f}")
            logger.info(f"  Selection entropy: {stats.get('mean_selection_entropy', 0.0):.4f}")
            logger.info(f"  Buffer size: {stats.get('buffer_size', 0)}")
            logger.info(f"  FPS: {fps:.2f}")
            
            # Save training statistics
            with open(os.path.join(log_dir, 'training_stats.json'), 'w') as f:
                json.dump(stats, f, indent=2)
        
        # Evaluation
        if timestep % eval_interval == 0 and timestep > 0:
            logger.info("Evaluating HRL agent...")
            
            # Simple evaluation
            eval_results = evaluate_hrl_agent(learner, env, n_eval_episodes)
            eval_results['timestep'] = timestep
            
            logger.info("Evaluation results:")
            logger.info(f"  Mean episode reward: {eval_results['mean_episode_reward']:.2f} ± {eval_results['std_episode_reward']:.2f}")
            logger.info(f"  Mean episode length: {eval_results['mean_episode_length']:.2f}")
            logger.info(f"  Mean coverage rate: {eval_results['mean_coverage_rate']:.4f} ± {eval_results['std_coverage_rate']:.4f} [PRIMARY METRIC]")
            logger.info(f"  Mean transport rate: {eval_results['mean_transport_rate']:.4f}")
            logger.info(f"  Selection efficiency: {eval_results['mean_selection_efficiency']:.4f}")
            
            # Save evaluation results
            with open(os.path.join(log_dir, f'eval_results_{timestep}.json'), 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            # Try comprehensive evaluation with ImprovedEvaluator
            try:
                logger.info("Running comprehensive evaluation with ImprovedEvaluator...")
                
                eval_config = create_evaluation_config(config)
                learner_group1 = learner
                learner_group2 = clone_hrl_learner(learner)
                
                evaluator = ImprovedEvaluator(eval_config)
                comprehensive_results = evaluator.evaluate(
                    learners=[learner_group1, learner_group2],
                    env=env,
                    timestep=timestep,
                    log_dir=log_dir
                )
                
                logger.info("Comprehensive evaluation completed!")
                
            except Exception as e:
                logger.warning(f"Comprehensive evaluation failed: {e}")
        
        # Save model
        if timestep % save_interval == 0 and timestep > 0:
            model_path = os.path.join(model_dir, f'qplex_hrl_{timestep}.pth')
            learner.save(model_path)
            logger.info(f"Model saved to {model_path}")
    
    # Final evaluation
    logger.info("Running final HRL evaluation...")
    final_eval_results = evaluate_hrl_agent(learner, env, n_eval_episodes * 2)
    final_eval_results['timestep'] = total_timesteps
    
    logger.info("Final evaluation results:")
    logger.info(f"  Mean episode reward: {final_eval_results['mean_episode_reward']:.2f} ± {final_eval_results['std_episode_reward']:.2f}")
    logger.info(f"  Mean episode length: {final_eval_results['mean_episode_length']:.2f}")
    logger.info(f"  Mean coverage rate: {final_eval_results['mean_coverage_rate']:.4f} ± {final_eval_results['std_coverage_rate']:.4f} [PRIMARY METRIC]")
    logger.info(f"  Mean transport rate: {final_eval_results['mean_transport_rate']:.4f}")
    logger.info(f"  Selection efficiency: {final_eval_results['mean_selection_efficiency']:.4f}")
    
    # Save final results
    with open(os.path.join(log_dir, 'final_eval_results.json'), 'w') as f:
        json.dump(final_eval_results, f, indent=2)
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'qplex_hrl_final.pth')
    learner.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    logger.info("QPLEX HRL training completed!")
    
    # Close environment
    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train QPLEX HRL - Hierarchical Reinforcement Learning')
    parser.add_argument('--config', type=str, default='configs/qplex_hrl_4v4_9.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_dir = config['logging']['log_dir']
    logger = setup_logging(log_dir, args.log_level)
    
    logger.info("Starting QPLEX HRL training on MATE environment")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Seed: {config.get('seed', 42)}")
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
    
    try:
        train_qplex_hrl(config, logger, resume_path=args.resume)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()