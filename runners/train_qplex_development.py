"""Training script for QPLEX development với mạng mới và ý tưởng từ deeprec."""

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

# Import MATE environment
import mate
from mate.environment import MultiAgentTracking
from mate.agents import GreedyTargetAgent, RandomTargetAgent

# Import dev QPLEX components
from algorithms.qplex_dev.learner import ReplayBuffer
from algorithms.qplex_dev.enhanced_learner import EnhancedQPLEXLearner

# Import new network components
from new_network.base_network import QNetwork, MixingNetwork, QPLEXNetwork, MLP, RNN, Attention
from new_network.rnn_network import RNNQNetwork, AttentionRNNQNetwork, BiRNNQNetwork, HierarchicalRNNQNetwork

# Import common_net components
from common_net.att_block import MAB, CrossMAB
from common_net.common import GatedMLP, ZeroCenteredRMSNorm

# Training utilities
from training_dev_qplex.curriculum import CurriculumExploration
from training_dev_qplex.monitoring import setup_comprehensive_monitoring
from training_dev_qplex.video import write_video, collect_episode_frames





def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("QPLEX_Development")
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


def create_environment(config: Dict[str, Any]) -> MultiAgentTracking:
    """Create MATE environment."""
    env_config = config['env']
    
    # Load environment configuration
    env = MultiAgentTracking(
        config=env_config['config_file'],
        render_mode=env_config.get('render_mode', 'human'),
        window_size=env_config.get('window_size', 800)
    )
    
    return env


def evaluate_agent(learner: EnhancedQPLEXLearner, env: MultiAgentTracking,
                  n_episodes: int = 10, render: bool = False) -> Dict[str, float]:
    """Evaluate the agent with focus on mean_coverage_rate optimization."""
    episode_rewards = []
    episode_lengths = []
    coverage_rates = []
    transport_rates = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        camera_obs, target_obs = obs
        state = env.state()
        
        learner.reset_hidden_states()
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select actions
            camera_actions, _ = learner.select_action(camera_obs, state, evaluate=True)
            
            # Convert discrete actions to continuous if needed
            if camera_actions.size == env.num_cameras:
                # Discrete actions - convert to continuous
                camera_actions_continuous = np.zeros((env.num_cameras, 2))
                for i in range(env.num_cameras):
                    action_idx = int(camera_actions[i]) if camera_actions.ndim > 0 else int(camera_actions)
                    camera_actions_continuous[i] = [(-1.0 if action_idx == 0 else 1.0), 0.0]
                camera_actions = camera_actions_continuous
            
            # Select actions for targets (random for now)
            target_actions = np.random.uniform(-1, 1, size=(env.num_targets, 2))
            
            # Combine actions
            actions = (camera_actions, target_actions)
            
            # Step environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            next_camera_obs, next_target_obs = next_obs
            camera_rewards, target_rewards = rewards
            next_state = env.state()
            done = terminated or truncated
            
            # Update statistics
            episode_reward += np.sum(camera_rewards)
            episode_length += 1
            
            # Update observations
            camera_obs = next_camera_obs
            state = next_state
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Extract metrics from info - focus on mean_coverage_rate
        if info and len(info) > 0:
            camera_infos, target_infos = info
            if camera_infos and len(camera_infos) > 0:
                # Get coverage_rate from environment or info
                coverage_rate = camera_infos[0].get('coverage_rate', getattr(env, 'coverage_rate', 0.0))
                coverage_rates.append(coverage_rate)
                transport_rates.append(camera_infos[0].get('mean_transport_rate', 0.0))
        else:
            # Fallback: try to get from environment directly
            coverage_rate = getattr(env, 'coverage_rate', 0.0)
            if coverage_rate > 0:
                coverage_rates.append(coverage_rate)
    
    return {
        'mean_episode_reward': np.mean(episode_rewards),
        'std_episode_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'mean_coverage_rate': np.mean(coverage_rates) if coverage_rates else 0.0,
        'std_coverage_rate': np.std(coverage_rates) if coverage_rates else 0.0,
        'mean_transport_rate': np.mean(transport_rates) if transport_rates else 0.0
    }


def train_qplex_development(config: Dict[str, Any], logger: logging.Logger, resume_path: Optional[str] = None):
    """Main training function cho QPLEX development."""
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
    
    # Create environment
    env = create_environment(config)
    logger.info(f"Environment created: {env}")
    
    # Get environment dimensions
    obs_dim = env.camera_observation_dim
    action_dim = 2  # Camera action dimension (rotation, zoom)
    state_dim = env.state_space.shape[0]
    n_agents = env.num_cameras
    
    logger.info(f"Environment dimensions:")
    logger.info(f"  Observation dim: {obs_dim}")
    logger.info(f"  Action dim: {action_dim}")
    logger.info(f"  State dim: {state_dim}")
    logger.info(f"  Number of agents: {n_agents}")
    
    # Create enhanced learner
    learner = EnhancedQPLEXLearner(config, device)
    learner.setup(obs_dim, action_dim, state_dim, n_agents)
    logger.info("Enhanced QPLEX learner created and setup")
    
    # Curriculum and monitoring
    curriculum = CurriculumExploration(config) if config.get('use_curriculum', True) else None
    try:
        metrics_logger = setup_comprehensive_monitoring(learner, env)
    except Exception:
        metrics_logger = None
    
    # RESUME LOGIC
    start_timestep = 0
    if resume_path is not None:
        try:
            logger.info(f"Loading checkpoint from {resume_path}...")
            learner.load(resume_path)
            start_timestep = learner.timestep
            logger.info(f"Resumed successfully! Starting from timestep {start_timestep}, episode {learner.episode_count}")
            logger.info(f"Buffer size: {learner.buffer.size if learner.buffer else 0}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            start_timestep = 0
    
    # Training parameters
    training_config = config['training']
    total_timesteps = training_config['total_timesteps']
    learning_starts = training_config['learning_starts']
    train_freq = training_config['train_freq']
    target_update_interval = training_config['target_update_interval']
    batch_size = training_config['batch_size']
    
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
    logger.info(f"Starting training from timestep {start_timestep}/{total_timesteps}...")
    start_time = time.time()
    
    episode_count = learner.episode_count if resume_path else 0
    episode_reward = 0
    episode_length = 0
    
    # Reset env và hidden states
    obs, info = env.reset()
    camera_obs, target_obs = obs
    state = env.state()
    learner.reset_hidden_states()
    
    eval_results = []
    
    for timestep in range(start_timestep, total_timesteps):
        # Select actions
        camera_actions, action_info = learner.select_action(camera_obs, state)
        
        # Convert discrete actions to continuous if needed
        # The learner returns discrete action indices, but environment expects continuous actions
        if camera_actions.size == env.num_cameras:
            # Discrete actions - convert to continuous
            camera_actions_continuous = np.zeros((env.num_cameras, 2))
            for i in range(env.num_cameras):
                # Map discrete action to continuous space [-1, 1]
                action_idx = int(camera_actions[i]) if camera_actions.ndim > 0 else int(camera_actions)
                # Simple mapping: action 0 -> [-1, 0], action 1 -> [1, 0]
                camera_actions_continuous[i] = [(-1.0 if action_idx == 0 else 1.0), 0.0]
            camera_actions = camera_actions_continuous
        
        # Select actions for targets (random for now)
        target_actions = np.random.uniform(-1, 1, size=(env.num_targets, 2))
        
        # Combine actions
        actions = (camera_actions, target_actions)
        
        # Step environment
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        next_camera_obs, next_target_obs = next_obs
        camera_rewards, target_rewards = rewards
        next_state = env.state()
        done = terminated or truncated
        
        # Learn from experience with info for mean_coverage optimization
        learn_info = {
            'episode_reward': episode_reward,
            'episode_length': episode_length
        }
        # Add environment info for coverage_state extraction
        if info is not None:
            learn_info['env_info'] = info
        
        learning_info = learner.learn(
            obs=camera_obs,
            actions=camera_actions,
            rewards=camera_rewards,
            next_obs=next_camera_obs,
            done=done,
            state=state,
            next_state=next_state,
            info=info,  # Pass actual env info for coverage_state extraction
            env=env
        )
        
        # Update episode statistics
        episode_reward += np.sum(camera_rewards)
        episode_length += 1
        
        # Update observations and state
        camera_obs = next_camera_obs
        state = next_state
        
        # Episode finished
        if done:
            episode_count += 1
            
            # Curriculum epsilon scheduling
            if curriculum is not None and learner.agent is not None:
                learner.agent.epsilon = curriculum.get_epsilon(episode_count)
            
            # Log episode statistics with mean_coverage focus
            if episode_count % 10 == 0:
                # Extract coverage_rate from environment or info
                coverage_rate = getattr(env, 'coverage_rate', 0.0)
                if info and len(info) > 0:
                    camera_infos, _ = info
                    if camera_infos and len(camera_infos) > 0:
                        coverage_rate = camera_infos[0].get('coverage_rate', coverage_rate)
                
                logger.info(f"Episode {episode_count}: Reward = {episode_reward:.2f}, Length = {episode_length}, "
                          f"Mean Coverage Rate = {coverage_rate:.4f} [PRIMARY]")
                
                if metrics_logger is not None:
                    try:
                        # Build coverage_state for monitoring
                        coverage_state = {
                            'coverage_scores': np.ones(env.num_cameras) * coverage_rate,
                            'target_coverage_rate': coverage_rate,  # PRIMARY METRIC
                            'obstacle_violations': 0
                        }
                        # Try to get from environment if available
                        if hasattr(env, 'coverage_state'):
                            coverage_state.update(env.coverage_state)
                        metrics_logger(episode_count, episode_reward, coverage_state)
                    except Exception as e:
                        logger.debug(f"Metrics logging failed: {e}")
            
            # Reset environment
            obs, info = env.reset()
            camera_obs, target_obs = obs
            state = env.state()
            
            # Reset hidden states
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
            logger.info(f"  Episode: {episode_count}")
            logger.info(f"  Mean Loss: {stats.get('mean_loss', 0):.4f}")
            logger.info(f"  Mean Q-values: {stats.get('mean_q_values', 0):.4f}")
            logger.info(f"  Mean TD Error: {stats.get('mean_td_error', 0):.4f}")
            logger.info(f"  Epsilon: {learner.agent.epsilon:.4f}")
            logger.info(f"  FPS: {fps:.2f}")
            
            # Save training stats
            training_stats = {
                'timestep': timestep,
                'episode_count': episode_count,
                'stats': stats,
                'fps': fps
            }
            
            with open(os.path.join(log_dir, 'training_stats.json'), 'w') as f:
                json.dump(training_stats, f, indent=2)
        
        # Evaluation
        if timestep % eval_interval == 0 and timestep > 0:
            logger.info("Running evaluation...")
            eval_results_dict = evaluate_agent(learner, env, n_eval_episodes)
            eval_results_dict['timestep'] = timestep
            eval_results.append(eval_results_dict)
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  Mean Reward: {eval_results_dict['mean_episode_reward']:.2f} ± {eval_results_dict['std_episode_reward']:.2f}")
            logger.info(f"  Mean Length: {eval_results_dict['mean_episode_length']:.2f} ± {eval_results_dict['std_episode_length']:.2f}")
            logger.info(f"  Mean Coverage Rate: {eval_results_dict['mean_coverage_rate']:.4f} ± {eval_results_dict.get('std_coverage_rate', 0.0):.4f} [PRIMARY METRIC]")
            logger.info(f"  Mean Transport Rate: {eval_results_dict.get('mean_transport_rate', 0.0):.4f}")
            
            # Save evaluation results
            with open(os.path.join(log_dir, f'eval_results_{timestep}.json'), 'w') as f:
                json.dump(eval_results_dict, f, indent=2)
            
            # Video recording
            video_cfg = config.get('video', {})
            if video_cfg.get('enabled', True) and timestep % config['video'].get('record_every', eval_interval) == 0:
                try:
                    # Roll out one evaluation episode with frame capture
                    frames = []
                    obs, _ = env.reset()
                    camera_obs, _ = obs
                    state = env.state()
                    learner.reset_hidden_states()
                    done = False
                    steps = 0
                    max_frames = int(video_cfg.get('max_frames', 2000))
                    while not done and steps < max_frames:
                        # capture frame
                        frame = None
                        try:
                            frame = env.render(mode='rgb_array')
                        except Exception:
                            try:
                                frame = env.render(return_image=True)
                            except Exception:
                                try:
                                    frame = env.render()
                                except Exception:
                                    frame = None
                        if isinstance(frame, np.ndarray):
                            frames.append(frame)
                        
                        # step
                        camera_actions, _ = learner.select_action(camera_obs, state, evaluate=True)
                        
                        # Convert discrete actions to continuous if needed
                        if camera_actions.size == env.num_cameras:
                            camera_actions_continuous = np.zeros((env.num_cameras, 2))
                            for i in range(env.num_cameras):
                                action_idx = int(camera_actions[i]) if camera_actions.ndim > 0 else int(camera_actions)
                                camera_actions_continuous[i] = [(-1.0 if action_idx == 0 else 1.0), 0.0]
                            camera_actions = camera_actions_continuous
                        
                        target_actions = np.random.uniform(-1, 1, size=(env.num_targets, 2))
                        actions = (camera_actions, target_actions)
                        next_obs, rewards, terminated, truncated, _ = env.step(actions)
                        next_camera_obs, _ = next_obs
                        state = env.state()
                        done = terminated or truncated
                        camera_obs = next_camera_obs
                        steps += 1
                    
                    if frames:
                        out_path = os.path.join('comparison', f'qplex_dev_eval_{timestep}.mp4')
                        write_video(frames, out_path, fps=int(video_cfg.get('fps', 30)))
                        logger.info(f"Saved evaluation video to {out_path}")
                except Exception as e:
                    logger.warning(f"Video recording failed: {e}")
        
        # Save model
        if timestep % save_interval == 0 and timestep > 0:
            model_path = os.path.join(model_dir, f'qplex_development_{timestep}.pth')
            learner.save(model_path)
            logger.info(f"Model saved to {model_path}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_eval_results = evaluate_agent(learner, env, n_eval_episodes * 2)
    final_eval_results['timestep'] = total_timesteps
    
    logger.info(f"Final Evaluation Results:")
    logger.info(f"  Mean Reward: {final_eval_results['mean_episode_reward']:.2f} ± {final_eval_results['std_episode_reward']:.2f}")
    logger.info(f"  Mean Length: {final_eval_results['mean_episode_length']:.2f} ± {final_eval_results['std_episode_length']:.2f}")
    logger.info(f"  Mean Coverage Rate: {final_eval_results['mean_coverage_rate']:.4f} ± {final_eval_results.get('std_coverage_rate', 0.0):.4f} [PRIMARY METRIC]")
    logger.info(f"  Mean Transport Rate: {final_eval_results.get('mean_transport_rate', 0.0):.4f}")
    
    # Save final results
    with open(os.path.join(log_dir, 'final_eval_results.json'), 'w') as f:
        json.dump(final_eval_results, f, indent=2)
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'qplex_development_final.pth')
    learner.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save all evaluation results
    with open(os.path.join(log_dir, 'all_eval_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info("Training completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train QPLEX Development với mạng mới')
    parser.add_argument('--config', type=str, default='configs/qplex_4v4_9_dev.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_dir = config['logging']['log_dir']
    logger = setup_logging(log_dir)
    
    # Start training
    train_qplex_development(config, logger, resume_path=args.resume)


if __name__ == '__main__':
    main()

