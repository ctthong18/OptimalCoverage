"""
Training script for MAStAC on MATE environment (two agent groups).

- Mirrors the structure and CLI of your previous QPLEX train script.
- Uses MAStACLearner (learners/mastac_learner.py) which internally uses alg.MAStACTrainer.
- Trains both cameras and targets as MAStAC agents (grouped).
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import time
import logging
import json
from typing import Optional

# ensure project root on path (adjust if needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MATE environment
import mate
from mate.environment import MultiAgentTracking
from mate.agents import GreedyTargetAgent, RandomTargetAgent

# MAStAC learner wrapper
from bayesian_model.networks.learner import MAStACLearner

# helper: setup logging (same pattern as your QPLEX script)
def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("MAStAC")
    logger.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_environment(env_cfg: dict):
    # match your previous create_environment behavior
    render_mode = env_cfg.get('render_mode', None)
    # Convert None string to actual None
    if render_mode == "None" or render_mode is None:
        render_mode = None
    env = MultiAgentTracking(config=env_cfg['config_file'],
                             render_mode=render_mode,
                             window_size=env_cfg.get('window_size', 800))
    return env

def create_target_agent(agent_type: str = "greedy", seed: int = 42):
    if agent_type == "greedy":
        return GreedyTargetAgent(seed=seed)
    elif agent_type == "random":
        return RandomTargetAgent(seed=seed)
    else:
        raise ValueError("Unsupported target agent type")

def evaluate_agent(learner: MAStACLearner, env: MultiAgentTracking, n_episodes: int = 5, render: bool = False, max_steps: int = 2000):
    rewards_list = []
    lengths = []
    coverage_rates = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        camera_obs, target_obs = obs
        state = env.state()
        learner.reset_hidden_states()
        done = False
        total_reward = 0.0
        episode_length = 0
        while not done and episode_length < max_steps:
            cam_actions, tar_actions = learner.select_action(camera_obs, target_obs, state, evaluate=True)
            actions = (cam_actions, tar_actions)
            (next_obs, rewards, terminated, truncated, info) = env.step(actions)
            camera_obs, target_obs = next_obs
            cam_rewards, tar_rewards = rewards
            state = env.state()
            total_reward += np.sum(cam_rewards) + np.sum(tar_rewards)
            episode_length += 1
            done = terminated or truncated
            if render:
                env.render()
        
        rewards_list.append(total_reward)
        lengths.append(episode_length)
        
        # Calculate coverage rate from environment
        # Try multiple methods to get coverage
        coverage = 0.0
        
        # Method 1: From info dict
        if info and len(info) > 0:
            if isinstance(info, tuple) and len(info) >= 2:
                camera_infos, target_infos = info
                if camera_infos and len(camera_infos) > 0:
                    coverage = camera_infos[0].get('coverage_rate', 0.0)
            elif isinstance(info, dict):
                coverage = info.get('coverage_rate', 0.0)
        
        # Method 2: Calculate from environment state if available
        if coverage == 0.0 and hasattr(env, 'get_coverage_rate'):
            try:
                coverage = env.get_coverage_rate()
            except:
                pass
        
        # Method 3: Calculate from environment metrics
        if coverage == 0.0 and hasattr(env, 'metrics'):
            try:
                coverage = env.metrics.get('coverage_rate', 0.0)
            except:
                pass
        
        coverage_rates.append(coverage)
    
    return {
        "mean_episode_reward": float(np.mean(rewards_list)),
        "std_episode_reward": float(np.std(rewards_list)),
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_coverage_rate": float(np.mean(coverage_rates)) if coverage_rates else 0.0,
        "std_coverage_rate": float(np.std(coverage_rates)) if coverage_rates else 0.0,
    }

def train_mastac(config: dict, logger: logging.Logger, resume_path: Optional[str] = None):
    # seeds
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if config['device']['use_cuda'] and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # env
    env = create_environment(config['env'])
    logger.info(f"Created env: {env}")

    # env dims (assuming MultiAgentTracking provides these attrs)
    num_cameras = env.num_cameras
    num_targets = env.num_targets
    obs_dim_cam = env.camera_observation_dim
    obs_dim_tgt = env.target_observation_dim if hasattr(env, 'target_observation_dim') else obs_dim_cam
    state_dim = env.state_space.shape[0] if hasattr(env, 'state_space') else obs_dim_cam
    act_dim_cam = config['env'].get('cam_action_dim', 2)
    act_dim_tgt = config['env'].get('tgt_action_dim', 2)

    logger.info(f"num_cameras={num_cameras}, num_targets={num_targets}, obs_dim_cam={obs_dim_cam}, act_dim_cam={act_dim_cam}")

    # build per-agent dim dicts expected by MAStACLearner.setup
    obs_dims_cam = {i: obs_dim_cam for i in range(num_cameras)}
    obs_dims_tgt = {i: obs_dim_tgt for i in range(num_targets)}
    state_dims = {}  # optional: supply per-agent state dims
    act_dims_cam = {i: act_dim_cam for i in range(num_cameras)}
    act_dims_tgt = {i: act_dim_tgt for i in range(num_targets)}

    # graph coupling: load from config or use empty (fully decentralized)
    ES = config.get('graphs', {}).get('ES', [])
    EO = config.get('graphs', {}).get('EO', [])
    ER = config.get('graphs', {}).get('ER', [])

    # create learner
    learner = MAStACLearner(config, device=device)
    learner.setup(
        num_cameras=num_cameras,
        num_targets=num_targets,
        obs_dims_cam=obs_dims_cam,
        obs_dims_tgt=obs_dims_tgt,
        state_dims=state_dims,
        act_dims_cam=act_dims_cam,
        act_dims_tgt=act_dims_tgt,
        ES=ES, EO=EO, ER=ER
    )
    logger.info("MAStAC learner created and setup")

    # resume if provided
    start_timestep = 0
    if resume_path:
        try:
            logger.info(f"Loading checkpoint {resume_path}")
            learner.load(resume_path)
            start_timestep = learner.timestep
            logger.info(f"Resumed from timestep {start_timestep}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting fresh.")
            start_timestep = 0

    training_cfg = config['training']
    total_timesteps = int(training_cfg['total_timesteps'])
    learning_starts = int(training_cfg.get('learning_starts', 1000))
    train_freq = int(training_cfg.get('train_freq', 1))
    batch_size = int(training_cfg.get('batch_size', 128))
    eval_interval = int(config['logging'].get('eval_interval', 10000))
    save_interval = int(config['logging'].get('save_interval', 10000))
    log_interval = int(config['logging'].get('log_interval', 1000))
    log_dir = config['logging']['log_dir']
    model_dir = config['logging']['model_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # create evaluation target agent if needed (we may still use greedy/random)
    target_controller_type = config['env'].get('target_controller', 'greedy')
    target_agent = create_target_agent(target_controller_type)

    # training loop (timestep-based)
    logger.info(f"Starting training for {total_timesteps} timesteps (start {start_timestep})")
    start_time = time.time()

    # reset env
    obs, info = env.reset()
    camera_obs, target_obs = obs
    state = env.state()
    learner.reset_hidden_states()

    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    # Add max episode steps for training to prevent infinite episodes
    max_episode_steps = config['env'].get('max_episode_steps', 2000)
    current_episode_steps = 0
    
    for t in range(start_timestep, total_timesteps):
        try:
            # Debug logging every 100 steps after evaluation
            if t > 1000 and t < 1200 and t % 10 == 0:
                logger.info(f"DEBUG: Timestep {t}, selecting actions...")
            
            # select actions (learners returns numpy arrays)
            cam_actions, tar_actions = learner.select_action(camera_obs, target_obs, state, evaluate=False)
            
            if t > 1000 and t < 1200 and t % 10 == 0:
                logger.info(f"DEBUG: Actions selected, stepping environment...")

            # If you want to use external target controller for targets instead of learner, do:
            # tar_actions = target_agent.act(target_obs)  # implement GreedyTargetAgent API

            actions = (cam_actions, tar_actions)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            next_camera_obs, next_target_obs = next_obs
            cam_rewards, tar_rewards = rewards
            next_state = env.state()
            done = terminated or truncated
            
            # Force episode termination if max steps reached
            current_episode_steps += 1
            if current_episode_steps >= max_episode_steps:
                done = True
                logger.info(f"FORCE TRUNCATION: Episode truncated at {current_episode_steps} steps (max={max_episode_steps})")

            # call learn
            learner.learn(
                obs=(camera_obs, target_obs),
                actions=(cam_actions, tar_actions),
                rewards=(cam_rewards, tar_rewards),
                next_obs=(next_camera_obs, next_target_obs),
                done=done,
                state=state,
                next_state=next_state,
                info={'timestep': t}
            )

            episode_reward += float(np.sum(cam_rewards) + np.sum(tar_rewards))
            episode_length += 1

            camera_obs = next_camera_obs
            target_obs = next_target_obs
            state = next_state

            if done:
                episode_count += 1
                termination_reason = "terminated" if terminated else "truncated"
                logger.info(f"Episode {episode_count} ended ({termination_reason}): reward={episode_reward:.2f}, length={episode_length}")
                obs, info = env.reset()
                camera_obs, target_obs = obs
                state = env.state()
                learner.reset_hidden_states()
                episode_reward = 0.0
                episode_length = 0
                current_episode_steps = 0  # Reset episode step counter

            # logging
            if t % log_interval == 0 and t > 0:
                stats = learner.get_training_stats()
                elapsed = time.time() - start_time
                steps_per_sec = t / elapsed if elapsed > 0 else 0
                logger.info(f"Timestep {t}/{total_timesteps} | Elapsed: {elapsed:.1f}s | Speed: {steps_per_sec:.1f} steps/s | Episodes: {episode_count}")
                logger.info(f"  Buffer: {stats.get('buffer_size', 0)} | Loss: {stats.get('mean_loss', 0.0):.6f} | Q-value: {stats.get('mean_q_values', 0.0):.4f}")
                logger.info(f"  Current episode: {current_episode_steps} steps, reward: {episode_reward:.2f}")

                with open(os.path.join(log_dir, "train_stats.json"), "w") as f:
                    json.dump(stats, f, indent=2)

            # evaluation
            if t % eval_interval == 0 and t > 0:
                logger.info("="*60)
                logger.info(f"EVALUATION at timestep {t}")
                logger.info("="*60)
                eval_res = evaluate_agent(learner, env, n_episodes=config['evaluation'].get('n_eval_episodes', 5))
                logger.info(f"Eval Results:")
                logger.info(f"  Reward: {eval_res['mean_episode_reward']:.3f} ± {eval_res['std_episode_reward']:.3f}")
                logger.info(f"  Length: {eval_res['mean_episode_length']:.1f}")
                logger.info(f"  Coverage: {eval_res.get('mean_coverage_rate', 0.0):.4f} ± {eval_res.get('std_coverage_rate', 0.0):.4f}")
                logger.info("="*60)
                
                # Save eval results
                eval_file = os.path.join(log_dir, f"eval_results_{t}.json")
                with open(eval_file, "w") as f:
                    json.dump(eval_res, f, indent=2)
                
                # Also append to a summary file for easy tracking
                summary_file = os.path.join(log_dir, "eval_summary.jsonl")
                with open(summary_file, "a") as f:
                    eval_res['timestep'] = t
                    f.write(json.dumps(eval_res) + "\n")
                
                # CRITICAL: Reset environment after evaluation to continue training
                logger.info("Resetting environment after evaluation...")
                obs, info = env.reset()
                camera_obs, target_obs = obs
                state = env.state()
                learner.reset_hidden_states()
                episode_reward = 0.0
                episode_length = 0
                current_episode_steps = 0
                logger.info("Environment reset complete, continuing training...")

            # saving
            if t % save_interval == 0 and t > start_timestep and t > 0:
                save_path = os.path.join(model_dir, f'mastac_model_{t}.pth')
                learner.save(save_path)
                logger.info(f"Saved model to {save_path}")
        
        except Exception as e:
            logger.error(f"ERROR at timestep {t}: {e}")
            logger.error(f"Episode steps: {current_episode_steps}, Episode count: {episode_count}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    # final save and eval
    final_save = os.path.join(model_dir, 'mastac_final.pth')
    learner.save(final_save)
    logger.info(f"Saved final model to {final_save}")
    final_eval = evaluate_agent(learner, env, n_episodes=config['evaluation'].get('n_eval_episodes', 10))
    logger.info(f"Final eval: {final_eval}")

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mastac_mate.yaml")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config['logging']['log_dir'], args.log_level)
    logger.info("Starting MAStAC training")
    if args.resume:
        logger.info(f"Resuming from {args.resume}")

    train_mastac(config, logger, resume_path=args.resume)

if __name__ == "__main__":
    main()
