"""
MAStAC Learner wrapper for train_mastac.py

This wrapper exposes an API similar to QPLEXLearner used by your previous train script:
- setup(obs_dim, action_dim, state_dim, n_agents_groups)
- select_action(obs_group1, obs_group2, state, evaluate=False)
- learn(obs, actions, rewards, next_obs, done, state, next_state, info)
- save(path), load(path)
- reset_hidden_states() (stub, kept for compatibility)
- get_training_stats()
- attributes: timestep, episode_count, buffer_size

This version trains two groups of agents:
 - group A: cameras (num_cameras)
 - group B: targets (num_targets)
It constructs a combined multi-agent problem with n_total = num_cameras + num_targets
and uses MAStACTrainer (alg.mastac.MAStACTrainer) to learn policies for all agents.

Assumptions:
- alg.MAStACTrainer exists and implements:
    - constructor with args (n_agents, ES, EO, ER, obs_dims, act_dims, state_dims, ...)
    - store_transition(states, actions, obs, rewards, next_states, next_obs, done)
    - update(batch_size)
    - save(path)/ load(path) optional (we implement saving wrapper for actor/critic weights)
- core/feature_builder and graph_utils exist (previously provided).

Note: Adapt hyperparams and graph coupling ES/EO/ER to your scenario in config.
"""

import os
import time
import json
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch

# If your project root differs, adjust path or use package imports
# from alg.mastac import MAStACTrainer  # previously implemented trainer
from bayesian_model.alg.mastac import MAStACTrainer

# Local path to the paper you uploaded (reference)
PAPER_PATH = "/mnt/data/2510.09937v1.pdf"


class MAStACLearner:
    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        """
        config: loaded YAML config for MAStAC. Expected keys (example):
          - env: {}
          - model: {actor_hidden, critic_gnn_hidden, ...}
          - training: {batch_size, train_freq, ...}
          - graphs: {ES, EO, ER, kappa}
        """
        self.config = config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Placeholders set at setup()
        self.trainer: Optional[MAStACTrainer] = None
        self.num_cameras = None
        self.num_targets = None
        self.timestep = 0
        self.episode_count = 0
        self._stats = {
            "losses": [],
            "q_values": [],
            "epsilons": [],
        }

    def setup(self,
              num_cameras: int,
              num_targets: int,
              obs_dims_cam: Dict[int, int],
              obs_dims_tgt: Dict[int, int],
              state_dims: Dict[int, int],
              act_dims_cam: Dict[int, int],
              act_dims_tgt: Dict[int, int],
              ES, EO, ER):
        """
        Build combined MAStACTrainer for both groups.

        We'll index agents with:
          0..num_cameras-1         -> camera agents
          num_cameras..num_total-1 -> target agents
        """
        n_c = int(num_cameras)
        n_t = int(num_targets)
        n_total = n_c + n_t
        self.num_cameras = n_c
        self.num_targets = n_t

        # Build per-agent dims (global indexing)
        obs_dims = {}
        state_dims_full = {}
        act_dims = {}

        # cameras first
        for i in range(n_c):
            obs_dims[i] = int(obs_dims_cam[i])
            act_dims[i] = int(act_dims_cam[i])
            state_dims_full[i] = int(state_dims.get(i, obs_dims_cam[i]))  # fallback

        # targets next (index offset = n_c)
        for k in range(n_t):
            idx = n_c + k
            obs_dims[idx] = int(obs_dims_tgt[k])
            act_dims[idx] = int(act_dims_tgt[k])
            state_dims_full[idx] = int(state_dims.get(idx, obs_dims_tgt[k]))

        # Model hyperparams from config
        model_cfg = self.config.get("model", {})
        actor_hidden = tuple(model_cfg.get("actor_hidden", (256, 256)))
        critic_gnn_hidden = tuple(model_cfg.get("critic_gnn_hidden", (128, 128)))
        critic_mlp_hidden = tuple(model_cfg.get("critic_mlp_hidden", (128,)))
        lr_actor = float(model_cfg.get("lr_actor", 1e-4))
        lr_critic = float(model_cfg.get("lr_critic", 1e-3))
        tau = float(model_cfg.get("tau", 0.01))
        gamma = float(model_cfg.get("gamma", 0.99))
        kappa = self.config.get("graphs", {}).get("kappa", None)

        # create MAStAC trainer for all agents
        self.trainer = MAStACTrainer(
            n_agents=n_total,
            ES=ES,
            EO=EO,
            ER=ER,
            obs_dims=obs_dims,
            act_dims=act_dims,
            state_dims=state_dims_full,
            actor_hidden=actor_hidden,
            critic_gnn_hidden=critic_gnn_hidden,
            critic_mlp_hidden=critic_mlp_hidden,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            device=self.device,
            kappa=kappa,
        )

        # training bookkeeping
        self.batch_size = int(self.config["training"].get("batch_size", 128))
        self.train_freq = int(self.config["training"].get("train_freq", 1))
        self.learning_starts = int(self.config["training"].get("learning_starts", 1000))
        self.timestep = 0
        self.episode_count = 0

    def select_action(self,
                      camera_obs: np.ndarray,
                      target_obs: Optional[np.ndarray],
                      state: Optional[np.ndarray],
                      evaluate: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given current observations, return actions for cameras and targets.

        camera_obs: shape (num_cameras, obs_dim_cam) or list of obs per camera
        target_obs: shape (num_targets, obs_dim_tgt) or list
        state: global state (unused here but kept for API compatibility)
        evaluate: if True, don't add exploration noise

        Returns:
          camera_actions: np.ndarray (num_cameras, act_dim_cam)
          target_actions: np.ndarray (num_targets, act_dim_tgt)
        """
        assert self.trainer is not None, "Trainer not initialized. Call setup() first."

        cams = []
        tars = []
        n_c = self.num_cameras
        n_t = self.num_targets

        # normalize input to list/dict
        # camera_obs assumed to be list/array length n_c, each obs vector
        for i in range(n_c):
            obs_i = torch.tensor(camera_obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                a_i = self.trainer.actors[i].act(obs_i, noise_std=0.0 if evaluate else float(self.config["training"].get("exploration_noise", 0.1)))
            cams.append(a_i.cpu().numpy().squeeze(0))

        # targets: their global indices offset
        for k in range(n_t):
            idx = n_c + k
            obs_k = torch.tensor(target_obs[k], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                a_k = self.trainer.actors[idx].act(obs_k, noise_std=0.0 if evaluate else float(self.config["training"].get("exploration_noise", 0.1)))
            tars.append(a_k.cpu().numpy().squeeze(0))

        cams_arr = np.stack(cams, axis=0)
        tars_arr = np.stack(tars, axis=0) if len(tars) > 0 else np.zeros((0, 0))
        return cams_arr, tars_arr

    def learn(self,
              obs: np.ndarray,
              actions: np.ndarray,
              rewards: np.ndarray,
              next_obs: np.ndarray,
              done: bool,
              state: Optional[np.ndarray],
              next_state: Optional[np.ndarray],
              info: Optional[dict] = None) -> Dict[str, Any]:
        """
        Called by train loop to store transition and trigger learning steps.

        Input formats:
          obs: tuple (camera_obs, target_obs) where each is list/array of per-agent obs
          actions: tuple (camera_actions, target_actions)
          rewards: tuple (camera_rewards, target_rewards)
          next_obs: (camera_obs_next, target_obs_next)

        Returns: dict of training info (losses, q-values etc)
        """
        assert self.trainer is not None, "Trainer not initialized."

        cam_obs, tar_obs = obs
        cam_actions, tar_actions = actions
        cam_rewards, tar_rewards = rewards
        cam_obs_next, tar_obs_next = next_obs

        # build global states / actions / obs dicts expected by trainer.store_transition:
        # global indexing: cameras first, then targets
        states = {}       # we don't always have full state per agent; use obs->state mapping or leave zeros
        actions_dict = {}
        obs_dict = {}
        next_states = {}
        next_actions_dict = {}
        next_obs_dict = {}

        n_c = self.num_cameras
        n_t = self.num_targets

        # if state dict provided as vector, we won't split by agent; trainer expects per-agent state arrays.
        # Here we try to use obs as state when per-agent states not available.
        for i in range(n_c):
            obs_i = np.asarray(cam_obs[i], dtype=np.float32)
            states[i] = obs_i  # fallback: use obs as state
            actions_dict[i] = np.asarray(cam_actions[i], dtype=np.float32)
            obs_dict[i] = obs_i
            next_obs_i = np.asarray(cam_obs_next[i], dtype=np.float32)
            next_states[i] = next_obs_i
            next_obs_dict[i] = next_obs_i  # FIX: populate next_obs_dict
            next_actions_dict[i] = np.asarray(cam_actions[i], dtype=np.float32)  # placeholder; trainer will compute next actions via actors_target

        for k in range(n_t):
            idx = n_c + k
            obs_k = np.asarray(tar_obs[k], dtype=np.float32)
            states[idx] = obs_k
            actions_dict[idx] = np.asarray(tar_actions[k], dtype=np.float32)
            obs_dict[idx] = obs_k
            next_obs_k = np.asarray(tar_obs_next[k], dtype=np.float32)
            next_states[idx] = next_obs_k
            next_obs_dict[idx] = next_obs_k  # FIX: populate next_obs_dict
            next_actions_dict[idx] = np.asarray(tar_actions[k], dtype=np.float32)

        # combine rewards into per-agent dict
        # Handle both scalar and array/list rewards from MATE environment
        rewards_dict = {}
        
        # Camera rewards: can be scalar (float) or list/array
        if isinstance(cam_rewards, (int, float)):
            # Scalar reward - distribute to all camera agents
            cam_reward_val = float(cam_rewards)
            for i in range(n_c):
                rewards_dict[i] = cam_reward_val
        elif isinstance(cam_rewards, (list, np.ndarray)):
            # Array/list reward - one per agent
            cam_rewards_arr = np.asarray(cam_rewards, dtype=np.float32)
            if len(cam_rewards_arr) == 1:
                # Single value repeated for all agents
                for i in range(n_c):
                    rewards_dict[i] = float(cam_rewards_arr[0])
            else:
                # One reward per agent
                for i in range(n_c):
                    rewards_dict[i] = float(cam_rewards_arr[i])
        else:
            # Fallback: treat as scalar
            cam_reward_val = float(cam_rewards)
            for i in range(n_c):
                rewards_dict[i] = cam_reward_val
        
        # Target rewards: can be scalar (float) or list/array
        if isinstance(tar_rewards, (int, float)):
            # Scalar reward - distribute to all target agents
            tar_reward_val = float(tar_rewards)
            for k in range(n_t):
                rewards_dict[n_c + k] = tar_reward_val
        elif isinstance(tar_rewards, (list, np.ndarray)):
            # Array/list reward - one per agent
            tar_rewards_arr = np.asarray(tar_rewards, dtype=np.float32)
            if len(tar_rewards_arr) == 1:
                # Single value repeated for all agents
                for k in range(n_t):
                    rewards_dict[n_c + k] = float(tar_rewards_arr[0])
            else:
                # One reward per agent
                for k in range(n_t):
                    rewards_dict[n_c + k] = float(tar_rewards_arr[k])
        else:
            # Fallback: treat as scalar
            tar_reward_val = float(tar_rewards)
            for k in range(n_t):
                rewards_dict[n_c + k] = tar_reward_val

        # store transition into trainer's buffer
        self.trainer.store_transition(states, actions_dict, obs_dict, rewards_dict, next_states, next_obs_dict, done)

        # increase timestep
        self.timestep += 1

        # call update periodically
        info_out = {}
        if (self.timestep >= self.learning_starts) and (self.timestep % self.train_freq == 0):
            self.trainer.update(batch_size=self.batch_size)
            # In a more detailed implementation we would capture losses and q-values returned by trainer.update
            info_out['updated'] = True
        else:
            info_out['updated'] = False

        return info_out

    def reset_hidden_states(self):
        # Placeholder for compatibility. If using RNN actors, reset hidden here.
        return

    def save(self, path: str):
        """Save model weights and learner bookkeeping."""
        assert self.trainer is not None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save actor and critic weights per agent
        save_dict = {
            "timestep": self.timestep,
            "episode_count": self.episode_count,
            "trainer": {}
        }
        for i in range(self.trainer.n_agents):
            save_dict["trainer"][f"actor_{i}"] = self.trainer.actors[i].state_dict()
            save_dict["trainer"][f"actor_target_{i}"] = self.trainer.actors_target[i].state_dict()
            save_dict["trainer"][f"critic_{i}"] = self.trainer.critics[i].state_dict()
            save_dict["trainer"][f"critic_target_{i}"] = self.trainer.critics_target[i].state_dict()

        torch.save(save_dict, path)

    def load(self, path: str):
        d = torch.load(path, map_location=self.device)
        # if trainer not created yet, cannot load -> user should call setup() first with matching dims
        assert self.trainer is not None, "Call setup() with matching model dims before load()"
        self.timestep = int(d.get("timestep", 0))
        self.episode_count = int(d.get("episode_count", 0))
        trainer_state = d.get("trainer", {})
        for i in range(self.trainer.n_agents):
            ai = trainer_state.get(f"actor_{i}", None)
            if ai is not None:
                self.trainer.actors[i].load_state_dict(ai)
            at = trainer_state.get(f"actor_target_{i}", None)
            if at is not None:
                self.trainer.actors_target[i].load_state_dict(at)
            ci = trainer_state.get(f"critic_{i}", None)
            if ci is not None:
                self.trainer.critics[i].load_state_dict(ci)
            ct = trainer_state.get(f"critic_target_{i}", None)
            if ct is not None:
                self.trainer.critics_target[i].load_state_dict(ct)

    def get_training_stats(self) -> Dict[str, Any]:
        # Return aggregate metrics; trainer could be extended to return exact losses and q-values
        return {
            "timestep": self.timestep,
            "episode_count": self.episode_count,
            "buffer_size": len(self.trainer.buffer) if self.trainer is not None else 0,
            "mean_loss": float(np.mean(self._stats["losses"])) if len(self._stats["losses"]) > 0 else 0.0,
            "mean_q_values": float(np.mean(self._stats["q_values"])) if len(self._stats["q_values"]) > 0 else 0.0,
            "mean_epsilon": float(np.mean(self._stats["epsilons"])) if len(self._stats["epsilons"]) > 0 else 0.0,
        }
