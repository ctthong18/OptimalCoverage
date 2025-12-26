# alg/mastac.py
"""
MAStAC trainer (Algorithm 1 in paper) — tích hợp core + networks + alg utilities.

Thiết kế:
- Mỗi agent i có: actor[i], actor_target[i], critic[i], critic_target[i], optimizers
- Replay buffer chứa global transitions; khi sample, ta build critic inputs cho từng agent dựa trên I_Q
- Critic update: TD-target sử dụng critic_target and actor_target for next actions
- Actor update: per agent i, compute L_i = - E[ sum_{j in I_i^GD} Q_j(...) ] theo compute_actor_loss_for_agent()
- Soft-update targets mỗi bước theo tau

Giả định:
- Các module core (feature_builder, graph_utils) và networks (Actor, GNNCritic) có sẵn
"""
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from bayesian_model.alg.buffer import ReplayBuffer
from bayesian_model.alg.update_rules import soft_update, hard_update
from bayesian_model.alg.loss_critic import compute_critic_targets, critic_mse_loss
from bayesian_model.alg.loss_actor import compute_actor_loss_for_agent

from bayesian_model.core.mabn import MABN
from bayesian_model.core.dependency import compute_value_dependency_sets
from bayesian_model.core.feature_builder import build_agent_critic_input, build_node_features
from bayesian_model.core.graph_utils import induced_edge_index

from bayesian_model.networks import Actor, GNNCritic

import time


class MAStACTrainer:
    def __init__(
        self,
        n_agents: int,
        ES, EO, ER,
        obs_dims: Dict[int, int],
        act_dims: Dict[int, int],
        state_dims: Dict[int, int],
        actor_hidden=(256, 256),
        critic_gnn_hidden=(128, 128),
        critic_mlp_hidden=(128,),
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.01,
        device: Optional[torch.device] = None,
        kappa: Optional[int] = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.n_agents = int(n_agents)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.state_dims = state_dims

        # Build MABN & dependency sets
        self.mabn = MABN(self.n_agents, ES, EO, ER)
        folded_edges = self.mabn.edge_list()
        # optionally build I_R explicitly; here we derive from ER parents
        I_R = self.mabn.parents_r
        self.I_Q, self.I_GD = compute_value_dependency_sets(self.n_agents, folded_edges, I_R=I_R, kappa=kappa)

        # create networks
        self.actors: Dict[int, Actor] = {}
        self.actors_target: Dict[int, Actor] = {}
        self.critics: Dict[int, GNNCritic] = {}
        self.critics_target: Dict[int, GNNCritic] = {}
        self.actor_opts: Dict[int, optim.Optimizer] = {}
        self.critic_opts: Dict[int, optim.Optimizer] = {}

        for i in range(self.n_agents):
            actor = Actor(obs_dim=obs_dims[i], act_dim=act_dims[i], hidden_sizes=actor_hidden).to(self.device)
            actor_t = Actor(obs_dim=obs_dims[i], act_dim=act_dims[i], hidden_sizes=actor_hidden).to(self.device)
            hard_update(actor, actor_t)
            self.actors[i] = actor
            self.actors_target[i] = actor_t
            self.actor_opts[i] = optim.Adam(actor.parameters(), lr=lr_actor)

            # critic input feature dim = state_dim + action_dim per node (we will use node features built this way)
            node_feat_dim = state_dims[i] + act_dims[i]  # note: used as an upper bound; critics will accept variable node counts
            # but for initialization we pick a nominal node_feature_dim — critic is agnostic to node count at runtime
            critic = GNNCritic(node_features_dim=state_dims[i] + act_dims[i], gnn_hidden=critic_gnn_hidden, mlp_hidden=critic_mlp_hidden).to(self.device)
            critic_t = GNNCritic(node_features_dim=state_dims[i] + act_dims[i], gnn_hidden=critic_gnn_hidden, mlp_hidden=critic_mlp_hidden).to(self.device)
            hard_update(critic, critic_t)
            self.critics[i] = critic
            self.critics_target[i] = critic_t
            self.critic_opts[i] = optim.Adam(critic.parameters(), lr=lr_critic)

        # buffer
        self.buffer = ReplayBuffer(capacity=200000)

    def store_transition(self, states, actions, obs, rewards, next_states, next_obs, done):
        # states/actions/obs are dict agent->array
        self.buffer.add(states, actions, obs, rewards, next_states, next_obs, done)

    def update(self, batch_size: int = 128):
        if len(self.buffer) < batch_size:
            return
        batch = self.buffer.sample(batch_size)

        # ========== CRITIC UPDATES ==========
        # For each agent j, build lists for:
        # - node_feats (current) for critic j (built from states & actions in transition)
        # - node_feats_next (next states & actions' from target actors)
        # - edge_index local (induced from folded)
        # - local_index (position of agent j in ordering)
        folded_edges = self.mabn.edge_list()

        for j in range(self.n_agents):
            # build batch lists
            batch_node_feats = []
            batch_edge_idx = []
            local_indices = []
            batch_rewards = []
            batch_node_feats_next = []
            batch_edge_idx_next = []
            local_indices_next = []
            for tr in batch:
                # current node features for I_Q[j] from stored states and actions
                states_k = tr["states"]
                actions_k = tr["actions"]
                ordering = sorted(list(self.I_Q[j]))
                # build node features via core.feature_builder (numpy)
                node_feats_np = build_agent_critic_input(j, self.I_Q[j], states_k, actions_k)[0]  # np.ndarray (N_j, feat_dim)
                # convert to tensor
                node_feats = torch.tensor(node_feats_np, dtype=torch.float32, device=self.device)
                batch_node_feats.append(node_feats)

                # build edge index local (numpy) then tensor
                edge_index_np = induced_edge_index(folded_edges, ordering)
                edge_index_t = torch.tensor(edge_index_np, dtype=torch.long, device=self.device) if edge_index_np.size != 0 else None
                batch_edge_idx.append(edge_index_t)
                local_indices.append(ordering.index(j))

                # rewards
                batch_rewards.append(tr["rewards"][j])

                # Next-state node features: we need actions for next state -> use target actors to compute
                # build next actions for agents in ordering via actors_target
                next_states_k = tr["next_states"]
                next_obs_k = tr["next_obs"]
                # compute next actions for agents in ordering using target actors (no grad)
                a_next_dict = {}
                for ag in ordering:
                    # Ensure agent ag exists in next_obs_k, otherwise use next_states_k as fallback
                    if ag not in next_obs_k:
                        # Fallback: use state if obs not available
                        if ag in next_states_k:
                            obs_ag = torch.tensor(next_states_k[ag], dtype=torch.float32, device=self.device).unsqueeze(0)
                        else:
                            # Last resort: skip this agent or use zeros (should not happen)
                            raise KeyError(f"Agent {ag} not found in next_obs_k or next_states_k. Available keys: {list(next_obs_k.keys())}, {list(next_states_k.keys())}")
                    else:
                        obs_ag = torch.tensor(next_obs_k[ag], dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        a_next = self.actors_target[ag](obs_ag).cpu().numpy().squeeze(0)
                    a_next_dict[ag] = a_next
                node_feats_next_np = build_agent_critic_input(j, self.I_Q[j], next_states_k, a_next_dict)[0]
                node_feats_next = torch.tensor(node_feats_next_np, dtype=torch.float32, device=self.device)
                batch_node_feats_next.append(node_feats_next)
                edge_index_next_np = edge_index_np  # same induced edges for ordering
                edge_index_next_t = edge_index_t
                batch_edge_idx_next.append(edge_index_next_t)
                local_indices_next.append(ordering.index(j))

            # Stack rewards into tensor
            rewards_tensor = torch.tensor(np.array(batch_rewards, dtype=np.float32)).unsqueeze(1).to(self.device)

            # Compute targets y = r + gamma * Q'_j(next)
            y = compute_critic_targets(self.critics_target[j], batch_node_feats_next, batch_edge_idx_next, local_indices_next, rewards_tensor, self.gamma, self.device)

            # Compute current Q estimates q(s,a)
            q_values_list = []
            for nf, ei, li in zip(batch_node_feats, batch_edge_idx, local_indices):
                q_val = self.critics[j](nf, ei, li)  # shape (1,)
                if q_val.dim() == 0:
                    q_val = q_val.unsqueeze(0)
                q_values_list.append(q_val)
            q_values = torch.stack(q_values_list, dim=0).float().to(self.device)  # (B,1)

            loss_q = critic_mse_loss(q_values, y)
            # gradient step for critic j
            self.critic_opts[j].zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[j].parameters(), max_norm=1.0)
            self.critic_opts[j].step()

        # ========== ACTOR UPDATES ==========
        # For each actor i, compute actor loss using critics j in I_GD[i]
        for i in range(self.n_agents):
            if len(self.I_GD[i]) == 0:
                continue
            # prepare batch transitions (list of dict as sampled)
            batch_transitions = batch  # list of transitions
            # compute actor loss (grad flows to actor i)
            loss_actor = compute_actor_loss_for_agent(
                agent_id=i,
                actor=self.actors[i],
                critics=self.critics,
                I_GD_of_i=self.I_GD[i],
                batch_transitions=batch_transitions,
                I_Q_dict=self.I_Q,
                feature_builder_fn=None,  # not used; we use core.feature_builder inside loss fn via induced_edge_index
                graph_utils_induced_fn=induced_edge_index,
                folded_edge_list=folded_edges,
                device=self.device,
            )
            self.actor_opts[i].zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), max_norm=1.0)
            self.actor_opts[i].step()

        # soft update targets
        for i in range(self.n_agents):
            soft_update(self.actors[i], self.actors_target[i], self.tau)
            soft_update(self.critics[i], self.critics_target[i], self.tau)
