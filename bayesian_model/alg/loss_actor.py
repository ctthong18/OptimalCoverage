# alg/loss_actor.py
"""
Actor loss for MAStAC per paper:

L_theta_i = - E_{s_{I_Q^hat}} [ sum_{j in I_i^GD} Q_j(s_{I_j^Q}, a_{I_j^Q}) * ].

Implementation detail:
- For each sampled transition in a minibatch:
    - For each j in I_i^GD:
        - build node_features for I_Q[j] where:
            * action of agent i replaced by actor_i(obs_i) (current actor output, requires grad)
            * other agents' actions are taken from the transition (stored actions, no grad)
    - Evaluate critic_j on these node_features to get Q_j (grad flows to actor_i).
- Sum Q_j over j in I_i^GD and average over batch, then neg as loss.
"""

from typing import List
import torch


def compute_actor_loss_for_agent(
    agent_id: int,
    actor,  # Actor module for agent i
    critics,  # dict j->critic module (current critics)
    I_GD_of_i,  # set/list of j indices
    batch_transitions,  # list of transitions (dict) from buffer
    I_Q_dict,  # dict j -> set of agents in I_Q[j]
    feature_builder_fn,  # function to build node features: (states, actions, ordering)->np.ndarray
    graph_utils_induced_fn,  # function to build induced edge_index: (edge_list, ordering) -> np.ndarray
    folded_edge_list,  # global folded edges list for induced subgraph
    device,
):
    """
    Returns actor_loss: torch scalar (requires grad wrt actor params).
    """
    batch_size = len(batch_transitions)
    losses = []
    # For each transition in batch, build actor output for agent i (with grad)
    # We collect per-sample actor actions
    obs_batch = []
    for tr in batch_transitions:
        obs_batch.append(torch.tensor(tr["obs"][agent_id], dtype=torch.float32).to(device))
    obs_batch_tensor = torch.stack(obs_batch, dim=0)  # (B, obs_dim)
    # Compute actions for agent i (requires grad) — actor must be in training mode
    a_i_batch = actor(obs_batch_tensor)  # (B, act_dim) with grad

    # For each sample k, compute sum_j Q_j(...) where actions for agent i are replaced by a_i_batch[k]
    total_qs = []
    for k, tr in enumerate(batch_transitions):
        q_sum = 0.0
        # prepare global states & actions dicts (numpy -> will convert to tensors later)
        states_k = tr["states"]
        actions_k = {int(a): tr["actions"][int(a)].copy() for a in tr["actions"].keys()}  # numpy arrays
        # replace actions_k[agent_id] with actor's current action (detach? NO — want grad to flow)
        # convert a_i_batch[k] to numpy-like or keep tensor and build node_features using tensors
        # We'll build node_features directly using torch for grad on agent_i action.
        a_i_tensor = a_i_batch[k]  # tensor shape (act_dim,)
        for j in I_GD_of_i:
            I_Qj = sorted(list(I_Q_dict[int(j)]))
            # build node features per node in ordering:
            node_feats_list = []
            local_index = None
            for idx_local, ag in enumerate(I_Qj):
                s_j = torch.tensor(states_k[ag], dtype=torch.float32, device=device)
                if ag == agent_id:
                    a_j = a_i_tensor  # tensor with grad
                else:
                    a_j = torch.tensor(actions_k[ag], dtype=torch.float32, device=device)
                node_feat = torch.cat([s_j, a_j], dim=0)  # (feat_dim,)
                node_feats_list.append(node_feat)
                if ag == agent_id:
                    local_index = idx_local
            node_features = torch.stack(node_feats_list, dim=0)  # (N_j, feat_dim)
            # build edge_index for induced subgraph (use provided util)
            edge_index_np = graph_utils_induced_fn(folded_edge_list, I_Qj)  # numpy (2,E)
            edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device) if edge_index_np.size != 0 else None
            q_j = critics[int(j)](node_features, edge_index, local_index)  # shape (1,)
            if q_j.dim() == 0:
                q_j = q_j.unsqueeze(0)
            q_sum = q_sum + q_j
        total_qs.append(q_sum)  # tensor (1,) with grad to actor params
    # stack and average
    qs_tensor = torch.stack(total_qs, dim=0).squeeze(-1)  # (B,)
    loss = - qs_tensor.mean()
    return loss
