import random
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

from .enhanced_model import EnhancedQPLEXNetwork


class EnhancedQPLEXAgent:
    """Enhanced QPLEX Agent với mạng mới và training tối ưu từ deeprec."""
    
    def __init__(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int,
                 config: Dict[str, Any], device: torch.device):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.config = config
        self.device = device
        
        # Algorithm parameters
        algo_config = config.get('algorithm', {})
        self.learning_rate = algo_config.get('learning_rate', 0.0005)
        self.gamma = algo_config.get('gamma', 0.99)
        self.tau = algo_config.get('tau', 0.005)
        self.epsilon_start = algo_config.get('epsilon_start', 1.0)
        self.epsilon_end = algo_config.get('epsilon_end', 0.05)
        self.epsilon_decay = algo_config.get('epsilon_decay', 0.9999)
        self.dueling = algo_config.get('dueling', True)
        self.double_q = algo_config.get('double_q', True)
        
        # Initialize enhanced networks
        self.q_network = EnhancedQPLEXNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            config=config
        ).to(device)
        
        self.target_q_network = EnhancedQPLEXNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            config=config
        ).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer và scheduler
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100000
        )
        
        # Exploration
        self.epsilon = self.epsilon_start
        
        # Hidden states for RNN
        self.hidden_states: List = [None] * n_agents
        self.target_hidden_states: List = [None] * n_agents
        
        # Training stats
        self.training_stats: Dict[str, List[float]] = {'loss': [], 'q_values': [], 'target_q_values': [], 'td_errors': []}
        self.training_step = 0
    
    def select_action(self, obs: np.ndarray, state: np.ndarray,
                    evaluate: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        with torch.no_grad():
            # Convert obs/state to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device) \
                        if len(obs.shape) == 2 else torch.FloatTensor(obs).to(self.device)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) \
                        if len(state.shape) == 1 else torch.FloatTensor(state).to(self.device)
        
            # Get Q-values
            q_values, _, new_hidden = self.q_network(obs_tensor, state_tensor, self.hidden_states)
            self.hidden_states = new_hidden

            # Decide actions
            if evaluate or random.random() > self.epsilon:
                # Greedy
                actions = q_values.argmax(dim=-1).squeeze(0).cpu().numpy()  # shape: (n_agents,)
            else:
                # Random
                num_actions = q_values.shape[-1]
                actions = np.random.randint(0, num_actions, size=(self.n_agents,))  # shape: (n_agents,)

            # Info dict
            info = {
                'q_values': q_values.cpu().numpy().squeeze(0),  # (n_agents, action_dim)
                'epsilon': self.epsilon,
                'actions': actions
            }

        return actions, info
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.training_step += 1
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        state = batch['state']
        next_state = batch['next_state']
        
        current_q_values, current_q_total, _ = self.q_network(obs, state)
        
        if self.double_q:
            next_q_values, _, _ = self.q_network(next_obs, next_state)
            next_actions = next_q_values.argmax(dim=-1)
            target_q_values, target_q_total, _ = self.target_q_network(next_obs, next_state)
            target_q_values = target_q_values.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
        else:
            target_q_values, target_q_total, _ = self.target_q_network(next_obs, next_state)
            target_q_values = target_q_values.max(dim=-1)[0]
        
        target_q_total = target_q_total.squeeze(-1)
        target_q_total = rewards.sum(dim=1) + self.gamma * target_q_total * (1 - dones.float())
        target_q_individual = rewards + self.gamma * target_q_values * (1 - dones.unsqueeze(1).float())
        
        if actions.dim() == 3 and actions.size(-1) > 1:
            actions = actions.argmax(dim=-1)
        elif actions.dim() == 2 and actions.size(-1) == 1:
            actions = actions.squeeze(-1)
        
        current_q_selected = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        current_q_total_selected = current_q_total.squeeze(-1)
        
        individual_loss = F.smooth_l1_loss(current_q_selected, target_q_individual)
        total_loss = F.smooth_l1_loss(current_q_total_selected, target_q_total)
        loss = individual_loss + total_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        
        self._soft_update_target_network()
        
        self.training_stats['loss'].append(loss.item())
        self.training_stats['q_values'].append(current_q_selected.mean().item())
        self.training_stats['target_q_values'].append(target_q_individual.mean().item())
        self.training_stats['td_errors'].append((target_q_individual - current_q_selected).abs().mean().item())
        self.update_epsilon()
        
        return {
            'loss': loss.item(),
            'individual_loss': individual_loss.item(),
            'total_loss': total_loss.item(),
            'q_values': current_q_selected.mean().item(),
            'target_q_values': target_q_individual.mean().item(),
            'td_error': (target_q_individual - current_q_selected).abs().mean().item(),
            'epsilon': self.epsilon,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def _soft_update_target_network(self):
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
    
    def reset_hidden_states(self):
        self.hidden_states = [None] * self.n_agents
        self.target_hidden_states = [None] * self.n_agents

