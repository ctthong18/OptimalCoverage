import numpy as np

class OptimizedTensorReward:
    """Reward shaping optimized for mean_coverage_rate as primary objective."""
    
    def __init__(self, coverage_weight: float = 5.0):
        """
        Args:
            coverage_weight: Weight for mean_coverage optimization (primary objective)
        """
        self.weights = {
            'coverage': coverage_weight,  # PRIMARY: Tối ưu mean_coverage
            'mean_coverage': coverage_weight * 2.0,  # Direct mean_coverage reward
            'tracking': 2.0,      # Secondary: tracking reward  
            'energy': -0.05,      # Giảm energy penalty
            'obstacle': -1.0,     # Tăng obstacle penalty
            'collaboration': 1.0, # Thêm collaboration bonus
            'coverage_balance': 0.5, # Tránh tập trung
        }
        
    def compute(self, coverage_state):
        """
        Compute reward shaping with focus on mean_coverage_rate optimization.
        
        Args:
            coverage_state: Dict containing coverage metrics including:
                - coverage_scores: array of coverage scores per agent
                - target_coverage_rate: mean coverage rate (PRIMARY METRIC)
                - n_tensors: number of agents
                - target_distances: distances to targets
                - rotation_costs: energy costs
                - obstacle_distances: distances to obstacles
                - target_velocity: target velocity
                - reliable_radius: reliable sensing radius
                - max_radius: maximum sensing radius
        """
        n_agents = coverage_state.get('n_tensors', len(coverage_state.get('coverage_scores', [])))
        rewards = np.zeros(n_agents)
        
        # PRIMARY OBJECTIVE: Mean coverage rate optimization
        mean_coverage = coverage_state.get('target_coverage_rate', 0.0)
        if mean_coverage > 0:
            # Reward all agents based on mean_coverage (shared reward)
            mean_coverage_reward = self.weights['mean_coverage'] * mean_coverage
            rewards += mean_coverage_reward
        
        # 1. Individual coverage scores (probabilistic coverage)
        coverage_scores = coverage_state.get('coverage_scores', np.zeros(n_agents))
        if len(coverage_scores) == n_agents:
            rewards += self.weights['coverage'] * coverage_scores
        
        # 2. Probabilistic coverage reward (Bài báo ACDRL)
        coverage_prob = self._compute_probabilistic_coverage(coverage_state)
        rewards += self.weights['coverage'] * 0.5 * coverage_prob  # Reduced weight since we have mean_coverage
        
        # 3. Target tracking với distance-based decay
        target_reward = self._compute_target_tracking(coverage_state)
        rewards += self.weights['tracking'] * target_reward
        
        # 4. Energy-efficient penalty (Bài báo vision-based)
        rotation_costs = coverage_state.get('rotation_costs', np.zeros(n_agents))
        if len(rotation_costs) == n_agents:
            energy_penalty = self.weights['energy'] * rotation_costs
            rewards += energy_penalty
        
        # 5. Obstacle avoidance với exponential penalty
        obstacle_distances = coverage_state.get('obstacle_distances', np.ones(n_agents) * 10.0)
        if len(obstacle_distances) == n_agents:
            obstacle_penalty = self.weights['obstacle'] * np.exp(-obstacle_distances)
            rewards += obstacle_penalty
        
        # 6. Collaboration bonus (Tránh overlap, encourage balanced coverage)
        collaboration_bonus = self.weights['collaboration'] * self._compute_coverage_balance(coverage_state)
        rewards += collaboration_bonus
        
        return rewards
    
    def _compute_probabilistic_coverage(self, state):
        """Probabilistic coverage model từ bài báo ACDRL"""
        n_agents = state.get('n_tensors', len(state.get('coverage_scores', [])))
        coverage_scores = []
        target_distances = state.get('target_distances', np.ones(n_agents) * 10.0)
        reliable_radius = state.get('reliable_radius', 1.0)
        max_radius = state.get('max_radius', 5.0)
        
        for i in range(n_agents):
            # Probabilistic sensing model
            distance_to_target = target_distances[i] if i < len(target_distances) else max_radius
            if distance_to_target <= reliable_radius:
                prob = 1.0
            elif distance_to_target <= max_radius:
                decay = np.exp(-0.5 * (distance_to_target - reliable_radius))
                prob = decay
            else:
                prob = 0.0
            coverage_scores.append(prob)
        return np.array(coverage_scores)
    
    def _compute_target_tracking(self, state):
        """Improved target tracking với velocity prediction"""
        n_agents = state.get('n_tensors', len(state.get('coverage_scores', [])))
        tracking_scores = []
        target_distances = state.get('target_distances', np.ones(n_agents) * 10.0)
        target_velocity = state.get('target_velocity', np.zeros(2))
        
        for i in range(n_agents):
            distance = target_distances[i] if i < len(target_distances) else 10.0
            # Adaptive decay based on target velocity
            velocity_factor = 1.0 / (1.0 + np.linalg.norm(target_velocity))
            score = np.exp(-distance * velocity_factor)
            tracking_scores.append(score)
        return np.array(tracking_scores)
    
    def _compute_coverage_balance(self, state):
        """Encourage balanced coverage distribution"""
        coverage_scores = state.get('coverage_scores', np.array([0.0]))
        if len(coverage_scores) > 1:
            balance_penalty = np.std(coverage_scores)  # Penalize imbalance
            return -balance_penalty
        return 0.0