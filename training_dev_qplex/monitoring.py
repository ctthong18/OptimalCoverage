import numpy as np


def setup_comprehensive_monitoring(learner, env):
    """Comprehensive monitoring system with mean_coverage_rate as PRIMARY METRIC"""
    
    metrics = {
        'coverage_metrics': [],
        'training_metrics': [], 
        'network_metrics': [],
        'exploration_metrics': []
    }
    
    def log_metrics(episode, eval_reward, coverage_state):
        """
        Log metrics with focus on mean_coverage_rate optimization.
        
        Args:
            episode: Current episode number
            eval_reward: Evaluation reward
            coverage_state: Dict containing coverage metrics, should include:
                - target_coverage_rate: mean_coverage_rate (PRIMARY METRIC)
                - coverage_scores: individual coverage scores
                - obstacle_violations: number of obstacle violations
        """
        # PRIMARY METRIC: Mean coverage rate
        target_coverage = coverage_state.get('target_coverage_rate', 0.0)
        
        # Individual coverage metrics
        coverage_scores = coverage_state.get('coverage_scores', [0.0])
        coverage_rate = np.mean(coverage_scores) if len(coverage_scores) > 0 else 0.0
        obstacle_violations = coverage_state.get('obstacle_violations', 0)
        
        # Training metrics
        stats = learner.get_training_stats()
        td_error = stats.get('mean_td_error', stats.get('td_error', 0))
        q_values = stats.get('mean_q_values', stats.get('q_values', 0))
        
        metrics['coverage_metrics'].append({
            'episode': episode,
            'mean_coverage_rate': target_coverage,  # PRIMARY METRIC
            'coverage_rate': coverage_rate,
            'target_coverage': target_coverage,
            'obstacle_violations': obstacle_violations
        })
        
        metrics['training_metrics'].append({
            'episode': episode,
            'eval_reward': eval_reward,
            'td_error': td_error,
            'q_values': q_values,
            'mean_coverage_rate': target_coverage  # Include in training metrics too
        })
        
        # Log every 100 episodes with PRIMARY METRIC highlighted
        if episode % 100 == 0:
            print(f"[PRIMARY] Mean Coverage Rate: {target_coverage:.4f} | "
                  f"Individual Coverage: {coverage_rate:.3f} | "
                  f"TD Error: {td_error:.3f} | "
                  f"Q-values: {q_values:.3f}")
    
    return log_metrics