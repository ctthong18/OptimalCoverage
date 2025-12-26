#!/usr/bin/env python3
"""
Evaluation utility script for QPLEX checkpoints.

This script allows you to:
1. Evaluate a single checkpoint
2. Compare two checkpoints
3. Use custom evaluation configurations
4. Generate visualizations and plots
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from algorithms.qplex.learner import QPLEXLearner
from mate import MultiAgentTracking
from evaluation_utils import (
    EvaluationConfig,
    ImprovedEvaluator,
    EvaluationLogger,
    AggregatedResults,
    ComparisonResults
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_learner_from_config(config: Dict[str, Any], device: str = 'cpu') -> QPLEXLearner:
    """Create a QPLEXLearner instance from configuration."""
    env_config_file = config['env']['config_file']
    env = MultiAgentTracking(config_file=env_config_file)
    
    n_agents = env.num_cameras
    obs_shape = env.observation_space[0].shape
    n_actions = env.action_space[0].n
    state_shape = env.state().shape
    
    learner = QPLEXLearner(
        n_agents=n_agents,
        obs_shape=obs_shape,
        n_actions=n_actions,
        state_shape=state_shape,
        config=config,
        device=device
    )
    
    env.close()
    return learner


def load_checkpoint(checkpoint_path: str, config: Dict[str, Any], device: str = 'cpu') -> QPLEXLearner:
    """Load a checkpoint and create a learner."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    learner = create_learner_from_config(config, device)
    learner.load(checkpoint_path)
    print(f"Checkpoint loaded successfully")
    return learner


def create_evaluation_config(config: Dict[str, Any], args: argparse.Namespace) -> EvaluationConfig:
    """Create EvaluationConfig from YAML config and command-line arguments."""
    eval_cfg = config.get('evaluation', {})
    
    n_eval_runs = args.n_eval_runs if args.n_eval_runs is not None else eval_cfg.get('n_eval_runs', 5)
    n_episodes_per_run = args.n_episodes_per_run if args.n_episodes_per_run is not None else eval_cfg.get('n_episodes_per_run', 400)
    n_warmup_episodes = args.n_warmup_episodes if args.n_warmup_episodes is not None else eval_cfg.get('n_warmup_episodes', 10)
    batch_size = eval_cfg.get('batch_size', 50)
    remove_outliers = args.remove_outliers if args.remove_outliers is not None else eval_cfg.get('remove_outliers', True)
    outlier_method = eval_cfg.get('outlier_method', 'iqr')
    outlier_threshold = eval_cfg.get('outlier_threshold', 1.5)
    confidence_level = eval_cfg.get('confidence_level', 0.95)
    seeds = eval_cfg.get('seeds', None)
    
    eval_config = EvaluationConfig(
        n_eval_runs=n_eval_runs,
        n_episodes_per_run=n_episodes_per_run,
        n_warmup_episodes=n_warmup_episodes,
        batch_size=batch_size,
        remove_outliers=remove_outliers,
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
        confidence_level=confidence_level,
        seeds=seeds
    )
    
    return eval_config


def evaluate_single_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    eval_config: EvaluationConfig,
    output_dir: str,
    device: str = 'cpu',
    visualize: bool = False
) -> Dict[str, Any]:
    """Evaluate a single checkpoint."""
    print("\n" + "=" * 80)
    print("SINGLE CHECKPOINT EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    learner = load_checkpoint(checkpoint_path, config, device)
    
    env_config_file = config['env']['config_file']
    env = MultiAgentTracking(config_file=env_config_file)
    
    evaluator = ImprovedEvaluator(eval_config)
    learners = [learner, learner]
    
    results = evaluator.evaluate(
        learners=learners,
        env=env,
        timestep=0,
        log_dir=output_dir
    )
    
    if visualize:
        print("\nGenerating visualizations...")
        visualize_results(results, output_dir, "single_checkpoint")
    
    env.close()
    return results


def compare_two_checkpoints(
    checkpoint1_path: str,
    checkpoint2_path: str,
    config: Dict[str, Any],
    eval_config: EvaluationConfig,
    output_dir: str,
    device: str = 'cpu',
    visualize: bool = False
) -> Dict[str, Any]:
    """Compare two checkpoints."""
    print("\n" + "=" * 80)
    print("TWO CHECKPOINT COMPARISON")
    print("=" * 80)
    print(f"Checkpoint 1: {checkpoint1_path}")
    print(f"Checkpoint 2: {checkpoint2_path}")
    print(f"Output directory: {output_dir}")
    
    learner1 = load_checkpoint(checkpoint1_path, config, device)
    learner2 = load_checkpoint(checkpoint2_path, config, device)
    
    env_config_file = config['env']['config_file']
    env = MultiAgentTracking(config_file=env_config_file)
    
    evaluator = ImprovedEvaluator(eval_config)
    learners = [learner1, learner2]
    
    results = evaluator.evaluate(
        learners=learners,
        env=env,
        timestep=0,
        log_dir=output_dir
    )
    
    if visualize:
        print("\nGenerating visualizations...")
        visualize_comparison(results, output_dir, checkpoint1_path, checkpoint2_path)
    
    env.close()
    return results


def visualize_results(results: Dict[str, Any], output_dir: str, prefix: str = "evaluation") -> None:
    """Generate visualization plots for evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    group1_results = results.get('group1_results', {})
    group2_results = results.get('group2_results', {})
    
    # Plot 1: Distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    ax = axes[0, 0]
    if 'all_rewards' in group1_results:
        ax.hist(group1_results['all_rewards'], bins=50, alpha=0.7, label='Group 0', color='blue')
    if 'all_rewards' in group2_results:
        ax.hist(group2_results['all_rewards'], bins=50, alpha=0.7, label='Group 1', color='orange')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    if 'all_coverages' in group1_results:
        ax.hist(group1_results['all_coverages'], bins=50, alpha=0.7, label='Group 0', color='blue')
    if 'all_coverages' in group2_results:
        ax.hist(group2_results['all_coverages'], bins=50, alpha=0.7, label='Group 1', color='orange')
    ax.set_xlabel('Coverage Rate')
    ax.set_ylabel('Frequency')
    ax.set_title('Coverage Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    data_to_plot = []
    labels = []
    if 'all_rewards' in group1_results:
        data_to_plot.append(group1_results['all_rewards'])
        labels.append('Group 0')
    if 'all_rewards' in group2_results:
        data_to_plot.append(group2_results['all_rewards'])
        labels.append('Group 1')
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightsalmon']):
            patch.set_facecolor(color)
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward Box Plot')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    metrics = ['mean', 'std', 'cv']
    group1_values = []
    group2_values = []
    
    if 'reward_stats' in group1_results:
        for metric in metrics:
            group1_values.append(group1_results['reward_stats'].get(metric, 0))
    if 'reward_stats' in group2_results:
        for metric in metrics:
            group2_values.append(group2_results['reward_stats'].get(metric, 0))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    if group1_values:
        ax.bar(x - width/2, group1_values, width, label='Group 0', color='blue', alpha=0.7)
    if group2_values:
        ax.bar(x + width/2, group2_values, width, label='Group 1', color='orange', alpha=0.7)
    
    ax.set_ylabel('Value')
    ax.set_title('Reward Statistics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{prefix}_distributions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plot to: {plot_path}")
    plt.close()
    
    # Plot 2: Confidence intervals
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = ['reward_stats', 'coverage_stats', 'transport_stats']
    metric_names = ['Reward', 'Coverage', 'Transport']
    
    x_pos = np.arange(len(metric_names))
    width = 0.35
    
    for i, (metric_key, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        if metric_key in group1_results:
            stats = group1_results[metric_key]
            mean = stats.get('mean', 0)
            ci_lower = stats.get('ci_lower', mean)
            ci_upper = stats.get('ci_upper', mean)
            error = [[mean - ci_lower], [ci_upper - mean]]
            
            ax.errorbar(x_pos[i] - width/2, mean, yerr=error, fmt='o', 
                       capsize=5, capthick=2, label='Group 0' if i == 0 else '', 
                       color='blue', markersize=8)
        
        if metric_key in group2_results:
            stats = group2_results[metric_key]
            mean = stats.get('mean', 0)
            ci_lower = stats.get('ci_lower', mean)
            ci_upper = stats.get('ci_upper', mean)
            error = [[mean - ci_lower], [ci_upper - mean]]
            
            ax.errorbar(x_pos[i] + width/2, mean, yerr=error, fmt='s', 
                       capsize=5, capthick=2, label='Group 1' if i == 0 else '', 
                       color='orange', markersize=8)
    
    ax.set_ylabel('Value')
    ax.set_title('Metrics with 95% Confidence Intervals')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{prefix}_confidence_intervals.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved confidence interval plot to: {plot_path}")
    plt.close()


def visualize_comparison(
    results: Dict[str, Any],
    output_dir: str,
    checkpoint1_path: str,
    checkpoint2_path: str
) -> None:
    """Generate comparison visualization plots."""
    visualize_results(results, output_dir, "comparison")
    
    comparison = results.get('comparison', {})
    
    if not comparison:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    reward_diff = comparison.get('reward_difference', 0)
    reward_diff_pct = comparison.get('reward_difference_percentage', 0)
    p_value = comparison.get('p_value', 1.0)
    
    colors = ['green' if reward_diff > 0 else 'red']
    ax.bar(['Checkpoint 1 vs 2'], [reward_diff], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Reward Difference')
    ax.set_title(f'Reward Difference\n({reward_diff_pct:+.1f}%, p={p_value:.4f})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    effect_size = comparison.get('effect_size', 0)
    
    if abs(effect_size) < 0.2:
        interpretation = 'Negligible'
        color = 'gray'
    elif abs(effect_size) < 0.5:
        interpretation = 'Small'
        color = 'yellow'
    elif abs(effect_size) < 0.8:
        interpretation = 'Medium'
        color = 'orange'
    else:
        interpretation = 'Large'
        color = 'red'
    
    ax.bar(['Effect Size'], [abs(effect_size)], color=color, alpha=0.7)
    ax.set_ylabel("Cohen's d")
    ax.set_title(f"Effect Size: {interpretation}\n(|d| = {abs(effect_size):.3f})")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison summary plot to: {plot_path}")
    plt.close()


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate QPLEX checkpoints with statistical analysis'
    )
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (for single evaluation)')
    parser.add_argument('--checkpoint1', type=str, default=None,
                       help='Path to first checkpoint (for comparison)')
    parser.add_argument('--checkpoint2', type=str, default=None,
                       help='Path to second checkpoint (for comparison)')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    parser.add_argument('--n-eval-runs', type=int, default=None,
                       help='Number of evaluation runs (overrides config)')
    parser.add_argument('--n-episodes-per-run', type=int, default=None,
                       help='Number of episodes per run (overrides config)')
    parser.add_argument('--n-warmup-episodes', type=int, default=None,
                       help='Number of warmup episodes (overrides config)')
    parser.add_argument('--remove-outliers', type=bool, default=None,
                       help='Whether to remove outliers (overrides config)')
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save results (default: auto-generated)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu or cuda, default: auto-detect)')
    
    args = parser.parse_args()
    
    if args.checkpoint is None and (args.checkpoint1 is None or args.checkpoint2 is None):
        parser.error('Must provide either --checkpoint or both --checkpoint1 and --checkpoint2')
    
    if args.checkpoint is not None and (args.checkpoint1 is not None or args.checkpoint2 is not None):
        parser.error('Cannot use --checkpoint with --checkpoint1/--checkpoint2')
    
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    eval_config = create_evaluation_config(config, args)
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.checkpoint is not None:
            output_dir = f"./evaluation_results/single_{timestamp}"
        else:
            output_dir = f"./evaluation_results/comparison_{timestamp}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if args.checkpoint is not None:
            results = evaluate_single_checkpoint(
                checkpoint_path=args.checkpoint,
                config=config,
                eval_config=eval_config,
                output_dir=output_dir,
                device=device,
                visualize=args.visualize
            )
        else:
            results = compare_two_checkpoints(
                checkpoint1_path=args.checkpoint1,
                checkpoint2_path=args.checkpoint2,
                config=config,
                eval_config=eval_config,
                output_dir=output_dir,
                device=device,
                visualize=args.visualize
            )
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
