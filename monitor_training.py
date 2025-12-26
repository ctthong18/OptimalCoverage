"""Monitor training progress for MAStAC/Bayesian training."""

import json
import os
import sys
from pathlib import Path

def monitor_training(log_dir="logs/mastac"):
    """Monitor training progress from log files."""
    
    summary_file = Path(log_dir) / "eval_summary.jsonl"
    
    if not summary_file.exists():
        print(f"No evaluation summary found at {summary_file}")
        print("Training may not have reached first evaluation yet.")
        return
    
    print("="*80)
    print(f"TRAINING PROGRESS MONITOR - {log_dir}")
    print("="*80)
    
    # Read all evaluation results
    results = []
    with open(summary_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if not results:
        print("No evaluation results yet.")
        return
    
    # Display results
    print(f"\nTotal evaluations: {len(results)}")
    print("\n" + "-"*80)
    print(f"{'Timestep':<12} {'Reward':<15} {'Coverage':<15} {'Length':<10}")
    print("-"*80)
    
    best_coverage = 0
    best_timestep = 0
    
    for res in results:
        timestep = res.get('timestep', 0)
        reward = res.get('mean_episode_reward', 0)
        reward_std = res.get('std_episode_reward', 0)
        coverage = res.get('mean_coverage_rate', 0)
        coverage_std = res.get('std_coverage_rate', 0)
        length = res.get('mean_episode_length', 0)
        
        print(f"{timestep:<12} {reward:>6.2f}±{reward_std:<5.2f} {coverage:>6.4f}±{coverage_std:<5.4f} {length:>8.1f}")
        
        if coverage > best_coverage:
            best_coverage = coverage
            best_timestep = timestep
    
    print("-"*80)
    print(f"\nBest Coverage: {best_coverage:.4f} at timestep {best_timestep}")
    print("="*80)
    
    # Check for latest model
    model_dir = Path(log_dir.replace('logs', 'models'))
    if model_dir.exists():
        models = list(model_dir.glob("mastac_model_*.pth"))
        if models:
            latest_model = max(models, key=lambda p: int(p.stem.split('_')[-1]))
            print(f"\nLatest model: {latest_model}")
    
    print()

if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/mastac"
    monitor_training(log_dir)
