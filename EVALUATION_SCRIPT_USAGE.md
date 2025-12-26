# Evaluation Script Usage Guide

## Overview

The `evaluate_checkpoint.py` script provides a comprehensive evaluation utility for QPLEX checkpoints with statistical analysis, outlier removal, and visualization capabilities.

## Features

1. **Single Checkpoint Evaluation**: Evaluate a single checkpoint with robust statistical analysis
2. **Two Checkpoint Comparison**: Compare two checkpoints with statistical significance testing
3. **Custom Evaluation Configuration**: Override config file parameters via command-line
4. **Visualization**: Generate distribution plots, confidence intervals, and comparison charts

## Installation

Ensure you have all required dependencies:

```bash
pip install numpy scipy matplotlib seaborn pyyaml torch
```

## Usage Examples

### 1. Evaluate a Single Checkpoint

Evaluate a single checkpoint using default settings from config file:

```bash
python evaluate_checkpoint.py \
    --checkpoint models/qplex/qplex_model_40000.pth \
    --config configs/qplex_4v4_9.yaml
```

With visualization:

```bash
python evaluate_checkpoint.py \
    --checkpoint models/qplex/qplex_model_40000.pth \
    --config configs/qplex_4v4_9.yaml \
    --visualize
```

### 2. Compare Two Checkpoints

Compare two checkpoints to see which performs better:

```bash
python evaluate_checkpoint.py \
    --checkpoint1 models/qplex/qplex_model_40000.pth \
    --checkpoint2 models/qplex/qplex_model_80000.pth \
    --config configs/qplex_4v4_9.yaml \
    --visualize
```

### 3. Custom Evaluation Parameters

Override config file parameters:

```bash
python evaluate_checkpoint.py \
    --checkpoint models/qplex/qplex_model_40000.pth \
    --config configs/qplex_4v4_9.yaml \
    --n-eval-runs 10 \
    --n-episodes-per-run 500 \
    --n-warmup-episodes 20 \
    --visualize
```

### 4. Specify Output Directory

Save results to a specific directory:

```bash
python evaluate_checkpoint.py \
    --checkpoint models/qplex/qplex_model_40000.pth \
    --config configs/qplex_4v4_9.yaml \
    --output-dir ./my_evaluation_results \
    --visualize
```

### 5. Use Specific Device

Force CPU or CUDA:

```bash
# Use CPU
python evaluate_checkpoint.py \
    --checkpoint models/qplex/qplex_model_40000.pth \
    --config configs/qplex_4v4_9.yaml \
    --device cpu

# Use CUDA
python evaluate_checkpoint.py \
    --checkpoint models/qplex/qplex_model_40000.pth \
    --config configs/qplex_4v4_9.yaml \
    --device cuda
```

## Command-Line Arguments

### Required Arguments

- `--config`: Path to YAML configuration file (required)
- `--checkpoint`: Path to checkpoint file (for single evaluation)
- `--checkpoint1`, `--checkpoint2`: Paths to two checkpoints (for comparison)

### Optional Arguments

#### Evaluation Parameters (override config file)

- `--n-eval-runs`: Number of evaluation runs (default: from config)
- `--n-episodes-per-run`: Number of episodes per run (default: from config)
- `--n-warmup-episodes`: Number of warmup episodes (default: from config)
- `--remove-outliers`: Whether to remove outliers (default: from config)

#### Output and Visualization

- `--output-dir`: Directory to save results (default: auto-generated with timestamp)
- `--visualize`: Generate visualization plots (flag, no value needed)

#### Device

- `--device`: Device to use - 'cpu' or 'cuda' (default: auto-detect)

## Output Files

The script generates the following files in the output directory:

### JSON Results

- `evaluation_results_<timestep>_<timestamp>.json`: Complete evaluation results in JSON format
  - Configuration used
  - Group 1 and Group 2 results
  - Statistical metrics (mean, std, confidence intervals, CV, etc.)
  - Comparison results (if comparing two checkpoints)

### CSV Raw Data

- `evaluation_raw_data_<timestep>_<timestamp>.csv`: Raw episode data
  - All episode rewards for both groups
  - All episode lengths
  - All coverage rates
  - All transport rates

### Visualization Plots (if --visualize is used)

#### Single Checkpoint Evaluation

- `single_checkpoint_distributions.png`: Distribution plots
  - Reward histogram
  - Coverage histogram
  - Reward box plot
  - Statistics comparison bar chart

- `single_checkpoint_confidence_intervals.png`: Confidence interval plots
  - Reward with 95% CI
  - Coverage with 95% CI
  - Transport with 95% CI

#### Two Checkpoint Comparison

- `comparison_distributions.png`: Distribution plots for both checkpoints
- `comparison_confidence_intervals.png`: Confidence intervals for both checkpoints
- `comparison_summary.png`: Comparison summary
  - Reward difference bar chart
  - Effect size (Cohen's d) with interpretation

## Understanding the Results

### Statistical Metrics

- **Mean**: Average value across all episodes
- **Std**: Standard deviation (measure of variability)
- **CI (Confidence Interval)**: 95% confidence interval for the mean
- **CV (Coefficient of Variation)**: Std/Mean, measures relative variability
- **Stability Score**: 1 - CV, higher is more stable (0-1 range)
- **Min/Max**: Minimum and maximum values
- **Median**: Middle value (50th percentile)
- **Q1/Q3**: 25th and 75th percentiles

### Comparison Metrics

- **Reward Difference**: Absolute difference in mean rewards (Checkpoint1 - Checkpoint2)
- **Reward Difference Percentage**: Percentage difference relative to Checkpoint2
- **P-value**: Statistical significance (p < 0.05 means significant difference)
- **Effect Size (Cohen's d)**: Magnitude of difference
  - |d| < 0.2: Negligible
  - 0.2 ≤ |d| < 0.5: Small
  - 0.5 ≤ |d| < 0.8: Medium
  - |d| ≥ 0.8: Large

### Outlier Removal

The script can automatically remove statistical outliers using:

- **IQR Method** (default): Removes values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- **Z-score Method**: Removes values with |z-score| > threshold

This helps reduce the impact of anomalous episodes on the final metrics.

## Configuration File

The script reads evaluation parameters from the YAML config file under the `evaluation` section:

```yaml
evaluation:
  n_eval_runs: 5
  n_episodes_per_run: 400
  n_warmup_episodes: 10
  batch_size: 50
  remove_outliers: true
  outlier_method: 'iqr'
  outlier_threshold: 1.5
  confidence_level: 0.95
  seeds: [42, 142, 242, 342, 442]
  n_groups: 2
```

Command-line arguments override these values.

## Tips for Best Results

1. **Use Multiple Runs**: Set `n_eval_runs` to 5-10 for stable results
2. **Balance Episodes**: Use 200-500 episodes per run (total ~2000 episodes)
3. **Enable Outlier Removal**: Helps reduce variance from anomalous episodes
4. **Use Warm-up**: Set 5-20 warm-up episodes to stabilize RNN states
5. **Visualize Results**: Always use `--visualize` to inspect distributions
6. **Check Significance**: When comparing, check p-value and effect size

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### CUDA Out of Memory

If you run out of GPU memory, use CPU:

```bash
python evaluate_checkpoint.py ... --device cpu
```

### Checkpoint Loading Fails

Ensure the checkpoint file exists and matches the config:

```bash
ls -lh models/qplex/qplex_model_40000.pth
```

### Environment Errors

Ensure the MATE environment config file exists:

```bash
ls -lh mate/assets/MATE-4v4-9.yaml
```

## Example Workflow

Here's a complete workflow for evaluating training progress:

```bash
# 1. Evaluate checkpoint at 40k steps
python evaluate_checkpoint.py \
    --checkpoint models/qplex/qplex_model_40000.pth \
    --config configs/qplex_4v4_9.yaml \
    --output-dir ./eval_40k \
    --visualize

# 2. Evaluate checkpoint at 80k steps
python evaluate_checkpoint.py \
    --checkpoint models/qplex/qplex_model_80000.pth \
    --config configs/qplex_4v4_9.yaml \
    --output-dir ./eval_80k \
    --visualize

# 3. Compare the two checkpoints
python evaluate_checkpoint.py \
    --checkpoint1 models/qplex/qplex_model_40000.pth \
    --checkpoint2 models/qplex/qplex_model_80000.pth \
    --config configs/qplex_4v4_9.yaml \
    --output-dir ./eval_comparison \
    --visualize

# 4. Review results
cat ./eval_comparison/evaluation_results_*.json
```

## Integration with Training

You can also use this script to evaluate checkpoints during training by calling it from a shell script:

```bash
#!/bin/bash
# evaluate_all_checkpoints.sh

for checkpoint in models/qplex/qplex_model_*.pth; do
    echo "Evaluating $checkpoint"
    python evaluate_checkpoint.py \
        --checkpoint "$checkpoint" \
        --config configs/qplex_4v4_9.yaml \
        --output-dir "./eval_$(basename $checkpoint .pth)" \
        --visualize
done
```

## References

- Requirements: `.kiro/specs/improved-evaluation/requirements.md`
- Design: `.kiro/specs/improved-evaluation/design.md`
- Implementation: `evaluation_utils.py`
