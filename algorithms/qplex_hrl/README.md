# QPLEX HRL - Hierarchical Reinforcement Learning with QPLEX

## Overview

QPLEX HRL combines hierarchical target selection with advanced Q-learning for multi-agent camera tracking. This implementation integrates:

1. **Hierarchical Target Selection** (inspired by MATE HRL examples)
   - Fast selector: Frame-by-frame target selection
   - Slow selector: Strategic target prioritization
   - Multi-timescale learning with frame skipping

2. **Advanced Q-Networks** (from qplex_dev)
   - AttentionRNN: Attention mechanism for target focus
   - HierarchicalRNN: Multi-timescale temporal processing
   - Adaptive mixing network for coordination

3. **Coverage Optimization**
   - Coverage-aware reward shaping
   - Selection entropy regularization
   - Exploration bonus for better coverage

## Architecture

```
QPLEX HRL Model
├── Hierarchical Target Selector (per agent)
│   ├── Fast Selector (every frame)
│   ├── Slow Selector (every frame_skip frames)
│   └── Attention Mechanism (target importance)
├── Enhanced Q-Networks (per agent)
│   ├── AttentionRNN / HierarchicalRNN
│   └── Augmented with target selections
└── Adaptive Mixing Network
    ├── Complexity Estimator
    ├── Simple Mixer (low complexity)
    ├── Complex Mixer (high complexity)
    └── Adaptive Gating
```

## Key Features

### 1. Hierarchical Target Selection
- **Multi-timescale**: Fast (every frame) + Slow (every 5 frames)
- **Attention-based**: Focus on important targets
- **Coverage-aware**: Encourages diverse target selection

### 2. Advanced Networks
- **AttentionRNN**: Better temporal modeling with attention
- **Adaptive Mixing**: Adjusts complexity based on task difficulty
- **State Encoder**: Enhanced global state processing

### 3. Coverage Optimization
- **Primary Metric**: mean_coverage_rate
- **Reward Shaping**: Coverage improvement bonus
- **Exploration**: Coverage-aware exploration bonus

## Usage

### Training Options

#### Option 1: Ray RLlib Training (Recommended - matches MATE HRL examples)

```bash
# Basic Ray RLlib training (similar to MATE HRL)
python runners/train_qplex_hrl_ray.py --config configs/qplex_hrl_4v4_9_ray.yaml

# With custom resources
python runners/train_qplex_hrl_ray.py --config configs/qplex_hrl_4v4_9_ray.yaml --num-workers 8 --num-gpus 0.5

# Resume from Ray checkpoint
python runners/train_qplex_hrl_ray.py --config configs/qplex_hrl_4v4_9_ray.yaml --resume path/to/checkpoint

# With W&B logging
python runners/train_qplex_hrl_ray.py --config configs/qplex_hrl_4v4_9_ray.yaml --project qplex-hrl --group experiment-1
```

#### Option 2: Standalone Training

```bash
# Basic standalone training
python runners/train_qplex_hrl.py --config configs/qplex_hrl_4v4_9.yaml

# Resume from checkpoint
python runners/train_qplex_hrl.py --config configs/qplex_hrl_4v4_9.yaml --resume models/qplex_hrl/qplex_hrl_40000.pth

# With debug logging
python runners/train_qplex_hrl.py --config configs/qplex_hrl_4v4_9.yaml --log-level DEBUG
```

### Configuration

#### Ray RLlib Configuration (configs/qplex_hrl_4v4_9_ray.yaml)

Key HRL parameters for Ray RLlib training:

```yaml
network:
  hrl:
    frame_skip: 5                    # Multi-timescale interval (matches MATE HRL)
    multi_selection: true            # Enable multi-target selection
    coverage_reward_weight: 0.5      # Coverage reward shaping
    selection_entropy_weight: 0.01   # Encourage diverse selection
    
    reward_coefficients:
      coverage_rate: 1.0             # Primary optimization target

ray:
  num_workers: 4                     # Parallel rollout workers
  num_envs_per_worker: 8            # Environments per worker
  num_gpus: 0.25                    # GPU allocation
```

#### Standalone Configuration (configs/qplex_hrl_4v4_9.yaml)

Key HRL parameters for standalone training:

```yaml
network:
  hrl:
    frame_skip: 5                    # Multi-timescale interval
    multi_selection: true            # Enable multi-target selection
    coverage_reward_weight: 0.5      # Coverage reward shaping
    selection_entropy_weight: 0.01   # Encourage diverse selection
    
    reward_coefficients:
      coverage_rate: 1.0             # Primary optimization target
```

### Evaluation

The training script automatically evaluates with hierarchical metrics:
- **Coverage Rate** (primary metric)
- **Selection Efficiency** (valid selections / total selections)
- **Transport Rate**
- **Episode Rewards**

Evaluation results are saved to:
- `logs/qplex_hrl/eval_results_{timestep}.json`
- `logs/qplex_hrl/final_eval_results.json`

## Comparison with Other Approaches

| Feature | QPLEX (standard) | QPLEX Dev | QPLEX HRL |
|---------|------------------|-----------|-----------|
| Network | Basic MLP | AttentionRNN | AttentionRNN + HRL |
| Target Selection | Implicit | Implicit | Explicit Hierarchical |
| Timescales | Single | Single | Multi (fast + slow) |
| Coverage Focus | No | Partial | Yes (primary) |
| Frame Skip | No | No | Yes (5 frames) |
| Complexity | Low | Medium | High |

## Expected Performance

Based on the architecture and HRL approach:

- **Coverage Rate**: Expected to achieve **higher coverage** than standard QPLEX
- **Selection Efficiency**: Should improve over time as hierarchical selection learns
- **Training Time**: Slower than standard QPLEX due to hierarchical complexity
- **Convergence**: May need 60k-100k timesteps for full convergence

## Monitoring Training

Key metrics to monitor:

1. **mean_coverage_rate** (PRIMARY): Should increase over training
2. **best_coverage**: Track best coverage achieved
3. **selection_entropy**: Should be moderate (not too low/high)
4. **mean_selection_efficiency**: Should increase (more valid selections)
5. **mean_loss**: Should decrease and stabilize

## Troubleshooting

### Low Coverage Rate
- Increase `coverage_reward_weight`
- Increase `exploration_bonus`
- Decrease `frame_skip` for more frequent selection updates

### Unstable Training
- Decrease `learning_rate`
- Increase `batch_size`
- Decrease `selection_entropy_weight`

### Slow Convergence
- Increase `learning_rate` slightly
- Decrease `epsilon_decay` for more exploration
- Increase `train_freq` for more frequent updates

## Files

- `model.py`: HRL model architecture (target selector + Q-networks + mixing)
- `agent.py`: HRL agent with hierarchical action selection
- `learner.py`: Training orchestrator with coverage optimization
- `rllib_policy.py`: Ray RLlib policy integration (matches MATE HRL)
- `README.md`: This file

## Training Scripts

- `runners/train_qplex_hrl_ray.py`: Ray RLlib training (recommended, matches MATE HRL)
- `runners/train_qplex_hrl.py`: Standalone training

## Configuration Files

- `configs/qplex_hrl_4v4_9_ray.yaml`: Ray RLlib configuration
- `configs/qplex_hrl_4v4_9.yaml`: Standalone configuration

## References

- MATE HRL Examples: `mate/examples/hrl/qplex/`
- QPLEX Dev: `algorithms/qplex_dev/`
- Evaluation Utils: `evaluation_utils.py`

## Citation

If you use this code, please cite:
- QPLEX paper
- MATE environment
- Your research paper (if applicable)