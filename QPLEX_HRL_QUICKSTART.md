# QPLEX HRL Quick Start Guide

## 🚀 Fast Training with Ray RLlib (Recommended)

This approach matches MATE HRL examples exactly - **fast, distributed, and efficient**.

### Why Ray RLlib is Faster:

1. **Distributed Rollouts**: Multiple workers collect experience in parallel
2. **Efficient Evaluation**: Ray Tune handles evaluation automatically
3. **Minimal Overhead**: No custom evaluation loops, just pure training
4. **Optimized Logging**: Built-in metrics tracking without slowdown

### Quick Start:

```bash
# Basic training (exactly like MATE HRL)
python -m runners.train_qplex_hrl_ray --config configs/qplex_hrl_4v4_9_ray.yaml

# With more workers for faster training
python -m runners.train_qplex_hrl_ray --config configs/qplex_hrl_4v4_9_ray.yaml --num-workers 8

# With GPU acceleration
python -m runners.train_qplex_hrl_ray --config configs/qplex_hrl_4v4_9_ray.yaml --num-gpus 0.5

# Full power (8 workers + GPU)
python -m runners.train_qplex_hrl_ray --config configs/qplex_hrl_4v4_9_ray.yaml --num-workers 8 --num-gpus 0.5
```

### Configuration:

The Ray config (`configs/qplex_hrl_4v4_9_ray.yaml`) is optimized for speed:

```yaml
# Key settings for fast training
ray:
  num_workers: 4              # Parallel rollout workers
  num_envs_per_worker: 8      # Environments per worker
  num_gpus: 0.25             # GPU allocation

training:
  total_timesteps: 10000000   # 10M timesteps
  train_batch_size: 1024      # Large batch for efficiency
  buffer_size: 2000           # Efficient buffer size
```

### Expected Performance:

- **Training Speed**: ~5-10x faster than standalone version
- **CPU Usage**: Efficiently uses all available cores
- **Memory**: Distributed across workers
- **Evaluation**: Automatic, no slowdown

### Monitoring:

Ray automatically logs to:
- **TensorBoard**: `logs/qplex_hrl_ray/ray_results/`
- **Console**: Real-time metrics every iteration
- **W&B** (optional): Enable in config

Key metrics to watch:
- `episode_reward_mean`: Average episode reward
- `custom_metrics/coverage_rate_mean`: **PRIMARY METRIC**
- `custom_metrics/num_selected_targets_mean`: Target selection efficiency
- `custom_metrics/invalid_target_selection_rate_mean`: Selection quality

### Checkpoints:

Ray saves checkpoints automatically:
- Location: `logs/qplex_hrl_ray/ray_results/HRL-QPLEX/*/checkpoint_*/`
- Frequency: Every 20 iterations
- Best checkpoint: Tracked by `episode_reward_mean`

Resume training:
```bash
# Ray automatically resumes from latest checkpoint if you use same local-dir
python -m runners.train_qplex_hrl_ray --config configs/qplex_hrl_4v4_9_ray.yaml --local-dir logs/qplex_hrl_ray/ray_results
```

## 🐌 Standalone Training (Slower, but Simpler)

Use this if you don't want Ray dependencies or need simpler debugging:

```bash
# Basic standalone training
python runners/train_qplex_hrl.py --config configs/qplex_hrl_4v4_9.yaml

# Resume from checkpoint
python runners/train_qplex_hrl.py --config configs/qplex_hrl_4v4_9.yaml --resume models/qplex_hrl/qplex_hrl_40000.pth
```

**Note**: Standalone version is ~5-10x slower because:
- No distributed rollouts
- Custom evaluation loops
- Single-threaded execution
- More logging overhead

## 📊 Comparison:

| Feature | Ray RLlib | Standalone |
|---------|-----------|------------|
| Speed | ⚡⚡⚡⚡⚡ (5-10x faster) | ⚡ (baseline) |
| Setup | Requires Ray | Simple |
| Debugging | Harder | Easier |
| Scalability | Excellent | Limited |
| Evaluation | Automatic | Manual |
| Logging | Built-in | Custom |
| **Recommended** | ✅ Yes | For debugging only |

## 🎯 Optimization Tips:

### For Maximum Speed (Ray RLlib):

1. **Increase workers**: `--num-workers 16` (if you have CPU cores)
2. **Use GPU**: `--num-gpus 1.0` (if available)
3. **Larger batches**: Edit config `train_batch_size: 2048`
4. **More envs per worker**: `--num-envs-per-worker 16`

### For Better Coverage:

1. **Increase frame_skip**: `frame_skip: 10` (slower but more strategic)
2. **Higher entropy weight**: `selection_entropy_weight: 0.02`
3. **More exploration**: `epsilon_end: 0.05`

### For Debugging:

1. Use standalone version
2. Reduce timesteps: `total_timesteps: 10000`
3. Enable debug logging: `--log-level DEBUG`
4. Render environment: Set `render_mode: "human"` in config

## 📈 Expected Results:

After training with Ray RLlib:

- **Coverage Rate**: Should reach 0.6-0.8 (60-80% coverage)
- **Training Time**: ~2-4 hours on 8 workers (vs 10-20 hours standalone)
- **Convergence**: Around 5-8M timesteps
- **Selection Efficiency**: Should improve from ~0.3 to ~0.7

## 🔧 Troubleshooting:

### Ray won't start:
```bash
# Check Ray installation
pip install ray[rllib]

# If still issues, try:
ray stop  # Stop any existing Ray processes
ray start --head  # Start fresh
```

### Out of memory:
- Reduce `num_workers`
- Reduce `num_envs_per_worker`
- Reduce `buffer_size`

### Training too slow:
- Increase `num_workers` (if you have CPU cores)
- Reduce `eval_interval` (evaluate less frequently)
- Disable rendering: `render_mode: null`

### Coverage not improving:
- Check `custom_metrics/coverage_rate_mean` in logs
- Increase `coverage_reward_weight` in config
- Increase `frame_skip` for more strategic selection
- Train longer (coverage improves slowly)

## 📚 Next Steps:

1. **Start training**: Use Ray RLlib command above
2. **Monitor progress**: Watch TensorBoard or console
3. **Evaluate results**: Check coverage_rate metric
4. **Compare with baselines**: Run qplex_dev for comparison
5. **Tune hyperparameters**: Adjust config based on results

## 🎉 Success Criteria:

Your training is successful if:
- ✅ Coverage rate > 0.6 (60%)
- ✅ Selection efficiency > 0.6
- ✅ Invalid selection rate < 0.3
- ✅ Episode reward increasing over time
- ✅ Training completes without errors

Good luck! 🚀