import numpy as np
from scipy import stats
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation.
    
    Attributes:
        n_eval_runs: Number of evaluation runs
        n_episodes_per_run: Number of episodes per run
        n_warmup_episodes: Number of warm-up episodes
        batch_size: Batch size for evaluation
        remove_outliers: Whether to remove outliers
        outlier_method: Method for outlier detection ('iqr' or 'zscore')
        outlier_threshold: Threshold for outlier detection
        confidence_level: Confidence level for confidence intervals
        seeds: Random seeds for each run
    """
    n_eval_runs: int = 5
    n_episodes_per_run: int = 400
    n_warmup_episodes: int = 10
    batch_size: int = 50
    remove_outliers: bool = True
    outlier_method: str = 'iqr'
    outlier_threshold: float = 1.5
    confidence_level: float = 0.95
    seeds: Optional[List[int]] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate n_eval_runs
        if self.n_eval_runs < 1:
            raise ValueError(f"n_eval_runs must be >= 1, got {self.n_eval_runs}")
        
        # Validate n_episodes_per_run
        if self.n_episodes_per_run < 1:
            raise ValueError(f"n_episodes_per_run must be >= 1, got {self.n_episodes_per_run}")
        
        # Validate n_warmup_episodes
        if self.n_warmup_episodes < 0:
            raise ValueError(f"n_warmup_episodes must be >= 0, got {self.n_warmup_episodes}")
        
        # Validate batch_size
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        
        # Validate outlier_method
        if self.outlier_method not in ['iqr', 'zscore']:
            raise ValueError(f"outlier_method must be 'iqr' or 'zscore', got {self.outlier_method}")
        
        # Validate outlier_threshold
        if self.outlier_threshold <= 0:
            raise ValueError(f"outlier_threshold must be > 0, got {self.outlier_threshold}")
        
        # Validate confidence_level
        if not 0 < self.confidence_level < 1:
            raise ValueError(f"confidence_level must be between 0 and 1, got {self.confidence_level}")
        
        # Generate default seeds if not provided
        if self.seeds is None:
            self.seeds = [42 + i * 100 for i in range(self.n_eval_runs)]
        
        # Validate seeds length
        if len(self.seeds) != self.n_eval_runs:
            raise ValueError(f"Length of seeds ({len(self.seeds)}) must match n_eval_runs ({self.n_eval_runs})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GroupResults:
    """
    Results for a single agent group from one evaluation run.
    
    Attributes:
        group_id: ID of the group (0 or 1)
        run_id: ID of the evaluation run
        seed: Random seed used for this run
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        coverage_rates: List of coverage rates
        transport_rates: List of transport rates
        mean_reward: Mean episode reward
        std_reward: Standard deviation of episode rewards
        mean_length: Mean episode length
        mean_coverage: Mean coverage rate
        mean_transport: Mean transport rate
    """
    group_id: int
    run_id: int
    seed: int
    episode_rewards: List[float]
    episode_lengths: List[float]
    coverage_rates: List[float]
    transport_rates: List[float]
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_length: float = 0.0
    mean_coverage: float = 0.0
    mean_transport: float = 0.0
    
    def __post_init__(self):
        """Calculate aggregated metrics from raw data."""
        # Validate group_id
        if self.group_id not in [0, 1]:
            raise ValueError(f"group_id must be 0 or 1, got {self.group_id}")
        
        # Validate run_id
        if self.run_id < 0:
            raise ValueError(f"run_id must be >= 0, got {self.run_id}")
        
        # Validate that all lists have the same length
        lengths = [
            len(self.episode_rewards),
            len(self.episode_lengths),
            len(self.coverage_rates),
            len(self.transport_rates)
        ]
        if len(set(lengths)) > 1:
            raise ValueError(f"All metric lists must have the same length, got {lengths}")
        
        # Calculate aggregated metrics if not already set
        if len(self.episode_rewards) > 0:
            rewards_array = np.array(self.episode_rewards)
            self.mean_reward = float(np.mean(rewards_array))
            self.std_reward = float(np.std(rewards_array, ddof=1) if len(rewards_array) > 1 else 0.0)
            self.mean_length = float(np.mean(self.episode_lengths))
            self.mean_coverage = float(np.mean(self.coverage_rates))
            self.mean_transport = float(np.mean(self.transport_rates))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroupResults':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AggregatedResults:
    """
    Aggregated results from multiple evaluation runs for one group.
    
    Attributes:
        group_id: ID of the group (0 or 1)
        n_runs: Number of evaluation runs
        n_episodes_total: Total number of episodes across all runs
        reward_stats: Statistics for rewards (mean, std, ci_lower, ci_upper, cv, etc.)
        length_stats: Statistics for episode lengths
        coverage_stats: Statistics for coverage rates
        transport_stats: Statistics for transport rates
        all_rewards: Raw rewards from all runs
        all_lengths: Raw lengths from all runs
        all_coverages: Raw coverage rates from all runs
        all_transports: Raw transport rates from all runs
        n_outliers_removed: Number of outliers removed
        outlier_percentage: Percentage of outliers removed
    """
    group_id: int
    n_runs: int
    n_episodes_total: int
    reward_stats: Dict[str, float]
    length_stats: Dict[str, float]
    coverage_stats: Dict[str, float]
    transport_stats: Dict[str, float]
    all_rewards: List[float] = field(default_factory=list)
    all_lengths: List[float] = field(default_factory=list)
    all_coverages: List[float] = field(default_factory=list)
    all_transports: List[float] = field(default_factory=list)
    n_outliers_removed: int = 0
    outlier_percentage: float = 0.0
    
    def __post_init__(self):
        """Validate aggregated results."""
        # Validate group_id
        if self.group_id not in [0, 1]:
            raise ValueError(f"group_id must be 0 or 1, got {self.group_id}")
        
        # Validate n_runs
        if self.n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {self.n_runs}")
        
        # Validate n_episodes_total
        if self.n_episodes_total < 1:
            raise ValueError(f"n_episodes_total must be >= 1, got {self.n_episodes_total}")
        
        # Validate that stats dictionaries contain required keys
        required_keys = ['mean', 'std', 'ci_lower', 'ci_upper']
        for stats_name, stats_dict in [
            ('reward_stats', self.reward_stats),
            ('length_stats', self.length_stats),
            ('coverage_stats', self.coverage_stats),
            ('transport_stats', self.transport_stats)
        ]:
            for key in required_keys:
                if key not in stats_dict:
                    raise ValueError(f"{stats_name} missing required key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AggregatedResults':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ComparisonResults:
    """
    Comparison results between two agent groups.
    
    Attributes:
        group1_results: Aggregated results for group 1
        group2_results: Aggregated results for group 2
        reward_difference: Difference in mean rewards (group1 - group2)
        reward_difference_percentage: Percentage difference in rewards
        statistical_significance: Whether the difference is statistically significant
        p_value: P-value from statistical test (t-test)
        effect_size: Cohen's d effect size
    """
    group1_results: AggregatedResults
    group2_results: AggregatedResults
    reward_difference: float = 0.0
    reward_difference_percentage: float = 0.0
    statistical_significance: bool = False
    p_value: float = 1.0
    effect_size: float = 0.0
    
    def __post_init__(self):
        """Calculate comparison metrics."""
        # Calculate reward difference
        mean1 = self.group1_results.reward_stats['mean']
        mean2 = self.group2_results.reward_stats['mean']
        self.reward_difference = mean1 - mean2
        
        # Calculate percentage difference
        if mean2 != 0:
            self.reward_difference_percentage = (self.reward_difference / abs(mean2)) * 100
        else:
            self.reward_difference_percentage = float('inf') if self.reward_difference != 0 else 0.0
        
        # Perform t-test if we have raw data
        if len(self.group1_results.all_rewards) > 0 and len(self.group2_results.all_rewards) > 0:
            t_stat, self.p_value = stats.ttest_ind(
                self.group1_results.all_rewards,
                self.group2_results.all_rewards
            )
            self.statistical_significance = self.p_value < 0.05
            
            # Calculate Cohen's d effect size
            std1 = self.group1_results.reward_stats['std']
            std2 = self.group2_results.reward_stats['std']
            n1 = len(self.group1_results.all_rewards)
            n2 = len(self.group2_results.all_rewards)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            if pooled_std != 0:
                self.effect_size = self.reward_difference / pooled_std
            else:
                self.effect_size = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'group1_results': self.group1_results.to_dict(),
            'group2_results': self.group2_results.to_dict(),
            'reward_difference': self.reward_difference,
            'reward_difference_percentage': self.reward_difference_percentage,
            'statistical_significance': self.statistical_significance,
            'p_value': self.p_value,
            'effect_size': self.effect_size
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComparisonResults':
        """Create from dictionary."""
        # Convert nested dictionaries back to dataclass instances
        group1_results = AggregatedResults.from_dict(data['group1_results'])
        group2_results = AggregatedResults.from_dict(data['group2_results'])
        
        return cls(
            group1_results=group1_results,
            group2_results=group2_results,
            reward_difference=data.get('reward_difference', 0.0),
            reward_difference_percentage=data.get('reward_difference_percentage', 0.0),
            statistical_significance=data.get('statistical_significance', False),
            p_value=data.get('p_value', 1.0),
            effect_size=data.get('effect_size', 0.0)
        )


class EvaluationError(Exception):
    """Base exception for evaluation errors."""
    pass


class InsufficientDataError(EvaluationError):
    """Raised when there is insufficient data to calculate statistics."""
    pass


class ConfigurationError(EvaluationError):
    """Raised when configuration is invalid."""
    pass


class GroupEvaluator:
    """
    Evaluator for a single agent group.
    
    This class handles evaluation of one group of agents, running multiple episodes
    and collecting metrics such as episode rewards, lengths, coverage rates, and
    transport rates.
    """
    
    def __init__(self, group_id: int):
        """
        Initialize GroupEvaluator.
        
        Args:
            group_id: ID of the group (0 or 1)
        
        Raises:
            ValueError: If group_id is not 0 or 1
        """
        if group_id not in [0, 1]:
            raise ValueError(f"group_id must be 0 or 1, got {group_id}")
        
        self.group_id = group_id
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[float] = []
        self.coverage_rates: List[float] = []
        self.transport_rates: List[float] = []
    
    def _run_single_episode(
        self,
        learner: Any,
        env: Any,
        render: bool = False,
        max_retries: int = 3
    ) -> Dict[str, float]:
        """
        Run a single episode and collect metrics.
        
        Args:
            learner: QPLEXLearner instance
            env: MATE environment
            render: Whether to render the environment
            max_retries: Maximum number of retries on error
        
        Returns:
            metrics: Dictionary containing episode metrics
        
        Raises:
            EvaluationError: If episode fails after max_retries
        """
        for attempt in range(max_retries):
            try:
                # Reset environment
                obs, info = env.reset()
                camera_obs, target_obs = obs
                state = env.state()
                
                # Reset hidden states
                learner.reset_hidden_states()
                
                episode_reward = 0.0
                episode_length = 0
                done = False
                
                # Track coverage and transport rates
                coverage_rates_episode = []
                transport_rates_episode = []
                
                while not done:
                    # Select actions for cameras (our agents)
                    camera_actions, _ = learner.select_action(camera_obs, state, evaluate=True)
                    
                    # Select actions for targets (default/zero actions)
                    target_actions = np.zeros((env.num_targets, 2))
                    
                    # Combine actions
                    actions = (camera_actions, target_actions)
                    
                    # Step environment
                    obs, rewards, terminated, truncated, info = env.step(actions)
                    camera_obs, target_obs = obs
                    camera_rewards, target_rewards = rewards
                    state = env.state()
                    
                    episode_reward += np.sum(camera_rewards)
                    episode_length += 1
                    done = terminated or truncated
                    
                    # Extract metrics from info
                    if info and len(info) > 0:
                        camera_infos, target_infos = info
                        if camera_infos and len(camera_infos) > 0:
                            coverage_rate = camera_infos[0].get('coverage_rate', 0.0)
                            transport_rate = camera_infos[0].get('mean_transport_rate', 0.0)
                            coverage_rates_episode.append(coverage_rate)
                            transport_rates_episode.append(transport_rate)
                    
                    if render:
                        env.render()
                
                # Calculate average coverage and transport rates for the episode
                avg_coverage = np.mean(coverage_rates_episode) if coverage_rates_episode else 0.0
                avg_transport = np.mean(transport_rates_episode) if transport_rates_episode else 0.0
                
                metrics = {
                    'episode_reward': float(episode_reward),
                    'episode_length': float(episode_length),
                    'coverage_rate': float(avg_coverage),
                    'transport_rate': float(avg_transport)
                }
                
                return metrics
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Log warning and retry
                    import warnings
                    warnings.warn(
                        f"Episode failed on attempt {attempt + 1}/{max_retries} "
                        f"for group {self.group_id}: {e}. Retrying..."
                    )
                    continue
                else:
                    # Final attempt failed, raise error
                    raise EvaluationError(
                        f"Episode failed after {max_retries} attempts for group {self.group_id}: {e}"
                    ) from e
        
        # Should never reach here
        raise EvaluationError(f"Unexpected error in _run_single_episode for group {self.group_id}")
    
    def evaluate_group(
        self,
        learner: Any,
        env: Any,
        n_episodes: int,
        seed: int,
        render: bool = False
    ) -> GroupResults:
        """
        Evaluate a group of agents over multiple episodes.
        
        Args:
            learner: QPLEXLearner instance
            env: MATE environment
            n_episodes: Number of episodes to evaluate
            seed: Random seed for reproducibility
            render: Whether to render the environment
        
        Returns:
            results: GroupResults containing all metrics
        
        Raises:
            ValueError: If n_episodes is less than 1
            EvaluationError: If evaluation fails
        """
        if n_episodes < 1:
            raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
        
        # Set random seed
        np.random.seed(seed)
        
        # Reset metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.coverage_rates = []
        self.transport_rates = []
        
        # Run episodes
        for episode_idx in range(n_episodes):
            try:
                metrics = self._run_single_episode(learner, env, render)
                
                self.episode_rewards.append(metrics['episode_reward'])
                self.episode_lengths.append(metrics['episode_length'])
                self.coverage_rates.append(metrics['coverage_rate'])
                self.transport_rates.append(metrics['transport_rate'])
                
            except EvaluationError as e:
                # Log error and continue with remaining episodes
                import warnings
                warnings.warn(
                    f"Failed to complete episode {episode_idx + 1}/{n_episodes} "
                    f"for group {self.group_id}: {e}"
                )
                # Skip this episode
                continue
        
        # Check if we have any successful episodes
        if len(self.episode_rewards) == 0:
            raise InsufficientDataError(
                f"No successful episodes completed for group {self.group_id}"
            )
        
        # Create GroupResults
        results = GroupResults(
            group_id=self.group_id,
            run_id=0,  # Will be set by caller
            seed=seed,
            episode_rewards=self.episode_rewards,
            episode_lengths=self.episode_lengths,
            coverage_rates=self.coverage_rates,
            transport_rates=self.transport_rates
        )
        
        return results


class EvaluationLogger:
    """
    Logger for evaluation results.
    
    Handles saving evaluation results to JSON and CSV files, and printing
    formatted summaries to console.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize EvaluationLogger.
        
        Args:
            log_dir: Directory to save log files
        
        Raises:
            ValueError: If log_dir is empty
        """
        if not log_dir:
            raise ValueError("log_dir cannot be empty")
        
        self.log_dir = log_dir
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Create log directory if it doesn't exist."""
        import os
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception as e:
            raise EvaluationError(f"Failed to create log directory {self.log_dir}: {e}") from e
    
    def save_json(
        self,
        results: Dict[str, Any],
        filepath: str
    ) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary to save
            filepath: Path to save JSON file (relative to log_dir or absolute)
        
        Raises:
            EvaluationError: If saving fails
        """
        import json
        import os
        
        # Make filepath absolute if it's relative
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.log_dir, filepath)
        
        # Ensure parent directory exists
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except Exception as e:
                raise EvaluationError(f"Failed to create directory {parent_dir}: {e}") from e
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            raise EvaluationError(f"Failed to save JSON to {filepath}: {e}") from e
    
    def save_csv(
        self,
        raw_data: Dict[str, List[float]],
        filepath: str
    ) -> None:
        """
        Save raw data to CSV file.
        
        Args:
            raw_data: Dictionary mapping column names to lists of values
            filepath: Path to save CSV file (relative to log_dir or absolute)
        
        Raises:
            EvaluationError: If saving fails
            ValueError: If raw_data is empty or columns have different lengths
        """
        import csv
        import os
        
        if not raw_data:
            raise ValueError("raw_data cannot be empty")
        
        # Check that all columns have the same length
        lengths = [len(values) for values in raw_data.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"All columns must have the same length, got {lengths}")
        
        # Make filepath absolute if it's relative
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.log_dir, filepath)
        
        # Ensure parent directory exists
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except Exception as e:
                raise EvaluationError(f"Failed to create directory {parent_dir}: {e}") from e
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                headers = list(raw_data.keys())
                writer.writerow(headers)
                
                # Write data rows
                n_rows = lengths[0] if lengths else 0
                for i in range(n_rows):
                    row = [raw_data[col][i] for col in headers]
                    writer.writerow(row)
        except Exception as e:
            raise EvaluationError(f"Failed to save CSV to {filepath}: {e}") from e
    
    def print_summary(
        self,
        results: Dict[str, Any]
    ) -> None:
        """
        Print formatted summary of results to console.
        
        Args:
            results: Results dictionary to print
        """
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        # Print timestamp if available
        if 'timestamp' in results:
            print(f"Timestamp: {results['timestamp']}")
        
        # Print timestep if available
        if 'timestep' in results:
            print(f"Training Timestep: {results['timestep']}")
        
        # Print configuration if available
        if 'config' in results:
            config = results['config']
            print(f"\nConfiguration:")
            print(f"  Evaluation Runs: {config.get('n_eval_runs', 'N/A')}")
            print(f"  Episodes per Run: {config.get('n_episodes_per_run', 'N/A')}")
            print(f"  Total Episodes: {config.get('n_eval_runs', 0) * config.get('n_episodes_per_run', 0)}")
            print(f"  Outlier Removal: {config.get('remove_outliers', 'N/A')}")
            print(f"  Confidence Level: {config.get('confidence_level', 'N/A')}")
        
        # Print group results
        for group_key in ['group1_results', 'group2_results']:
            if group_key in results:
                group_results = results[group_key]
                group_id = group_results.get('group_id', 'N/A')
                
                print(f"\n{'-' * 80}")
                print(f"GROUP {group_id} RESULTS")
                print(f"{'-' * 80}")
                
                # Print reward statistics
                if 'reward_stats' in group_results:
                    stats = group_results['reward_stats']
                    print(f"\nReward Statistics:")
                    print(f"  Mean: {stats.get('mean', 0):.4f} ± {stats.get('std', 0):.4f}")
                    print(f"  95% CI: [{stats.get('ci_lower', 0):.4f}, {stats.get('ci_upper', 0):.4f}]")
                    print(f"  CV: {stats.get('cv', 0):.4f}")
                    print(f"  Min: {stats.get('min', 0):.4f}, Max: {stats.get('max', 0):.4f}")
                    print(f"  Median: {stats.get('median', 0):.4f}")
                
                # Print coverage statistics
                if 'coverage_stats' in group_results:
                    stats = group_results['coverage_stats']
                    print(f"\nCoverage Statistics:")
                    print(f"  Mean: {stats.get('mean', 0):.4f} ± {stats.get('std', 0):.4f}")
                    print(f"  95% CI: [{stats.get('ci_lower', 0):.4f}, {stats.get('ci_upper', 0):.4f}]")
                
                # Print transport statistics
                if 'transport_stats' in group_results:
                    stats = group_results['transport_stats']
                    print(f"\nTransport Statistics:")
                    print(f"  Mean: {stats.get('mean', 0):.4f} ± {stats.get('std', 0):.4f}")
                    print(f"  95% CI: [{stats.get('ci_lower', 0):.4f}, {stats.get('ci_upper', 0):.4f}]")
                
                # Print outlier information
                if 'n_outliers_removed' in group_results:
                    n_outliers = group_results['n_outliers_removed']
                    outlier_pct = group_results.get('outlier_percentage', 0)
                    print(f"\nOutliers Removed: {n_outliers} ({outlier_pct:.2f}%)")
        
        # Print comparison results if available
        if 'comparison' in results:
            comparison = results['comparison']
            print(f"\n{'-' * 80}")
            print(f"GROUP COMPARISON")
            print(f"{'-' * 80}")
            
            reward_diff = comparison.get('reward_difference', 0)
            reward_diff_pct = comparison.get('reward_difference_percentage', 0)
            p_value = comparison.get('p_value', 1.0)
            effect_size = comparison.get('effect_size', 0)
            significant = comparison.get('statistical_significance', False)
            
            print(f"\nReward Difference (Group 1 - Group 2):")
            print(f"  Absolute: {reward_diff:.4f}")
            print(f"  Percentage: {reward_diff_pct:.2f}%")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Effect Size (Cohen's d): {effect_size:.4f}")
            print(f"  Statistically Significant (p < 0.05): {significant}")
        
        print("\n" + "=" * 80 + "\n")
    
    def log_results(
        self,
        results: Dict[str, Any],
        timestep: int
    ) -> None:
        """
        Log results by saving to JSON, CSV, and printing summary.
        
        Args:
            results: Results dictionary to log
            timestep: Current training timestep
        
        Raises:
            EvaluationError: If logging fails
        """
        import datetime
        
        # Add timestamp and timestep to results
        results['timestamp'] = datetime.datetime.now().isoformat()
        results['timestep'] = timestep
        
        # Generate filenames
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"evaluation_results_{timestep}_{timestamp_str}.json"
        csv_filename = f"evaluation_raw_data_{timestep}_{timestamp_str}.csv"
        
        # Save JSON
        try:
            self.save_json(results, json_filename)
            print(f"Saved evaluation results to: {json_filename}")
        except Exception as e:
            print(f"Warning: Failed to save JSON: {e}")
        
        # Save CSV if raw data is available
        try:
            raw_data = {}
            
            # Collect raw data from both groups
            for group_key in ['group1_results', 'group2_results']:
                if group_key in results:
                    group_results = results[group_key]
                    group_id = group_results.get('group_id', 'N/A')
                    
                    # Add raw data with group prefix
                    for metric in ['all_rewards', 'all_lengths', 'all_coverages', 'all_transports']:
                        if metric in group_results:
                            col_name = f"group{group_id}_{metric.replace('all_', '')}"
                            raw_data[col_name] = group_results[metric]
            
            if raw_data:
                self.save_csv(raw_data, csv_filename)
                print(f"Saved raw evaluation data to: {csv_filename}")
        except Exception as e:
            print(f"Warning: Failed to save CSV: {e}")
        
        # Print summary
        try:
            self.print_summary(results)
        except Exception as e:
            print(f"Warning: Failed to print summary: {e}")


class StatisticsCalculator:
    """
    Utility class for calculating advanced statistics on evaluation data.
    
    Provides static methods for:
    - Outlier detection and removal (IQR and Z-score methods)
    - Confidence interval calculation using t-distribution
    - Convergence metrics (CV, stability score)
    - Summary statistics (min, max, median, quartiles)
    """
    
    @staticmethod
    def remove_outliers(
        data: np.ndarray,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from data using IQR or Z-score method.
        
        Args:
            data: Input data array
            method: Method to detect outliers ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
                      - For IQR: multiplier for IQR (default 1.5)
                      - For Z-score: number of standard deviations (default 1.5)
        
        Returns:
            cleaned_data: Data after removing outliers
            outlier_mask: Boolean mask where True indicates outliers
        
        Raises:
            ValueError: If method is not 'iqr' or 'zscore'
            ValueError: If data is empty or all NaN
        """
        if len(data) == 0:
            raise ValueError("Cannot remove outliers from empty data")
        
        data = np.asarray(data, dtype=float)
        
        if np.all(np.isnan(data)):
            raise ValueError("Cannot remove outliers from all-NaN data")
        
        # Remove NaN values first
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            raise ValueError("No valid data after removing NaN values")
        
        if method == 'iqr':
            # IQR method
            q1 = np.percentile(valid_data, 25)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outlier_mask = (valid_data < lower_bound) | (valid_data > upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            mean = np.mean(valid_data)
            std = np.std(valid_data, ddof=1)
            
            if std == 0:
                # All values are the same, no outliers
                outlier_mask = np.zeros(len(valid_data), dtype=bool)
            else:
                z_scores = np.abs((valid_data - mean) / std)
                outlier_mask = z_scores > threshold
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")
        
        cleaned_data = valid_data[~outlier_mask]
        
        return cleaned_data, outlier_mask
    
    @staticmethod
    def calculate_confidence_interval(
        data: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval using t-distribution.
        
        Args:
            data: Input data array
            confidence: Confidence level (0-1), default 0.95 for 95% CI
        
        Returns:
            mean: Mean value
            lower_bound: Lower bound of confidence interval
            upper_bound: Upper bound of confidence interval
        
        Raises:
            ValueError: If confidence is not between 0 and 1
            ValueError: If data has less than 2 samples
        """
        if not 0 < confidence < 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        data = np.asarray(data, dtype=float)
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) < 2:
            raise ValueError("Need at least 2 samples to calculate confidence interval")
        
        mean = np.mean(valid_data)
        std_err = stats.sem(valid_data)  # Standard error of the mean
        
        # Degrees of freedom
        df = len(valid_data) - 1
        
        # t-value for the given confidence level
        t_value = stats.t.ppf((1 + confidence) / 2, df)
        
        # Margin of error
        margin = t_value * std_err
        
        lower_bound = mean - margin
        upper_bound = mean + margin
        
        return mean, lower_bound, upper_bound

    @staticmethod
    def calculate_convergence_metrics(
        data: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate convergence metrics for evaluation data.
        
        Metrics include:
        - Coefficient of Variation (CV): std/mean, measures relative variability
        - Stability Score: 1 - CV, higher is more stable (0-1 range)
        - Range: max - min
        - Relative Range: (max - min) / mean
        
        Args:
            data: Input data array
        
        Returns:
            metrics: Dictionary containing convergence metrics
        
        Raises:
            ValueError: If data is empty or all NaN
        """
        data = np.asarray(data, dtype=float)
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            raise ValueError("Cannot calculate convergence metrics from empty data")
        
        mean = np.mean(valid_data)
        std = np.std(valid_data, ddof=1)
        
        # Coefficient of Variation
        if mean != 0:
            cv = std / abs(mean)
        else:
            cv = float('inf') if std != 0 else 0.0
        
        # Stability score (inverse of CV, capped at 1)
        if cv == float('inf'):
            stability_score = 0.0
        else:
            stability_score = max(0.0, 1.0 - cv)
        
        # Range metrics
        data_range = np.max(valid_data) - np.min(valid_data)
        
        if mean != 0:
            relative_range = data_range / abs(mean)
        else:
            relative_range = float('inf') if data_range != 0 else 0.0
        
        metrics = {
            'cv': cv,
            'stability_score': stability_score,
            'range': data_range,
            'relative_range': relative_range,
            'std': std,
            'mean': mean
        }
        
        return metrics
    
    @staticmethod
    def calculate_summary_statistics(
        data: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate summary statistics for data.
        
        Statistics include:
        - min, max, mean, median
        - Q1 (25th percentile), Q3 (75th percentile)
        - IQR (Interquartile Range)
        - std (standard deviation)
        - count (number of valid samples)
        
        Args:
            data: Input data array
        
        Returns:
            stats: Dictionary containing summary statistics
        
        Raises:
            ValueError: If data is empty or all NaN
        """
        data = np.asarray(data, dtype=float)
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            raise ValueError("Cannot calculate summary statistics from empty data")
        
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1
        
        stats_dict = {
            'count': len(valid_data),
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'mean': np.mean(valid_data),
            'median': np.median(valid_data),
            'std': np.std(valid_data, ddof=1),
            'q1': q1,
            'q3': q3,
            'iqr': iqr
        }
        
        return stats_dict


class ImprovedEvaluator:
    """
    Improved evaluator for multi-group agent evaluation.
    
    This class orchestrates the complete evaluation process including:
    - Warm-up episodes to stabilize hidden states
    - Multiple evaluation runs with different seeds
    - Statistical aggregation with outlier removal
    - Group comparison with statistical tests
    - Comprehensive logging and reporting
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize ImprovedEvaluator.
        
        Args:
            config: EvaluationConfig instance with evaluation parameters
        
        Raises:
            ConfigurationError: If config is invalid
        """
        if not isinstance(config, EvaluationConfig):
            raise ConfigurationError("config must be an EvaluationConfig instance")
        
        self.config = config
        self.logger = None  # Will be set when log_dir is provided
    
    def run_warmup_episodes(
        self,
        learner: Any,
        env: Any,
        n_episodes: int
    ) -> None:
        """
        Run warm-up episodes to stabilize hidden states.
        
        Warm-up episodes are run before the actual evaluation to ensure that
        the agent's hidden states (for RNN-based models) are properly initialized
        and stabilized.
        
        Args:
            learner: QPLEXLearner instance
            env: MATE environment
            n_episodes: Number of warm-up episodes to run
        
        Raises:
            ValueError: If n_episodes is negative
        """
        if n_episodes < 0:
            raise ValueError(f"n_episodes must be >= 0, got {n_episodes}")
        
        if n_episodes == 0:
            return
        
        print(f"Running {n_episodes} warm-up episodes...")
        
        for episode_idx in range(n_episodes):
            try:
                # Reset environment
                obs, info = env.reset()
                camera_obs, target_obs = obs
                state = env.state()
                
                # Reset hidden states
                learner.reset_hidden_states()
                
                done = False
                
                while not done:
                    # Select actions
                    camera_actions, _ = learner.select_action(camera_obs, state, evaluate=True)
                    target_actions = np.zeros((env.num_targets, 2))
                    actions = (camera_actions, target_actions)
                    
                    # Step environment
                    obs, rewards, terminated, truncated, info = env.step(actions)
                    camera_obs, target_obs = obs
                    state = env.state()
                    done = terminated or truncated
                
                if (episode_idx + 1) % max(1, n_episodes // 5) == 0:
                    print(f"  Warm-up progress: {episode_idx + 1}/{n_episodes}")
                    
            except Exception as e:
                import warnings
                warnings.warn(f"Warm-up episode {episode_idx + 1} failed: {e}")
                continue
        
        print("Warm-up complete.")
    
    def _evaluate_single_run(
        self,
        learners: List[Any],
        env: Any,
        run_id: int,
        seed: int
    ) -> Tuple[GroupResults, GroupResults]:
        """
        Evaluate both groups in a single run.
        
        Args:
            learners: List of 2 QPLEXLearner instances (one for each group)
            env: MATE environment
            run_id: ID of this evaluation run
            seed: Random seed for this run
        
        Returns:
            group1_results: Results for group 1
            group2_results: Results for group 2
        
        Raises:
            ValueError: If learners list doesn't contain exactly 2 learners
            EvaluationError: If evaluation fails
        """
        if len(learners) != 2:
            raise ValueError(f"Expected 2 learners, got {len(learners)}")
        
        print(f"\nRun {run_id + 1}/{self.config.n_eval_runs} (seed={seed}):")
        
        # Evaluate group 1
        print(f"  Evaluating Group 0...")
        evaluator1 = GroupEvaluator(group_id=0)
        group1_results = evaluator1.evaluate_group(
            learner=learners[0],
            env=env,
            n_episodes=self.config.n_episodes_per_run,
            seed=seed,
            render=False
        )
        group1_results.run_id = run_id
        print(f"    Mean reward: {group1_results.mean_reward:.2f} ± {group1_results.std_reward:.2f}")
        
        # Evaluate group 2
        print(f"  Evaluating Group 1...")
        evaluator2 = GroupEvaluator(group_id=1)
        group2_results = evaluator2.evaluate_group(
            learner=learners[1],
            env=env,
            n_episodes=self.config.n_episodes_per_run,
            seed=seed + 1,  # Use different seed for group 2
            render=False
        )
        group2_results.run_id = run_id
        print(f"    Mean reward: {group2_results.mean_reward:.2f} ± {group2_results.std_reward:.2f}")
        
        return group1_results, group2_results
    
    def _aggregate_results(
        self,
        all_group_results: List[GroupResults],
        group_id: int
    ) -> AggregatedResults:
        """
        Aggregate results from multiple runs for a single group.
        
        This method:
        1. Collects all raw data from multiple runs
        2. Optionally removes outliers
        3. Calculates statistics with confidence intervals
        4. Computes convergence metrics
        
        Args:
            all_group_results: List of GroupResults from multiple runs
            group_id: ID of the group being aggregated
        
        Returns:
            aggregated_results: AggregatedResults with all statistics
        
        Raises:
            InsufficientDataError: If no valid data is available
        """
        if not all_group_results:
            raise InsufficientDataError(f"No results to aggregate for group {group_id}")
        
        print(f"\nAggregating results for Group {group_id}...")
        
        # Collect all raw data
        all_rewards = []
        all_lengths = []
        all_coverages = []
        all_transports = []
        
        for result in all_group_results:
            all_rewards.extend(result.episode_rewards)
            all_lengths.extend(result.episode_lengths)
            all_coverages.extend(result.coverage_rates)
            all_transports.extend(result.transport_rates)
        
        n_episodes_total = len(all_rewards)
        n_runs = len(all_group_results)
        
        print(f"  Total episodes: {n_episodes_total} from {n_runs} runs")
        
        # Calculate statistics for each metric
        def calculate_metric_stats(data: List[float], metric_name: str) -> Dict[str, float]:
            """Helper function to calculate statistics for a metric."""
            data_array = np.array(data)
            
            # Remove outliers if configured
            n_outliers = 0
            if self.config.remove_outliers:
                try:
                    cleaned_data, outlier_mask = StatisticsCalculator.remove_outliers(
                        data_array,
                        method=self.config.outlier_method,
                        threshold=self.config.outlier_threshold
                    )
                    n_outliers = np.sum(outlier_mask)
                    data_array = cleaned_data
                    print(f"  {metric_name}: Removed {n_outliers} outliers ({n_outliers/len(data)*100:.1f}%)")
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to remove outliers for {metric_name}: {e}")
            
            # Calculate confidence interval
            try:
                mean, ci_lower, ci_upper = StatisticsCalculator.calculate_confidence_interval(
                    data_array,
                    confidence=self.config.confidence_level
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to calculate confidence interval for {metric_name}: {e}")
                mean = float(np.mean(data_array))
                ci_lower = mean
                ci_upper = mean
            
            # Calculate convergence metrics
            try:
                convergence = StatisticsCalculator.calculate_convergence_metrics(data_array)
                cv = convergence['cv']
                stability_score = convergence['stability_score']
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to calculate convergence metrics for {metric_name}: {e}")
                cv = 0.0
                stability_score = 1.0
            
            # Calculate summary statistics
            try:
                summary = StatisticsCalculator.calculate_summary_statistics(data_array)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to calculate summary statistics for {metric_name}: {e}")
                summary = {
                    'min': float(np.min(data_array)),
                    'max': float(np.max(data_array)),
                    'median': float(np.median(data_array)),
                    'std': float(np.std(data_array, ddof=1))
                }
            
            stats = {
                'mean': mean,
                'std': summary.get('std', float(np.std(data_array, ddof=1))),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'cv': cv,
                'stability_score': stability_score,
                'min': summary.get('min', float(np.min(data_array))),
                'max': summary.get('max', float(np.max(data_array))),
                'median': summary.get('median', float(np.median(data_array))),
                'q1': summary.get('q1', float(np.percentile(data_array, 25))),
                'q3': summary.get('q3', float(np.percentile(data_array, 75)))
            }
            
            return stats, n_outliers
        
        # Calculate statistics for each metric
        reward_stats, reward_outliers = calculate_metric_stats(all_rewards, "Rewards")
        length_stats, _ = calculate_metric_stats(all_lengths, "Lengths")
        coverage_stats, _ = calculate_metric_stats(all_coverages, "Coverage")
        transport_stats, _ = calculate_metric_stats(all_transports, "Transport")
        
        # Print summary
        print(f"  Reward: {reward_stats['mean']:.2f} ± {reward_stats['std']:.2f} "
              f"[{reward_stats['ci_lower']:.2f}, {reward_stats['ci_upper']:.2f}] "
              f"(CV: {reward_stats['cv']:.3f})")
        print(f"  Coverage: {coverage_stats['mean']:.3f} ± {coverage_stats['std']:.3f}")
        print(f"  Transport: {transport_stats['mean']:.3f} ± {transport_stats['std']:.3f}")
        
        # Create AggregatedResults
        aggregated = AggregatedResults(
            group_id=group_id,
            n_runs=n_runs,
            n_episodes_total=n_episodes_total,
            reward_stats=reward_stats,
            length_stats=length_stats,
            coverage_stats=coverage_stats,
            transport_stats=transport_stats,
            all_rewards=all_rewards,
            all_lengths=all_lengths,
            all_coverages=all_coverages,
            all_transports=all_transports,
            n_outliers_removed=reward_outliers,
            outlier_percentage=reward_outliers / n_episodes_total * 100 if n_episodes_total > 0 else 0.0
        )
        
        return aggregated
    
    def _compare_groups(
        self,
        group1_results: AggregatedResults,
        group2_results: AggregatedResults
    ) -> ComparisonResults:
        """
        Compare two groups using statistical tests.
        
        Performs:
        - Independent t-test for statistical significance
        - Cohen's d effect size calculation
        - Percentage difference calculation
        
        Args:
            group1_results: Aggregated results for group 1
            group2_results: Aggregated results for group 2
        
        Returns:
            comparison_results: ComparisonResults with all comparison metrics
        """
        print("\nComparing groups...")
        
        # Create ComparisonResults (it will calculate everything in __post_init__)
        comparison = ComparisonResults(
            group1_results=group1_results,
            group2_results=group2_results
        )
        
        # Print comparison summary
        print(f"  Reward difference: {comparison.reward_difference:.2f} "
              f"({comparison.reward_difference_percentage:+.1f}%)")
        print(f"  P-value: {comparison.p_value:.4f}")
        print(f"  Effect size (Cohen's d): {comparison.effect_size:.3f}")
        print(f"  Statistically significant: {comparison.statistical_significance}")
        
        return comparison
    
    def evaluate(
        self,
        learners: List[Any],
        env: Any,
        timestep: int = 0,
        log_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation process for both groups.
        
        This is the main entry point for evaluation. It orchestrates:
        1. Warm-up episodes
        2. Multiple evaluation runs with different seeds
        3. Statistical aggregation
        4. Group comparison
        5. Logging and reporting
        
        Args:
            learners: List of 2 QPLEXLearner instances (one for each group)
            env: MATE environment
            timestep: Current training timestep (for logging)
            log_dir: Directory to save logs (optional)
        
        Returns:
            results: Dictionary containing all evaluation results
        
        Raises:
            ValueError: If learners list doesn't contain exactly 2 learners
            EvaluationError: If evaluation fails
        """
        if len(learners) != 2:
            raise ValueError(f"Expected 2 learners, got {len(learners)}")
        
        print("\n" + "=" * 80)
        print("STARTING IMPROVED EVALUATION")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Evaluation runs: {self.config.n_eval_runs}")
        print(f"  Episodes per run: {self.config.n_episodes_per_run}")
        print(f"  Total episodes: {self.config.n_eval_runs * self.config.n_episodes_per_run}")
        print(f"  Warm-up episodes: {self.config.n_warmup_episodes}")
        print(f"  Outlier removal: {self.config.remove_outliers}")
        print(f"  Confidence level: {self.config.confidence_level}")
        
        # Setup logger if log_dir is provided
        if log_dir:
            self.logger = EvaluationLogger(log_dir)
        
        # Run warm-up episodes for both learners
        if self.config.n_warmup_episodes > 0:
            print("\n" + "-" * 80)
            print("WARM-UP PHASE")
            print("-" * 80)
            
            for i, learner in enumerate(learners):
                print(f"\nWarm-up for Group {i}:")
                self.run_warmup_episodes(learner, env, self.config.n_warmup_episodes)
        
        # Run multiple evaluation runs
        print("\n" + "-" * 80)
        print("EVALUATION PHASE")
        print("-" * 80)
        
        group1_all_results = []
        group2_all_results = []
        
        for run_id in range(self.config.n_eval_runs):
            seed = self.config.seeds[run_id]
            
            try:
                group1_result, group2_result = self._evaluate_single_run(
                    learners=learners,
                    env=env,
                    run_id=run_id,
                    seed=seed
                )
                
                group1_all_results.append(group1_result)
                group2_all_results.append(group2_result)
                
            except Exception as e:
                import warnings
                warnings.warn(f"Run {run_id + 1} failed: {e}")
                continue
        
        # Check if we have any successful runs
        if not group1_all_results or not group2_all_results:
            raise InsufficientDataError("No successful evaluation runs completed")
        
        # Aggregate results
        print("\n" + "-" * 80)
        print("AGGREGATION PHASE")
        print("-" * 80)
        
        group1_aggregated = self._aggregate_results(group1_all_results, group_id=0)
        group2_aggregated = self._aggregate_results(group2_all_results, group_id=1)
        
        # Compare groups
        print("\n" + "-" * 80)
        print("COMPARISON PHASE")
        print("-" * 80)
        
        comparison = self._compare_groups(group1_aggregated, group2_aggregated)
        
        # Prepare results dictionary
        results = {
            'config': self.config.to_dict(),
            'group1_results': group1_aggregated.to_dict(),
            'group2_results': group2_aggregated.to_dict(),
            'comparison': comparison.to_dict()
        }
        
        # Log results if logger is available
        if self.logger:
            print("\n" + "-" * 80)
            print("LOGGING PHASE")
            print("-" * 80)
            self.logger.log_results(results, timestep)
        else:
            # Print summary even if no logger
            print("\n" + "-" * 80)
            print("SUMMARY")
            print("-" * 80)
            temp_logger = EvaluationLogger("/tmp")
            temp_logger.print_summary(results)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80 + "\n")
        
        return results
