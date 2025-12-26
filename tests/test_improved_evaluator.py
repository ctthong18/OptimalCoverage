"""
Integration tests for ImprovedEvaluator class.

Tests cover:
- Full evaluation flow with mock learners and environment
- Warm-up episodes
- Multiple runs with different seeds
- Aggregation and comparison
- Error handling
"""

import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from evaluation_utils import (
    ImprovedEvaluator,
    EvaluationConfig,
    GroupResults,
    AggregatedResults,
    ComparisonResults,
    EvaluationError,
    InsufficientDataError,
    ConfigurationError
)


class MockLearner:
    """Mock QPLEXLearner for testing."""
    
    def __init__(self, group_id=0):
        self.group_id = group_id
        self.hidden_states_reset = False
        self.action_count = 0
    
    def reset_hidden_states(self):
        """Mock reset hidden states."""
        self.hidden_states_reset = True
    
    def select_action(self, obs, state, evaluate=False):
        """Mock select action."""
        self.action_count += 1
        # Return random actions and empty info
        n_agents = obs.shape[0] if len(obs.shape) > 1 else 1
        # Add some variation based on group_id for testing
        actions = np.random.uniform(-1, 1, size=(n_agents, 2)) + self.group_id * 0.1
        return actions, {}


class MockEnvironment:
    """Mock MATE environment for testing."""
    
    def __init__(self, num_cameras=4, num_targets=8, episode_length=50):
        self.num_cameras = num_cameras
        self.num_targets = num_targets
        self.episode_length = episode_length
        self.current_step = 0
        self.should_fail = False
        self.fail_on_step = -1
    
    def reset(self):
        """Mock reset."""
        self.current_step = 0
        
        # Return mock observations and info
        camera_obs = np.random.randn(self.num_cameras, 50)
        target_obs = np.random.randn(self.num_targets, 40)
        obs = (camera_obs, target_obs)
        info = {}
        
        return obs, info
    
    def state(self):
        """Mock state."""
        return np.random.randn(200)
    
    def step(self, actions):
        """Mock step."""
        if self.should_fail or self.current_step == self.fail_on_step:
            raise RuntimeError("Mock environment error")
        
        self.current_step += 1
        
        # Mock observations
        camera_obs = np.random.randn(self.num_cameras, 50)
        target_obs = np.random.randn(self.num_targets, 40)
        obs = (camera_obs, target_obs)
        
        # Mock rewards with some variation
        camera_rewards = np.random.randn(self.num_cameras) * 5 + 10
        target_rewards = np.random.randn(self.num_targets) * 5
        rewards = (camera_rewards, target_rewards)
        
        # Mock termination
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Mock info with metrics
        camera_infos = [
            {
                'coverage_rate': 0.75 + np.random.randn() * 0.05,
                'mean_transport_rate': 0.65 + np.random.randn() * 0.05
            }
            for _ in range(self.num_cameras)
        ]
        target_infos = [{} for _ in range(self.num_targets)]
        info = (camera_infos, target_infos)
        
        return obs, rewards, terminated, truncated, info
    
    def render(self):
        """Mock render."""
        pass


class TestImprovedEvaluatorInit:
    """Tests for ImprovedEvaluator initialization."""
    
    def test_valid_initialization(self):
        """Test initialization with valid config."""
        config = EvaluationConfig(n_eval_runs=3, n_episodes_per_run=10)
        evaluator = ImprovedEvaluator(config)
        
        assert evaluator.config == config
        assert evaluator.logger is None
    
    def test_initialization_with_invalid_config(self):
        """Test initialization with invalid config type."""
        with pytest.raises(ConfigurationError, match="config must be an EvaluationConfig instance"):
            ImprovedEvaluator({"n_eval_runs": 3})
    
    def test_initialization_stores_config(self):
        """Test that config is properly stored."""
        config = EvaluationConfig(
            n_eval_runs=5,
            n_episodes_per_run=100,
            n_warmup_episodes=5,
            remove_outliers=True
        )
        evaluator = ImprovedEvaluator(config)
        
        assert evaluator.config.n_eval_runs == 5
        assert evaluator.config.n_episodes_per_run == 100
        assert evaluator.config.n_warmup_episodes == 5
        assert evaluator.config.remove_outliers is True


class TestRunWarmupEpisodes:
    """Tests for run_warmup_episodes method."""
    
    def test_warmup_episodes_run_successfully(self):
        """Test that warm-up episodes run without error."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        learner = MockLearner()
        env = MockEnvironment(episode_length=10)
        
        # Should not raise error
        evaluator.run_warmup_episodes(learner, env, n_episodes=3)
        
        # Learner should have been used
        assert learner.action_count > 0
    
    def test_warmup_with_zero_episodes(self):
        """Test warm-up with zero episodes."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        learner = MockLearner()
        env = MockEnvironment()
        
        # Should return immediately without error
        evaluator.run_warmup_episodes(learner, env, n_episodes=0)
        
        # Learner should not have been used
        assert learner.action_count == 0
    
    def test_warmup_with_negative_episodes_raises_error(self):
        """Test warm-up with negative episodes."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        learner = MockLearner()
        env = MockEnvironment()
        
        with pytest.raises(ValueError, match="n_episodes must be >= 0"):
            evaluator.run_warmup_episodes(learner, env, n_episodes=-1)
    
    def test_warmup_continues_after_episode_failure(self):
        """Test that warm-up continues after individual episode failures."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        learner = MockLearner()
        env = MockEnvironment(episode_length=10)
        
        # Make first episode fail
        call_count = [0]
        original_reset = env.reset
        
        def failing_reset():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("First episode fails")
            return original_reset()
        
        env.reset = failing_reset
        
        # Should complete with warning
        with pytest.warns(UserWarning, match="Warm-up episode .* failed"):
            evaluator.run_warmup_episodes(learner, env, n_episodes=3)


class TestEvaluateSingleRun:
    """Tests for _evaluate_single_run method."""
    
    def test_single_run_evaluates_both_groups(self):
        """Test that single run evaluates both groups."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        learner1 = MockLearner(group_id=0)
        learner2 = MockLearner(group_id=1)
        learners = [learner1, learner2]
        env = MockEnvironment(episode_length=10)
        
        group1_result, group2_result = evaluator._evaluate_single_run(
            learners=learners,
            env=env,
            run_id=0,
            seed=42
        )
        
        # Check results
        assert isinstance(group1_result, GroupResults)
        assert isinstance(group2_result, GroupResults)
        assert group1_result.group_id == 0
        assert group2_result.group_id == 1
        assert group1_result.run_id == 0
        assert group2_result.run_id == 0
        assert len(group1_result.episode_rewards) == 5
        assert len(group2_result.episode_rewards) == 5
    
    def test_single_run_with_invalid_learners_count(self):
        """Test single run with wrong number of learners."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        learner = MockLearner()
        env = MockEnvironment()
        
        with pytest.raises(ValueError, match="Expected 2 learners"):
            evaluator._evaluate_single_run([learner], env, 0, 42)
        
        with pytest.raises(ValueError, match="Expected 2 learners"):
            evaluator._evaluate_single_run([learner, learner, learner], env, 0, 42)
    
    def test_single_run_uses_different_seeds_for_groups(self):
        """Test that different seeds are used for each group."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment(episode_length=10)
        
        group1_result, group2_result = evaluator._evaluate_single_run(
            learners=learners,
            env=env,
            run_id=0,
            seed=42
        )
        
        # Seeds should be different (42 and 43)
        assert group1_result.seed == 42
        assert group2_result.seed == 43


class TestAggregateResults:
    """Tests for _aggregate_results method."""
    
    def test_aggregate_results_basic(self):
        """Test basic aggregation of results."""
        config = EvaluationConfig(n_eval_runs=3, n_episodes_per_run=10, remove_outliers=False)
        evaluator = ImprovedEvaluator(config)
        
        # Create mock results
        results = []
        for i in range(3):
            result = GroupResults(
                group_id=0,
                run_id=i,
                seed=42 + i,
                episode_rewards=[10.0 + j for j in range(10)],
                episode_lengths=[50.0] * 10,
                coverage_rates=[0.75] * 10,
                transport_rates=[0.65] * 10
            )
            results.append(result)
        
        aggregated = evaluator._aggregate_results(results, group_id=0)
        
        # Check aggregated results
        assert isinstance(aggregated, AggregatedResults)
        assert aggregated.group_id == 0
        assert aggregated.n_runs == 3
        assert aggregated.n_episodes_total == 30
        assert len(aggregated.all_rewards) == 30
        assert 'mean' in aggregated.reward_stats
        assert 'std' in aggregated.reward_stats
        assert 'ci_lower' in aggregated.reward_stats
        assert 'ci_upper' in aggregated.reward_stats
    
    def test_aggregate_results_with_outlier_removal(self):
        """Test aggregation with outlier removal."""
        config = EvaluationConfig(
            n_eval_runs=2,
            n_episodes_per_run=10,
            remove_outliers=True,
            outlier_method='iqr',
            outlier_threshold=1.5
        )
        evaluator = ImprovedEvaluator(config)
        
        # Create results with outliers
        results = []
        for i in range(2):
            rewards = [10.0] * 9 + [100.0]  # One outlier
            result = GroupResults(
                group_id=0,
                run_id=i,
                seed=42 + i,
                episode_rewards=rewards,
                episode_lengths=[50.0] * 10,
                coverage_rates=[0.75] * 10,
                transport_rates=[0.65] * 10
            )
            results.append(result)
        
        aggregated = evaluator._aggregate_results(results, group_id=0)
        
        # Check that outliers were removed
        assert aggregated.n_outliers_removed > 0
        assert aggregated.outlier_percentage > 0
    
    def test_aggregate_results_with_empty_list_raises_error(self):
        """Test aggregation with empty results list."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        
        with pytest.raises(InsufficientDataError, match="No results to aggregate"):
            evaluator._aggregate_results([], group_id=0)
    
    def test_aggregate_results_calculates_all_metrics(self):
        """Test that all metrics are calculated."""
        config = EvaluationConfig(n_eval_runs=2, n_episodes_per_run=5, remove_outliers=False)
        evaluator = ImprovedEvaluator(config)
        
        results = []
        for i in range(2):
            result = GroupResults(
                group_id=0,
                run_id=i,
                seed=42 + i,
                episode_rewards=[10.0 + j for j in range(5)],
                episode_lengths=[50.0 + j for j in range(5)],
                coverage_rates=[0.75 + j * 0.01 for j in range(5)],
                transport_rates=[0.65 + j * 0.01 for j in range(5)]
            )
            results.append(result)
        
        aggregated = evaluator._aggregate_results(results, group_id=0)
        
        # Check all stats dictionaries have required keys
        required_keys = ['mean', 'std', 'ci_lower', 'ci_upper', 'cv', 'min', 'max', 'median']
        for stats_dict in [
            aggregated.reward_stats,
            aggregated.length_stats,
            aggregated.coverage_stats,
            aggregated.transport_stats
        ]:
            for key in required_keys:
                assert key in stats_dict


class TestCompareGroups:
    """Tests for _compare_groups method."""
    
    def test_compare_groups_basic(self):
        """Test basic group comparison."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        
        # Create mock aggregated results
        group1 = AggregatedResults(
            group_id=0,
            n_runs=1,
            n_episodes_total=5,
            reward_stats={'mean': 100.0, 'std': 10.0, 'ci_lower': 95.0, 'ci_upper': 105.0},
            length_stats={'mean': 50.0, 'std': 5.0, 'ci_lower': 48.0, 'ci_upper': 52.0},
            coverage_stats={'mean': 0.75, 'std': 0.05, 'ci_lower': 0.72, 'ci_upper': 0.78},
            transport_stats={'mean': 0.65, 'std': 0.05, 'ci_lower': 0.62, 'ci_upper': 0.68},
            all_rewards=[95.0, 100.0, 105.0, 98.0, 102.0],
            all_lengths=[50.0] * 5,
            all_coverages=[0.75] * 5,
            all_transports=[0.65] * 5
        )
        
        group2 = AggregatedResults(
            group_id=1,
            n_runs=1,
            n_episodes_total=5,
            reward_stats={'mean': 90.0, 'std': 10.0, 'ci_lower': 85.0, 'ci_upper': 95.0},
            length_stats={'mean': 50.0, 'std': 5.0, 'ci_lower': 48.0, 'ci_upper': 52.0},
            coverage_stats={'mean': 0.70, 'std': 0.05, 'ci_lower': 0.67, 'ci_upper': 0.73},
            transport_stats={'mean': 0.60, 'std': 0.05, 'ci_lower': 0.57, 'ci_upper': 0.63},
            all_rewards=[85.0, 90.0, 95.0, 88.0, 92.0],
            all_lengths=[50.0] * 5,
            all_coverages=[0.70] * 5,
            all_transports=[0.60] * 5
        )
        
        comparison = evaluator._compare_groups(group1, group2)
        
        # Check comparison results
        assert isinstance(comparison, ComparisonResults)
        assert comparison.reward_difference == 10.0
        assert comparison.reward_difference_percentage > 0
        assert 0 <= comparison.p_value <= 1
        assert comparison.effect_size != 0


class TestEvaluate:
    """Tests for the main evaluate method."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_full_evaluation_flow(self, temp_log_dir):
        """Test complete evaluation flow."""
        config = EvaluationConfig(
            n_eval_runs=2,
            n_episodes_per_run=5,
            n_warmup_episodes=2,
            remove_outliers=False
        )
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment(episode_length=10)
        
        results = evaluator.evaluate(
            learners=learners,
            env=env,
            timestep=1000,
            log_dir=temp_log_dir
        )
        
        # Check results structure
        assert 'config' in results
        assert 'group1_results' in results
        assert 'group2_results' in results
        assert 'comparison' in results
        
        # Check group results
        assert results['group1_results']['group_id'] == 0
        assert results['group2_results']['group_id'] == 1
        assert results['group1_results']['n_runs'] == 2
        assert results['group2_results']['n_runs'] == 2
        
        # Check comparison
        assert 'reward_difference' in results['comparison']
        assert 'p_value' in results['comparison']
        assert 'effect_size' in results['comparison']
    
    def test_evaluation_with_warmup(self):
        """Test evaluation with warm-up episodes."""
        config = EvaluationConfig(
            n_eval_runs=1,
            n_episodes_per_run=3,
            n_warmup_episodes=5
        )
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment(episode_length=10)
        
        # Track warm-up calls
        initial_action_counts = [l.action_count for l in learners]
        
        results = evaluator.evaluate(learners, env, timestep=0)
        
        # Both learners should have been used for warm-up
        for i, learner in enumerate(learners):
            assert learner.action_count > initial_action_counts[i]
        
        # Results should still be valid
        assert 'group1_results' in results
        assert 'group2_results' in results
    
    def test_evaluation_without_warmup(self):
        """Test evaluation without warm-up episodes."""
        config = EvaluationConfig(
            n_eval_runs=1,
            n_episodes_per_run=3,
            n_warmup_episodes=0
        )
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment(episode_length=10)
        
        results = evaluator.evaluate(learners, env, timestep=0)
        
        # Should complete successfully
        assert 'group1_results' in results
        assert 'group2_results' in results
    
    def test_evaluation_with_multiple_runs(self):
        """Test evaluation with multiple runs."""
        config = EvaluationConfig(
            n_eval_runs=3,
            n_episodes_per_run=5,
            n_warmup_episodes=0
        )
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment(episode_length=10)
        
        results = evaluator.evaluate(learners, env, timestep=0)
        
        # Check that results include data from all runs
        assert results['group1_results']['n_runs'] == 3
        assert results['group2_results']['n_runs'] == 3
        assert results['group1_results']['n_episodes_total'] == 15
        assert results['group2_results']['n_episodes_total'] == 15
    
    def test_evaluation_with_invalid_learners_count(self):
        """Test evaluation with wrong number of learners."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        learner = MockLearner()
        env = MockEnvironment()
        
        with pytest.raises(ValueError, match="Expected 2 learners"):
            evaluator.evaluate([learner], env, timestep=0)
    
    def test_evaluation_creates_logger_when_log_dir_provided(self, temp_log_dir):
        """Test that logger is created when log_dir is provided."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=3)
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment(episode_length=10)
        
        assert evaluator.logger is None
        
        evaluator.evaluate(learners, env, timestep=0, log_dir=temp_log_dir)
        
        # Logger should be created
        assert evaluator.logger is not None
        
        # Log files should be created
        import os
        files = os.listdir(temp_log_dir)
        json_files = [f for f in files if f.endswith('.json')]
        csv_files = [f for f in files if f.endswith('.csv')]
        
        assert len(json_files) > 0
        assert len(csv_files) > 0
    
    def test_evaluation_without_log_dir(self):
        """Test evaluation without log directory."""
        config = EvaluationConfig(n_eval_runs=1, n_episodes_per_run=3)
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment(episode_length=10)
        
        # Should complete without error
        results = evaluator.evaluate(learners, env, timestep=0, log_dir=None)
        
        assert 'group1_results' in results
        assert evaluator.logger is None
    
    def test_evaluation_continues_after_run_failure(self):
        """Test that evaluation continues after individual run failures."""
        config = EvaluationConfig(n_eval_runs=3, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment(episode_length=10)
        
        # Mock _evaluate_single_run to fail on first call
        call_count = [0]
        original_method = evaluator._evaluate_single_run
        
        def failing_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise EvaluationError("First run fails")
            return original_method(*args, **kwargs)
        
        evaluator._evaluate_single_run = failing_run
        
        # Should complete with warning
        with pytest.warns(UserWarning, match="Run .* failed"):
            results = evaluator.evaluate(learners, env, timestep=0)
        
        # Should have 2 successful runs
        assert results['group1_results']['n_runs'] == 2
        assert results['group2_results']['n_runs'] == 2
    
    def test_evaluation_fails_with_no_successful_runs(self):
        """Test that evaluation fails when no runs succeed."""
        config = EvaluationConfig(n_eval_runs=2, n_episodes_per_run=5)
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment()
        
        # Make all runs fail
        def always_fail(*args, **kwargs):
            raise EvaluationError("All runs fail")
        
        evaluator._evaluate_single_run = always_fail
        
        # Should raise InsufficientDataError
        with pytest.raises(InsufficientDataError, match="No successful evaluation runs completed"):
            evaluator.evaluate(learners, env, timestep=0)
    
    def test_evaluation_uses_configured_seeds(self):
        """Test that evaluation uses seeds from config."""
        custom_seeds = [100, 200, 300]
        config = EvaluationConfig(
            n_eval_runs=3,
            n_episodes_per_run=3,
            seeds=custom_seeds
        )
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment(episode_length=10)
        
        # Track which seeds were used
        used_seeds = []
        original_method = evaluator._evaluate_single_run
        
        def track_seeds(learners, env, run_id, seed):
            used_seeds.append(seed)
            return original_method(learners, env, run_id, seed)
        
        evaluator._evaluate_single_run = track_seeds
        
        evaluator.evaluate(learners, env, timestep=0)
        
        # Check that custom seeds were used
        assert used_seeds == custom_seeds


class TestImprovedEvaluatorIntegration:
    """Integration tests for complete evaluation scenarios."""
    
    def test_evaluation_with_outlier_removal(self):
        """Test full evaluation with outlier removal enabled."""
        config = EvaluationConfig(
            n_eval_runs=2,
            n_episodes_per_run=10,
            n_warmup_episodes=0,
            remove_outliers=True,
            outlier_method='iqr',
            outlier_threshold=1.5
        )
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment(episode_length=10)
        
        results = evaluator.evaluate(learners, env, timestep=0)
        
        # Check that outlier information is present
        assert 'n_outliers_removed' in results['group1_results']
        assert 'outlier_percentage' in results['group1_results']
        assert 'n_outliers_removed' in results['group2_results']
        assert 'outlier_percentage' in results['group2_results']
    
    def test_evaluation_with_high_confidence_level(self):
        """Test evaluation with high confidence level."""
        config = EvaluationConfig(
            n_eval_runs=2,
            n_episodes_per_run=10,
            confidence_level=0.99
        )
        evaluator = ImprovedEvaluator(config)
        learners = [MockLearner(0), MockLearner(1)]
        env = MockEnvironment(episode_length=10)
        
        results = evaluator.evaluate(learners, env, timestep=0)
        
        # Check that confidence intervals are present
        reward_stats = results['group1_results']['reward_stats']
        assert 'ci_lower' in reward_stats
        assert 'ci_upper' in reward_stats
        
        # CI should be wider with 99% confidence
        ci_width = reward_stats['ci_upper'] - reward_stats['ci_lower']
        assert ci_width > 0
    
    def test_evaluation_reproducibility_with_same_seed(self):
        """Test that same seeds produce similar results."""
        config = EvaluationConfig(
            n_eval_runs=1,
            n_episodes_per_run=5,
            seeds=[42]
        )
        
        evaluator1 = ImprovedEvaluator(config)
        evaluator2 = ImprovedEvaluator(config)
        
        learners1 = [MockLearner(0), MockLearner(1)]
        learners2 = [MockLearner(0), MockLearner(1)]
        
        env1 = MockEnvironment(episode_length=10)
        env2 = MockEnvironment(episode_length=10)
        
        results1 = evaluator1.evaluate(learners1, env1, timestep=0)
        results2 = evaluator2.evaluate(learners2, env2, timestep=0)
        
        # Results should have same structure (values may differ due to randomness)
        assert results1['group1_results']['n_episodes_total'] == results2['group1_results']['n_episodes_total']
        assert results1['group2_results']['n_episodes_total'] == results2['group2_results']['n_episodes_total']
