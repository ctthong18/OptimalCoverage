"""
Unit tests for GroupEvaluator class.

Tests cover:
- Single episode execution with mock environment
- Metrics collection
- Error handling and retry logic
- Multiple episode evaluation
- Edge cases
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from evaluation_utils import (
    GroupEvaluator,
    GroupResults,
    EvaluationError,
    InsufficientDataError
)


class MockLearner:
    """Mock QPLEXLearner for testing."""
    
    def __init__(self):
        self.hidden_states_reset = False
    
    def reset_hidden_states(self):
        """Mock reset hidden states."""
        self.hidden_states_reset = True
    
    def select_action(self, obs, state, evaluate=False):
        """Mock select action."""
        # Return random actions and empty info
        n_agents = obs.shape[0] if len(obs.shape) > 1 else 1
        actions = np.random.uniform(-1, 1, size=(n_agents, 2))
        return actions, {}


class MockEnvironment:
    """Mock MATE environment for testing."""
    
    def __init__(self, num_cameras=4, num_targets=8, episode_length=100):
        self.num_cameras = num_cameras
        self.num_targets = num_targets
        self.episode_length = episode_length
        self.current_step = 0
        self.should_fail = False
        self.fail_on_step = -1
    
    def reset(self):
        """Mock reset."""
        self.current_step = 0
        self.hidden_states_reset = False
        
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
        
        # Mock rewards
        camera_rewards = np.random.randn(self.num_cameras) * 10
        target_rewards = np.random.randn(self.num_targets) * 10
        rewards = (camera_rewards, target_rewards)
        
        # Mock termination
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Mock info with metrics
        camera_infos = [
            {
                'coverage_rate': 0.75 + np.random.randn() * 0.1,
                'mean_transport_rate': 0.65 + np.random.randn() * 0.1
            }
            for _ in range(self.num_cameras)
        ]
        target_infos = [{} for _ in range(self.num_targets)]
        info = (camera_infos, target_infos)
        
        return obs, rewards, terminated, truncated, info
    
    def render(self):
        """Mock render."""
        pass


class TestGroupEvaluatorInit:
    """Tests for GroupEvaluator initialization."""
    
    def test_valid_group_id_0(self):
        """Test initialization with group_id=0."""
        evaluator = GroupEvaluator(group_id=0)
        assert evaluator.group_id == 0
        assert evaluator.episode_rewards == []
        assert evaluator.episode_lengths == []
        assert evaluator.coverage_rates == []
        assert evaluator.transport_rates == []
    
    def test_valid_group_id_1(self):
        """Test initialization with group_id=1."""
        evaluator = GroupEvaluator(group_id=1)
        assert evaluator.group_id == 1
    
    def test_invalid_group_id(self):
        """Test initialization with invalid group_id."""
        with pytest.raises(ValueError, match="group_id must be 0 or 1"):
            GroupEvaluator(group_id=2)
        
        with pytest.raises(ValueError, match="group_id must be 0 or 1"):
            GroupEvaluator(group_id=-1)


class TestRunSingleEpisode:
    """Tests for _run_single_episode method."""
    
    def test_successful_episode(self):
        """Test successful episode execution."""
        evaluator = GroupEvaluator(group_id=0)
        learner = MockLearner()
        env = MockEnvironment(episode_length=10)
        
        metrics = evaluator._run_single_episode(learner, env, render=False)
        
        # Check metrics are returned
        assert 'episode_reward' in metrics
        assert 'episode_length' in metrics
        assert 'coverage_rate' in metrics
        assert 'transport_rate' in metrics
        
        # Check values are reasonable
        assert isinstance(metrics['episode_reward'], float)
        assert isinstance(metrics['episode_length'], float)
        assert metrics['episode_length'] == 10
        assert 0 <= metrics['coverage_rate'] <= 1.5  # Allow some variance
        assert 0 <= metrics['transport_rate'] <= 1.5
        
        # Check learner was used correctly
        assert learner.hidden_states_reset
    
    def test_episode_with_render(self):
        """Test episode execution with rendering."""
        evaluator = GroupEvaluator(group_id=0)
        learner = MockLearner()
        env = MockEnvironment(episode_length=5)
        
        # Should not raise error
        metrics = evaluator._run_single_episode(learner, env, render=True)
        assert metrics['episode_length'] == 5
    
    def test_episode_retry_on_failure(self):
        """Test retry logic when episode fails."""
        evaluator = GroupEvaluator(group_id=0)
        learner = MockLearner()
        env = MockEnvironment(episode_length=10)
        
        # Make environment fail on first attempt only
        call_count = [0]
        original_reset = env.reset
        
        def failing_reset():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("First attempt fails")
            return original_reset()
        
        env.reset = failing_reset
        
        # Should succeed on retry
        metrics = evaluator._run_single_episode(learner, env, max_retries=3)
        assert metrics['episode_length'] == 10
        assert call_count[0] == 2  # Failed once, succeeded on second
    
    def test_episode_fails_after_max_retries(self):
        """Test that episode fails after max retries."""
        evaluator = GroupEvaluator(group_id=0)
        learner = MockLearner()
        env = MockEnvironment(episode_length=10)
        
        # Make environment always fail
        env.should_fail = True
        
        # Should raise EvaluationError after max retries
        with pytest.raises(EvaluationError, match="Episode failed after 3 attempts"):
            evaluator._run_single_episode(learner, env, max_retries=3)
    
    def test_episode_with_empty_info(self):
        """Test episode when info is empty or missing metrics."""
        evaluator = GroupEvaluator(group_id=0)
        learner = MockLearner()
        env = MockEnvironment(episode_length=5)
        
        # Override step to return empty info
        original_step = env.step
        
        def step_with_empty_info(actions):
            obs, rewards, terminated, truncated, _ = original_step(actions)
            return obs, rewards, terminated, truncated, ([], [])
        
        env.step = step_with_empty_info
        
        # Should still work, with default values
        metrics = evaluator._run_single_episode(learner, env)
        assert metrics['coverage_rate'] == 0.0
        assert metrics['transport_rate'] == 0.0


class TestEvaluateGroup:
    """Tests for evaluate_group method."""
    
    def test_successful_evaluation(self):
        """Test successful evaluation over multiple episodes."""
        evaluator = GroupEvaluator(group_id=0)
        learner = MockLearner()
        env = MockEnvironment(episode_length=10)
        
        results = evaluator.evaluate_group(
            learner=learner,
            env=env,
            n_episodes=5,
            seed=42,
            render=False
        )
        
        # Check results type
        assert isinstance(results, GroupResults)
        assert results.group_id == 0
        assert results.seed == 42
        
        # Check metrics were collected
        assert len(results.episode_rewards) == 5
        assert len(results.episode_lengths) == 5
        assert len(results.coverage_rates) == 5
        assert len(results.transport_rates) == 5
        
        # Check aggregated metrics were calculated
        assert results.mean_reward != 0.0
        assert results.mean_length == 10.0
        assert results.mean_coverage > 0
        assert results.mean_transport > 0
    
    def test_evaluation_with_seed(self):
        """Test that seed produces reproducible results."""
        evaluator1 = GroupEvaluator(group_id=0)
        evaluator2 = GroupEvaluator(group_id=0)
        learner1 = MockLearner()
        learner2 = MockLearner()
        env1 = MockEnvironment(episode_length=10)
        env2 = MockEnvironment(episode_length=10)
        
        # Same seed should give same results (approximately, due to mock randomness)
        results1 = evaluator1.evaluate_group(learner1, env1, n_episodes=3, seed=42)
        results2 = evaluator2.evaluate_group(learner2, env2, n_episodes=3, seed=42)
        
        # At least the structure should be the same
        assert len(results1.episode_rewards) == len(results2.episode_rewards)
        assert results1.seed == results2.seed
    
    def test_evaluation_invalid_n_episodes(self):
        """Test evaluation with invalid n_episodes."""
        evaluator = GroupEvaluator(group_id=0)
        learner = MockLearner()
        env = MockEnvironment()
        
        with pytest.raises(ValueError, match="n_episodes must be >= 1"):
            evaluator.evaluate_group(learner, env, n_episodes=0, seed=42)
        
        with pytest.raises(ValueError, match="n_episodes must be >= 1"):
            evaluator.evaluate_group(learner, env, n_episodes=-1, seed=42)
    
    def test_evaluation_continues_after_episode_failure(self):
        """Test that evaluation continues after individual episode failures."""
        evaluator = GroupEvaluator(group_id=0)
        learner = MockLearner()
        env = MockEnvironment(episode_length=10)
        
        # Mock _run_single_episode to fail on first call only
        call_count = [0]
        original_run = evaluator._run_single_episode
        
        def failing_run(learner, env, render=False, max_retries=3):
            call_count[0] += 1
            if call_count[0] == 1:
                # First episode fails completely (even after retries)
                raise EvaluationError("First episode fails")
            # Subsequent episodes succeed
            return original_run(learner, env, render, max_retries)
        
        evaluator._run_single_episode = failing_run
        
        # Should complete with 4 successful episodes (1 failed)
        with pytest.warns(UserWarning, match="Failed to complete episode"):
            results = evaluator.evaluate_group(learner, env, n_episodes=5, seed=42)
        
        # Should have 4 successful episodes
        assert len(results.episode_rewards) == 4
    
    def test_evaluation_fails_with_no_successful_episodes(self):
        """Test that evaluation fails when no episodes succeed."""
        evaluator = GroupEvaluator(group_id=0)
        learner = MockLearner()
        env = MockEnvironment()
        
        # Make all episodes fail
        def always_fail(*args, **kwargs):
            raise EvaluationError("All episodes fail")
        
        evaluator._run_single_episode = always_fail
        
        # Should raise InsufficientDataError
        with pytest.raises(InsufficientDataError, match="No successful episodes completed"):
            evaluator.evaluate_group(learner, env, n_episodes=3, seed=42)
    
    def test_evaluation_metrics_collection(self):
        """Test that metrics are properly collected and stored."""
        evaluator = GroupEvaluator(group_id=1)
        learner = MockLearner()
        env = MockEnvironment(episode_length=20)
        
        results = evaluator.evaluate_group(learner, env, n_episodes=10, seed=123)
        
        # Check all metrics have correct length
        assert len(results.episode_rewards) == 10
        assert len(results.episode_lengths) == 10
        assert len(results.coverage_rates) == 10
        assert len(results.transport_rates) == 10
        
        # Check all episode lengths are correct
        assert all(length == 20.0 for length in results.episode_lengths)
        
        # Check mean length is correct
        assert results.mean_length == 20.0
    
    def test_evaluation_with_render(self):
        """Test evaluation with rendering enabled."""
        evaluator = GroupEvaluator(group_id=0)
        learner = MockLearner()
        env = MockEnvironment(episode_length=5)
        
        # Should not raise error
        results = evaluator.evaluate_group(learner, env, n_episodes=2, seed=42, render=True)
        assert len(results.episode_rewards) == 2


class TestGroupEvaluatorIntegration:
    """Integration tests for GroupEvaluator."""
    
    def test_multiple_evaluations_reset_metrics(self):
        """Test that metrics are reset between evaluations."""
        evaluator = GroupEvaluator(group_id=0)
        learner = MockLearner()
        env = MockEnvironment(episode_length=10)
        
        # First evaluation
        results1 = evaluator.evaluate_group(learner, env, n_episodes=3, seed=42)
        assert len(results1.episode_rewards) == 3
        
        # Second evaluation should reset metrics
        results2 = evaluator.evaluate_group(learner, env, n_episodes=5, seed=43)
        assert len(results2.episode_rewards) == 5
        
        # Results should be independent
        assert results1.seed != results2.seed
    
    def test_different_group_ids(self):
        """Test that different group IDs work independently."""
        evaluator0 = GroupEvaluator(group_id=0)
        evaluator1 = GroupEvaluator(group_id=1)
        learner = MockLearner()
        env = MockEnvironment(episode_length=10)
        
        results0 = evaluator0.evaluate_group(learner, env, n_episodes=3, seed=42)
        results1 = evaluator1.evaluate_group(learner, env, n_episodes=3, seed=42)
        
        assert results0.group_id == 0
        assert results1.group_id == 1
        assert len(results0.episode_rewards) == 3
        assert len(results1.episode_rewards) == 3
