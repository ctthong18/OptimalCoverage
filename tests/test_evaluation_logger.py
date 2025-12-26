"""
Unit tests for EvaluationLogger class.
"""

import pytest
import json
import csv
import os
import tempfile
import shutil
from evaluation_utils import (
    EvaluationLogger,
    EvaluationError,
    AggregatedResults
)


class TestEvaluationLogger:
    """Test suite for EvaluationLogger class."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def logger(self, temp_log_dir):
        """Create an EvaluationLogger instance for testing."""
        return EvaluationLogger(temp_log_dir)
    
    @pytest.fixture
    def sample_results(self):
        """Create sample evaluation results for testing."""
        return {
            'timestamp': '2024-01-01T12:00:00',
            'timestep': 10000,
            'config': {
                'n_eval_runs': 5,
                'n_episodes_per_run': 400,
                'remove_outliers': True,
                'confidence_level': 0.95
            },
            'group1_results': {
                'group_id': 0,
                'n_runs': 5,
                'n_episodes_total': 2000,
                'reward_stats': {
                    'mean': 100.5,
                    'std': 10.2,
                    'ci_lower': 95.3,
                    'ci_upper': 105.7,
                    'cv': 0.101,
                    'min': 80.0,
                    'max': 120.0,
                    'median': 101.0
                },
                'coverage_stats': {
                    'mean': 0.85,
                    'std': 0.05,
                    'ci_lower': 0.82,
                    'ci_upper': 0.88
                },
                'transport_stats': {
                    'mean': 0.75,
                    'std': 0.08,
                    'ci_lower': 0.70,
                    'ci_upper': 0.80
                },
                'all_rewards': [100.0, 102.0, 98.0, 105.0, 95.0],
                'all_lengths': [200, 210, 195, 205, 198],
                'all_coverages': [0.85, 0.87, 0.83, 0.86, 0.84],
                'all_transports': [0.75, 0.77, 0.73, 0.76, 0.74],
                'n_outliers_removed': 2,
                'outlier_percentage': 0.1
            },
            'group2_results': {
                'group_id': 1,
                'n_runs': 5,
                'n_episodes_total': 2000,
                'reward_stats': {
                    'mean': 95.0,
                    'std': 12.0,
                    'ci_lower': 88.5,
                    'ci_upper': 101.5,
                    'cv': 0.126,
                    'min': 75.0,
                    'max': 115.0,
                    'median': 96.0
                },
                'coverage_stats': {
                    'mean': 0.80,
                    'std': 0.06,
                    'ci_lower': 0.76,
                    'ci_upper': 0.84
                },
                'transport_stats': {
                    'mean': 0.70,
                    'std': 0.09,
                    'ci_lower': 0.64,
                    'ci_upper': 0.76
                },
                'all_rewards': [95.0, 97.0, 93.0, 100.0, 90.0],
                'all_lengths': [200, 205, 198, 210, 195],
                'all_coverages': [0.80, 0.82, 0.78, 0.81, 0.79],
                'all_transports': [0.70, 0.72, 0.68, 0.71, 0.69],
                'n_outliers_removed': 3,
                'outlier_percentage': 0.15
            },
            'comparison': {
                'reward_difference': 5.5,
                'reward_difference_percentage': 5.79,
                'p_value': 0.032,
                'effect_size': 0.48,
                'statistical_significance': True
            }
        }
    
    def test_init_creates_directory(self, temp_log_dir):
        """Test that __init__ creates the log directory."""
        log_dir = os.path.join(temp_log_dir, 'new_logs')
        assert not os.path.exists(log_dir)
        
        logger = EvaluationLogger(log_dir)
        
        assert os.path.exists(log_dir)
        assert os.path.isdir(log_dir)
    
    def test_init_with_empty_log_dir_raises_error(self):
        """Test that __init__ raises ValueError for empty log_dir."""
        with pytest.raises(ValueError, match="log_dir cannot be empty"):
            EvaluationLogger("")
    
    def test_save_json_creates_file(self, logger, temp_log_dir, sample_results):
        """Test that save_json creates a JSON file with correct content."""
        filename = 'test_results.json'
        logger.save_json(sample_results, filename)
        
        filepath = os.path.join(temp_log_dir, filename)
        assert os.path.exists(filepath)
        
        # Verify content
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == sample_results
    
    def test_save_json_with_absolute_path(self, logger, temp_log_dir, sample_results):
        """Test that save_json works with absolute paths."""
        filepath = os.path.join(temp_log_dir, 'absolute_test.json')
        logger.save_json(sample_results, filepath)
        
        assert os.path.exists(filepath)
        
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == sample_results
    
    def test_save_json_creates_nested_directories(self, logger, temp_log_dir, sample_results):
        """Test that save_json creates nested directories if needed."""
        filename = 'nested/dir/test_results.json'
        logger.save_json(sample_results, filename)
        
        filepath = os.path.join(temp_log_dir, filename)
        assert os.path.exists(filepath)
        
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == sample_results
    
    def test_save_json_with_invalid_path_raises_error(self, logger):
        """Test that save_json raises EvaluationError for invalid paths."""
        # Try to save to a path that cannot be created (e.g., invalid characters on Windows)
        # This test might be platform-specific, so we'll use a simpler approach
        # by trying to write to a directory that exists as a file
        
        # Create a file
        file_path = os.path.join(logger.log_dir, 'existing_file')
        with open(file_path, 'w') as f:
            f.write('test')
        
        # Try to save JSON to a path that uses this file as a directory
        invalid_path = os.path.join('existing_file', 'test.json')
        
        with pytest.raises(EvaluationError, match="Failed to create directory"):
            logger.save_json({'test': 'data'}, invalid_path)
    
    def test_save_csv_creates_file(self, logger, temp_log_dir):
        """Test that save_csv creates a CSV file with correct content."""
        raw_data = {
            'episode': [1, 2, 3, 4, 5],
            'reward': [100.0, 102.0, 98.0, 105.0, 95.0],
            'length': [200, 210, 195, 205, 198]
        }
        
        filename = 'test_data.csv'
        logger.save_csv(raw_data, filename)
        
        filepath = os.path.join(temp_log_dir, filename)
        assert os.path.exists(filepath)
        
        # Verify content
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 5
        assert rows[0]['episode'] == '1'
        assert rows[0]['reward'] == '100.0'
        assert rows[0]['length'] == '200'
    
    def test_save_csv_with_empty_data_raises_error(self, logger):
        """Test that save_csv raises ValueError for empty data."""
        with pytest.raises(ValueError, match="raw_data cannot be empty"):
            logger.save_csv({}, 'test.csv')
    
    def test_save_csv_with_mismatched_lengths_raises_error(self, logger):
        """Test that save_csv raises ValueError for mismatched column lengths."""
        raw_data = {
            'col1': [1, 2, 3],
            'col2': [4, 5]  # Different length
        }
        
        with pytest.raises(ValueError, match="All columns must have the same length"):
            logger.save_csv(raw_data, 'test.csv')
    
    def test_save_csv_creates_nested_directories(self, logger, temp_log_dir):
        """Test that save_csv creates nested directories if needed."""
        raw_data = {
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        }
        
        filename = 'nested/dir/test_data.csv'
        logger.save_csv(raw_data, filename)
        
        filepath = os.path.join(temp_log_dir, filename)
        assert os.path.exists(filepath)
    
    def test_print_summary_runs_without_error(self, logger, sample_results, capsys):
        """Test that print_summary runs without error and prints expected content."""
        logger.print_summary(sample_results)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check for key elements in output
        assert "EVALUATION SUMMARY" in output
        assert "GROUP 0 RESULTS" in output
        assert "GROUP 1 RESULTS" in output
        assert "GROUP COMPARISON" in output
        assert "Reward Statistics:" in output
        assert "Coverage Statistics:" in output
        assert "Transport Statistics:" in output
        assert "Mean: 100.5000" in output
        assert "P-value: 0.0320" in output
    
    def test_print_summary_with_minimal_results(self, logger, capsys):
        """Test that print_summary works with minimal results."""
        minimal_results = {
            'group1_results': {
                'group_id': 0,
                'reward_stats': {
                    'mean': 100.0,
                    'std': 10.0,
                    'ci_lower': 95.0,
                    'ci_upper': 105.0,
                    'cv': 0.1,
                    'min': 90.0,
                    'max': 110.0,
                    'median': 100.0
                }
            }
        }
        
        logger.print_summary(minimal_results)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "EVALUATION SUMMARY" in output
        assert "GROUP 0 RESULTS" in output
    
    def test_log_results_saves_all_formats(self, logger, temp_log_dir, sample_results):
        """Test that log_results saves JSON, CSV, and prints summary."""
        timestep = 10000
        
        # Remove timestamp from sample_results to test that log_results adds it
        sample_results_copy = sample_results.copy()
        if 'timestamp' in sample_results_copy:
            del sample_results_copy['timestamp']
        if 'timestep' in sample_results_copy:
            del sample_results_copy['timestep']
        
        logger.log_results(sample_results_copy, timestep)
        
        # Check that JSON file was created
        json_files = [f for f in os.listdir(temp_log_dir) if f.endswith('.json')]
        assert len(json_files) == 1
        assert f'evaluation_results_{timestep}' in json_files[0]
        
        # Check that CSV file was created
        csv_files = [f for f in os.listdir(temp_log_dir) if f.endswith('.csv')]
        assert len(csv_files) == 1
        assert f'evaluation_raw_data_{timestep}' in csv_files[0]
        
        # Verify JSON content includes timestamp and timestep
        json_path = os.path.join(temp_log_dir, json_files[0])
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert 'timestamp' in loaded_data
        assert loaded_data['timestep'] == timestep
    
    def test_log_results_handles_missing_raw_data_gracefully(self, logger, temp_log_dir, capsys):
        """Test that log_results handles missing raw data gracefully."""
        results_without_raw_data = {
            'group1_results': {
                'group_id': 0,
                'reward_stats': {
                    'mean': 100.0,
                    'std': 10.0,
                    'ci_lower': 95.0,
                    'ci_upper': 105.0
                }
            }
        }
        
        # Should not raise an error
        logger.log_results(results_without_raw_data, 5000)
        
        # Check that JSON was still created
        json_files = [f for f in os.listdir(temp_log_dir) if f.endswith('.json')]
        assert len(json_files) == 1
        
        # CSV might not be created or might be empty, but should not crash
        captured = capsys.readouterr()
        # Should not have error messages about JSON
        assert "Failed to save JSON" not in captured.out
