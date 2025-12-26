"""
Unit tests for StatisticsCalculator class.

Tests cover:
- Outlier removal with IQR and Z-score methods
- Confidence interval calculation
- Convergence metrics calculation
- Summary statistics calculation
- Edge cases and error handling
"""

import pytest
import numpy as np
from evaluation_utils import StatisticsCalculator


class TestRemoveOutliers:
    """Tests for remove_outliers method."""
    
    def test_iqr_method_basic(self):
        """Test IQR method with known data."""
        # Data with clear outliers
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        cleaned, mask = StatisticsCalculator.remove_outliers(data, method='iqr', threshold=1.5)
        
        # 100 should be detected as outlier
        assert len(cleaned) < len(data)
        assert 100 not in cleaned
        assert mask[-1] == True  # Last element is outlier
    
    def test_iqr_method_no_outliers(self):
        """Test IQR method when no outliers present."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cleaned, mask = StatisticsCalculator.remove_outliers(data, method='iqr', threshold=1.5)
        
        # No outliers should be detected
        assert len(cleaned) == len(data)
        assert np.all(~mask)
    
    def test_zscore_method_basic(self):
        """Test Z-score method with known data."""
        # Data with outliers - use lower threshold to detect outlier
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 50])
        cleaned, mask = StatisticsCalculator.remove_outliers(data, method='zscore', threshold=2.0)
        
        # 50 should be detected as outlier with threshold=2.0
        assert len(cleaned) < len(data)
        assert 50 not in cleaned
    
    def test_zscore_method_no_outliers(self):
        """Test Z-score method when no outliers present."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cleaned, mask = StatisticsCalculator.remove_outliers(data, method='zscore', threshold=3.0)
        
        # No outliers should be detected with high threshold
        assert len(cleaned) == len(data)
        assert np.all(~mask)
    
    def test_empty_data(self):
        """Test with empty data."""
        data = np.array([])
        with pytest.raises(ValueError, match="Cannot remove outliers from empty data"):
            StatisticsCalculator.remove_outliers(data)
    
    def test_all_nan_data(self):
        """Test with all NaN data."""
        data = np.array([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError, match="Cannot remove outliers from all-NaN data"):
            StatisticsCalculator.remove_outliers(data)
    
    def test_data_with_nan(self):
        """Test with data containing NaN values."""
        data = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 100])
        cleaned, mask = StatisticsCalculator.remove_outliers(data, method='iqr')
        
        # NaN should be removed, and outliers detected
        assert not np.any(np.isnan(cleaned))
        assert len(cleaned) < 9  # Less than valid data count
    
    def test_identical_values(self):
        """Test with all identical values (std=0)."""
        data = np.array([5, 5, 5, 5, 5])
        cleaned, mask = StatisticsCalculator.remove_outliers(data, method='zscore')
        
        # No outliers when all values are identical
        assert len(cleaned) == len(data)
        assert np.all(~mask)
    
    def test_invalid_method(self):
        """Test with invalid method."""
        data = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Unknown method"):
            StatisticsCalculator.remove_outliers(data, method='invalid')


class TestCalculateConfidenceInterval:
    """Tests for calculate_confidence_interval method."""
    
    def test_basic_calculation(self):
        """Test basic confidence interval calculation."""
        # Known data
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        mean, lower, upper = StatisticsCalculator.calculate_confidence_interval(data, confidence=0.95)
        
        # Check mean is correct
        assert np.isclose(mean, 5.5)
        
        # Check bounds are symmetric around mean
        assert lower < mean < upper
        assert np.isclose(mean - lower, upper - mean)
    
    def test_high_confidence(self):
        """Test that higher confidence gives wider interval."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        mean_95, lower_95, upper_95 = StatisticsCalculator.calculate_confidence_interval(data, confidence=0.95)
        mean_99, lower_99, upper_99 = StatisticsCalculator.calculate_confidence_interval(data, confidence=0.99)
        
        # Same mean
        assert np.isclose(mean_95, mean_99)
        
        # 99% CI should be wider than 95% CI
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        assert width_99 > width_95
    
    def test_small_sample(self):
        """Test with small sample size."""
        data = np.array([1, 2])
        mean, lower, upper = StatisticsCalculator.calculate_confidence_interval(data)
        
        # Should work with 2 samples
        assert lower < mean < upper
    
    def test_single_sample_error(self):
        """Test that single sample raises error."""
        data = np.array([5])
        with pytest.raises(ValueError, match="Need at least 2 samples"):
            StatisticsCalculator.calculate_confidence_interval(data)
    
    def test_empty_data_error(self):
        """Test that empty data raises error."""
        data = np.array([])
        with pytest.raises(ValueError, match="Need at least 2 samples"):
            StatisticsCalculator.calculate_confidence_interval(data)
    
    def test_invalid_confidence(self):
        """Test with invalid confidence level."""
        data = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            StatisticsCalculator.calculate_confidence_interval(data, confidence=1.5)
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            StatisticsCalculator.calculate_confidence_interval(data, confidence=0)
    
    def test_with_nan_values(self):
        """Test with NaN values in data."""
        data = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        mean, lower, upper = StatisticsCalculator.calculate_confidence_interval(data)
        
        # Should ignore NaN and calculate correctly
        assert not np.isnan(mean)
        assert lower < mean < upper


class TestCalculateConvergenceMetrics:
    """Tests for calculate_convergence_metrics method."""
    
    def test_basic_metrics(self):
        """Test basic convergence metrics calculation."""
        data = np.array([10, 12, 11, 13, 9, 10, 11, 12])
        metrics = StatisticsCalculator.calculate_convergence_metrics(data)
        
        # Check all expected keys are present
        assert 'cv' in metrics
        assert 'stability_score' in metrics
        assert 'range' in metrics
        assert 'relative_range' in metrics
        assert 'std' in metrics
        assert 'mean' in metrics
        
        # Check values are reasonable
        assert metrics['cv'] > 0
        assert 0 <= metrics['stability_score'] <= 1
        assert metrics['range'] > 0
    
    def test_perfect_convergence(self):
        """Test with perfectly converged data (all same values)."""
        data = np.array([5, 5, 5, 5, 5])
        metrics = StatisticsCalculator.calculate_convergence_metrics(data)
        
        # CV should be 0 for identical values
        assert metrics['cv'] == 0
        assert metrics['stability_score'] == 1.0
        assert metrics['range'] == 0
        assert metrics['std'] == 0
    
    def test_high_variance(self):
        """Test with high variance data."""
        data = np.array([1, 100, 2, 99, 3, 98])
        metrics = StatisticsCalculator.calculate_convergence_metrics(data)
        
        # CV should be high
        assert metrics['cv'] > 0.5
        # Stability score should be low
        assert metrics['stability_score'] < 0.5
    
    def test_zero_mean(self):
        """Test with data that has zero mean."""
        data = np.array([-5, -3, 0, 3, 5])
        metrics = StatisticsCalculator.calculate_convergence_metrics(data)
        
        # CV should be inf when mean is 0
        assert metrics['cv'] == float('inf')
        assert metrics['stability_score'] == 0.0
    
    def test_empty_data_error(self):
        """Test with empty data."""
        data = np.array([])
        with pytest.raises(ValueError, match="Cannot calculate convergence metrics from empty data"):
            StatisticsCalculator.calculate_convergence_metrics(data)
    
    def test_with_nan_values(self):
        """Test with NaN values."""
        data = np.array([10, np.nan, 12, 11, np.nan, 13])
        metrics = StatisticsCalculator.calculate_convergence_metrics(data)
        
        # Should ignore NaN and calculate correctly
        assert not np.isnan(metrics['cv'])
        assert not np.isnan(metrics['mean'])


class TestCalculateSummaryStatistics:
    """Tests for calculate_summary_statistics method."""
    
    def test_basic_statistics(self):
        """Test basic summary statistics calculation."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stats = StatisticsCalculator.calculate_summary_statistics(data)
        
        # Check all expected keys
        assert 'count' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'q1' in stats
        assert 'q3' in stats
        assert 'iqr' in stats
        
        # Check values
        assert stats['count'] == 10
        assert stats['min'] == 1
        assert stats['max'] == 10
        assert np.isclose(stats['mean'], 5.5)
        assert np.isclose(stats['median'], 5.5)
        assert np.isclose(stats['q1'], 3.25)
        assert np.isclose(stats['q3'], 7.75)
        assert np.isclose(stats['iqr'], 4.5)
    
    def test_odd_count(self):
        """Test with odd number of samples."""
        data = np.array([1, 2, 3, 4, 5])
        stats = StatisticsCalculator.calculate_summary_statistics(data)
        
        assert stats['count'] == 5
        assert stats['median'] == 3
    
    def test_single_value(self):
        """Test with single value."""
        data = np.array([42])
        stats = StatisticsCalculator.calculate_summary_statistics(data)
        
        assert stats['count'] == 1
        assert stats['min'] == 42
        assert stats['max'] == 42
        assert stats['mean'] == 42
        assert stats['median'] == 42
        # std is NaN for single value with ddof=1
        assert np.isnan(stats['std'])
        assert stats['iqr'] == 0
    
    def test_empty_data_error(self):
        """Test with empty data."""
        data = np.array([])
        with pytest.raises(ValueError, match="Cannot calculate summary statistics from empty data"):
            StatisticsCalculator.calculate_summary_statistics(data)
    
    def test_with_nan_values(self):
        """Test with NaN values."""
        data = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10])
        stats = StatisticsCalculator.calculate_summary_statistics(data)
        
        # Should ignore NaN
        assert stats['count'] == 8
        assert not np.isnan(stats['mean'])
        assert not np.isnan(stats['median'])
    
    def test_negative_values(self):
        """Test with negative values."""
        data = np.array([-10, -5, 0, 5, 10])
        stats = StatisticsCalculator.calculate_summary_statistics(data)
        
        assert stats['min'] == -10
        assert stats['max'] == 10
        assert stats['mean'] == 0
        assert stats['median'] == 0
