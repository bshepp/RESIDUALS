"""
Tests for the analysis module (differentials and features).

Covers:
- Differential computation correctness
- Feature analysis returns expected keys
- Ranking function returns sorted results
- Edge cases
"""

import numpy as np
import pytest

from src.analysis.differential import (
    compute_differential,
    compute_all_differentials,
    compute_method_differentials,
)
from src.analysis.features import (
    analyze_features,
    rank_for_features,
    compute_feature_snr,
    compute_spatial_autocorrelation,
)


class TestComputeDifferential:
    """Tests for the differential computation function."""

    def test_subtract_method(self):
        a = np.ones((10, 10)) * 5
        b = np.ones((10, 10)) * 3
        diff = compute_differential(a, b, method="subtract")
        np.testing.assert_allclose(diff, 2.0)

    def test_absolute_method(self):
        a = np.ones((10, 10)) * 3
        b = np.ones((10, 10)) * 5
        diff = compute_differential(a, b, method="absolute")
        np.testing.assert_allclose(diff, 2.0)

    def test_normalized_method(self):
        a = np.arange(100, dtype=float).reshape(10, 10)
        b = np.zeros((10, 10))
        diff = compute_differential(a, b, method="normalized")
        # Normalized divides by std, so std of result should be ~1
        assert abs(np.std(diff) - 1.0) < 0.1

    def test_ratio_method(self):
        a = np.ones((10, 10)) * 6
        b = np.ones((10, 10)) * 3
        diff = compute_differential(a, b, method="ratio")
        np.testing.assert_allclose(diff, 2.0, atol=1e-6)

    def test_unknown_method_raises(self):
        a = np.ones((10, 10))
        b = np.ones((10, 10))
        with pytest.raises(ValueError, match="Unknown differential method"):
            compute_differential(a, b, method="bogus")

    def test_different_shapes_cropped(self):
        """Arrays of different shapes should be cropped to minimum."""
        a = np.ones((10, 12))
        b = np.ones((8, 10))
        diff = compute_differential(a, b, method="subtract")
        assert diff.shape == (8, 10)

    def test_identical_arrays_zero_diff(self):
        a = np.random.randn(20, 20)
        diff = compute_differential(a, a.copy(), method="subtract")
        np.testing.assert_allclose(diff, 0.0, atol=1e-15)


class TestAnalyzeFeatures:
    """Tests for the feature analysis function."""

    def test_returns_expected_keys(self, small_dem):
        """Analysis should return all expected metric keys."""
        analysis = analyze_features(small_dem)
        expected_keys = [
            "mean", "std", "min", "max", "range",
            "skewness", "kurtosis",
            "linear_feature_count", "max_linear_strength",
            "spatial_autocorr",
            "low_freq_energy", "mid_freq_energy", "high_freq_energy",
            "feature_snr",
        ]
        for key in expected_keys:
            assert key in analysis, f"Missing key: {key}"

    def test_all_values_are_finite(self, small_dem):
        analysis = analyze_features(small_dem)
        for key, value in analysis.items():
            if isinstance(value, float):
                assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_constant_array_low_std(self):
        """A constant array should have zero std."""
        flat = np.ones((32, 32)) * 100
        analysis = analyze_features(flat)
        assert analysis["std"] == 0.0


class TestRankForFeatures:
    """Tests for the ranking function."""

    def test_returns_sorted_by_score(self, small_dem):
        # Create some fake differentials and analyses
        diffs = {
            "a_vs_b": np.random.randn(32, 32),
            "c_vs_d": np.random.randn(32, 32) * 5,  # more variation
        }
        analyses = {key: analyze_features(diff) for key, diff in diffs.items()}

        ranked = rank_for_features(diffs, analyses)

        assert len(ranked) == 2
        # Should be sorted descending by score
        assert ranked[0][1] >= ranked[1][1]

    def test_returns_tuples_of_three(self, small_dem):
        diffs = {"a_vs_b": np.random.randn(32, 32)}
        analyses = {key: analyze_features(diff) for key, diff in diffs.items()}
        ranked = rank_for_features(diffs, analyses)

        assert len(ranked) == 1
        key, score, analysis = ranked[0]
        assert isinstance(key, str)
        assert isinstance(score, float)
        assert isinstance(analysis, dict)


class TestFeatureMetrics:
    """Tests for individual feature metric functions."""

    def test_spatial_autocorrelation_structured(self):
        """A structured signal should have high spatial autocorrelation."""
        # Create a smooth gradient (high autocorrelation)
        x = np.linspace(0, 10, 64)
        structured = np.outer(x, x)
        moran = compute_spatial_autocorrelation(structured)
        assert moran > 0.5, f"Structured signal should have high Moran's I, got {moran}"

    def test_spatial_autocorrelation_noise(self):
        """Random noise should have low spatial autocorrelation."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((64, 64))
        moran = compute_spatial_autocorrelation(noise)
        assert abs(moran) < 0.1, f"Noise should have low Moran's I, got {moran}"

    def test_feature_snr_constant(self):
        """A constant array should have zero or NaN SNR (no variance)."""
        flat = np.ones((32, 32))
        snr = compute_feature_snr(flat)
        assert snr == 0.0 or np.isnan(snr)
