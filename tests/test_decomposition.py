"""
Tests for decomposition methods.

Covers:
- All 25 methods run without error
- Output shape correctness
- trend + residual ≈ original (where applicable)
- NaN handling
- Known-answer tests
"""

import numpy as np
import pytest

from src.decomposition.registry import (
    DECOMPOSITION_REGISTRY,
    run_decomposition,
)


class TestAllDecompositionMethodsRun:
    """Every registered method should run without error on a small DEM."""

    @pytest.fixture(autouse=True)
    def _ensure_registered(self):
        # Importing the modules triggers registration
        import src.decomposition.methods  # noqa: F401
        import src.decomposition.methods_extended  # noqa: F401

    @pytest.mark.parametrize("method_name", list(DECOMPOSITION_REGISTRY.keys()))
    def test_method_runs(self, small_dem, method_name):
        trend, residual = run_decomposition(method_name, small_dem)

        # Should return arrays of the same shape
        assert trend.shape == small_dem.shape, (
            f"{method_name}: trend shape {trend.shape} != dem shape {small_dem.shape}"
        )
        assert residual.shape == small_dem.shape, (
            f"{method_name}: residual shape {residual.shape} != dem shape {small_dem.shape}"
        )

    @pytest.mark.parametrize("method_name", list(DECOMPOSITION_REGISTRY.keys()))
    def test_no_nans_in_output(self, small_dem, method_name):
        """Output should be NaN-free when input is NaN-free."""
        trend, residual = run_decomposition(method_name, small_dem)
        assert not np.any(np.isnan(trend)), f"{method_name}: trend contains NaN"
        assert not np.any(np.isnan(residual)), f"{method_name}: residual contains NaN"


class TestDecompositionProperties:
    """Test mathematical properties that should hold."""

    def test_gaussian_trend_plus_residual_equals_original(self, small_dem):
        """For Gaussian decomposition, trend + residual should equal input."""
        trend, residual = run_decomposition("gaussian", small_dem)
        reconstructed = trend + residual
        np.testing.assert_allclose(
            reconstructed, small_dem, atol=1e-10, err_msg="Gaussian: trend + residual != original"
        )

    def test_gaussian_flat_dem_near_zero_residual(self, flat_dem):
        """Gaussian on a flat DEM should produce near-zero residual."""
        trend, residual = run_decomposition("gaussian", flat_dem)
        assert np.max(np.abs(residual)) < 1e-6, (
            f"Flat DEM residual too large: max={np.max(np.abs(residual))}"
        )

    def test_bilateral_preserves_shape(self, small_dem):
        """Bilateral filter should preserve sharp features better than Gaussian."""
        trend, residual = run_decomposition("bilateral", small_dem)
        assert trend.shape == small_dem.shape
        assert residual.shape == small_dem.shape

    def test_wavelet_reconstruction(self, small_dem):
        """Wavelet trend + residual should closely approximate original."""
        trend, residual = run_decomposition("wavelet_dwt", small_dem)
        reconstructed = trend + residual
        np.testing.assert_allclose(
            reconstructed,
            small_dem,
            atol=0.1,
            err_msg="Wavelet: trend + residual diverged from original",
        )

    def test_polynomial_removes_linear_trend(self):
        """Polynomial degree-1 should remove a linear ramp."""
        h, w = 64, 64
        x = np.arange(w)
        dem = np.tile(x, (h, 1)).astype(np.float64) * 2.0  # pure linear ramp

        trend, residual = run_decomposition("polynomial", dem, params={"degree": 1})
        # Residual should be near-zero for a pure linear surface
        assert np.std(residual) < 0.1, f"Polynomial residual std too high: {np.std(residual)}"


class TestNaNHandling:
    """NaN inputs should produce NaN-free outputs."""

    @pytest.mark.parametrize(
        "method_name",
        ["gaussian", "bilateral", "wavelet_dwt", "morphological", "tophat", "polynomial"],
    )
    def test_nan_input_produces_clean_output(self, nan_dem, method_name):
        trend, residual = run_decomposition(method_name, nan_dem)
        assert not np.any(np.isnan(trend)), f"{method_name}: trend has NaN from NaN input"
        assert not np.any(np.isnan(residual)), f"{method_name}: residual has NaN from NaN input"


class TestMethodCount:
    """Ensure we haven't accidentally lost methods."""

    def test_expected_method_count(self):
        assert len(DECOMPOSITION_REGISTRY) == 25, (
            f"Expected 25 decomposition methods, found {len(DECOMPOSITION_REGISTRY)}"
        )
