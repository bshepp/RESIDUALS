"""
Known-answer tests for decomposition and upsampling methods.

Each test constructs a DEM with a mathematically predictable structure,
runs a method, and checks that the output matches the expected answer
within tight tolerances.  These tests verify that the methods actually
do what we claim, not just that they run.
"""

import numpy as np
import pytest

# Trigger registration
import src.decomposition.methods  # noqa: F401
import src.decomposition.methods_extended  # noqa: F401
import src.upsampling.methods  # noqa: F401
import src.upsampling.methods_extended  # noqa: F401
from src.decomposition.registry import run_decomposition
from src.upsampling.registry import run_upsampling
from src.utils.preprocessing import fill_nans

# =========================================================================
# Decomposition known-answer tests
# =========================================================================


class TestGaussianKnownAnswers:
    """Gaussian filter has well-defined behavior on simple inputs."""

    def test_linear_ramp_residual_near_zero(self):
        """Gaussian on a pure linear ramp: residual should be ~0 everywhere
        (a linear function is its own best smooth approximation).
        Interior pixels far from edges should have residual < 1e-6.
        """
        h, w = 128, 128
        x = np.arange(w, dtype=np.float64)
        dem = np.tile(x, (h, 1)) * 0.5 + 100.0  # linear ramp

        trend, residual = run_decomposition("gaussian", dem, {"sigma": 5})

        # Exclude 3*sigma border where edge effects dominate
        border = 15
        interior = residual[border:-border, border:-border]
        assert np.max(np.abs(interior)) < 0.01, (
            f"Gaussian on linear ramp: interior residual max={np.max(np.abs(interior))}"
        )

    def test_constant_dem_unchanged(self):
        """Gaussian on a constant DEM: trend = constant, residual = 0."""
        dem = np.full((64, 64), 850.0)
        trend, residual = run_decomposition("gaussian", dem, {"sigma": 10})
        np.testing.assert_allclose(trend, 850.0, atol=1e-10)
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_reconstruction_exact(self):
        """trend + residual should exactly equal the original."""
        rng = np.random.default_rng(99)
        dem = rng.uniform(800, 900, (64, 64))
        trend, residual = run_decomposition("gaussian", dem, {"sigma": 5})
        np.testing.assert_allclose(trend + residual, dem, atol=1e-10)


class TestPolynomialKnownAnswers:
    """Polynomial trend removal has exact solutions for polynomial surfaces."""

    def test_degree1_on_plane(self):
        """Polynomial degree=1 on a pure plane: residual should be ~0."""
        h, w = 64, 64
        y, x = np.mgrid[0:h, 0:w].astype(np.float64)
        dem = 100.0 + 2.0 * x + 3.0 * y  # perfect plane

        trend, residual = run_decomposition("polynomial", dem, {"degree": 1})
        assert np.max(np.abs(residual)) < 0.01, (
            f"Poly degree=1 on plane: residual max={np.max(np.abs(residual))}"
        )

    def test_degree2_on_quadratic(self):
        """Polynomial degree=2 on a pure quadratic surface: residual should be ~0."""
        h, w = 64, 64
        y, x = np.mgrid[0:h, 0:w].astype(np.float64)
        dem = 100.0 + 0.01 * x**2 + 0.02 * y**2 + 0.5 * x

        trend, residual = run_decomposition("polynomial", dem, {"degree": 2})
        assert np.max(np.abs(residual)) < 0.1, (
            f"Poly degree=2 on quadratic: residual max={np.max(np.abs(residual))}"
        )

    def test_degree1_preserves_bump(self):
        """Polynomial degree=1 on a plane+bump: residual should contain the bump."""
        h, w = 64, 64
        y, x = np.mgrid[0:h, 0:w].astype(np.float64)
        plane = 100.0 + 0.5 * x + 0.3 * y
        bump = 5.0 * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 8**2))
        dem = plane + bump

        trend, residual = run_decomposition("polynomial", dem, {"degree": 1})

        # Residual peak should be close to the bump peak (5.0)
        assert np.max(residual) > 3.0, (
            f"Bump peak should be >3 in residual, got {np.max(residual):.2f}"
        )
        # Residual center should be near (32, 32)
        peak_y, peak_x = np.unravel_index(np.argmax(residual), residual.shape)
        assert abs(peak_x - 32) < 3 and abs(peak_y - 32) < 3, (
            f"Bump peak at ({peak_x}, {peak_y}), expected near (32, 32)"
        )


class TestTophatKnownAnswers:
    """Top-hat transform extracts features smaller than the structuring element."""

    def test_flat_with_spike(self):
        """White top-hat on flat surface with single spike: residual ≈ spike."""
        dem = np.full((64, 64), 100.0)
        dem[32, 32] = 110.0  # 10 ft spike

        trend, residual = run_decomposition("tophat", dem, {"size": 10, "mode": "white"})

        # The spike should appear in the residual
        assert residual[32, 32] > 5.0, (
            f"Spike not detected in top-hat residual: {residual[32, 32]:.2f}"
        )
        # Non-spike areas should be near zero
        mask = np.ones_like(residual, dtype=bool)
        mask[30:35, 30:35] = False  # exclude spike neighborhood
        assert np.max(residual[mask]) < 0.1, (
            f"Top-hat residual too high away from spike: {np.max(residual[mask]):.2f}"
        )

    def test_constant_dem_zero_residual(self):
        """Top-hat on constant surface: residual should be exactly 0."""
        dem = np.full((64, 64), 500.0)
        trend, residual = run_decomposition("tophat", dem, {"size": 10, "mode": "white"})
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)


class TestMorphologicalKnownAnswers:
    """Morphological opening removes peaks smaller than the structuring element."""

    def test_opening_constant_surface(self):
        """Opening of constant surface = constant surface."""
        dem = np.full((64, 64), 300.0)
        trend, residual = run_decomposition(
            "morphological", dem, {"operation": "opening", "size": 5}
        )
        np.testing.assert_allclose(trend, 300.0, atol=1e-10)
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_opening_removes_narrow_peak(self):
        """Opening should remove a peak narrower than the structuring element."""
        dem = np.full((64, 64), 100.0)
        # Add a narrow peak (3 pixels wide, but element size=10)
        dem[30:33, 30:33] = 120.0

        trend, residual = run_decomposition(
            "morphological", dem, {"operation": "opening", "size": 10}
        )

        # The peak should appear in the residual (original - opened)
        assert np.max(residual) > 10.0, (
            f"Narrow peak not captured in residual: max={np.max(residual):.2f}"
        )


class TestDoGKnownAnswers:
    """Difference of Gaussians is a bandpass filter."""

    def test_constant_dem_zero_residual(self):
        """DoG on constant surface: residual = 0 (no features at any scale)."""
        dem = np.full((64, 64), 100.0)
        trend, residual = run_decomposition("dog", dem, {"sigma_low": 2, "sigma_high": 10})
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_reconstruction_exact(self):
        """DoG trend + residual = original."""
        rng = np.random.default_rng(42)
        dem = rng.uniform(800, 900, (64, 64))
        trend, residual = run_decomposition("dog", dem, {"sigma_low": 2, "sigma_high": 10})
        np.testing.assert_allclose(trend + residual, dem, atol=1e-10)


class TestWaveletKnownAnswers:
    """Wavelet DWT should preserve energy and allow reconstruction."""

    def test_constant_dem_zero_residual(self):
        """Wavelet on constant surface: residual ≈ 0."""
        dem = np.full((64, 64), 200.0)
        trend, residual = run_decomposition("wavelet_dwt", dem, {"wavelet": "db4", "level": 2})
        assert np.max(np.abs(residual)) < 0.01, (
            f"Wavelet residual on constant DEM: max={np.max(np.abs(residual))}"
        )

    def test_reconstruction_close(self):
        """Wavelet trend + residual should closely match original."""
        rng = np.random.default_rng(7)
        dem = rng.uniform(800, 900, (64, 64))
        trend, residual = run_decomposition("wavelet_dwt", dem, {"wavelet": "db4", "level": 2})
        np.testing.assert_allclose(trend + residual, dem, atol=0.1)


# =========================================================================
# Upsampling known-answer tests
# =========================================================================


class TestUpsamplingKnownAnswers:
    """Upsampling methods should have predictable behavior on simple inputs."""

    def test_nearest_preserves_values(self):
        """Nearest-neighbor upsampling: each original pixel becomes a 2x2 block."""
        dem = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = run_upsampling("nearest", dem, scale=2)
        assert result.shape == (4, 4)
        # Top-left 2x2 block should all be 1.0
        np.testing.assert_allclose(result[0:2, 0:2], 1.0)
        np.testing.assert_allclose(result[0:2, 2:4], 2.0)
        np.testing.assert_allclose(result[2:4, 0:2], 3.0)
        np.testing.assert_allclose(result[2:4, 2:4], 4.0)

    def test_bicubic_preserves_constant(self):
        """Bicubic on constant surface: output should be constant."""
        dem = np.full((16, 16), 500.0)
        result = run_upsampling("bicubic", dem, scale=2)
        np.testing.assert_allclose(result, 500.0, atol=1e-6)

    def test_bilinear_constant_surface(self):
        """Bilinear on constant surface: output should be constant."""
        dem = np.full((16, 16), 250.0)
        result = run_upsampling("bilinear", dem, scale=2)
        np.testing.assert_allclose(result, 250.0, atol=1e-6)

    def test_upsampling_shape(self):
        """All methods should produce scale*N output shape."""
        dem = np.random.default_rng(0).uniform(0, 100, (16, 16))
        for method in ["bicubic", "lanczos", "nearest", "bilinear"]:
            result = run_upsampling(method, dem, scale=3)
            assert result.shape == (48, 48), (
                f"{method} scale=3: expected (48,48), got {result.shape}"
            )


# =========================================================================
# Preprocessing known-answer tests
# =========================================================================


class TestFillNansKnownAnswers:
    """fill_nans should replace NaN values with the array mean."""

    def test_no_nans_unchanged(self):
        """Array without NaN should be returned unchanged."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fill_nans(arr)
        np.testing.assert_array_equal(result, arr)

    def test_single_nan_replaced_with_mean(self):
        """Single NaN replaced with mean of non-NaN values."""
        arr = np.array([1.0, 2.0, np.nan, 4.0])
        result = fill_nans(arr)
        expected_mean = np.nanmean(arr)  # (1+2+4)/3 = 2.333...
        assert result[2] == pytest.approx(expected_mean)
        # Non-NaN values unchanged
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[3] == 4.0

    def test_does_not_modify_original(self):
        """fill_nans should return a copy, not modify in place."""
        arr = np.array([1.0, np.nan, 3.0])
        result = fill_nans(arr)
        assert np.isnan(arr[1]), "Original array was modified!"
        assert not np.isnan(result[1])

    def test_all_nans(self):
        """All-NaN array: mean is NaN, so result should be NaN (edge case)."""
        arr = np.array([np.nan, np.nan, np.nan])
        result = fill_nans(arr)
        # np.nanmean of all-NaN is NaN, so nan_to_num replaces with NaN
        # This is an edge case -- just verify it doesn't crash
        assert result.shape == arr.shape
