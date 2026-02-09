"""
Tests for upsampling methods.

Covers:
- All 19 methods run without error
- Output shape correctness (scale factor applied)
- NaN handling
- Known-answer test for nearest-neighbor
"""

import numpy as np
import pytest

from src.upsampling.registry import (
    UPSAMPLING_REGISTRY,
    run_upsampling,
    list_upsamplings,
)


# Small DEM for upsampling tests (upsampling can be memory-heavy)
@pytest.fixture
def tiny_dem():
    """A very small 16x16 DEM for upsampling tests."""
    rng = np.random.default_rng(42)
    return rng.random((16, 16)) * 100 + 500


@pytest.fixture
def tiny_nan_dem(tiny_dem):
    """Tiny DEM with a few NaN values."""
    dem = tiny_dem.copy()
    dem[0, 0] = np.nan
    dem[8, 8] = np.nan
    return dem


class TestAllUpsamplingMethodsRun:
    """Every registered method should run without error."""

    @pytest.fixture(autouse=True)
    def _ensure_registered(self):
        import src.upsampling.methods  # noqa: F401
        import src.upsampling.methods_extended  # noqa: F401

    @pytest.mark.parametrize("method_name", list(UPSAMPLING_REGISTRY.keys()))
    def test_method_runs(self, tiny_dem, method_name):
        # Use scale=2 for all methods (safe default)
        result = run_upsampling(method_name, tiny_dem, scale=2)
        assert result is not None
        assert result.ndim == 2

    @pytest.mark.parametrize("method_name", list(UPSAMPLING_REGISTRY.keys()))
    def test_no_nans_in_output(self, tiny_dem, method_name):
        result = run_upsampling(method_name, tiny_dem, scale=2)
        assert not np.any(np.isnan(result)), f"{method_name}: output contains NaN"


class TestUpsamplingOutputShape:
    """Output dimensions should match scale factor."""

    @pytest.mark.parametrize("method_name", [
        "bicubic", "lanczos", "bspline", "fft_zeropad",
        "nearest", "bilinear", "quadratic",
    ])
    def test_2x_scale_doubles_dimensions(self, tiny_dem, method_name):
        result = run_upsampling(method_name, tiny_dem, scale=2)
        expected_h = tiny_dem.shape[0] * 2
        expected_w = tiny_dem.shape[1] * 2
        assert result.shape == (expected_h, expected_w), (
            f"{method_name}: expected {(expected_h, expected_w)}, got {result.shape}"
        )


class TestUpsamplingKnownAnswers:
    """Known-answer tests for predictable methods."""

    def test_nearest_neighbor_block_pattern(self):
        """Nearest-neighbor 2x should produce 2x2 blocks."""
        dem = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
        result = run_upsampling("nearest", dem, scale=2)

        assert result.shape == (4, 4)
        # Each pixel should be replicated into a 2x2 block
        assert result[0, 0] == 1.0
        assert result[0, 1] == 1.0
        assert result[1, 0] == 1.0
        assert result[1, 1] == 1.0
        assert result[2, 2] == 4.0
        assert result[3, 3] == 4.0

    def test_bicubic_preserves_constant(self):
        """Bicubic on a constant surface should return the same constant."""
        dem = np.full((8, 8), 42.0)
        result = run_upsampling("bicubic", dem, scale=2)
        np.testing.assert_allclose(result, 42.0, atol=1e-6)


class TestNaNHandling:
    """NaN inputs should produce NaN-free outputs."""

    @pytest.mark.parametrize("method_name", ["bicubic", "lanczos", "bspline", "nearest"])
    def test_nan_input_produces_clean_output(self, tiny_nan_dem, method_name):
        result = run_upsampling(method_name, tiny_nan_dem, scale=2)
        assert not np.any(np.isnan(result)), f"{method_name}: output has NaN from NaN input"


class TestMethodCount:
    """Ensure we haven't accidentally lost methods."""

    def test_expected_method_count(self):
        assert len(UPSAMPLING_REGISTRY) == 19, (
            f"Expected 19 upsampling methods, found {len(UPSAMPLING_REGISTRY)}"
        )
