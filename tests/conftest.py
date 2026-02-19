"""
Shared test fixtures for RESIDUALS test suite.
"""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def small_dem():
    """A small 64x64 synthetic DEM with known features.

    Contains:
    - A flat background at elevation 100
    - A linear ramp across the X axis (regional trend)
    - A Gaussian bump at (32, 32) simulating a mound
    """
    h, w = 64, 64
    dem = np.full((h, w), 100.0, dtype=np.float64)

    # Add a gentle X-direction slope (regional trend)
    x = np.arange(w)
    dem += x[np.newaxis, :] * 0.5  # 0 to 31.5 ft rise

    # Add a Gaussian bump (local feature)
    y, x = np.ogrid[:h, :w]
    bump = 10.0 * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 8**2))
    dem += bump

    return dem


@pytest.fixture
def flat_dem():
    """A perfectly flat 32x32 DEM at constant elevation."""
    return np.full((32, 32), 500.0, dtype=np.float64)


@pytest.fixture
def nan_dem(small_dem):
    """A DEM with scattered NaN values."""
    dem = small_dem.copy()
    rng = np.random.default_rng(42)
    nan_mask = rng.random(dem.shape) < 0.05  # ~5% NaN
    dem[nan_mask] = np.nan
    return dem


@pytest.fixture
def test_dem_path():
    """Path to the included Fairfield test DEM."""
    path = Path(__file__).parent.parent / "data" / "test_dems" / "fairfield_sample_1.5ft.npy"
    if not path.exists():
        pytest.skip(f"Test DEM not found: {path}")
    return path
