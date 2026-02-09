"""
Preprocessing utilities for DEM data.

Common operations applied before decomposition/upsampling methods.
"""

import numpy as np


def fill_nans(dem: np.ndarray) -> np.ndarray:
    """Replace NaN values with the array mean.

    Standard preprocessing step for all decomposition and upsampling methods.
    Returns a copy with NaN values replaced; the original array is not modified.

    Args:
        dem: Input DEM array, possibly containing NaN values.

    Returns:
        Array with NaN values replaced by the mean of non-NaN values.
    """
    return np.nan_to_num(dem, nan=np.nanmean(dem))
