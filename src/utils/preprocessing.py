"""
Preprocessing utilities for DEM data.

Common operations applied before decomposition/upsampling methods.
"""

import numpy as np
from scipy.ndimage import zoom


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


def roundtrip_resample(
    arr: np.ndarray,
    factor: float,
    down_order: int = 1,
    up_order: int = 3,
    transform=None,
    target_shape: tuple = None,
) -> np.ndarray:
    """Downsample arr by 1/factor, optionally transform on the coarse grid,
    then upsample back by factor.

    Used by decompositions that operate on a coarser grid for memory or
    speed reasons (e.g. rolling_ball at large radii). Returning the
    round-tripped array as a single utility makes the two interpolation
    orders visible to callers — they used to be hardcoded inside callers,
    invisible to any parameter sweep or reproducibility log.

    Args:
        arr: 2-D input array.
        factor: Downsample factor (>1 means coarser intermediate grid).
        down_order: scipy.ndimage.zoom interpolation order on the way down.
            Default 1 (linear).
        up_order: scipy.ndimage.zoom interpolation order on the way up.
            Default 3 (cubic). down_order and up_order are deliberately
            independent — callers can pick asymmetric pairs that match
            their tolerance for smoothing vs. ringing in each direction.
        transform: Optional callable applied to the downsampled array
            before upsampling, signature transform(small_arr) -> small_arr.
            Lets callers do work on the coarse grid (e.g. morphological
            opening at a smaller effective radius) while keeping the
            resample bookkeeping in one place.
        target_shape: Optional (h, w) to crop/pad the output to. Useful
            when zoom rounding produces an off-by-one mismatch with the
            input shape and the caller needs an exact size match.

    Returns:
        Resampled 2-D array. Shape matches target_shape if given,
        otherwise approximately matches the original (may differ by 1
        pixel per axis due to zoom rounding at non-integer factors).
    """
    small = zoom(arr, 1.0 / factor, order=down_order)
    if transform is not None:
        small = transform(small)
    upscaled = zoom(small, factor, order=up_order)
    if target_shape is not None and upscaled.shape != target_shape:
        out = np.zeros(target_shape, dtype=upscaled.dtype)
        min_h = min(upscaled.shape[0], target_shape[0])
        min_w = min(upscaled.shape[1], target_shape[1])
        out[:min_h, :min_w] = upscaled[:min_h, :min_w]
        return out
    return upscaled
