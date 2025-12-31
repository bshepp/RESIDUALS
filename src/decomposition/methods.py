"""
Core Decomposition Methods

Implements 6 key decomposition methods from different categories:
- Gaussian (Classical)
- Bilateral (Edge-preserving)
- Wavelet DWT (Multi-scale)
- Morphological Opening (Shape-based)
- Top-Hat (Small features)
- Polynomial (Trend removal)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, grey_opening
from scipy.ndimage import white_tophat
from skimage.morphology import disk
import cv2
import pywt

from .registry import register_decomposition


# =============================================================================
# Classical Signal Processing
# =============================================================================

@register_decomposition(
    name='gaussian',
    category='classical',
    default_params={'sigma': 10},
    param_ranges={'sigma': [2, 5, 10, 20, 50, 100]},
    preserves='smooth regions, large-scale features',
    destroys='all high-frequency equally (isotropic)'
)
def decompose_gaussian(dem: np.ndarray, sigma: float = 10) -> tuple:
    """
    Gaussian low-pass filtering for trend extraction.
    
    Simple and fast baseline method. Treats all directions equally.
    """
    # Handle NaN values
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    trend = gaussian_filter(dem_filled, sigma=sigma)
    residual = dem_filled - trend
    
    return trend, residual


@register_decomposition(
    name='bilateral',
    category='edge_preserving',
    default_params={'d': 9, 'sigma_color': 75, 'sigma_space': 75},
    param_ranges={
        'd': [5, 9, 15, 21],
        'sigma_color': [25, 50, 75, 100, 150],
        'sigma_space': [25, 50, 75, 100, 150]
    },
    preserves='sharp edges, discontinuities (roads, walls)',
    destroys='gradual transitions in flat areas, texture'
)
def decompose_bilateral(
    dem: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> tuple:
    """
    Bilateral filtering - edge-preserving smoothing.
    
    Smooths while preserving edges. Good for roads, walls, embankments.
    """
    # Handle NaN values
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    # Normalize to 0-255 range for cv2
    dem_min, dem_max = dem_filled.min(), dem_filled.max()
    dem_range = dem_max - dem_min
    if dem_range == 0:
        dem_range = 1
    
    dem_norm = ((dem_filled - dem_min) / dem_range * 255).astype(np.float32)
    
    # Apply bilateral filter
    trend_norm = cv2.bilateralFilter(dem_norm, d, sigma_color, sigma_space)
    
    # Rescale back to original range
    trend = trend_norm / 255 * dem_range + dem_min
    residual = dem_filled - trend
    
    return trend, residual


# =============================================================================
# Wavelet Methods
# =============================================================================

@register_decomposition(
    name='wavelet_dwt',
    category='wavelet',
    default_params={'wavelet': 'db4', 'level': 3},
    param_ranges={
        'wavelet': ['haar', 'db2', 'db4', 'db8', 'sym4', 'coif2'],
        'level': [1, 2, 3, 4, 5]
    },
    preserves='multi-scale structure, low-frequency approximation',
    destroys='high-frequency detail (depends on level)'
)
def decompose_wavelet_dwt(
    dem: np.ndarray,
    wavelet: str = 'db4',
    level: int = 3
) -> tuple:
    """
    Discrete Wavelet Transform decomposition.
    
    Separates into approximation (trend) and detail (residual) coefficients.
    """
    # Handle NaN values
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    # Perform 2D DWT
    coeffs = pywt.wavedec2(dem_filled, wavelet, level=level)
    
    # Trend = approximation coefficients only (zero out details)
    trend_coeffs = [coeffs[0]] + [
        tuple(np.zeros_like(d) for d in detail) 
        for detail in coeffs[1:]
    ]
    
    # Residual = detail coefficients only (zero out approximation)
    residual_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
    
    # Reconstruct
    trend = pywt.waverec2(trend_coeffs, wavelet)
    residual = pywt.waverec2(residual_coeffs, wavelet)
    
    # Trim to original size (wavelet may pad)
    trend = trend[:dem.shape[0], :dem.shape[1]]
    residual = residual[:dem.shape[0], :dem.shape[1]]
    
    return trend, residual


# =============================================================================
# Morphological Methods
# =============================================================================

@register_decomposition(
    name='morphological',
    category='morphological',
    default_params={'operation': 'opening', 'size': 10},
    param_ranges={
        'operation': ['opening', 'closing', 'average'],
        'size': [5, 10, 20, 50, 100]
    },
    preserves='features matching structuring element shape',
    destroys='features smaller than element'
)
def decompose_morphological(
    dem: np.ndarray,
    operation: str = 'opening',
    size: int = 10
) -> tuple:
    """
    Morphological filtering for shape-based decomposition.
    
    Opening removes bright features smaller than the element.
    Closing removes dark features smaller than the element.
    """
    from scipy.ndimage import grey_opening, grey_closing
    
    # Handle NaN values
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    # Create structuring element (disk shape)
    selem = disk(size)
    
    if operation == 'opening':
        trend = grey_opening(dem_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(dem_filled, footprint=selem)
    else:  # 'average' - opening-closing average
        opened = grey_opening(dem_filled, footprint=selem)
        closed = grey_closing(dem_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = dem_filled - trend
    
    return trend, residual


@register_decomposition(
    name='tophat',
    category='morphological',
    default_params={'size': 20, 'mode': 'white'},
    param_ranges={
        'size': [10, 20, 50, 100],
        'mode': ['white', 'black']
    },
    preserves='small bright (white) or dark (black) features',
    destroys='large-scale variation, background'
)
def decompose_tophat(
    dem: np.ndarray,
    size: int = 20,
    mode: str = 'white'
) -> tuple:
    """
    Top-hat transform for small feature extraction.
    
    White top-hat: extracts bright features smaller than element (mounds)
    Black top-hat: extracts dark features smaller than element (pits)
    """
    from scipy.ndimage import white_tophat, black_tophat
    
    # Handle NaN values
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    # Create structuring element
    selem = disk(size)
    
    if mode == 'white':
        residual = white_tophat(dem_filled, footprint=selem)
    else:
        residual = black_tophat(dem_filled, footprint=selem)
    
    trend = dem_filled - residual
    
    return trend, residual


# =============================================================================
# Polynomial/Surface Fitting
# =============================================================================

@register_decomposition(
    name='polynomial',
    category='trend_removal',
    default_params={'degree': 2},
    param_ranges={'degree': [1, 2, 3]},
    preserves='local deviations from regional trend',
    destroys='large-scale topographic slope'
)
def decompose_polynomial(dem: np.ndarray, degree: int = 2) -> tuple:
    """
    Polynomial surface fitting for trend removal.
    
    Fits a polynomial surface and subtracts it to reveal local features.
    """
    # Handle NaN values
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    rows, cols = dem_filled.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Flatten for fitting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = dem_filled.flatten()
    
    # Build design matrix for 2D polynomial
    if degree == 1:
        A = np.column_stack([
            np.ones_like(X_flat),
            X_flat,
            Y_flat
        ])
    elif degree == 2:
        A = np.column_stack([
            np.ones_like(X_flat),
            X_flat, Y_flat,
            X_flat**2, X_flat*Y_flat, Y_flat**2
        ])
    else:  # degree 3
        A = np.column_stack([
            np.ones_like(X_flat),
            X_flat, Y_flat,
            X_flat**2, X_flat*Y_flat, Y_flat**2,
            X_flat**3, X_flat**2*Y_flat, X_flat*Y_flat**2, Y_flat**3
        ])
    
    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)
    
    # Compute trend surface
    trend_flat = A @ coeffs
    trend = trend_flat.reshape(dem_filled.shape)
    residual = dem_filled - trend
    
    return trend, residual

