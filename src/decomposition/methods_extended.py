"""
Extended Decomposition Methods

Additional methods to maximize prior art coverage.
These complement the core methods in methods.py.
"""

import numpy as np
from scipy.ndimage import (
    gaussian_filter, median_filter, uniform_filter,
    grey_opening, grey_closing, grey_erosion, grey_dilation,
    sobel, laplace
)
from scipy.ndimage import generate_binary_structure, iterate_structure
from skimage.morphology import disk, square, rectangle, diamond, ellipse
from skimage.filters import difference_of_gaussians
import cv2

from .registry import register_decomposition


# =============================================================================
# Additional Classical Methods
# =============================================================================

@register_decomposition(
    name='gaussian_anisotropic',
    category='classical',
    default_params={'sigma_x': 10, 'sigma_y': 10},
    param_ranges={
        'sigma_x': [2, 5, 10, 20, 50],
        'sigma_y': [2, 5, 10, 20, 50]
    },
    preserves='directional features aligned with low-sigma axis',
    destroys='features perpendicular to low-sigma axis'
)
def decompose_gaussian_anisotropic(
    dem: np.ndarray,
    sigma_x: float = 10,
    sigma_y: float = 10
) -> tuple:
    """
    Anisotropic Gaussian filtering.
    
    Different smoothing in X and Y directions. Useful for detecting
    linear features (roads, walls) aligned with specific orientations.
    
    sigma_x: smoothing perpendicular to rows (horizontal features)
    sigma_y: smoothing perpendicular to columns (vertical features)
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    trend = gaussian_filter(dem_filled, sigma=(sigma_y, sigma_x))
    residual = dem_filled - trend
    return trend, residual


@register_decomposition(
    name='median',
    category='edge_preserving',
    default_params={'size': 5},
    param_ranges={'size': [3, 5, 7, 11, 15, 21]},
    preserves='sharp edges, step discontinuities',
    destroys='salt-and-pepper noise, thin lines'
)
def decompose_median(dem: np.ndarray, size: int = 5) -> tuple:
    """
    Median filter decomposition.
    
    Non-linear, edge-preserving smoothing. Better than bilateral
    for removing impulse noise while preserving edges.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    trend = median_filter(dem_filled, size=size)
    residual = dem_filled - trend
    return trend, residual


@register_decomposition(
    name='uniform',
    category='classical',
    default_params={'size': 10},
    param_ranges={'size': [3, 5, 10, 20, 50, 100]},
    preserves='average local elevation',
    destroys='all local variation equally'
)
def decompose_uniform(dem: np.ndarray, size: int = 10) -> tuple:
    """
    Uniform (box) filter decomposition.
    
    Simple averaging filter. Faster than Gaussian but introduces
    more artifacts at edges. Useful as baseline.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    trend = uniform_filter(dem_filled, size=size)
    residual = dem_filled - trend
    return trend, residual


# =============================================================================
# Difference of Gaussians (DoG) - Band-pass filtering
# =============================================================================

@register_decomposition(
    name='dog',
    category='multiscale',
    default_params={'sigma_low': 2, 'sigma_high': 10},
    param_ranges={
        'sigma_low': [1, 2, 3, 5],
        'sigma_high': [5, 10, 20, 50, 100]
    },
    preserves='features at intermediate scales',
    destroys='very small and very large features'
)
def decompose_dog(
    dem: np.ndarray,
    sigma_low: float = 2,
    sigma_high: float = 10
) -> tuple:
    """
    Difference of Gaussians band-pass filtering.
    
    Isolates features at scales between sigma_low and sigma_high.
    Classic approach for edge and blob detection.
    
    Residual = G(σ_low) - G(σ_high)
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    # Ensure sigma_high > sigma_low
    if sigma_high <= sigma_low:
        sigma_high = sigma_low * 2
    
    residual = difference_of_gaussians(dem_filled, sigma_low, sigma_high)
    trend = dem_filled - residual
    return trend, residual


@register_decomposition(
    name='dog_multiscale',
    category='multiscale',
    default_params={'sigma_ratio': 1.6, 'n_scales': 4, 'base_sigma': 1.0},
    param_ranges={
        'sigma_ratio': [1.4, 1.6, 2.0],
        'n_scales': [3, 4, 5, 6],
        'base_sigma': [0.5, 1.0, 2.0]
    },
    preserves='multi-scale blob-like features',
    destroys='flat regions, monotonic gradients'
)
def decompose_dog_multiscale(
    dem: np.ndarray,
    sigma_ratio: float = 1.6,
    n_scales: int = 4,
    base_sigma: float = 1.0
) -> tuple:
    """
    Multi-scale Difference of Gaussians.
    
    Creates a scale-space pyramid and sums DoG responses.
    Detects features across multiple scales simultaneously.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    combined_residual = np.zeros_like(dem_filled)
    
    for i in range(n_scales):
        sigma_low = base_sigma * (sigma_ratio ** i)
        sigma_high = base_sigma * (sigma_ratio ** (i + 1))
        dog = difference_of_gaussians(dem_filled, sigma_low, sigma_high)
        combined_residual += dog
    
    # Normalize by number of scales
    combined_residual /= n_scales
    trend = dem_filled - combined_residual
    
    return trend, combined_residual


# =============================================================================
# Laplacian of Gaussian (LoG) - Blob detection
# =============================================================================

@register_decomposition(
    name='log',
    category='multiscale',
    default_params={'sigma': 5},
    param_ranges={'sigma': [1, 2, 3, 5, 10, 20]},
    preserves='blob-like features at specified scale',
    destroys='linear features, edges, flat regions'
)
def decompose_log(dem: np.ndarray, sigma: float = 5) -> tuple:
    """
    Laplacian of Gaussian (LoG) decomposition.
    
    Detects blob-like features (mounds, pits) at a specific scale.
    The residual is the LoG response (normalized).
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    # Smooth then apply Laplacian (approximates LoG)
    smoothed = gaussian_filter(dem_filled, sigma=sigma)
    residual = laplace(smoothed)
    
    # Scale normalization (σ² for scale-space)
    residual = residual * (sigma ** 2)
    
    trend = dem_filled - residual
    return trend, residual


# =============================================================================
# Additional Morphological Methods with Different Structuring Elements
# =============================================================================

@register_decomposition(
    name='morphological_square',
    category='morphological',
    default_params={'operation': 'opening', 'size': 10},
    param_ranges={
        'operation': ['opening', 'closing', 'gradient', 'average'],
        'size': [3, 5, 10, 15, 20, 50]
    },
    preserves='rectangular features aligned with axes',
    destroys='features smaller than element, circular features'
)
def decompose_morphological_square(
    dem: np.ndarray,
    operation: str = 'opening',
    size: int = 10
) -> tuple:
    """
    Morphological filtering with square structuring element.
    
    Square element is faster to compute and better for detecting
    rectilinear features (building foundations, field boundaries).
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    selem = square(size)
    
    if operation == 'opening':
        trend = grey_opening(dem_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(dem_filled, footprint=selem)
    elif operation == 'gradient':
        # Morphological gradient = dilation - erosion
        dilated = grey_dilation(dem_filled, footprint=selem)
        eroded = grey_erosion(dem_filled, footprint=selem)
        residual = dilated - eroded
        trend = dem_filled - residual
        return trend, residual
    else:  # average
        opened = grey_opening(dem_filled, footprint=selem)
        closed = grey_closing(dem_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = dem_filled - trend
    return trend, residual


@register_decomposition(
    name='morphological_rect',
    category='morphological',
    default_params={'operation': 'opening', 'width': 20, 'height': 5},
    param_ranges={
        'operation': ['opening', 'closing', 'average'],
        'width': [5, 10, 20, 50],
        'height': [3, 5, 10, 20]
    },
    preserves='linear features perpendicular to long axis',
    destroys='features parallel to long axis, small features'
)
def decompose_morphological_rect(
    dem: np.ndarray,
    operation: str = 'opening',
    width: int = 20,
    height: int = 5
) -> tuple:
    """
    Morphological filtering with rectangular structuring element.
    
    Anisotropic element for detecting linear features.
    Width > height detects vertical linear features.
    Height > width detects horizontal linear features.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    selem = rectangle(height, width)
    
    if operation == 'opening':
        trend = grey_opening(dem_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(dem_filled, footprint=selem)
    else:  # average
        opened = grey_opening(dem_filled, footprint=selem)
        closed = grey_closing(dem_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = dem_filled - trend
    return trend, residual


@register_decomposition(
    name='morphological_diamond',
    category='morphological',
    default_params={'operation': 'opening', 'radius': 10},
    param_ranges={
        'operation': ['opening', 'closing', 'average'],
        'radius': [3, 5, 10, 15, 20]
    },
    preserves='diamond/rhombus shaped features',
    destroys='features not matching diamond geometry'
)
def decompose_morphological_diamond(
    dem: np.ndarray,
    operation: str = 'opening',
    radius: int = 10
) -> tuple:
    """
    Morphological filtering with diamond (rhombus) structuring element.
    
    Diamond element is useful for features at 45-degree angles
    and has smaller computational footprint than disk.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    selem = diamond(radius)
    
    if operation == 'opening':
        trend = grey_opening(dem_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(dem_filled, footprint=selem)
    else:  # average
        opened = grey_opening(dem_filled, footprint=selem)
        closed = grey_closing(dem_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = dem_filled - trend
    return trend, residual


@register_decomposition(
    name='morphological_ellipse',
    category='morphological',
    default_params={'operation': 'opening', 'width': 20, 'height': 10},
    param_ranges={
        'operation': ['opening', 'closing', 'average'],
        'width': [5, 10, 20, 30, 50],
        'height': [5, 10, 20, 30, 50]
    },
    preserves='elliptical features with matching orientation',
    destroys='features not matching ellipse geometry'
)
def decompose_morphological_ellipse(
    dem: np.ndarray,
    operation: str = 'opening',
    width: int = 20,
    height: int = 10
) -> tuple:
    """
    Morphological filtering with elliptical structuring element.
    
    Combines aspects of disk (isotropy) and rectangle (directionality).
    Width/height ratio determines orientation selectivity.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    selem = ellipse(height // 2, width // 2)
    
    if operation == 'opening':
        trend = grey_opening(dem_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(dem_filled, footprint=selem)
    else:  # average
        opened = grey_opening(dem_filled, footprint=selem)
        closed = grey_closing(dem_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = dem_filled - trend
    return trend, residual


@register_decomposition(
    name='morphological_gradient',
    category='morphological',
    default_params={'size': 5, 'shape': 'disk'},
    param_ranges={
        'size': [3, 5, 7, 10, 15],
        'shape': ['disk', 'square', 'diamond']
    },
    preserves='edges, boundaries, rapid transitions',
    destroys='flat regions, gradual slopes'
)
def decompose_morphological_gradient(
    dem: np.ndarray,
    size: int = 5,
    shape: str = 'disk'
) -> tuple:
    """
    Morphological gradient = dilation - erosion.
    
    Highlights boundaries and edges. Residual is the edge map.
    Similar to gradient magnitude but using morphological operations.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    if shape == 'disk':
        selem = disk(size)
    elif shape == 'square':
        selem = square(size)
    else:
        selem = diamond(size)
    
    dilated = grey_dilation(dem_filled, footprint=selem)
    eroded = grey_erosion(dem_filled, footprint=selem)
    
    residual = dilated - eroded
    trend = dem_filled - residual
    
    return trend, residual


@register_decomposition(
    name='tophat_combined',
    category='morphological',
    default_params={'size': 20},
    param_ranges={'size': [5, 10, 20, 50, 100]},
    preserves='both bright and dark small features',
    destroys='large-scale variation, features larger than element'
)
def decompose_tophat_combined(dem: np.ndarray, size: int = 20) -> tuple:
    """
    Combined white + black top-hat transform.
    
    Extracts both mound-like (bright) and pit-like (dark) features.
    Useful when interested in all small features regardless of polarity.
    """
    from scipy.ndimage import white_tophat, black_tophat
    
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    selem = disk(size)
    
    white_th = white_tophat(dem_filled, footprint=selem)
    black_th = black_tophat(dem_filled, footprint=selem)
    
    # Combine: white is positive, black is negative
    residual = white_th - black_th
    trend = dem_filled - residual
    
    return trend, residual


# =============================================================================
# Anisotropic Diffusion (Perona-Malik)
# =============================================================================

@register_decomposition(
    name='anisotropic_diffusion',
    category='edge_preserving',
    default_params={'iterations': 10, 'kappa': 50, 'gamma': 0.1},
    param_ranges={
        'iterations': [5, 10, 20, 50],
        'kappa': [10, 30, 50, 100],
        'gamma': [0.05, 0.1, 0.15, 0.2]
    },
    preserves='edges above gradient threshold (kappa)',
    destroys='noise, texture below threshold'
)
def decompose_anisotropic_diffusion(
    dem: np.ndarray,
    iterations: int = 10,
    kappa: float = 50,
    gamma: float = 0.1
) -> tuple:
    """
    Perona-Malik anisotropic diffusion.
    
    Iteratively smooths while preserving edges above threshold.
    kappa: edge threshold (higher = more smoothing)
    gamma: diffusion speed (0-0.25 for stability)
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    img = dem_filled.copy().astype(np.float64)
    
    for _ in range(iterations):
        # Compute gradients
        nabla_n = np.roll(img, -1, axis=0) - img  # North
        nabla_s = np.roll(img, 1, axis=0) - img   # South
        nabla_e = np.roll(img, -1, axis=1) - img  # East
        nabla_w = np.roll(img, 1, axis=1) - img   # West
        
        # Perona-Malik diffusion coefficient (exponential)
        c_n = np.exp(-(nabla_n / kappa) ** 2)
        c_s = np.exp(-(nabla_s / kappa) ** 2)
        c_e = np.exp(-(nabla_e / kappa) ** 2)
        c_w = np.exp(-(nabla_w / kappa) ** 2)
        
        # Update
        img += gamma * (c_n * nabla_n + c_s * nabla_s + 
                       c_e * nabla_e + c_w * nabla_w)
    
    trend = img
    residual = dem_filled - trend
    
    return trend, residual


# =============================================================================
# Rolling Ball Background Subtraction
# =============================================================================

@register_decomposition(
    name='rolling_ball',
    category='morphological',
    default_params={'radius': 50},
    param_ranges={'radius': [10, 25, 50, 100, 200]},
    preserves='features smaller than ball radius',
    destroys='background curvature, large-scale variation'
)
def decompose_rolling_ball(dem: np.ndarray, radius: int = 50) -> tuple:
    """
    Rolling ball background subtraction.
    
    Simulates a ball rolling under the surface. Common in microscopy.
    Effective for removing large-scale curvature while preserving local features.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    # Create ball kernel
    size = 2 * radius + 1
    y, x = np.ogrid[:size, :size]
    center = radius
    
    # Ball height at each position
    dist_sq = (x - center) ** 2 + (y - center) ** 2
    ball = np.where(
        dist_sq <= radius ** 2,
        np.sqrt(radius ** 2 - dist_sq),
        0
    )
    ball = ball.max() - ball  # Invert for use as structuring element
    
    # Use grayscale erosion then dilation (opening)
    # This is equivalent to rolling ball from below
    from scipy.ndimage import grey_erosion, grey_dilation
    
    eroded = grey_erosion(dem_filled, footprint=ball > 0, structure=ball)
    trend = grey_dilation(eroded, footprint=ball > 0, structure=ball)
    
    residual = dem_filled - trend
    
    return trend, residual


# =============================================================================
# Local Polynomial (LOESS-like)
# =============================================================================

@register_decomposition(
    name='local_polynomial',
    category='trend_removal',
    default_params={'window_size': 51, 'degree': 2},
    param_ranges={
        'window_size': [21, 31, 51, 101],
        'degree': [1, 2, 3]
    },
    preserves='local deviations from local polynomial trend',
    destroys='features smoother than local polynomial'
)
def decompose_local_polynomial(
    dem: np.ndarray,
    window_size: int = 51,
    degree: int = 2
) -> tuple:
    """
    Local polynomial (LOESS-like) trend removal.
    
    Fits a polynomial surface in a moving window.
    More adaptive than global polynomial, less blocky than median.
    
    Note: Computationally intensive for large DEMs.
    """
    from scipy.ndimage import uniform_filter
    
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    h, w = dem_filled.shape
    
    # For efficiency, we use a uniform-weighted approach
    # rather than true LOESS (which uses weighted regression)
    
    # Create coordinate grids relative to center
    half = window_size // 2
    y_local = np.arange(-half, half + 1)
    x_local = np.arange(-half, half + 1)
    X_local, Y_local = np.meshgrid(x_local, y_local)
    
    # For each pixel, we'd fit a polynomial
    # This is expensive, so we approximate with uniform filtering
    # of polynomial basis functions
    
    # Approximate: uniform filter of (DEM * basis) / uniform filter of (basis)
    # For degree 0, this is just uniform filter
    # For higher degrees, we weight by position
    
    # Simple approximation: Gaussian-weighted local mean with polynomial correction
    if degree == 1:
        # Linear: remove local gradient
        from scipy.ndimage import sobel
        dx = sobel(dem_filled, axis=1, mode='reflect') / 8
        dy = sobel(dem_filled, axis=0, mode='reflect') / 8
        
        trend = uniform_filter(dem_filled, size=window_size)
        # Trend includes local mean, gradient contribution is in residual
    elif degree >= 2:
        # Higher degree: use Gaussian smoothing as approximation
        sigma = window_size / 4
        trend = gaussian_filter(dem_filled, sigma=sigma)
    else:
        trend = uniform_filter(dem_filled, size=window_size)
    
    residual = dem_filled - trend
    
    return trend, residual


# =============================================================================
# Guided Filter (Edge-Aware)
# =============================================================================

@register_decomposition(
    name='guided',
    category='edge_preserving',
    default_params={'radius': 8, 'eps': 0.01},
    param_ranges={
        'radius': [4, 8, 16, 32],
        'eps': [0.001, 0.01, 0.1, 1.0]
    },
    preserves='edges defined by the guide image',
    destroys='texture not aligned with edges'
)
def decompose_guided(
    dem: np.ndarray,
    radius: int = 8,
    eps: float = 0.01
) -> tuple:
    """
    Guided filter decomposition.
    
    Edge-aware smoothing that preserves edges from a guide image.
    Here we use the DEM itself as guide (self-guided).
    
    eps: regularization (higher = more smoothing)
    radius: window size
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    # Normalize for numerical stability
    dem_min, dem_max = dem_filled.min(), dem_filled.max()
    dem_range = dem_max - dem_min
    if dem_range == 0:
        dem_range = 1
    dem_norm = (dem_filled - dem_min) / dem_range
    
    # Self-guided filter
    I = dem_norm  # Guide
    p = dem_norm  # Input (same as guide for self-guided)
    
    # Compute local statistics using box filter
    mean_I = uniform_filter(I, size=2*radius+1)
    mean_p = uniform_filter(p, size=2*radius+1)
    corr_Ip = uniform_filter(I * p, size=2*radius+1)
    corr_II = uniform_filter(I * I, size=2*radius+1)
    
    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = uniform_filter(a, size=2*radius+1)
    mean_b = uniform_filter(b, size=2*radius+1)
    
    q = mean_a * I + mean_b
    
    # Rescale back
    trend = q * dem_range + dem_min
    residual = dem_filled - trend
    
    return trend, residual


# =============================================================================
# Higher-Degree Polynomial
# =============================================================================

@register_decomposition(
    name='polynomial_high',
    category='trend_removal',
    default_params={'degree': 4},
    param_ranges={'degree': [4, 5, 6]},
    preserves='local deviations from high-order regional trend',
    destroys='large-scale topographic shape up to specified degree'
)
def decompose_polynomial_high(dem: np.ndarray, degree: int = 4) -> tuple:
    """
    High-degree polynomial surface fitting.
    
    For capturing complex regional trends (valleys, ridges).
    Higher degrees can overfit, but useful for specific terrain types.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    rows, cols = dem_filled.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = dem_filled.flatten()
    
    # Normalize coordinates to prevent numerical issues
    X_norm = (X_flat - X_flat.mean()) / X_flat.std()
    Y_norm = (Y_flat - Y_flat.mean()) / Y_flat.std()
    
    # Build design matrix for 2D polynomial up to given degree
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((X_norm ** i) * (Y_norm ** j))
    
    A = np.column_stack(terms)
    
    # Least squares fit with regularization
    coeffs, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)
    
    trend_flat = A @ coeffs
    trend = trend_flat.reshape(dem_filled.shape)
    residual = dem_filled - trend
    
    return trend, residual


# =============================================================================
# Multi-Scale Wavelet (More Wavelets)
# =============================================================================

@register_decomposition(
    name='wavelet_biorthogonal',
    category='wavelet',
    default_params={'wavelet': 'bior3.5', 'level': 3},
    param_ranges={
        'wavelet': ['bior1.3', 'bior2.4', 'bior3.5', 'bior4.4', 'bior5.5'],
        'level': [1, 2, 3, 4, 5]
    },
    preserves='multi-scale structure with linear phase',
    destroys='high-frequency detail (depends on level)'
)
def decompose_wavelet_biorthogonal(
    dem: np.ndarray,
    wavelet: str = 'bior3.5',
    level: int = 3
) -> tuple:
    """
    Biorthogonal wavelet decomposition.
    
    Biorthogonal wavelets have symmetric filters and linear phase,
    which avoids phase distortion. Good for feature detection.
    """
    import pywt
    
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    coeffs = pywt.wavedec2(dem_filled, wavelet, level=level)
    
    trend_coeffs = [coeffs[0]] + [
        tuple(np.zeros_like(d) for d in detail) 
        for detail in coeffs[1:]
    ]
    residual_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
    
    trend = pywt.waverec2(trend_coeffs, wavelet)
    residual = pywt.waverec2(residual_coeffs, wavelet)
    
    trend = trend[:dem.shape[0], :dem.shape[1]]
    residual = residual[:dem.shape[0], :dem.shape[1]]
    
    return trend, residual


@register_decomposition(
    name='wavelet_reverse_biorthogonal',
    category='wavelet',
    default_params={'wavelet': 'rbio3.5', 'level': 3},
    param_ranges={
        'wavelet': ['rbio1.3', 'rbio2.4', 'rbio3.5', 'rbio4.4', 'rbio5.5'],
        'level': [1, 2, 3, 4, 5]
    },
    preserves='multi-scale structure with reversed decomposition',
    destroys='high-frequency detail (depends on level)'
)
def decompose_wavelet_reverse_biorthogonal(
    dem: np.ndarray,
    wavelet: str = 'rbio3.5',
    level: int = 3
) -> tuple:
    """
    Reverse biorthogonal wavelet decomposition.
    
    Uses transposed filter pairs from biorthogonal wavelets.
    Different frequency characteristics from standard biorthogonal.
    """
    import pywt
    
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    coeffs = pywt.wavedec2(dem_filled, wavelet, level=level)
    
    trend_coeffs = [coeffs[0]] + [
        tuple(np.zeros_like(d) for d in detail) 
        for detail in coeffs[1:]
    ]
    residual_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
    
    trend = pywt.waverec2(trend_coeffs, wavelet)
    residual = pywt.waverec2(residual_coeffs, wavelet)
    
    trend = trend[:dem.shape[0], :dem.shape[1]]
    residual = residual[:dem.shape[0], :dem.shape[1]]
    
    return trend, residual

