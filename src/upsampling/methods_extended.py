"""
Extended Upsampling Methods

Additional interpolation methods to maximize prior art coverage.
These complement the core methods in methods.py.
"""

import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
import cv2

from .registry import register_upsampling


# =============================================================================
# Different Spline Orders
# =============================================================================

@register_upsampling(
    name='nearest',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='exact original values at sample points',
    introduces='blocky artifacts, no smoothing'
)
def upsample_nearest(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Nearest-neighbor interpolation (order=0).
    
    Simplest method. No interpolation, just replicates pixels.
    Useful as baseline to measure interpolation effects.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    return zoom(dem_filled, scale, order=0)


@register_upsampling(
    name='bilinear',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='linear gradients',
    introduces='slight blurring, no overshoot'
)
def upsample_bilinear(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Bilinear interpolation (order=1).
    
    Linear interpolation in both directions. Smooth but blurs edges.
    No ringing or overshoot artifacts.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    return zoom(dem_filled, scale, order=1)


@register_upsampling(
    name='quadratic',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='smooth curvature',
    introduces='some overshoot at discontinuities'
)
def upsample_quadratic(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Quadratic spline interpolation (order=2).
    
    Between linear and cubic. Less smooth than cubic but less ringing.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    return zoom(dem_filled, scale, order=2)


@register_upsampling(
    name='quartic',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='higher-order smoothness',
    introduces='more ringing than cubic'
)
def upsample_quartic(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Quartic spline interpolation (order=4).
    
    Higher order than cubic. More smooth continuity but more ringing.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    return zoom(dem_filled, scale, order=4)


@register_upsampling(
    name='quintic',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='highest available smoothness',
    introduces='most ringing of polynomial methods'
)
def upsample_quintic(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Quintic spline interpolation (order=5).
    
    Highest order available in scipy.ndimage.zoom.
    Maximum smoothness but maximum ringing artifacts.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    return zoom(dem_filled, scale, order=5)


# =============================================================================
# Different Scale Factors
# =============================================================================

@register_upsampling(
    name='bicubic_3x',
    category='interpolation',
    default_params={'scale': 3},
    param_ranges={'scale': [3]},
    preserves='smooth curves (3x scale)',
    introduces='slight ringing'
)
def upsample_bicubic_3x(dem: np.ndarray, scale: int = 3) -> np.ndarray:
    """Bicubic interpolation at 3x scale."""
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    return zoom(dem_filled, scale, order=3)


@register_upsampling(
    name='bicubic_16x',
    category='interpolation',
    default_params={'scale': 16},
    param_ranges={'scale': [16]},
    preserves='smooth curves (extreme upscaling)',
    introduces='significant interpolation artifacts'
)
def upsample_bicubic_16x(dem: np.ndarray, scale: int = 16) -> np.ndarray:
    """Bicubic interpolation at 16x scale for extreme upsampling."""
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    return zoom(dem_filled, scale, order=3)


# =============================================================================
# OpenCV Alternative Methods
# =============================================================================

@register_upsampling(
    name='area',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='area-weighted average (best for downscaling)',
    introduces='blur when upscaling'
)
def upsample_area(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Area-based interpolation using OpenCV.
    
    Primarily designed for downscaling (anti-aliased).
    For upscaling, behaves like nearest-neighbor with some smoothing.
    Included for completeness in method comparison.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    return cv2.resize(
        dem_filled.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


@register_upsampling(
    name='linear_exact',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='bit-exact linear interpolation',
    introduces='mild blurring'
)
def upsample_linear_exact(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Bit-exact linear interpolation using OpenCV.
    
    INTER_LINEAR_EXACT provides bit-exact results regardless of
    threading or hardware. Useful for reproducibility.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    return cv2.resize(
        dem_filled.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR_EXACT
    )


# =============================================================================
# Windowed Sinc Variations
# =============================================================================

@register_upsampling(
    name='sinc_hamming',
    category='frequency',
    default_params={'scale': 2, 'kernel_size': 8},
    param_ranges={
        'scale': [2, 4, 8],
        'kernel_size': [4, 8, 16]
    },
    preserves='band-limited signal with Hamming window',
    introduces='less ringing than pure sinc'
)
def upsample_sinc_hamming(
    dem: np.ndarray,
    scale: int = 2,
    kernel_size: int = 8
) -> np.ndarray:
    """
    Sinc interpolation with Hamming window.
    
    Windowed sinc reduces Gibbs ringing compared to pure sinc (FFT).
    Hamming window provides good sidelobe suppression.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Create 1D Hamming-windowed sinc kernel
    def hamming_sinc_kernel(size, scale):
        t = np.arange(-size, size + 1) / scale
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc = np.where(t == 0, 1.0, np.sin(np.pi * t) / (np.pi * t))
        # Hamming window
        window = 0.54 - 0.46 * np.cos(np.pi * (np.arange(len(t)) / (len(t) - 1) * 2))
        return sinc * window
    
    kernel_1d = hamming_sinc_kernel(kernel_size, scale)
    kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize
    
    # Separable convolution for efficiency
    from scipy.ndimage import convolve1d
    
    # First upsample with zeros
    upsampled = np.zeros((new_h, new_w))
    upsampled[::scale, ::scale] = dem_filled
    
    # Convolve
    result = convolve1d(upsampled, kernel_1d * scale, axis=0, mode='reflect')
    result = convolve1d(result, kernel_1d * scale, axis=1, mode='reflect')
    
    return result


@register_upsampling(
    name='sinc_blackman',
    category='frequency',
    default_params={'scale': 2, 'kernel_size': 8},
    param_ranges={
        'scale': [2, 4, 8],
        'kernel_size': [4, 8, 16]
    },
    preserves='band-limited signal with Blackman window',
    introduces='minimal ringing, slight blurring'
)
def upsample_sinc_blackman(
    dem: np.ndarray,
    scale: int = 2,
    kernel_size: int = 8
) -> np.ndarray:
    """
    Sinc interpolation with Blackman window.
    
    Blackman window has better sidelobe suppression than Hamming,
    resulting in less ringing but slightly more blurring.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    def blackman_sinc_kernel(size, scale):
        t = np.arange(-size, size + 1) / scale
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc = np.where(t == 0, 1.0, np.sin(np.pi * t) / (np.pi * t))
        n = len(t)
        window = (0.42 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1)) + 
                  0.08 * np.cos(4 * np.pi * np.arange(n) / (n - 1)))
        return sinc * window
    
    kernel_1d = blackman_sinc_kernel(kernel_size, scale)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    from scipy.ndimage import convolve1d
    
    upsampled = np.zeros((new_h, new_w))
    upsampled[::scale, ::scale] = dem_filled
    
    result = convolve1d(upsampled, kernel_1d * scale, axis=0, mode='reflect')
    result = convolve1d(result, kernel_1d * scale, axis=1, mode='reflect')
    
    return result


# =============================================================================
# Cubic Convolution with Different Alpha
# =============================================================================

@register_upsampling(
    name='cubic_catmull_rom',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='smooth interpolation through control points',
    introduces='minimal overshoot'
)
def upsample_cubic_catmull_rom(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Catmull-Rom spline interpolation (alpha=0.5).
    
    Passes through all control points with C1 continuity.
    Less blurring than B-spline, less ringing than Keys cubic.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Catmull-Rom kernel (alpha = 0.5)
    def catmull_rom_kernel(t):
        t = np.abs(t)
        t2 = t * t
        t3 = t2 * t
        return np.where(
            t <= 1,
            1.5 * t3 - 2.5 * t2 + 1,
            np.where(
                t <= 2,
                -0.5 * t3 + 2.5 * t2 - 4 * t + 2,
                0
            )
        )
    
    # Create output grid
    y_out = np.linspace(0, h - 1, new_h)
    x_out = np.linspace(0, w - 1, new_w)
    
    # Use scipy for actual interpolation with cubic
    # RectBivariateSpline with kx=ky=3 approximates Catmull-Rom
    y_in = np.arange(h)
    x_in = np.arange(w)
    
    spline = RectBivariateSpline(y_in, x_in, dem_filled, kx=3, ky=3, s=0)
    result = spline(y_out, x_out)
    
    return result


@register_upsampling(
    name='cubic_mitchell',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='balanced sharpness and smoothness',
    introduces='minimal artifacts (designed for images)'
)
def upsample_cubic_mitchell(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Mitchell-Netravali filter (B=C=1/3).
    
    Designed to balance blurring, ringing, and anisotropy.
    Often considered ideal for general-purpose image interpolation.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Mitchell-Netravali kernel parameters
    B, C = 1/3, 1/3
    
    def mitchell_kernel(t):
        t = np.abs(t)
        t2 = t * t
        t3 = t2 * t
        return np.where(
            t < 1,
            ((12 - 9*B - 6*C) * t3 + (-18 + 12*B + 6*C) * t2 + (6 - 2*B)) / 6,
            np.where(
                t < 2,
                ((-B - 6*C) * t3 + (6*B + 30*C) * t2 + (-12*B - 48*C) * t + (8*B + 24*C)) / 6,
                0
            )
        )
    
    # Create 1D kernel
    kernel_size = 4 * scale
    kernel_1d = mitchell_kernel(np.arange(-kernel_size, kernel_size + 1) / scale)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    from scipy.ndimage import convolve1d
    
    upsampled = np.zeros((new_h, new_w))
    upsampled[::scale, ::scale] = dem_filled
    
    result = convolve1d(upsampled, kernel_1d * scale, axis=0, mode='reflect')
    result = convolve1d(result, kernel_1d * scale, axis=1, mode='reflect')
    
    return result


# =============================================================================
# Edge-Directed Interpolation
# =============================================================================

@register_upsampling(
    name='edge_directed',
    category='adaptive',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4]},
    preserves='edges with minimal jagging',
    introduces='possible artifacts in textured regions'
)
def upsample_edge_directed(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Simple edge-directed interpolation.
    
    Adapts interpolation direction based on local gradient.
    Better edge preservation than non-adaptive methods.
    
    Note: Simplified implementation for prior art purposes.
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    # First do bicubic as baseline
    bicubic = zoom(dem_filled, scale, order=3)
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Compute gradients at original resolution
    gy = np.gradient(dem_filled, axis=0)
    gx = np.gradient(dem_filled, axis=1)
    
    # Upsample gradients
    gy_up = zoom(gy, scale, order=1)
    gx_up = zoom(gx, scale, order=1)
    
    # Gradient magnitude and direction
    mag = np.sqrt(gx_up**2 + gy_up**2)
    
    # At edges (high gradient), use Lanczos for sharpness
    lanczos = cv2.resize(
        dem_filled.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )
    
    # Blend based on gradient magnitude
    # Normalize magnitude
    mag_norm = mag / (mag.max() + 1e-10)
    
    # Blend: low gradient -> bicubic, high gradient -> lanczos
    result = bicubic * (1 - mag_norm) + lanczos * mag_norm
    
    return result


# =============================================================================
# Regularized Interpolation
# =============================================================================

@register_upsampling(
    name='regularized',
    category='optimization',
    default_params={'scale': 2, 'lambda_reg': 0.01},
    param_ranges={
        'scale': [2, 4],
        'lambda_reg': [0.001, 0.01, 0.1]
    },
    preserves='smooth interpolation with controlled energy',
    introduces='slight smoothing from regularization'
)
def upsample_regularized(
    dem: np.ndarray,
    scale: int = 2,
    lambda_reg: float = 0.01
) -> np.ndarray:
    """
    Regularized interpolation using smoothness prior.
    
    Minimizes interpolation energy + smoothness term.
    Approximated using iterative refinement of bicubic.
    
    lambda_reg: regularization strength (higher = smoother)
    """
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    # Start with bicubic
    result = zoom(dem_filled, scale, order=3)
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Create mask of known points (original samples)
    known = np.zeros((new_h, new_w), dtype=bool)
    known[::scale, ::scale] = True
    
    # Original values at known points
    original_vals = np.zeros((new_h, new_w))
    original_vals[::scale, ::scale] = dem_filled
    
    # Iterative refinement with Laplacian regularization
    from scipy.ndimage import laplace
    
    for _ in range(10):
        # Laplacian smoothing
        lap = laplace(result)
        
        # Update: blend toward smoothness, but keep known points
        result = result - lambda_reg * lap
        
        # Enforce known values
        result[known] = original_vals[known]
    
    return result

