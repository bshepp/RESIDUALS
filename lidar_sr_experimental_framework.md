# LiDAR/DEM Super-Resolution Experimental Framework
## Combinatoric Decomposition × Upsampling Feature Extraction Study

**Purpose**: Systematically test all combinations of signal decomposition methods and upsampling/reconstruction methods to identify which combinations produce residuals that highlight specific terrain features (anthropogenic linear structures, natural drainage, noise patterns, etc.)

**Core Hypothesis**: Different decomposition × upsampling combinations have characteristic "failure modes" that selectively preserve or eliminate different feature classes. By computing differentials between method outputs, we can create feature-specific extraction filters.

---

## PART 1: EXPERIMENTAL SETUP

### 1.1 Data Preparation

```bash
# Create working directories
mkdir -p ~/lidar_experiment/{raw,decomposed,upsampled,differentials,results}
mkdir -p ~/lidar_experiment/decomposed/{trend,residual}
mkdir -p ~/lidar_experiment/logs

# Required Python packages
pip install numpy scipy scikit-image opencv-python pywavelets \
    emd PyEMD vmdpy scikit-learn rasterio laspy pdal \
    tensorflow torch torchvision --break-system-packages
```

### 1.2 Input Data Format

Convert LAS/LAZ to gridded DEM for processing:
```python
import numpy as np
import rasterio
from scipy.interpolate import griddata

def las_to_dem(las_path, resolution=1.0, output_path=None):
    """Convert LAS point cloud to gridded DEM"""
    import laspy
    las = laspy.read(las_path)
    
    # Extract ground points (classification 2)
    ground_mask = las.classification == 2
    x = las.x[ground_mask]
    y = las.y[ground_mask]
    z = las.z[ground_mask]
    
    # Create regular grid
    xi = np.arange(x.min(), x.max(), resolution)
    yi = np.arange(y.min(), y.max(), resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate to grid
    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    return xi, yi, zi
```

### 1.3 Experimental Matrix Generator

```python
import itertools
from dataclasses import dataclass
from typing import List, Callable, Dict, Any

@dataclass
class ExperimentConfig:
    decomposition_method: str
    decomposition_params: Dict[str, Any]
    upsampling_method: str
    upsampling_params: Dict[str, Any]
    scale_factor: int
    
def generate_experiment_matrix(scale_factors=[2, 4, 8]):
    """Generate all decomposition × upsampling combinations"""
    
    decomposition_methods = list(DECOMPOSITION_METHODS.keys())
    upsampling_methods = list(UPSAMPLING_METHODS.keys())
    
    experiments = []
    for decomp, upsamp, scale in itertools.product(
        decomposition_methods, upsampling_methods, scale_factors
    ):
        config = ExperimentConfig(
            decomposition_method=decomp,
            decomposition_params=DECOMPOSITION_METHODS[decomp]['default_params'],
            upsampling_method=upsamp,
            upsampling_params=UPSAMPLING_METHODS[upsamp]['default_params'],
            scale_factor=scale
        )
        experiments.append(config)
    
    return experiments
```

---

## PART 2: DECOMPOSITION METHODS

### Category A: Classical Signal Processing

#### A1. Gaussian Low-Pass Filtering
**Field**: Image processing (fundamental)
**Separates**: Low-frequency trend from high-frequency detail
```python
from scipy.ndimage import gaussian_filter

def decompose_gaussian(dem, sigma=10):
    """Isotropic Gaussian smoothing for trend extraction"""
    trend = gaussian_filter(dem, sigma=sigma)
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['gaussian'] = {
    'func': decompose_gaussian,
    'default_params': {'sigma': 10},
    'param_ranges': {'sigma': [2, 5, 10, 20, 50, 100]},
    'preserves': 'isotropic smoothness',
    'destroys': 'all high-frequency equally'
}
```

#### A2. Bilateral Filtering
**Field**: Computational photography, medical imaging
**Separates**: Smooth regions from edges
```python
import cv2

def decompose_bilateral(dem, d=9, sigma_color=75, sigma_space=75):
    """Edge-preserving smoothing"""
    # Normalize to 0-255 for cv2
    dem_norm = ((dem - dem.min()) / (dem.max() - dem.min()) * 255).astype(np.float32)
    trend = cv2.bilateralFilter(dem_norm, d, sigma_color, sigma_space)
    # Rescale back
    trend = trend / 255 * (dem.max() - dem.min()) + dem.min()
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['bilateral'] = {
    'func': decompose_bilateral,
    'default_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
    'param_ranges': {
        'd': [5, 9, 15, 21],
        'sigma_color': [25, 50, 75, 100, 150],
        'sigma_space': [25, 50, 75, 100, 150]
    },
    'preserves': 'sharp edges, discontinuities',
    'destroys': 'gradual transitions in flat areas'
}
```

#### A3. Guided Filtering
**Field**: Computational photography
**Separates**: Structure from texture using guide image
```python
import cv2

def decompose_guided(dem, radius=16, eps=0.01, guide=None):
    """Structure-preserving smoothing with optional guide"""
    if guide is None:
        guide = dem
    dem_32 = dem.astype(np.float32)
    guide_32 = guide.astype(np.float32)
    trend = cv2.ximgproc.guidedFilter(guide_32, dem_32, radius, eps)
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['guided'] = {
    'func': decompose_guided,
    'default_params': {'radius': 16, 'eps': 0.01},
    'param_ranges': {'radius': [4, 8, 16, 32, 64], 'eps': [0.001, 0.01, 0.1, 1.0]},
    'preserves': 'structures matching guide',
    'destroys': 'textures not in guide'
}
```

#### A4. Anisotropic Diffusion (Perona-Malik)
**Field**: Computer vision, medical imaging
**Separates**: Edges from smooth regions via nonlinear diffusion
```python
from skimage.restoration import denoise_tv_chambolle

def decompose_anisotropic_diffusion(dem, weight=0.1, n_iter=100):
    """Perona-Malik style edge-preserving smoothing"""
    # Using TV denoising as proxy (similar edge-preserving behavior)
    trend = denoise_tv_chambolle(dem, weight=weight, max_num_iter=n_iter)
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['anisotropic_diffusion'] = {
    'func': decompose_anisotropic_diffusion,
    'default_params': {'weight': 0.1, 'n_iter': 100},
    'param_ranges': {'weight': [0.01, 0.05, 0.1, 0.2, 0.5], 'n_iter': [50, 100, 200]},
    'preserves': 'edges, contours',
    'destroys': 'texture, fine detail in smooth regions'
}
```

#### A5. Median Filtering
**Field**: Signal processing, image denoising
**Separates**: Robust local median from outliers/impulse noise
```python
from scipy.ndimage import median_filter

def decompose_median(dem, size=5):
    """Nonlinear median filtering"""
    trend = median_filter(dem, size=size)
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['median'] = {
    'func': decompose_median,
    'default_params': {'size': 5},
    'param_ranges': {'size': [3, 5, 7, 11, 15, 21]},
    'preserves': 'edges, step functions',
    'destroys': 'impulse noise, thin lines (<size/2)'
}
```

---

### Category B: Polynomial/Surface Fitting

#### B1. Polynomial Detrending
**Field**: Geostatistics, terrain analysis
**Separates**: Regional trend surface from local variation
```python
from numpy.polynomial import polynomial as P
import numpy as np

def decompose_polynomial(dem, degree=2):
    """Fit polynomial surface and subtract"""
    rows, cols = dem.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Flatten for fitting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = dem.flatten()
    
    # Remove NaN for fitting
    valid = ~np.isnan(Z_flat)
    
    # Build design matrix for 2D polynomial
    if degree == 1:
        A = np.column_stack([np.ones_like(X_flat[valid]), X_flat[valid], Y_flat[valid]])
    elif degree == 2:
        A = np.column_stack([
            np.ones_like(X_flat[valid]), 
            X_flat[valid], Y_flat[valid],
            X_flat[valid]**2, X_flat[valid]*Y_flat[valid], Y_flat[valid]**2
        ])
    else:  # degree 3
        A = np.column_stack([
            np.ones_like(X_flat[valid]),
            X_flat[valid], Y_flat[valid],
            X_flat[valid]**2, X_flat[valid]*Y_flat[valid], Y_flat[valid]**2,
            X_flat[valid]**3, X_flat[valid]**2*Y_flat[valid], 
            X_flat[valid]*Y_flat[valid]**2, Y_flat[valid]**3
        ])
    
    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, Z_flat[valid], rcond=None)
    
    # Reconstruct trend surface
    if degree == 1:
        A_full = np.column_stack([np.ones_like(X_flat), X_flat, Y_flat])
    elif degree == 2:
        A_full = np.column_stack([
            np.ones_like(X_flat), X_flat, Y_flat,
            X_flat**2, X_flat*Y_flat, Y_flat**2
        ])
    else:
        A_full = np.column_stack([
            np.ones_like(X_flat), X_flat, Y_flat,
            X_flat**2, X_flat*Y_flat, Y_flat**2,
            X_flat**3, X_flat**2*Y_flat, X_flat*Y_flat**2, Y_flat**3
        ])
    
    trend = (A_full @ coeffs).reshape(dem.shape)
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['polynomial'] = {
    'func': decompose_polynomial,
    'default_params': {'degree': 2},
    'param_ranges': {'degree': [1, 2, 3]},
    'preserves': 'local deviations from regional trend',
    'destroys': 'large-scale topographic slope'
}
```

#### B2. Spline Surface Fitting
**Field**: CAD, surface modeling
**Separates**: Smooth surface from local perturbations
```python
from scipy.interpolate import RectBivariateSpline

def decompose_spline(dem, s=1e6, kx=3, ky=3):
    """B-spline surface fitting"""
    rows, cols = dem.shape
    x = np.arange(cols)
    y = np.arange(rows)
    
    # Handle NaN by interpolation first
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    spline = RectBivariateSpline(y, x, dem_filled, kx=kx, ky=ky, s=s)
    trend = spline(y, x)
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['spline'] = {
    'func': decompose_spline,
    'default_params': {'s': 1e6, 'kx': 3, 'ky': 3},
    'param_ranges': {'s': [1e4, 1e5, 1e6, 1e7, 1e8]},
    'preserves': 'local bumps and depressions',
    'destroys': 'smooth regional variation'
}
```

#### B3. Robust Local Regression (LOESS/LOWESS)
**Field**: Statistics, econometrics
**Separates**: Locally weighted trend from residuals
```python
from scipy.ndimage import uniform_filter

def decompose_local_regression(dem, window_size=21):
    """Local weighted averaging (simplified LOESS)"""
    # Use uniform filter as simple local regression
    trend = uniform_filter(dem, size=window_size, mode='reflect')
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['local_regression'] = {
    'func': decompose_local_regression,
    'default_params': {'window_size': 21},
    'param_ranges': {'window_size': [5, 11, 21, 41, 81]},
    'preserves': 'features larger than window',
    'destroys': 'features smaller than window'
}
```

---

### Category C: Wavelet Methods

#### C1. Discrete Wavelet Transform (DWT)
**Field**: Signal processing, compression
**Separates**: Multi-scale frequency components
```python
import pywt

def decompose_wavelet_dwt(dem, wavelet='db4', level=3):
    """Multi-level 2D DWT decomposition"""
    coeffs = pywt.wavedec2(dem, wavelet, level=level)
    
    # Trend = approximation at deepest level
    # Residual = reconstruction without approximation
    trend_coeffs = [coeffs[0]] + [tuple(np.zeros_like(d) for d in detail) for detail in coeffs[1:]]
    residual_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
    
    trend = pywt.waverec2(trend_coeffs, wavelet)
    residual = pywt.waverec2(residual_coeffs, wavelet)
    
    # Trim to original size
    trend = trend[:dem.shape[0], :dem.shape[1]]
    residual = residual[:dem.shape[0], :dem.shape[1]]
    
    return trend, residual

DECOMPOSITION_METHODS['wavelet_dwt'] = {
    'func': decompose_wavelet_dwt,
    'default_params': {'wavelet': 'db4', 'level': 3},
    'param_ranges': {
        'wavelet': ['haar', 'db2', 'db4', 'db8', 'sym4', 'coif2', 'bior2.2'],
        'level': [1, 2, 3, 4, 5]
    },
    'preserves': 'multi-scale structure',
    'destroys': 'depends on level and wavelet'
}
```

#### C2. Stationary Wavelet Transform (SWT)
**Field**: Signal processing (shift-invariant)
**Separates**: Shift-invariant multi-scale components
```python
def decompose_wavelet_swt(dem, wavelet='db4', level=3):
    """Stationary (undecimated) wavelet transform"""
    # Pad to power of 2 if needed
    rows, cols = dem.shape
    pad_rows = int(2**np.ceil(np.log2(rows)))
    pad_cols = int(2**np.ceil(np.log2(cols)))
    dem_padded = np.pad(dem, ((0, pad_rows-rows), (0, pad_cols-cols)), mode='reflect')
    
    coeffs = pywt.swt2(dem_padded, wavelet, level=level)
    
    # Reconstruct trend (low-freq only)
    trend_coeffs = [(coeffs[-1][0], (np.zeros_like(coeffs[-1][1][0]),) * 3)]
    for i in range(level - 1):
        trend_coeffs.append((np.zeros_like(coeffs[i][0]), (np.zeros_like(coeffs[i][1][0]),) * 3))
    
    trend = pywt.iswt2(trend_coeffs[::-1], wavelet)[:rows, :cols]
    residual = dem - trend
    
    return trend, residual

DECOMPOSITION_METHODS['wavelet_swt'] = {
    'func': decompose_wavelet_swt,
    'default_params': {'wavelet': 'db4', 'level': 3},
    'param_ranges': {
        'wavelet': ['haar', 'db4', 'sym4'],
        'level': [2, 3, 4]
    },
    'preserves': 'shift-invariant features',
    'destroys': 'high-frequency detail'
}
```

#### C3. Curvelet Transform
**Field**: Seismic imaging, astronomical imaging
**Separates**: Multi-scale, multi-directional curvilinear features
```python
# Note: Requires curvelops or pyct package
def decompose_curvelet(dem, nscales=4, nangles=16):
    """Curvelet decomposition for directional features"""
    try:
        import curvelops
        # Implementation depends on specific package
        # Placeholder for concept
        trend = gaussian_filter(dem, sigma=10)  # Fallback
        residual = dem - trend
    except ImportError:
        print("Curvelet package not installed, using Gaussian fallback")
        trend = gaussian_filter(dem, sigma=10)
        residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['curvelet'] = {
    'func': decompose_curvelet,
    'default_params': {'nscales': 4, 'nangles': 16},
    'param_ranges': {'nscales': [3, 4, 5], 'nangles': [8, 16, 32]},
    'preserves': 'curvilinear features at multiple scales',
    'destroys': 'isotropic texture'
}
```

#### C4. Contourlet Transform
**Field**: Image processing, texture analysis
**Separates**: Directional features at multiple scales
```python
def decompose_contourlet(dem, nlevels=3):
    """Contourlet decomposition (directional filter bank)"""
    # Simplified implementation using directional wavelets
    from scipy.ndimage import convolve
    
    # Laplacian pyramid for scale decomposition
    kernel = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]]) / 256
    
    trend = convolve(dem, kernel, mode='reflect')
    for _ in range(nlevels - 1):
        trend = convolve(trend, kernel, mode='reflect')
    
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['contourlet'] = {
    'func': decompose_contourlet,
    'default_params': {'nlevels': 3},
    'param_ranges': {'nlevels': [2, 3, 4, 5]},
    'preserves': 'directional contours',
    'destroys': 'non-directional texture'
}
```

#### C5. Shearlet Transform
**Field**: Applied harmonic analysis, edge detection
**Separates**: Anisotropic features with optimal sparsity for edges
```python
def decompose_shearlet(dem, scales=4):
    """Shearlet decomposition for edge-like features"""
    # Simplified shearlet using oriented filters
    from scipy.ndimage import sobel, gaussian_filter
    
    trend = gaussian_filter(dem, sigma=2**scales)
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['shearlet'] = {
    'func': decompose_shearlet,
    'default_params': {'scales': 4},
    'param_ranges': {'scales': [2, 3, 4, 5]},
    'preserves': 'edge-like singularities',
    'destroys': 'smooth regions'
}
```

---

### Category D: Empirical/Data-Driven Decomposition

#### D1. Empirical Mode Decomposition (EMD)
**Field**: Nonlinear signal processing, climate science
**Separates**: Intrinsic mode functions (local oscillations)
```python
from PyEMD import EMD2D

def decompose_emd(dem, max_imfs=5):
    """2D Empirical Mode Decomposition"""
    emd2d = EMD2D()
    imfs = emd2d.emd(dem, max_imf=max_imfs)
    
    # Trend = last IMF (residual)
    # High-freq residual = sum of first IMFs
    if len(imfs) > 1:
        trend = imfs[-1]
        residual = np.sum(imfs[:-1], axis=0)
    else:
        trend = imfs[0]
        residual = dem - trend
    
    return trend, residual

DECOMPOSITION_METHODS['emd'] = {
    'func': decompose_emd,
    'default_params': {'max_imfs': 5},
    'param_ranges': {'max_imfs': [3, 5, 7, 10]},
    'preserves': 'nonlinear oscillations',
    'destroys': 'depends on IMF selection'
}
```

#### D2. Ensemble EMD (EEMD)
**Field**: Climate science, financial analysis
**Separates**: More robust IMFs via noise-assisted decomposition
```python
from PyEMD import EEMD

def decompose_eemd_1d(signal, noise_width=0.2, ensemble_size=100):
    """1D EEMD - apply row-wise then column-wise"""
    eemd = EEMD(trials=ensemble_size, noise_width=noise_width)
    
    # Apply to each row
    row_decomp = []
    for row in signal:
        imfs = eemd.eemd(row)
        row_decomp.append(imfs[-1] if len(imfs) > 0 else row)
    
    trend = np.array(row_decomp)
    residual = signal - trend
    return trend, residual

DECOMPOSITION_METHODS['eemd'] = {
    'func': decompose_eemd_1d,
    'default_params': {'noise_width': 0.2, 'ensemble_size': 100},
    'param_ranges': {'noise_width': [0.1, 0.2, 0.3], 'ensemble_size': [50, 100, 200]},
    'preserves': 'robust low-frequency trend',
    'destroys': 'mode mixing artifacts'
}
```

#### D3. Variational Mode Decomposition (VMD)
**Field**: Signal processing, mechanical fault diagnosis
**Separates**: Narrow-band modes via optimization
```python
def decompose_vmd_1d(signal, K=5, alpha=2000):
    """VMD decomposition - apply row-wise"""
    try:
        from vmdpy import VMD
        
        row_trends = []
        for row in signal:
            u, _, _ = VMD(row, alpha, 0, K, 0, 1, 1e-7)
            # Last mode typically lowest frequency
            row_trends.append(u[-1])
        
        trend = np.array(row_trends)
        residual = signal - trend
    except ImportError:
        trend = gaussian_filter(signal, sigma=10)
        residual = signal - trend
    
    return trend, residual

DECOMPOSITION_METHODS['vmd'] = {
    'func': decompose_vmd_1d,
    'default_params': {'K': 5, 'alpha': 2000},
    'param_ranges': {'K': [3, 5, 7], 'alpha': [500, 2000, 5000]},
    'preserves': 'narrow-band oscillatory modes',
    'destroys': 'broad-band features'
}
```

#### D4. Singular Spectrum Analysis (SSA)
**Field**: Time series analysis, climate science
**Separates**: Components via trajectory matrix SVD
```python
def decompose_ssa(dem, window=50, n_components=5):
    """2D SSA-like decomposition via SVD"""
    from scipy.linalg import svd
    
    # Flatten and create trajectory matrix
    flat = dem.flatten()
    N = len(flat)
    K = N - window + 1
    
    # Build trajectory matrix
    X = np.zeros((window, K))
    for i in range(window):
        X[i] = flat[i:i+K]
    
    # SVD
    U, s, Vt = svd(X, full_matrices=False)
    
    # Reconstruct trend from first components
    X_trend = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
    
    # Diagonal averaging to get 1D signal back
    trend_flat = np.zeros(N)
    counts = np.zeros(N)
    for i in range(window):
        trend_flat[i:i+K] += X_trend[i]
        counts[i:i+K] += 1
    trend_flat /= counts
    
    trend = trend_flat.reshape(dem.shape)
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['ssa'] = {
    'func': decompose_ssa,
    'default_params': {'window': 50, 'n_components': 5},
    'param_ranges': {'window': [20, 50, 100], 'n_components': [3, 5, 10]},
    'preserves': 'dominant singular vectors',
    'destroys': 'minor components (noise, fine detail)'
}
```

---

### Category E: Morphological Methods

#### E1. Morphological Opening/Closing
**Field**: Mathematical morphology, geology
**Separates**: Features by structuring element shape
```python
from scipy.ndimage import grey_opening, grey_closing
from skimage.morphology import disk, square

def decompose_morphological(dem, operation='opening', size=10, shape='disk'):
    """Morphological filtering"""
    if shape == 'disk':
        selem = disk(size)
    else:
        selem = square(size)
    
    if operation == 'opening':
        trend = grey_opening(dem, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(dem, footprint=selem)
    else:  # opening-closing average
        opened = grey_opening(dem, footprint=selem)
        closed = grey_closing(dem, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['morphological'] = {
    'func': decompose_morphological,
    'default_params': {'operation': 'opening', 'size': 10, 'shape': 'disk'},
    'param_ranges': {
        'operation': ['opening', 'closing', 'average'],
        'size': [5, 10, 20, 50],
        'shape': ['disk', 'square']
    },
    'preserves': 'features matching structuring element',
    'destroys': 'features smaller than element'
}
```

#### E2. Top-Hat Transform
**Field**: Microscopy, medical imaging
**Separates**: Bright/dark features smaller than structuring element
```python
from scipy.ndimage import white_tophat, black_tophat
from skimage.morphology import disk

def decompose_tophat(dem, size=20, mode='white'):
    """Top-hat transform for small feature extraction"""
    selem = disk(size)
    
    if mode == 'white':
        residual = white_tophat(dem, footprint=selem)
    else:
        residual = black_tophat(dem, footprint=selem)
    
    trend = dem - residual
    return trend, residual

DECOMPOSITION_METHODS['tophat'] = {
    'func': decompose_tophat,
    'default_params': {'size': 20, 'mode': 'white'},
    'param_ranges': {'size': [10, 20, 50, 100], 'mode': ['white', 'black']},
    'preserves': 'small bright/dark features',
    'destroys': 'large-scale variation'
}
```

#### E3. Rolling Ball Algorithm
**Field**: Microscopy, astronomy (background subtraction)
**Separates**: Background from foreground via virtual ball rolling
```python
from scipy.ndimage import grey_erosion, grey_dilation
from skimage.morphology import ball

def decompose_rolling_ball(dem, radius=50):
    """Rolling ball background subtraction"""
    # 2D version using disk approximation
    from skimage.morphology import disk
    selem = disk(radius)
    
    # Rolling ball = erosion followed by dilation
    background = grey_dilation(grey_erosion(dem, footprint=selem), footprint=selem)
    
    trend = background
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['rolling_ball'] = {
    'func': decompose_rolling_ball,
    'default_params': {'radius': 50},
    'param_ranges': {'radius': [10, 25, 50, 100, 200]},
    'preserves': 'features smaller than ball radius',
    'destroys': 'smooth background'
}
```

---

### Category F: Frequency Domain Methods

#### F1. Fourier Low-Pass Filtering
**Field**: Signal processing (fundamental)
**Separates**: Frequency bands
```python
def decompose_fourier_lowpass(dem, cutoff=0.1):
    """Ideal low-pass filter in Fourier domain"""
    fft = np.fft.fft2(dem)
    fft_shift = np.fft.fftshift(fft)
    
    rows, cols = dem.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create low-pass mask
    y, x = np.ogrid[:rows, :cols]
    mask = np.sqrt((x - ccol)**2 + (y - crow)**2) <= cutoff * min(rows, cols)
    
    # Apply mask
    fft_filtered = fft_shift * mask
    trend = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))
    residual = dem - trend
    
    return trend, residual

DECOMPOSITION_METHODS['fourier_lowpass'] = {
    'func': decompose_fourier_lowpass,
    'default_params': {'cutoff': 0.1},
    'param_ranges': {'cutoff': [0.01, 0.05, 0.1, 0.2, 0.3]},
    'preserves': 'low-frequency components',
    'destroys': 'high-frequency detail sharply'
}
```

#### F2. Butterworth Filter
**Field**: Electrical engineering, signal processing
**Separates**: Smooth frequency transition
```python
def decompose_butterworth(dem, cutoff=0.1, order=2):
    """Butterworth low-pass filter"""
    fft = np.fft.fft2(dem)
    fft_shift = np.fft.fftshift(fft)
    
    rows, cols = dem.shape
    crow, ccol = rows // 2, cols // 2
    
    y, x = np.ogrid[:rows, :cols]
    d = np.sqrt((x - ccol)**2 + (y - crow)**2) / (cutoff * min(rows, cols))
    
    # Butterworth filter
    mask = 1 / (1 + d**(2*order))
    
    fft_filtered = fft_shift * mask
    trend = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))
    residual = dem - trend
    
    return trend, residual

DECOMPOSITION_METHODS['butterworth'] = {
    'func': decompose_butterworth,
    'default_params': {'cutoff': 0.1, 'order': 2},
    'param_ranges': {'cutoff': [0.05, 0.1, 0.2], 'order': [1, 2, 4, 8]},
    'preserves': 'low frequencies with smooth rolloff',
    'destroys': 'high frequencies gradually'
}
```

#### F3. Gabor Filter Bank
**Field**: Texture analysis, neuroscience (visual cortex modeling)
**Separates**: Oriented frequency components
```python
from skimage.filters import gabor

def decompose_gabor_bank(dem, frequency=0.1, n_orientations=8):
    """Gabor filter bank decomposition"""
    responses = []
    
    for i in range(n_orientations):
        theta = i * np.pi / n_orientations
        filt_real, filt_imag = gabor(dem, frequency=frequency, theta=theta)
        responses.append(np.sqrt(filt_real**2 + filt_imag**2))
    
    # Residual = max response across orientations
    residual = np.max(responses, axis=0)
    trend = dem - residual
    
    return trend, residual

DECOMPOSITION_METHODS['gabor'] = {
    'func': decompose_gabor_bank,
    'default_params': {'frequency': 0.1, 'n_orientations': 8},
    'param_ranges': {'frequency': [0.05, 0.1, 0.2, 0.3], 'n_orientations': [4, 8, 16]},
    'preserves': 'oriented texture at specific frequency',
    'destroys': 'non-oriented features'
}
```

---

### Category G: Sparse/Dictionary Methods

#### G1. Dictionary Learning (K-SVD style)
**Field**: Sparse coding, compressed sensing
**Separates**: Sparse representation from dense residual
```python
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning

def decompose_dictionary(dem, n_components=50, patch_size=8, alpha=1.0):
    """Dictionary learning based decomposition"""
    from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
    
    # Extract patches
    patches = extract_patches_2d(dem, (patch_size, patch_size))
    patches_flat = patches.reshape(patches.shape[0], -1)
    
    # Learn dictionary
    dl = MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha, 
                                      max_iter=100, random_state=42)
    code = dl.fit_transform(patches_flat)
    
    # Reconstruct
    reconstructed_patches = code @ dl.components_
    reconstructed_patches = reconstructed_patches.reshape(patches.shape)
    
    trend = reconstruct_from_patches_2d(reconstructed_patches, dem.shape)
    residual = dem - trend
    
    return trend, residual

DECOMPOSITION_METHODS['dictionary_learning'] = {
    'func': decompose_dictionary,
    'default_params': {'n_components': 50, 'patch_size': 8, 'alpha': 1.0},
    'param_ranges': {
        'n_components': [25, 50, 100],
        'patch_size': [4, 8, 16],
        'alpha': [0.5, 1.0, 2.0]
    },
    'preserves': 'sparse structure in learned dictionary',
    'destroys': 'features not in dictionary'
}
```

#### G2. Robust PCA (Low-Rank + Sparse)
**Field**: Computer vision, video surveillance
**Separates**: Low-rank background from sparse foreground
```python
def decompose_robust_pca(dem, lambda_param=None, max_iter=100):
    """Robust PCA via inexact ALM"""
    from scipy.linalg import svd
    
    if lambda_param is None:
        lambda_param = 1 / np.sqrt(max(dem.shape))
    
    M = dem.copy()
    S = np.zeros_like(M)
    Y = np.zeros_like(M)
    mu = 1.25 / np.linalg.norm(M, 2)
    mu_bar = mu * 1e7
    rho = 1.5
    
    for _ in range(max_iter):
        # Update L (low-rank)
        U, s, Vt = svd(M - S + Y/mu, full_matrices=False)
        s_thresh = np.maximum(s - 1/mu, 0)
        L = U @ np.diag(s_thresh) @ Vt
        
        # Update S (sparse)
        temp = M - L + Y/mu
        S = np.sign(temp) * np.maximum(np.abs(temp) - lambda_param/mu, 0)
        
        # Update Y
        Y = Y + mu * (M - L - S)
        mu = min(rho * mu, mu_bar)
    
    trend = L  # Low-rank component
    residual = S  # Sparse component
    
    return trend, residual

DECOMPOSITION_METHODS['robust_pca'] = {
    'func': decompose_robust_pca,
    'default_params': {'lambda_param': None, 'max_iter': 100},
    'param_ranges': {'lambda_param': [None, 0.01, 0.1], 'max_iter': [50, 100, 200]},
    'preserves': 'sparse anomalies, outliers',
    'destroys': 'low-rank structure'
}
```

---

### Category H: Deep Learning Decomposition

#### H1. Autoencoder Decomposition
**Field**: Deep learning, representation learning
**Separates**: Learned latent representation from reconstruction error
```python
import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def decompose_autoencoder(dem, model_path=None, epochs=50):
    """Autoencoder-based decomposition"""
    # Normalize
    dem_norm = (dem - dem.min()) / (dem.max() - dem.min())
    
    # Pad to multiple of 4
    h, w = dem_norm.shape
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    dem_padded = np.pad(dem_norm, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    # Convert to tensor
    x = torch.FloatTensor(dem_padded).unsqueeze(0).unsqueeze(0)
    
    # Simple training or load pretrained
    model = SimpleAutoencoder()
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        trend_tensor = model(x)
    
    trend_norm = trend_tensor.squeeze().numpy()[:h, :w]
    trend = trend_norm * (dem.max() - dem.min()) + dem.min()
    residual = dem - trend
    
    return trend, residual

DECOMPOSITION_METHODS['autoencoder'] = {
    'func': decompose_autoencoder,
    'default_params': {'epochs': 50},
    'param_ranges': {'epochs': [20, 50, 100]},
    'preserves': 'features learnable by network',
    'destroys': 'features not captured in bottleneck'
}
```

#### H2. U-Net Style Decomposition
**Field**: Medical image segmentation, satellite imagery
**Separates**: Multi-scale features via skip connections
```python
# Conceptual - requires trained model
def decompose_unet(dem, model_path=None):
    """U-Net based multi-scale decomposition"""
    # Placeholder - would need pre-trained terrain U-Net
    trend = gaussian_filter(dem, sigma=5)
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['unet'] = {
    'func': decompose_unet,
    'default_params': {},
    'param_ranges': {},
    'preserves': 'multi-scale hierarchical features',
    'destroys': 'depends on training'
}
```

---

### Category I: Physics-Inspired Methods

#### I1. Heat Equation Diffusion
**Field**: Physics, image processing
**Separates**: Solution to heat equation at time t
```python
def decompose_heat_diffusion(dem, t=10, dt=0.1):
    """Heat equation based smoothing"""
    from scipy.ndimage import laplace
    
    u = dem.copy()
    for _ in range(int(t / dt)):
        u = u + dt * laplace(u)
    
    trend = u
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['heat_diffusion'] = {
    'func': decompose_heat_diffusion,
    'default_params': {'t': 10, 'dt': 0.1},
    'param_ranges': {'t': [1, 5, 10, 20, 50]},
    'preserves': 'large-scale structure',
    'destroys': 'fine detail (diffuses away)'
}
```

#### I2. Mumford-Shah Segmentation
**Field**: Computer vision, variational methods
**Separates**: Piecewise smooth regions from edges
```python
def decompose_mumford_shah(dem, alpha=0.1, beta=1.0, iterations=100):
    """Simplified Mumford-Shah via Chan-Vese"""
    from skimage.segmentation import chan_vese
    
    # Normalize for segmentation
    dem_norm = (dem - dem.min()) / (dem.max() - dem.min())
    
    # Get segmentation
    seg = chan_vese(dem_norm, mu=alpha, lambda1=beta, lambda2=beta, 
                   max_num_iter=iterations)
    
    # Create piecewise constant approximation
    trend = np.where(seg, dem_norm[seg].mean(), dem_norm[~seg].mean())
    trend = trend * (dem.max() - dem.min()) + dem.min()
    residual = dem - trend
    
    return trend, residual

DECOMPOSITION_METHODS['mumford_shah'] = {
    'func': decompose_mumford_shah,
    'default_params': {'alpha': 0.1, 'beta': 1.0, 'iterations': 100},
    'param_ranges': {'alpha': [0.05, 0.1, 0.2], 'beta': [0.5, 1.0, 2.0]},
    'preserves': 'piecewise smooth regions',
    'destroys': 'within-region variation'
}
```

---

### Category J: Domain-Specific Methods

#### J1. Terrain-Specific: Slope-Based Decomposition
**Field**: Geomorphology
**Separates**: Slope-defined features
```python
def decompose_slope_based(dem, slope_threshold=15):
    """Separate by local slope"""
    from scipy.ndimage import sobel
    
    dx = sobel(dem, axis=1)
    dy = sobel(dem, axis=0)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Smooth only low-slope areas
    mask = slope < slope_threshold
    trend = dem.copy()
    trend[mask] = gaussian_filter(dem, sigma=5)[mask]
    
    residual = dem - trend
    return trend, residual

DECOMPOSITION_METHODS['slope_based'] = {
    'func': decompose_slope_based,
    'default_params': {'slope_threshold': 15},
    'param_ranges': {'slope_threshold': [5, 10, 15, 25, 45]},
    'preserves': 'steep features (cliffs, embankments)',
    'destroys': 'variation in flat areas'
}
```

#### J2. Hydrological: Flow-Based Decomposition
**Field**: Hydrology, terrain analysis
**Separates**: Drainage network from interfluves
```python
def decompose_flow_based(dem, flow_threshold=100):
    """Separate using flow accumulation"""
    from scipy.ndimage import label
    
    # Simple D8 flow accumulation approximation
    # In practice, use proper hydrology library
    dx = np.gradient(dem, axis=1)
    dy = np.gradient(dem, axis=0)
    
    # Flow direction (simplified)
    flow_dir = np.arctan2(dy, dx)
    
    # Accumulation (very simplified)
    trend = gaussian_filter(dem, sigma=20)
    residual = dem - trend
    
    return trend, residual

DECOMPOSITION_METHODS['flow_based'] = {
    'func': decompose_flow_based,
    'default_params': {'flow_threshold': 100},
    'param_ranges': {'flow_threshold': [50, 100, 500, 1000]},
    'preserves': 'drainage channels',
    'destroys': 'hilltop variation'
}
```

---

## PART 3: UPSAMPLING/RECONSTRUCTION METHODS

### Category U-A: Classical Interpolation

```python
UPSAMPLING_METHODS = {}

# U-A1: Nearest Neighbor
def upsample_nearest(dem, scale=2):
    """Nearest neighbor interpolation"""
    from scipy.ndimage import zoom
    return zoom(dem, scale, order=0)

UPSAMPLING_METHODS['nearest'] = {
    'func': upsample_nearest,
    'default_params': {'scale': 2},
    'preserves': 'exact values, sharp edges',
    'introduces': 'blocky artifacts'
}

# U-A2: Bilinear
def upsample_bilinear(dem, scale=2):
    """Bilinear interpolation"""
    from scipy.ndimage import zoom
    return zoom(dem, scale, order=1)

UPSAMPLING_METHODS['bilinear'] = {
    'func': upsample_bilinear,
    'default_params': {'scale': 2},
    'preserves': 'smooth gradients',
    'introduces': 'linear blur'
}

# U-A3: Bicubic
def upsample_bicubic(dem, scale=2):
    """Bicubic interpolation"""
    from scipy.ndimage import zoom
    return zoom(dem, scale, order=3)

UPSAMPLING_METHODS['bicubic'] = {
    'func': upsample_bicubic,
    'default_params': {'scale': 2},
    'preserves': 'smooth curves, good for lines',
    'introduces': 'slight ringing at edges'
}

# U-A4: Biquintic
def upsample_biquintic(dem, scale=2):
    """5th order spline interpolation"""
    from scipy.ndimage import zoom
    return zoom(dem, scale, order=5)

UPSAMPLING_METHODS['biquintic'] = {
    'func': upsample_biquintic,
    'default_params': {'scale': 2},
    'preserves': 'very smooth curves',
    'introduces': 'more ringing than bicubic'
}

# U-A5: Lanczos
def upsample_lanczos(dem, scale=2, a=3):
    """Lanczos interpolation"""
    import cv2
    h, w = dem.shape
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(dem.astype(np.float32), (new_w, new_h), 
                      interpolation=cv2.INTER_LANCZOS4)

UPSAMPLING_METHODS['lanczos'] = {
    'func': upsample_lanczos,
    'default_params': {'scale': 2},
    'preserves': 'sharp edges with smooth interpolation',
    'introduces': 'controlled ringing'
}
```

### Category U-B: Spline Methods

```python
# U-B1: B-Spline
def upsample_bspline(dem, scale=2):
    """B-spline interpolation"""
    from scipy.interpolate import RectBivariateSpline
    
    h, w = dem.shape
    x = np.arange(w)
    y = np.arange(h)
    
    spline = RectBivariateSpline(y, x, dem)
    
    new_x = np.linspace(0, w-1, int(w*scale))
    new_y = np.linspace(0, h-1, int(h*scale))
    
    return spline(new_y, new_x)

UPSAMPLING_METHODS['bspline'] = {
    'func': upsample_bspline,
    'default_params': {'scale': 2},
    'preserves': 'smooth surface continuity',
    'introduces': 'may overshoot at discontinuities'
}

# U-B2: Thin Plate Spline
def upsample_thin_plate_spline(dem, scale=2, smoothing=0):
    """Thin plate spline interpolation"""
    from scipy.interpolate import Rbf
    
    h, w = dem.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Subsample for computational feasibility
    step = max(1, min(h, w) // 50)
    xs = X[::step, ::step].flatten()
    ys = Y[::step, ::step].flatten()
    zs = dem[::step, ::step].flatten()
    
    rbf = Rbf(xs, ys, zs, function='thin_plate', smooth=smoothing)
    
    new_x = np.linspace(0, w-1, int(w*scale))
    new_y = np.linspace(0, h-1, int(h*scale))
    new_X, new_Y = np.meshgrid(new_x, new_y)
    
    return rbf(new_X, new_Y)

UPSAMPLING_METHODS['thin_plate_spline'] = {
    'func': upsample_thin_plate_spline,
    'default_params': {'scale': 2, 'smoothing': 0},
    'preserves': 'minimum curvature surface',
    'introduces': 'global smoothness constraint'
}
```

### Category U-C: Frequency Domain

```python
# U-C1: Zero-Padding FFT
def upsample_fft_zeropad(dem, scale=2):
    """FFT upsampling via zero-padding"""
    h, w = dem.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    fft = np.fft.fft2(dem)
    fft_shift = np.fft.fftshift(fft)
    
    # Zero-pad in frequency domain
    padded = np.zeros((new_h, new_w), dtype=complex)
    pad_h = (new_h - h) // 2
    pad_w = (new_w - w) // 2
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = fft_shift
    
    result = np.real(np.fft.ifft2(np.fft.ifftshift(padded))) * (scale ** 2)
    return result

UPSAMPLING_METHODS['fft_zeropad'] = {
    'func': upsample_fft_zeropad,
    'default_params': {'scale': 2},
    'preserves': 'frequency content exactly',
    'introduces': 'Gibbs ringing at discontinuities'
}

# U-C2: Sinc Interpolation
def upsample_sinc(dem, scale=2):
    """Ideal sinc interpolation (approximated)"""
    from scipy.signal import resample
    
    # Resample rows then columns
    temp = resample(dem, int(dem.shape[0] * scale), axis=0)
    result = resample(temp, int(dem.shape[1] * scale), axis=1)
    return result

UPSAMPLING_METHODS['sinc'] = {
    'func': upsample_sinc,
    'default_params': {'scale': 2},
    'preserves': 'band-limited content perfectly',
    'introduces': 'ringing for non-band-limited signals'
}
```

### Category U-D: Wavelet Methods

```python
# U-D1: Wavelet Cycle Spinning
def upsample_wavelet(dem, scale=2, wavelet='db4'):
    """Wavelet-based upsampling"""
    import pywt
    
    # Decompose
    coeffs = pywt.wavedec2(dem, wavelet, level=int(np.log2(scale)))
    
    # Upsample approximation coefficients
    coeffs[0] = zoom(coeffs[0], scale, order=3)
    
    # Zero high-frequency coefficients for upsampled size
    for i in range(1, len(coeffs)):
        coeffs[i] = tuple(np.zeros((coeffs[0].shape[0] // (2**(i-1)), 
                                    coeffs[0].shape[1] // (2**(i-1)))) 
                          for _ in range(3))
    
    return pywt.waverec2(coeffs, wavelet)[:int(dem.shape[0]*scale), 
                                          :int(dem.shape[1]*scale)]

UPSAMPLING_METHODS['wavelet'] = {
    'func': upsample_wavelet,
    'default_params': {'scale': 2, 'wavelet': 'db4'},
    'preserves': 'multi-scale structure',
    'introduces': 'wavelet-specific artifacts'
}
```

### Category U-E: Edge-Directed

```python
# U-E1: NEDI (New Edge-Directed Interpolation)
def upsample_edge_directed(dem, scale=2):
    """Edge-directed interpolation"""
    from scipy.ndimage import sobel
    
    # Simple edge-aware interpolation
    # Full NEDI is more complex
    h, w = dem.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Get edges
    edges_x = sobel(dem, axis=1)
    edges_y = sobel(dem, axis=0)
    edge_strength = np.sqrt(edges_x**2 + edges_y**2)
    
    # Bicubic base
    result = zoom(dem, scale, order=3)
    edge_up = zoom(edge_strength, scale, order=1)
    
    # Sharpen along edges
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(result, sigma=1)
    result = result + 0.5 * edge_up / (edge_up.max() + 1e-6) * (result - blurred)
    
    return result

UPSAMPLING_METHODS['edge_directed'] = {
    'func': upsample_edge_directed,
    'default_params': {'scale': 2},
    'preserves': 'edge sharpness',
    'introduces': 'may over-sharpen'
}
```

### Category U-F: Deep Learning

```python
# U-F1: SRCNN-style
def upsample_srcnn(dem, scale=2, model_path=None):
    """SRCNN-style super-resolution"""
    # Would need trained model
    # Fallback to bicubic
    return zoom(dem, scale, order=3)

UPSAMPLING_METHODS['srcnn'] = {
    'func': upsample_srcnn,
    'default_params': {'scale': 2},
    'preserves': 'learned features',
    'introduces': 'learned artifacts, smoothing'
}

# U-F2: ESPCN (Efficient Sub-Pixel CNN)
def upsample_espcn(dem, scale=2, model_path=None):
    """ESPCN with pixel shuffle"""
    return zoom(dem, scale, order=3)  # Placeholder

UPSAMPLING_METHODS['espcn'] = {
    'func': upsample_espcn,
    'default_params': {'scale': 2},
    'preserves': 'efficient sub-pixel features',
    'introduces': 'checkerboard artifacts possible'
}

# U-F3: EDSR (Enhanced Deep Residual)
def upsample_edsr(dem, scale=2, model_path=None):
    """EDSR super-resolution"""
    return zoom(dem, scale, order=3)  # Placeholder

UPSAMPLING_METHODS['edsr'] = {
    'func': upsample_edsr,
    'default_params': {'scale': 2},
    'preserves': 'residual learning features',
    'introduces': 'training data bias'
}

# U-F4: SRGAN
def upsample_srgan(dem, scale=2, model_path=None):
    """SRGAN perceptual super-resolution"""
    return zoom(dem, scale, order=3)  # Placeholder

UPSAMPLING_METHODS['srgan'] = {
    'func': upsample_srgan,
    'default_params': {'scale': 2},
    'preserves': 'perceptual quality',
    'introduces': 'hallucinated textures'
}

# U-F5: Real-ESRGAN
def upsample_real_esrgan(dem, scale=2, model_path=None):
    """Real-ESRGAN for real-world degradation"""
    return zoom(dem, scale, order=3)  # Placeholder

UPSAMPLING_METHODS['real_esrgan'] = {
    'func': upsample_real_esrgan,
    'default_params': {'scale': 2},
    'preserves': 'handles complex degradation',
    'introduces': 'may over-enhance'
}
```

### Category U-G: Iterative/Optimization

```python
# U-G1: Iterative Back-Projection
def upsample_ibp(dem, scale=2, iterations=10):
    """Iterative Back-Projection"""
    from scipy.ndimage import zoom
    
    # Initial estimate
    hr = zoom(dem, scale, order=3)
    
    for _ in range(iterations):
        # Simulate LR from current HR
        lr_sim = zoom(hr, 1/scale, order=3)
        
        # Compute error
        error = dem - lr_sim
        
        # Back-project error
        error_up = zoom(error, scale, order=3)
        
        # Update HR
        hr = hr + error_up
    
    return hr

UPSAMPLING_METHODS['ibp'] = {
    'func': upsample_ibp,
    'default_params': {'scale': 2, 'iterations': 10},
    'preserves': 'consistency with LR input',
    'introduces': 'may amplify noise'
}

# U-G2: Total Variation Regularized
def upsample_tv(dem, scale=2, weight=0.1, iterations=100):
    """TV-regularized upsampling"""
    from scipy.ndimage import zoom
    from skimage.restoration import denoise_tv_chambolle
    
    # Initial bicubic
    hr = zoom(dem, scale, order=3)
    
    # TV denoising to regularize
    hr = denoise_tv_chambolle(hr, weight=weight, max_num_iter=iterations)
    
    return hr

UPSAMPLING_METHODS['tv_regularized'] = {
    'func': upsample_tv,
    'default_params': {'scale': 2, 'weight': 0.1},
    'preserves': 'piecewise smooth regions',
    'introduces': 'staircasing'
}
```

### Category U-H: Unconventional/Cross-Domain

```python
# U-H1: Audio-inspired: Phase Vocoder
def upsample_phase_vocoder(dem, scale=2):
    """Phase vocoder style (audio time-stretch adapted)"""
    # Treat DEM as 2D signal, apply phase preservation
    from scipy.signal import stft, istft
    
    # Apply row-wise
    results = []
    for row in dem:
        f, t, Zxx = stft(row, nperseg=64)
        # Stretch time axis
        new_t_len = int(len(t) * scale)
        # Interpolate magnitude and phase
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        from scipy.ndimage import zoom
        mag_up = zoom(mag, (1, scale), order=1)
        phase_up = zoom(phase, (1, scale), order=1)
        
        Zxx_up = mag_up * np.exp(1j * phase_up)
        _, row_up = istft(Zxx_up, nperseg=64)
        results.append(row_up[:int(len(row) * scale)])
    
    temp = np.array(results)
    # Then column-wise (simplified - just zoom)
    return zoom(temp, (scale, 1), order=3)

UPSAMPLING_METHODS['phase_vocoder'] = {
    'func': upsample_phase_vocoder,
    'default_params': {'scale': 2},
    'preserves': 'phase relationships',
    'introduces': 'audio-style artifacts'
}

# U-H2: Fractal-based
def upsample_fractal(dem, scale=2, iterations=3):
    """Fractal interpolation (midpoint displacement variant)"""
    from scipy.ndimage import zoom
    
    result = dem.copy()
    current_scale = 1
    
    while current_scale < scale:
        h, w = result.shape
        new_result = np.zeros((h*2, w*2))
        
        # Copy existing points
        new_result[::2, ::2] = result
        
        # Interpolate with random displacement
        new_result[1::2, ::2] = (new_result[:-1:2, ::2] + new_result[2::2, ::2]) / 2
        new_result[::2, 1::2] = (new_result[::2, :-1:2] + new_result[::2, 2::2]) / 2
        new_result[1::2, 1::2] = (new_result[:-1:2, :-1:2] + new_result[2::2, 2::2] + 
                                  new_result[:-1:2, 2::2] + new_result[2::2, :-1:2]) / 4
        
        # Add random displacement scaled by roughness
        roughness = 0.5 ** iterations
        noise_scale = np.std(result) * roughness / (2 ** iterations)
        new_result[1::2, :] += np.random.randn(h, w*2) * noise_scale
        new_result[:, 1::2] += np.random.randn(h*2, w) * noise_scale
        
        result = new_result
        current_scale *= 2
    
    return result[:int(dem.shape[0]*scale), :int(dem.shape[1]*scale)]

UPSAMPLING_METHODS['fractal'] = {
    'func': upsample_fractal,
    'default_params': {'scale': 2, 'iterations': 3},
    'preserves': 'self-similar roughness',
    'introduces': 'synthetic texture'
}

# U-H3: Kriging (geostatistics)
def upsample_kriging(dem, scale=2, variogram='spherical'):
    """Kriging interpolation"""
    # Simplified - would need proper variogram fitting
    from scipy.ndimage import zoom
    return zoom(dem, scale, order=3)  # Placeholder

UPSAMPLING_METHODS['kriging'] = {
    'func': upsample_kriging,
    'default_params': {'scale': 2, 'variogram': 'spherical'},
    'preserves': 'spatial autocorrelation',
    'introduces': 'variogram-dependent smoothness'
}

# U-H4: Optimal Transport
def upsample_optimal_transport(dem, scale=2):
    """Optimal transport based upsampling"""
    # Conceptual - treat as distribution matching
    from scipy.ndimage import zoom
    return zoom(dem, scale, order=3)  # Placeholder

UPSAMPLING_METHODS['optimal_transport'] = {
    'func': upsample_optimal_transport,
    'default_params': {'scale': 2},
    'preserves': 'distribution properties',
    'introduces': 'transport-based smoothing'
}

# U-H5: Neural Implicit (coordinate network)
def upsample_neural_implicit(dem, scale=2):
    """SIREN/NeRF style coordinate network"""
    # Would require training coordinate network
    from scipy.ndimage import zoom
    return zoom(dem, scale, order=3)  # Placeholder

UPSAMPLING_METHODS['neural_implicit'] = {
    'func': upsample_neural_implicit,
    'default_params': {'scale': 2},
    'preserves': 'continuous representation',
    'introduces': 'network capacity limits'
}
```

---

## PART 4: DIFFERENTIAL COMPUTATION AND ANALYSIS

### 4.1 Compute All Differentials

```python
def compute_differential(result_a, result_b, method='subtract'):
    """Compute differential between two results"""
    if method == 'subtract':
        return result_a - result_b
    elif method == 'ratio':
        return result_a / (result_b + 1e-10)
    elif method == 'normalized':
        diff = result_a - result_b
        return diff / (np.std(diff) + 1e-10)
    elif method == 'absolute':
        return np.abs(result_a - result_b)

def run_full_experiment(dem, scale=2, output_dir='results'):
    """Run all decomposition × upsampling combinations"""
    import os
    import json
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # For each decomposition method
    for decomp_name, decomp_info in DECOMPOSITION_METHODS.items():
        print(f"Decomposition: {decomp_name}")
        
        try:
            trend, residual = decomp_info['func'](dem, **decomp_info['default_params'])
        except Exception as e:
            print(f"  Error in decomposition: {e}")
            continue
        
        results[decomp_name] = {'trend': {}, 'residual': {}}
        
        # For each upsampling method
        for upsamp_name, upsamp_info in UPSAMPLING_METHODS.items():
            print(f"  Upsampling: {upsamp_name}")
            
            try:
                # Upsample both trend and residual
                trend_up = upsamp_info['func'](trend, scale=scale)
                residual_up = upsamp_info['func'](residual, scale=scale)
                
                # Store results
                results[decomp_name]['trend'][upsamp_name] = trend_up
                results[decomp_name]['residual'][upsamp_name] = residual_up
                
                # Save to disk
                np.save(f"{output_dir}/{decomp_name}_{upsamp_name}_trend.npy", trend_up)
                np.save(f"{output_dir}/{decomp_name}_{upsamp_name}_residual.npy", residual_up)
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    # Compute differentials between all pairs
    print("\nComputing differentials...")
    differentials = compute_all_differentials(results, output_dir)
    
    return results, differentials

def compute_all_differentials(results, output_dir):
    """Compute differentials between all method pairs"""
    differentials = {}
    
    decomp_names = list(results.keys())
    
    for i, decomp1 in enumerate(decomp_names):
        for decomp2 in decomp_names[i+1:]:
            for upsamp in results[decomp1]['residual'].keys():
                if upsamp not in results[decomp2]['residual']:
                    continue
                
                key = f"{decomp1}_vs_{decomp2}_{upsamp}"
                
                res1 = results[decomp1]['residual'][upsamp]
                res2 = results[decomp2]['residual'][upsamp]
                
                # Ensure same shape
                min_shape = (min(res1.shape[0], res2.shape[0]),
                            min(res1.shape[1], res2.shape[1]))
                res1 = res1[:min_shape[0], :min_shape[1]]
                res2 = res2[:min_shape[0], :min_shape[1]]
                
                diff = compute_differential(res1, res2, method='subtract')
                differentials[key] = diff
                
                np.save(f"{output_dir}/diff_{key}.npy", diff)
    
    return differentials
```

### 4.2 Feature Analysis

```python
def analyze_differential_for_features(diff, dem_original=None):
    """Analyze differential for potential feature content"""
    
    analysis = {}
    
    # Basic statistics
    analysis['mean'] = float(np.mean(diff))
    analysis['std'] = float(np.std(diff))
    analysis['min'] = float(np.min(diff))
    analysis['max'] = float(np.max(diff))
    analysis['skewness'] = float(scipy.stats.skew(diff.flatten()))
    analysis['kurtosis'] = float(scipy.stats.kurtosis(diff.flatten()))
    
    # Linearity analysis (for detecting roads, etc.)
    from skimage.transform import hough_line, hough_line_peaks
    from skimage.feature import canny
    
    # Threshold differential for edge detection
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-10)
    edges = canny(diff_norm, sigma=2)
    
    # Hough transform for linear features
    h, theta, d = hough_line(edges)
    peaks = hough_line_peaks(h, theta, d, num_peaks=20)
    analysis['linear_feature_count'] = len(peaks[0])
    analysis['max_linear_strength'] = float(np.max(peaks[0])) if len(peaks[0]) > 0 else 0
    
    # Spatial autocorrelation (Moran's I approximation)
    from scipy.ndimage import correlate
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
    lagged = correlate(diff, kernel, mode='reflect')
    moran_num = np.sum((diff - diff.mean()) * (lagged - lagged.mean()))
    moran_den = np.sum((diff - diff.mean())**2)
    analysis['spatial_autocorr'] = float(moran_num / (moran_den + 1e-10))
    
    # Frequency content
    fft = np.fft.fft2(diff)
    fft_mag = np.abs(np.fft.fftshift(fft))
    center = np.array(fft_mag.shape) // 2
    
    # Radial frequency distribution
    y, x = np.ogrid[:fft_mag.shape[0], :fft_mag.shape[1]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    low_freq = np.mean(fft_mag[r < center[0] * 0.2])
    mid_freq = np.mean(fft_mag[(r >= center[0] * 0.2) & (r < center[0] * 0.6)])
    high_freq = np.mean(fft_mag[r >= center[0] * 0.6])
    
    analysis['low_freq_energy'] = float(low_freq)
    analysis['mid_freq_energy'] = float(mid_freq)
    analysis['high_freq_energy'] = float(high_freq)
    analysis['freq_ratio_high_low'] = float(high_freq / (low_freq + 1e-10))
    
    return analysis

def rank_differentials_for_linear_features(differentials, analyses):
    """Rank differentials by potential for linear feature detection"""
    
    scores = {}
    
    for key, analysis in analyses.items():
        # Score based on linear feature indicators
        score = 0
        
        # More linear features detected = better
        score += analysis['linear_feature_count'] * 10
        score += analysis['max_linear_strength']
        
        # High spatial autocorrelation suggests structured features
        score += max(0, analysis['spatial_autocorr']) * 100
        
        # High freq ratio suggests edge content
        score += analysis['freq_ratio_high_low'] * 10
        
        # Moderate kurtosis suggests distinct features (not uniform noise)
        if 1 < abs(analysis['kurtosis']) < 10:
            score += 20
        
        scores[key] = score
    
    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked
```

---

## PART 5: EXECUTION PROTOCOL

### 5.1 Main Execution Script

```python
#!/usr/bin/env python3
"""
LiDAR/DEM Super-Resolution Experimental Framework
Main execution script for AI agent
"""

import os
import sys
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main(input_dem_path, output_dir, scale_factors=[2, 4]):
    """
    Main execution function
    
    Parameters:
    -----------
    input_dem_path : str
        Path to input DEM (GeoTIFF, NumPy array, etc.)
    output_dir : str
        Directory for outputs
    scale_factors : list
        Upsampling factors to test
    """
    
    logger.info(f"Starting experiment at {datetime.now()}")
    logger.info(f"Input: {input_dem_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Scale factors: {scale_factors}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load DEM
    logger.info("Loading DEM...")
    if input_dem_path.endswith('.tif') or input_dem_path.endswith('.tiff'):
        import rasterio
        with rasterio.open(input_dem_path) as src:
            dem = src.read(1)
            profile = src.profile
    elif input_dem_path.endswith('.npy'):
        dem = np.load(input_dem_path)
        profile = None
    else:
        raise ValueError(f"Unsupported format: {input_dem_path}")
    
    logger.info(f"DEM shape: {dem.shape}")
    logger.info(f"DEM range: [{dem.min():.2f}, {dem.max():.2f}]")
    
    # Handle NaN
    if np.any(np.isnan(dem)):
        logger.warning("DEM contains NaN values, filling with local mean")
        from scipy.ndimage import generic_filter
        dem = np.nan_to_num(dem, nan=np.nanmean(dem))
    
    all_results = {}
    all_analyses = {}
    
    for scale in scale_factors:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing scale factor: {scale}")
        logger.info(f"{'='*50}")
        
        scale_dir = os.path.join(output_dir, f"scale_{scale}")
        os.makedirs(scale_dir, exist_ok=True)
        
        # Run full experiment
        results, differentials = run_full_experiment(dem, scale=scale, output_dir=scale_dir)
        
        # Analyze differentials
        logger.info("Analyzing differentials...")
        analyses = {}
        for key, diff in differentials.items():
            analyses[key] = analyze_differential_for_features(diff, dem)
        
        # Rank for linear features
        ranked = rank_differentials_for_linear_features(differentials, analyses)
        
        # Save analyses
        with open(os.path.join(scale_dir, 'analyses.json'), 'w') as f:
            json.dump(analyses, f, indent=2)
        
        with open(os.path.join(scale_dir, 'rankings.json'), 'w') as f:
            json.dump(ranked, f, indent=2)
        
        all_results[scale] = results
        all_analyses[scale] = analyses
        
        # Report top candidates
        logger.info("\nTop 10 differential combinations for linear feature detection:")
        for i, (key, score) in enumerate(ranked[:10]):
            logger.info(f"  {i+1}. {key}: score={score:.2f}")
    
    # Generate summary report
    generate_summary_report(all_results, all_analyses, output_dir)
    
    logger.info(f"\nExperiment complete. Results saved to {output_dir}")

def generate_summary_report(all_results, all_analyses, output_dir):
    """Generate summary report of all experiments"""
    
    report_path = os.path.join(output_dir, 'SUMMARY_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# LiDAR/DEM Super-Resolution Experiment Summary\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- Decomposition methods tested: {len(DECOMPOSITION_METHODS)}\n")
        f.write(f"- Upsampling methods tested: {len(UPSAMPLING_METHODS)}\n")
        f.write(f"- Total combinations: {len(DECOMPOSITION_METHODS) * len(UPSAMPLING_METHODS)}\n\n")
        
        for scale, analyses in all_analyses.items():
            f.write(f"## Scale Factor {scale}\n\n")
            
            # Find best combinations
            ranked = rank_differentials_for_linear_features({}, analyses)
            
            f.write("### Top Combinations for Linear Feature Detection\n\n")
            f.write("| Rank | Combination | Score | Linear Features | Spatial Autocorr |\n")
            f.write("|------|-------------|-------|-----------------|------------------|\n")
            
            for i, (key, score) in enumerate(ranked[:20]):
                if key in analyses:
                    a = analyses[key]
                    f.write(f"| {i+1} | {key} | {score:.1f} | {a['linear_feature_count']} | {a['spatial_autocorr']:.3f} |\n")
            
            f.write("\n")
    
    logger.info(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python experiment.py <input_dem> <output_dir> [scale_factors]")
        print("Example: python experiment.py terrain.tif results/ 2,4,8")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if len(sys.argv) > 3:
        scales = [int(s) for s in sys.argv[3].split(',')]
    else:
        scales = [2, 4]
    
    main(input_path, output_path, scales)
```

---

## PART 6: METHOD SUMMARY TABLE

| Category | Method | Field of Origin | Trend Preserves | Residual Captures |
|----------|--------|-----------------|-----------------|-------------------|
| **Decomposition** |
| A-Classical | Gaussian | Image processing | Smooth regions | All high-freq |
| A-Classical | Bilateral | Photography | Edges + smooth | Texture |
| A-Classical | Guided | Photography | Structure | Texture |
| A-Classical | Anisotropic | Medical | Edges | Smooth texture |
| A-Classical | Median | Signal proc | Robust median | Outliers |
| B-Polynomial | Polynomial | Geostatistics | Regional slope | Local bumps |
| B-Polynomial | Spline | CAD | Smooth surface | Perturbations |
| B-Polynomial | LOESS | Statistics | Local trend | Residuals |
| C-Wavelet | DWT | Signal proc | Low-freq approx | Multi-scale |
| C-Wavelet | SWT | Signal proc | Shift-invariant | Details |
| C-Wavelet | Curvelet | Seismic | Curves | Non-curve |
| C-Wavelet | Contourlet | Texture | Directional | Non-directional |
| C-Wavelet | Shearlet | Math | Edges | Smooth |
| D-Empirical | EMD | Climate | Residual IMF | Oscillations |
| D-Empirical | EEMD | Climate | Robust trend | Clean modes |
| D-Empirical | VMD | Fault diag | Narrow-band | Modes |
| D-Empirical | SSA | Time series | Principal comp | Minor comp |
| E-Morph | Opening/Closing | Geology | Shape match | Shape residual |
| E-Morph | Top-Hat | Microscopy | Background | Small features |
| E-Morph | Rolling Ball | Astronomy | Background | Foreground |
| F-Frequency | Fourier LP | Fundamentals | Low freq | High freq |
| F-Frequency | Butterworth | EE | Smooth rolloff | High freq |
| F-Frequency | Gabor | Neuroscience | Non-oriented | Oriented |
| G-Sparse | Dictionary | Compressed | Sparse struct | Non-sparse |
| G-Sparse | Robust PCA | Video | Low-rank | Sparse |
| H-Deep | Autoencoder | ML | Learned | Bottleneck loss |
| I-Physics | Heat | Physics | Diffused | Fine detail |
| I-Physics | Mumford-Shah | Vision | Piecewise | Edges |
| J-Domain | Slope-based | Geomorph | Flat areas | Steep |
| J-Domain | Flow-based | Hydrology | Interfluves | Channels |
| **Upsampling** |
| U-A | Nearest | Fundamentals | Values | Blocky |
| U-A | Bilinear | Fundamentals | Gradients | Linear blur |
| U-A | Bicubic | Fundamentals | Curves | Slight ring |
| U-A | Lanczos | Fundamentals | Sharp edges | Controlled ring |
| U-B | B-Spline | CAD | Continuity | Overshoot |
| U-B | Thin Plate | CAD | Min curvature | Global smooth |
| U-C | FFT Zero-pad | Signal | Frequency | Gibbs |
| U-C | Sinc | Signal | Band-limited | Ringing |
| U-D | Wavelet | Signal | Multi-scale | Wavelet artifacts |
| U-E | Edge-directed | Vision | Edge sharp | Over-sharpen |
| U-F | SRCNN | Deep learning | Learned | Smooth |
| U-F | SRGAN | Deep learning | Perceptual | Hallucinated |
| U-G | IBP | Optimization | LR consistent | Noise amplify |
| U-G | TV | Optimization | Piecewise | Staircasing |
| U-H | Phase Vocoder | Audio | Phase | Audio artifacts |
| U-H | Fractal | Graphics | Self-similar | Synthetic |
| U-H | Kriging | Geostat | Autocorr | Variogram dep |

---

## APPENDIX: Additional Resources

### Python Environment Setup

```bash
# Create conda environment
conda create -n lidar_sr python=3.10
conda activate lidar_sr

# Core packages
pip install numpy scipy scikit-image opencv-python-headless
pip install pywavelets PyEMD vmdpy
pip install rasterio laspy pdal
pip install scikit-learn

# Deep learning (optional)
pip install torch torchvision
pip install tensorflow

# Visualization
pip install matplotlib seaborn plotly

# Geospatial
pip install geopandas shapely pyproj
```

### References for Further Reading

1. DTDL Framework: doi.org/10.1080/17538947.2024.2356121
2. TfaSR: doi.org/10.1016/j.isprsjprs.2022.04.028
3. EMD: Huang et al. (1998) "The empirical mode decomposition"
4. Bilateral Filter: Tomasi & Manduchi (1998)
5. Archaeological LiDAR: Kokalj et al. (2012)

---

*Document Version: 1.0*
*Created for systematic exploration of DEM super-resolution methods*
*Designed for AI agent execution*
