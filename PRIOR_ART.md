# RESIDUALS: Exhaustive Parameter Exploration - Prior Art Documentation

**Generated**: 2026-01-05  
**Analysis Completed**: 2026-01-09  
**Total Data Size**: 4.28 TB (39,731 output files)

**Purpose**: This document establishes prior art for all tested combinations 
of signal decomposition and upsampling methods applied to Digital Elevation 
Model (DEM) super-resolution for feature detection.

---

## 1. Method Parameter Spaces

### 1.1 Decomposition Methods

#### gaussian
- **Category**: classical
- **Preserves**: smooth regions, large-scale features
- **Destroys**: all high-frequency equally (isotropic)
- **Default Parameters**: `{'sigma': 10}`
- **Parameter Ranges**:
  - `sigma`: [2, 5, 10, 20, 50, 100]
- **Total Combinations**: 6

#### bilateral
- **Category**: edge_preserving
- **Preserves**: sharp edges, discontinuities (roads, walls)
- **Destroys**: gradual transitions in flat areas, texture
- **Default Parameters**: `{'d': 9, 'sigma_color': 75, 'sigma_space': 75}`
- **Parameter Ranges**:
  - `d`: [5, 9, 15, 21]
  - `sigma_color`: [25, 50, 75, 100, 150]
  - `sigma_space`: [25, 50, 75, 100, 150]
- **Total Combinations**: 100

#### wavelet_dwt
- **Category**: wavelet
- **Preserves**: multi-scale structure, low-frequency approximation
- **Destroys**: high-frequency detail (depends on level)
- **Default Parameters**: `{'wavelet': 'db4', 'level': 3}`
- **Parameter Ranges**:
  - `wavelet`: ['haar', 'db2', 'db4', 'db8', 'sym4', 'coif2']
  - `level`: [1, 2, 3, 4, 5]
- **Total Combinations**: 30

#### morphological
- **Category**: morphological
- **Preserves**: features matching structuring element shape
- **Destroys**: features smaller than element
- **Default Parameters**: `{'operation': 'opening', 'size': 10}`
- **Parameter Ranges**:
  - `operation`: ['opening', 'closing', 'average']
  - `size`: [5, 10, 20, 50, 100]
- **Total Combinations**: 15

#### tophat
- **Category**: morphological
- **Preserves**: small bright (white) or dark (black) features
- **Destroys**: large-scale variation, background
- **Default Parameters**: `{'size': 20, 'mode': 'white'}`
- **Parameter Ranges**:
  - `size`: [10, 20, 50, 100]
  - `mode`: ['white', 'black']
- **Total Combinations**: 8

#### polynomial
- **Category**: trend_removal
- **Preserves**: local deviations from regional trend
- **Destroys**: large-scale topographic slope
- **Default Parameters**: `{'degree': 2}`
- **Parameter Ranges**:
  - `degree`: [1, 2, 3]
- **Total Combinations**: 3

#### gaussian_anisotropic
- **Category**: classical
- **Preserves**: directional features aligned with low-sigma axis
- **Destroys**: features perpendicular to low-sigma axis
- **Default Parameters**: `{'sigma_x': 10, 'sigma_y': 10}`
- **Parameter Ranges**:
  - `sigma_x`: [2, 5, 10, 20, 50]
  - `sigma_y`: [2, 5, 10, 20, 50]
- **Total Combinations**: 25

#### median
- **Category**: edge_preserving
- **Preserves**: sharp edges, step discontinuities
- **Destroys**: salt-and-pepper noise, thin lines
- **Default Parameters**: `{'size': 5}`
- **Parameter Ranges**:
  - `size`: [3, 5, 7, 11, 15, 21]
- **Total Combinations**: 6

#### uniform
- **Category**: classical
- **Preserves**: average local elevation
- **Destroys**: all local variation equally
- **Default Parameters**: `{'size': 10}`
- **Parameter Ranges**:
  - `size`: [3, 5, 10, 20, 50, 100]
- **Total Combinations**: 6

#### dog
- **Category**: multiscale
- **Preserves**: features at intermediate scales
- **Destroys**: very small and very large features
- **Default Parameters**: `{'sigma_low': 2, 'sigma_high': 10}`
- **Parameter Ranges**:
  - `sigma_low`: [1, 2, 3, 5]
  - `sigma_high`: [5, 10, 20, 50, 100]
- **Total Combinations**: 20

#### dog_multiscale
- **Category**: multiscale
- **Preserves**: multi-scale blob-like features
- **Destroys**: flat regions, monotonic gradients
- **Default Parameters**: `{'sigma_ratio': 1.6, 'n_scales': 4, 'base_sigma': 1.0}`
- **Parameter Ranges**:
  - `sigma_ratio`: [1.4, 1.6, 2.0]
  - `n_scales`: [3, 4, 5, 6]
  - `base_sigma`: [0.5, 1.0, 2.0]
- **Total Combinations**: 36

#### log
- **Category**: multiscale
- **Preserves**: blob-like features at specified scale
- **Destroys**: linear features, edges, flat regions
- **Default Parameters**: `{'sigma': 5}`
- **Parameter Ranges**:
  - `sigma`: [1, 2, 3, 5, 10, 20]
- **Total Combinations**: 6

#### morphological_square
- **Category**: morphological
- **Preserves**: rectangular features aligned with axes
- **Destroys**: features smaller than element, circular features
- **Default Parameters**: `{'operation': 'opening', 'size': 10}`
- **Parameter Ranges**:
  - `operation`: ['opening', 'closing', 'gradient', 'average']
  - `size`: [3, 5, 10, 15, 20, 50]
- **Total Combinations**: 24

#### morphological_rect
- **Category**: morphological
- **Preserves**: linear features perpendicular to long axis
- **Destroys**: features parallel to long axis, small features
- **Default Parameters**: `{'operation': 'opening', 'width': 20, 'height': 5}`
- **Parameter Ranges**:
  - `operation`: ['opening', 'closing', 'average']
  - `width`: [5, 10, 20, 50]
  - `height`: [3, 5, 10, 20]
- **Total Combinations**: 48

#### morphological_diamond
- **Category**: morphological
- **Preserves**: diamond/rhombus shaped features
- **Destroys**: features not matching diamond geometry
- **Default Parameters**: `{'operation': 'opening', 'radius': 10}`
- **Parameter Ranges**:
  - `operation`: ['opening', 'closing', 'average']
  - `radius`: [3, 5, 10, 15, 20]
- **Total Combinations**: 15

#### morphological_ellipse
- **Category**: morphological
- **Preserves**: elliptical features with matching orientation
- **Destroys**: features not matching ellipse geometry
- **Default Parameters**: `{'operation': 'opening', 'width': 20, 'height': 10}`
- **Parameter Ranges**:
  - `operation`: ['opening', 'closing', 'average']
  - `width`: [5, 10, 20, 30, 50]
  - `height`: [5, 10, 20, 30, 50]
- **Total Combinations**: 75

#### morphological_gradient
- **Category**: morphological
- **Preserves**: edges, boundaries, rapid transitions
- **Destroys**: flat regions, gradual slopes
- **Default Parameters**: `{'size': 5, 'shape': 'disk'}`
- **Parameter Ranges**:
  - `size`: [3, 5, 7, 10, 15]
  - `shape`: ['disk', 'square', 'diamond']
- **Total Combinations**: 15

#### tophat_combined
- **Category**: morphological
- **Preserves**: both bright and dark small features
- **Destroys**: large-scale variation, features larger than element
- **Default Parameters**: `{'size': 20}`
- **Parameter Ranges**:
  - `size`: [5, 10, 20, 50, 100]
- **Total Combinations**: 5

#### anisotropic_diffusion
- **Category**: edge_preserving
- **Preserves**: edges above gradient threshold (kappa)
- **Destroys**: noise, texture below threshold
- **Default Parameters**: `{'iterations': 10, 'kappa': 50, 'gamma': 0.1}`
- **Parameter Ranges**:
  - `iterations`: [5, 10, 20, 50]
  - `kappa`: [10, 30, 50, 100]
  - `gamma`: [0.05, 0.1, 0.15, 0.2]
- **Total Combinations**: 64

#### rolling_ball
- **Category**: morphological
- **Preserves**: features smaller than ball radius
- **Destroys**: background curvature, large-scale variation
- **Default Parameters**: `{'radius': 50}`
- **Parameter Ranges**:
  - `radius`: [10, 25, 50, 100, 200]
- **Total Combinations**: 5

#### local_polynomial
- **Category**: trend_removal
- **Preserves**: local deviations from local polynomial trend
- **Destroys**: features smoother than local polynomial
- **Default Parameters**: `{'window_size': 51, 'degree': 2}`
- **Parameter Ranges**:
  - `window_size`: [21, 31, 51, 101]
  - `degree`: [1, 2, 3]
- **Total Combinations**: 12

#### guided
- **Category**: edge_preserving
- **Preserves**: edges defined by the guide image
- **Destroys**: texture not aligned with edges
- **Default Parameters**: `{'radius': 8, 'eps': 0.01}`
- **Parameter Ranges**:
  - `radius`: [4, 8, 16, 32]
  - `eps`: [0.001, 0.01, 0.1, 1.0]
- **Total Combinations**: 16

#### polynomial_high
- **Category**: trend_removal
- **Preserves**: local deviations from high-order regional trend
- **Destroys**: large-scale topographic shape up to specified degree
- **Default Parameters**: `{'degree': 4}`
- **Parameter Ranges**:
  - `degree`: [4, 5, 6]
- **Total Combinations**: 3

#### wavelet_biorthogonal
- **Category**: wavelet
- **Preserves**: multi-scale structure with linear phase
- **Destroys**: high-frequency detail (depends on level)
- **Default Parameters**: `{'wavelet': 'bior3.5', 'level': 3}`
- **Parameter Ranges**:
  - `wavelet`: ['bior1.3', 'bior2.4', 'bior3.5', 'bior4.4', 'bior5.5']
  - `level`: [1, 2, 3, 4, 5]
- **Total Combinations**: 25

#### wavelet_reverse_biorthogonal
- **Category**: wavelet
- **Preserves**: multi-scale structure with reversed decomposition
- **Destroys**: high-frequency detail (depends on level)
- **Default Parameters**: `{'wavelet': 'rbio3.5', 'level': 3}`
- **Parameter Ranges**:
  - `wavelet`: ['rbio1.3', 'rbio2.4', 'rbio3.5', 'rbio4.4', 'rbio5.5']
  - `level`: [1, 2, 3, 4, 5]
- **Total Combinations**: 25

### 1.2 Upsampling Methods

#### bicubic
- **Category**: interpolation
- **Preserves**: smooth curves, good for continuous surfaces
- **Introduces**: slight ringing at sharp edges
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### lanczos
- **Category**: interpolation
- **Preserves**: sharp edges with smooth interpolation
- **Introduces**: controlled ringing (less than sinc)
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### bspline
- **Category**: spline
- **Preserves**: smooth surface, less ringing than cubic
- **Introduces**: slightly more smoothing than cubic
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### fft_zeropad
- **Category**: frequency
- **Preserves**: frequency content exactly (band-limited)
- **Introduces**: Gibbs ringing at discontinuities
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### nearest
- **Category**: interpolation
- **Preserves**: exact original values at sample points
- **Introduces**: blocky artifacts, no smoothing
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### bilinear
- **Category**: interpolation
- **Preserves**: linear gradients
- **Introduces**: slight blurring, no overshoot
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### quadratic
- **Category**: interpolation
- **Preserves**: smooth curvature
- **Introduces**: some overshoot at discontinuities
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### quartic
- **Category**: interpolation
- **Preserves**: higher-order smoothness
- **Introduces**: more ringing than cubic
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### quintic
- **Category**: interpolation
- **Preserves**: highest available smoothness
- **Introduces**: most ringing of polynomial methods
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### bicubic_3x
- **Category**: interpolation
- **Preserves**: smooth curves (3x scale)
- **Introduces**: slight ringing
- **Default Parameters**: `{'scale': 3}`
- **Parameter Ranges**:
  - `scale`: [3]
- **Total Combinations**: 1

#### bicubic_16x
- **Category**: interpolation
- **Preserves**: smooth curves (extreme upscaling)
- **Introduces**: significant interpolation artifacts
- **Default Parameters**: `{'scale': 16}`
- **Parameter Ranges**:
  - `scale`: [16]
- **Total Combinations**: 1

#### area
- **Category**: interpolation
- **Preserves**: area-weighted average (best for downscaling)
- **Introduces**: blur when upscaling
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### linear_exact
- **Category**: interpolation
- **Preserves**: bit-exact linear interpolation
- **Introduces**: mild blurring
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### sinc_hamming
- **Category**: frequency
- **Preserves**: band-limited signal with Hamming window
- **Introduces**: less ringing than pure sinc
- **Default Parameters**: `{'scale': 2, 'kernel_size': 8}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
  - `kernel_size`: [4, 8, 16]
- **Total Combinations**: 9

#### sinc_blackman
- **Category**: frequency
- **Preserves**: band-limited signal with Blackman window
- **Introduces**: minimal ringing, slight blurring
- **Default Parameters**: `{'scale': 2, 'kernel_size': 8}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
  - `kernel_size`: [4, 8, 16]
- **Total Combinations**: 9

#### cubic_catmull_rom
- **Category**: interpolation
- **Preserves**: smooth interpolation through control points
- **Introduces**: minimal overshoot
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### cubic_mitchell
- **Category**: interpolation
- **Preserves**: balanced sharpness and smoothness
- **Introduces**: minimal artifacts (designed for images)
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4, 8]
- **Total Combinations**: 3

#### edge_directed
- **Category**: adaptive
- **Preserves**: edges with minimal jagging
- **Introduces**: possible artifacts in textured regions
- **Default Parameters**: `{'scale': 2}`
- **Parameter Ranges**:
  - `scale`: [2, 4]
- **Total Combinations**: 2

#### regularized
- **Category**: optimization
- **Preserves**: smooth interpolation with controlled energy
- **Introduces**: slight smoothing from regularization
- **Default Parameters**: `{'scale': 2, 'lambda_reg': 0.01}`
- **Parameter Ranges**:
  - `scale`: [2, 4]
  - `lambda_reg`: [0.001, 0.01, 0.1]
- **Total Combinations**: 6

---

## 2. Tested Combinations Summary

**Total Combinations Tested**: 39,731

**Total Data Generated**: 4.28 TB

### Redundancy Analysis

Analysis of all 39,731 outputs using SHA-256 checksums and statistical fingerprinting:

| Metric | Value |
|--------|-------|
| Exact duplicate groups | 3,345 |
| Near-duplicate pairs (r > 0.99) | 4,754,489 |
| Distinct method clusters | 20 |
| Verification accuracy | 100/100 sampled pairs confirmed |

**Key equivalences discovered:**
- `bspline` ≡ `quadratic` upsampling (identical outputs for all decompositions)
- Anisotropic Diffusion variations cluster together (42% of all methods)
- Morphological methods cluster by structuring element shape

### Top 20 Method Clusters (by output similarity)

| Cluster | Representative Method | Count | % |
|---------|----------------------|-------|---|
| 4 | Anisotropic Diffusion | 16,760 | 42.2% |
| 2 | Gaussian Anisotropic | 7,197 | 18.1% |
| 0 | Morphological (square) | 4,375 | 11.0% |
| 9 | Wavelet DWT | 2,696 | 6.8% |
| 19 | Morphological Ellipse | 2,454 | 6.2% |
| 1 | Wavelet DWT | 1,548 | 3.9% |
| 6 | Morphological Gradient | 1,159 | 2.9% |
| 15 | Morphological Rectangle | 1,039 | 2.6% |
| 3 | Gaussian (large σ) | 736 | 1.9% |
| 7 | DoG Multiscale | 695 | 1.7% |
| 10 | Morphological Rectangle | 312 | 0.8% |
| 18 | DoG Multiscale | 303 | 0.8% |
| 13 | Wavelet Biorthogonal | 181 | 0.5% |
| 5 | Rolling Ball | 128 | 0.3% |
| 14 | Polynomial | 61 | 0.2% |
| 11 | Difference of Gaussians | 59 | 0.1% |
| 16 | Laplacian of Gaussian | 18 | 0.05% |
| 8 | Rolling Ball | 6 | 0.02% |
| 12 | Polynomial | 2 | 0.01% |
| 17 | Rolling Ball | 2 | 0.01% |

### Combinations by Decomposition Method

| Decomposition | Combinations Tested |
|---------------|---------------------|
| gaussian | 402 |
| bilateral | 6,700 |
| wavelet_dwt | 2,010 |
| morphological | 1,005 |
| tophat | 536 |
| polynomial | 201 |
| gaussian_anisotropic | 1,675 |
| median | 402 |
| uniform | 402 |
| dog | 1,340 |
| dog_multiscale | 2,412 |
| log | 402 |
| morphological_square | 1,608 |
| morphological_rect | 3,216 |
| morphological_diamond | 1,005 |
| morphological_ellipse | 5,025 |
| morphological_gradient | 1,005 |
| tophat_combined | 335 |
| anisotropic_diffusion | 4,288 |
| rolling_ball | 335 |
| local_polynomial | 804 |
| guided | 1,072 |
| polynomial_high | 201 |
| wavelet_biorthogonal | 1,675 |
| wavelet_reverse_biorthogonal | 1,675 |

---

## 3. Detailed Results

Each combination produces a unique residual image capturing 
terrain features at specific scales and with specific characteristics.

### Sample Results (first 50)

| Combo ID | Decomp Params | Upsamp Params | Mean | Std | Hash |
|----------|---------------|---------------|------|-----|------|
| wavelet_biorthogonal_level4_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8, 'kernel_size': 8} | 0.0076 | 1.1853 | 43f970a3d2911c0f |
| wavelet_biorthogonal_level4_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8, 'kernel_size': 16... | 0.0076 | 1.0883 | c2ae641823e49283 |
| wavelet_biorthogonal_level4_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2, 'lambda_reg': 0.0... | 0.0079 | 1.0858 | 8da8c451d271cf69 |
| wavelet_biorthogonal_level4_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2, 'lambda_reg': 0.0... | 0.0079 | 1.0909 | 473bb6df9ec56fbb |
| wavelet_biorthogonal_level4_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2, 'lambda_reg': 0.1... | 0.0060 | 2.3158 | dc6edc9d2f744329 |
| wavelet_biorthogonal_level4_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4, 'lambda_reg': 0.0... | 0.0079 | 1.0831 | 2ac3528a55f851d2 |
| wavelet_biorthogonal_level4_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4, 'lambda_reg': 0.0... | 0.0078 | 1.0848 | 69b30df71dab52a5 |
| wavelet_biorthogonal_level4_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4, 'lambda_reg': 0.1... | 0.0073 | 1.5075 | 8152174345f081dc |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2} | 0.0042 | 2.0727 | 570ed66793894531 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4} | 0.0042 | 2.0727 | e7e0e288bc969ae1 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8} | 0.0042 | 2.0727 | c989100a6f225989 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4} | 0.0038 | 2.0692 | 2f0bbb5917d5f9d8 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8} | 0.0038 | 2.0674 | 0c48733ab3708141 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2} | 0.0038 | 2.0576 | fc2b3a21199c5c98 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4} | 0.0037 | 2.0555 | 0aa1378907f12328 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8} | 0.0036 | 2.0545 | e7182b146ae78760 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2} | 0.0038 | 2.0679 | a6b62622868a2ac5 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4} | 0.0037 | 2.0664 | d1ae78e5145901a7 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8} | 0.0036 | 2.0655 | c2adf475b8f1a330 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2} | 0.0038 | 2.0686 | 9130ee2f345ff525 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4} | 0.0037 | 2.0669 | 2bbc75a9ccdef2ea |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8} | 0.0036 | 2.0661 | db0274736cf01e2a |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2} | 0.0038 | 2.0687 | 538e9a0cdeb556a4 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4} | 0.0037 | 2.0671 | c5369cf5c7c6f41f |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8} | 0.0036 | 2.0662 | 80cb6a2d96be3860 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 3} | 0.0037 | 2.0671 | ac999cc791e1d224 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 16} | 0.0036 | 2.0653 | 2617492403b7bad1 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2} | 0.0042 | 2.0727 | 0fbc1e6212f9367f |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4} | 0.0042 | 2.0727 | 5f03fd00f883dcf9 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8} | 0.0042 | 2.0727 | 70fb31aedae10004 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2} | 0.0042 | 2.0587 | 7e64293316d3bcc1 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4} | 0.0042 | 2.0599 | 7dd44811f1a9fead |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8} | 0.0042 | 2.0602 | ee43c50ae7874b08 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2, 'kernel_size': 4} | 0.0042 | 2.0742 | 74b30db59b9bf0bf |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2, 'kernel_size': 8} | 0.0042 | 2.0800 | 4d195c0e83511cee |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2, 'kernel_size': 16... | 0.0042 | 2.0823 | a502951dc3f5c08e |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4, 'kernel_size': 4} | 0.0042 | 2.1125 | 6185dca4d386385a |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4, 'kernel_size': 8} | 0.0042 | 2.0777 | d2346d1844e71778 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4, 'kernel_size': 16... | 0.0042 | 2.0840 | de0a2cda1f384d05 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8, 'kernel_size': 4} | 0.0042 | 2.9445 | a43ebdc2121982b1 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8, 'kernel_size': 8} | 0.0042 | 2.1143 | a9d6d56309958221 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8, 'kernel_size': 16... | 0.0042 | 2.0791 | 39b95d74b91fbdb5 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2, 'kernel_size': 4} | 0.0042 | 2.0717 | 1097f77f5988f965 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2, 'kernel_size': 8} | 0.0042 | 2.0779 | 0bc1950719e97295 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 2, 'kernel_size': 16... | 0.0042 | 2.0794 | 30d681c3f9d8086c |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4, 'kernel_size': 4} | 0.0042 | 2.2441 | 2e2ec6615a6a5908 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4, 'kernel_size': 8} | 0.0042 | 2.0744 | 87467c7d5cb6fc2b |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 4, 'kernel_size': 16... | 0.0042 | 2.0813 | ca43a411534a056c |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8, 'kernel_size': 4} | 0.0042 | 3.7390 | 9362d2d8345d4ff9 |
| wavelet_biorthogonal_level5_waveletbior3... | {'wavelet': 'bior3.5', 'level'... | {'scale': 8, 'kernel_size': 8} | 0.0042 | 2.2461 | f8785843a18d1c56 |

---

## 4. Reproducibility

All results can be reproduced by running:

```bash
# Generate all 39,731 combinations
python run_exhaustive.py --dem <path_to_dem>

# Generate SHA-256 checksums
python generate_checksums.py --source <results_dir>

# Run redundancy analysis
python analyze_redundancy.py --results <results_dir> --checksums results/CHECKSUMS.txt
```

### Verification

- **SHA-256 checksums**: All 39,731 output files are hashed and recorded in `results/CHECKSUMS.txt`
- **Statistical fingerprints**: Compact 26-dimensional signatures for near-duplicate detection
- **Correlation verification**: 100/100 sampled near-duplicate pairs verified with r > 0.99

---

## 5. License

This work is released under the Apache License 2.0.

The methods, parameter combinations, and results documented herein 
constitute prior art and are disclosed publicly to prevent exclusive 
claims or patents on these specific applications of signal processing 
to Digital Elevation Model analysis for feature detection.
