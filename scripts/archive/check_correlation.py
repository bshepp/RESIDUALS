#!/usr/bin/env python3
"""Check correlation between Div and ΔDiv as sanity check."""

import numpy as np
from pathlib import Path
from scipy.ndimage import zoom

# Load data
dem = np.load('data/test_dems/fairfield_sample_1.5ft.npy')
decomp_methods = ['gaussian', 'bilateral', 'wavelet_dwt', 'morphological', 'tophat', 'polynomial']
upsamp_methods = ['bicubic', 'lanczos', 'bspline', 'fft_zeropad']

# Load residuals
residuals = {}
for decomp in decomp_methods:
    residuals[decomp] = {}
    for upsamp in upsamp_methods:
        path = Path(f'results/combinations/{decomp}_{upsamp}_residual.npy')
        if path.exists():
            residuals[decomp][upsamp] = np.load(path)

# Get first residual shape for GT scaling
first_res = residuals[decomp_methods[0]][upsamp_methods[0]]
img_h, img_w = first_res.shape

# Compute GT gradient
scale_h = img_h / dem.shape[0]
scale_w = img_w / dem.shape[1]
dem_scaled = zoom(dem, (scale_h, scale_w), order=1)
dy, dx = np.gradient(dem_scaled)
gt_gradient = np.sqrt(dx**2 + dy**2)
gt_norm = gt_gradient / np.percentile(gt_gradient, 99)

print('Correlation between Div and DeltaDiv:')
print('='*50)

for decomp in decomp_methods:
    # Compute Div (std of residuals)
    all_res = np.stack([residuals[decomp][u] for u in upsamp_methods], axis=0)
    div = np.std(all_res, axis=0)
    
    # Compute ΔDiv (std of delta comparisons)
    deltas = []
    for upsamp in upsamp_methods:
        res = residuals[decomp][upsamp]
        res_abs = np.abs(res)
        res_norm = res_abs / np.percentile(res_abs, 99) if res_abs.max() > 0 else res_abs
        delta = res_norm - gt_norm
        deltas.append(delta)
    
    all_deltas = np.stack(deltas, axis=0)
    delta_div = np.std(all_deltas, axis=0)
    
    # Correlation
    corr = np.corrcoef(div.flatten(), delta_div.flatten())[0, 1]
    print(f'{decomp:15} correlation: {corr:.4f}')

print()
print('High correlation = sanity check passed!')
print('(Div and DeltaDiv measure similar structural disagreement)')

