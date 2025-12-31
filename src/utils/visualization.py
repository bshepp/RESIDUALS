"""
Visualization Utilities

Functions for visualizing DEMs, differentials, and analysis results.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging

logger = logging.getLogger(__name__)


def _timestamped_filename(base_name: str, ext: str = 'png') -> str:
    """Generate a timestamped filename to avoid overwriting."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.{ext}"


def visualize_dem(
    dem: np.ndarray,
    title: str = "DEM",
    output_path: Optional[str] = None,
    cmap: str = 'terrain',
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Visualize a DEM with optional hillshading.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(dem, cmap=cmap, origin='lower')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Elevation')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def visualize_decomposition(
    original: np.ndarray,
    trend: np.ndarray,
    residual: np.ndarray,
    method_name: str,
    output_path: Optional[str] = None
) -> None:
    """
    Visualize a decomposition result showing original, trend, and residual.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    im0 = axes[0].imshow(original, cmap='terrain', origin='lower')
    axes[0].set_title('Original DEM')
    plt.colorbar(im0, ax=axes[0])
    
    # Trend
    im1 = axes[1].imshow(trend, cmap='terrain', origin='lower')
    axes[1].set_title(f'Trend ({method_name})')
    plt.colorbar(im1, ax=axes[1])
    
    # Residual
    vmax = np.percentile(np.abs(residual), 99)
    im2 = axes[2].imshow(residual, cmap='RdBu_r', origin='lower', 
                         vmin=-vmax, vmax=vmax)
    axes[2].set_title(f'Residual ({method_name})')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def visualize_differential(
    diff: np.ndarray,
    title: str,
    output_path: Optional[str] = None,
    percentile_clip: float = 99
) -> None:
    """
    Visualize a differential with symmetric colormap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmax = np.percentile(np.abs(diff), percentile_clip)
    
    im = ax.imshow(diff, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Difference')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def visualize_top_results(
    differentials: Dict[str, np.ndarray],
    ranked_results: List[Tuple[str, float, Dict[str, Any]]],
    output_dir: str,
    top_n: int = 10
) -> None:
    """
    Visualize the top-ranked differential results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (key, score, analysis) in enumerate(ranked_results[:top_n]):
        if key not in differentials:
            continue
        
        diff = differentials[key]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), 
                                  gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.1})
        
        # Differential image
        vmax = np.percentile(np.abs(diff), 99)
        im = axes[0].imshow(diff, cmap='RdBu_r', origin='lower', 
                           vmin=-vmax, vmax=vmax, aspect='auto')
        
        # Shorter title with wrapped key name
        key_short = key.replace('_vs_', '\nvs ')
        axes[0].set_title(f'#{i+1}: {key_short}', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=axes[0], shrink=0.8)
        
        # Analysis metrics
        metrics_text = [
            f"Score: {score:.1f}",
            f"Linear Features: {analysis.get('linear_feature_count', 0)}",
            f"Linear Strength: {analysis.get('max_linear_strength', 0):.1f}",
            f"Spatial Autocorr: {analysis.get('spatial_autocorr', 0):.3f}",
            f"Feature SNR: {analysis.get('feature_snr', 0):.2f}",
            f"Wavelength: {analysis.get('dominant_freq_wavelength', 0):.1f} px",
            f"Kurtosis: {analysis.get('kurtosis', 0):.2f}"
        ]
        
        axes[1].text(0.05, 0.5, '\n'.join(metrics_text), 
                    transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].axis('off')
        axes[1].set_title('Metrics', fontsize=9, fontweight='bold')
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"rank_{i+1:02d}_{key[:35]}_{timestamp}.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    logger.info(f"Saved top {top_n} visualizations to {output_dir}")


def visualize_results(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: str,
    original_dem: Optional[np.ndarray] = None
) -> None:
    """
    Visualize all decomposition × upsampling results in a grid.
    Stitches images directly for minimal spacing.
    
    Args:
        results: Nested dict of results[decomp][upsamp]['residual']
        output_dir: Output directory
        original_dem: Optional original DEM to show as ground truth column
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    decomp_names = list(results.keys())
    if not decomp_names:
        return
    
    upsamp_names = list(results[decomp_names[0]].keys())
    if not upsamp_names:
        return
    
    n_decomp = len(decomp_names)
    n_upsamp = len(upsamp_names)
    
    # Get dimensions from first image
    first_res = results[decomp_names[0]][upsamp_names[0]]['residual']
    img_h, img_w = first_res.shape
    
    # Small gap between images (in pixels)
    gap = 4
    
    # Add ground truth column if provided, plus divergence columns
    # Layout: DEM | (residual | Δ from GT) × n_upsamp | Div | ΔDiv
    has_ground_truth = original_dem is not None
    has_divergence = n_upsamp > 1  # Need multiple methods to show divergence
    has_gt_comparison = has_ground_truth  # Show residual vs ground truth comparison
    has_delta_divergence = has_gt_comparison and has_divergence  # Level 4: divergence of Δ columns
    
    # Each upsampling method gets 2 columns if GT comparison enabled
    upsamp_cols = n_upsamp * (2 if has_gt_comparison else 1)
    n_cols = (upsamp_cols + 
              (1 if has_ground_truth else 0) + 
              (1 if has_divergence else 0) +
              (1 if has_delta_divergence else 0))
    
    # Calculate total size
    total_h = n_decomp * img_h + (n_decomp - 1) * gap
    total_w = n_cols * img_w + (n_cols - 1) * gap
    
    # Create composite array (white background)
    composite = np.ones((total_h, total_w, 3))
    
    # Colormaps
    cmap_residual = plt.cm.RdBu_r
    
    # If we have ground truth, resize it and compute hillshading + gradient magnitude
    gt_gradient = None
    if has_ground_truth:
        from scipy.ndimage import zoom
        scale_h = img_h / original_dem.shape[0]
        scale_w = img_w / original_dem.shape[1]
        dem_scaled = zoom(original_dem, (scale_h, scale_w), order=1)
        
        # Compute hillshading
        # Calculate gradients
        dy, dx = np.gradient(dem_scaled)
        
        # Sun position (azimuth and altitude in radians)
        azimuth = 315 * np.pi / 180  # NW illumination
        altitude = 45 * np.pi / 180  # 45 degree sun angle
        
        # Calculate hillshade
        slope = np.sqrt(dx**2 + dy**2)
        aspect = np.arctan2(-dx, dy)
        
        shaded = (np.cos(altitude) * np.sin(np.arctan(slope)) * 
                  np.cos(azimuth - aspect) + 
                  np.sin(altitude) * np.cos(np.arctan(slope)))
        
        # Normalize to 0-1 range
        shaded = (shaded - shaded.min()) / (shaded.max() - shaded.min())
        
        # Convert to RGB (grayscale)
        hillshade_rgb = np.stack([shaded, shaded, shaded], axis=-1)
        
        # Store gradient magnitude for comparison (normalized)
        # This represents "ground truth features" - where terrain has structure
        gt_gradient = slope  # Already computed above
        
        # Place hillshade in first column for each row
        for i in range(n_decomp):
            y_start = i * (img_h + gap)
            composite[y_start:y_start + img_h, 0:img_w] = hillshade_rgb
    
    # Column offset if ground truth is present
    col_offset = 1 if has_ground_truth else 0
    
    # Compute shared vmax per row (same decomposition, all upsampling methods)
    # This makes differences between upsampling methods visible
    row_vmax = {}
    for decomp_name in decomp_names:
        all_residuals = [results[decomp_name][u]['residual'] 
                        for u in upsamp_names if u in results[decomp_name]]
        if all_residuals:
            combined = np.concatenate([np.abs(r).flatten() for r in all_residuals])
            row_vmax[decomp_name] = np.percentile(combined, 99)
            if row_vmax[decomp_name] == 0:
                row_vmax[decomp_name] = 1
    
    # Colormap for GT comparison (green = residual matches GT features, magenta = mismatch)
    cmap_comparison = plt.cm.PiYG  # Diverging green-magenta
    
    # Store delta values for computing delta-divergence (Level 4)
    delta_values = {decomp: [] for decomp in decomp_names}
    
    for i, decomp_name in enumerate(decomp_names):
        vmax = row_vmax.get(decomp_name, 1)
        
        for j, upsamp_name in enumerate(upsamp_names):
            if upsamp_name not in results[decomp_name]:
                continue
                
            residual = results[decomp_name][upsamp_name]['residual']
            
            # Normalize using shared row vmax
            norm = (residual + vmax) / (2 * vmax)
            norm = np.clip(norm, 0, 1)
            
            # Apply colormap
            colored = cmap_residual(norm)[:, :, :3]
            
            # Position in composite (offset by ground truth column)
            # Each upsampling method gets 2 columns if GT comparison enabled
            y_start = i * (img_h + gap)
            col_idx = col_offset + j * (2 if has_gt_comparison else 1)
            x_start = col_idx * (img_w + gap)
            composite[y_start:y_start + img_h, x_start:x_start + img_w] = colored
            
            # Add GT comparison column if enabled
            if has_gt_comparison and gt_gradient is not None:
                # Compare absolute residual to GT gradient magnitude
                # Normalize both to same scale
                res_abs = np.abs(residual)
                res_norm = res_abs / np.percentile(res_abs, 99) if res_abs.max() > 0 else res_abs
                gt_norm = gt_gradient / np.percentile(gt_gradient, 99) if gt_gradient.max() > 0 else gt_gradient
                
                # Difference: positive = residual stronger, negative = GT stronger
                diff = res_norm - gt_norm
                
                # Store for delta-divergence calculation
                delta_values[decomp_name].append(diff)
                
                diff_vmax = np.percentile(np.abs(diff), 99)
                if diff_vmax == 0:
                    diff_vmax = 1
                diff_normalized = (diff + diff_vmax) / (2 * diff_vmax)
                diff_normalized = np.clip(diff_normalized, 0, 1)
                
                diff_colored = cmap_comparison(diff_normalized)[:, :, :3]
                
                # Place in next column
                x_start_diff = (col_idx + 1) * (img_w + gap)
                composite[y_start:y_start + img_h, x_start_diff:x_start_diff + img_w] = diff_colored
    
    # Add divergence column (standard deviation across upsampling methods)
    if has_divergence:
        cmap_divergence = plt.cm.hot  # Hot colormap for divergence
        # Divergence is the last column
        divergence_col = col_offset + n_upsamp * (2 if has_gt_comparison else 1)
        
        for i, decomp_name in enumerate(decomp_names):
            # Stack all residuals for this decomposition
            all_res = np.stack([
                results[decomp_name][u]['residual'] 
                for u in upsamp_names if u in results[decomp_name]
            ], axis=0)
            
            # Compute standard deviation across upsampling methods
            divergence = np.std(all_res, axis=0)
            
            # Normalize divergence to 0-1 (use 99th percentile for vmax)
            div_vmax = np.percentile(divergence, 99)
            if div_vmax == 0:
                div_vmax = 1
            div_norm = np.clip(divergence / div_vmax, 0, 1)
            
            # Apply hot colormap
            div_colored = cmap_divergence(div_norm)[:, :, :3]
            
            # Place in composite
            y_start = i * (img_h + gap)
            x_start = divergence_col * (img_w + gap)
            composite[y_start:y_start + img_h, x_start:x_start + img_w] = div_colored
    
    # Add delta-divergence column (Level 4: divergence of Δ columns)
    if has_delta_divergence:
        cmap_delta_div = plt.cm.viridis  # Different colormap for meta-divergence
        delta_div_col = divergence_col + 1
        
        for i, decomp_name in enumerate(decomp_names):
            if delta_values[decomp_name]:
                # Stack all delta values for this decomposition
                all_deltas = np.stack(delta_values[decomp_name], axis=0)
                
                # Compute standard deviation across upsampling methods' deltas
                delta_divergence = np.std(all_deltas, axis=0)
                
                # Normalize
                ddiv_vmax = np.percentile(delta_divergence, 99)
                if ddiv_vmax == 0:
                    ddiv_vmax = 1
                ddiv_norm = np.clip(delta_divergence / ddiv_vmax, 0, 1)
                
                # Apply colormap
                ddiv_colored = cmap_delta_div(ddiv_norm)[:, :, :3]
                
                # Place in composite
                y_start = i * (img_h + gap)
                x_start = delta_div_col * (img_w + gap)
                composite[y_start:y_start + img_h, x_start:x_start + img_w] = ddiv_colored
    
    # Create figure sized to content
    dpi = 150
    fig_w = total_w / dpi + 0.8  # Extra for labels
    fig_h = total_h / dpi + 0.5  # Extra for title
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(composite, origin='upper')
    ax.axis('off')
    
    # Add column labels
    all_col_names = []
    if has_ground_truth:
        all_col_names.append('DEM')
    for upsamp in upsamp_names:
        all_col_names.append(upsamp)
        if has_gt_comparison:
            all_col_names.append(f'Δ{upsamp[:3]}')  # Short delta label
    if has_divergence:
        all_col_names.append('Div')
    if has_delta_divergence:
        all_col_names.append('ΔDiv')  # Level 4: meta-divergence
    
    for j, col_name in enumerate(all_col_names):
        x = j * (img_w + gap) + img_w / 2
        is_delta = col_name.startswith('Δ')
        is_special = col_name in ('DEM', 'Div', 'ΔDiv')
        ax.text(x, -10, col_name, ha='center', va='bottom', fontsize=6,
               fontweight='bold' if is_special else 'normal',
               style='italic' if is_delta else 'normal')
    
    # Add row labels
    for i, decomp_name in enumerate(decomp_names):
        y = i * (img_h + gap) + img_h / 2
        ax.text(-10, y, decomp_name, ha='right', va='center', fontsize=6, rotation=0)
    
    ax.set_title('Decomposition × Upsampling Residuals', fontsize=9, pad=15)
    
    # Save with timestamp to avoid overwriting
    filename = _timestamped_filename('residuals_grid')
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    plt.close()
    
    logger.info(f"Saved results grid: {filepath}")

