"""
Feature Detection and Ranking Module

Analyzes differential outputs to detect and rank potential features:
- Linear features (roads, walls) via Hough transform
- Circular features (mounds) via morphological analysis
- Periodic structures via frequency analysis
- Spatial autocorrelation patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.ndimage import correlate
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import logging

logger = logging.getLogger(__name__)


def analyze_features(
    diff: np.ndarray,
    original_dem: np.ndarray = None
) -> Dict[str, Any]:
    """
    Comprehensive feature analysis of a differential.
    
    Args:
        diff: Differential array to analyze
        original_dem: Optional original DEM for context
        
    Returns:
        Dict of analysis metrics
    """
    analysis = {}
    
    # Handle NaN
    diff_clean = np.nan_to_num(diff, nan=0)
    
    # ==========================================================================
    # Basic Statistics
    # ==========================================================================
    analysis['mean'] = float(np.mean(diff_clean))
    analysis['std'] = float(np.std(diff_clean))
    analysis['min'] = float(np.min(diff_clean))
    analysis['max'] = float(np.max(diff_clean))
    analysis['range'] = analysis['max'] - analysis['min']
    
    # Distribution shape
    flat = diff_clean.flatten()
    analysis['skewness'] = float(stats.skew(flat))
    analysis['kurtosis'] = float(stats.kurtosis(flat))
    
    # ==========================================================================
    # Linear Feature Detection (Hough Transform)
    # ==========================================================================
    try:
        linear_metrics = detect_linear_features(diff_clean)
        analysis.update(linear_metrics)
    except Exception as e:
        logger.warning(f"Linear feature detection failed: {e}")
        analysis['linear_feature_count'] = 0
        analysis['max_linear_strength'] = 0.0
    
    # ==========================================================================
    # Spatial Autocorrelation (Moran's I approximation)
    # ==========================================================================
    try:
        moran = compute_spatial_autocorrelation(diff_clean)
        analysis['spatial_autocorr'] = moran
    except Exception as e:
        logger.warning(f"Spatial autocorrelation failed: {e}")
        analysis['spatial_autocorr'] = 0.0
    
    # ==========================================================================
    # Frequency Content Analysis
    # ==========================================================================
    try:
        freq_metrics = analyze_frequency_content(diff_clean)
        analysis.update(freq_metrics)
    except Exception as e:
        logger.warning(f"Frequency analysis failed: {e}")
        analysis['low_freq_energy'] = 0.0
        analysis['mid_freq_energy'] = 0.0
        analysis['high_freq_energy'] = 0.0
    
    # ==========================================================================
    # Feature Distinctness
    # ==========================================================================
    # How much the features stand out from background
    analysis['feature_snr'] = compute_feature_snr(diff_clean)
    
    return analysis


def detect_linear_features(
    diff: np.ndarray,
    canny_sigma: float = 2.0,
    num_peaks: int = 20
) -> Dict[str, Any]:
    """
    Detect linear features using Hough transform.
    
    Good for roads, walls, ancient pathways.
    """
    # Normalize for edge detection
    diff_norm = diff - diff.min()
    diff_max = diff_norm.max()
    if diff_max > 0:
        diff_norm = diff_norm / diff_max
    
    # Edge detection
    edges = canny(diff_norm, sigma=canny_sigma)
    
    # Hough transform
    h, theta, d = hough_line(edges)
    
    # Find peaks
    accum, angles, dists = hough_line_peaks(h, theta, d, num_peaks=num_peaks)
    
    # Analyze orientations
    if len(angles) > 0:
        # Convert to degrees
        angles_deg = np.degrees(angles)
        
        # Detect dominant orientations
        hist, bin_edges = np.histogram(angles_deg, bins=18, range=(-90, 90))
        dominant_orientation = bin_edges[np.argmax(hist)]
        orientation_entropy = stats.entropy(hist + 1e-10)
    else:
        dominant_orientation = 0.0
        orientation_entropy = 0.0
    
    return {
        'linear_feature_count': len(accum),
        'max_linear_strength': float(np.max(accum)) if len(accum) > 0 else 0.0,
        'mean_linear_strength': float(np.mean(accum)) if len(accum) > 0 else 0.0,
        'dominant_orientation_deg': float(dominant_orientation),
        'orientation_entropy': float(orientation_entropy)
    }


def compute_spatial_autocorrelation(diff: np.ndarray) -> float:
    """
    Compute Moran's I spatial autocorrelation.
    
    High values indicate spatially structured features.
    """
    # 4-neighbor kernel
    kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]) / 4
    
    # Compute lagged values
    lagged = correlate(diff, kernel, mode='reflect')
    
    # Moran's I numerator and denominator
    diff_centered = diff - diff.mean()
    lagged_centered = lagged - lagged.mean()
    
    numerator = np.sum(diff_centered * lagged_centered)
    denominator = np.sum(diff_centered ** 2)
    
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)


def analyze_frequency_content(diff: np.ndarray) -> Dict[str, float]:
    """
    Analyze frequency distribution using FFT.
    
    Helps identify periodic structures and noise patterns.
    """
    # Compute 2D FFT
    fft = np.fft.fft2(diff)
    fft_mag = np.abs(np.fft.fftshift(fft))
    
    # Compute radial distance from center
    center = np.array(fft_mag.shape) // 2
    y, x = np.ogrid[:fft_mag.shape[0], :fft_mag.shape[1]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    max_r = min(center)
    
    # Define frequency bands
    low_mask = r < max_r * 0.2
    mid_mask = (r >= max_r * 0.2) & (r < max_r * 0.6)
    high_mask = r >= max_r * 0.6
    
    # Compute energy in each band
    low_energy = np.mean(fft_mag[low_mask])
    mid_energy = np.mean(fft_mag[mid_mask])
    high_energy = np.mean(fft_mag[high_mask])
    
    # Compute dominant frequency
    # Mask out DC component
    fft_mag_nodc = fft_mag.copy()
    fft_mag_nodc[center[0]-2:center[0]+3, center[1]-2:center[1]+3] = 0
    
    peak_idx = np.unravel_index(np.argmax(fft_mag_nodc), fft_mag.shape)
    dominant_freq_r = np.sqrt(
        (peak_idx[0] - center[0])**2 + (peak_idx[1] - center[1])**2
    )
    
    return {
        'low_freq_energy': float(low_energy),
        'mid_freq_energy': float(mid_energy),
        'high_freq_energy': float(high_energy),
        'freq_ratio_high_low': float(high_energy / (low_energy + 1e-10)),
        'dominant_freq_radius': float(dominant_freq_r),
        'dominant_freq_wavelength': float(min(diff.shape) / (dominant_freq_r + 1e-10))
    }


def compute_feature_snr(diff: np.ndarray, percentile: float = 95) -> float:
    """
    Compute signal-to-noise ratio for features.
    
    Features are defined as values above the percentile threshold.
    """
    threshold = np.percentile(np.abs(diff), percentile)
    
    # "Signal" = high values
    signal_mask = np.abs(diff) >= threshold
    signal_values = diff[signal_mask]
    
    # "Noise" = low values
    noise_values = diff[~signal_mask]
    
    if len(signal_values) == 0 or np.std(noise_values) == 0:
        return 0.0
    
    snr = np.std(signal_values) / np.std(noise_values)
    return float(snr)


def rank_for_features(
    differentials: Dict[str, np.ndarray],
    analyses: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Rank differentials by potential for feature detection.
    
    Scoring emphasizes:
    - Linear features (roads, walls)
    - High spatial autocorrelation (structured features)
    - Mid-frequency content (human-scale features)
    - Feature distinctness
    
    Returns:
        List of (key, score, analysis) tuples, sorted by score descending
    """
    scores = []
    
    for key, analysis in analyses.items():
        score = 0.0
        
        # Linear features are key for roads, walls, pathways
        score += analysis.get('linear_feature_count', 0) * 10
        score += analysis.get('max_linear_strength', 0) * 0.1
        
        # Spatial autocorrelation indicates structured features
        autocorr = analysis.get('spatial_autocorr', 0)
        if autocorr > 0:
            score += autocorr * 100
        
        # Mid-frequency content is human-scale (not too large, not noise)
        mid_freq = analysis.get('mid_freq_energy', 0)
        low_freq = analysis.get('low_freq_energy', 1)
        high_freq = analysis.get('high_freq_energy', 1)
        
        # Prefer high mid relative to both low and high
        if low_freq > 0 and high_freq > 0:
            mid_ratio = mid_freq / np.sqrt(low_freq * high_freq)
            score += mid_ratio * 20
        
        # Feature distinctness
        score += analysis.get('feature_snr', 0) * 15
        
        # Moderate kurtosis suggests distinct features (not uniform or noise)
        kurtosis = abs(analysis.get('kurtosis', 0))
        if 1 < kurtosis < 10:
            score += 20
        elif kurtosis > 10:
            score += 10  # Very peaked, might be too concentrated
        
        # Penalize very low std (essentially flat)
        if analysis.get('std', 0) < 0.01:
            score *= 0.1
        
        scores.append((key, score, analysis))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores


def generate_analysis_report(
    ranked_results: List[Tuple[str, float, Dict[str, Any]]],
    top_n: int = 20
) -> str:
    """
    Generate a markdown report of analysis results.
    """
    lines = [
        "# Differential Analysis Report",
        "",
        "## Top Candidates for Feature Detection",
        "",
        "| Rank | Combination | Score | Linear Features | Spatial Autocorr | Feature SNR |",
        "|------|-------------|-------|-----------------|------------------|-------------|"
    ]
    
    for i, (key, score, analysis) in enumerate(ranked_results[:top_n]):
        lines.append(
            f"| {i+1} | {key} | {score:.1f} | "
            f"{analysis.get('linear_feature_count', 0)} | "
            f"{analysis.get('spatial_autocorr', 0):.3f} | "
            f"{analysis.get('feature_snr', 0):.2f} |"
        )
    
    lines.extend([
        "",
        "## Metric Explanations",
        "",
        "- **Linear Features**: Count of detected linear structures (roads, walls)",
        "- **Spatial Autocorr**: Moran's I (high = spatially structured)",
        "- **Feature SNR**: Signal-to-noise ratio of prominent features",
        ""
    ])
    
    return "\n".join(lines)

