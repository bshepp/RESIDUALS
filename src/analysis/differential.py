"""
Differential Computation Module

Computes differences between decomposition × upsampling method outputs
to identify feature-specific extraction filters.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from itertools import combinations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_differential(
    result_a: np.ndarray,
    result_b: np.ndarray,
    method: str = 'subtract'
) -> np.ndarray:
    """
    Compute differential between two method outputs.
    
    Args:
        result_a: First result array
        result_b: Second result array
        method: Differential method
            - 'subtract': Simple difference (a - b)
            - 'absolute': Absolute difference |a - b|
            - 'normalized': Normalized difference (a - b) / std
            - 'ratio': Ratio a / b
    
    Returns:
        Differential array
    """
    # Ensure same shape
    min_h = min(result_a.shape[0], result_b.shape[0])
    min_w = min(result_a.shape[1], result_b.shape[1])
    a = result_a[:min_h, :min_w]
    b = result_b[:min_h, :min_w]
    
    if method == 'subtract':
        return a - b
    elif method == 'absolute':
        return np.abs(a - b)
    elif method == 'normalized':
        diff = a - b
        std = np.std(diff)
        return diff / (std + 1e-10)
    elif method == 'ratio':
        return a / (b + 1e-10)
    else:
        raise ValueError(f"Unknown differential method: {method}")


def run_all_combinations(
    dem: np.ndarray,
    decomposition_methods: List[str],
    upsampling_methods: List[str],
    scale: int = 2,
    output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run all decomposition × upsampling combinations.
    
    Args:
        dem: Input DEM array
        decomposition_methods: List of decomposition method names
        upsampling_methods: List of upsampling method names
        scale: Upsampling scale factor
        output_dir: Optional directory to save intermediate results
    
    Returns:
        Nested dict: results[decomp_name][upsamp_name] = {'trend': ..., 'residual': ...}
    """
    from ..decomposition import run_decomposition
    from ..upsampling import run_upsampling
    
    results = {}
    total = len(decomposition_methods) * len(upsampling_methods)
    current = 0
    
    for decomp_name in decomposition_methods:
        logger.info(f"Decomposition: {decomp_name}")
        
        try:
            trend, residual = run_decomposition(decomp_name, dem)
        except Exception as e:
            logger.error(f"  Error in {decomp_name}: {e}")
            continue
        
        results[decomp_name] = {}
        
        for upsamp_name in upsampling_methods:
            current += 1
            logger.info(f"  [{current}/{total}] Upsampling: {upsamp_name}")
            
            try:
                trend_up = run_upsampling(upsamp_name, trend, scale=scale)
                residual_up = run_upsampling(upsamp_name, residual, scale=scale)
                
                results[decomp_name][upsamp_name] = {
                    'trend': trend_up,
                    'residual': residual_up
                }
                
                # Optionally save to disk
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    np.save(
                        output_dir / f"{decomp_name}_{upsamp_name}_trend.npy",
                        trend_up
                    )
                    np.save(
                        output_dir / f"{decomp_name}_{upsamp_name}_residual.npy",
                        residual_up
                    )
                    
            except Exception as e:
                logger.error(f"    Error: {e}")
                continue
    
    return results


def compute_all_differentials(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    differential_method: str = 'subtract',
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Compute differentials between all pairs of method outputs.
    
    Args:
        results: Output from run_all_combinations()
        differential_method: Method for computing differences
        output_dir: Optional directory to save differentials
    
    Returns:
        Dict mapping differential key to array
        Key format: "{decomp1}_{upsamp1}_vs_{decomp2}_{upsamp2}"
    """
    differentials = {}
    
    # Get all (decomp, upsamp) combinations that succeeded
    all_combos = []
    for decomp_name, upsamp_results in results.items():
        for upsamp_name in upsamp_results.keys():
            all_combos.append((decomp_name, upsamp_name))
    
    logger.info(f"Computing differentials for {len(all_combos)} combinations "
                f"({len(all_combos) * (len(all_combos) - 1) // 2} pairs)")
    
    # Compare all pairs (residuals)
    for i, (decomp1, upsamp1) in enumerate(all_combos):
        for decomp2, upsamp2 in all_combos[i + 1:]:
            key = f"{decomp1}_{upsamp1}_vs_{decomp2}_{upsamp2}"
            
            res1 = results[decomp1][upsamp1]['residual']
            res2 = results[decomp2][upsamp2]['residual']
            
            diff = compute_differential(res1, res2, method=differential_method)
            differentials[key] = diff
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                np.save(output_dir / f"diff_{key}.npy", diff)
    
    logger.info(f"Computed {len(differentials)} differentials")
    
    return differentials


def compute_method_differentials(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    differential_method: str = 'subtract'
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute differentials organized by comparison type.
    
    Returns separate dicts for:
    - Same decomp, different upsamp
    - Different decomp, same upsamp
    - Both different
    
    This helps identify which component (decomp vs upsamp) drives differences.
    """
    same_decomp = {}  # Same decomposition, different upsampling
    same_upsamp = {}  # Different decomposition, same upsampling
    both_diff = {}    # Both different
    
    all_combos = []
    for decomp_name, upsamp_results in results.items():
        for upsamp_name in upsamp_results.keys():
            all_combos.append((decomp_name, upsamp_name))
    
    for i, (decomp1, upsamp1) in enumerate(all_combos):
        for decomp2, upsamp2 in all_combos[i + 1:]:
            res1 = results[decomp1][upsamp1]['residual']
            res2 = results[decomp2][upsamp2]['residual']
            diff = compute_differential(res1, res2, method=differential_method)
            
            key = f"{decomp1}_{upsamp1}_vs_{decomp2}_{upsamp2}"
            
            if decomp1 == decomp2:
                same_decomp[key] = diff
            elif upsamp1 == upsamp2:
                same_upsamp[key] = diff
            else:
                both_diff[key] = diff
    
    return {
        'same_decomp_diff_upsamp': same_decomp,
        'diff_decomp_same_upsamp': same_upsamp,
        'both_different': both_diff
    }

