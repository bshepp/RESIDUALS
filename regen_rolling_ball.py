#!/usr/bin/env python3
"""
Regenerate rolling_ball_radius200 combinations after fixing the edge case.
"""

import numpy as np
from pathlib import Path
import sys
from itertools import product
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.decomposition.methods_extended import decompose_rolling_ball
from src.upsampling.registry import UPSAMPLING_REGISTRY, run_upsampling


def main():
    # Load the DEM used in exhaustive run
    dem_path = Path('data/test_dems/fairfield_sample_1.5ft.npy')
    dem = np.load(dem_path)
    print(f"Loaded DEM: {dem.shape}")
    
    # Output directory
    output_dir = Path('D:/DIVERGE_exhaustive/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run decomposition with radius=200
    print("Running rolling_ball decomposition with radius=200...")
    trend, residual = decompose_rolling_ball(dem, radius=200)
    print(f"Decomposition complete: trend={trend.shape}, residual={residual.shape}")
    
    # Get all upsampling combinations
    upsampling_combos = []
    for name, method in UPSAMPLING_REGISTRY.items():
        param_ranges = method.param_ranges
        if not param_ranges:
            upsampling_combos.append((name, method.default_params.copy()))
        else:
            param_names = list(param_ranges.keys())
            param_values = [param_ranges[p] for p in param_names]
            for values in product(*param_values):
                combo = dict(zip(param_names, values))
                upsampling_combos.append((name, combo))
    
    print(f"Generating {len(upsampling_combos)} upsampled outputs...")
    
    success = 0
    failed = 0
    
    for upsample_name, upsample_params in tqdm(upsampling_combos):
        # Create unique filename
        decomp_str = "rolling_ball_radius200"
        upsample_str = upsample_name
        for k, v in sorted(upsample_params.items()):
            upsample_str += f"_{k}{v}"
        
        output_file = output_dir / f"{decomp_str}___{upsample_str}.npy"
        
        if output_file.exists():
            print(f"Skipping existing: {output_file.name}")
            continue
        
        try:
            # Extract scale from params, pass rest as params dict
            params_copy = upsample_params.copy()
            scale = params_copy.pop('scale', 2)
            extra_params = params_copy if params_copy else None
            upsampled = run_upsampling(upsample_name, residual, scale=scale, params=extra_params)
            np.save(output_file, upsampled.astype(np.float32))
            success += 1
        except Exception as e:
            print(f"Failed: {output_file.name}: {e}")
            failed += 1
    
    print(f"\nComplete: {success} success, {failed} failed")


if __name__ == '__main__':
    main()
