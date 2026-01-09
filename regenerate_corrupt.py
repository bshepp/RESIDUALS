"""
Regenerate corrupt files from the exhaustive run.
These files were truncated due to disk space issues.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

from src.decomposition.registry import DECOMPOSITION_REGISTRY, run_decomposition
from src.upsampling.registry import UPSAMPLING_REGISTRY, run_upsampling


def parse_filename(filename: str) -> tuple:
    """Parse a result filename back into decomposition and upsampling specs."""
    # Format: decomp_params___upsample_params.npy
    name = filename.replace('.npy', '')
    parts = name.split('___')
    if len(parts) != 2:
        return None, None, None, None
    
    decomp_str, upsample_str = parts
    
    # Parse decomposition
    # e.g., "wavelet_biorthogonal_level1_waveletbior2.4"
    # Need to find the method name and parameters
    decomp_method = None
    decomp_params = {}
    
    for method_name in DECOMPOSITION_REGISTRY:
        if decomp_str.startswith(method_name):
            decomp_method = method_name
            param_str = decomp_str[len(method_name):]
            # Parse params like "_level1_waveletbior2.4"
            if param_str.startswith('_'):
                param_str = param_str[1:]
            # Extract key-value pairs
            if param_str:
                decomp_params = parse_params(param_str, DECOMPOSITION_REGISTRY[method_name].param_ranges)
            break
    
    # Parse upsampling
    upsample_method = None
    upsample_params = {}
    
    for method_name in UPSAMPLING_REGISTRY:
        if upsample_str.startswith(method_name):
            upsample_method = method_name
            param_str = upsample_str[len(method_name):]
            if param_str.startswith('_'):
                param_str = param_str[1:]
            if param_str:
                upsample_params = parse_params(param_str, UPSAMPLING_REGISTRY[method_name].param_ranges)
            break
    
    return decomp_method, decomp_params, upsample_method, upsample_params


def parse_params(param_str: str, param_ranges: dict) -> dict:
    """Parse parameter string like 'level1_waveletbior2.4' into dict."""
    params = {}
    
    # Try to match against known parameter names
    remaining = param_str
    for param_name, values in param_ranges.items():
        for value in values:
            pattern = f"{param_name}{value}"
            if pattern in remaining:
                params[param_name] = value
                remaining = remaining.replace(pattern, '', 1)
                if remaining.startswith('_'):
                    remaining = remaining[1:]
                break
    
    return params


def main():
    # Load corrupt file list
    corrupt_list = Path('results/corrupt_files.txt')
    if not corrupt_list.exists():
        print("Error: results/corrupt_files.txt not found")
        return
    
    corrupt_files = [f.strip() for f in open(corrupt_list) if f.strip()]
    print(f"Found {len(corrupt_files)} corrupt files to regenerate")
    
    # Load DEM
    dem_path = Path('data/test_dems/fairfield_sample_1.5ft.npy')
    dem = np.load(dem_path)
    print(f"Loaded DEM: {dem.shape}")
    
    output_dir = Path('D:/DIVERGE_exhaustive/results')
    
    # Cache decomposition results to avoid recomputing
    decomp_cache = {}
    
    success = 0
    failed = 0
    skipped = 0
    
    for filename in tqdm(corrupt_files, desc="Regenerating"):
        output_path = output_dir / filename
        
        # Parse the filename to get method specs
        decomp_method, decomp_params, upsample_method, upsample_params = parse_filename(filename)
        
        if not decomp_method or not upsample_method:
            print(f"Could not parse: {filename}")
            failed += 1
            continue
        
        try:
            # Get or compute decomposition
            decomp_key = f"{decomp_method}_{decomp_params}"
            if decomp_key not in decomp_cache:
                trend, residual = run_decomposition(decomp_method, dem, decomp_params)
                decomp_cache[decomp_key] = residual
                # Keep cache small
                if len(decomp_cache) > 50:
                    oldest = list(decomp_cache.keys())[0]
                    del decomp_cache[oldest]
                    gc.collect()
            
            residual = decomp_cache[decomp_key]
            
            # Run upsampling
            scale = upsample_params.pop('scale', 2)
            extra_params = upsample_params if upsample_params else None
            
            upsampled = run_upsampling(upsample_method, residual, scale=scale, params=extra_params)
            
            # Delete corrupt file and save new one
            if output_path.exists():
                output_path.unlink()
            
            np.save(output_path, upsampled)
            success += 1
            
            # Verify size
            if output_path.stat().st_size < 10000:
                print(f"Warning: {filename} still small after regeneration")
            
            del upsampled
            gc.collect()
            
        except Exception as e:
            print(f"Failed {filename}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")


if __name__ == '__main__':
    main()
