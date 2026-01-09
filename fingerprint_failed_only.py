#!/usr/bin/env python3
"""
Fingerprint ONLY the files that failed in the original redundancy analysis.
These are the ~1,030 large files that caused memory errors.
"""

import numpy as np
from pathlib import Path
import json
import gc
import sys
from tqdm import tqdm
import warnings
import re

sys.path.insert(0, str(Path(__file__).parent))
from analyze_redundancy import compute_fingerprint


def extract_failed_files_from_log():
    """
    Extract filenames that failed with memory errors from the original run.
    These patterns were logged during the redundancy analysis.
    """
    # Files that failed were primarily:
    # - morphological_diamond_operationopening_radius10/15 with various upsampling at scale 4/8
    # - wavelet_biorthogonal level 1-5 with various upsampling
    # These are the large upsampled files that exceeded memory
    
    # Pattern: look for files with these characteristics
    failed_patterns = [
        # Morphological diamond with large scales
        r'morphological_diamond.*_scale[48]\.npy$',
        r'morphological_diamond.*_scale16\.npy$',
        # Wavelet biorthogonal level 1+ at large scales  
        r'wavelet_biorthogonal_level[1-5].*_scale[48]\.npy$',
        r'wavelet_biorthogonal_level[1-5].*_scale16\.npy$',
        # Wavelet reverse biorthogonal
        r'wavelet_reverse_biorthogonal.*_scale[48]\.npy$',
    ]
    
    return failed_patterns


def find_large_files(results_dir: Path, size_threshold_mb: float = 500) -> list:
    """Find files larger than threshold that might have failed."""
    large_files = []
    for f in results_dir.glob('*.npy'):
        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb > size_threshold_mb:
            large_files.append(f)
    return sorted(large_files, key=lambda x: x.stat().st_size, reverse=True)


def main():
    results_dir = Path(r'D:\DIVERGE_exhaustive\results')
    
    # Load the FULL fingerprint data from the original analysis if it exists
    # The original run completed 38,634 files successfully
    original_json = Path(r'F:\science-projects\DIVERGE\results\REDUNDANCY_REPORT.json')
    
    # Load our checkpoint (has ~1,600 fingerprints from recent run)
    checkpoint_file = Path(r'F:\science-projects\DIVERGE\results\fingerprint_checkpoint.json')
    
    existing = {}
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} fingerprints from checkpoint")
    
    # Find files that are likely the ones that failed (large files >500MB)
    print("\nFinding large files (>500MB) that likely failed...")
    large_files = find_large_files(results_dir, size_threshold_mb=500)
    print(f"Found {len(large_files)} large files")
    
    # Filter to ones not already fingerprinted
    to_process = [f for f in large_files if f.name not in existing]
    print(f"Files needing fingerprints: {len(to_process)}")
    
    if not to_process:
        print("All large files already fingerprinted!")
        return
    
    # Show size distribution
    sizes = [f.stat().st_size / (1024*1024) for f in to_process[:10]]
    print(f"\nLargest files to process (MB): {[f'{s:.0f}' for s in sizes]}")
    
    # Process with aggressive memory management
    success = 0
    failed = 0
    failed_files = []
    
    print(f"\nProcessing {len(to_process)} files...")
    
    for i, fpath in enumerate(tqdm(to_process, desc="Fingerprinting large files")):
        try:
            warnings.filterwarnings('ignore')
            # Use memory-mapped loading
            arr = np.load(fpath, mmap_mode='r')
            # Use smaller sample size for very large files
            sample_size = 250_000  # Smaller sample for memory safety
            fp = compute_fingerprint(arr, max_sample_size=sample_size)
            existing[fpath.name] = fp.tolist()
            success += 1
            
            # Aggressive cleanup
            del arr
            gc.collect()
            
            # Checkpoint every 50 files
            if (i + 1) % 50 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(existing, f)
                print(f"\nCheckpoint: {len(existing)} fingerprints saved")
                
        except Exception as e:
            failed += 1
            failed_files.append((fpath.name, str(e)))
            gc.collect()  # Clean up even on failure
    
    # Final save
    with open(checkpoint_file, 'w') as f:
        json.dump(existing, f)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    print(f"Total fingerprints now: {len(existing)}")
    
    if failed_files:
        print(f"\nFailed files:")
        for name, err in failed_files:
            print(f"  {name}: {err[:80]}...")


if __name__ == '__main__':
    main()
