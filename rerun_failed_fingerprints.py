#!/usr/bin/env python3
"""
Rerun fingerprinting on files that previously failed due to memory issues.
Uses the optimized fingerprinting with sampling.
"""

import numpy as np
from pathlib import Path
import json
import gc
import sys
from tqdm import tqdm
import warnings

sys.path.insert(0, str(Path(__file__).parent))
from analyze_redundancy import compute_fingerprint


def main():
    results_dir = Path(r'D:\DIVERGE_exhaustive\results')
    
    # Load existing fingerprints from the redundancy report
    existing_fp_file = Path(r'F:\science-projects\DIVERGE\results\fingerprint_checkpoint.json')
    
    if existing_fp_file.exists():
        with open(existing_fp_file, 'r') as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing fingerprints")
    else:
        existing = {}
        print("No existing fingerprints found")
    
    # Get all .npy files
    all_files = list(results_dir.glob('*.npy'))
    print(f"Total files in results: {len(all_files)}")
    
    # Find missing fingerprints
    missing = [f for f in all_files if f.name not in existing]
    print(f"Files needing fingerprints: {len(missing)}")
    
    if not missing:
        print("All files already fingerprinted!")
        return
    
    # Process missing files with the optimized fingerprinter
    success = 0
    failed = 0
    failed_files = []
    
    for i, fpath in enumerate(tqdm(missing, desc="Fingerprinting")):
        try:
            warnings.filterwarnings('ignore')
            arr = np.load(fpath, mmap_mode='r')
            fp = compute_fingerprint(arr, max_sample_size=500_000)
            existing[fpath.name] = fp.tolist()
            success += 1
            
            # Cleanup
            del arr
            if (i + 1) % 50 == 0:
                gc.collect()
            
            # Checkpoint every 100 files
            if (i + 1) % 100 == 0:
                with open(existing_fp_file, 'w') as f:
                    json.dump(existing, f)
                print(f"\nCheckpoint: {len(existing)} fingerprints saved")
                
        except Exception as e:
            failed += 1
            failed_files.append((fpath.name, str(e)))
    
    # Final save
    with open(existing_fp_file, 'w') as f:
        json.dump(existing, f)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    print(f"Total fingerprints: {len(existing)}")
    
    if failed_files:
        print(f"\nFailed files (first 10):")
        for name, err in failed_files[:10]:
            print(f"  {name}: {err[:50]}...")


if __name__ == '__main__':
    main()
