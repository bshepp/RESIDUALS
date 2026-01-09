#!/usr/bin/env python3
"""
Parallel Fingerprint Generator for DIVERGE Exhaustive Results

Uses ProcessPoolExecutor with 2 workers (HDD-optimized) to fingerprint
all output files for redundancy analysis.

Supports checkpointing - can be interrupted and resumed.
"""

import numpy as np
from pathlib import Path
import json
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse


# Configuration
RESULTS_DIR = Path('D:/DIVERGE_exhaustive/results')
CHECKPOINT_FILE = Path('F:/science-projects/DIVERGE/results/fingerprint_checkpoint.json')
NUM_WORKERS = 2  # Optimized for HDD
CHECKPOINT_INTERVAL = 500  # Save every N files


def compute_fingerprint(arr: np.ndarray, max_sample_size: int = 1_000_000) -> np.ndarray:
    """Compute lightweight statistical fingerprint of array.
    
    Returns ~20-dimensional feature vector.
    For large arrays, uses sampling to reduce memory usage.
    """
    total_elements = arr.size
    use_sampling = total_elements > max_sample_size
    
    if use_sampling:
        rng = np.random.default_rng(42)  # Reproducible
        flat_indices = rng.choice(total_elements, size=max_sample_size, replace=False)
        flat_sample = arr.flat[sorted(flat_indices)]
    else:
        flat_sample = arr.flatten()
    
    # Basic statistics
    stats = [
        float(np.mean(flat_sample)),
        float(np.std(flat_sample)),
        float(np.min(flat_sample)),
        float(np.max(flat_sample)),
        float(np.median(flat_sample)),
        float(np.percentile(flat_sample, 25)),
        float(np.percentile(flat_sample, 75)),
    ]
    
    # Histogram (10 bins, normalized)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist, _ = np.histogram(flat_sample, bins=10, density=True)
        if np.any(np.isnan(hist)):
            hist = np.zeros(10)
    
    del flat_sample
    
    # Gradient magnitude (edge density)
    if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
        if use_sampling:
            step = max(1, int(np.sqrt(total_elements / max_sample_size)))
            arr_sub = arr[::step, ::step]
        else:
            arr_sub = arr
        
        try:
            gy, gx = np.gradient(arr_sub.astype(np.float32))
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_stats = [float(np.mean(grad_mag)), float(np.std(grad_mag)), float(np.max(grad_mag))]
            del gy, gx, grad_mag
        except MemoryError:
            grad_stats = [float(np.std(arr_sub)), 0.0, 0.0]
        
        if use_sampling:
            del arr_sub
    else:
        grad_stats = [0.0, 0.0, 0.0]
    
    return np.array(stats + list(hist) + grad_stats)


def process_file(filepath: Path) -> tuple:
    """Process a single file and return (filename, fingerprint).
    
    This function runs in a worker process.
    """
    try:
        arr = np.load(filepath, mmap_mode='r')
        fp = compute_fingerprint(arr)
        del arr
        return (filepath.name, fp.tolist(), None)
    except Exception as e:
        return (filepath.name, None, str(e))


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load existing fingerprints from checkpoint."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint(fingerprints: dict, checkpoint_path: Path):
    """Save fingerprints to checkpoint file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(fingerprints, f)


def main():
    parser = argparse.ArgumentParser(description='Parallel fingerprint generation')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help=f'Number of parallel workers (default: {NUM_WORKERS})')
    parser.add_argument('--source', type=str, default=str(RESULTS_DIR),
                        help='Source directory with .npy files')
    parser.add_argument('--checkpoint', type=str, default=str(CHECKPOINT_FILE),
                        help='Checkpoint file path')
    parser.add_argument('--interval', type=int, default=CHECKPOINT_INTERVAL,
                        help=f'Checkpoint save interval (default: {CHECKPOINT_INTERVAL})')
    args = parser.parse_args()
    
    results_dir = Path(args.source)
    checkpoint_path = Path(args.checkpoint)
    
    print(f"{'='*60}")
    print(f"DIVERGE Parallel Fingerprint Generator")
    print(f"{'='*60}")
    print(f"Source: {results_dir}")
    print(f"Workers: {args.workers}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Load existing fingerprints
    fingerprints = load_checkpoint(checkpoint_path)
    print(f"Loaded {len(fingerprints):,} existing fingerprints from checkpoint")
    
    # Get all files
    all_files = sorted(results_dir.glob('*.npy'))
    print(f"Total files in source: {len(all_files):,}")
    
    # Filter to only unprocessed files
    files_to_process = [f for f in all_files if f.name not in fingerprints]
    print(f"Files needing fingerprints: {len(files_to_process):,}")
    
    if not files_to_process:
        print("\nAll files already fingerprinted!")
        return
    
    print(f"\nProcessing {len(files_to_process):,} files with {args.workers} workers...")
    print()
    
    success = 0
    failed = 0
    processed_since_checkpoint = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, f): f for f in files_to_process}
        
        # Process as they complete
        with tqdm(total=len(files_to_process), desc="Fingerprinting") as pbar:
            for future in as_completed(future_to_file):
                filename, fp, error = future.result()
                
                if error:
                    failed += 1
                    tqdm.write(f"Failed: {filename}: {error}")
                else:
                    fingerprints[filename] = fp
                    success += 1
                
                processed_since_checkpoint += 1
                pbar.update(1)
                
                # Periodic checkpoint
                if processed_since_checkpoint >= args.interval:
                    save_checkpoint(fingerprints, checkpoint_path)
                    tqdm.write(f"Checkpoint: {len(fingerprints):,} fingerprints saved")
                    processed_since_checkpoint = 0
    
    # Final save
    save_checkpoint(fingerprints, checkpoint_path)
    
    print()
    print(f"{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Success: {success:,}")
    print(f"Failed:  {failed}")
    print(f"Total fingerprints: {len(fingerprints):,}")
    print(f"Saved to: {checkpoint_path}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
