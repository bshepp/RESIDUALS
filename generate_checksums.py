#!/usr/bin/env python3
"""
Generate SHA256 checksums for all exhaustive run results.
Supports resuming - skips files already checksummed.
"""

import hashlib
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse


def load_existing_checksums(manifest_path: Path) -> dict:
    """Load already-computed checksums from manifest."""
    existing = {}
    if manifest_path.exists():
        with open(manifest_path, 'r') as fp:
            for line in fp:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('  ', 1)
                if len(parts) == 2:
                    existing[parts[1]] = parts[0]
    return existing


def compute_checksum(filepath: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as fp:
        # Read in chunks for large files
        for chunk in iter(lambda: fp.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Generate checksums for exhaustive results')
    parser.add_argument('--source', type=str, default=r'D:\DIVERGE_exhaustive\results',
                        help='Source directory with .npy files')
    parser.add_argument('--output', type=str, default=r'F:\science-projects\DIVERGE\results\CHECKSUMS.txt',
                        help='Output manifest file')
    parser.add_argument('--force', action='store_true',
                        help='Recompute all checksums (ignore existing)')
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    manifest_path = Path(args.output)
    
    # Ensure output directory exists
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all .npy files
    files = sorted(source_dir.glob('*.npy'))
    print(f"Found {len(files)} .npy files in {source_dir}")
    
    # Load existing checksums if resuming
    if args.force:
        existing = {}
        print("Force mode: recomputing all checksums")
    else:
        existing = load_existing_checksums(manifest_path)
        print(f"Loaded {len(existing)} existing checksums")
    
    # Compute missing checksums
    checksums = dict(existing)  # Start with existing
    new_count = 0
    skipped_count = 0
    
    for f in tqdm(files, desc="Computing checksums"):
        if f.name in checksums:
            skipped_count += 1
            continue
        checksums[f.name] = compute_checksum(f)
        new_count += 1
        
        # Write incrementally every 100 files (in case of interruption)
        if new_count % 100 == 0:
            write_manifest(manifest_path, checksums, source_dir)
    
    # Final write
    write_manifest(manifest_path, checksums, source_dir)
    
    print(f"\nComplete!")
    print(f"  New checksums computed: {new_count}")
    print(f"  Skipped (already done): {skipped_count}")
    print(f"  Total in manifest: {len(checksums)}")
    print(f"  Saved to: {manifest_path}")


def write_manifest(manifest_path: Path, checksums: dict, source_dir: Path):
    """Write checksums to manifest file."""
    with open(manifest_path, 'w') as fp:
        fp.write(f'# RESIDUALS Exhaustive Run - SHA256 Checksums\n')
        fp.write(f'# Generated: {datetime.now().isoformat()}\n')
        fp.write(f'# Total files: {len(checksums)}\n')
        fp.write(f'# Source: {source_dir}\n')
        fp.write(f'#\n')
        fp.write(f'# Format: SHA256_HASH  FILENAME\n')
        fp.write(f'# Verify with: sha256sum -c CHECKSUMS.txt\n')
        fp.write(f'#\n')
        for name in sorted(checksums.keys()):
            fp.write(f'{checksums[name]}  {name}\n')


if __name__ == '__main__':
    main()

