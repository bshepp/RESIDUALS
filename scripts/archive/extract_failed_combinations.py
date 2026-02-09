#!/usr/bin/env python3
"""
Extract failed combination IDs from exhaustive run log.

Usage:
    python extract_failed_combinations.py --log exhaustive_experiment.log --output failed_combinations.txt
"""

import re
import argparse
from pathlib import Path
from collections import Counter


def extract_failures(log_path: Path) -> dict:
    """Extract failures categorized by type."""
    failures = {
        'disk_space': [],
        'disk_write': [],
        'decomposition': [],
        'other': []
    }
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'WARNING - Failed:' in line:
                # Extract the combination name
                match = re.search(r'Failed: ([^:]+):', line)
                if match:
                    combo = match.group(1).strip()
                    
                    if 'No space left on device' in line:
                        failures['disk_space'].append(combo)
                    elif 'requested and 0 written' in line:
                        failures['disk_write'].append(combo)
                    else:
                        failures['other'].append(combo)
            
            elif 'WARNING - Decomposition failed:' in line:
                # Extract method name
                match = re.search(r'Decomposition failed: ([^:]+):', line)
                if match:
                    failures['decomposition'].append(match.group(1).strip())
    
    return failures


def main():
    parser = argparse.ArgumentParser(description='Extract failed combinations from log')
    parser.add_argument('--log', required=True, help='Path to exhaustive_experiment.log')
    parser.add_argument('--output', required=True, help='Output file for failed combinations')
    parser.add_argument('--summary', action='store_true', help='Print summary only')
    
    args = parser.parse_args()
    
    failures = extract_failures(Path(args.log))
    
    # Summary
    print("=" * 60)
    print("FAILURE SUMMARY")
    print("=" * 60)
    print(f"Disk space failures:    {len(failures['disk_space']):,}")
    print(f"Disk write failures:    {len(failures['disk_write']):,}")
    print(f"Decomposition failures: {len(failures['decomposition']):,}")
    print(f"Other failures:         {len(failures['other']):,}")
    print("-" * 60)
    total = sum(len(v) for v in failures.values())
    print(f"TOTAL:                  {total:,}")
    print("=" * 60)
    
    if failures['decomposition']:
        print("\nDecomposition failures (code bugs):")
        for d in failures['decomposition']:
            print(f"  - {d}")
    
    if args.summary:
        return
    
    # Write all failed combinations (excluding decomposition bugs)
    all_combos = failures['disk_space'] + failures['disk_write'] + failures['other']
    
    # Deduplicate
    unique_combos = list(set(all_combos))
    
    with open(args.output, 'w') as f:
        for combo in sorted(unique_combos):
            f.write(combo + '\n')
    
    print(f"\nWrote {len(unique_combos)} unique failed combinations to {args.output}")
    
    # Analyze patterns
    print("\nPattern Analysis:")
    
    # Count by decomposition method
    decomp_methods = Counter()
    upsample_methods = Counter()
    
    for combo in unique_combos:
        if '___' in combo:
            decomp, upsample = combo.rsplit('___', 1)
            decomp_methods[decomp.split('_')[0]] += 1
            upsample_methods[upsample.split('_scale')[0]] += 1
    
    print("\nTop 10 decomposition prefixes:")
    for method, count in decomp_methods.most_common(10):
        print(f"  {method}: {count}")
    
    print("\nTop 10 upsampling methods:")
    for method, count in upsample_methods.most_common(10):
        print(f"  {method}: {count}")


if __name__ == '__main__':
    main()
