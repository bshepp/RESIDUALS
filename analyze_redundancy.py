#!/usr/bin/env python3
"""
RESIDUALS Redundancy Analyzer

Analyzes exhaustive run results to find redundant or near-identical 
method combinations. Helps prune the method space to truly distinct approaches.

NOTE: This script is READ-ONLY on the results dataset. It only writes 
analysis outputs to the specified output directory.
"""

import numpy as np
from pathlib import Path
import argparse
import json
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import warnings


def parse_checksums(checksum_file: Path) -> dict:
    """Parse CHECKSUMS.txt into {filename: hash} dict."""
    checksums = {}
    with open(checksum_file, 'r') as fp:
        for line in fp:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('  ', 1)
            if len(parts) == 2:
                checksums[parts[1]] = parts[0]
    return checksums


def find_exact_duplicates(checksums: dict) -> dict:
    """Find files with identical hashes.
    
    Returns: {hash: [list of filenames]} for hashes with 2+ files
    """
    hash_to_files = defaultdict(list)
    for filename, file_hash in checksums.items():
        hash_to_files[file_hash].append(filename)
    
    # Keep only duplicates
    return {h: files for h, files in hash_to_files.items() if len(files) > 1}


def parse_filename(filename: str) -> dict:
    """Parse method info from filename.
    
    Format: {decomposition}___{upsampling}.npy
    """
    name = filename.replace('.npy', '')
    parts = name.split('___')
    if len(parts) == 2:
        return {'decomposition': parts[0], 'upsampling': parts[1], 'full': name}
    return {'decomposition': name, 'upsampling': 'unknown', 'full': name}


def compute_fingerprint(arr: np.ndarray) -> np.ndarray:
    """Compute lightweight statistical fingerprint of array.
    
    Returns ~20-dimensional feature vector.
    """
    flat = arr.flatten()
    
    # Basic statistics
    stats = [
        np.mean(flat),
        np.std(flat),
        np.min(flat),
        np.max(flat),
        np.median(flat),
        np.percentile(flat, 25),
        np.percentile(flat, 75),
    ]
    
    # Histogram (10 bins, normalized)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist, _ = np.histogram(flat, bins=10, density=True)
        # Handle edge case of constant array
        if np.any(np.isnan(hist)):
            hist = np.zeros(10)
    
    # Gradient magnitude (edge density)
    if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
        gy, gx = np.gradient(arr)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_stats = [np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag)]
    else:
        grad_stats = [0, 0, 0]
    
    return np.array(stats + list(hist) + grad_stats)


def compute_all_fingerprints(results_dir: Path, file_list: list = None, 
                              limit: int = None) -> dict:
    """Compute fingerprints for all result files.
    
    Returns: {filename: fingerprint_array}
    """
    if file_list is None:
        file_list = sorted(results_dir.glob('*.npy'))
    else:
        file_list = [results_dir / f for f in file_list]
    
    if limit:
        file_list = file_list[:limit]
    
    fingerprints = {}
    for fpath in tqdm(file_list, desc="Computing fingerprints"):
        try:
            # Use memmap for memory efficiency - READ ONLY
            arr = np.load(fpath, mmap_mode='r')
            fingerprints[fpath.name] = compute_fingerprint(arr)
        except Exception as e:
            print(f"Warning: Could not process {fpath.name}: {e}")
    
    return fingerprints


def fingerprint_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute cosine similarity between fingerprints."""
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(fp1, fp2) / (norm1 * norm2))


def find_near_duplicates_by_fingerprint(fingerprints: dict, 
                                         threshold: float = 0.999) -> list:
    """Find pairs with very similar fingerprints.
    
    Returns list of (file1, file2, similarity) tuples.
    """
    names = list(fingerprints.keys())
    fps = np.array([fingerprints[n] for n in names])
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(fps, axis=1, keepdims=True)
    norms[norms == 0] = 1
    fps_norm = fps / norms
    
    near_dupes = []
    
    # Compute similarity matrix in chunks to save memory
    chunk_size = 1000
    for i in tqdm(range(0, len(names), chunk_size), desc="Finding near-duplicates"):
        chunk_end = min(i + chunk_size, len(names))
        chunk = fps_norm[i:chunk_end]
        
        # Compare chunk against all files after it
        for j in range(i, len(names), chunk_size):
            j_end = min(j + chunk_size, len(names))
            other = fps_norm[j:j_end]
            
            # Compute similarities
            sims = chunk @ other.T
            
            # Find high similarities (excluding self-comparisons)
            for ci in range(chunk.shape[0]):
                for cj in range(other.shape[0]):
                    global_i = i + ci
                    global_j = j + cj
                    if global_i >= global_j:
                        continue
                    if sims[ci, cj] >= threshold:
                        near_dupes.append((
                            names[global_i], 
                            names[global_j], 
                            float(sims[ci, cj])
                        ))
    
    return sorted(near_dupes, key=lambda x: -x[2])


def sample_correlation_analysis(results_dir: Path, pairs: list, 
                                 n_samples: int = 100) -> list:
    """Compute actual array correlation for sampled pairs.
    
    Returns list of (file1, file2, correlation) tuples.
    """
    correlations = []
    sample_pairs = pairs[:n_samples] if len(pairs) > n_samples else pairs
    
    for f1, f2, _ in tqdm(sample_pairs, desc="Computing correlations"):
        try:
            # READ ONLY access
            arr1 = np.load(results_dir / f1, mmap_mode='r').flatten()
            arr2 = np.load(results_dir / f2, mmap_mode='r').flatten()
            
            # Handle size mismatch
            if len(arr1) != len(arr2):
                correlations.append((f1, f2, 0.0, "size_mismatch"))
                continue
            
            corr = np.corrcoef(arr1, arr2)[0, 1]
            correlations.append((f1, f2, float(corr), "ok"))
        except Exception as e:
            correlations.append((f1, f2, 0.0, str(e)))
    
    return correlations


def analyze_parameter_sensitivity(fingerprints: dict) -> dict:
    """Analyze how much parameters affect output.
    
    Groups methods by base name and measures variation.
    """
    # Group by decomposition base method
    decomp_groups = defaultdict(list)
    upsamp_groups = defaultdict(list)
    
    for filename, fp in fingerprints.items():
        info = parse_filename(filename)
        decomp_groups[info['decomposition']].append((filename, fp))
        upsamp_groups[info['upsampling']].append((filename, fp))
    
    # Measure within-group variation for decompositions
    decomp_sensitivity = {}
    for decomp, items in decomp_groups.items():
        if len(items) < 2:
            continue
        fps = np.array([fp for _, fp in items])
        # Coefficient of variation across upsampling methods
        mean_fp = np.mean(fps, axis=0)
        std_fp = np.std(fps, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = np.mean(std_fp / (np.abs(mean_fp) + 1e-10))
        decomp_sensitivity[decomp] = float(cv)
    
    # Measure within-group variation for upsampling
    upsamp_sensitivity = {}
    for upsamp, items in upsamp_groups.items():
        if len(items) < 2:
            continue
        fps = np.array([fp for _, fp in items])
        mean_fp = np.mean(fps, axis=0)
        std_fp = np.std(fps, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = np.mean(std_fp / (np.abs(mean_fp) + 1e-10))
        upsamp_sensitivity[upsamp] = float(cv)
    
    return {
        'decomposition': decomp_sensitivity,
        'upsampling': upsamp_sensitivity
    }


def cluster_methods(fingerprints: dict, n_clusters: int = 20) -> dict:
    """Cluster methods by fingerprint similarity.
    
    Returns cluster assignments and representatives.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Warning: sklearn not available, skipping clustering")
        return {}
    
    names = list(fingerprints.keys())
    fps = np.array([fingerprints[n] for n in names])
    
    # Normalize
    scaler = StandardScaler()
    fps_scaled = scaler.fit_transform(fps)
    
    # Cluster
    n_clusters = min(n_clusters, len(names) // 2)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(fps_scaled)
    
    # Group by cluster
    clusters = defaultdict(list)
    for name, label in zip(names, labels):
        clusters[int(label)].append(name)
    
    # Find representative (closest to centroid) for each cluster
    representatives = {}
    for label, members in clusters.items():
        member_fps = np.array([fingerprints[m] for m in members])
        centroid = np.mean(member_fps, axis=0)
        distances = np.linalg.norm(member_fps - centroid, axis=1)
        rep_idx = np.argmin(distances)
        representatives[label] = members[rep_idx]
    
    return {
        'clusters': dict(clusters),
        'representatives': representatives,
        'n_clusters': n_clusters
    }


def generate_report(exact_dupes: dict, near_dupes: list, correlations: list,
                    sensitivity: dict, clusters: dict, output_path: Path):
    """Generate markdown report."""
    
    with open(output_path, 'w', encoding='utf-8') as fp:
        fp.write("# RESIDUALS Redundancy Analysis\n\n")
        fp.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        
        # Summary
        fp.write("## Summary\n\n")
        total_exact = sum(len(files) for files in exact_dupes.values())
        fp.write(f"- **Exact duplicate groups**: {len(exact_dupes)} ")
        fp.write(f"({total_exact} files)\n")
        fp.write(f"- **Near-duplicate pairs** (fingerprint similarity > 0.999): ")
        fp.write(f"{len(near_dupes)}\n")
        if clusters:
            fp.write(f"- **Method clusters**: {clusters.get('n_clusters', 'N/A')}\n")
        fp.write("\n")
        
        # Exact duplicates
        fp.write("## Exact Duplicate Groups\n\n")
        fp.write("These method combinations produce byte-for-byte identical outputs:\n\n")
        if exact_dupes:
            for i, (hash_val, files) in enumerate(list(exact_dupes.items())[:20]):
                fp.write(f"### Group {i+1} (hash: `{hash_val[:12]}...`)\n\n")
                for f in files[:10]:
                    info = parse_filename(f)
                    fp.write(f"- `{info['decomposition']}` + `{info['upsampling']}`\n")
                if len(files) > 10:
                    fp.write(f"- ... and {len(files) - 10} more\n")
                fp.write("\n")
            if len(exact_dupes) > 20:
                fp.write(f"*... and {len(exact_dupes) - 20} more groups*\n\n")
        else:
            fp.write("*No exact duplicates found.*\n\n")
        
        # Near duplicates
        fp.write("## Near-Duplicate Pairs\n\n")
        fp.write("These pairs have extremely similar statistical fingerprints:\n\n")
        if near_dupes:
            fp.write("| File 1 | File 2 | Similarity |\n")
            fp.write("|--------|--------|------------|\n")
            for f1, f2, sim in near_dupes[:30]:
                fp.write(f"| `{f1[:40]}...` | `{f2[:40]}...` | {sim:.4f} |\n")
            if len(near_dupes) > 30:
                fp.write(f"\n*... and {len(near_dupes) - 30} more pairs*\n")
        else:
            fp.write("*No near-duplicates found at threshold.*\n")
        fp.write("\n")
        
        # Correlation verification
        if correlations:
            fp.write("## Correlation Verification\n\n")
            fp.write("Actual array correlations for sampled near-duplicate pairs:\n\n")
            high_corr = [(f1, f2, c) for f1, f2, c, s in correlations if c > 0.99]
            fp.write(f"- **Pairs with r > 0.99**: {len(high_corr)}\n")
            med_corr = [(f1, f2, c) for f1, f2, c, s in correlations if 0.95 < c <= 0.99]
            fp.write(f"- **Pairs with 0.95 < r <= 0.99**: {len(med_corr)}\n\n")
        
        # Sensitivity
        fp.write("## Parameter Sensitivity\n\n")
        fp.write("Methods with LOW sensitivity produce similar outputs regardless of ")
        fp.write("other parameters (candidates for pruning).\n\n")
        
        if sensitivity.get('upsampling'):
            fp.write("### Upsampling Methods (by variation)\n\n")
            sorted_upsamp = sorted(sensitivity['upsampling'].items(), key=lambda x: x[1])
            fp.write("| Method | Variation |\n")
            fp.write("|--------|----------|\n")
            for method, var in sorted_upsamp[:15]:
                fp.write(f"| `{method}` | {var:.4f} |\n")
            fp.write("\n")
        
        # Clusters
        if clusters and clusters.get('clusters'):
            fp.write("## Method Clusters\n\n")
            fp.write("Methods grouped by fingerprint similarity:\n\n")
            for label, members in list(clusters['clusters'].items())[:10]:
                rep = clusters['representatives'].get(label, members[0])
                fp.write(f"### Cluster {label} ({len(members)} methods)\n")
                fp.write(f"- **Representative**: `{rep}`\n")
                fp.write(f"- **Members**: {', '.join(f'`{m[:30]}...`' for m in members[:5])}")
                if len(members) > 5:
                    fp.write(f" + {len(members) - 5} more")
                fp.write("\n\n")
        
        # Recommendations
        fp.write("## Recommendations\n\n")
        if exact_dupes:
            fp.write(f"1. **{total_exact - len(exact_dupes)} files can be pruned** ")
            fp.write("(exact duplicates - keep one per group)\n")
        if near_dupes:
            fp.write(f"2. **Review {len(near_dupes)} near-duplicate pairs** ")
            fp.write("for potential consolidation\n")
        if clusters and clusters.get('representatives'):
            fp.write(f"3. **{len(clusters['representatives'])} representative methods** ")
            fp.write("may capture most variation\n")
        
    print(f"Report saved to: {output_path}")


def save_json_results(exact_dupes: dict, near_dupes: list, correlations: list,
                      sensitivity: dict, clusters: dict, output_path: Path):
    """Save detailed results as JSON for further analysis."""
    
    results = {
        'generated': datetime.now().isoformat(),
        'exact_duplicates': {h: files for h, files in exact_dupes.items()},
        'near_duplicates': [
            {'file1': f1, 'file2': f2, 'similarity': sim}
            for f1, f2, sim in near_dupes
        ],
        'correlations': [
            {'file1': f1, 'file2': f2, 'correlation': c, 'status': s}
            for f1, f2, c, s in correlations
        ],
        'sensitivity': sensitivity,
        'clusters': {
            'assignments': clusters.get('clusters', {}),
            'representatives': clusters.get('representatives', {})
        } if clusters else {}
    }
    
    with open(output_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    
    print(f"JSON results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze RESIDUALS exhaustive results for redundancy'
    )
    parser.add_argument('--checksums', type=str, 
                        default=r'F:\science-projects\DIVERGE\results\CHECKSUMS.txt',
                        help='Path to checksums file')
    parser.add_argument('--results', type=str,
                        default=r'D:\DIVERGE_exhaustive\results',
                        help='Path to results directory (READ ONLY)')
    parser.add_argument('--output', type=str,
                        default=r'F:\science-projects\DIVERGE\results\REDUNDANCY_REPORT.md',
                        help='Output report path')
    parser.add_argument('--fast', action='store_true',
                        help='Skip correlation sampling (faster)')
    parser.add_argument('--fingerprint-limit', type=int, default=None,
                        help='Limit fingerprint computation (for testing)')
    parser.add_argument('--correlation-samples', type=int, default=100,
                        help='Number of pairs to correlate')
    parser.add_argument('--skip-clustering', action='store_true',
                        help='Skip clustering analysis')
    parser.add_argument('--near-dupe-threshold', type=float, default=0.999,
                        help='Fingerprint similarity threshold for near-duplicates')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("RESIDUALS Redundancy Analyzer")
    print("=" * 60)
    print(f"Results directory: {results_dir} (READ ONLY)")
    print(f"Output: {output_path}")
    print()
    
    # Stage 1: Exact duplicates from checksums
    print("Stage 1: Analyzing checksums for exact duplicates...")
    checksums_path = Path(args.checksums)
    if checksums_path.exists():
        checksums = parse_checksums(checksums_path)
        print(f"  Loaded {len(checksums)} checksums")
        exact_dupes = find_exact_duplicates(checksums)
        print(f"  Found {len(exact_dupes)} duplicate groups")
    else:
        print(f"  Warning: Checksums file not found, skipping exact duplicate detection")
        checksums = {}
        exact_dupes = {}
    
    # Stage 2: Statistical fingerprints
    print("\nStage 2: Computing statistical fingerprints...")
    file_list = list(checksums.keys()) if checksums else None
    fingerprints = compute_all_fingerprints(
        results_dir, 
        file_list=file_list,
        limit=args.fingerprint_limit
    )
    print(f"  Computed {len(fingerprints)} fingerprints")
    
    # Stage 3: Find near-duplicates by fingerprint
    print("\nStage 3: Finding near-duplicates by fingerprint similarity...")
    near_dupes = find_near_duplicates_by_fingerprint(
        fingerprints, 
        threshold=args.near_dupe_threshold
    )
    print(f"  Found {len(near_dupes)} near-duplicate pairs")
    
    # Stage 4: Correlation sampling (optional)
    correlations = []
    if not args.fast and near_dupes:
        print("\nStage 4: Sampling correlations for verification...")
        correlations = sample_correlation_analysis(
            results_dir, 
            near_dupes, 
            n_samples=args.correlation_samples
        )
        high_corr = sum(1 for _, _, c, _ in correlations if c > 0.99)
        print(f"  {high_corr}/{len(correlations)} pairs have r > 0.99")
    
    # Stage 5: Parameter sensitivity
    print("\nStage 5: Analyzing parameter sensitivity...")
    sensitivity = analyze_parameter_sensitivity(fingerprints)
    print(f"  Analyzed {len(sensitivity.get('decomposition', {}))} decomposition methods")
    print(f"  Analyzed {len(sensitivity.get('upsampling', {}))} upsampling methods")
    
    # Stage 6: Clustering (optional)
    clusters = {}
    if not args.skip_clustering and len(fingerprints) > 20:
        print("\nStage 6: Clustering methods...")
        clusters = cluster_methods(fingerprints)
        if clusters:
            print(f"  Created {clusters.get('n_clusters', 0)} clusters")
    
    # Generate outputs
    print("\nGenerating reports...")
    generate_report(exact_dupes, near_dupes, correlations, 
                    sensitivity, clusters, output_path)
    
    json_path = output_path.with_suffix('.json')
    save_json_results(exact_dupes, near_dupes, correlations,
                      sensitivity, clusters, json_path)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

