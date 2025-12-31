#!/usr/bin/env python3
"""
RESIDUALS Exhaustive Parameter Exploration

Runs ALL parameter combinations to generate comprehensive prior art.
Outputs detailed documentation of each combination's behavior.

This script is designed to:
1. Test every documented parameter variation
2. Generate timestamped, reproducible results
3. Create machine-readable and human-readable prior art documentation
"""

import numpy as np
from pathlib import Path
import logging
import sys
import json
import hashlib
from datetime import datetime
from itertools import product
from tqdm import tqdm
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exhaustive_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_all_parameter_combinations():
    """
    Generate all parameter combinations for all methods.
    
    Returns dict of:
        method_name: [list of param dicts]
    """
    from src.decomposition.registry import DECOMPOSITION_REGISTRY
    from src.upsampling.registry import UPSAMPLING_REGISTRY
    
    all_combos = {
        'decomposition': {},
        'upsampling': {}
    }
    
    # Decomposition methods
    for name, method in DECOMPOSITION_REGISTRY.items():
        param_ranges = method.param_ranges
        
        if not param_ranges:
            # No parameters, just default
            all_combos['decomposition'][name] = [method.default_params.copy()]
        else:
            # Generate all combinations
            param_names = list(param_ranges.keys())
            param_values = [param_ranges[p] for p in param_names]
            
            combinations = []
            for values in product(*param_values):
                combo = dict(zip(param_names, values))
                combinations.append(combo)
            
            all_combos['decomposition'][name] = combinations
    
    # Upsampling methods
    for name, method in UPSAMPLING_REGISTRY.items():
        param_ranges = method.param_ranges
        
        if not param_ranges:
            all_combos['upsampling'][name] = [method.default_params.copy()]
        else:
            param_names = list(param_ranges.keys())
            param_values = [param_ranges[p] for p in param_names]
            
            combinations = []
            for values in product(*param_values):
                combo = dict(zip(param_names, values))
                combinations.append(combo)
            
            all_combos['upsampling'][name] = combinations
    
    return all_combos


def compute_result_hash(residual: np.ndarray) -> str:
    """Compute deterministic hash of result for verification."""
    return hashlib.sha256(residual.tobytes()).hexdigest()[:16]


def compute_statistics(arr: np.ndarray) -> dict:
    """Compute comprehensive statistics for documentation."""
    return {
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        'skewness': float(compute_skewness(arr)),
        'kurtosis': float(compute_kurtosis(arr)),
        'energy': float(np.sum(arr ** 2)),
        'entropy': float(compute_entropy(arr)),
        'zero_crossings': int(count_zero_crossings(arr)),
    }


def compute_skewness(arr):
    """Compute skewness of array."""
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0
    return np.mean(((arr - mean) / std) ** 3)


def compute_kurtosis(arr):
    """Compute excess kurtosis of array."""
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0
    return np.mean(((arr - mean) / std) ** 4) - 3


def compute_entropy(arr):
    """Compute approximate entropy of array."""
    # Discretize to histogram
    hist, _ = np.histogram(arr.flatten(), bins=256)
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zeros
    return -np.sum(hist * np.log2(hist))


def count_zero_crossings(arr):
    """Count zero crossings in array."""
    signs = np.sign(arr)
    # Count transitions
    return np.sum(np.abs(np.diff(signs.flatten())) > 0)


def run_exhaustive_experiment(
    dem_path: str,
    output_dir: str = 'results/exhaustive',
    max_decomp_combos: int = None,
    max_upsamp_combos: int = None,
    skip_existing: bool = True
):
    """
    Run exhaustive parameter exploration.
    
    Args:
        dem_path: Path to input DEM
        output_dir: Output directory
        max_decomp_combos: Limit decomposition combinations (for testing)
        max_upsamp_combos: Limit upsampling combinations (for testing)
        skip_existing: Skip if output already exists
    """
    from src.decomposition import get_decomposition
    from src.upsampling import get_upsampling
    from src.utils import load_dem
    
    start_time = datetime.now()
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    results_dir = output_dir / 'results'
    docs_dir = output_dir / 'documentation'
    results_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)
    
    # Load DEM
    logger.info(f"Loading DEM: {dem_path}")
    dem = load_dem(dem_path)
    logger.info(f"DEM shape: {dem.shape}, range: [{dem.min():.1f}, {dem.max():.1f}]")
    
    # Get all parameter combinations
    all_combos = get_all_parameter_combinations()
    
    decomp_combos = all_combos['decomposition']
    upsamp_combos = all_combos['upsampling']
    
    # Count total
    total_decomp = sum(len(v) for v in decomp_combos.values())
    total_upsamp = sum(len(v) for v in upsamp_combos.values())
    total_combinations = total_decomp * total_upsamp
    
    logger.info(f"Decomposition methods: {len(decomp_combos)}")
    logger.info(f"Decomposition parameter combinations: {total_decomp}")
    logger.info(f"Upsampling methods: {len(upsamp_combos)}")
    logger.info(f"Upsampling parameter combinations: {total_upsamp}")
    logger.info(f"Total combinations: {total_combinations}")
    
    # Master documentation
    master_doc = {
        'run_timestamp': timestamp,
        'dem_path': str(dem_path),
        'dem_shape': dem.shape,
        'dem_stats': compute_statistics(dem),
        'total_combinations': total_combinations,
        'decomposition_methods': {},
        'upsampling_methods': {},
        'results': []
    }
    
    # Document all methods first (even before running)
    logger.info("\n" + "=" * 60)
    logger.info("DOCUMENTING ALL METHOD PARAMETER SPACES")
    logger.info("=" * 60)
    
    from src.decomposition.registry import DECOMPOSITION_REGISTRY
    from src.upsampling.registry import UPSAMPLING_REGISTRY
    
    for name, method in DECOMPOSITION_REGISTRY.items():
        master_doc['decomposition_methods'][name] = {
            'category': method.category,
            'preserves': method.preserves,
            'destroys': method.destroys,
            'default_params': method.default_params,
            'param_ranges': method.param_ranges,
            'total_combinations': len(decomp_combos.get(name, []))
        }
        logger.info(f"Decomposition '{name}': {len(decomp_combos.get(name, []))} combinations")
    
    for name, method in UPSAMPLING_REGISTRY.items():
        master_doc['upsampling_methods'][name] = {
            'category': method.category,
            'preserves': method.preserves,
            'introduces': getattr(method, 'introduces', ''),
            'default_params': method.default_params,
            'param_ranges': method.param_ranges,
            'total_combinations': len(upsamp_combos.get(name, []))
        }
        logger.info(f"Upsampling '{name}': {len(upsamp_combos.get(name, []))} combinations")
    
    # Save method documentation (prior art for parameter spaces)
    method_doc_path = docs_dir / f'method_parameters_{timestamp}.json'
    with open(method_doc_path, 'w') as f:
        json.dump(master_doc, f, indent=2)
    logger.info(f"Saved method documentation: {method_doc_path}")
    
    # Run all combinations
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING EXHAUSTIVE PARAMETER EXPLORATION")
    logger.info("=" * 60)
    
    completed = 0
    failed = 0
    skipped = 0
    
    # Create progress bar
    pbar = tqdm(total=total_combinations, desc="Processing")
    
    for decomp_name, decomp_param_list in decomp_combos.items():
        if max_decomp_combos:
            decomp_param_list = decomp_param_list[:max_decomp_combos]
        
        decomp_method = get_decomposition(decomp_name)
        decomp_func = decomp_method.func
        
        for decomp_params in decomp_param_list:
            # Create unique ID for this decomposition configuration
            decomp_id = f"{decomp_name}_" + "_".join(
                f"{k}{v}" for k, v in sorted(decomp_params.items())
            )
            
            # Run decomposition
            try:
                trend, residual = decomp_func(dem, **decomp_params)
            except Exception as e:
                logger.warning(f"Decomposition failed: {decomp_id}: {e}")
                failed += len(upsamp_combos) * len(list(upsamp_combos.values())[0]) if upsamp_combos else 1
                pbar.update(sum(len(v) for v in upsamp_combos.values()))
                continue
            
            for upsamp_name, upsamp_param_list in upsamp_combos.items():
                if max_upsamp_combos:
                    upsamp_param_list = upsamp_param_list[:max_upsamp_combos]
                
                upsamp_method = get_upsampling(upsamp_name)
                upsamp_func = upsamp_method.func
                
                for upsamp_params in upsamp_param_list:
                    # Create unique ID for this full configuration
                    upsamp_id = f"{upsamp_name}_" + "_".join(
                        f"{k}{v}" for k, v in sorted(upsamp_params.items())
                    )
                    combo_id = f"{decomp_id}___{upsamp_id}"
                    
                    # Check if already exists
                    result_path = results_dir / f"{combo_id}.npy"
                    if skip_existing and result_path.exists():
                        skipped += 1
                        pbar.update(1)
                        continue
                    
                    # Run upsampling on residual
                    try:
                        residual_up = upsamp_func(residual, **upsamp_params)
                        
                        # Compute statistics
                        stats = compute_statistics(residual_up)
                        result_hash = compute_result_hash(residual_up)
                        
                        # Save result
                        np.save(result_path, residual_up)
                        
                        # Document
                        result_doc = {
                            'combo_id': combo_id,
                            'decomposition': {
                                'name': decomp_name,
                                'params': decomp_params
                            },
                            'upsampling': {
                                'name': upsamp_name,
                                'params': upsamp_params
                            },
                            'output_shape': list(residual_up.shape),
                            'statistics': stats,
                            'hash': result_hash,
                            'result_file': str(result_path.name)
                        }
                        
                        master_doc['results'].append(result_doc)
                        completed += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed: {combo_id}: {e}")
                        failed += 1
                    
                    pbar.update(1)
    
    pbar.close()
    
    # Save complete documentation
    final_doc_path = docs_dir / f'exhaustive_results_{timestamp}.json'
    with open(final_doc_path, 'w') as f:
        json.dump(master_doc, f, indent=2)
    
    # Generate human-readable prior art document
    prior_art_path = docs_dir / f'PRIOR_ART_{timestamp}.md'
    generate_prior_art_document(master_doc, prior_art_path)
    
    # Summary
    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("EXHAUSTIVE EXPLORATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (existing): {skipped}")
    logger.info(f"Time elapsed: {elapsed}")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Documentation saved to: {docs_dir}")
    logger.info(f"Prior art document: {prior_art_path}")
    
    return master_doc


def generate_prior_art_document(doc: dict, output_path: Path):
    """
    Generate human-readable prior art documentation.
    
    This document is designed to be citable and serve as evidence
    of prior disclosure for all tested parameter combinations.
    """
    
    lines = [
        "# RESIDUALS: Exhaustive Parameter Exploration - Prior Art Documentation",
        "",
        f"**Generated**: {doc['run_timestamp']}",
        "",
        "**Purpose**: This document establishes prior art for all tested combinations ",
        "of signal decomposition and upsampling methods applied to Digital Elevation ",
        "Model (DEM) super-resolution for feature detection.",
        "",
        "---",
        "",
        "## 1. Method Parameter Spaces",
        "",
        "### 1.1 Decomposition Methods",
        "",
    ]
    
    for name, info in doc['decomposition_methods'].items():
        lines.append(f"#### {name}")
        lines.append(f"- **Category**: {info['category']}")
        lines.append(f"- **Preserves**: {info['preserves']}")
        lines.append(f"- **Destroys**: {info['destroys']}")
        lines.append(f"- **Default Parameters**: `{info['default_params']}`")
        lines.append(f"- **Parameter Ranges**:")
        for param, values in info['param_ranges'].items():
            lines.append(f"  - `{param}`: {values}")
        lines.append(f"- **Total Combinations**: {info['total_combinations']}")
        lines.append("")
    
    lines.append("### 1.2 Upsampling Methods")
    lines.append("")
    
    for name, info in doc['upsampling_methods'].items():
        lines.append(f"#### {name}")
        lines.append(f"- **Category**: {info['category']}")
        lines.append(f"- **Preserves**: {info['preserves']}")
        lines.append(f"- **Introduces**: {info['introduces']}")
        lines.append(f"- **Default Parameters**: `{info['default_params']}`")
        lines.append(f"- **Parameter Ranges**:")
        for param, values in info['param_ranges'].items():
            lines.append(f"  - `{param}`: {values}")
        lines.append(f"- **Total Combinations**: {info['total_combinations']}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## 2. Tested Combinations Summary")
    lines.append("")
    lines.append(f"**Total Combinations Tested**: {len(doc['results'])}")
    lines.append("")
    
    # Group by decomposition method
    by_decomp = {}
    for result in doc['results']:
        decomp_name = result['decomposition']['name']
        if decomp_name not in by_decomp:
            by_decomp[decomp_name] = []
        by_decomp[decomp_name].append(result)
    
    lines.append("### Combinations by Decomposition Method")
    lines.append("")
    lines.append("| Decomposition | Combinations Tested |")
    lines.append("|---------------|---------------------|")
    for decomp_name, results in sorted(by_decomp.items()):
        lines.append(f"| {decomp_name} | {len(results)} |")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## 3. Detailed Results")
    lines.append("")
    lines.append("Each combination produces a unique residual image capturing ")
    lines.append("terrain features at specific scales and with specific characteristics.")
    lines.append("")
    
    # Sample of results (not all, to keep document manageable)
    lines.append("### Sample Results (first 50)")
    lines.append("")
    lines.append("| Combo ID | Decomp Params | Upsamp Params | Mean | Std | Hash |")
    lines.append("|----------|---------------|---------------|------|-----|------|")
    
    for result in doc['results'][:50]:
        decomp_params = str(result['decomposition']['params'])
        upsamp_params = str(result['upsampling']['params'])
        mean = result['statistics']['mean']
        std = result['statistics']['std']
        hash_val = result['hash']
        
        # Truncate for table
        decomp_params = decomp_params[:30] + "..." if len(decomp_params) > 30 else decomp_params
        upsamp_params = upsamp_params[:30] + "..." if len(upsamp_params) > 30 else upsamp_params
        
        lines.append(f"| {result['combo_id'][:40]}... | {decomp_params} | {upsamp_params} | {mean:.4f} | {std:.4f} | {hash_val} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Reproducibility")
    lines.append("")
    lines.append("All results can be reproduced by running:")
    lines.append("")
    lines.append("```bash")
    lines.append("python run_exhaustive.py --dem <path_to_dem>")
    lines.append("```")
    lines.append("")
    lines.append("Result hashes (SHA-256) are provided for verification.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. License")
    lines.append("")
    lines.append("This work is released under the Apache License 2.0.")
    lines.append("")
    lines.append("The methods, parameter combinations, and results documented herein ")
    lines.append("constitute prior art and are disclosed publicly to prevent exclusive ")
    lines.append("claims or patents on these specific applications of signal processing ")
    lines.append("to Digital Elevation Model analysis for feature detection.")
    lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RESIDUALS Exhaustive Parameter Exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script tests ALL parameter combinations to create comprehensive prior art.

Examples:
    # Full exhaustive run (may take hours)
    python run_exhaustive.py --dem data/test_dems/fairfield_sample_1.5ft.npy
    
    # Limited run for testing
    python run_exhaustive.py --dem data/test_dems/fairfield_sample_1.5ft.npy --max-decomp 2 --max-upsamp 2
    
    # Resume interrupted run (skip existing)
    python run_exhaustive.py --dem data/test_dems/fairfield_sample_1.5ft.npy --skip-existing
        """
    )
    
    parser.add_argument('--dem', type=str, 
                       default='data/test_dems/fairfield_sample_1.5ft.npy',
                       help='Path to input DEM (.npy)')
    parser.add_argument('--output', type=str, default='results/exhaustive',
                       help='Output directory')
    parser.add_argument('--max-decomp', type=int, default=None,
                       help='Max decomposition param combos per method (for testing)')
    parser.add_argument('--max-upsamp', type=int, default=None,
                       help='Max upsampling param combos per method (for testing)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip combinations with existing results')
    
    args = parser.parse_args()
    
    # Check if DEM exists
    dem_path = Path(args.dem)
    if not dem_path.exists():
        logger.error(f"DEM not found: {dem_path}")
        logger.error("Run 'python generate_test_dem.py' first to create test DEM")
        sys.exit(1)
    
    run_exhaustive_experiment(
        dem_path=str(dem_path),
        output_dir=args.output,
        max_decomp_combos=args.max_decomp,
        max_upsamp_combos=args.max_upsamp,
        skip_existing=args.skip_existing
    )


if __name__ == '__main__':
    main()

