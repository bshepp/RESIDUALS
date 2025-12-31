#!/usr/bin/env python3
"""
DIVERGE: Main Experiment Runner

Runs the decomposition × upsampling experimental framework.
Computes differentials and ranks results for archaeological feature detection.
"""

import numpy as np
from pathlib import Path
import logging
import sys
import json
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_experiment(
    dem_path: str,
    output_dir: str = 'results',
    scale: int = 2,
    decomposition_methods: list = None,
    upsampling_methods: list = None
):
    """
    Run the full decomposition × upsampling experiment.
    
    Args:
        dem_path: Path to input DEM (.npy file)
        output_dir: Directory for outputs
        scale: Upsampling scale factor
        decomposition_methods: List of decomposition methods to use
        upsampling_methods: List of upsampling methods to use
    """
    from src.decomposition import list_decompositions, run_decomposition
    from src.upsampling import list_upsamplings, run_upsampling
    from src.analysis import (
        compute_all_differentials,
        analyze_features,
        rank_for_archaeology,
        generate_analysis_report
    )
    from src.utils import (
        load_dem,
        visualize_results,
        visualize_top_results
    )
    
    start_time = datetime.now()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load DEM
    logger.info(f"Loading DEM: {dem_path}")
    dem = load_dem(dem_path)
    logger.info(f"DEM shape: {dem.shape}")
    logger.info(f"DEM range: {dem.min():.1f} - {dem.max():.1f}")
    
    # Get available methods
    available_decomp = list_decompositions()
    available_upsamp = list_upsamplings()
    
    logger.info(f"Available decomposition methods: {available_decomp}")
    logger.info(f"Available upsampling methods: {available_upsamp}")
    
    # Use specified methods or all available
    decomp_methods = decomposition_methods or available_decomp
    upsamp_methods = upsampling_methods or available_upsamp
    
    logger.info(f"Using decomposition: {decomp_methods}")
    logger.info(f"Using upsampling: {upsamp_methods}")
    
    total_combos = len(decomp_methods) * len(upsamp_methods)
    logger.info(f"Total combinations: {total_combos}")
    
    # ==========================================================================
    # Phase 1: Run all decomposition × upsampling combinations
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Computing decomposition × upsampling combinations")
    logger.info("="*60)
    
    results = {}
    combo_dir = output_dir / 'combinations'
    combo_dir.mkdir(exist_ok=True)
    
    current = 0
    for decomp_name in decomp_methods:
        logger.info(f"\nDecomposition: {decomp_name}")
        
        try:
            trend, residual = run_decomposition(decomp_name, dem)
            logger.info(f"  Trend range: {trend.min():.1f} - {trend.max():.1f}")
            logger.info(f"  Residual range: {residual.min():.2f} - {residual.max():.2f}")
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            continue
        
        results[decomp_name] = {}
        
        for upsamp_name in upsamp_methods:
            current += 1
            logger.info(f"  [{current}/{total_combos}] Upsampling: {upsamp_name}")
            
            try:
                trend_up = run_upsampling(upsamp_name, trend, scale=scale)
                residual_up = run_upsampling(upsamp_name, residual, scale=scale)
                
                results[decomp_name][upsamp_name] = {
                    'trend': trend_up,
                    'residual': residual_up
                }
                
                # Save intermediate results
                np.save(combo_dir / f"{decomp_name}_{upsamp_name}_residual.npy", residual_up)
                
            except Exception as e:
                logger.error(f"    ERROR: {e}")
                continue
    
    # ==========================================================================
    # Phase 2: Compute differentials
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Computing differentials between all pairs")
    logger.info("="*60)
    
    differentials = {}
    diff_dir = output_dir / 'differentials'
    diff_dir.mkdir(exist_ok=True)
    
    # Get all successful combinations
    all_combos = []
    for decomp_name, upsamp_results in results.items():
        for upsamp_name in upsamp_results.keys():
            all_combos.append((decomp_name, upsamp_name))
    
    n_pairs = len(all_combos) * (len(all_combos) - 1) // 2
    logger.info(f"Computing {n_pairs} pairwise differentials...")
    
    pair_count = 0
    for i, (decomp1, upsamp1) in enumerate(tqdm(all_combos, desc="Computing differentials")):
        for decomp2, upsamp2 in all_combos[i + 1:]:
            key = f"{decomp1}_{upsamp1}_vs_{decomp2}_{upsamp2}"
            
            res1 = results[decomp1][upsamp1]['residual']
            res2 = results[decomp2][upsamp2]['residual']
            
            # Ensure same shape
            min_h = min(res1.shape[0], res2.shape[0])
            min_w = min(res1.shape[1], res2.shape[1])
            
            diff = res1[:min_h, :min_w] - res2[:min_h, :min_w]
            differentials[key] = diff
            
            np.save(diff_dir / f"{key}.npy", diff)
            pair_count += 1
    
    logger.info(f"Computed {pair_count} differentials")
    
    # ==========================================================================
    # Phase 3: Analyze and rank
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: Analyzing differentials for feature detection")
    logger.info("="*60)
    
    analyses = {}
    
    for key, diff in tqdm(differentials.items(), desc="Analyzing"):
        try:
            analysis = analyze_features(diff, dem)
            analyses[key] = analysis
        except Exception as e:
            logger.warning(f"Analysis failed for {key}: {e}")
            continue
    
    # Rank for archaeological features
    ranked = rank_for_archaeology(differentials, analyses)
    
    logger.info("\nTop 10 candidates for archaeological feature detection:")
    for i, (key, score, analysis) in enumerate(ranked[:10]):
        logger.info(f"  {i+1}. {key}")
        logger.info(f"     Score: {score:.1f}, Linear: {analysis.get('linear_feature_count', 0)}, "
                   f"Autocorr: {analysis.get('spatial_autocorr', 0):.3f}")
    
    # ==========================================================================
    # Phase 4: Generate outputs
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 4: Generating outputs")
    logger.info("="*60)
    
    # Save analyses
    analyses_path = output_dir / 'analyses.json'
    with open(analyses_path, 'w') as f:
        json.dump(analyses, f, indent=2)
    logger.info(f"Saved analyses: {analyses_path}")
    
    # Save rankings
    rankings = [(key, score, {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                              for k, v in analysis.items()})
                for key, score, analysis in ranked]
    rankings_path = output_dir / 'rankings.json'
    with open(rankings_path, 'w') as f:
        json.dump(rankings, f, indent=2)
    logger.info(f"Saved rankings: {rankings_path}")
    
    # Generate report
    report = generate_analysis_report(ranked)
    report_path = output_dir / 'REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved report: {report_path}")
    
    # Visualize results
    viz_dir = output_dir / 'visualizations'
    visualize_results(results, str(viz_dir), original_dem=dem)
    visualize_top_results(differentials, ranked, str(viz_dir / 'top_results'))
    
    # Summary
    elapsed = datetime.now() - start_time
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*60)
    logger.info(f"Combinations tested: {len(all_combos)}")
    logger.info(f"Differentials computed: {len(differentials)}")
    logger.info(f"Time elapsed: {elapsed}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)
    
    return ranked


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DIVERGE: Decomposition × Upsampling Experimental Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default test DEM
    python run_experiment.py
    
    # Run with custom DEM
    python run_experiment.py --dem data/test_dems/my_dem.npy
    
    # Run with specific methods
    python run_experiment.py --decomp gaussian bilateral --upsamp bicubic lanczos
        """
    )
    
    parser.add_argument('--dem', type=str, 
                       default='data/test_dems/fairfield_sample_1.5ft.npy',
                       help='Path to input DEM (.npy)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--scale', type=int, default=2,
                       help='Upsampling scale factor')
    parser.add_argument('--decomp', nargs='+', default=None,
                       help='Decomposition methods to use')
    parser.add_argument('--upsamp', nargs='+', default=None,
                       help='Upsampling methods to use')
    
    args = parser.parse_args()
    
    # Check if DEM exists
    dem_path = Path(args.dem)
    if not dem_path.exists():
        logger.error(f"DEM not found: {dem_path}")
        logger.error("Run 'python generate_test_dem.py' first to create test DEM")
        sys.exit(1)
    
    run_experiment(
        dem_path=str(dem_path),
        output_dir=args.output,
        scale=args.scale,
        decomposition_methods=args.decomp,
        upsampling_methods=args.upsamp
    )


if __name__ == '__main__':
    main()

