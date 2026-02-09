#!/usr/bin/env python3
"""
RESIDUALS: Best-Of Pipeline

Runs 20 distinct decomposition methods (one representative per cluster from
the exhaustive 39,731-combination analysis) paired with 3 upsamplers, giving
60 total combinations that cover the full diversity of the method space.

This is the recommended entry point for new DEMs.  For the full parameter
sweep, use run_exhaustive.py instead.

Usage:
    python run_bestof.py --dem data/test_dems/fairfield_sample_1.5ft.npy
    python run_bestof.py --dem data/test_dems/licking_hopewell_2.5ft.npy --output results/hopewell

Cluster source: PRIOR_ART.md, Section 2 (Top 20 Method Clusters)
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
        logging.FileHandler('bestof_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==========================================================================
# The 20 cluster representatives
# ==========================================================================
# Each entry: (method_name, params_dict, cluster_description)
# Selected for maximum interpretive diversity from the redundancy analysis.

BESTOF_DECOMPOSITIONS = [
    # --- Classical / Gaussian family ---
    ("gaussian",              {"sigma": 5},
     "Fine local features (small-sigma Gaussian)"),

    ("gaussian",              {"sigma": 50},
     "Regional trend removal (large-sigma Gaussian)"),

    # --- Edge-preserving ---
    ("bilateral",             {"d": 9, "sigma_color": 75, "sigma_space": 75},
     "Edge-preserving smoothing"),

    ("anisotropic_diffusion", {"iterations": 10, "kappa": 50, "gamma": 0.1},
     "Edge-preserving via iterative diffusion"),

    ("guided",                {"radius": 8, "eps": 0.01},
     "Self-guided edge-preserving filter"),

    # --- Wavelet family ---
    ("wavelet_dwt",           {"wavelet": "db4", "level": 2},
     "Wavelet high-frequency detail (shallow)"),

    ("wavelet_dwt",           {"wavelet": "db4", "level": 4},
     "Wavelet low-frequency structure (deep)"),

    ("wavelet_biorthogonal",  {"wavelet": "bior3.5", "level": 3},
     "Linear-phase wavelet multi-scale"),

    # --- Morphological family ---
    ("morphological",         {"operation": "opening", "size": 15},
     "Circular structuring element (disk)"),

    ("morphological_square",  {"operation": "opening", "size": 10},
     "Axis-aligned square element"),

    ("morphological_ellipse", {"operation": "opening", "width": 20, "height": 10},
     "Elongated elliptical element"),

    ("morphological_rect",    {"operation": "opening", "width": 20, "height": 5},
     "Directional rectangular element (linear features)"),

    ("morphological_gradient", {"size": 5, "shape": "disk"},
     "Morphological edge detection"),

    # --- Top-hat family ---
    ("tophat",                {"size": 20, "mode": "white"},
     "Small bright features (white top-hat)"),

    ("tophat_combined",       {"size": 20},
     "Both bright and dark small features"),

    # --- Rolling ball ---
    ("rolling_ball",          {"radius": 25},
     "Fine features (small ball)"),

    # --- Multi-scale / bandpass ---
    ("dog",                   {"sigma_low": 2, "sigma_high": 10},
     "Bandpass: features at intermediate scales"),

    ("dog_multiscale",        {"sigma_ratio": 1.6, "n_scales": 4, "base_sigma": 1.0},
     "Multi-scale blob detection"),

    ("log",                   {"sigma": 5},
     "Laplacian of Gaussian blob detection"),

    # --- Polynomial trend removal ---
    ("polynomial",            {"degree": 2},
     "Quadratic regional trend removal"),
]

BESTOF_UPSAMPLINGS = [
    ("bicubic",    {"scale": 2}, "Smooth baseline"),
    ("lanczos",    {"scale": 2}, "Sharp edges, good for linear features"),
    ("fft_zeropad", {"scale": 2}, "Frequency-domain, band-limited"),
]


# ==========================================================================
# Pipeline
# ==========================================================================

def run_bestof(
    dem_path: str,
    output_dir: str = 'results/bestof',
    scale: int = 2,
):
    """
    Run the best-of decomposition pipeline.

    Args:
        dem_path: Path to input DEM
        output_dir: Output directory
        scale: Upsampling scale factor (overrides BESTOF_UPSAMPLINGS default)
    """
    from src.decomposition.registry import run_decomposition, get_decomposition
    from src.upsampling.registry import run_upsampling
    from src.analysis import analyze_features, rank_for_features, generate_analysis_report
    from src.utils import load_dem, visualize_results, visualize_top_results

    start_time = datetime.now()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combo_dir = output_dir / 'combinations'
    combo_dir.mkdir(exist_ok=True)

    # --- Load DEM ---
    logger.info(f"Loading DEM: {dem_path}")
    dem = load_dem(dem_path)
    logger.info(f"DEM shape: {dem.shape}, range: [{dem.min():.1f}, {dem.max():.1f}]")

    n_decomp = len(BESTOF_DECOMPOSITIONS)
    n_upsamp = len(BESTOF_UPSAMPLINGS)
    total = n_decomp * n_upsamp
    logger.info(f"Best-of pipeline: {n_decomp} decompositions x {n_upsamp} upsamplings = {total} combinations")

    # --- Phase 1: Decompose + Upsample ---
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Running best-of decomposition x upsampling")
    logger.info("=" * 60)

    results = {}
    current = 0

    for decomp_name, decomp_params, desc in BESTOF_DECOMPOSITIONS:
        # Build a unique label for this decomposition configuration
        param_tag = "_".join(f"{k}{v}" for k, v in sorted(decomp_params.items()))
        decomp_label = f"{decomp_name}_{param_tag}" if param_tag else decomp_name

        logger.info(f"\n[{decomp_label}] {desc}")

        try:
            trend, residual = run_decomposition(decomp_name, dem, decomp_params)
            logger.info(f"  Residual: std={residual.std():.4f}, range=[{residual.min():.3f}, {residual.max():.3f}]")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            current += n_upsamp
            continue

        results[decomp_label] = {}

        for upsamp_name, upsamp_params, upsamp_desc in BESTOF_UPSAMPLINGS:
            current += 1
            logger.info(f"  [{current}/{total}] {upsamp_name} ({upsamp_desc})")

            try:
                trend_up = run_upsampling(upsamp_name, trend, scale=scale)
                residual_up = run_upsampling(upsamp_name, residual, scale=scale)

                results[decomp_label][upsamp_name] = {
                    'trend': trend_up,
                    'residual': residual_up,
                }

                np.save(combo_dir / f"{decomp_label}_{upsamp_name}_residual.npy", residual_up)

            except Exception as e:
                logger.error(f"    FAILED: {e}")

    # --- Phase 2: Pairwise differentials ---
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Computing pairwise differentials")
    logger.info("=" * 60)

    all_combos = []
    for decomp_label, upsamp_results in results.items():
        for upsamp_name in upsamp_results:
            all_combos.append((decomp_label, upsamp_name))

    differentials = {}
    diff_dir = output_dir / 'differentials'
    diff_dir.mkdir(exist_ok=True)

    for i, (d1, u1) in enumerate(tqdm(all_combos, desc="Differentials")):
        for d2, u2 in all_combos[i + 1:]:
            key = f"{d1}_{u1}_vs_{d2}_{u2}"
            r1 = results[d1][u1]['residual']
            r2 = results[d2][u2]['residual']
            min_h = min(r1.shape[0], r2.shape[0])
            min_w = min(r1.shape[1], r2.shape[1])
            diff = r1[:min_h, :min_w] - r2[:min_h, :min_w]
            differentials[key] = diff
            np.save(diff_dir / f"{key}.npy", diff)

    logger.info(f"Computed {len(differentials)} differentials")

    # --- Phase 3: Analyze + rank ---
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Analyzing differentials")
    logger.info("=" * 60)

    analyses = {}
    for key, diff in tqdm(differentials.items(), desc="Analyzing"):
        try:
            analyses[key] = analyze_features(diff, dem)
        except Exception as e:
            logger.warning(f"Analysis failed for {key}: {e}")

    ranked = rank_for_features(differentials, analyses)

    logger.info("\nTop 10 candidates for feature detection:")
    for i, (key, score, analysis) in enumerate(ranked[:10]):
        logger.info(f"  {i+1}. {key}")
        logger.info(f"     Score: {score:.1f}")

    # --- Phase 4: Outputs ---
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Generating outputs")
    logger.info("=" * 60)

    # Save analyses
    with open(output_dir / 'analyses.json', 'w') as f:
        json.dump(analyses, f, indent=2)

    # Save rankings
    rankings = [
        (key, score, {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                      for k, v in analysis.items()})
        for key, score, analysis in ranked
    ]
    with open(output_dir / 'rankings.json', 'w') as f:
        json.dump(rankings, f, indent=2)

    # Generate report
    report = generate_analysis_report(ranked)
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.write(report)

    # Visualize
    viz_dir = output_dir / 'visualizations'
    visualize_results(results, str(viz_dir), original_dem=dem)
    visualize_top_results(differentials, ranked, str(viz_dir / 'top_results'))

    # Summary
    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("BEST-OF PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Combinations: {len(all_combos)} ({n_decomp} decomp x {n_upsamp} upsamp)")
    logger.info(f"Differentials: {len(differentials)}")
    logger.info(f"Time: {elapsed}")
    logger.info(f"Output: {output_dir}")

    # Save a summary of which methods were used
    method_summary = []
    for name, params, desc in BESTOF_DECOMPOSITIONS:
        method_summary.append({"name": name, "params": params, "description": desc})
    with open(output_dir / 'methods_used.json', 'w') as f:
        json.dump({
            "decompositions": method_summary,
            "upsamplings": [{"name": n, "params": p, "description": d}
                            for n, p, d in BESTOF_UPSAMPLINGS],
            "total_combinations": total,
            "scale": scale,
        }, f, indent=2)

    return ranked


# ==========================================================================
# CLI
# ==========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RESIDUALS Best-Of Pipeline: 20 distinct methods x 3 upsamplers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This runs the distilled set of 20 decomposition methods (one per cluster from
the exhaustive 39,731-combination analysis) with 3 upsamplers = 60 combinations.

Examples:
    python run_bestof.py --dem data/test_dems/fairfield_sample_1.5ft.npy
    python run_bestof.py --dem data/test_dems/licking_hopewell_2.5ft.npy --scale 2
        """
    )

    parser.add_argument('--dem', type=str,
                        default='data/test_dems/fairfield_sample_1.5ft.npy',
                        help='Path to input DEM')
    parser.add_argument('--output', type=str, default='results/bestof',
                        help='Output directory')
    parser.add_argument('--scale', type=int, default=2,
                        help='Upsampling scale factor')

    args = parser.parse_args()

    dem_path = Path(args.dem)
    if not dem_path.exists():
        logger.error(f"DEM not found: {dem_path}")
        sys.exit(1)

    run_bestof(
        dem_path=str(dem_path),
        output_dir=args.output,
        scale=args.scale,
    )


if __name__ == '__main__':
    main()
