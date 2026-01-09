# RESIDUALS: Multi-Method Differential Feature Detection

**A framework for feature detection in Digital Elevation Models using systematic decomposition and differential analysis.**

![Sample Output](results/visualizations/residuals_grid_preview.png)

## What Is This?

RESIDUALS systematically tests combinations of signal decomposition and upsampling methods to identify which combinations best reveal features in elevation data. The core insight:

> *Different method combinations have characteristic behaviors that selectively preserve or eliminate different feature types. By computing differentials between outputs, we create feature-specific extraction filters.*

## The 4-Level Differential Hierarchy

| Level | What It Shows | Column |
|-------|---------------|--------|
| 0 | Ground truth (hillshade) | DEM |
| 1 | Decomposition residuals | bicubic, lanczos, bspline, fft |
| 2 | Residual vs ground truth | Δbic, Δlan, Δbsp, Δfft |
| 3 | Divergence across methods | Div |
| 4 | Meta-divergence (uncertainty of uncertainty) | ΔDiv |

## Quick Start

```bash
# Clone and install
git clone https://github.com/bshepp/RESIDUALS.git
cd RESIDUALS
pip install -r requirements.txt

# Run with included test DEM
python run_experiment.py

# Or generate your own from LiDAR
python generate_test_dem.py --lidar-dir /path/to/las/files --grid-rows 4 --grid-cols 4
python run_experiment.py --dem data/test_dems/your_dem.npy
```

## Decomposition Methods

| Method | How It Works | Good For |
|--------|--------------|----------|
| **Gaussian** | Local blur → subtract | Smooth gradients, valleys |
| **Bilateral** | Edge-preserving blur | Features with sharp boundaries |
| **Wavelet** | Multi-scale frequency separation | Scale-specific features |
| **Morphological** | Shape-based opening/closing | Peaks, ridges, depressions |
| **Top-Hat** | Small feature extraction | Small isolated features |
| **Polynomial** | Global surface fitting | Regional-scale variations |

Extended methods include: anisotropic Gaussian, median, DoG, LoG, guided filter, anisotropic diffusion, rolling ball, and multiple structuring element shapes.

## Upsampling Methods

| Method | Characteristics |
|--------|----------------|
| **Bicubic** | Balanced baseline (order 3) |
| **Lanczos** | Sharper edges, some ringing |
| **B-Spline** | Smoother (order 2) |
| **FFT Zero-pad** | Band-limited, Gibbs ringing at edges |

Extended methods include: nearest, bilinear, quadratic, quartic, quintic, windowed sinc (Hamming, Blackman), Catmull-Rom, Mitchell-Netravali, and edge-directed interpolation.

## Exhaustive Parameter Exploration

```bash
# Run all 39,731 parameter combinations
python run_exhaustive.py --output results/exhaustive

# Limited test run
python run_exhaustive.py --max-decomp 2 --max-upsamp 2

# Analyze redundancy across outputs
python analyze_redundancy.py --results D:\path\to\results --checksums results\CHECKSUMS.txt
```

Generates comprehensive documentation of all method combinations with statistics and hashes.

### Redundancy Analysis Results

Analysis of all 39,731 combinations revealed significant redundancy:

| Metric | Value |
|--------|-------|
| Total combinations | 39,731 |
| Exact duplicates | 3,345 groups |
| Near-duplicate pairs | 4,754,489 (0.6% of all pairs) |
| Distinct method clusters | 20 |

**Key findings:**
- `bspline` and `quadratic` upsampling produce **identical outputs** for all decomposition methods
- **Anisotropic Diffusion** dominates — 42% of combinations cluster together regardless of upsampling
- Decomposition method matters more than upsampling method for output characteristics
- 99.4% of method combinations produce genuinely distinct outputs

See `results/REDUNDANCY_REPORT.md` for full analysis.

## Output Visualization

The main output is a grid showing:
- **Rows**: Decomposition methods
- **Columns**: Upsampling methods + ground truth comparisons + divergence metrics

Each cell reveals different features. The Δ columns show where each method matches or misses ground truth features. The divergence columns show where methods disagree — useful for identifying features that are method-sensitive.

## Project Structure

```
RESIDUALS/
├── src/
│   ├── decomposition/     # 25 decomposition algorithms
│   ├── upsampling/        # 19 upsampling methods  
│   ├── analysis/          # Differential computation, feature detection
│   └── utils/             # Visualization, I/O
├── data/test_dems/        # Sample DEMs
├── results/
│   ├── combinations/      # Raw residual arrays (.npy)
│   ├── differentials/     # Pairwise differences
│   ├── visualizations/    # Output images (timestamped)
│   └── debug_archive/     # Bug documentation
├── generate_test_dem.py   # Create DEM from LiDAR tiles
├── run_experiment.py      # Main experiment runner
└── run_exhaustive.py      # Full parameter space exploration
```

## Applications

Feature detection in:
- **Terrain analysis** — ridges, valleys, drainage patterns
- **Infrastructure** — roads, embankments, foundations
- **Natural features** — geological formations, vegetation patterns
- **Change detection** — comparing DEMs over time
- **Quality assessment** — identifying artifacts in elevation data

Different decomposition methods excel at different feature types — the grid visualization helps identify which combination works best for your specific use case.

## Known Limitations

- **Edge artifacts**: Morphological methods show artifacts where terrain is cut off at image boundaries
- **Polynomial regional-scale**: Polynomial decomposition captures regional trends, not local features
- **Memory usage**: Large DEMs (>4000×4000) produce very large visualization files

See `results/debug_archive/README.md` for documented bugs and fixes.

## Contributing

Contributions welcome. Areas of interest:
- Additional decomposition methods
- GPU acceleration for large DEMs
- Machine learning feature classifiers
- Integration with GIS workflows

## Citation

If you use RESIDUALS in research, please cite:
```
@software{residuals2025,
  title={RESIDUALS: Multi-Method Differential Feature Detection},
  author={bshepp},
  year={2025},
  url={https://github.com/bshepp/RESIDUALS}
}
```

## License

Apache License 2.0

## Acknowledgments

- Sample LiDAR data: Connecticut Environmental Conditions Online (CT ECO)
- Built in CUrsor with: Opus 4.5, NumPy, SciPy, scikit-image, PyWavelets, OpenCV, Matplotlib
