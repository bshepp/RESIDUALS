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

# Run the demo (auto-selects best available DEM)
python demo.py

# Run with a specific archaeological site
python demo.py --site hopewell_road     # Great Hopewell Road
python demo.py --site great_circle      # Great Circle Earthworks (Newark)

# Run the best-of pipeline (20 methods x 3 upsamplers = 60 combos)
python run_bestof.py --dem data/test_dems/fairfield_sample_1.5ft.npy

# Full experiment (all methods x all upsamplers)
python run_experiment.py --dem data/test_dems/fairfield_sample_1.5ft.npy

# Generate a DEM from LiDAR tiles
python generate_test_dem.py --lidar-dir /path/to/las/files --grid-rows 4 --grid-cols 4

# Generate the Licking County Hopewell sites DEM (requires local LiDAR data)
python generate_licking_dem.py
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

## Best-Of Pipeline

The exhaustive analysis of 39,731 combinations found 20 truly distinct method clusters. The best-of pipeline runs one representative per cluster with 3 upsamplers = **60 combinations** that cover the full diversity:

```bash
python run_bestof.py --dem data/test_dems/fairfield_sample_1.5ft.npy
python run_bestof.py --dem data/test_dems/licking_hopewell_2.5ft.npy --output results/hopewell
```

See `run_bestof.py` for the full cluster table with descriptions of what each method reveals.

## Archaeological Demo

The demo targets two Hopewell-culture sites in Licking County, Ohio:

- **Great Hopewell Road** — a subtle ~60 ft wide raised embankment, often <1 ft above surrounding terrain, nearly invisible in raw hillshade but visible in DoG and morphological residuals.
- **Great Circle Earthworks** — a ~1050 ft diameter circular enclosure (part of the Newark Earthworks, UNESCO World Heritage Site), visible in top-hat and rolling ball residuals.

```bash
python demo.py --site hopewell_road      # side-by-side: hillshade vs. residuals
python demo.py --site great_circle       # same for the Great Circle
python demo.py                           # full DEM grid view
```

## Exhaustive Parameter Exploration

```bash
# Run all 39,731 parameter combinations
python run_exhaustive.py --output results/exhaustive

# Limited test run
python run_exhaustive.py --max-decomp 2 --max-upsamp 2
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

## Testing

166 tests covering decomposition, upsampling, registry, analysis, and preprocessing:

```bash
python -m pytest tests/ -v
```

Includes 22 known-answer tests that verify methods against mathematically predictable inputs (Gaussian on linear ramps, polynomial on planes, top-hat on spikes, etc.).

## Project Structure

```
RESIDUALS/
├── src/
│   ├── decomposition/        # 25 decomposition algorithms
│   ├── upsampling/           # 19 upsampling methods
│   ├── analysis/             # Differential computation, feature detection
│   └── utils/                # Visualization, I/O, preprocessing
├── tests/                    # 166 pytest tests (known-answer, registry, etc.)
├── data/test_dems/           # Sample DEMs + metadata
├── results/
│   ├── combinations/         # Raw residual arrays (.npy)
│   ├── differentials/        # Pairwise differences
│   ├── visualizations/       # Output images
│   └── debug_archive/        # Bug documentation
├── scripts/archive/          # One-off diagnostic scripts (not part of core)
├── demo.py                   # Archaeological site demo (Hopewell Road, Great Circle)
├── run_bestof.py             # Best-of pipeline (20 clusters x 3 upsamplers)
├── run_experiment.py         # Full experiment runner
├── run_exhaustive.py         # Exhaustive parameter space exploration
├── generate_test_dem.py      # Create DEM from LiDAR tiles
└── generate_licking_dem.py   # Generate Licking County Hopewell sites DEM
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

- Licking County LiDAR: Ohio Statewide Imagery Program (OSIP) 2015
- Fairfield County LiDAR: Connecticut Environmental Conditions Online (CT ECO)
- Built in Cursor with: Claude, NumPy, SciPy, scikit-image, PyWavelets, OpenCV, Matplotlib
