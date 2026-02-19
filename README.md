# RESIDUALS: Multi-Method Differential Feature Detection

[![CI](https://github.com/bshepp/RESIDUALS/actions/workflows/ci.yml/badge.svg)](https://github.com/bshepp/RESIDUALS/actions/workflows/ci.yml)

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
pip install -e .              # core dependencies
pip install -e ".[geo]"       # + geospatial (rasterio, geopandas, laspy)
pip install -e ".[all]"       # everything including dev tools

# Or use requirements.txt (equivalent to core + geo)
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
python generate_licking_dem.py --lidar-dir /path/to/licking_2015
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

## Road Corridor Trace

Extends the Hopewell Road detection across all of Licking County. Generates a corridor of DEM strips along the detected road bearing, applies multi-method residual analysis with ridge detection, and outputs georeferenced GeoTIFFs and a GeoJSON centerline.

```bash
# Full corridor trace (~4 hours, processes ~13 DEM strips)
python trace_road.py --lidar-dir /path/to/licking_2015

# Quick test with fewer strips
python trace_road.py --lidar-dir /path/to/licking_2015 --max-strips 2

# Reuse existing strip DEMs (skip regeneration)
python trace_road.py --skip-dem-gen
```

All LiDAR-dependent scripts accept `--lidar-dir` or read the `RESIDUALS_LIDAR_DIR` environment variable. See `.env.example`.

Outputs:
- `results/hopewell_trace/corridor_confidence.tif` — multi-method consensus heatmap (GeoTIFF)
- `results/hopewell_trace/corridor_residual_tophat.tif` — best single-method residual (GeoTIFF)
- `results/hopewell_trace/road_centerline.geojson` — detected road polyline
- `results/hopewell_trace/corridor_overview.png` — summary figure

## County Grid Tiling

Blankets all of Licking County with a regular 2-mile grid of DEM tiles, computing professional archaeological visualizations for each. Each tile produces a 12-panel PNG comparing visualization methods, terrain shape analysis, and feature extraction techniques.

```bash
# Full county (~200 tiles, ~20 hours first run, ~7 hours cached re-render)
python tile_county.py --lidar-dir /path/to/licking_2015

# Process a subset of tiles (0-indexed row/col ranges)
python tile_county.py --lidar-dir /path/to/licking_2015 --rows 3-5 --cols 7-10

# Re-render PNGs from cached DEMs + SVF (skips expensive generation)
python tile_county.py --png-only

# Just regenerate the county index map
python tile_county.py --index-only
```

Each tile's 12-panel figure includes:

| Row | Purpose | Panels |
|-----|---------|--------|
| Visualization | Professional methods | Multi-Dir Hillshade, SVF, RRIM, Slope |
| Terrain Shape | Openness + baselines | Positive Openness, Negative Openness, SLRM, TPI |
| Feature Extraction | TopHat multi-scale | r=50ft (road), r=7.5ft (fine), r=25ft (embankment), r=100ft (landscape) |

Outputs:
- `results/county_tiles/R04_C08.png` — 12-panel visualization per tile
- `results/county_tiles/R04_C08_dem.npy` — cached DEM array (~143 MB)
- `results/county_tiles/R04_C08_svf.npy` — cached Sky-View Factor (~71 MB)
- `results/county_tiles/index_map.png` — county overview with numbered tile boxes
- `results/county_tiles/metadata.json` — grid dimensions and processing status

Disk usage: ~84 GB for all 200 tiles (DEMs + SVF/openness caches + PNGs).

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
pytest                     # uses settings from pyproject.toml
python -m pytest tests/ -v # equivalent
```

Includes 22 known-answer tests that verify methods against mathematically predictable inputs (Gaussian on linear ramps, polynomial on planes, top-hat on spikes, etc.).

## Development

```bash
# Install with dev dependencies
pip install -e ".[all]"

# Run tests
pytest

# Lint
ruff check .

# Auto-format
ruff format .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add new methods and submit changes.

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
├── .github/workflows/        # CI: lint + test on push/PR
├── pyproject.toml            # Package metadata, dependencies, tool config
├── demo.py                   # Archaeological site demo (Hopewell Road, Great Circle)
├── tile_county.py            # County grid tiling: 200 tiles x 12-panel visualizations
├── trace_road.py             # Corridor trace: extend road detection across county
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

## Roadmap

### Phase 1: County Corpus (current)
Complete the Licking County grid tiling — 200 tiles with cached DEMs, SVF, openness, and multi-scale TopHat residuals. This produces the searchable corpus for all downstream work.

### Phase 2: Index the Corpus
Run [terravector](https://github.com/bshepp/terravector) on the cached DEMs to build an HNSW index of ~1.7 million terrain patches across Licking County. Enable queries like "find all patches that look like the Great Circle mound."

### Phase 3: Recommendation API
Build the RESIDUALS-as-a-Service prototype (see `terravector/docs/RESIDUALS_SERVICE_CONCEPT.md`): given a terrain patch, recommend which decomposition method will best reveal its features, using the indexed corpus as training data.

### Phase 4: GIS Plugins
Package the indexing and query pipeline as a QGIS plugin. Target UX: "Right-click a patch → Find Similar Terrain" with sub-millisecond results. Consider ArcGIS plugin as a second target.

### Phase 5: GPU Acceleration
Move the expensive per-tile computations (SVF horizon scanning, morphological operations, decomposition methods) to GPU via CuPy or CUDA kernels. Target: real-time processing for interactive use.

### Phase 6: Cloud Deployment
Host the index and recommendation API on AWS/GCP for public access. Enable researchers to query against the corpus without local infrastructure.

### Phase 7: ML Feature Classifiers
Train classifiers on the indexed corpus to automatically identify feature types (mounds, roads, enclosures, natural formations). This sits as a **separate layer on top** of the pure algorithmic core — RESIDUALS and terravector remain training-free signal processing tools that work without any ML dependency.

### Architecture Note

The stack is intentionally layered so each level works independently:

```
Layer 3: ML classifiers (optional, trained on indexed corpus)
Layer 2: terravector (HNSW indexing + similarity search)
Layer 1: RESIDUALS (pure signal decomposition + feature extraction)
```

RESIDUALS must never depend on ML. terravector must never depend on ML. The ML layer consumes their outputs but the algorithmic tools remain usable, interpretable, and reproducible without it.

## Contributing

Contributions welcome across any roadmap phase. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions, and how to add new methods.

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
