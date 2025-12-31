# DIVERGE: Differential Verification of Ground-truth Extraction

**A framework for archaeological feature detection in LiDAR DEMs using multi-level differential analysis.**

![Sample Output](results/visualizations/residuals_grid_preview.png)

## What Is This?

DIVERGE systematically tests combinations of signal decomposition and upsampling methods to identify which combinations best reveal hidden terrain features. The core insight:

> *Different method combinations have characteristic "failure modes" that selectively preserve or eliminate different feature types. By computing differentials between outputs, we create feature-specific extraction filters.*

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
git clone https://github.com/YOUR_USERNAME/DIVERGE.git
cd DIVERGE
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
| **Gaussian** | Local blur → subtract | Drainage channels, valleys |
| **Bilateral** | Edge-preserving blur | Features with sharp boundaries |
| **Wavelet** | Multi-scale frequency separation | Scale-specific features |
| **Morphological** | Shape-based opening/closing | Mounds, peaks, ridges |
| **Top-Hat** | Small feature extraction | Small isolated features |
| **Polynomial** | Global surface fitting | Regional-scale modifications |

## Upsampling Methods

| Method | Characteristics |
|--------|----------------|
| **Bicubic** | Balanced baseline (order 3) |
| **Lanczos** | Sharper edges, some ringing |
| **B-Spline** | Smoother (order 2) |
| **FFT Zero-pad** | Band-limited, Gibbs ringing at edges |

## Output Visualization

The main output is a grid showing:
- **Rows**: Decomposition methods
- **Columns**: Upsampling methods + ground truth comparisons + divergence metrics

Each cell reveals different terrain features. The Δ columns (pink-green) show where each method matches or misses ground truth features. The divergence columns (hot/viridis) show where methods disagree.

## Project Structure

```
DIVERGE/
├── src/
│   ├── decomposition/     # 6 decomposition algorithms
│   ├── upsampling/        # 4 upsampling methods  
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
└── check_correlation.py   # Sanity check script
```

## Archaeological Applications

The framework detects:
- **Linear features** (buried roads, walls, field boundaries)
- **Mound/pit features** (burial mounds, storage pits)
- **Periodic patterns** (agricultural terracing, urban grids)
- **Landscape modifications** (land leveling, canal systems)

Different decomposition methods excel at different feature types — the grid visualization helps identify which combination works best for your specific targets.

## Known Limitations

- **Edge artifacts**: Morphological methods show artifacts where terrain is cut off at image boundaries
- **Polynomial regional-scale**: Polynomial decomposition captures regional trends, not local features
- **Memory usage**: Large DEMs (>4000×4000) produce very large visualization files

See `results/debug_archive/README.md` for documented bugs and fixes.

## Contributing

We welcome contributions! Areas of interest:
- Additional decomposition methods
- GPU acceleration for large DEMs
- Machine learning feature classifiers
- Integration with archaeological GIS workflows
- Validation on sites with known features

## Citation

If you use DIVERGE in research, please cite:
```
@software{diverge2024,
  title={DIVERGE: Differential Verification of Ground-truth Extraction},
  author={Your Name},
  year={2024},
  url={https://github.com/YOUR_USERNAME/DIVERGE}
}
```

## License

MIT License - Use freely, attribution appreciated.

## Acknowledgments

- LiDAR data: Connecticut Environmental Conditions Online (CT ECO)
- Built with: NumPy, SciPy, scikit-image, PyWavelets, OpenCV, Matplotlib
