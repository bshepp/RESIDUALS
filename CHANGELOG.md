# Changelog

All notable changes to RESIDUALS are documented here.

## [Unreleased]

## [0.2.0] - 2026-10-10

### Added
- `pyproject.toml` for modern Python packaging (replaces bare `requirements.txt` as primary config).
- Ruff linter/formatter configuration.
- GitHub Actions CI pipeline (lint + test).
- `.editorconfig` for consistent editor settings.
- `CONTRIBUTING.md` with development setup and method registration guide.
- `.env.example` documenting the `RESIDUALS_LIDAR_DIR` environment variable.
- `roundtrip_resample()` helper in `residuals/utils/preprocessing.py` for down→process→up round trips.
- `decompose_rolling_ball` now exposes `down_order`, `up_order`, `target_effective_radius`, and
  `large_radius_threshold` as kwargs (defaults preserve previous behavior bit-for-bit). The
  interpolation orders are registered so future sweeps record them.
- README links to the published exhaustive-sweep dataset (Hugging Face, Zenodo DOI
  10.5281/zenodo.19903273) and project page, plus a dataset citation.

### Changed
- **Package renamed from `src` to `residuals`.** Import as `from residuals.decomposition ...`
  instead of `from src.decomposition ...`. Installed via `pip install -e .` as before.
- LiDAR-dependent scripts (`tile_county.py`, `trace_road.py`, `generate_licking_dem.py`,
  `generate_test_dem.py`) now accept `--lidar-dir` CLI arg or `RESIDUALS_LIDAR_DIR` env var
  instead of hardcoded paths.
- Long-running scripts write their logs to `logs/`; `demo.py` writes to `results/demo/` by default.
- Minimum Python is now 3.10 (3.9 is end of life). CI tests 3.11, 3.12, and 3.13.
- Roadmap: Phase 1 (county corpus) marked complete; Phase 2 (indexing) is current.

## [0.1.0] - 2026-02-17

### Added
- County grid tiling: blanket Licking County with ~200 tiles, each producing a 12-panel archaeological visualization (SVF, openness, RRIM, multi-scale TopHat). (`tile_county.py`)
- Road corridor trace: extend Hopewell Road detection across the county with consensus heatmap and GeoJSON centerline. (`trace_road.py`)
- Leaflet tile viewer for browsing county results in a web browser. (`build_viewer.py`, `viewer.html`)
- Archaeological demo with side-by-side hillshade vs. residual comparisons for Great Hopewell Road and Great Circle Earthworks. (`demo.py`)
- Best-of pipeline: 20 cluster representatives x 3 upsamplers = 60 combinations. (`run_bestof.py`)
- 166 pytest tests covering all methods, registries, known-answer correctness, and analysis.
- Project refactor: extracted `src/` library structure with registry-based decomposition and upsampling, analysis module, and utilities.

### Changed
- Archived one-off exhaustive-run scripts to `scripts/archive/`.

## [0.0.2] - 2026-01-09

### Added
- Redundancy analysis: SHA-256 checksums + statistical fingerprinting across all 39,731 combinations.
- Identified 20 distinct method clusters and 3,345 exact duplicate groups.
- Parallel fingerprint generator optimized for HDD I/O.
- Rolling ball fix for large radii using downsampling approach.
- Resumable fingerprinting with checkpoint support.
- Prior art documentation (`PRIOR_ART.md`) with full method parameter spaces and results.

## [0.0.1] - 2025-12-30

### Added
- Initial release: 4-level differential framework for archaeological feature detection in LiDAR DEMs.
- 25 decomposition methods (Gaussian, bilateral, wavelet, morphological, tophat, polynomial, and 19 extended methods).
- 19 upsampling methods (bicubic, Lanczos, B-spline, FFT zero-pad, and 15 extended methods).
- Exhaustive parameter exploration: 39,731 combinations generating 4.28 TB of prior art data.
- Apache License 2.0.
