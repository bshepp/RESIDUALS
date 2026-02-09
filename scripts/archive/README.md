# Archived Scripts

One-off and maintenance scripts from the exhaustive parameter exploration run
(Jan 2026). These are not part of the core RESIDUALS pipeline but are preserved
for reference.

## Exhaustive Run Maintenance

| Script | Purpose |
|--------|---------|
| `analyze_redundancy.py` | Post-run redundancy analysis (SHA256 + fingerprinting) |
| `fingerprint_all_parallel.py` | Parallel fingerprinting of 39K result files |
| `fingerprint_failed_only.py` | Retry fingerprinting for memory-failed large files |
| `rerun_failed_fingerprints.py` | Another retry pass at failed fingerprints |
| `regenerate_corrupt.py` | Regenerate files truncated by disk-full errors |
| `regen_rolling_ball.py` | Regenerate rolling_ball_radius200 after edge case fix |
| `extract_failed_combinations.py` | Parse exhaustive log for failed combo IDs |
| `generate_checksums.py` | Generate SHA256 checksums for all result files |

## Diagnostic / One-Off

| Script | Purpose |
|--------|---------|
| `check_coordinate.py` | Verify a lat/lon falls within Licking County LiDAR tiles |
| `check_correlation.py` | Sanity check: correlation between Div and DeltaDiv columns |

## Data / Docs

| File | Purpose |
|------|---------|
| `RERUN_PLAN.md` | Notes on re-running 3,399 failed exhaustive combos |
| `failed_combinations.txt` | List of failed combo IDs from exhaustive run |
