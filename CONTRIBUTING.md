# Contributing to RESIDUALS

## Development Setup

```bash
git clone https://github.com/bshepp/RESIDUALS.git
cd RESIDUALS
pip install -e ".[all]"   # installs core + geo + dev dependencies
```

## Running Tests

```bash
pytest                # full suite (166 tests)
pytest -x             # stop on first failure
pytest -k gaussian    # run only tests matching "gaussian"
```

## Linting and Formatting

```bash
ruff check .          # lint (pyflakes, pycodestyle, isort)
ruff check . --fix    # auto-fix what it can
ruff format .         # auto-format
```

CI runs both `ruff check` and `ruff format --check` on every push and PR.

## Adding a New Decomposition Method

All decomposition methods live in `src/decomposition/methods.py` (core) or `src/decomposition/methods_extended.py` (extended). Use the `@register_decomposition` decorator:

```python
from .registry import register_decomposition
from ..utils.preprocessing import fill_nans

@register_decomposition(
    name='my_method',
    category='classical',                    # classical, edge_preserving, wavelet, morphological, etc.
    default_params={'sigma': 10},
    param_ranges={'sigma': [2, 5, 10, 20]},
    preserves='what features survive',
    destroys='what features are removed',
)
def decompose_my_method(dem, sigma=10):
    dem_filled = fill_nans(dem)
    trend = some_filter(dem_filled, sigma=sigma)
    residual = dem_filled - trend
    return trend, residual
```

Requirements:
- Call `fill_nans(dem)` as the first operation.
- Return a `(trend, residual)` tuple where both have the same shape as the input.
- The method auto-registers on import; no other wiring needed.
- Add the method name to `test_decomposition.py::TestMethodCount::test_expected_method_count` if it changes the total.

## Adding a New Upsampling Method

Same pattern in `src/upsampling/methods.py` or `src/upsampling/methods_extended.py`:

```python
from .registry import register_upsampling
from ..utils.preprocessing import fill_nans

@register_upsampling(
    name='my_upsampler',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='what it preserves',
    introduces='what artifacts it introduces',
)
def upsample_my_method(dem, scale=2):
    dem_filled = fill_nans(dem)
    return some_interpolation(dem_filled, scale)
```

Requirements:
- Call `fill_nans(dem)` first.
- Return a 2D array with dimensions scaled by `scale`.
- Update the method count in `test_upsampling.py::TestMethodCount::test_expected_method_count`.

## LiDAR Data

Scripts that load LiDAR tiles (`tile_county.py`, `trace_road.py`, `generate_licking_dem.py`, `generate_test_dem.py`) resolve the data directory from:

1. `--lidar-dir` CLI argument (highest priority)
2. `RESIDUALS_LIDAR_DIR` environment variable
3. Failure with a clear error message

Copy `.env.example` to `.env` and set `RESIDUALS_LIDAR_DIR` to avoid passing `--lidar-dir` every time.

## Project Conventions

- **snake_case** for everything: files, functions, variables, method registry keys.
- **No classes** except the two registry dataclasses. Keep it functional.
- **Logging** via `logging.getLogger(__name__)`. No print statements in library code.
- **Error handling**: try/except around individual methods in batch runs so one failure doesn't halt the pipeline.
- **Tests**: parametrized over all registered methods where possible. Known-answer tests for mathematical correctness.
