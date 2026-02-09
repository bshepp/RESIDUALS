#!/usr/bin/env python3
"""
RESIDUALS -- Label-Free Feature Revelation Demo

Demonstrates unsupervised feature enhancement on a DEM using signal
decomposition. No labeled data is required: decomposition separates
regional topography (trend) from local anomalies (residual), and
earthworks, ditches, mounds, and other subtle features appear in the
residuals automatically.

Usage:
    python demo.py                          # uses default DEM
    python demo.py --dem path/to/dem.npy    # uses a custom DEM
    python demo.py --output my_figure.png   # saves to a specific file

Output:
    A comparison figure showing raw terrain alongside residual maps
    from 5 decomposition methods, plus a printed text summary.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.decomposition.registry import run_decomposition, list_decompositions
from src.upsampling.registry import run_upsampling
from src.utils.dem_io import load_dem

# Trigger method registration
import src.decomposition.methods       # noqa: F401
import src.decomposition.methods_extended  # noqa: F401
import src.upsampling.methods           # noqa: F401
import src.upsampling.methods_extended  # noqa: F401


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DEM = Path("data/test_dems/fairfield_sample_1.5ft.npy")

DECOMP_METHODS = [
    ("gaussian",     {"sigma": 5}),
    ("tophat",       {"size": 15, "mode": "white"}),
    ("morphological", {"operation": "opening", "size": 15}),
    ("dog",          {"sigma_low": 2, "sigma_high": 10}),
    ("wavelet_dwt",  {"wavelet": "db4", "level": 3}),
]

UPSAMP_METHODS = ["bicubic", "lanczos", "nearest"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hillshade(dem: np.ndarray, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    """Compute a hillshade from elevation data (returns 0-1 float array)."""
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    dy, dx = np.gradient(dem)
    slope = np.sqrt(dx ** 2 + dy ** 2)
    aspect = np.arctan2(-dx, dy)
    shaded = (np.cos(alt) * np.sin(np.arctan(slope)) *
              np.cos(az - aspect) +
              np.sin(alt) * np.cos(np.arctan(slope)))
    lo, hi = shaded.min(), shaded.max()
    if hi - lo > 0:
        shaded = (shaded - lo) / (hi - lo)
    return shaded


def make_synthetic_dem(rows: int = 256, cols: int = 256) -> np.ndarray:
    """
    Generate a synthetic DEM with a regional slope, a circular mound,
    and a linear ditch -- known earthwork-like features embedded in
    realistic-ish terrain.  Used as fallback when no real DEM is available.
    """
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    # Regional slope
    dem = 900.0 + 0.15 * x + 0.10 * y
    # Gentle rolling hills
    dem += 3.0 * np.sin(2 * np.pi * x / 80) * np.cos(2 * np.pi * y / 120)
    # Circular mound (subtle, ~1 ft rise)
    cx, cy, r = cols * 0.35, rows * 0.40, 25.0
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    dem += 1.2 * np.clip(1.0 - dist / r, 0, 1)
    # Linear ditch (subtle, ~0.8 ft depression)
    ditch_dist = np.abs((y - rows * 0.65) - 0.3 * (x - cols * 0.2))
    dem -= 0.8 * np.clip(1.0 - ditch_dist / 6.0, 0, 1)
    # Light noise
    rng = np.random.default_rng(42)
    dem += rng.normal(0, 0.05, dem.shape)
    return dem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(dem_path: Path, output_path: Path, scale: int = 2) -> None:
    """Run the full demo pipeline and produce the comparison figure."""

    # --- Load DEM -----------------------------------------------------------
    if dem_path.exists():
        print(f"Loading DEM: {dem_path}")
        dem = load_dem(str(dem_path))
        dem_label = dem_path.stem
    else:
        print(f"DEM not found at {dem_path}, generating synthetic terrain ...")
        dem = make_synthetic_dem()
        dem_label = "synthetic (mound + ditch)"

    print(f"  Shape: {dem.shape}  |  Range: {dem.min():.1f} - {dem.max():.1f}")

    # --- Decompose ----------------------------------------------------------
    results = {}  # {method_name: {"trend": ..., "residual": ...}}
    for name, params in DECOMP_METHODS:
        print(f"  Decomposing with {name} ...", end=" ", flush=True)
        trend, residual = run_decomposition(name, dem, params)
        results[name] = {"trend": trend, "residual": residual}
        print(f"residual std={residual.std():.4f}")

    # --- Upsample residuals (one representative: bicubic 2x) ---------------
    upsamp_name = "bicubic"
    print(f"\n  Upsampling residuals with {upsamp_name} ({scale}x) ...")
    for name in results:
        res = results[name]["residual"]
        results[name]["upsampled"] = run_upsampling(upsamp_name, res, scale=scale)

    # --- Build figure -------------------------------------------------------
    n_methods = len(DECOMP_METHODS)
    n_rows = n_methods + 1
    n_cols = 3

    # Compute figure size from DEM shape so pixels stay square.
    # Each column shows an image of width dem.shape[1]; each row height dem.shape[0].
    dem_h, dem_w = dem.shape
    cell_w = 4.0                          # target inches per subplot column
    cell_h = cell_w * (dem_h / dem_w)     # preserve aspect ratio
    cell_h = min(cell_h, 12.0)            # cap so figure stays reasonable
    fig_w = cell_w * n_cols + 3.5         # extra room for colorbars + labels
    fig_h = cell_h * n_rows + 1.5         # extra room for title + spacing

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.25, "hspace": 0.20},
    )

    # Row 0: raw terrain
    hillshade = make_hillshade(dem)
    ax = axes[0, 0]
    ax.imshow(hillshade, cmap="gray", origin="lower", aspect="equal")
    ax.set_title("Hillshade", fontsize=10, fontweight="bold")
    ax.set_ylabel("Raw Terrain", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 1]
    im = ax.imshow(dem, cmap="terrain", origin="lower", aspect="equal")
    ax.set_title("Elevation", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ft")

    # Row 0, col 2: blank / legend
    ax = axes[0, 2]
    legend_text = (
        "RESIDUALS Demo\n"
        "──────────────────\n"
        f"DEM: {dem_label}\n"
        f"Shape: {dem.shape[0]}x{dem.shape[1]}\n"
        f"Elev: {dem.min():.1f} - {dem.max():.1f} ft\n\n"
        "Residual maps reveal\n"
        "local anomalies hidden\n"
        "under regional topography.\n\n"
        "Red/blue = positive/negative\n"
        "deviations from trend."
    )
    ax.text(0.5, 0.5, legend_text, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f0f0", alpha=0.9))
    ax.axis("off")

    # Rows 1..n: decomposition results
    for i, (name, _params) in enumerate(DECOMP_METHODS):
        row = i + 1
        data = results[name]
        trend = data["trend"]
        residual = data["residual"]
        upsampled = data["upsampled"]

        # Trend
        ax = axes[row, 0]
        ax.imshow(trend, cmap="terrain", origin="lower", aspect="equal")
        ax.set_title("Trend", fontsize=9)
        ax.set_ylabel(name, fontsize=10, fontweight="bold", rotation=90, labelpad=8)
        ax.set_xticks([]); ax.set_yticks([])

        # Residual (original resolution)
        ax = axes[row, 1]
        vmax = np.percentile(np.abs(residual), 99)
        if vmax == 0:
            vmax = 1
        im = ax.imshow(residual, cmap="RdBu_r", origin="lower", aspect="equal",
                        vmin=-vmax, vmax=vmax)
        ax.set_title("Residual", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Upsampled residual
        ax = axes[row, 2]
        vmax_up = np.percentile(np.abs(upsampled), 99)
        if vmax_up == 0:
            vmax_up = 1
        im = ax.imshow(upsampled, cmap="RdBu_r", origin="lower", aspect="equal",
                        vmin=-vmax_up, vmax=vmax_up)
        ax.set_title(f"Residual {scale}x ({upsamp_name})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "RESIDUALS: Label-Free Feature Enhancement via Signal Decomposition",
        fontsize=13, fontweight="bold", y=0.995,
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nFigure saved: {output_path}")

    # --- Text summary -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESIDUALS -- Demo Summary")
    print("=" * 60)
    print(f"  DEM        : {dem_label}")
    print(f"  Shape      : {dem.shape[0]} x {dem.shape[1]}")
    print(f"  Elevation  : {dem.min():.1f} - {dem.max():.1f} ft")
    print(f"  Upsampling : {upsamp_name} {scale}x")
    print("-" * 60)
    print(f"  {'Method':<22s} {'Residual Std':>13s} {'Range':>18s}")
    print("-" * 60)

    best_name, best_std = None, 0.0
    for name, _params in DECOMP_METHODS:
        res = results[name]["residual"]
        std = res.std()
        rng = f"{res.min():.3f} .. {res.max():.3f}"
        print(f"  {name:<22s} {std:>13.4f} {rng:>18s}")
        if std > best_std:
            best_std = std
            best_name = name

    print("-" * 60)
    print(f"  Highest-contrast residual: {best_name} (std={best_std:.4f})")
    print("  -> Likely best for revealing subtle terrain features.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RESIDUALS demo: label-free feature enhancement on a DEM."
    )
    parser.add_argument(
        "--dem", type=Path, default=DEFAULT_DEM,
        help=f"Path to DEM file (.npy, .tif, .las).  Default: {DEFAULT_DEM}"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("demo_output.png"),
        help="Output figure path.  Default: demo_output.png"
    )
    parser.add_argument(
        "--scale", type=int, default=2,
        help="Upsampling scale factor.  Default: 2"
    )
    args = parser.parse_args()
    run_demo(args.dem, args.output, args.scale)


if __name__ == "__main__":
    main()
