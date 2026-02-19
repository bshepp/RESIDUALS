#!/usr/bin/env python3
"""
RESIDUALS -- Label-Free Feature Revelation Demo

Demonstrates unsupervised feature enhancement on a DEM using signal
decomposition.  No labeled data is required: decomposition separates
regional topography (trend) from local anomalies (residual), and
earthworks, ditches, mounds, and other subtle features appear in the
residuals automatically.

Usage:
    python demo.py                                          # full DEM grid
    python demo.py --site hopewell_road                     # crop to site
    python demo.py --site great_circle                      # crop to site
    python demo.py --dem path/to/dem.npy                    # custom DEM
    python demo.py --dem path/to/dem.npy --meta path.json   # with site metadata
    python demo.py --output my_figure.png                   # custom output path

Output:
    A comparison figure showing raw hillshade alongside residual maps
    from 5 decomposition methods, plus a printed text summary.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Trigger method registration
import src.decomposition.methods  # noqa: F401
import src.decomposition.methods_extended  # noqa: F401
import src.upsampling.methods  # noqa: F401
import src.upsampling.methods_extended  # noqa: F401

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.decomposition.registry import run_decomposition
from src.upsampling.registry import run_upsampling
from src.utils.dem_io import load_dem

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LICKING_DEM = Path("data/test_dems/licking_hopewell_2.5ft.npy")
LICKING_META = Path("data/test_dems/licking_hopewell_2.5ft.json")
FAIRFIELD_DEM = Path("data/test_dems/fairfield_sample_1.5ft.npy")

DECOMP_METHODS = [
    ("gaussian", {"sigma": 5}, "Gaussian (sigma=5)"),
    ("tophat", {"size": 15, "mode": "white"}, "Top-hat (white, 15)"),
    ("morphological", {"operation": "opening", "size": 15}, "Morph opening (disk 15)"),
    ("dog", {"sigma_low": 2, "sigma_high": 10}, "DoG (2-10)"),
    ("wavelet_dwt", {"wavelet": "db4", "level": 3}, "Wavelet DWT (db4 L3)"),
]

# Site crop radius in pixels (used when --site is given)
SITE_CROP_RADIUS = 400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_hillshade(dem, azimuth=315, altitude=45):
    """Compute a hillshade (returns 0-1 float array)."""
    az, alt = np.radians(azimuth), np.radians(altitude)
    dy, dx = np.gradient(dem)
    slope = np.sqrt(dx**2 + dy**2)
    aspect = np.arctan2(-dx, dy)
    shaded = np.cos(alt) * np.sin(np.arctan(slope)) * np.cos(az - aspect) + np.sin(alt) * np.cos(
        np.arctan(slope)
    )
    lo, hi = shaded.min(), shaded.max()
    return (shaded - lo) / (hi - lo) if hi > lo else shaded


def make_synthetic_dem(rows=256, cols=256):
    """Synthetic DEM with embedded mound + ditch (fallback if no real data)."""
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    dem = 900.0 + 0.15 * x + 0.10 * y
    dem += 3.0 * np.sin(2 * np.pi * x / 80) * np.cos(2 * np.pi * y / 120)
    cx, cy, r = cols * 0.35, rows * 0.40, 25.0
    dem += 1.2 * np.clip(1.0 - np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / r, 0, 1)
    ditch_dist = np.abs((y - rows * 0.65) - 0.3 * (x - cols * 0.2))
    dem -= 0.8 * np.clip(1.0 - ditch_dist / 6.0, 0, 1)
    rng = np.random.default_rng(42)
    dem += rng.normal(0, 0.05, dem.shape)
    return dem


def load_site_metadata(meta_path):
    """Load site coordinate metadata from the JSON sidecar."""
    if meta_path and Path(meta_path).exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def crop_to_site(dem, meta, site_name, radius=SITE_CROP_RADIUS):
    """Crop the DEM around a named site.  Returns (cropped_dem, site_info)."""
    if meta is None or "sites" not in meta or site_name not in meta["sites"]:
        raise ValueError(
            f"Site '{site_name}' not found in metadata.  "
            f"Available: {list(meta.get('sites', {}).keys()) if meta else 'none'}"
        )

    site = meta["sites"][site_name]
    cx = int(round(site["pixel_x"]))
    cy = int(round(site["pixel_y"]))

    h, w = dem.shape
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius)

    cropped = dem[y0:y1, x0:x1]
    # Compute site center in the cropped coordinate system
    site_local = {"pixel_x": cx - x0, "pixel_y": cy - y0}
    site_local.update(site)
    return cropped, site_local


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_site_comparison(dem, site_info, results, output_path, site_name):
    """
    The money shot: hillshade vs. best residuals for a single site.

    Layout (2 columns x N+1 rows):
        Row 0:  Hillshade  |  Elevation
        Row 1+: Hillshade  |  Residual  (one per method)

    Site location is marked with a crosshair on every panel.
    """
    n_methods = len(DECOMP_METHODS)
    n_rows = n_methods + 1
    n_cols = 2

    dem_h, dem_w = dem.shape
    cell_w = 5.0
    cell_h = cell_w * (dem_h / dem_w)
    cell_h = min(cell_h, 8.0)
    fig_w = cell_w * n_cols + 2.0
    fig_h = cell_h * n_rows + 1.5

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), gridspec_kw={"wspace": 0.15, "hspace": 0.22}
    )

    hillshade = make_hillshade(dem)
    sx = site_info.get("pixel_x")
    sy = site_info.get("pixel_y")
    desc = site_info.get("description", site_name)

    def mark_site(ax):
        if sx is not None and sy is not None:
            ax.plot(sx, sy, "x", color="lime", markersize=10, markeredgewidth=2)

    # Row 0: hillshade + elevation
    ax = axes[0, 0]
    ax.imshow(hillshade, cmap="gray", origin="lower", aspect="equal")
    ax.set_title("Hillshade (raw LiDAR)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Raw", fontsize=10, fontweight="bold")
    mark_site(ax)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 1]
    im = ax.imshow(dem, cmap="terrain", origin="lower", aspect="equal")
    ax.set_title("Elevation", fontsize=10, fontweight="bold")
    mark_site(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ft")

    # Rows 1..n: hillshade | residual
    for i, (name, _params, label) in enumerate(DECOMP_METHODS):
        row = i + 1
        residual = results[name]["residual"]

        # Left: hillshade (repeated for direct comparison)
        ax = axes[row, 0]
        ax.imshow(hillshade, cmap="gray", origin="lower", aspect="equal")
        ax.set_ylabel(label, fontsize=9, fontweight="bold", rotation=90, labelpad=8)
        mark_site(ax)
        ax.set_xticks([])
        ax.set_yticks([])

        # Right: residual
        ax = axes[row, 1]
        vmax = np.percentile(np.abs(residual), 99)
        if vmax == 0:
            vmax = 1
        im = ax.imshow(
            residual, cmap="RdBu_r", origin="lower", aspect="equal", vmin=-vmax, vmax=vmax
        )
        ax.set_title("Residual", fontsize=9)
        mark_site(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"RESIDUALS: {desc}\nRaw hillshade (left) vs. decomposition residuals (right)",
        fontsize=12,
        fontweight="bold",
        y=1.0,
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_grid(dem, dem_label, results, output_path, scale, upsamp_name):
    """Original grid view: trend | residual | upsampled for each method."""
    n_methods = len(DECOMP_METHODS)
    n_rows = n_methods + 1
    n_cols = 3

    dem_h, dem_w = dem.shape
    cell_w = 4.0
    cell_h = min(cell_w * (dem_h / dem_w), 12.0)
    fig_w = cell_w * n_cols + 3.5
    fig_h = cell_h * n_rows + 1.5

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), gridspec_kw={"wspace": 0.25, "hspace": 0.20}
    )

    hillshade = make_hillshade(dem)

    # Row 0: raw terrain
    ax = axes[0, 0]
    ax.imshow(hillshade, cmap="gray", origin="lower", aspect="equal")
    ax.set_title("Hillshade", fontsize=10, fontweight="bold")
    ax.set_ylabel("Raw Terrain", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 1]
    im = ax.imshow(dem, cmap="terrain", origin="lower", aspect="equal")
    ax.set_title("Elevation", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ft")

    ax = axes[0, 2]
    legend_text = (
        "RESIDUALS Demo\n"
        f"DEM: {dem_label}\n"
        f"Shape: {dem.shape[0]}x{dem.shape[1]}\n"
        f"Elev: {dem.min():.1f} - {dem.max():.1f} ft\n\n"
        "Residual maps reveal\n"
        "local anomalies hidden\n"
        "under regional topography.\n\n"
        "Red/blue = +/- deviations."
    )
    ax.text(
        0.5,
        0.5,
        legend_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f0f0", alpha=0.9),
    )
    ax.axis("off")

    for i, (name, _params, label) in enumerate(DECOMP_METHODS):
        row = i + 1
        data = results[name]

        ax = axes[row, 0]
        ax.imshow(data["trend"], cmap="terrain", origin="lower", aspect="equal")
        ax.set_title("Trend", fontsize=9)
        ax.set_ylabel(label, fontsize=9, fontweight="bold", rotation=90, labelpad=8)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row, 1]
        vmax = np.percentile(np.abs(data["residual"]), 99) or 1
        im = ax.imshow(
            data["residual"], cmap="RdBu_r", origin="lower", aspect="equal", vmin=-vmax, vmax=vmax
        )
        ax.set_title("Residual", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row, 2]
        up = data["upsampled"]
        vmax_up = np.percentile(np.abs(up), 99) or 1
        im = ax.imshow(
            up, cmap="RdBu_r", origin="lower", aspect="equal", vmin=-vmax_up, vmax=vmax_up
        )
        ax.set_title(f"Residual {scale}x ({upsamp_name})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "RESIDUALS: Label-Free Feature Enhancement via Signal Decomposition",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_demo(dem_path, output_path, scale=2, site=None, meta_path=None):
    """Run the demo pipeline."""

    # --- Choose DEM ---
    if dem_path.exists():
        print(f"Loading DEM: {dem_path}")
        dem_full = load_dem(str(dem_path))
        dem_label = dem_path.stem
    else:
        print(f"DEM not found at {dem_path}, generating synthetic terrain ...")
        dem_full = make_synthetic_dem()
        dem_label = "synthetic (mound + ditch)"

    # --- Load metadata for site info ---
    meta = load_site_metadata(meta_path)

    # --- Crop to site if requested ---
    site_info = None
    if site:
        if meta is None:
            print("ERROR: --site requires a metadata JSON with site coordinates.")
            print("       Use --meta path/to/metadata.json")
            sys.exit(1)
        dem, site_info = crop_to_site(dem_full, meta, site, SITE_CROP_RADIUS)
        dem_label = f"{site_info.get('description', site)}"
        print(f"  Cropped to site '{site}': {dem.shape}")
    else:
        dem = dem_full

    print(f"  Shape: {dem.shape}  |  Range: {dem.min():.1f} - {dem.max():.1f}")

    # --- Decompose ---
    results = {}
    for name, params, label in DECOMP_METHODS:
        print(f"  Decomposing with {label} ...", end=" ", flush=True)
        trend, residual = run_decomposition(name, dem, params)
        results[name] = {"trend": trend, "residual": residual}
        print(f"residual std={residual.std():.4f}")

    # --- Upsample ---
    upsamp_name = "bicubic"
    print(f"\n  Upsampling residuals with {upsamp_name} ({scale}x) ...")
    for name in results:
        results[name]["upsampled"] = run_upsampling(
            upsamp_name, results[name]["residual"], scale=scale
        )

    # --- Plot ---
    if site and site_info:
        plot_site_comparison(dem, site_info, results, output_path, site)
    else:
        plot_grid(dem, dem_label, results, output_path, scale, upsamp_name)

    print(f"\nFigure saved: {output_path}")

    # --- Text summary ---
    print("\n" + "=" * 60)
    print("  RESIDUALS -- Demo Summary")
    print("=" * 60)
    print(f"  DEM        : {dem_label}")
    print(f"  Shape      : {dem.shape[0]} x {dem.shape[1]}")
    print(f"  Elevation  : {dem.min():.1f} - {dem.max():.1f} ft")
    if site:
        print(f"  Site       : {site}")
    print(f"  Upsampling : {upsamp_name} {scale}x")
    print("-" * 60)
    print(f"  {'Method':<25s} {'Residual Std':>13s} {'Range':>20s}")
    print("-" * 60)

    best_name, best_std = None, 0.0
    for name, _params, label in DECOMP_METHODS:
        res = results[name]["residual"]
        std = res.std()
        rng = f"{res.min():.3f} .. {res.max():.3f}"
        print(f"  {label:<25s} {std:>13.4f} {rng:>20s}")
        if std > best_std:
            best_std = std
            best_name = label

    print("-" * 60)
    print(f"  Highest-contrast residual: {best_name} (std={best_std:.4f})")
    print("  -> Likely best for revealing subtle terrain features.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    # Choose default DEM: prefer Licking (archaeological sites), fall back to Fairfield
    default_dem = LICKING_DEM if LICKING_DEM.exists() else FAIRFIELD_DEM
    default_meta = str(LICKING_META) if LICKING_META.exists() else None

    parser = argparse.ArgumentParser(
        description="RESIDUALS demo: label-free feature enhancement on a DEM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py                                  # full DEM overview
    python demo.py --site hopewell_road             # Great Hopewell Road
    python demo.py --site great_circle              # Great Circle Earthworks
    python demo.py --dem custom.npy --meta custom.json --site my_site
        """,
    )
    parser.add_argument(
        "--dem", type=Path, default=default_dem, help=f"Path to DEM file.  Default: {default_dem}"
    )
    parser.add_argument(
        "--meta", type=str, default=default_meta, help="Path to JSON metadata with site coordinates"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo_output.png"),
        help="Output figure path.  Default: demo_output.png",
    )
    parser.add_argument("--scale", type=int, default=2, help="Upsampling scale factor.  Default: 2")
    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help="Crop to named site (e.g. hopewell_road, great_circle)",
    )
    args = parser.parse_args()
    run_demo(args.dem, args.output, args.scale, args.site, args.meta)


if __name__ == "__main__":
    main()
