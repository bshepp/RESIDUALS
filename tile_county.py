#!/usr/bin/env python3
"""
RESIDUALS: Licking County Grid Tiling

Blankets the entire Licking County LiDAR extent with a regular grid of
DEM tiles, computes professional archaeological visualizations for each
(SVF, openness, RRIM, multi-directional hillshade, tophat residual),
and renders 6-panel PNGs plus a county-wide index map.

Usage:
    python tile_county.py                          # full county (~208 tiles)
    python tile_county.py --rows 5-8               # process rows 5 through 8
    python tile_county.py --rows 5-8 --cols 6-10   # subset of tiles
    python tile_county.py --png-only               # re-render PNGs from cached DEMs
    python tile_county.py --index-only             # just regenerate the index map

Output:
    results/county_tiles/index_map.png       County overview with numbered tile boxes
    results/county_tiles/metadata.json       Grid dimensions and processing status
    results/county_tiles/R03_C07.png              6-panel visualization per tile
    results/county_tiles/R03_C07_dem.npy           Cached DEM array per tile
    results/county_tiles/R03_C07_svf.npy           Cached Sky-View Factor
    results/county_tiles/R03_C07_pos_openness.npy  Cached positive openness
    results/county_tiles/R03_C07_neg_openness.npy  Cached negative openness
"""

import argparse
import gc
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import box

# Trigger method registration
import src.decomposition.methods  # noqa: F401
import src.decomposition.methods_extended  # noqa: F401

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.decomposition.registry import run_decomposition

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tile_county.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("results/county_tiles")


def _resolve_lidar_dir(cli_arg=None):
    """Resolve LiDAR directory from CLI arg, env var, or fail."""
    candidates = [cli_arg, os.environ.get("RESIDUALS_LIDAR_DIR")]
    for raw in candidates:
        if raw:
            p = Path(raw)
            if p.exists():
                return p
    raise FileNotFoundError(
        "LiDAR directory not found. Provide --lidar-dir or set RESIDUALS_LIDAR_DIR"
    )


CRS = "EPSG:3735"

RESOLUTION = 2.5  # ft per pixel
TILE_SIZE = 10560  # 2 miles in ft (same as corridor strip length)

# Known site coordinates (State Plane EPSG:3735)
HR_X, HR_Y = 1979967, 738723  # Hopewell Road segment
GC_X, GC_Y = 1987788, 743540  # Great Circle Earthworks

KNOWN_SITES = {
    "Hopewell Road": (HR_X, HR_Y, "c^"),
    "Great Circle": (GC_X, GC_Y, "gs"),
}

# Tophat parameters (proven best for archaeological linear features)
TOPHAT_PARAMS = {"size": 20, "mode": "white"}


# =========================================================================
# Hillshade
# =========================================================================


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


# =========================================================================
# Multi-directional hillshade
# =========================================================================


def make_multi_hillshade(dem, n_dir=16):
    """
    Average hillshade from n_dir evenly-spaced azimuths.
    Removes the directional bias of single-azimuth hillshading.
    """
    azimuths = np.linspace(0, 360, n_dir, endpoint=False)
    result = np.zeros_like(dem, dtype=np.float64)
    for az in azimuths:
        result += make_hillshade(dem, azimuth=az)
    result /= n_dir
    return result


# =========================================================================
# Sky-View Factor + Openness (Zakšek 2011 / Yokoyama 2002)
# =========================================================================


def compute_svf_openness(dem, n_dir=32, radius=50):
    """
    Compute Sky-View Factor, positive openness, and negative openness.

    Uses an optimized horizon-angle scan: for each of n_dir directions,
    march outward tracking the maximum rise/drop ratio (tan of angle),
    then convert to angles only once per direction at the end.

    References:
        Zakšek, K. et al. (2011). Sky-View Factor as a Relief Visualization
            Technique. Remote Sensing, 3(2), 398-415.
        Yokoyama, R. et al. (2002). Visualizing Topography by Openness.
            Photogrammetric Engineering & Remote Sensing, 68(3), 257-265.

    Args:
        dem: 2D elevation array
        n_dir: number of azimuth directions (default 32)
        radius: search radius in pixels (default 50 = 125 ft at 2.5 ft/px)

    Returns:
        (svf, pos_openness, neg_openness) -- each a 2D float array
        svf: 0-1, higher = more sky visible
        pos_openness: degrees, higher = more exposed/convex
        neg_openness: degrees, higher = more enclosed/concave
    """
    h, w = dem.shape
    pixel_size = RESOLUTION

    # Pad DEM to handle edge lookups -- use float32 to save memory/speed
    padded = np.pad(dem, radius, mode="edge").astype(np.float32)
    dem32 = dem.astype(np.float32)

    # Accumulators across directions
    svf_sin_sum = np.zeros((h, w), dtype=np.float64)
    pos_open_sum = np.zeros((h, w), dtype=np.float64)
    neg_open_sum = np.zeros((h, w), dtype=np.float64)

    angles = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)

    for d_idx, theta in enumerate(angles):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Track max tangent of elevation angle (rise/run) to avoid
        # expensive arctan2 inside the inner loop. We only call arctan
        # once per direction at the end.
        max_tan_elev = np.full((h, w), -1e30, dtype=np.float32)
        max_tan_depr = np.full((h, w), -1e30, dtype=np.float32)

        prev_dj, prev_di = 0, 0
        for r in range(1, radius + 1):
            dj = int(round(r * cos_t))
            di = int(round(r * sin_t))

            if dj == prev_dj and di == prev_di:
                continue
            prev_dj, prev_di = dj, di

            shifted = padded[radius + di : radius + di + h, radius + dj : radius + dj + w]

            # Use 1/distance as multiplier instead of dividing
            inv_dist = 1.0 / (np.sqrt(float(di**2 + dj**2)) * pixel_size)

            # tan(elevation_angle) = (z_shifted - z_self) / distance
            dz = shifted - dem32
            tan_elev = dz * inv_dist

            # tan(depression_angle) = (z_self - z_shifted) / distance
            tan_depr = -dz * inv_dist

            np.maximum(max_tan_elev, tan_elev, out=max_tan_elev)
            np.maximum(max_tan_depr, tan_depr, out=max_tan_depr)

        # Convert from tangent back to angle (once per direction)
        max_elev = np.arctan(max_tan_elev.astype(np.float64))
        max_depr = np.arctan(max_tan_depr.astype(np.float64))

        # SVF: accumulate sin(horizon_angle) where horizon >= 0
        horizon = np.maximum(max_elev, 0.0)
        svf_sin_sum += np.sin(horizon)

        # Positive openness: zenith angle to horizon = 90° - elevation angle
        pos_open_sum += np.pi / 2 - max_elev

        # Negative openness: nadir angle to deepest drop = 90° - depression angle
        neg_open_sum += np.pi / 2 - max_depr

        if (d_idx + 1) % 8 == 0:
            logger.info(f"    SVF progress: {d_idx + 1}/{n_dir} directions")

    svf = 1.0 - svf_sin_sum / n_dir
    pos_openness = np.degrees(pos_open_sum / n_dir)
    neg_openness = np.degrees(neg_open_sum / n_dir)

    return svf, pos_openness, neg_openness


# =========================================================================
# Red Relief Image Map (Chiba et al. 2008)
# =========================================================================


def make_rrim(dem, pos_openness):
    """
    Red Relief Image Map (Chiba et al. 2008).

    Composites slope inclination with positive openness into a red-toned
    image where elevated features (mounds, embankments) glow bright red
    and depressions appear dark.

    Args:
        dem: 2D elevation array
        pos_openness: 2D positive openness array (degrees)

    Returns:
        RGB array (h, w, 3) with values in 0-1
    """
    dy, dx = np.gradient(dem.astype(np.float64))
    slope = np.arctan(np.sqrt(dx**2 + dy**2))  # slope angle in radians

    # Normalize slope to 0-1
    slope_norm = slope / (np.pi / 4)  # 45° = 1.0
    slope_norm = np.clip(slope_norm, 0, 1)

    # Normalize and invert openness: low openness (enclosed) -> high value
    o_min, o_max = pos_openness.min(), pos_openness.max()
    if o_max > o_min:
        open_norm = (pos_openness - o_min) / (o_max - o_min)
    else:
        open_norm = np.full_like(pos_openness, 0.5)

    # RRIM composite:
    # - Slope drives the red intensity (steep edges glow)
    # - Inverted openness adds red to enclosed/elevated areas
    # - Openness controls overall brightness (open = brighter, enclosed = darker)
    red_intensity = np.clip(0.6 * slope_norm + 0.4 * (1.0 - open_norm), 0, 1)
    brightness = np.clip(0.3 + 0.7 * open_norm, 0, 1)

    rgb = np.zeros((*dem.shape, 3), dtype=np.float64)
    rgb[:, :, 0] = np.clip(red_intensity + brightness * 0.3, 0, 1)  # Red
    rgb[:, :, 1] = brightness * 0.35  # Green (subdued)
    rgb[:, :, 2] = brightness * 0.25  # Blue (very subdued)

    return np.clip(rgb, 0, 1)


# =========================================================================
# Simple Local Relief Model (Hesse 2010)
# =========================================================================


def compute_slrm(dem, radius=20):
    """
    Simple Local Relief Model: subtract a circular moving-average surface.

    The professional standard for archaeological feature extraction.
    Conceptually identical to TopHat (remove trend, keep residual) but uses
    a mean filter for the background instead of morphological opening.

    Args:
        dem: 2D elevation array
        radius: averaging radius in pixels (default 20 = 50 ft at 2.5 ft/px,
                matching TopHat's structuring element size for fair comparison)

    Returns:
        2D residual array (DEM minus local mean surface)
    """
    from scipy.ndimage import uniform_filter

    # Use a square uniform filter with width = 2*radius+1 as a fast
    # approximation of circular averaging (standard in SLRM implementations)
    kernel_size = 2 * radius + 1
    trend = uniform_filter(dem.astype(np.float64), size=kernel_size, mode="nearest")
    return dem - trend


# =========================================================================
# Topographic Position Index (Weiss 2001)
# =========================================================================


def compute_tpi(dem, radius=20):
    """
    Topographic Position Index: how much higher/lower each pixel is compared
    to the mean of its annular neighborhood.

    Positive = locally elevated (mound, ridge)
    Negative = locally depressed (ditch, pit)
    Near zero = slope or flat ground

    Args:
        dem: 2D elevation array
        radius: neighborhood radius in pixels (default 20 = 50 ft at 2.5 ft/px)

    Returns:
        2D TPI array (in same units as DEM, typically feet)
    """
    from scipy.ndimage import uniform_filter

    kernel_size = 2 * radius + 1
    local_mean = uniform_filter(dem.astype(np.float64), size=kernel_size, mode="nearest")
    return dem - local_mean


# =========================================================================
# Grid computation
# =========================================================================


def compute_grid(tile_gdf: gpd.GeoDataFrame) -> list:
    """
    Compute a regular grid of tiles covering the county LiDAR extent.

    Returns:
        List of tile dicts with keys: row, col, tile_id, x_min, x_max,
        y_min, y_max, n_tiles (number of LAS tiles that intersect).
    """
    bounds = tile_gdf.total_bounds  # (x_min, y_min, x_max, y_max)
    county_x_min, county_y_min, county_x_max, county_y_max = bounds

    logger.info(
        f"County extent: X=[{county_x_min:.0f}, {county_x_max:.0f}], "
        f"Y=[{county_y_min:.0f}, {county_y_max:.0f}]"
    )
    logger.info(
        f"County size: {county_x_max - county_x_min:.0f} x {county_y_max - county_y_min:.0f} ft"
    )

    n_cols = int(np.ceil((county_x_max - county_x_min) / TILE_SIZE))
    n_rows = int(np.ceil((county_y_max - county_y_min) / TILE_SIZE))

    logger.info(
        f"Grid: {n_rows} rows x {n_cols} cols = {n_rows * n_cols} tiles "
        f"({TILE_SIZE} ft / {TILE_SIZE / 5280:.1f} mi per tile)"
    )

    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            x_min = county_x_min + c * TILE_SIZE
            x_max = x_min + TILE_SIZE
            y_min = county_y_min + r * TILE_SIZE
            y_max = y_min + TILE_SIZE

            # Check how many LAS tiles intersect
            tile_box = box(x_min, y_min, x_max, y_max)
            n_las = len(tile_gdf[tile_gdf.intersects(tile_box)])

            tiles.append(
                {
                    "row": r,
                    "col": c,
                    "tile_id": f"R{r:02d}_C{c:02d}",
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "n_tiles": n_las,
                }
            )

    return tiles, n_rows, n_cols, bounds


# =========================================================================
# DEM generation (reused from trace_road.py)
# =========================================================================


def generate_tile_dem(tile: dict, tile_gdf: gpd.GeoDataFrame, lidar_dir: Path = None):
    """
    Generate a DEM array for one grid tile.

    Returns:
        dem (2D array) or None if not enough data.
    """
    import laspy

    x_min, x_max = tile["x_min"], tile["x_max"]
    y_min, y_max = tile["y_min"], tile["y_max"]

    tile_box = box(x_min, y_min, x_max, y_max)
    matching = tile_gdf[tile_gdf.intersects(tile_box)]

    all_px, all_py, all_pz = [], [], []
    for _, row in matching.iterrows():
        tile_name = row["TileName"]
        las_path = lidar_dir / f"{tile_name}.las"
        if not las_path.exists():
            continue

        las = laspy.read(las_path)

        if hasattr(las, "classification"):
            mask = las.classification == 2
            if mask.sum() == 0:
                mask = np.ones(len(las.x), dtype=bool)
        else:
            mask = np.ones(len(las.x), dtype=bool)

        x = np.array(las.x[mask])
        y = np.array(las.y[mask])
        z = np.array(las.z[mask])
        in_bbox = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        all_px.extend(x[in_bbox])
        all_py.extend(y[in_bbox])
        all_pz.extend(z[in_bbox])

    if len(all_px) < 100:
        return None

    px = np.array(all_px)
    py = np.array(all_py)
    pz = np.array(all_pz)

    xi = np.arange(x_min, x_max, RESOLUTION)
    yi = np.arange(y_min, y_max, RESOLUTION)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((px, py), pz, (Xi, Yi), method="linear")

    nan_count = np.isnan(Zi).sum()
    if nan_count > 0:
        Zi_nearest = griddata((px, py), pz, (Xi, Yi), method="nearest")
        Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)

    return Zi


# =========================================================================
# Render per-tile PNG
# =========================================================================


def render_tile_png(
    dem, residual, svf, pos_openness, neg_openness, tile_id, output_path, sites_in_tile=None
):
    """
    Render a 9-panel archaeological visualization for one tile.

    Layout (3 rows x 3 cols):
        Row 0 (Visualization):  Multi-Dir Hillshade | Sky-View Factor  | Red Relief (RRIM)
        Row 1 (Context):        Positive Openness   | Negative Openness| Slope
        Row 2 (Extraction):     SLRM                | TPI              | Tophat Residual

    Row 2 is the head-to-head comparison: professional standard methods
    (SLRM, TPI) versus morphological TopHat, all at the same scale.

    Args:
        dem: 2D elevation array
        residual: 2D tophat residual array
        svf: 2D Sky-View Factor array (0-1)
        pos_openness: 2D positive openness array (degrees)
        neg_openness: 2D negative openness array (degrees)
        tile_id: string like 'R03_C07'
        output_path: Path for the PNG
        sites_in_tile: list of (name, px_x, px_y) for sites within this tile
    """
    # Compute derived visualizations
    mdhs = make_multi_hillshade(dem)
    rrim = make_rrim(dem, pos_openness)
    slrm = compute_slrm(dem, radius=20)
    tpi = compute_tpi(dem, radius=20)

    # Fine-scale tophat: size=3 → disk radius 3px = 7.5 ft radius ≈ 15 ft
    # diameter, targeting individual embankment ridges / mound edges
    try:
        _trend_fine, residual_fine = run_decomposition("tophat", dem, {"size": 3, "mode": "white"})
    except Exception:
        residual_fine = residual  # fallback

    # Slope for context panel
    dy, dx = np.gradient(dem.astype(np.float64))
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    fig, axes = plt.subplots(3, 4, figsize=(48, 36))

    def mark_sites(ax):
        if sites_in_tile:
            for _name, px_x, px_y in sites_in_tile:
                ax.plot(px_x, px_y, "+", color="lime", markersize=12, markeredgewidth=1.5)

    def strip_axes(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared vmax for SLRM/TPI fair comparison
    abs_slrm = np.abs(slrm)
    abs_tpi = np.abs(tpi)
    shared_vmax = max(
        np.percentile(abs_slrm, 99),
        np.percentile(abs_tpi, 99),
        1e-6,
    )

    # =================================================================
    # Row 0: Visualization methods
    # =================================================================

    # --- Multi-directional hillshade ---
    ax = axes[0, 0]
    ax.imshow(mdhs, cmap="gray", origin="lower", aspect="equal", interpolation="none")
    ax.set_title("Multi-Dir Hillshade (16 dir)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Visualization", fontsize=13, fontweight="bold", rotation=90, labelpad=10)
    strip_axes(ax)
    mark_sites(ax)

    # --- Sky-View Factor ---
    ax = axes[0, 1]
    ax.imshow(
        svf, cmap="gray", origin="lower", aspect="equal", interpolation="none", vmin=0.7, vmax=1.0
    )
    ax.set_title("Sky-View Factor (SVF)", fontsize=12, fontweight="bold")
    strip_axes(ax)
    mark_sites(ax)

    # --- Red Relief Image Map ---
    ax = axes[0, 2]
    ax.imshow(rrim, origin="lower", aspect="equal", interpolation="none")
    ax.set_title("Red Relief Image Map (RRIM)", fontsize=12, fontweight="bold")
    strip_axes(ax)
    mark_sites(ax)

    # --- Slope ---
    ax = axes[0, 3]
    ax.imshow(
        slope,
        cmap="gray",
        origin="lower",
        aspect="equal",
        interpolation="none",
        vmin=0,
        vmax=np.percentile(slope, 99),
    )
    ax.set_title("Slope (degrees)", fontsize=12, fontweight="bold")
    strip_axes(ax)
    mark_sites(ax)

    # =================================================================
    # Row 1: Terrain shape (openness)
    # =================================================================

    # --- Positive Openness ---
    ax = axes[1, 0]
    p01 = np.percentile(pos_openness, 1)
    p99 = np.percentile(pos_openness, 99)
    ax.imshow(
        pos_openness,
        cmap="gray",
        origin="lower",
        aspect="equal",
        interpolation="none",
        vmin=p01,
        vmax=p99,
    )
    ax.set_title("Positive Openness (Yokoyama 2002)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Terrain Shape", fontsize=13, fontweight="bold", rotation=90, labelpad=10)
    strip_axes(ax)
    mark_sites(ax)

    # --- Negative Openness ---
    ax = axes[1, 1]
    p01 = np.percentile(neg_openness, 1)
    p99 = np.percentile(neg_openness, 99)
    ax.imshow(
        neg_openness,
        cmap="gray",
        origin="lower",
        aspect="equal",
        interpolation="none",
        vmin=p01,
        vmax=p99,
    )
    ax.set_title("Negative Openness (Yokoyama 2002)", fontsize=12, fontweight="bold")
    strip_axes(ax)
    mark_sites(ax)

    # --- SLRM ---
    ax = axes[1, 2]
    im_slrm = ax.imshow(
        slrm,
        cmap="hot",
        origin="lower",
        aspect="equal",
        vmin=0,
        vmax=shared_vmax,
        interpolation="none",
    )
    ax.set_title("SLRM (Hesse 2010) r=50ft", fontsize=12, fontweight="bold")
    strip_axes(ax)
    mark_sites(ax)
    plt.colorbar(im_slrm, ax=ax, fraction=0.046, pad=0.04, label="ft")

    # --- TPI ---
    ax = axes[1, 3]
    im_tpi = ax.imshow(
        tpi,
        cmap="hot",
        origin="lower",
        aspect="equal",
        vmin=0,
        vmax=shared_vmax,
        interpolation="none",
    )
    ax.set_title("TPI (Weiss 2001) r=50ft", fontsize=12, fontweight="bold")
    strip_axes(ax)
    mark_sites(ax)
    plt.colorbar(im_tpi, ax=ax, fraction=0.046, pad=0.04, label="ft")

    # =================================================================
    # Row 2: TopHat at two scales -- road width vs embankment width
    # =================================================================

    # --- Tophat r=50ft (road-scale) ---
    ax = axes[2, 0]
    abs_tophat = np.abs(residual)
    th_p99 = np.percentile(abs_tophat, 99)
    th_vmax = max(th_p99, 1e-6)
    im_th = ax.imshow(
        residual,
        cmap="hot",
        origin="lower",
        aspect="equal",
        vmin=0,
        vmax=th_vmax,
        interpolation="none",
    )
    ax.set_title("TopHat r=50ft (road width)", fontsize=12, fontweight="bold")
    ax.set_ylabel("TopHat Extraction", fontsize=13, fontweight="bold", rotation=90, labelpad=10)
    strip_axes(ax)
    mark_sites(ax)
    plt.colorbar(im_th, ax=ax, fraction=0.046, pad=0.04, label="ft")

    # --- Tophat r=7.5ft (embankment-scale, ~1m features) ---
    ax = axes[2, 1]
    abs_fine = np.abs(residual_fine)
    fine_p99 = np.percentile(abs_fine, 99)
    fine_vmax = max(fine_p99, 1e-6)
    im_fine = ax.imshow(
        residual_fine,
        cmap="hot",
        origin="lower",
        aspect="equal",
        vmin=0,
        vmax=fine_vmax,
        interpolation="none",
    )
    ax.set_title("TopHat r=7.5ft / 15ft dia (~1m features)", fontsize=12, fontweight="bold")
    strip_axes(ax)
    mark_sites(ax)
    plt.colorbar(im_fine, ax=ax, fraction=0.046, pad=0.04, label="ft")

    # --- Tophat r=25ft (intermediate) ---
    try:
        _trend_mid, residual_mid = run_decomposition("tophat", dem, {"size": 10, "mode": "white"})
    except Exception:
        residual_mid = residual
    ax = axes[2, 2]
    abs_mid = np.abs(residual_mid)
    mid_p99 = np.percentile(abs_mid, 99)
    mid_vmax = max(mid_p99, 1e-6)
    im_mid = ax.imshow(
        residual_mid,
        cmap="hot",
        origin="lower",
        aspect="equal",
        vmin=0,
        vmax=mid_vmax,
        interpolation="none",
    )
    ax.set_title("TopHat r=25ft / 50ft dia (~5m features)", fontsize=12, fontweight="bold")
    strip_axes(ax)
    mark_sites(ax)
    plt.colorbar(im_mid, ax=ax, fraction=0.046, pad=0.04, label="ft")

    # --- Tophat r=100ft (landscape-scale) ---
    try:
        _trend_wide, residual_wide = run_decomposition("tophat", dem, {"size": 40, "mode": "white"})
    except Exception:
        residual_wide = residual
    ax = axes[2, 3]
    abs_wide = np.abs(residual_wide)
    wide_p99 = np.percentile(abs_wide, 99)
    wide_vmax = max(wide_p99, 1e-6)
    im_wide = ax.imshow(
        residual_wide,
        cmap="hot",
        origin="lower",
        aspect="equal",
        vmin=0,
        vmax=wide_vmax,
        interpolation="none",
    )
    ax.set_title("TopHat r=100ft / 200ft dia (~30m features)", fontsize=12, fontweight="bold")
    strip_axes(ax)
    mark_sites(ax)
    plt.colorbar(im_wide, ax=ax, fraction=0.046, pad=0.04, label="ft")

    fig.suptitle(
        f"RESIDUALS: County Tile {tile_id}\n"
        f"Row 1: Professional viz | Row 2: SLRM/TPI vs context | "
        f"Row 3: TopHat multi-scale (7.5ft to 100ft radius)",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# =========================================================================
# Index map
# =========================================================================


def generate_index_map(
    all_tiles: list,
    n_rows: int,
    n_cols: int,
    county_bounds: tuple,
    output_path: Path,
    processed_tiles: set = None,
):
    """
    Generate a county-wide index map with numbered tile boxes.

    Args:
        all_tiles: list of tile dicts
        n_rows, n_cols: grid dimensions
        county_bounds: (x_min, y_min, x_max, y_max) of county LiDAR extent
        output_path: where to save the PNG
        processed_tiles: set of tile_id strings that have been processed
    """
    if processed_tiles is None:
        processed_tiles = set()

    logger.info("Generating index map ...")

    cx_min, cy_min, cx_max, cy_max = county_bounds
    width_ft = cx_max - cx_min
    height_ft = cy_max - cy_min
    aspect = width_ft / height_ft

    fig_h = 16
    fig_w = fig_h * aspect
    fig_w = max(12, min(fig_w, 28))

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # Draw tile boxes
    for tile in all_tiles:
        if tile["n_tiles"] == 0:
            continue  # no LAS data for this tile, skip

        x_min, y_min = tile["x_min"], tile["y_min"]
        w = tile["x_max"] - tile["x_min"]
        h = tile["y_max"] - tile["y_min"]
        tid = tile["tile_id"]

        is_processed = tid in processed_tiles

        rect = mpatches.Rectangle(
            (x_min, y_min),
            w,
            h,
            linewidth=0.8,
            edgecolor="green" if is_processed else "#888888",
            facecolor="#e0ffe0" if is_processed else "#f0f0f0",
            alpha=0.6 if is_processed else 0.3,
        )
        ax.add_patch(rect)

        # Label tile
        cx = x_min + w / 2
        cy = y_min + h / 2
        fontsize = max(4, min(7, 200 / max(n_rows, n_cols)))
        ax.text(
            cx,
            cy,
            tid,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="darkgreen" if is_processed else "#999999",
        )

    # Mark known archaeological sites
    for name, (sx, sy, marker) in KNOWN_SITES.items():
        ax.plot(sx, sy, marker, markersize=14, markeredgewidth=2, label=name, zorder=10)

    ax.set_xlim(cx_min - 2000, cx_max + 2000)
    ax.set_ylim(cy_min - 2000, cy_max + 2000)
    ax.set_aspect("equal")
    ax.set_xlabel("State Plane X (ft)", fontsize=11)
    ax.set_ylabel("State Plane Y (ft)", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")

    ax.set_title(
        f"RESIDUALS: Licking County Tile Index\n"
        f"{n_rows} rows x {n_cols} cols = {n_rows * n_cols} tiles "
        f"({TILE_SIZE / 5280:.1f} mi each)  |  "
        f"{len(processed_tiles)} processed",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Index map saved: {output_path}")


# =========================================================================
# CLI helpers
# =========================================================================


def parse_range(s: str) -> tuple:
    """Parse a range string like '5-8' into (5, 8) inclusive."""
    if "-" in s:
        parts = s.split("-", 1)
        return int(parts[0]), int(parts[1])
    else:
        v = int(s)
        return v, v


def sites_in_tile(tile: dict) -> list:
    """Return list of (name, px_x, px_y) for known sites within the tile."""
    result = []
    for name, (sx, sy, _marker) in KNOWN_SITES.items():
        if tile["x_min"] <= sx <= tile["x_max"] and tile["y_min"] <= sy <= tile["y_max"]:
            # Convert to pixel coordinates within the tile
            px_x = (sx - tile["x_min"]) / RESOLUTION
            px_y = (sy - tile["y_min"]) / RESOLUTION
            result.append((name, px_x, px_y))
    return result


# =========================================================================
# Safe I/O helpers
# =========================================================================


def safe_save_npy(path: Path, arr: np.ndarray):
    """Atomic .npy write: save to temp file then rename to prevent corruption."""
    tmp = path.parent / (path.stem + "_tmp.npy")
    np.save(tmp, arr)
    os.replace(str(tmp), str(path))


def safe_load_npy(path: Path):
    """Load .npy with corruption handling. Returns None and deletes file on error."""
    try:
        return np.load(path)
    except Exception as e:
        logger.warning(f"  Corrupt cache {path.name}: {e} -- deleting")
        try:
            path.unlink()
        except OSError:
            pass
        return None


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="RESIDUALS: Licking County grid tiling")
    parser.add_argument(
        "--rows", type=str, default=None, help='Row range to process, e.g. "5-8" (0-indexed)'
    )
    parser.add_argument(
        "--cols", type=str, default=None, help='Column range to process, e.g. "6-10" (0-indexed)'
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Only re-render PNGs from cached DEMs (skip DEM generation)",
    )
    parser.add_argument("--index-only", action="store_true", help="Only regenerate the index map")
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--lidar-dir",
        type=str,
        default=None,
        help="LiDAR tile directory (or set RESIDUALS_LIDAR_DIR)",
    )
    parser.add_argument("--tile-index", type=str, default=None, help="Path to tile index shapefile")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    LIDAR_DIR = _resolve_lidar_dir(args.lidar_dir)
    TILE_INDEX = (
        Path(args.tile_index)
        if args.tile_index
        else (LIDAR_DIR / "_LIC_1250_Tile_Index" / "Tile_Index.shp")
    )

    logger.info("=" * 70)
    logger.info("RESIDUALS: Licking County Grid Tiling")
    logger.info("=" * 70)

    # Load tile index
    logger.info(f"Loading tile index: {TILE_INDEX}")
    tile_gdf = gpd.read_file(TILE_INDEX)
    logger.info(f"  {len(tile_gdf)} LAS tiles in index")

    # Compute grid
    all_tiles, n_rows, n_cols, county_bounds = compute_grid(tile_gdf)
    tiles_with_data = [t for t in all_tiles if t["n_tiles"] > 0]
    logger.info(
        f"  {len(tiles_with_data)} tiles have LAS data (of {len(all_tiles)} total grid cells)"
    )

    # --index-only: just regenerate the index map and exit
    if args.index_only:
        processed = set()
        for t in all_tiles:
            png_path = output_dir / f"{t['tile_id']}.png"
            if png_path.exists():
                processed.add(t["tile_id"])
        generate_index_map(
            all_tiles, n_rows, n_cols, county_bounds, output_dir / "index_map.png", processed
        )
        return

    # Filter tiles by row/col range if specified
    row_min, row_max = 0, n_rows - 1
    col_min, col_max = 0, n_cols - 1
    if args.rows:
        row_min, row_max = parse_range(args.rows)
    if args.cols:
        col_min, col_max = parse_range(args.cols)

    work_tiles = [
        t
        for t in tiles_with_data
        if row_min <= t["row"] <= row_max and col_min <= t["col"] <= col_max
    ]
    logger.info(
        f"Processing {len(work_tiles)} tiles (rows {row_min}-{row_max}, cols {col_min}-{col_max})"
    )

    # Process tiles
    t_start = time.time()
    processed_ids = set()

    n_failed = 0

    for i, tile in enumerate(work_tiles):
        tid = tile["tile_id"]
        t_tile_start = time.time()

        logger.info(f"\n--- {tid} ({i + 1}/{len(work_tiles)}) LAS tiles={tile['n_tiles']} ---")

        dem_cache = output_dir / f"{tid}_dem.npy"
        svf_cache = output_dir / f"{tid}_svf.npy"
        pos_open_cache = output_dir / f"{tid}_pos_openness.npy"
        neg_open_cache = output_dir / f"{tid}_neg_openness.npy"
        png_path = output_dir / f"{tid}.png"

        # Skip if PNG already exists (unless --png-only forces re-render)
        if png_path.exists() and not args.png_only:
            logger.info("  PNG exists, skipping")
            processed_ids.add(tid)
            continue

        try:
            # --- Load or generate DEM ---
            dem = None
            if dem_cache.exists():
                dem = safe_load_npy(dem_cache)
                if dem is not None:
                    logger.info(f"  DEM (cached): {dem.shape}")

            if dem is None and args.png_only:
                logger.info("  No cached DEM and --png-only set, skipping")
                continue

            if dem is None:
                dem = generate_tile_dem(tile, tile_gdf, lidar_dir=LIDAR_DIR)
                if dem is None:
                    logger.warning("  Not enough LAS points, skipping")
                    continue
                safe_save_npy(dem_cache, dem)
                logger.info(
                    f"  DEM: {dem.shape} ({dem.size:,} px), elev=[{dem.min():.0f}, {dem.max():.0f}]"
                )

            # --- Load or compute SVF + openness ---
            svf, pos_openness, neg_openness = None, None, None
            if svf_cache.exists() and pos_open_cache.exists() and neg_open_cache.exists():
                svf = safe_load_npy(svf_cache)
                pos_openness = safe_load_npy(pos_open_cache)
                neg_openness = safe_load_npy(neg_open_cache)
                if svf is not None and pos_openness is not None and neg_openness is not None:
                    logger.info("  SVF/openness (cached)")

            if svf is None or pos_openness is None or neg_openness is None:
                t0 = time.time()
                svf, pos_openness, neg_openness = compute_svf_openness(dem)
                t_svf = time.time() - t0
                logger.info(f"  SVF/openness: {t_svf:.1f}s  SVF=[{svf.min():.3f}, {svf.max():.3f}]")
                safe_save_npy(svf_cache, svf.astype(np.float32))
                safe_save_npy(pos_open_cache, pos_openness.astype(np.float32))
                safe_save_npy(neg_open_cache, neg_openness.astype(np.float32))

            # --- Compute tophat residual ---
            t0 = time.time()
            _trend, residual = run_decomposition("tophat", dem, TOPHAT_PARAMS)
            logger.info(f"  Tophat: {time.time() - t0:.1f}s")

            # --- Render multi-panel PNG ---
            site_marks = sites_in_tile(tile)
            render_tile_png(
                dem, residual, svf, pos_openness, neg_openness, tid, png_path, site_marks
            )
            processed_ids.add(tid)

        except Exception as e:
            n_failed += 1
            logger.error(f"  FAILED: {e}")
            logger.error(f"  Skipping {tid} and continuing ({n_failed} failures so far)")
            # Clean up partial PNG if it was written
            if png_path.exists():
                try:
                    png_path.unlink()
                except OSError:
                    pass
        finally:
            # Free large arrays to prevent memory creep over 200 tiles
            for var in ("dem", "svf", "pos_openness", "neg_openness", "residual", "_trend"):
                if var in locals():
                    try:
                        del locals()[var]
                    except Exception:
                        pass
            dem = svf = pos_openness = neg_openness = residual = _trend = None
            gc.collect()

        elapsed = time.time() - t_start
        remaining_tiles = len(work_tiles) - (i + 1)
        per_tile = elapsed / (i + 1)
        est_remaining = timedelta(seconds=remaining_tiles * per_tile)
        logger.info(
            f"  Tile time: {time.time() - t_tile_start:.1f}s  |  "
            f"Elapsed: {timedelta(seconds=int(elapsed))}  |  "
            f"Est. remaining: {est_remaining}"
        )

    if n_failed > 0:
        logger.warning(f"\n{n_failed} tiles failed during processing")

    # Also count any previously processed tiles for the index map
    all_processed = set()
    for t in all_tiles:
        png_path = output_dir / f"{t['tile_id']}.png"
        if png_path.exists():
            all_processed.add(t["tile_id"])

    # Generate index map
    generate_index_map(
        all_tiles, n_rows, n_cols, county_bounds, output_dir / "index_map.png", all_processed
    )

    # Save metadata
    meta = {
        "grid_rows": n_rows,
        "grid_cols": n_cols,
        "tile_size_ft": TILE_SIZE,
        "resolution_ft": RESOLUTION,
        "total_tiles": len(all_tiles),
        "tiles_with_data": len(tiles_with_data),
        "tiles_processed": len(all_processed),
        "county_bounds": {
            "x_min": float(county_bounds[0]),
            "y_min": float(county_bounds[1]),
            "x_max": float(county_bounds[2]),
            "y_max": float(county_bounds[3]),
        },
        "known_sites": {
            name: {"x": float(xy[0]), "y": float(xy[1])}
            for name, xy in [("Hopewell Road", (HR_X, HR_Y)), ("Great Circle", (GC_X, GC_Y))]
        },
        "crs": CRS,
        "processed_tile_ids": sorted(all_processed),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata: {meta_path}")

    total_elapsed = time.time() - t_start
    logger.info(
        f"\nDone. {len(all_processed)} tiles processed in {timedelta(seconds=int(total_elapsed))}"
    )


if __name__ == "__main__":
    main()
