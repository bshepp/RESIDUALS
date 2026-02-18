#!/usr/bin/env python3
"""
RESIDUALS: Hopewell Road Corridor Trace

Extends the road detection from the known segment across all of Licking County
by generating corridor DEM strips, applying multi-method residual analysis,
building a consensus confidence heatmap, and tracing the road centerline.

Outputs:
    results/hopewell_trace/corridor_residual_tophat.tif   GeoTIFF residual
    results/hopewell_trace/corridor_confidence.tif         Confidence heatmap
    results/hopewell_trace/road_centerline.geojson         Detected centerline
    results/hopewell_trace/corridor_overview.png           Summary figure
    results/hopewell_trace/metadata.json                   Processing metadata

Usage:
    python trace_road.py                    # full run
    python trace_road.py --max-strips 5     # quick test with 5 strips per direction
    python trace_road.py --skip-dem-gen     # reuse existing strip DEMs
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.merge import merge as rasterio_merge
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from shapely.geometry import box, LineString, mapping
from pyproj import Transformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trace_road.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LIDAR_DIR = Path(r'F:\science-projects\lidar_super_rez\data\licking_2015')
TILE_INDEX = LIDAR_DIR / '_LIC_1250_Tile_Index' / 'Tile_Index.shp'
EXISTING_DEM = Path('data/test_dems/licking_hopewell_2.5ft.npy')
EXISTING_META = Path('data/test_dems/licking_hopewell_2.5ft.json')
OUTPUT_DIR = Path('results/hopewell_trace')
CRS = 'EPSG:3735'

RESOLUTION = 2.5        # ft per pixel
STRIP_LENGTH = 10560    # 2 miles in ft
STRIP_OVERLAP = 500     # ft overlap between strips
CORRIDOR_WIDTH = 4000   # ft total width

# Known site coordinates (State Plane, from metadata)
HR_X, HR_Y = 1979967, 738723   # Hopewell Road segment
GC_X, GC_Y = 1987788, 743540   # Great Circle

# Methods for linear feature detection
RESIDUAL_METHODS = [
    ('tophat',              {'size': 20, 'mode': 'white'}),
    ('morphological_rect',  {'operation': 'opening', 'width': 20, 'height': 5}),
    ('dog',                 {'sigma_low': 2, 'sigma_high': 10}),
]


# =========================================================================
# Step 1: Detect road bearing from existing data
# =========================================================================

def detect_bearing(dem_path: Path, meta_path: Path) -> float:
    """
    Detect the dominant linear feature bearing from the existing DEM
    using residual analysis + Meijering ridge filter + Hough transform.

    Returns:
        Bearing in radians (angle of road axis in State Plane coordinates,
        measured counter-clockwise from the +X axis).
    """
    from skimage.filters import meijering
    from skimage.transform import probabilistic_hough_line
    from skimage.feature import canny

    logger.info("Step 1: Detecting road bearing from existing data ...")

    # Load DEM and crop to Hopewell Road area
    dem_full = np.load(dem_path)
    with open(meta_path) as f:
        meta = json.load(f)

    site = meta['sites']['hopewell_road']
    cx, cy = int(site['pixel_x']), int(site['pixel_y'])
    r = 400
    h, w = dem_full.shape
    dem = dem_full[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)]
    logger.info(f"  Cropped DEM around Hopewell Road: {dem.shape}")

    # Compute tophat residual (best performer from demo)
    from src.decomposition.registry import run_decomposition
    import src.decomposition.methods           # noqa: F401
    import src.decomposition.methods_extended  # noqa: F401

    _, residual = run_decomposition('tophat', dem, {'size': 15, 'mode': 'white'})

    # Apply Meijering ridge filter to enhance linear features
    ridge = meijering(residual, sigmas=range(1, 5), black_ridges=False)
    ridge = ridge / (ridge.max() + 1e-10)
    logger.info(f"  Ridge filter: max={ridge.max():.4f}")

    # Edge detection + Hough for dominant angle
    edges = canny(ridge, sigma=2, low_threshold=0.1, high_threshold=0.3)
    lines = probabilistic_hough_line(
        edges, threshold=20, line_length=80, line_gap=15
    )

    if not lines:
        logger.warning("  No lines detected, falling back to site-to-site bearing")
        dx = HR_X - GC_X
        dy = HR_Y - GC_Y
        return np.arctan2(dy, dx)

    # Compute angle of each line segment, weight by length
    angles = []
    weights = []
    for (x0, y0), (x1, y1) in lines:
        length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        angle = np.arctan2(y1-y0, x1-x0)
        angles.append(angle)
        weights.append(length)

    # Find dominant angle using weighted circular mean
    # First, double angles to handle 180-degree ambiguity
    angles = np.array(angles)
    weights = np.array(weights)
    double_angles = 2 * angles
    mean_double = np.arctan2(
        np.sum(weights * np.sin(double_angles)),
        np.sum(weights * np.cos(double_angles))
    )
    bearing_pixel = mean_double / 2  # back to single angle

    # Convert pixel bearing to State Plane bearing
    # In the DEM, +x = east (+X in State Plane), +y = north (+Y in State Plane)
    # since origin='lower' and axes align
    bearing_sp = bearing_pixel
    bearing_deg = np.degrees(bearing_sp)
    logger.info(f"  Detected bearing: {bearing_deg:.1f} degrees from +X axis")
    logger.info(f"  ({len(lines)} Hough line segments detected)")

    return bearing_sp


# =========================================================================
# Step 2: Generate corridor geometry (strip bounding boxes)
# =========================================================================

def generate_corridor_strips(
    origin_x: float,
    origin_y: float,
    bearing: float,
    max_strips_per_dir: int = 50,
) -> list:
    """
    Generate a list of strip definitions along the corridor.

    Each strip is a dict with:
        center_x, center_y: strip center in State Plane
        x_min, x_max, y_min, y_max: axis-aligned bounding box
        strip_id: sequential index
        direction: 'forward' or 'backward'

    Returns list of strip dicts.
    """
    logger.info("Step 2: Computing corridor strip geometry ...")

    # Load tile index to find county bounds
    tiles = gpd.read_file(TILE_INDEX)
    county_bounds = tiles.total_bounds  # (x_min, y_min, x_max, y_max)
    logger.info(f"  County bounds: X=[{county_bounds[0]:.0f}, {county_bounds[2]:.0f}], "
                f"Y=[{county_bounds[1]:.0f}, {county_bounds[3]:.0f}]")

    # Direction vectors
    dx = np.cos(bearing)
    dy = np.sin(bearing)
    # Perpendicular vector
    px = -dy
    py = dx

    step = STRIP_LENGTH - STRIP_OVERLAP
    half_w = CORRIDOR_WIDTH / 2

    strips = []

    for direction_sign, direction_name in [(1, 'forward'), (-1, 'backward')]:
        for i in range(max_strips_per_dir):
            dist = (i + 0.5) * step * direction_sign
            cx = origin_x + dx * dist
            cy = origin_y + dy * dist

            # Four corners of the rotated strip
            half_len = STRIP_LENGTH / 2
            corners = []
            for sl, sw in [(-half_len, -half_w), (-half_len, half_w),
                           (half_len, -half_w), (half_len, half_w)]:
                corners.append((
                    cx + dx * sl + px * sw,
                    cy + dy * sl + py * sw,
                ))

            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Check if strip center is within county bounds (with margin)
            margin = 2000
            if (cx < county_bounds[0] + margin or cx > county_bounds[2] - margin or
                cy < county_bounds[1] + margin or cy > county_bounds[3] - margin):
                logger.info(f"  {direction_name} strip {i}: outside county bounds, stopping")
                break

            # Check if tiles exist in this area
            strip_box = box(x_min, y_min, x_max, y_max)
            matching = tiles[tiles.intersects(strip_box)]
            if len(matching) == 0:
                logger.info(f"  {direction_name} strip {i}: no tiles, stopping")
                break

            strips.append({
                'strip_id': len(strips),
                'direction': direction_name,
                'dir_index': i,
                'center_x': cx,
                'center_y': cy,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'n_tiles': len(matching),
            })

    logger.info(f"  Generated {len(strips)} corridor strips")
    return strips


# =========================================================================
# Step 3: Generate strip DEM from LAS tiles
# =========================================================================

def generate_strip_dem(strip: dict, tile_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Generate a DEM array for one corridor strip.

    Returns:
        (dem, xi, yi) where dem is the 2D elevation array and
        xi, yi are the coordinate arrays for the grid axes.
    """
    import laspy

    x_min = strip['x_min']
    x_max = strip['x_max']
    y_min = strip['y_min']
    y_max = strip['y_max']

    strip_box = box(x_min, y_min, x_max, y_max)
    matching = tile_gdf[tile_gdf.intersects(strip_box)]

    # Load ground points
    all_px, all_py, all_pz = [], [], []
    for _, row in matching.iterrows():
        tile_name = row['TileName']
        las_path = LIDAR_DIR / f'{tile_name}.las'
        if not las_path.exists():
            continue

        las = laspy.read(las_path)

        if hasattr(las, 'classification'):
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
        return None, None, None

    px = np.array(all_px)
    py = np.array(all_py)
    pz = np.array(all_pz)

    xi = np.arange(x_min, x_max, RESOLUTION)
    yi = np.arange(y_min, y_max, RESOLUTION)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((px, py), pz, (Xi, Yi), method='linear')

    nan_count = np.isnan(Zi).sum()
    if nan_count > 0:
        Zi_nearest = griddata((px, py), pz, (Xi, Yi), method='nearest')
        Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)

    return Zi, xi, yi


# =========================================================================
# Step 4: Compute residuals for one strip
# =========================================================================

def compute_strip_residuals(dem: np.ndarray) -> dict:
    """
    Apply all residual methods to a strip DEM.

    Returns:
        Dict of {method_name: residual_array}
    """
    from src.decomposition.registry import run_decomposition

    residuals = {}
    for method_name, params in RESIDUAL_METHODS:
        try:
            _, residual = run_decomposition(method_name, dem, params)
            residuals[method_name] = residual
        except Exception as e:
            logger.warning(f"    Method {method_name} failed: {e}")

    return residuals


# =========================================================================
# Step 5: Compute confidence heatmap for one strip
# =========================================================================

def compute_strip_confidence(
    dem: np.ndarray,
    residuals: dict,
    bearing: float,
) -> np.ndarray:
    """
    Build a multi-method consensus confidence heatmap.

    For each pixel:
    1. Count methods that detect a feature (above threshold)
    2. Weight by normalized residual magnitude
    3. Add Meijering ridge filter response
    4. Smooth spatially
    5. Normalize to 0-1

    Returns:
        Confidence array (0.0 - 1.0) same shape as dem.
    """
    from skimage.filters import meijering

    h, w = dem.shape
    votes = np.zeros((h, w), dtype=np.float64)
    n_methods = 0

    for name, res in residuals.items():
        # Normalize residual to [0, 1] using percentile
        abs_res = np.abs(res)
        p99 = np.percentile(abs_res, 99)
        if p99 > 0:
            norm = np.clip(abs_res / p99, 0, 1)
        else:
            norm = np.zeros_like(abs_res)

        # Threshold: pixel is "detected" if above 30th percentile of non-zero
        threshold = np.percentile(abs_res[abs_res > 0], 70) if np.any(abs_res > 0) else 0
        detected = (abs_res > threshold).astype(np.float64)

        # Add weighted vote: detection flag * normalized magnitude
        votes += detected * norm
        n_methods += 1

    if n_methods > 0:
        votes /= n_methods

    # Add Meijering ridge filter (directly on DEM for tubular structures)
    try:
        ridge = meijering(dem, sigmas=range(2, 8), black_ridges=False)
        ridge_norm = ridge / (ridge.max() + 1e-10)
        # Weight ridge filter as equivalent to one method
        votes = (votes * n_methods + ridge_norm) / (n_methods + 1)
    except Exception as e:
        logger.warning(f"    Meijering filter failed: {e}")

    # Smooth spatially
    confidence = gaussian_filter(votes, sigma=3.0)

    # Normalize to 0-1
    c_min, c_max = confidence.min(), confidence.max()
    if c_max > c_min:
        confidence = (confidence - c_min) / (c_max - c_min)

    return confidence


# =========================================================================
# Step 6: Write a GeoTIFF for one strip
# =========================================================================

def write_strip_geotiff(
    data: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    output_path: Path,
    nodata: float = None,
):
    """Write a single-band GeoTIFF with State Plane coordinates."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform = from_bounds(x_min, y_min, x_max, y_max, data.shape[1], data.shape[0])
    data_out = np.flipud(data).astype(np.float32)

    profile = {
        'driver': 'GTiff',
        'height': data_out.shape[0],
        'width': data_out.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': CRS,
        'transform': transform,
        'compress': 'lzw',
    }
    if nodata is not None:
        profile['nodata'] = nodata

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data_out, 1)


# =========================================================================
# Step 7: Trace centerline from stitched confidence heatmap
# =========================================================================

def trace_centerline(
    confidence_tif: Path,
    threshold: float = 0.3,
) -> list:
    """
    Extract road centerline from the merged confidence GeoTIFF.

    Returns:
        List of (x, y, confidence) tuples in State Plane coordinates.
    """
    from skimage.morphology import skeletonize, remove_small_objects
    from skimage.transform import probabilistic_hough_line

    logger.info("Step 7: Tracing road centerline ...")

    with rasterio.open(confidence_tif) as src:
        conf = src.read(1)
        transform = src.transform

    # Threshold
    binary = conf > threshold
    # Remove small objects (noise)
    binary = remove_small_objects(binary, min_size=50)

    if not np.any(binary):
        logger.warning("  No features above threshold!")
        return []

    # Skeletonize to 1-pixel centerline
    skeleton = skeletonize(binary)
    logger.info(f"  Skeleton pixels: {skeleton.sum()}")

    # Use probabilistic Hough to find line segments
    lines = probabilistic_hough_line(
        skeleton, threshold=10, line_length=30, line_gap=20
    )
    logger.info(f"  Hough line segments: {len(lines)}")

    if not lines:
        # Fall back to just the skeleton pixels
        ys, xs = np.where(skeleton)
        points = []
        for px_x, px_y in zip(xs, ys):
            sp_x, sp_y = rasterio.transform.xy(transform, px_y, px_x)
            c = conf[px_y, px_x]
            points.append((sp_x, sp_y, float(c)))
        return points

    # Collect all line segment endpoints + midpoints
    points = []
    for (x0, y0), (x1, y1) in lines:
        for px_x, px_y in [(x0, y0), (x1, y1), ((x0+x1)//2, (y0+y1)//2)]:
            if 0 <= px_y < conf.shape[0] and 0 <= px_x < conf.shape[1]:
                sp_x, sp_y = rasterio.transform.xy(transform, px_y, px_x)
                c = conf[px_y, px_x]
                points.append((sp_x, sp_y, float(c)))

    # Sort by position along the road axis for a coherent polyline
    if len(points) > 2:
        pts = np.array([(p[0], p[1]) for p in points])
        # PCA to find main axis, then sort along it
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        main_axis = eigenvectors[:, -1]  # eigenvector with largest eigenvalue
        projections = centered @ main_axis
        order = np.argsort(projections)
        points = [points[i] for i in order]

    # Remove duplicate/very close points
    filtered = [points[0]]
    for p in points[1:]:
        dx = p[0] - filtered[-1][0]
        dy = p[1] - filtered[-1][1]
        if np.sqrt(dx**2 + dy**2) > RESOLUTION * 3:
            filtered.append(p)
    points = filtered

    logger.info(f"  Centerline points: {len(points)}")
    return points


# =========================================================================
# Step 8: Export GeoJSON
# =========================================================================

def export_geojson(points: list, output_path: Path):
    """Export centerline points as a GeoJSON LineString with confidence."""
    if len(points) < 2:
        logger.warning("  Not enough points for a LineString")
        return

    coords = [(p[0], p[1]) for p in points]
    confidences = [p[2] for p in points]

    # Also compute WGS84 coordinates for convenience
    t = Transformer.from_crs(CRS, 'EPSG:4326', always_xy=True)
    coords_wgs84 = [t.transform(x, y) for x, y in coords]

    feature = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {'name': CRS},
        },
        'features': [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': coords,
                },
                'properties': {
                    'name': 'Great Hopewell Road (detected)',
                    'method': 'RESIDUALS multi-method consensus',
                    'mean_confidence': float(np.mean(confidences)),
                    'n_points': len(points),
                    'crs': CRS,
                },
            },
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [(lon, lat) for lon, lat in coords_wgs84],
                },
                'properties': {
                    'name': 'Great Hopewell Road (detected) - WGS84',
                    'crs': 'EPSG:4326',
                },
            },
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(feature, f, indent=2)
    logger.info(f"  Saved GeoJSON: {output_path}")


# =========================================================================
# Step 9: Merge strip GeoTIFFs into one corridor raster
# =========================================================================

def merge_strip_tiffs(strip_dir: Path, pattern: str, output_path: Path):
    """Merge all strip GeoTIFFs matching pattern into one corridor raster."""
    tif_files = sorted(strip_dir.glob(pattern))
    if not tif_files:
        logger.warning(f"  No files matching {pattern} in {strip_dir}")
        return

    logger.info(f"  Merging {len(tif_files)} strip rasters -> {output_path}")

    src_files = [rasterio.open(f) for f in tif_files]
    try:
        mosaic, out_transform = rasterio_merge(src_files)

        profile = src_files[0].profile.copy()
        profile.update({
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_transform,
            'compress': 'lzw',
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mosaic)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Merged raster: {output_path} ({size_mb:.1f} MB)")
    finally:
        for src in src_files:
            src.close()


# =========================================================================
# Step 10: Generate summary figure
# =========================================================================

def generate_summary_figure(
    confidence_tif: Path,
    centerline_points: list,
    output_path: Path,
):
    """Generate a corridor overview figure.

    Large merged rasters are downsampled to fit in memory for plotting.
    """
    import matplotlib.pyplot as plt
    from skimage.transform import downscale_local_mean

    logger.info("Generating summary figure ...")

    with rasterio.open(confidence_tif) as src:
        conf = src.read(1)
        transform = src.transform
        bounds = src.bounds

    # Downsample if the raster is huge (target ~4000px on the long side)
    max_dim = max(conf.shape)
    if max_dim > 4000:
        factor = int(np.ceil(max_dim / 4000))
        logger.info(f"  Downsampling {conf.shape} by {factor}x for plotting ...")
        conf = downscale_local_mean(conf, (factor, factor))
        logger.info(f"  Plot raster: {conf.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: confidence heatmap
    ax = axes[0]
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im = ax.imshow(conf, cmap='hot', extent=extent, aspect='equal',
                   origin='upper', vmin=0, vmax=1)
    ax.set_title('Multi-Method Confidence Heatmap', fontsize=12, fontweight='bold')
    ax.set_xlabel('State Plane X (ft)')
    ax.set_ylabel('State Plane Y (ft)')
    plt.colorbar(im, ax=ax, label='Confidence', fraction=0.046, pad=0.04)

    # Mark known sites
    ax.plot(HR_X, HR_Y, 'c^', markersize=12, label='Hopewell Road segment')
    ax.plot(GC_X, GC_Y, 'gs', markersize=12, label='Great Circle')
    ax.legend(fontsize=9, loc='upper left')

    # Right: confidence + centerline
    ax = axes[1]
    ax.imshow(conf, cmap='gray_r', extent=extent, aspect='equal',
              origin='upper', vmin=0, vmax=1, alpha=0.7)

    if len(centerline_points) > 1:
        cl_x = [p[0] for p in centerline_points]
        cl_y = [p[1] for p in centerline_points]
        cl_c = [p[2] for p in centerline_points]
        sc = ax.scatter(cl_x, cl_y, c=cl_c, cmap='RdYlGn', s=3,
                        vmin=0, vmax=1, zorder=5)
        plt.colorbar(sc, ax=ax, label='Centerline confidence', fraction=0.046, pad=0.04)

    ax.plot(HR_X, HR_Y, 'c^', markersize=12, label='Hopewell Road')
    ax.plot(GC_X, GC_Y, 'gs', markersize=12, label='Great Circle')
    ax.set_title('Detected Road Centerline', fontsize=12, fontweight='bold')
    ax.set_xlabel('State Plane X (ft)')
    ax.set_ylabel('State Plane Y (ft)')
    ax.legend(fontsize=9, loc='upper left')

    fig.suptitle(
        'RESIDUALS: Great Hopewell Road Corridor Trace\n'
        'Multi-method residual analysis of Licking County LiDAR',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Summary figure: {output_path}")


# =========================================================================
# Main pipeline
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RESIDUALS: Great Hopewell Road Corridor Trace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--max-strips', type=int, default=50,
                        help='Max strips per direction (default: 50)')
    parser.add_argument('--skip-dem-gen', action='store_true',
                        help='Skip DEM generation (reuse existing strip files)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Confidence threshold for centerline (default: 0.3)')
    parser.add_argument('--bearing', type=float, default=None,
                        help='Manual bearing override (degrees from +X axis, CCW positive)')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR),
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()

    output_dir = Path(args.output)
    strip_dir = output_dir / 'strips'
    strip_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("RESIDUALS: Great Hopewell Road Corridor Trace")
    logger.info("=" * 70)

    # Import decomposition methods (triggers registration)
    import src.decomposition.methods           # noqa: F401
    import src.decomposition.methods_extended  # noqa: F401

    # ---- Step 1: Determine bearing ----
    # The Hopewell Road ran from the Great Circle (Newark) SSW toward
    # Chillicothe.  Use the known site-to-site bearing, which is reliable.
    # The Hough-based detection picks up whatever linear feature dominates
    # the crop (modern roads, field edges) and can point the wrong way.
    dx = HR_X - GC_X
    dy = HR_Y - GC_Y
    bearing = np.arctan2(dy, dx)  # site-to-site bearing
    logger.info(f"Site-to-site bearing: {np.degrees(bearing):.1f} deg from +X axis")

    if args.bearing is not None:
        # Manual override (degrees from +X axis, CCW positive)
        bearing = np.radians(args.bearing)
        logger.info(f"Using manual bearing override: {args.bearing:.1f} deg")

    bearing_deg = np.degrees(bearing)
    logger.info(f"Road bearing: {bearing_deg:.1f} degrees from +X axis")

    # ---- Step 2: Generate corridor geometry ----
    # Origin: midpoint between known sites
    origin_x = (HR_X + GC_X) / 2
    origin_y = (HR_Y + GC_Y) / 2
    logger.info(f"Corridor origin: ({origin_x:.0f}, {origin_y:.0f})")

    strips = generate_corridor_strips(
        origin_x, origin_y, bearing, args.max_strips
    )

    if not strips:
        logger.error("No valid strips generated!")
        sys.exit(1)

    # ---- Step 3-5: Process each strip ----
    if not args.skip_dem_gen:
        tile_gdf = gpd.read_file(TILE_INDEX)
        logger.info(f"Loaded tile index: {len(tile_gdf)} tiles")

        total_strips = len(strips)
        for i, strip in enumerate(strips):
            sid = strip['strip_id']
            logger.info(f"\n--- Strip {sid} ({i+1}/{total_strips}) "
                        f"[{strip['direction']} #{strip['dir_index']}] "
                        f"center=({strip['center_x']:.0f}, {strip['center_y']:.0f}) "
                        f"tiles={strip['n_tiles']} ---")

            conf_path = strip_dir / f'strip_{sid:03d}_confidence.tif'
            tophat_path = strip_dir / f'strip_{sid:03d}_tophat.tif'

            # Check if all outputs already exist
            dem_cache = strip_dir / f'strip_{sid:03d}_dem.npy'
            all_method_tifs = [strip_dir / f'strip_{sid:03d}_{name}.tif'
                               for name, _ in RESIDUAL_METHODS]
            if conf_path.exists() and all(f.exists() for f in all_method_tifs):
                logger.info("  Already processed, skipping")
                continue

            # Step 3: Generate or load cached DEM
            t0 = time.time()
            if dem_cache.exists():
                dem = np.load(dem_cache)
                logger.info(f"  DEM (cached): {dem.shape}")
            else:
                dem, xi, yi = generate_strip_dem(strip, tile_gdf)
                if dem is None:
                    logger.warning("  Not enough points, skipping")
                    continue
                np.save(dem_cache, dem)
            t_dem = time.time() - t0
            logger.info(f"  DEM: {dem.shape} ({dem.size:,} px) in {t_dem:.1f}s, "
                        f"elev=[{dem.min():.0f}, {dem.max():.0f}]")

            # Step 4: Compute residuals
            t0 = time.time()
            residuals = compute_strip_residuals(dem)
            t_res = time.time() - t0
            logger.info(f"  Residuals: {len(residuals)} methods in {t_res:.1f}s")

            if not residuals:
                logger.warning("  No residuals computed, skipping")
                continue

            # Step 5: Confidence heatmap
            t0 = time.time()
            confidence = compute_strip_confidence(dem, residuals, bearing)
            t_conf = time.time() - t0
            logger.info(f"  Confidence: max={confidence.max():.3f} in {t_conf:.1f}s")

            # Write strip GeoTIFFs -- confidence + each method individually
            write_strip_geotiff(
                confidence,
                strip['x_min'], strip['x_max'],
                strip['y_min'], strip['y_max'],
                conf_path,
            )

            for method_name, residual in residuals.items():
                method_path = strip_dir / f'strip_{sid:03d}_{method_name}.tif'
                write_strip_geotiff(
                    residual,
                    strip['x_min'], strip['x_max'],
                    strip['y_min'], strip['y_max'],
                    method_path,
                )

            elapsed = datetime.now() - start_time
            remaining = elapsed / (i + 1) * (total_strips - i - 1)
            logger.info(f"  Elapsed: {elapsed}, est. remaining: {remaining}")

    # ---- Step 6: Merge strips ----
    logger.info("\n" + "=" * 70)
    logger.info("Merging corridor rasters ...")
    logger.info("=" * 70)

    conf_merged = output_dir / 'corridor_confidence.tif'
    merge_strip_tiffs(strip_dir, 'strip_*_confidence.tif', conf_merged)

    # Merge each individual method into its own corridor GeoTIFF
    for method_name, _ in RESIDUAL_METHODS:
        method_merged = output_dir / f'corridor_residual_{method_name}.tif'
        merge_strip_tiffs(
            strip_dir, f'strip_*_{method_name}.tif', method_merged
        )

    # ---- Step 7: Trace centerline ----
    logger.info("\n" + "=" * 70)
    logger.info("Tracing road centerline ...")
    logger.info("=" * 70)

    centerline_points = []
    if conf_merged.exists():
        centerline_points = trace_centerline(conf_merged, args.threshold)

    # ---- Step 8: Export GeoJSON ----
    geojson_path = output_dir / 'road_centerline.geojson'
    export_geojson(centerline_points, geojson_path)

    # ---- Save metadata (before figure, which can OOM on large corridors) ----
    meta = {
        'bearing_degrees': float(bearing_deg),
        'corridor_width_ft': CORRIDOR_WIDTH,
        'strip_length_ft': STRIP_LENGTH,
        'resolution_ft': RESOLUTION,
        'n_strips': len(strips),
        'threshold': args.threshold,
        'methods': [{'name': n, 'params': p} for n, p in RESIDUAL_METHODS],
        'elapsed_seconds': (datetime.now() - start_time).total_seconds(),
        'known_sites': {
            'hopewell_road': {'x': HR_X, 'y': HR_Y},
            'great_circle': {'x': GC_X, 'y': GC_Y},
        },
        'centerline_points': len(centerline_points),
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # ---- Step 9: Summary figure ----
    if conf_merged.exists():
        generate_summary_figure(
            conf_merged,
            centerline_points,
            output_dir / 'corridor_overview.png',
        )

    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("TRACE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Strips processed: {len(strips)}")
    logger.info(f"Centerline points: {len(centerline_points)}")
    logger.info(f"Time: {elapsed}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
