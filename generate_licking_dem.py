#!/usr/bin/env python3
"""
Generate a Licking County DEM covering the Great Hopewell Road and
Great Circle Earthworks archaeological sites.

Outputs:
    data/test_dems/licking_hopewell_2.5ft.npy   (DEM array)
    data/test_dems/licking_hopewell_2.5ft.json   (metadata)
"""

import numpy as np
import json
import sys
from pathlib import Path
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import box
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Archaeological site coordinates (WGS84)
# ---------------------------------------------------------------------------
SITES = {
    'hopewell_road': {
        'lat': 40.02818536392136,
        'lon': -82.45905475647373,
        'description': 'Great Hopewell Road segment',
    },
    'great_circle': {
        'lat': 40.04139440223933,
        'lon': -82.43111687044818,
        'description': 'Great Circle Earthworks (Newark Earthworks)',
    },
}

LIDAR_DIR = Path(r'F:\science-projects\lidar_super_rez\data\licking_2015')
TILE_INDEX = LIDAR_DIR / '_LIC_1250_Tile_Index' / 'Tile_Index.shp'
OUTPUT_NPY = Path('data/test_dems/licking_hopewell_2.5ft.npy')
OUTPUT_JSON = Path('data/test_dems/licking_hopewell_2.5ft.json')

RESOLUTION = 2.5   # feet per pixel
MARGIN = 2000       # feet of padding around sites


def main():
    import laspy
    from scipy.interpolate import griddata

    # --- Convert site coords to State Plane --------------------------------
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3735', always_xy=True)
    site_coords_sp = {}
    for name, info in SITES.items():
        x, y = transformer.transform(info['lon'], info['lat'])
        site_coords_sp[name] = {'x': x, 'y': y}
        logger.info(f"{info['description']}: X={x:.0f}, Y={y:.0f}")

    # --- Compute bounding box ----------------------------------------------
    all_x = [c['x'] for c in site_coords_sp.values()]
    all_y = [c['y'] for c in site_coords_sp.values()]
    x_min = min(all_x) - MARGIN
    x_max = max(all_x) + MARGIN
    y_min = min(all_y) - MARGIN
    y_max = max(all_y) + MARGIN
    logger.info(f"Bounding box: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}]")
    logger.info(f"Extent: {x_max - x_min:.0f} x {y_max - y_min:.0f} ft")

    # --- Find tiles --------------------------------------------------------
    logger.info(f"Loading tile index: {TILE_INDEX}")
    tiles = gpd.read_file(TILE_INDEX)
    bbox_geom = box(x_min, y_min, x_max, y_max)
    matching = tiles[tiles.intersects(bbox_geom)]
    logger.info(f"Tiles covering area: {len(matching)}")

    if len(matching) == 0:
        logger.error("No tiles found covering the target area!")
        sys.exit(1)

    # --- Load LAS points ---------------------------------------------------
    all_px, all_py, all_pz = [], [], []
    for _, row in matching.iterrows():
        tile_name = row['TileName']
        las_path = LIDAR_DIR / f'{tile_name}.las'
        if not las_path.exists():
            logger.warning(f"  Missing: {las_path}")
            continue

        logger.info(f"  Loading {tile_name}.las ...")
        las = laspy.read(las_path)

        # Filter to ground points (classification 2)
        if hasattr(las, 'classification'):
            mask = las.classification == 2
            if mask.sum() == 0:
                mask = np.ones(len(las.x), dtype=bool)
        else:
            mask = np.ones(len(las.x), dtype=bool)

        # Clip to bounding box
        x = np.array(las.x[mask])
        y = np.array(las.y[mask])
        z = np.array(las.z[mask])
        in_bbox = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        all_px.extend(x[in_bbox])
        all_py.extend(y[in_bbox])
        all_pz.extend(z[in_bbox])
        logger.info(f"    {in_bbox.sum()} ground points in bbox")

    px = np.array(all_px)
    py = np.array(all_py)
    pz = np.array(all_pz)
    logger.info(f"Total points: {len(px):,}")
    logger.info(f"Z range: {pz.min():.1f} - {pz.max():.1f} ft")

    # --- Interpolate to grid -----------------------------------------------
    xi = np.arange(x_min, x_max, RESOLUTION)
    yi = np.arange(y_min, y_max, RESOLUTION)
    Xi, Yi = np.meshgrid(xi, yi)
    logger.info(f"Grid size: {Xi.shape[1]} x {Xi.shape[0]} ({Xi.size:,} pixels)")
    logger.info(f"Resolution: {RESOLUTION} ft/pixel")

    logger.info("Interpolating (this may take a few minutes) ...")
    Zi = griddata((px, py), pz, (Xi, Yi), method='linear')

    nan_count = np.isnan(Zi).sum()
    if nan_count > 0:
        logger.info(f"Filling {nan_count} NaN values with nearest neighbor ...")
        Zi_nearest = griddata((px, py), pz, (Xi, Yi), method='nearest')
        Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)

    # --- Save ---------------------------------------------------------------
    OUTPUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_NPY, Zi)
    logger.info(f"Saved DEM: {OUTPUT_NPY} ({OUTPUT_NPY.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save metadata with site coordinates in pixel space
    meta = {
        'resolution_ft': RESOLUTION,
        'shape': list(Zi.shape),
        'x_min': float(x_min),
        'x_max': float(x_max),
        'y_min': float(y_min),
        'y_max': float(y_max),
        'z_min': float(Zi.min()),
        'z_max': float(Zi.max()),
        'crs': 'EPSG:3735',
        'sites': {},
        'source_tiles': [row['TileName'] for _, row in matching.iterrows()],
    }

    for name, sp in site_coords_sp.items():
        px_x = (sp['x'] - x_min) / RESOLUTION
        px_y = (sp['y'] - y_min) / RESOLUTION
        meta['sites'][name] = {
            'description': SITES[name]['description'],
            'lat': SITES[name]['lat'],
            'lon': SITES[name]['lon'],
            'state_plane_x': sp['x'],
            'state_plane_y': sp['y'],
            'pixel_x': float(px_x),
            'pixel_y': float(px_y),
        }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata: {OUTPUT_JSON}")

    logger.info(f"DEM shape: {Zi.shape}")
    logger.info(f"Elevation range: {Zi.min():.1f} - {Zi.max():.1f} ft")


if __name__ == '__main__':
    main()
