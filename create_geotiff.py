#!/usr/bin/env python3
"""Create a GeoTIFF from a DEM numpy array."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import json
import argparse


def create_geotiff(npy_path, json_path=None, output_path=None):
    """Create a GeoTIFF from a DEM .npy file with metadata."""
    
    npy_path = Path(npy_path)
    
    # Derive paths if not provided
    if json_path is None:
        json_path = npy_path.with_suffix('.json')
    if output_path is None:
        output_path = npy_path.with_suffix('.tif')
    
    # Load DEM
    dem = np.load(npy_path)
    print(f'DEM shape: {dem.shape}')
    
    # Load metadata
    with open(json_path) as f:
        meta = json.load(f)
    
    x_min, x_max = meta['x_min'], meta['x_max']
    y_min, y_max = meta['y_min'], meta['y_max']
    print(f'Bounds: X {x_min:.0f}-{x_max:.0f}, Y {y_min:.0f}-{y_max:.0f}')
    
    # Get CRS (default to Ohio South State Plane)
    crs = meta.get('crs', 'EPSG:3735')
    
    # Create transform (affine transformation from pixel to world coordinates)
    transform = from_bounds(x_min, y_min, x_max, y_max, dem.shape[1], dem.shape[0])
    
    # Flip the DEM vertically since GeoTIFF uses top-left origin
    dem_flipped = np.flipud(dem).astype(np.float32)
    
    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=dem_flipped.shape[0],
        width=dem_flipped.shape[1],
        count=1,
        dtype=dem_flipped.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(dem_flipped, 1)
        
        # Add metadata tags
        tags = {
            'TIFFTAG_IMAGEDESCRIPTION': f'DEM from {npy_path.stem}',
        }
        if 'source_tiles' in meta:
            tags['source_tiles'] = ','.join(meta['source_tiles'])
        if 'target_coordinate' in meta:
            tags['target_lat'] = str(meta['target_coordinate'].get('lat', ''))
            tags['target_lon'] = str(meta['target_coordinate'].get('lon', ''))
        dst.update_tags(**tags)
    
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f'Saved: {output_path} ({size_mb:.1f} MB)')
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create GeoTIFF from DEM numpy array')
    parser.add_argument('npy_path', help='Path to .npy DEM file')
    parser.add_argument('--json', help='Path to metadata JSON (default: same name as npy)')
    parser.add_argument('--output', help='Output GeoTIFF path (default: same name as npy)')
    
    args = parser.parse_args()
    create_geotiff(args.npy_path, args.json, args.output)

