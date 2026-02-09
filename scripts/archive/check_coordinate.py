#!/usr/bin/env python3
"""Check if a lat/lon coordinate falls within our LAS/DEM data."""

import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from pathlib import Path

# User's coordinate (WGS84 lat/lon)
lat, lon = 40.02785285896847, -82.45898950567279

# Transform to Ohio South State Plane (EPSG:3735 - feet)
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3735', always_xy=True)
x, y = transformer.transform(lon, lat)

print(f'Input: {lat}, {lon}')
print(f'Ohio South State Plane (ft): X={x:.2f}, Y={y:.2f}')
print()

# Check Licking County tile index
print('=' * 50)
print('Checking Licking County tile index...')
print('=' * 50)

index_path = Path(r'F:\science-projects\lidar_super_rez\data\licking_2015\_LIC_1250_Tile_Index\Tile_Index.shp')
las_dir = Path(r'F:\science-projects\lidar_super_rez\data\licking_2015')

if index_path.exists():
    tiles = gpd.read_file(index_path)
    print(f'Loaded {len(tiles)} tiles from index')
    print(f'Index CRS: {tiles.crs}')
    
    # Create point in same CRS
    point = Point(x, y)
    
    # Find tiles containing the point
    containing = tiles[tiles.contains(point)]
    print(f'\nTiles containing your coordinate: {len(containing)}')
    
    if len(containing) > 0:
        for _, row in containing.iterrows():
            tile_name = row['TileName']
            print(f'\n  Tile: {tile_name}')
            
            # Check if LAS file exists
            las_path = las_dir / f'{tile_name}.las'
            if las_path.exists():
                size_mb = las_path.stat().st_size / (1024 * 1024)
                print(f'  FILE EXISTS: {las_path}')
                print(f'  Size: {size_mb:.1f} MB')
            else:
                print(f'  FILE MISSING: {las_path}')
    else:
        # Show bounds for reference
        bounds = tiles.total_bounds
        print(f'\nYour coordinate is OUTSIDE the tile index coverage.')
        print(f'Index bounds:')
        print(f'  X: {bounds[0]:.0f} - {bounds[2]:.0f}')
        print(f'  Y: {bounds[1]:.0f} - {bounds[3]:.0f}')
else:
    print(f'Tile index not found: {index_path}')

