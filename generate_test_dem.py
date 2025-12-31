#!/usr/bin/env python3
"""
Generate Test DEM from LiDAR Data

Creates a test DEM from Fairfield County LiDAR tiles for use in experiments.
"""

import numpy as np
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_contiguous_tiles(lidar_dir: Path, grid_size: tuple = (4, 4)):
    """
    Find tiles that form a contiguous rectangular grid.
    
    Tile naming: BSxxxyyy where xxx=X index, yyy=Y index
    Each tile is 1250x1250 ft.
    
    Args:
        lidar_dir: Directory containing LAS files
        grid_size: (rows, cols) of tiles to select
        
    Returns:
        List of tile paths forming a contiguous grid
    """
    # Parse all available tiles into (x_idx, y_idx) -> path mapping
    tile_map = {}
    for las_file in lidar_dir.glob('*.las'):
        name = las_file.stem
        if name.startswith('BS') and len(name) == 8:
            try:
                x_idx = int(name[2:5])
                y_idx = int(name[5:8])
                tile_map[(x_idx, y_idx)] = las_file
            except ValueError:
                continue
    
    if not tile_map:
        return []
    
    # Find all possible grid origins and check for complete coverage
    all_coords = list(tile_map.keys())
    x_coords = sorted(set(c[0] for c in all_coords))
    y_coords = sorted(set(c[1] for c in all_coords))
    
    rows, cols = grid_size
    best_tiles = []
    
    # Search for a complete grid
    for x_start in x_coords:
        for y_start in y_coords:
            # Check if we have complete coverage for this grid
            tiles = []
            complete = True
            for dx in range(cols):
                for dy in range(rows):
                    coord = (x_start + dx, y_start + dy)
                    if coord in tile_map:
                        tiles.append(tile_map[coord])
                    else:
                        complete = False
                        break
                if not complete:
                    break
            
            if complete and len(tiles) == rows * cols:
                return tiles
    
    # If no complete grid found, find the largest available
    logger.warning(f"No complete {rows}x{cols} grid found, using largest available")
    
    # Fall back to finding any contiguous tiles
    for x_start in x_coords:
        for y_start in y_coords:
            tiles = []
            for dx in range(cols):
                for dy in range(rows):
                    coord = (x_start + dx, y_start + dy)
                    if coord in tile_map:
                        tiles.append(tile_map[coord])
            if len(tiles) > len(best_tiles):
                best_tiles = tiles
    
    return best_tiles


def generate_test_dem(
    lidar_dir: Path,
    output_path: Path,
    resolution: float = 1.5,
    grid_size: tuple = (4, 4),
    target_size: int = 2000
):
    """
    Generate a test DEM from LiDAR tiles.
    
    Args:
        lidar_dir: Directory containing LAS files
        output_path: Output path for .npy file
        resolution: Grid resolution in feet
        grid_size: (rows, cols) of tiles to select for contiguous coverage
        target_size: Target size in pixels (will use ~target_size x target_size)
    """
    import laspy
    from scipy.interpolate import griddata
    
    # Find LAS files
    all_las_files = list(lidar_dir.glob('*.las'))
    if not all_las_files:
        raise FileNotFoundError(f"No LAS files found in {lidar_dir}")
    
    logger.info(f"Found {len(all_las_files)} LAS files")
    
    # Select contiguous tiles
    las_files = find_contiguous_tiles(lidar_dir, grid_size)
    if not las_files:
        raise ValueError("Could not find contiguous tile coverage")
    
    logger.info(f"Using {len(las_files)} tiles in {grid_size[0]}x{grid_size[1]} grid:")
    for f in las_files:
        logger.info(f"  {f.name}")
    
    # Load all points
    all_x, all_y, all_z = [], [], []
    
    for las_path in las_files:
        logger.info(f"Loading: {las_path.name}")
        las = laspy.read(las_path)
        
        # Filter to ground points (classification 2)
        if hasattr(las, 'classification'):
            mask = las.classification == 2
            if mask.sum() == 0:
                logger.warning(f"  No ground points, using all")
                mask = np.ones(len(las.x), dtype=bool)
        else:
            mask = np.ones(len(las.x), dtype=bool)
        
        all_x.extend(las.x[mask])
        all_y.extend(las.y[mask])
        all_z.extend(las.z[mask])
        
        logger.info(f"  {mask.sum()} ground points")
    
    x = np.array(all_x)
    y = np.array(all_y)
    z = np.array(all_z)
    
    logger.info(f"Total: {len(x):,} points")
    logger.info(f"X range: {x.min():.0f} - {x.max():.0f}")
    logger.info(f"Y range: {y.min():.0f} - {y.max():.0f}")
    logger.info(f"Z range: {z.min():.1f} - {z.max():.1f}")
    
    # Create grid
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    
    # Adjust resolution to hit target size approximately
    actual_res = max(x_range, y_range) / target_size
    if actual_res > resolution:
        logger.info(f"Adjusting resolution from {resolution} to {actual_res:.2f} to fit target size")
        resolution = actual_res
    
    xi = np.arange(x.min(), x.max(), resolution)
    yi = np.arange(y.min(), y.max(), resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    logger.info(f"Grid size: {Xi.shape[1]} x {Xi.shape[0]} ({Xi.size:,} pixels)")
    logger.info(f"Resolution: {resolution:.2f} ft/pixel")
    
    # Interpolate
    logger.info("Interpolating (this may take a minute)...")
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    
    # Fill NaN with nearest
    nan_count = np.isnan(Zi).sum()
    if nan_count > 0:
        logger.info(f"Filling {nan_count} NaN values with nearest neighbor...")
        Zi_nearest = griddata((x, y), z, (Xi, Yi), method='nearest')
        Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, Zi)
    
    logger.info(f"Saved: {output_path}")
    logger.info(f"Shape: {Zi.shape}")
    logger.info(f"Elevation range: {Zi.min():.1f} - {Zi.max():.1f} ft")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Also save metadata
    meta = {
        'resolution_ft': resolution,
        'shape': list(Zi.shape),
        'x_min': float(x.min()),
        'y_min': float(y.min()),
        'z_min': float(Zi.min()),
        'z_max': float(Zi.max()),
        'source_tiles': [f.name for f in las_files]
    }
    
    import json
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata: {meta_path}")
    
    return Zi


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test DEM from LiDAR")
    parser.add_argument('--lidar-dir', type=str, 
                       default='../lidar_super_rez/data/fairfield_2015',
                       help='Directory containing LAS files')
    parser.add_argument('--output', type=str,
                       default='data/test_dems/fairfield_sample_1.5ft.npy',
                       help='Output path')
    parser.add_argument('--resolution', type=float, default=1.5,
                       help='Grid resolution in feet')
    parser.add_argument('--grid-rows', type=int, default=4,
                       help='Number of tile rows (each tile is 1250 ft)')
    parser.add_argument('--grid-cols', type=int, default=4,
                       help='Number of tile columns (each tile is 1250 ft)')
    parser.add_argument('--target-size', type=int, default=2000,
                       help='Target grid size in pixels')
    
    args = parser.parse_args()
    
    lidar_dir = Path(args.lidar_dir)
    if not lidar_dir.exists():
        # Try absolute path
        lidar_dir = Path('F:/science-projects/lidar_super_rez/data/fairfield_2015')
    
    if not lidar_dir.exists():
        logger.error(f"LiDAR directory not found: {lidar_dir}")
        logger.error("Please specify --lidar-dir with path to LAS files")
        sys.exit(1)
    
    output_path = Path(args.output)
    
    generate_test_dem(
        lidar_dir=lidar_dir,
        output_path=output_path,
        resolution=args.resolution,
        grid_size=(args.grid_rows, args.grid_cols),
        target_size=args.target_size
    )


if __name__ == '__main__':
    main()

