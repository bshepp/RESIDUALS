"""
DEM Input/Output Utilities

Functions for loading and saving DEM data from various sources.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_dem(path: str) -> np.ndarray:
    """
    Load a DEM from file.
    
    Supports:
    - .npy: NumPy array
    - .tif/.tiff: GeoTIFF (requires rasterio)
    - .las: LiDAR point cloud (converts to raster)
    
    Args:
        path: Path to DEM file
        
    Returns:
        2D NumPy array of elevation values
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"DEM file not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.npy':
        return np.load(path)
    
    elif suffix in ['.tif', '.tiff']:
        try:
            import rasterio
            with rasterio.open(path) as src:
                dem = src.read(1)
            return dem
        except ImportError:
            raise ImportError("rasterio required for GeoTIFF support: pip install rasterio")
    
    elif suffix == '.las':
        return load_las_to_dem(path)
    
    else:
        raise ValueError(f"Unsupported DEM format: {suffix}")


def save_dem(dem: np.ndarray, path: str) -> None:
    """
    Save a DEM to file.
    
    Args:
        dem: 2D NumPy array
        path: Output path (.npy recommended)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = path.suffix.lower()
    
    if suffix == '.npy':
        np.save(path, dem)
    else:
        # Default to npy
        np.save(path.with_suffix('.npy'), dem)


def load_las_to_dem(
    las_path: Path,
    resolution: float = 1.5,
    classification: int = 2
) -> np.ndarray:
    """
    Convert a LAS file to a gridded DEM.
    
    Args:
        las_path: Path to LAS file
        resolution: Grid resolution in feet
        classification: Point classification to use (2 = ground)
        
    Returns:
        2D NumPy array of elevation values
    """
    try:
        import laspy
        from scipy.interpolate import griddata
    except ImportError:
        raise ImportError("laspy and scipy required: pip install laspy scipy")
    
    logger.info(f"Loading LAS file: {las_path}")
    las = laspy.read(las_path)
    
    # Filter to ground points
    if hasattr(las, 'classification'):
        mask = las.classification == classification
        if mask.sum() == 0:
            logger.warning(f"No points with classification {classification}, using all points")
            mask = np.ones(len(las.x), dtype=bool)
    else:
        mask = np.ones(len(las.x), dtype=bool)
    
    x = np.array(las.x[mask])
    y = np.array(las.y[mask])
    z = np.array(las.z[mask])
    
    logger.info(f"  {len(x)} points, X: {x.min():.0f}-{x.max():.0f}, Y: {y.min():.0f}-{y.max():.0f}")
    
    # Create regular grid
    xi = np.arange(x.min(), x.max(), resolution)
    yi = np.arange(y.min(), y.max(), resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    logger.info(f"  Grid size: {Xi.shape}")
    
    # Interpolate
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    
    # Fill NaN with nearest
    if np.any(np.isnan(Zi)):
        Zi_nearest = griddata((x, y), z, (Xi, Yi), method='nearest')
        Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)
    
    return Zi


def load_multiple_las_to_dem(
    las_paths: list,
    resolution: float = 1.5,
    classification: int = 2,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Load multiple LAS files and merge into a single DEM.
    
    Args:
        las_paths: List of paths to LAS files
        resolution: Grid resolution in feet
        classification: Point classification (2 = ground)
        bounds: Optional (x_min, x_max, y_min, y_max) bounds
        
    Returns:
        (dem_array, (x_min, y_min)) tuple
    """
    try:
        import laspy
        from scipy.interpolate import griddata
    except ImportError:
        raise ImportError("laspy and scipy required")
    
    all_x, all_y, all_z = [], [], []
    
    for las_path in las_paths:
        logger.info(f"Loading: {las_path}")
        las = laspy.read(las_path)
        
        if hasattr(las, 'classification'):
            mask = las.classification == classification
            if mask.sum() == 0:
                mask = np.ones(len(las.x), dtype=bool)
        else:
            mask = np.ones(len(las.x), dtype=bool)
        
        all_x.extend(las.x[mask])
        all_y.extend(las.y[mask])
        all_z.extend(las.z[mask])
    
    x = np.array(all_x)
    y = np.array(all_y)
    z = np.array(all_z)
    
    logger.info(f"Total: {len(x)} points")
    
    # Determine bounds
    if bounds:
        x_min, x_max, y_min, y_max = bounds
    else:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
    
    # Filter to bounds
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    x, y, z = x[mask], y[mask], z[mask]
    
    # Create grid
    xi = np.arange(x_min, x_max, resolution)
    yi = np.arange(y_min, y_max, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    logger.info(f"Grid size: {Xi.shape}")
    
    # Interpolate
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    
    # Fill NaN
    if np.any(np.isnan(Zi)):
        Zi_nearest = griddata((x, y), z, (Xi, Yi), method='nearest')
        Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)
    
    return Zi, (x_min, y_min)

