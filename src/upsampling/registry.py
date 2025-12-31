"""
Upsampling Method Registry

Provides a decorator-based registration system for upsampling methods.
Each method should take a DEM and scale factor, return upsampled DEM.
"""

from typing import Dict, Callable, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class UpsamplingMethod:
    """Metadata for a registered upsampling method."""
    name: str
    func: Callable[[np.ndarray, int], np.ndarray]
    category: str
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_ranges: Dict[str, List[Any]] = field(default_factory=dict)
    preserves: str = ""
    introduces: str = ""
    description: str = ""


# Global registry
UPSAMPLING_REGISTRY: Dict[str, UpsamplingMethod] = {}


def register_upsampling(
    name: str,
    category: str,
    default_params: Optional[Dict[str, Any]] = None,
    param_ranges: Optional[Dict[str, List[Any]]] = None,
    preserves: str = "",
    introduces: str = "",
    description: str = ""
):
    """
    Decorator to register an upsampling method.
    
    Usage:
        @register_upsampling(
            name='bicubic',
            category='interpolation',
            default_params={'scale': 2},
            preserves='smooth curves',
            introduces='slight ringing at edges'
        )
        def upsample_bicubic(dem, scale=2):
            return zoom(dem, scale, order=3)
    """
    def decorator(func: Callable) -> Callable:
        method = UpsamplingMethod(
            name=name,
            func=func,
            category=category,
            default_params=default_params or {},
            param_ranges=param_ranges or {},
            preserves=preserves,
            introduces=introduces,
            description=description or func.__doc__ or ""
        )
        UPSAMPLING_REGISTRY[name] = method
        return func
    return decorator


def get_upsampling(name: str) -> UpsamplingMethod:
    """Get an upsampling method by name."""
    if name not in UPSAMPLING_REGISTRY:
        available = list(UPSAMPLING_REGISTRY.keys())
        raise ValueError(f"Unknown upsampling method: {name}. Available: {available}")
    return UPSAMPLING_REGISTRY[name]


def list_upsamplings() -> List[str]:
    """List all registered upsampling methods."""
    return list(UPSAMPLING_REGISTRY.keys())


def run_upsampling(
    name: str,
    dem: np.ndarray,
    scale: int = 2,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Run an upsampling method on a DEM.
    
    Args:
        name: Method name
        dem: Input DEM array
        scale: Upsampling scale factor
        params: Optional parameter overrides
        
    Returns:
        Upsampled DEM array
    """
    method = get_upsampling(name)
    
    # Merge default params with overrides
    run_params = {**method.default_params}
    run_params['scale'] = scale
    if params:
        run_params.update(params)
    
    return method.func(dem, **run_params)


def get_all_methods_info() -> Dict[str, Dict[str, Any]]:
    """Get info about all registered methods for reporting."""
    info = {}
    for name, method in UPSAMPLING_REGISTRY.items():
        info[name] = {
            'category': method.category,
            'default_params': method.default_params,
            'preserves': method.preserves,
            'introduces': method.introduces,
            'description': method.description
        }
    return info

