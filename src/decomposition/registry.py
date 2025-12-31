"""
Decomposition Method Registry

Provides a decorator-based registration system for decomposition methods.
Each method should return (trend, residual) tuple.
"""

from typing import Dict, Callable, Any, Tuple, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DecompositionMethod:
    """Metadata for a registered decomposition method."""
    name: str
    func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
    category: str
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_ranges: Dict[str, List[Any]] = field(default_factory=dict)
    preserves: str = ""
    destroys: str = ""
    description: str = ""


# Global registry
DECOMPOSITION_REGISTRY: Dict[str, DecompositionMethod] = {}


def register_decomposition(
    name: str,
    category: str,
    default_params: Optional[Dict[str, Any]] = None,
    param_ranges: Optional[Dict[str, List[Any]]] = None,
    preserves: str = "",
    destroys: str = "",
    description: str = ""
):
    """
    Decorator to register a decomposition method.
    
    Usage:
        @register_decomposition(
            name='gaussian',
            category='classical',
            default_params={'sigma': 10},
            param_ranges={'sigma': [2, 5, 10, 20, 50]},
            preserves='smooth regions',
            destroys='all high-frequency equally'
        )
        def decompose_gaussian(dem, sigma=10):
            trend = gaussian_filter(dem, sigma=sigma)
            residual = dem - trend
            return trend, residual
    """
    def decorator(func: Callable) -> Callable:
        method = DecompositionMethod(
            name=name,
            func=func,
            category=category,
            default_params=default_params or {},
            param_ranges=param_ranges or {},
            preserves=preserves,
            destroys=destroys,
            description=description or func.__doc__ or ""
        )
        DECOMPOSITION_REGISTRY[name] = method
        return func
    return decorator


def get_decomposition(name: str) -> DecompositionMethod:
    """Get a decomposition method by name."""
    if name not in DECOMPOSITION_REGISTRY:
        available = list(DECOMPOSITION_REGISTRY.keys())
        raise ValueError(f"Unknown decomposition method: {name}. Available: {available}")
    return DECOMPOSITION_REGISTRY[name]


def list_decompositions() -> List[str]:
    """List all registered decomposition methods."""
    return list(DECOMPOSITION_REGISTRY.keys())


def run_decomposition(
    name: str,
    dem: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a decomposition method on a DEM.
    
    Args:
        name: Method name
        dem: Input DEM array
        params: Optional parameter overrides
        
    Returns:
        (trend, residual) tuple
    """
    method = get_decomposition(name)
    
    # Merge default params with overrides
    run_params = {**method.default_params}
    if params:
        run_params.update(params)
    
    return method.func(dem, **run_params)


def get_all_methods_info() -> Dict[str, Dict[str, Any]]:
    """Get info about all registered methods for reporting."""
    info = {}
    for name, method in DECOMPOSITION_REGISTRY.items():
        info[name] = {
            'category': method.category,
            'default_params': method.default_params,
            'preserves': method.preserves,
            'destroys': method.destroys,
            'description': method.description
        }
    return info

