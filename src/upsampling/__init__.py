"""Upsampling/interpolation methods for DEM super-resolution."""

# Import methods to trigger registration
from . import (
    methods,
    methods_extended,  # Extended methods for exhaustive prior art
)
from .registry import (
    UPSAMPLING_REGISTRY,
    get_all_methods_info,
    get_upsampling,
    list_upsamplings,
    register_upsampling,
    run_upsampling,
)

__all__ = [
    "methods",
    "methods_extended",
    "UPSAMPLING_REGISTRY",
    "get_all_methods_info",
    "get_upsampling",
    "list_upsamplings",
    "register_upsampling",
    "run_upsampling",
]
