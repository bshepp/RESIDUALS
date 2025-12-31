"""Upsampling/interpolation methods for DEM super-resolution."""
from .registry import (
    UPSAMPLING_REGISTRY,
    register_upsampling,
    get_upsampling,
    list_upsamplings,
    run_upsampling,
    get_all_methods_info
)

# Import methods to trigger registration
from . import methods

