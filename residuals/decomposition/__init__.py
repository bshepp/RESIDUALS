"""Signal decomposition methods for DEM analysis."""

# Import methods to trigger registration
from . import (
    methods,
    methods_extended,  # Extended methods for exhaustive prior art
)
from .registry import (
    DECOMPOSITION_REGISTRY,
    get_all_methods_info,
    get_decomposition,
    list_decompositions,
    register_decomposition,
    run_decomposition,
)

__all__ = [
    "methods",
    "methods_extended",
    "DECOMPOSITION_REGISTRY",
    "get_all_methods_info",
    "get_decomposition",
    "list_decompositions",
    "register_decomposition",
    "run_decomposition",
]
