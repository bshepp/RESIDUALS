"""Signal decomposition methods for DEM analysis."""
from .registry import (
    DECOMPOSITION_REGISTRY,
    register_decomposition,
    get_decomposition,
    list_decompositions,
    run_decomposition,
    get_all_methods_info
)

# Import methods to trigger registration
from . import methods
from . import methods_extended  # Extended methods for exhaustive prior art

