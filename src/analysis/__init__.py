"""Differential computation and feature analysis."""
from .differential import (
    compute_differential,
    compute_all_differentials,
    run_all_combinations,
    compute_method_differentials
)
from .features import (
    analyze_features,
    rank_for_features,
    generate_analysis_report,
    detect_linear_features,
    analyze_frequency_content
)

