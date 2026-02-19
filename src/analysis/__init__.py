"""Differential computation and feature analysis."""

from .differential import (
    compute_all_differentials,
    compute_differential,
    compute_method_differentials,
    run_all_combinations,
)
from .features import (
    analyze_features,
    analyze_frequency_content,
    detect_linear_features,
    generate_analysis_report,
    rank_for_features,
)

__all__ = [
    "compute_all_differentials",
    "compute_differential",
    "compute_method_differentials",
    "run_all_combinations",
    "analyze_features",
    "analyze_frequency_content",
    "detect_linear_features",
    "generate_analysis_report",
    "rank_for_features",
]
