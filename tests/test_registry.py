"""
Tests for the decomposition and upsampling registry systems.

Covers:
- Registration completeness
- Metadata integrity
- Error handling for unknown methods
- Registry API functions
"""

import pytest

from src.decomposition.registry import (
    DECOMPOSITION_REGISTRY,
    get_all_methods_info,
    get_decomposition,
    list_decompositions,
)
from src.upsampling.registry import (
    UPSAMPLING_REGISTRY,
    get_upsampling,
    list_upsamplings,
)


class TestDecompositionRegistry:
    """Tests for the decomposition registry."""

    def test_all_methods_have_name(self):
        for name, method in DECOMPOSITION_REGISTRY.items():
            assert method.name == name
            assert isinstance(method.name, str)
            assert len(method.name) > 0

    def test_all_methods_have_category(self):
        for name, method in DECOMPOSITION_REGISTRY.items():
            assert isinstance(method.category, str), f"{name}: missing category"
            assert len(method.category) > 0, f"{name}: empty category"

    def test_all_methods_have_default_params(self):
        for name, method in DECOMPOSITION_REGISTRY.items():
            assert isinstance(method.default_params, dict), f"{name}: default_params is not a dict"

    def test_all_methods_have_param_ranges(self):
        for name, method in DECOMPOSITION_REGISTRY.items():
            assert isinstance(method.param_ranges, dict), f"{name}: param_ranges is not a dict"

    def test_all_methods_are_callable(self):
        for name, method in DECOMPOSITION_REGISTRY.items():
            assert callable(method.func), f"{name}: func is not callable"

    def test_get_unknown_method_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown decomposition method"):
            get_decomposition("nonexistent_method_xyz")

    def test_list_decompositions_returns_list(self):
        result = list_decompositions()
        assert isinstance(result, list)
        assert len(result) == len(DECOMPOSITION_REGISTRY)

    def test_get_all_methods_info(self):
        info = get_all_methods_info()
        assert isinstance(info, dict)
        assert len(info) == len(DECOMPOSITION_REGISTRY)
        for name, details in info.items():
            assert "category" in details
            assert "default_params" in details


class TestUpsamplingRegistry:
    """Tests for the upsampling registry."""

    def test_all_methods_have_name(self):
        for name, method in UPSAMPLING_REGISTRY.items():
            assert method.name == name
            assert isinstance(method.name, str)
            assert len(method.name) > 0

    def test_all_methods_have_category(self):
        for name, method in UPSAMPLING_REGISTRY.items():
            assert isinstance(method.category, str), f"{name}: missing category"
            assert len(method.category) > 0, f"{name}: empty category"

    def test_all_methods_are_callable(self):
        for name, method in UPSAMPLING_REGISTRY.items():
            assert callable(method.func), f"{name}: func is not callable"

    def test_get_unknown_method_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown upsampling method"):
            get_upsampling("nonexistent_method_xyz")

    def test_list_upsamplings_returns_list(self):
        result = list_upsamplings()
        assert isinstance(result, list)
        assert len(result) == len(UPSAMPLING_REGISTRY)


class TestRegistryMetadataCompleteness:
    """Ensure all methods have documentation metadata."""

    def test_decomp_methods_have_preserves_destroys(self):
        for name, method in DECOMPOSITION_REGISTRY.items():
            assert isinstance(method.preserves, str), f"{name}: missing preserves"
            assert isinstance(method.destroys, str), f"{name}: missing destroys"
            assert len(method.preserves) > 0, f"{name}: empty preserves"
            assert len(method.destroys) > 0, f"{name}: empty destroys"

    def test_upsamp_methods_have_preserves_introduces(self):
        for name, method in UPSAMPLING_REGISTRY.items():
            assert isinstance(method.preserves, str), f"{name}: missing preserves"
            assert isinstance(method.introduces, str), f"{name}: missing introduces"
            assert len(method.preserves) > 0, f"{name}: empty preserves"
            assert len(method.introduces) > 0, f"{name}: empty introduces"
