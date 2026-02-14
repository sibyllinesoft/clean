"""Tests for the detection method registry."""

import pytest

from sibylline_clean.methods import (
    HeuristicMethod,
    SemiMarkovCRFMethod,
    get_method,
    list_methods,
    register_method,
)
from sibylline_clean.methods.base import DetectionMethod, MethodResult


class TestRegistry:
    """Tests for method registration and lookup."""

    def test_list_methods_includes_builtins(self):
        methods = list_methods()
        assert "heuristic" in methods
        assert "semi-markov-crf" in methods

    def test_get_method_heuristic(self):
        cls = get_method("heuristic")
        assert cls is HeuristicMethod

    def test_get_method_crf(self):
        cls = get_method("semi-markov-crf")
        assert cls is SemiMarkovCRFMethod

    def test_get_method_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown detection method"):
            get_method("nonexistent")

    def test_register_custom_method(self):
        class CustomMethod(DetectionMethod):
            @classmethod
            def name(cls) -> str:
                return "_test_custom"

            def analyze(self, text, normalized_text, include_matches=False):
                return MethodResult(score=0.0, pattern_features={})

            @property
            def mode(self) -> str:
                return "custom"

            @property
            def is_loaded(self) -> bool:
                return True

        register_method(CustomMethod)
        try:
            assert get_method("_test_custom") is CustomMethod
            assert "_test_custom" in list_methods()
        finally:
            # Clean up registry
            from sibylline_clean.methods import _REGISTRY

            _REGISTRY.pop("_test_custom", None)
