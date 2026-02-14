"""Clean: Prompt injection detection library for LLM applications."""

from .detector import InjectionAnalysis, InjectionDetector
from .methods import DetectionMethod, MethodResult, get_method, list_methods, register_method
from .motifs import HAS_RAPIDFUZZ, MotifFeatureExtractor, MotifMatcher, MotifSignal
from .normalizer import TextNormalizer
from .patterns import PatternExtractor, PatternFeatures
from .windowing import AdaptiveWindowAnalyzer, SlidingWindowAnalyzer

__all__ = [
    "InjectionDetector",
    "InjectionAnalysis",
    "TextNormalizer",
    "PatternExtractor",
    "PatternFeatures",
    "MotifMatcher",
    "MotifFeatureExtractor",
    "MotifSignal",
    "SlidingWindowAnalyzer",
    "AdaptiveWindowAnalyzer",
    "HAS_RAPIDFUZZ",
    "DetectionMethod",
    "MethodResult",
    "get_method",
    "list_methods",
    "register_method",
    "ContentScanner",
]

_CONTENT_NAMES = {
    "ContentScanner",
    "ScanResult",
    "AnnotationMode",
    "Detection",
    "ExtractedString",
    "SpanMap",
    "OriginalSpan",
    "register_extractor",
    "get_extractor",
}


def __getattr__(name: str):
    if name in _CONTENT_NAMES:
        # Cache on module to avoid repeated imports
        import sys

        from .content import (  # noqa: F811
            AnnotationMode,
            ContentScanner,
            Detection,
            ExtractedString,
            OriginalSpan,
            ScanResult,
            SpanMap,
            get_extractor,
            register_extractor,
        )

        mod = sys.modules[__name__]
        for n, v in {
            "ContentScanner": ContentScanner,
            "ScanResult": ScanResult,
            "AnnotationMode": AnnotationMode,
            "Detection": Detection,
            "ExtractedString": ExtractedString,
            "SpanMap": SpanMap,
            "OriginalSpan": OriginalSpan,
            "register_extractor": register_extractor,
            "get_extractor": get_extractor,
        }.items():
            setattr(mod, n, v)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
