"""Content-type-aware scanning for structured documents."""

from .extractors import get_extractor, register_extractor
from .scanner import AnnotationMode, ContentScanner, Detection, ScanResult
from .spans import ExtractedString, OriginalSpan, SpanMap

__all__ = [
    "ContentScanner",
    "ScanResult",
    "AnnotationMode",
    "Detection",
    "ExtractedString",
    "SpanMap",
    "OriginalSpan",
    "register_extractor",
    "get_extractor",
]
