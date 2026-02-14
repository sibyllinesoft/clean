"""Content-type-aware scanning orchestrator.

Extracts strings from structured documents (JSON, CSV, XML, YAML),
runs prompt-injection detection on the joined virtual text, maps
detection spans back to original byte positions, and annotates/redacts
the result without breaking document structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..detector import InjectionDetector
from . import annotators
from .extractors import get_extractor
from .heat import HeatScorer, HeatSpan
from .spans import OriginalSpan, SpanMap


class AnnotationMode(Enum):
    STRUCTURED = "structured"
    CLI = "cli"
    PLAIN = "plain"


@dataclass
class Detection:
    """A single detection mapped back to original document positions."""

    category: str
    heat: float
    original_spans: list[OriginalSpan]
    text_snippet: str


@dataclass
class ScanResult:
    """Full result of a content scan."""

    flagged: bool
    score: float
    detections: list[Detection]
    annotated: bytes
    """Annotated/redacted document bytes."""

    metadata: dict = field(default_factory=dict)
    """Sidecar metadata (populated in PLAIN mode, or with extra info)."""

    content_type: str = ""


class ContentScanner:
    """Orchestrates the content-aware scanning pipeline.

    Pipeline:
        1. Select extractor by content type
        2. Extract strings with byte offsets → ``SpanMap``
        3. Run ``InjectionDetector.analyze()`` on virtual text
        4. If flagged: score sub-spans with ``HeatScorer``
        5. Filter to hot spans (``heat >= heat_threshold``)
        6. Map back to original byte positions
        7. Annotate/redact via mode-specific formatter
        8. Unknown content types fall back to plain-text scanning
    """

    def __init__(
        self,
        detector: InjectionDetector | None = None,
        threshold: float = 0.3,
        heat_threshold: float = 1.5,
        mode: AnnotationMode = AnnotationMode.STRUCTURED,
        languages: list[str] | None = None,
        use_embeddings: bool = True,
    ) -> None:
        self._detector = detector or InjectionDetector(
            threshold=threshold,
            use_embeddings=use_embeddings,
            languages=languages,
        )
        self._threshold = threshold
        self._heat_threshold = heat_threshold
        self._mode = mode
        self._heat_scorer = HeatScorer(languages=languages)

    def scan(self, raw: bytes, content_type: str) -> ScanResult:
        """Extract → detect → heat-score → map back → annotate/redact."""
        extractor_cls = get_extractor(content_type)

        if extractor_cls is None:
            return self._scan_plain_text(raw, content_type)

        # Step 1-2: Extract strings and build virtual text
        extractor = extractor_cls()
        entries = extractor.extract(raw)

        if not entries:
            return ScanResult(
                flagged=False,
                score=0.0,
                detections=[],
                annotated=raw,
                content_type=content_type,
            )

        span_map = SpanMap.build(entries)

        # Step 3: Run detection on virtual text
        analysis = self._detector.analyze(span_map.virtual_text)

        if not analysis.flagged:
            return ScanResult(
                flagged=False,
                score=analysis.score,
                detections=[],
                annotated=raw,
                content_type=content_type,
            )

        # Step 4: Fine-grained heat scoring on virtual text
        heat_spans = self._heat_scorer.score(span_map.virtual_text)

        # Step 5: Filter to hot spans
        hot_spans = [s for s in heat_spans if s.heat >= self._heat_threshold]

        # Step 6: Map back to original byte positions
        detections = self._build_detections(hot_spans, span_map)

        if not detections:
            # Flagged by detector but no spans exceeded heat threshold —
            # still report as flagged with score but no redaction
            return ScanResult(
                flagged=True,
                score=analysis.score,
                detections=[],
                annotated=raw,
                metadata={"note": "flagged but no spans exceeded heat threshold"},
                content_type=content_type,
            )

        # Step 7: Annotate/redact
        annotated, metadata = self._annotate(raw, detections, analysis.score, content_type)

        return ScanResult(
            flagged=True,
            score=analysis.score,
            detections=detections,
            annotated=annotated,
            metadata=metadata,
            content_type=content_type,
        )

    def _scan_plain_text(self, raw: bytes, content_type: str) -> ScanResult:
        """Fallback: scan as plain text without structured extraction."""
        text = raw.decode("utf-8", errors="replace")
        analysis = self._detector.analyze(text)

        return ScanResult(
            flagged=analysis.flagged,
            score=analysis.score,
            detections=[],
            annotated=raw,
            metadata={"mode": "plain_text_fallback"},
            content_type=content_type,
        )

    def _build_detections(
        self,
        hot_spans: list[HeatSpan],
        span_map: SpanMap,
    ) -> list[Detection]:
        """Map hot virtual-text spans back to original document positions."""
        detections: list[Detection] = []

        for hs in hot_spans:
            original_spans = span_map.map_span(hs.virt_start, hs.virt_end)
            if not original_spans:
                continue

            snippet = span_map.virtual_text[hs.virt_start : hs.virt_end]

            detections.append(
                Detection(
                    category=hs.category,
                    heat=hs.heat,
                    original_spans=original_spans,
                    text_snippet=snippet,
                )
            )

        return detections

    def _annotate(
        self,
        raw: bytes,
        detections: list[Detection],
        score: float,
        content_type: str,
    ) -> tuple[bytes, dict]:
        """Dispatch to the appropriate annotator based on mode."""
        if self._mode is AnnotationMode.STRUCTURED:
            annotated = annotators.annotate_structured(raw, detections, content_type, score)
            return annotated, {}

        if self._mode is AnnotationMode.CLI:
            annotated = annotators.annotate_cli(raw, detections)
            return annotated, {}

        # PLAIN
        return annotators.annotate_plain(raw, detections, score, content_type)
