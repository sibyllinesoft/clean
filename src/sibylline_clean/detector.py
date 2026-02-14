"""Core prompt injection detection engine."""

from dataclasses import dataclass
from typing import Any

from .methods import get_method
from .normalizer import TextNormalizer


@dataclass
class InjectionAnalysis:
    """Results of prompt injection analysis."""

    score: float
    """Probability of prompt injection (0.0 to 1.0)."""

    threshold: float
    """Detection threshold used."""

    flagged: bool
    """Whether the score exceeds the threshold."""

    pattern_features: dict[str, float]
    """Individual pattern category scores."""

    matched_patterns: dict[str, list[str]]
    """Actual pattern matches by category (for debugging)."""

    matched_spans: list[tuple[int, int]]
    """Character spans (start, end) of all matched patterns in original text."""


class InjectionDetector:
    """Detect prompt injection attempts in text content.

    Delegates to a pluggable DetectionMethod selected by name.
    The default 'heuristic' method uses pattern extraction, motif matching,
    and optional ML classification. Alternative methods can be registered
    via the methods registry.
    """

    _DEFAULT_THRESHOLD = 0.3

    def __init__(
        self,
        threshold: float | None = None,
        method: str = "heuristic",
        *,
        use_embeddings: bool = True,
        use_windowing: bool = True,
        lazy_load: bool = True,
        languages: list[str] | None = None,
        app_name: str = "clean",
        **method_kwargs: Any,
    ):
        """Initialize the detector.

        Args:
            threshold: Detection threshold (0.0 to 1.0). Higher = fewer false positives.
                       If not set, uses the method's default_threshold or 0.3.
            method: Name of the detection method to use (default: "heuristic").
            use_embeddings: Whether to use embedding-based detection (heuristic method).
            use_windowing: Whether to use sliding window analysis (heuristic method).
            lazy_load: Whether to lazy-load heavy components.
            languages: List of language codes for pattern detection.
            app_name: Application name for config directory resolution.
            **method_kwargs: Additional keyword arguments passed to the method constructor.
        """
        self._normalizer = TextNormalizer()

        # Look up and instantiate the detection method
        method_cls = get_method(method)
        self._method = method_cls(
            use_embeddings=use_embeddings,
            use_windowing=use_windowing,
            lazy_load=lazy_load,
            languages=languages,
            app_name=app_name,
            **method_kwargs,
        )

        # Use explicit threshold > method default > global default
        if threshold is not None:
            self.threshold = threshold
        elif getattr(self._method, "default_threshold", None) is not None:
            self.threshold = self._method.default_threshold
        else:
            self.threshold = self._DEFAULT_THRESHOLD

    def analyze(self, text: str, include_matches: bool = False) -> InjectionAnalysis:
        """Analyze text for prompt injection.

        Args:
            text: Text content to analyze.
            include_matches: Whether to include actual pattern matches
                            (useful for debugging but slower).

        Returns:
            InjectionAnalysis with detection results.
        """
        normalized = self._normalizer.normalize(text)
        result = self._method.analyze(text, normalized, include_matches=include_matches)

        flagged = result.score >= self.threshold

        return InjectionAnalysis(
            score=result.score,
            threshold=self.threshold,
            flagged=flagged,
            pattern_features=result.pattern_features,
            matched_patterns=result.matched_patterns,
            matched_spans=result.matched_spans if flagged else [],
        )

    @property
    def mode(self) -> str:
        """Return current detection mode."""
        return self._method.mode
