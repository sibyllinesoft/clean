"""Base classes for detection methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MethodResult:
    """Raw output from a detection method.

    Methods return this; InjectionDetector wraps it into InjectionAnalysis
    by adding threshold/flagged policy.
    """

    score: float
    """Probability of prompt injection (0.0 to 1.0)."""

    pattern_features: dict[str, float]
    """Individual pattern category scores."""

    matched_patterns: dict[str, list[str]] = field(default_factory=dict)
    """Actual pattern matches by category (for debugging)."""

    matched_spans: list[tuple[int, int]] = field(default_factory=list)
    """Character spans (start, end) of all matched patterns in original text."""


class DetectionMethod(ABC):
    """Abstract base class for prompt injection detection methods.

    Each method implements a different detection strategy. Methods are
    registered in the method registry and selected by name.
    """

    default_threshold: float | None = None
    """Method-specific default threshold. When set, InjectionDetector uses
    this instead of its own default when the caller hasn't overridden."""

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the unique name of this detection method."""
        ...

    @abstractmethod
    def analyze(
        self, text: str, normalized_text: str, include_matches: bool = False
    ) -> MethodResult:
        """Analyze text for prompt injection.

        Args:
            text: Original text content.
            normalized_text: Text after normalization (obfuscation defeated).
            include_matches: Whether to include actual pattern matches.

        Returns:
            MethodResult with detection scores and matches.
        """
        ...

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return the current operating mode (e.g. 'pattern-only', 'ml')."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return whether this method is ready for use."""
        ...
