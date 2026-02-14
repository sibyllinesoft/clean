"""Heuristic detection method — wraps the existing pattern/ML pipeline."""

from ..classifier import InjectionClassifier, PatternOnlyClassifier
from ..motifs import MotifFeatureExtractor, MotifMatcher
from ..patterns import PatternExtractor
from ..windowing import AdaptiveWindowAnalyzer
from .base import DetectionMethod, MethodResult


class HeuristicMethod(DetectionMethod):
    """Pattern + ML heuristic detection pipeline.

    Combines text normalization, regex pattern extraction, fuzzy motif matching,
    optional sliding-window analysis, and optional embedding-based ML classification.
    Falls back to pattern-only scoring when ML components are unavailable.
    """

    def __init__(
        self,
        use_embeddings: bool = True,
        use_windowing: bool = True,
        lazy_load: bool = True,
        languages: list[str] | None = None,
        app_name: str = "clean",
        **kwargs,
    ):
        self._use_embeddings = use_embeddings
        self._use_windowing = use_windowing
        self._languages = languages
        self._app_name = app_name

        # Always-available components
        self._pattern_extractor = PatternExtractor(languages=languages, app_name=app_name)
        self._pattern_classifier = PatternOnlyClassifier(threshold=0.3)

        # Motif-based detection
        self._motif_matcher = MotifMatcher(threshold=75, languages=languages, app_name=app_name)
        self._motif_extractor = MotifFeatureExtractor(
            threshold=75, languages=languages, app_name=app_name
        )

        # Windowing analyzer for long texts
        self._window_analyzer = AdaptiveWindowAnalyzer(
            coarse_window=4096,
            coarse_step=2048,
            fine_window=512,
            fine_step=256,
            hotspot_threshold=0.3,
            batch_max_size=1024,
            batch_gap_tolerance=256,
        )

        # Heavy components (lazy-loaded)
        self._embedder = None
        self._classifier = None
        self._load_failed = False

        if not lazy_load:
            self._ensure_heavy_components()

    @classmethod
    def name(cls) -> str:
        return "heuristic"

    @property
    def mode(self) -> str:
        if self._embedder is not None:
            return "ml"
        return "pattern-only"

    @property
    def is_loaded(self) -> bool:
        return True

    def _ensure_heavy_components(self) -> bool:
        """Try to load embedder and classifier. Returns True if successful."""
        if self._load_failed:
            return False

        if self._embedder is not None and self._classifier is not None:
            return True

        try:
            from ..embedder import MiniLMEmbedder

            self._embedder = MiniLMEmbedder()
            self._classifier = InjectionClassifier()
            return True
        except (ImportError, RuntimeError):
            self._load_failed = True
            return False

    def _find_pattern_spans(self, text: str) -> list[tuple[int, int]]:
        """Find all pattern match spans in original text."""
        # Use Rust-accelerated span finding when available
        pattern_spans = self._pattern_extractor.find_pattern_spans(text)
        motif_spans = self._motif_matcher.get_match_positions(text)

        if not pattern_spans and not motif_spans:
            return []

        # Merge both span sources
        spans: set[tuple[int, int]] = set()
        for start, end in pattern_spans:
            spans.add((start, end))
        for start, end in motif_spans:
            spans.add((max(0, start), min(len(text), end)))

        sorted_spans = sorted(spans)
        merged = [sorted_spans[0]]

        for start, end in sorted_spans[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        return merged

    def analyze(
        self, text: str, normalized_text: str, include_matches: bool = False
    ) -> MethodResult:
        """Analyze text for prompt injection using heuristic pipeline."""
        import numpy as np

        normalized = normalized_text

        # Extract pattern features
        pattern_features = self._pattern_extractor.extract(normalized)

        # Extract motif features
        motif_features = self._motif_extractor.extract(normalized)

        # Early exit: if no suspicious patterns detected, skip expensive embedding
        pattern_score = self._pattern_classifier.predict_proba(pattern_features)
        motif_signal = self._motif_matcher.compute_signal(normalized)
        has_suspicious_patterns = (
            pattern_features.instruction_override > 0
            or pattern_features.role_injection > 0
            or pattern_features.system_manipulation > 0
            or pattern_features.prompt_leak > 0
            or pattern_features.jailbreak_keywords > 0
            or pattern_features.encoding_markers > 0
            or pattern_features.suspicious_delimiters > 0
            or motif_signal.density > 0
        )

        if not has_suspicious_patterns and pattern_score < 0.1:
            return MethodResult(
                score=pattern_score,
                pattern_features={
                    "instruction_override": pattern_features.instruction_override,
                    "role_injection": pattern_features.role_injection,
                    "system_manipulation": pattern_features.system_manipulation,
                    "prompt_leak": pattern_features.prompt_leak,
                    "jailbreak_keywords": pattern_features.jailbreak_keywords,
                    "encoding_markers": pattern_features.encoding_markers,
                    "suspicious_delimiters": pattern_features.suspicious_delimiters,
                },
                matched_patterns={},
                matched_spans=[],
            )

        # For long texts, use windowing to find hotspots
        hotspot_regions = []
        if self._use_windowing and len(text) > 4096:
            analysis_result = self._window_analyzer.analyze(normalized)
            hotspot_regions = analysis_result.batched_regions

        # Try to use full ML pipeline
        if self._use_embeddings and self._ensure_heavy_components():
            pattern_spans = self._find_pattern_spans(normalized) if has_suspicious_patterns else []

            if pattern_spans and len(normalized) > 1024:
                WINDOW_SIZE = 512
                MERGE_GAP = 256
                windows = []

                for start, end in pattern_spans:
                    center = (start + end) // 2
                    win_start = max(0, center - WINDOW_SIZE // 2)
                    win_end = min(len(normalized), center + WINDOW_SIZE // 2)

                    if windows and win_start - windows[-1][1] < MERGE_GAP:
                        windows[-1] = (windows[-1][0], win_end)
                    else:
                        windows.append((win_start, win_end))

                region_embeddings = []
                for start, end in windows:
                    region_text = normalized[start:end]
                    if len(region_text) > 20:
                        region_embeddings.append(self._embedder.embed(region_text))

                if region_embeddings:
                    embedding = np.max(region_embeddings, axis=0)
                else:
                    embedding = self._embedder.embed(normalized[:1024])
            elif hotspot_regions:
                region_embeddings = []
                for start, end in hotspot_regions:
                    region_text = normalized[start:end]
                    region_embeddings.append(self._embedder.embed(region_text))

                if region_embeddings:
                    embedding = np.max(region_embeddings, axis=0)
                else:
                    embedding = self._embedder.embed(normalized[:4096])
            else:
                embedding = self._embedder.embed(normalized[:2048])

            pattern_array = np.array(pattern_features.to_array(), dtype=np.float32)
            motif_array = np.array(motif_features, dtype=np.float32)
            features = np.concatenate([pattern_array, motif_array, embedding])

            score = self._classifier.predict_proba(features)
        else:
            motif_signal = self._motif_matcher.compute_signal(normalized)
            motif_boost = min(motif_signal.density / 10, 0.3)
            base_score = self._pattern_classifier.predict_proba(pattern_features)
            score = min(base_score + motif_boost, 1.0)

        # Get pattern matches if requested
        matched_patterns: dict[str, list[str]] = {}
        if include_matches:
            matched_patterns = self._pattern_extractor.get_matches(normalized)

        # Find spans on original text (always compute when suspicious;
        # the detector decides whether to use them based on threshold)
        matched_spans = self._find_pattern_spans(text)

        return MethodResult(
            score=score,
            pattern_features={
                "instruction_override": pattern_features.instruction_override,
                "role_injection": pattern_features.role_injection,
                "system_manipulation": pattern_features.system_manipulation,
                "prompt_leak": pattern_features.prompt_leak,
                "jailbreak_keywords": pattern_features.jailbreak_keywords,
                "encoding_markers": pattern_features.encoding_markers,
                "suspicious_delimiters": pattern_features.suspicious_delimiters,
            },
            matched_patterns=matched_patterns,
            matched_spans=matched_spans,
        )
