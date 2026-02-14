"""Fine-grained sub-span heat scoring.

Uses :class:`~sibylline_clean.patterns.PatternExtractor` and
:class:`~sibylline_clean.motifs.MotifMatcher` to identify precise sub-spans
of injected content, then weights them by category severity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..motifs import MotifMatcher
from ..patterns import PatternExtractor

# Weights sourced from PatternOnlyClassifier.WEIGHTS
CATEGORY_WEIGHTS: dict[str, float] = {
    "instruction_override": 3.0,
    "role_injection": 2.5,
    "system_manipulation": 3.0,
    "prompt_leak": 2.0,
    "jailbreak_keywords": 2.5,
    "encoding_markers": 1.0,
    "suspicious_delimiters": 1.5,
    # Motif-only categories (mapped from motif library keys)
    "jailbreak": 2.5,
    "delimiters": 1.5,
}


@dataclass(slots=True)
class HeatSpan:
    """A sub-span with a weighted heat score."""

    virt_start: int
    virt_end: int
    heat: float
    """Weighted score — higher means more confident injection signal."""

    category: str
    """Dominant category for this span."""

    sources: list[str] = field(default_factory=list)
    """Detection sources: ``["pattern"]``, ``["motif"]``, or both."""


class HeatScorer:
    """Score sub-spans of virtual text by injection severity.

    Runs pattern extraction and motif matching on the virtual text, then
    assigns a weighted heat value to each detected region. Overlapping
    spans from different sources are merged, keeping the maximum heat.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        app_name: str = "clean",
    ) -> None:
        self._pattern_extractor = PatternExtractor(languages=languages, app_name=app_name)
        self._motif_matcher = MotifMatcher(threshold=75, languages=languages, app_name=app_name)

    def score(self, text: str) -> list[HeatSpan]:
        """Find and score all sub-spans in *text*.

        Returns a sorted, merged list of :class:`HeatSpan` objects.
        """
        raw_spans: list[HeatSpan] = []

        # Pattern spans (with category information)
        for start, end, category in self._pattern_extractor.find_categorized_spans(text):
            weight = CATEGORY_WEIGHTS.get(category, 1.0)
            raw_spans.append(
                HeatSpan(
                    virt_start=start,
                    virt_end=end,
                    heat=weight,
                    category=category,
                    sources=["pattern"],
                )
            )

        # Motif spans
        for match in self._motif_matcher.find_matches(text):
            weight = CATEGORY_WEIGHTS.get(match.category, 1.0)
            heat = weight * (match.score / 100.0)
            raw_spans.append(
                HeatSpan(
                    virt_start=match.position,
                    virt_end=match.position + match.length,
                    heat=heat,
                    category=match.category,
                    sources=["motif"],
                )
            )

        if not raw_spans:
            return []

        # Sort by start position
        raw_spans.sort(key=lambda s: (s.virt_start, s.virt_end))

        # Merge overlapping spans, taking max heat and combining sources
        merged: list[HeatSpan] = [raw_spans[0]]
        for span in raw_spans[1:]:
            last = merged[-1]
            if span.virt_start <= last.virt_end:
                # Overlapping — extend and take max heat
                new_end = max(last.virt_end, span.virt_end)
                new_heat = max(last.heat, span.heat)
                new_cat = last.category if last.heat >= span.heat else span.category
                new_sources = list(set(last.sources + span.sources))
                merged[-1] = HeatSpan(
                    virt_start=last.virt_start,
                    virt_end=new_end,
                    heat=new_heat,
                    category=new_cat,
                    sources=new_sources,
                )
            else:
                merged.append(span)

        return merged
