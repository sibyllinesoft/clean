"""Shared test fixtures for sibylline-clean."""

import pytest

from sibylline_clean.normalizer import TextNormalizer
from sibylline_clean.patterns import PatternExtractor


@pytest.fixture
def normalizer():
    """Create a TextNormalizer instance."""
    return TextNormalizer()


@pytest.fixture
def pattern_extractor():
    """Create a PatternExtractor instance with default English patterns."""
    return PatternExtractor()
