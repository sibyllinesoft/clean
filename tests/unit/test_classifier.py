"""Tests for classifier feature-schema compatibility."""

import numpy as np
import pytest

from sibylline_clean.classifier import InjectionClassifier
from sibylline_clean.patterns import PatternFeatures


class StubModel:
    def __init__(self, n_features):
        self.n_features_in_ = n_features
        self.received = None

    def predict_proba(self, features):
        self.received = features
        return np.array([[0.25, 0.75]])


def classifier_with_model(n_features):
    classifier = InjectionClassifier()
    classifier._model = StubModel(n_features)
    return classifier


def test_current_schema_is_passed_through():
    classifier = classifier_with_model(408)
    features = np.arange(408, dtype=np.float32)

    assert classifier.predict_proba(features) == 0.75
    np.testing.assert_array_equal(classifier._model.received[0], features)


def test_current_schema_is_adapted_for_bundled_legacy_model():
    classifier = classifier_with_model(405)
    features = np.arange(408, dtype=np.float32)
    omitted = [
        PatternFeatures.feature_names().index(name)
        for name in classifier._LEGACY_OMITTED_PATTERN_FEATURES
    ]

    assert classifier.predict_proba(features) == 0.75
    np.testing.assert_array_equal(classifier._model.received[0], np.delete(features, omitted))


def test_unknown_schema_mismatch_has_actionable_error():
    classifier = classifier_with_model(400)

    with pytest.raises(ValueError, match="Retrain the classifier"):
        classifier.predict_proba(np.zeros(408, dtype=np.float32))
