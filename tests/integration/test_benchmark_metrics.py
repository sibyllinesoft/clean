"""Tests for clean.benchmarks.metrics — no HuggingFace download needed."""

import math

import numpy as np
import pytest

from sibylline_clean.benchmarks.metrics import (
    BenchmarkMetrics,
    compute_metrics,
    tpr_at_fpr_point,
)

# ---------------------------------------------------------------------------
# tpr_at_fpr_point
# ---------------------------------------------------------------------------


class TestTprAtFprPoint:
    def test_interpolation_on_known_curve(self):
        fpr = np.array([0.0, 0.1, 0.5, 1.0])
        tpr = np.array([0.0, 0.4, 0.8, 1.0])
        # Linear interpolation: at fpr=0.3, between (0.1,0.4) and (0.5,0.8)
        result = tpr_at_fpr_point(fpr, tpr, 0.3)
        assert abs(result - 0.6) < 1e-6

    def test_at_exact_point(self):
        fpr = np.array([0.0, 0.25, 0.5, 1.0])
        tpr = np.array([0.0, 0.5, 0.75, 1.0])
        assert abs(tpr_at_fpr_point(fpr, tpr, 0.25) - 0.5) < 1e-6

    def test_at_zero(self):
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.7, 1.0])
        assert tpr_at_fpr_point(fpr, tpr, 0.0) == 0.0

    def test_at_one(self):
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.7, 1.0])
        assert abs(tpr_at_fpr_point(fpr, tpr, 1.0) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_metrics — perfect classifier
# ---------------------------------------------------------------------------


class TestComputeMetricsPerfect:
    @pytest.fixture()
    def perfect(self):
        y_true = [1] * 50 + [0] * 50
        y_scores = [0.95] * 50 + [0.05] * 50
        return compute_metrics(y_true, y_scores, threshold=0.5)

    def test_auc_near_one(self, perfect):
        assert perfect.auc > 0.99

    def test_precision_recall_f1(self, perfect):
        assert perfect.precision == pytest.approx(1.0)
        assert perfect.recall == pytest.approx(1.0)
        assert perfect.f1 == pytest.approx(1.0)

    def test_tpr_at_fpr_high(self, perfect):
        for val in perfect.tpr_at_fpr.values():
            assert val > 0.9

    def test_counts(self, perfect):
        assert perfect.num_samples == 100
        assert perfect.num_positive == 50
        assert perfect.num_negative == 50

    def test_threshold_stored(self, perfect):
        assert perfect.threshold_used == 0.5


# ---------------------------------------------------------------------------
# compute_metrics — random classifier
# ---------------------------------------------------------------------------


class TestComputeMetricsRandom:
    @pytest.fixture()
    def random_clf(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=2000).tolist()
        y_scores = rng.uniform(0, 1, size=2000).tolist()
        return compute_metrics(y_true, y_scores, threshold=0.5)

    def test_auc_near_half(self, random_clf):
        assert 0.4 < random_clf.auc < 0.6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_positive(self):
        m = compute_metrics([1, 1, 1], [0.9, 0.8, 0.7], threshold=0.5)
        assert math.isnan(m.auc)
        assert all(math.isnan(v) for v in m.tpr_at_fpr.values())
        assert m.num_negative == 0

    def test_all_negative(self):
        m = compute_metrics([0, 0, 0], [0.1, 0.2, 0.05], threshold=0.5)
        assert math.isnan(m.auc)
        assert m.num_positive == 0

    def test_single_sample_positive(self):
        m = compute_metrics([1], [0.9], threshold=0.5)
        assert math.isnan(m.auc)
        assert m.num_samples == 1

    def test_single_sample_negative(self):
        m = compute_metrics([0], [0.1], threshold=0.5)
        assert math.isnan(m.auc)
        assert m.num_samples == 1


# ---------------------------------------------------------------------------
# BenchmarkMetrics fields
# ---------------------------------------------------------------------------


class TestBenchmarkMetricsFields:
    def test_all_fields_populated(self):
        m = compute_metrics(
            [1, 0, 1, 0, 1, 0],
            [0.8, 0.2, 0.7, 0.3, 0.9, 0.1],
            threshold=0.5,
        )
        assert isinstance(m, BenchmarkMetrics)
        assert isinstance(m.auc, float)
        assert isinstance(m.tpr_at_fpr, dict)
        assert isinstance(m.precision, float)
        assert isinstance(m.recall, float)
        assert isinstance(m.f1, float)
        assert isinstance(m.threshold_used, float)
        assert isinstance(m.num_samples, int)
        assert isinstance(m.num_positive, int)
        assert isinstance(m.num_negative, int)

    def test_custom_fpr_targets(self):
        m = compute_metrics(
            [1, 0, 1, 0],
            [0.9, 0.1, 0.8, 0.2],
            threshold=0.5,
            fpr_targets=[0.05, 0.10],
        )
        assert "5%" in m.tpr_at_fpr
        assert "10%" in m.tpr_at_fpr
        assert len(m.tpr_at_fpr) == 2
