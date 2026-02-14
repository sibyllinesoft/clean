"""Dataset-agnostic metrics for benchmark evaluation.

Provides AUC, TPR at specified FPR thresholds, and standard
precision / recall / F1 at a given decision threshold.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

DEFAULT_FPR_TARGETS = [0.01, 0.005, 0.001, 0.0005]


@dataclass
class BenchmarkMetrics:
    """Container for benchmark evaluation results."""

    auc: float
    tpr_at_fpr: dict[str, float]
    precision: float
    recall: float
    f1: float
    threshold_used: float
    num_samples: int
    num_positive: int
    num_negative: int


def tpr_at_fpr_point(fpr_arr: np.ndarray, tpr_arr: np.ndarray, target_fpr: float) -> float:
    """Interpolate TPR at a specific FPR from an ROC curve.

    Args:
        fpr_arr: False positive rates from ``sklearn.metrics.roc_curve``.
        tpr_arr: True positive rates from ``sklearn.metrics.roc_curve``.
        target_fpr: The FPR at which to evaluate TPR.

    Returns:
        Interpolated TPR value.
    """
    return float(np.interp(target_fpr, fpr_arr, tpr_arr))


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_scores: list[float] | np.ndarray,
    threshold: float,
    fpr_targets: list[float] | None = None,
) -> BenchmarkMetrics:
    """Compute benchmark metrics from ground-truth labels and continuous scores.

    Args:
        y_true: Binary ground-truth labels (1 = injection, 0 = benign).
        y_scores: Continuous detection scores in [0, 1].
        threshold: Decision threshold for binary predictions.
        fpr_targets: FPR points at which to evaluate TPR.
            Defaults to [0.01, 0.005, 0.001, 0.0005].

    Returns:
        Populated :class:`BenchmarkMetrics` instance.
    """
    if fpr_targets is None:
        fpr_targets = DEFAULT_FPR_TARGETS

    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    num_positive = int(y_true.sum())
    num_negative = int(len(y_true) - num_positive)

    # AUC — requires both classes present
    if num_positive == 0 or num_negative == 0:
        auc = float("nan")
        tpr_fpr_map: dict[str, float] = {f"{t * 100:g}%": float("nan") for t in fpr_targets}
    else:
        auc = float(roc_auc_score(y_true, y_scores))
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
        tpr_fpr_map = {f"{t * 100:g}%": tpr_at_fpr_point(fpr_arr, tpr_arr, t) for t in fpr_targets}

    # P / R / F1 at the decision threshold
    y_pred = (y_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0.0,
    )

    return BenchmarkMetrics(
        auc=auc,
        tpr_at_fpr=tpr_fpr_map,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        threshold_used=threshold,
        num_samples=len(y_true),
        num_positive=num_positive,
        num_negative=num_negative,
    )
