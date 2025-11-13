from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_accuracy_metrics
from src.evaluation.fairness import fairness_at_target_rate, fairness_metrics
from src.evaluation.thresholds import threshold_for_acceptance_rate


def test_metrics_and_fairness_outputs():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    A = np.array([0, 1, 0, 1])

    accuracy = compute_accuracy_metrics(y_true, y_proba)
    for key in ("roc_auc", "pr_auc", "brier", "log_loss"):
        assert key in accuracy
        assert np.isfinite(accuracy[key])

    fairness = fairness_metrics(y_true, y_proba, A, threshold=0.5)
    expected_keys = {
        "eo_gap_tpr",
        "eo_gap_fpr",
        "dp_diff",
        "dp_ratio",
        "tpr_0",
        "tpr_1",
        "fpr_0",
        "fpr_1",
        "selection_rate_0",
        "selection_rate_1",
        "auc_roc_group0",
        "auc_roc_group1",
    }
    assert expected_keys.issubset(fairness.keys())

    for key in ("selection_rate_0", "selection_rate_1"):
        val = fairness[key]
        assert 0.0 <= val <= 1.0

    for key in ("eo_gap_tpr", "eo_gap_fpr"):
        val = fairness[key]
        assert 0.0 <= val <= 1.0

    assert fairness["selection_rate_0"] == 0.5
    assert fairness["selection_rate_1"] == 0.5


def test_threshold_for_acceptance_rate():
    y_proba = np.linspace(0.0, 1.0, 101)
    target_rate = 0.1
    thr = threshold_for_acceptance_rate(y_proba, target_rate)
    actual_rate = (y_proba >= thr).mean()
    assert actual_rate >= target_rate
    assert actual_rate - target_rate < 0.05


def test_fairness_at_target_rate():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.05, 0.9, 0.7, 0.2, 0.8, 0.4])
    A = np.array([0, 0, 1, 1, 0, 1])
    target_rate = 0.5

    metrics = fairness_at_target_rate(y_true, y_proba, A, target_rate)
    assert "threshold" in metrics and "actual_rate" in metrics
    assert abs(metrics["actual_rate"] - target_rate) < 0.2
    assert 0.0 <= metrics["threshold"] <= 1.0
