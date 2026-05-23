import numpy as np
import pandas as pd
import pytest

from ..training.evaluate import (
    check_performance_targets,
    check_subgroup_gaps,
    compute_primary_metrics,
    compute_secondary_metrics,
)


def _perfect_case():
    y_true = np.array([0] * 64 + [1] * 36)
    y_prob = np.array([0.1] * 64 + [0.9] * 36)
    return y_true, y_prob


def test_primary_metrics_perfect():
    y_true, y_prob = _perfect_case()
    m = compute_primary_metrics(y_true, y_prob, threshold=0.5)
    assert m["auroc"] == pytest.approx(1.0)
    assert m["sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["brier_score"] < 0.05


def test_secondary_metrics_keys():
    y_true, y_prob = _perfect_case()
    m = compute_secondary_metrics(y_true, y_prob, threshold=0.5)
    for key in ("f1", "mcc", "precision", "false_alarm_rate_per_100"):
        assert key in m


def test_accuracy_not_in_metrics():
    y_true, y_prob = _perfect_case()
    primary = compute_primary_metrics(y_true, y_prob, threshold=0.5)
    secondary = compute_secondary_metrics(y_true, y_prob, threshold=0.5)
    assert "accuracy" not in primary
    assert "accuracy" not in secondary


def test_performance_targets_pass():
    metrics = {
        "auroc": 0.80, "auprc": 0.55,
        "sensitivity": 0.82, "specificity": 0.70, "brier_score": 0.20,
    }
    targets = {
        "auroc": 0.75, "auprc": 0.50,
        "sensitivity": 0.80, "specificity": 0.65, "brier_score": 0.25,
    }
    checks = check_performance_targets(metrics, targets)
    assert all(checks.values())


def test_performance_targets_fail():
    metrics = {
        "auroc": 0.60, "auprc": 0.40,
        "sensitivity": 0.85, "specificity": 0.70, "brier_score": 0.20,
    }
    targets = {
        "auroc": 0.75, "auprc": 0.50,
        "sensitivity": 0.80, "specificity": 0.65, "brier_score": 0.25,
    }
    checks = check_performance_targets(metrics, targets)
    assert not checks["auroc"]
    assert not checks["auprc"]


def test_subgroup_gap_detection():
    subgroup_df = pd.DataFrame(
        {"auroc": {"Age < 40": 0.72, "Age 40-60": 0.89}},
    )
    flagged = check_subgroup_gaps(subgroup_df, overall_auroc=0.85, max_gap=0.05)
    assert "Age < 40" in flagged
    assert "Age 40-60" not in flagged
