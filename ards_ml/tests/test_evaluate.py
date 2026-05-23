import numpy as np
import pytest

from ..training.evaluate import (
    check_performance_targets,
    check_subgroup_gaps,
    compute_primary_metrics,
    compute_secondary_metrics,
)

import pandas as pd


def _perfect_case():
    y_true = np.array([0] * 70 + [1] * 30)
    y_prob = np.array([0.1] * 70 + [0.9] * 30)
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
        "auroc": 0.90, "auprc": 0.75,
        "sensitivity": 0.82, "specificity": 0.78, "brier_score": 0.15,
    }
    targets = {
        "auroc": 0.85, "auprc": 0.70,
        "sensitivity": 0.80, "specificity": 0.75, "brier_score": 0.20,
    }
    checks = check_performance_targets(metrics, targets)
    assert all(checks.values())


def test_performance_targets_fail():
    metrics = {
        "auroc": 0.80, "auprc": 0.65,  # both below target
        "sensitivity": 0.85, "specificity": 0.80, "brier_score": 0.15,
    }
    targets = {
        "auroc": 0.85, "auprc": 0.70,
        "sensitivity": 0.80, "specificity": 0.75, "brier_score": 0.20,
    }
    checks = check_performance_targets(metrics, targets)
    assert not checks["auroc"]
    assert not checks["auprc"]


def test_subgroup_gap_detection():
    subgroup_df = pd.DataFrame(
        {"auroc": {"Age < 50": 0.75, "Age 50–70": 0.88}},
    )
    flagged = check_subgroup_gaps(subgroup_df, overall_auroc=0.87, max_gap=0.05)
    assert "Age < 50" in flagged
    assert "Age 50–70" not in flagged
