import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from stroke_ml.training.evaluate import (
    check_subgroup_recall_gaps,
    compute_primary_metrics,
    subgroup_analysis,
)


def _make_binary(n_pos=30, n_neg=270, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([1] * n_pos + [0] * n_neg)
    prob = rng.uniform(0.0, 1.0, len(y))
    return y, prob


def test_compute_primary_metrics_keys():
    y, prob = _make_binary()
    metrics = compute_primary_metrics(y, prob, threshold=0.3)
    for key in ("recall", "auroc", "auprc", "f2", "tp", "fp", "tn", "fn"):
        assert key in metrics


def test_recall_in_unit_interval():
    y, prob = _make_binary()
    metrics = compute_primary_metrics(y, prob, threshold=0.3)
    assert 0.0 <= metrics["recall"] <= 1.0


def test_check_subgroup_recall_gaps_flags_low_subgroup():
    subgroup_df = pd.DataFrame(
        {"recall": [0.80, 0.55, 0.78], "n": [100, 80, 90], "n_positive": [20, 15, 18]},
        index=["age < 40", "age 40–60", "age 60–80"],
    )
    flagged = check_subgroup_recall_gaps(subgroup_df, overall_recall=0.80, max_gap=0.10)
    assert "age 40–60" in flagged
    assert "age < 40" not in flagged


def test_check_subgroup_recall_gaps_empty_df():
    flagged = check_subgroup_recall_gaps(pd.DataFrame(), overall_recall=0.80)
    assert flagged == []


def test_subgroup_analysis_returns_dataframe():
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame({
        "age": rng.uniform(20, 85, n),
        "gender_Male": rng.integers(0, 2, n),
        "Residence_type": rng.integers(0, 2, n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    prob = rng.uniform(0, 1, n)
    df = subgroup_analysis(X, y, prob, threshold=0.3)
    assert isinstance(df, pd.DataFrame)
