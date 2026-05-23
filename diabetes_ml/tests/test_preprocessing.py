"""
Unit tests for preprocessing and clinical rule logic.
Run with: pytest diabetes_ml/tests/test_preprocessing.py -v
"""
import numpy as np
import pandas as pd
import pytest

from diabetes_ml.pipeline.pipeline import BMICapper, build_pipeline, build_preprocessor
from diabetes_ml.training.evaluate import clinical_risk_band
from diabetes_ml.training.data import split_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal synthetic dataset with the exact schema of the real dataset."""
    rng = np.random.default_rng(0)
    n = 200
    positive = 17  # ~8.5% to match real imbalance
    y = np.array([1] * positive + [0] * (n - positive))
    return pd.DataFrame({
        "gender": rng.choice(["Male", "Female", "Other"], n),
        "age": rng.uniform(0.1, 80.0, n),
        "hypertension": rng.integers(0, 2, n),
        "heart_disease": rng.integers(0, 2, n),
        "smoking_history": rng.choice(
            ["never", "No Info", "current", "former", "not current", "ever"], n
        ),
        "bmi": np.append(rng.uniform(15.0, 45.0, n - 5), [70.0, 80.0, 85.0, 90.0, 95.7]),
        "HbA1c_level": rng.uniform(3.5, 9.0, n),
        "blood_glucose_level": rng.uniform(80, 300, n),
        "diabetes": y,
    })


# ---------------------------------------------------------------------------
# BMICapper
# ---------------------------------------------------------------------------

def test_bmi_capper_clips_outliers(sample_df):
    X = sample_df.drop(columns=["diabetes"])
    capper = BMICapper(percentile=99)
    capper.fit(X)
    X_out = capper.transform(X)
    assert X_out["bmi"].max() <= capper.cap_ + 1e-9


def test_bmi_capper_does_not_touch_other_columns(sample_df):
    X = sample_df.drop(columns=["diabetes"])
    capper = BMICapper(percentile=99)
    capper.fit(X)
    X_out = capper.transform(X)
    pd.testing.assert_series_equal(X["age"], X_out["age"])


def test_bmi_capper_cap_fitted_on_train_not_test(sample_df):
    """Cap must come from training data; test values above the cap are clipped."""
    X_train = sample_df.drop(columns=["diabetes"]).iloc[:100].copy()
    X_test = sample_df.drop(columns=["diabetes"]).iloc[100:].copy()
    X_test.loc[X_test.index[0], "bmi"] = 999.0  # extreme outlier in test

    capper = BMICapper(percentile=99)
    capper.fit(X_train)
    X_test_out = capper.transform(X_test)
    assert X_test_out["bmi"].max() <= capper.cap_ + 1e-9


# ---------------------------------------------------------------------------
# Preprocessor / feature dimensions
# ---------------------------------------------------------------------------

def test_preprocessor_output_shape(sample_df):
    """Pipeline should produce exactly 14 features after encoding."""
    # 4 scaled numerics + 2 gender (drop=first from 3) + 6 smoking + 2 binary = 14
    X = sample_df.drop(columns=["diabetes"])
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    pipeline = build_pipeline(lr)
    pipeline.fit(X, sample_df["diabetes"])
    # Extract output of the preprocessor step (before classifier)
    preprocessor_pipeline = pipeline[:-1]
    X_out = preprocessor_pipeline.transform(X)
    assert X_out.shape[1] == 14


def test_smoking_no_info_preserved(sample_df):
    """'No Info' smoking category must appear as its own OHE column."""
    from sklearn.linear_model import LogisticRegression
    X = sample_df.drop(columns=["diabetes"])
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, sample_df["diabetes"])
    ct = pipeline.named_steps["preprocessor"]
    smoking_encoder = ct.named_transformers_["smoking"]
    categories = list(smoking_encoder.categories_[0])
    assert "No Info" in categories


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def test_split_preserves_positive_rate(sample_df):
    """Stratified split must keep ~8.5% positive rate in every partition."""
    X = sample_df.drop(columns=["diabetes"])
    y = sample_df["diabetes"]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    overall_rate = y.mean()
    for part_y, name in [(y_train, "train"), (y_val, "val"), (y_test, "test")]:
        rate = part_y.mean()
        assert abs(rate - overall_rate) < 0.03, (
            f"{name} positive rate {rate:.3f} deviates >3% from overall {overall_rate:.3f}"
        )


# ---------------------------------------------------------------------------
# Clinical rule hard flags
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("glucose,hba1c,expected_flag", [
    (250, 5.0, True),   # glucose hard flag
    (150, 6.5, True),   # HbA1c hard flag
    (250, 6.5, True),   # both
    (150, 5.0, False),  # neither
    (199, 6.49, False), # just below both thresholds
])
def test_clinical_hard_flag(glucose, hba1c, expected_flag):
    result = clinical_risk_band(0.3, blood_glucose=glucose, hba1c=hba1c)
    assert result["hard_flag"] == expected_flag


@pytest.mark.parametrize("prob,expected_band", [
    (0.60, "High"),
    (0.50, "High"),
    (0.40, "Moderate"),
    (0.25, "Moderate"),
    (0.24, "Low"),
    (0.00, "Low"),
])
def test_clinical_risk_bands(prob, expected_band):
    result = clinical_risk_band(prob)
    assert result["risk_band"] == expected_band


def test_clinical_output_is_labelled_decision_support():
    result = clinical_risk_band(0.8)
    assert "decision-support" in result["label"].lower() or "not a diagnosis" in result["label"].lower()
