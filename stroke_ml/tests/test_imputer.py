import numpy as np
import pandas as pd
import pytest

from stroke_ml.pipeline.imputer import AgeDecileBMIImputer

TRAIN_DF = pd.DataFrame({
    "age": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 25.0, 35.0, 45.0, 55.0, 65.0],
    "bmi": [22.0, 24.0, np.nan, 28.0, 30.0, 32.0, 34.0, 23.0, 25.0, 27.0, 29.0, 31.0],
    "smoking_status": ["smokes"] * 12,
})


def test_bmi_was_missing_indicator():
    imp = AgeDecileBMIImputer()
    imp.fit(TRAIN_DF)
    out = imp.transform(TRAIN_DF)
    # Row 2 has null bmi → indicator should be 1
    assert out.loc[2, "bmi_was_missing"] == 1
    assert out.loc[0, "bmi_was_missing"] == 0


def test_no_null_bmi_after_transform():
    imp = AgeDecileBMIImputer()
    imp.fit(TRAIN_DF)
    out = imp.transform(TRAIN_DF)
    assert out["bmi"].isnull().sum() == 0


def test_smoking_status_unchanged():
    imp = AgeDecileBMIImputer()
    imp.fit(TRAIN_DF)
    out = imp.transform(TRAIN_DF)
    assert "smoking_status" in out.columns
    assert list(out["smoking_status"]) == list(TRAIN_DF["smoking_status"])


def test_decile_medians_from_training_only():
    # Fit on train, transform on a different df with all null BMI
    imp = AgeDecileBMIImputer()
    imp.fit(TRAIN_DF)

    test_df = pd.DataFrame({
        "age": [22.0, 52.0, 72.0],
        "bmi": [np.nan, np.nan, np.nan],
        "smoking_status": ["Unknown", "Unknown", "Unknown"],
    })
    out = imp.transform(test_df)
    # All BMIs must be filled using training medians (no NaN remaining)
    assert out["bmi"].isnull().sum() == 0
    # All indicators must be 1
    assert list(out["bmi_was_missing"]) == [1, 1, 1]
    # Filled values must be plausible (between global min and max of training bmis)
    assert out["bmi"].between(20.0, 36.0).all()
