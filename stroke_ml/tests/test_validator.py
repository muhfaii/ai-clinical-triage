import warnings

import pandas as pd
import pytest

from stroke_ml.pipeline.validator import FeatureValidator

VALID_ROW = {
    "age": 45.0,
    "avg_glucose_level": 120.0,
    "bmi": 28.0,
    "bmi_was_missing": 0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": 1,
    "Residence_type": 1,
    "gender_Male": 1,
    "work_type_Never_worked": 0,
    "work_type_Private": 1,
    "work_type_Self-employed": 0,
    "work_type_children": 0,
    "smoking_status_Unknown": 0,
    "smoking_status_formerly smoked": 0,
    "smoking_status_never smoked": 1,
    "smoking_status_smokes": 0,
    "age_hypertension": 0.0,
    "cardiometabolic_risk": 1,
    "glucose_bmi_ratio": 4.286,
}


def test_valid_row_passes_silently():
    df = pd.DataFrame([VALID_ROW])
    validator = FeatureValidator()
    validator.fit(df)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = validator.transform(df)
    assert len(w) == 0
    assert set(out.columns) == set(VALID_ROW.keys())


def test_out_of_range_triggers_warning():
    row = VALID_ROW.copy()
    row["age"] = 200.0  # out of [0, 120]
    df = pd.DataFrame([row])
    validator = FeatureValidator()
    validator.fit(df)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validator.transform(df)
    assert any("age" in str(warning.message) for warning in w)


def test_only_schema_columns_returned():
    row = VALID_ROW.copy()
    row["extra_col"] = 99
    df = pd.DataFrame([row])
    validator = FeatureValidator()
    validator.fit(df)
    out = validator.transform(df)
    assert "extra_col" not in out.columns


def test_missing_column_triggers_warning():
    row = {k: v for k, v in VALID_ROW.items() if k != "bmi"}
    df = pd.DataFrame([row])
    validator = FeatureValidator()
    validator.fit(df)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validator.transform(df)
    assert any("bmi" in str(warning.message) for warning in w)
