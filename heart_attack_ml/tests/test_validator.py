import warnings

import pandas as pd
import pytest

from ..pipeline.validator import FeatureValidator, EXPECTED_FEATURES


def _valid_row():
    return {
        "Age": 55, "Sex": 1, "Cholesterol": 250,
        "BP_Systolic": 140, "BP_Diastolic": 90, "Heart_Rate": 80,
        "Diabetes": 1, "Family_History": 0, "Smoking": 1,
        "Obesity": 0, "Alcohol_Consumption": 0,
        "Exercise_Hours_Per_Week": 3.0, "Diet": 1,
        "Previous_Heart_Problems": 0, "Medication_Use": 1,
        "Stress_Level": 7, "Sedentary_Hours_Per_Day": 5.0,
        "Income": 80000, "BMI": 27.5, "Triglycerides": 200,
        "Physical_Activity_Days_Per_Week": 3, "Sleep_Hours_Per_Day": 7,
        "Pulse_Pressure": 50, "Risk_Score": 0.45,
    }


def test_valid_input_passes():
    v = FeatureValidator()
    df = pd.DataFrame([_valid_row()])
    out = v.fit(df).transform(df)
    assert list(out.columns) == EXPECTED_FEATURES


def test_missing_column_warns():
    v = FeatureValidator(raise_on_error=False)
    row = _valid_row()
    del row["Age"]
    df = pd.DataFrame([row])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v.fit(df).transform(df)
        assert any("Missing required features" in str(x.message) for x in w)


def test_missing_column_raises():
    v = FeatureValidator(raise_on_error=True)
    row = _valid_row()
    del row["Age"]
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="Missing required features"):
        v.fit(df).transform(df)


def test_out_of_range_warns():
    v = FeatureValidator(raise_on_error=False)
    row = _valid_row()
    row["Age"] = 200  # outside [18, 90]
    df = pd.DataFrame([row])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v.fit(df).transform(df)
        assert any("Age" in str(x.message) for x in w)


def test_invalid_sex_value_warns():
    v = FeatureValidator(raise_on_error=False)
    row = _valid_row()
    row["Sex"] = 5  # not in [0, 1]
    df = pd.DataFrame([row])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v.fit(df).transform(df)
        assert any("Sex" in str(x.message) for x in w)


def test_invalid_diet_value_warns():
    v = FeatureValidator(raise_on_error=False)
    row = _valid_row()
    row["Diet"] = 9  # not in [0, 1, 2]
    df = pd.DataFrame([row])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v.fit(df).transform(df)
        assert any("Diet" in str(x.message) for x in w)
