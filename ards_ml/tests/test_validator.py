import pytest
import pandas as pd
import warnings

from ..pipeline.validator import FeatureValidator, EXPECTED_FEATURES


def _valid_row():
    return {
        "Age": 60, "Sex": 1, "BMI": 28.5, "Smoking_Status": 0,
        "Hypertension": 1, "Diabetes": 0, "COPD": 0,
        "Cardiovascular_Disease": 0, "Chronic_Kidney_Disease": 0, "Liver_Disease": 0,
        "Oxygen_Saturation": 85.0, "PaO2_FiO2_Ratio": 250.0,
        "Blood_Pressure_Systolic": 130, "Blood_Pressure_Diastolic": 80,
        "Heart_Rate": 95, "Respiratory_Rate": 22,
        "CRP_Level": 45.0, "D_Dimer": 2.5, "Lactate_Level": 2.0,
        "Ventilation_Type": 1,
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
    row["Age"] = 150  # outside [18, 120]
    df = pd.DataFrame([row])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v.fit(df).transform(df)
        assert any("Age" in str(x.message) for x in w)


def test_icu_los_flagged():
    v = FeatureValidator(raise_on_error=False)
    row = _valid_row()
    row["ICU_Length_of_Stay"] = 7
    df = pd.DataFrame([row])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v.fit(df).transform(df)
        assert any("ICU_Length_of_Stay" in str(x.message) for x in w)


def test_invalid_sex_value_warns():
    v = FeatureValidator(raise_on_error=False)
    row = _valid_row()
    row["Sex"] = 3  # not in [0, 1]
    df = pd.DataFrame([row])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v.fit(df).transform(df)
        assert any("Sex" in str(x.message) for x in w)
