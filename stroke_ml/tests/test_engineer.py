import numpy as np
import pandas as pd
import pytest

from stroke_ml.pipeline.engineer import FeatureEngineer

SAMPLE = pd.DataFrame([
    {
        "id": 1, "gender": "Male", "age": 45.0, "hypertension": 0, "heart_disease": 1,
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 120.0, "bmi": 28.0, "smoking_status": "never smoked",
    },
    {
        "id": 2, "gender": "Female", "age": 60.0, "hypertension": 1, "heart_disease": 0,
        "ever_married": "No", "work_type": "Govt_job", "Residence_type": "Rural",
        "avg_glucose_level": 90.0, "bmi": "N/A", "smoking_status": "Unknown",
    },
    {
        "id": 3, "gender": "Other", "age": 30.0, "hypertension": 0, "heart_disease": 0,
        "ever_married": "Yes", "work_type": "Self-employed", "Residence_type": "Urban",
        "avg_glucose_level": 75.0, "bmi": 22.5, "smoking_status": "smokes",
    },
])


def _fit_transform(df):
    eng = FeatureEngineer()
    eng.fit(df)
    return eng.transform(df)


def test_id_dropped():
    out = _fit_transform(SAMPLE)
    assert "id" not in out.columns


def test_gender_other_maps_to_female():
    out = _fit_transform(SAMPLE)
    # "Other" should behave like "Female" → gender_Male == 0
    assert out.loc[2, "gender_Male"] == 0


def test_gender_male_encoded():
    out = _fit_transform(SAMPLE)
    assert out.loc[0, "gender_Male"] == 1
    assert out.loc[1, "gender_Male"] == 0


def test_bmi_na_string_becomes_nan():
    out = _fit_transform(SAMPLE)
    assert np.isnan(out.loc[1, "bmi"])


def test_ever_married_encoding():
    out = _fit_transform(SAMPLE)
    assert out.loc[0, "ever_married"] == 1
    assert out.loc[1, "ever_married"] == 0


def test_residence_type_encoding():
    out = _fit_transform(SAMPLE)
    assert out.loc[0, "Residence_type"] == 1   # Urban
    assert out.loc[1, "Residence_type"] == 0   # Rural


def test_expected_ohe_columns_present():
    out = _fit_transform(SAMPLE)
    expected = [
        "gender_Male",
        "work_type_Never_worked", "work_type_Private", "work_type_Self-employed", "work_type_children",
        "smoking_status_Unknown", "smoking_status_formerly smoked",
        "smoking_status_never smoked", "smoking_status_smokes",
    ]
    for col in expected:
        assert col in out.columns, f"Missing column: {col}"


def test_smoking_unknown_preserved():
    out = _fit_transform(SAMPLE)
    assert out.loc[1, "smoking_status_Unknown"] == 1
    assert out.loc[0, "smoking_status_Unknown"] == 0
