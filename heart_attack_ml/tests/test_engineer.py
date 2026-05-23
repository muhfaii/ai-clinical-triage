import numpy as np
import pandas as pd
import pytest

from ..pipeline.engineer import FeatureEngineer


def _raw_row(**overrides):
    row = {
        "Patient ID": "TST001",
        "Age": 55,
        "Sex": "Male",
        "Cholesterol": 250,
        "Blood Pressure": "140/90",
        "Heart Rate": 80,
        "Diabetes": 1,
        "Family History": 0,
        "Smoking": 1,
        "Obesity": 0,
        "Alcohol Consumption": 0,
        "Exercise Hours Per Week": 3.0,
        "Diet": "Average",
        "Previous Heart Problems": 0,
        "Medication Use": 1,
        "Stress Level": 7,
        "Sedentary Hours Per Day": 5.0,
        "Income": 80000,
        "BMI": 27.5,
        "Triglycerides": 200,
        "Physical Activity Days Per Week": 3,
        "Sleep Hours Per Day": 7,
        "Country": "USA",
        "Continent": "North America",
        "Hemisphere": "Northern Hemisphere",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_bp_parsed():
    out = FeatureEngineer().fit_transform(_raw_row())
    assert "BP_Systolic" in out.columns
    assert "BP_Diastolic" in out.columns
    assert out.loc[0, "BP_Systolic"] == 140
    assert out.loc[0, "BP_Diastolic"] == 90


def test_blood_pressure_raw_dropped():
    out = FeatureEngineer().fit_transform(_raw_row())
    assert "Blood Pressure" not in out.columns


def test_sex_encoded():
    out_m = FeatureEngineer().fit_transform(_raw_row(**{"Sex": "Male"}))
    out_f = FeatureEngineer().fit_transform(_raw_row(**{"Sex": "Female"}))
    assert out_m.loc[0, "Sex"] == 1
    assert out_f.loc[0, "Sex"] == 0


def test_diet_encoded():
    out_h = FeatureEngineer().fit_transform(_raw_row(**{"Diet": "Healthy"}))
    out_a = FeatureEngineer().fit_transform(_raw_row(**{"Diet": "Average"}))
    out_u = FeatureEngineer().fit_transform(_raw_row(**{"Diet": "Unhealthy"}))
    assert out_h.loc[0, "Diet"] == 0
    assert out_a.loc[0, "Diet"] == 1
    assert out_u.loc[0, "Diet"] == 2


def test_pulse_pressure():
    out = FeatureEngineer().fit_transform(_raw_row())
    assert out.loc[0, "Pulse_Pressure"] == 140 - 90


def test_risk_score_in_zero_one():
    out = FeatureEngineer().fit_transform(_raw_row())
    assert 0.0 <= out.loc[0, "Risk_Score"] <= 1.0


def test_risk_score_increases_with_severity():
    low_risk = _raw_row(**{"Age": 20, "BMI": 15.0, "Cholesterol": 130, "Triglycerides": 50, "Stress Level": 1})
    high_risk = _raw_row(**{"Age": 88, "BMI": 55.0, "Cholesterol": 390, "Triglycerides": 950, "Stress Level": 10})
    eng = FeatureEngineer()
    low_out = eng.fit_transform(low_risk)
    high_out = eng.fit_transform(high_risk)
    assert high_out.loc[0, "Risk_Score"] > low_out.loc[0, "Risk_Score"]


def test_non_predictive_columns_dropped():
    out = FeatureEngineer().fit_transform(_raw_row())
    for col in ["Patient ID", "Country", "Continent", "Hemisphere"]:
        assert col not in out.columns


def test_column_names_renamed():
    out = FeatureEngineer().fit_transform(_raw_row())
    assert "Heart_Rate" in out.columns
    assert "Heart Rate" not in out.columns
    assert "Family_History" in out.columns
    assert "Stress_Level" in out.columns
