import numpy as np
import pandas as pd
import pytest

from ..pipeline.engineer import FeatureEngineer


def _sample():
    return pd.DataFrame([{
        "Age": 60, "Sex": 1, "BMI": 28.0, "Smoking_Status": 1,
        "Hypertension": 1, "Diabetes": 1, "COPD": 0,
        "Cardiovascular_Disease": 1, "Chronic_Kidney_Disease": 0, "Liver_Disease": 0,
        "Oxygen_Saturation": 85.0, "PaO2_FiO2_Ratio": 250.0,
        "Blood_Pressure_Systolic": 140, "Blood_Pressure_Diastolic": 90,
        "Heart_Rate": 110, "Respiratory_Rate": 35,
        "CRP_Level": 50.0, "D_Dimer": 2.5, "Lactate_Level": 2.2,
        "Ventilation_Type": 1,
    }])


def test_cci_computed():
    eng = FeatureEngineer()
    out = eng.fit_transform(_sample())
    assert "CCI" in out.columns
    # Diabetes=1, Cardiovascular_Disease=1 → CCI = 2
    assert out.loc[0, "CCI"] == 2


def test_hypoxaemia_tier_mild():
    df = _sample()
    df.loc[0, "PaO2_FiO2_Ratio"] = 250.0  # mild
    out = FeatureEngineer().fit_transform(df)
    assert out.loc[0, "hypoxaemia_tier"] == 1


def test_hypoxaemia_tier_moderate():
    df = _sample()
    df.loc[0, "PaO2_FiO2_Ratio"] = 150.0
    out = FeatureEngineer().fit_transform(df)
    assert out.loc[0, "hypoxaemia_tier"] == 2


def test_pulse_pressure():
    out = FeatureEngineer().fit_transform(_sample())
    assert out.loc[0, "pulse_pressure"] == 140 - 90


def test_log_crp():
    out = FeatureEngineer().fit_transform(_sample())
    assert abs(out.loc[0, "log_CRP"] - np.log1p(50.0)) < 1e-6


def test_tachycardia_flag():
    out = FeatureEngineer().fit_transform(_sample())
    assert out.loc[0, "tachycardia_flag"] == 1  # HR=110 > 100


def test_tachypnoea_flag():
    out = FeatureEngineer().fit_transform(_sample())
    assert out.loc[0, "tachypnoea_flag"] == 1  # RR=35 > 30


def test_no_tachycardia():
    df = _sample()
    df.loc[0, "Heart_Rate"] = 80
    out = FeatureEngineer().fit_transform(df)
    assert out.loc[0, "tachycardia_flag"] == 0
