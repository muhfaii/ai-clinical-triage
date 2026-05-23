import numpy as np
import pandas as pd
import pytest

from ..pipeline.imputer import MissingValueImputer


def _base_df(n=20):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Age": rng.integers(30, 89, n).astype(float),
        "BMI": rng.uniform(18, 40, n),
        "Oxygen_Saturation": rng.uniform(70, 100, n),
        "Lactate_Level": rng.uniform(0.5, 4.0, n),
        "Blood_Pressure_Systolic": rng.integers(90, 180, n).astype(float),
        "Blood_Pressure_Diastolic": rng.integers(50, 110, n).astype(float),
        "Heart_Rate": rng.integers(50, 150, n).astype(float),
        "Respiratory_Rate": rng.integers(10, 40, n).astype(float),
        "PaO2_FiO2_Ratio": rng.uniform(100, 400, n),
        "CRP_Level": rng.uniform(0.1, 100, n),
        "D_Dimer": rng.uniform(0.1, 5, n),
        "Smoking_Status": rng.integers(0, 3, n).astype(float),
        "Ventilation_Type": rng.integers(0, 3, n).astype(float),
        "Hypertension": rng.integers(0, 2, n).astype(float),
        "Diabetes": rng.integers(0, 2, n).astype(float),
        "COPD": rng.integers(0, 2, n).astype(float),
        "Cardiovascular_Disease": rng.integers(0, 2, n).astype(float),
        "Chronic_Kidney_Disease": rng.integers(0, 2, n).astype(float),
        "Liver_Disease": rng.integers(0, 2, n).astype(float),
    })


def test_no_nulls_passthrough():
    df = _base_df()
    imp = MissingValueImputer()
    imp.fit(df)
    out = imp.transform(df)
    assert out.isnull().sum().sum() == 0


def test_median_imputation_fills_nulls():
    df = _base_df()
    df.loc[0, "Age"] = np.nan
    df.loc[1, "BMI"] = np.nan
    imp = MissingValueImputer()
    imp.fit(df)
    out = imp.transform(df)
    assert not out["Age"].isnull().any()
    assert not out["BMI"].isnull().any()


def test_zero_fill_comorbidities():
    df = _base_df()
    df.loc[0, "Hypertension"] = np.nan
    df.loc[1, "Liver_Disease"] = np.nan
    imp = MissingValueImputer()
    imp.fit(df)
    out = imp.transform(df)
    assert out.loc[0, "Hypertension"] == 0
    assert out.loc[1, "Liver_Disease"] == 0


def test_missing_indicator_added_when_threshold_exceeded():
    df = _base_df(100)
    # Introduce >5% missingness on Age
    df.loc[:6, "Age"] = np.nan
    imp = MissingValueImputer()
    imp.fit(df)
    out = imp.transform(df)
    assert "Age_missing" in out.columns
