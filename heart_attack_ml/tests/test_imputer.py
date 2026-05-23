import numpy as np
import pandas as pd

from ..pipeline.imputer import MissingValueImputer


def _post_engineer_df(n=20):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Age": rng.integers(18, 90, n).astype(float),
        "Sex": rng.integers(0, 2, n).astype(float),
        "Cholesterol": rng.integers(120, 400, n).astype(float),
        "BP_Systolic": rng.integers(90, 180, n).astype(float),
        "BP_Diastolic": rng.integers(60, 110, n).astype(float),
        "Heart_Rate": rng.integers(40, 110, n).astype(float),
        "Diabetes": rng.integers(0, 2, n).astype(float),
        "Family_History": rng.integers(0, 2, n).astype(float),
        "Smoking": rng.integers(0, 2, n).astype(float),
        "Obesity": rng.integers(0, 2, n).astype(float),
        "Alcohol_Consumption": rng.integers(0, 2, n).astype(float),
        "Exercise_Hours_Per_Week": rng.uniform(0, 20, n),
        "Diet": rng.integers(0, 3, n).astype(float),
        "Previous_Heart_Problems": rng.integers(0, 2, n).astype(float),
        "Medication_Use": rng.integers(0, 2, n).astype(float),
        "Stress_Level": rng.integers(1, 11, n).astype(float),
        "Sedentary_Hours_Per_Day": rng.uniform(0, 12, n),
        "Income": rng.integers(20000, 300000, n).astype(float),
        "BMI": rng.uniform(15, 45, n),
        "Triglycerides": rng.integers(30, 800, n).astype(float),
        "Physical_Activity_Days_Per_Week": rng.integers(0, 8, n).astype(float),
        "Sleep_Hours_Per_Day": rng.integers(4, 11, n).astype(float),
        "Pulse_Pressure": rng.integers(20, 80, n).astype(float),
        "Risk_Score": rng.uniform(0, 1, n),
    })


def test_no_nulls_passthrough():
    df = _post_engineer_df()
    imp = MissingValueImputer()
    imp.fit(df)
    out = imp.transform(df)
    assert out.isnull().sum().sum() == 0


def test_median_fills_nulls():
    df = _post_engineer_df()
    df.loc[0, "Age"] = np.nan
    df.loc[1, "BMI"] = np.nan
    imp = MissingValueImputer()
    imp.fit(df)
    out = imp.transform(df)
    assert not out["Age"].isnull().any()
    assert not out["BMI"].isnull().any()


def test_mode_fills_nulls():
    df = _post_engineer_df()
    df.loc[0, "Diabetes"] = np.nan
    df.loc[1, "Diet"] = np.nan
    imp = MissingValueImputer()
    imp.fit(df)
    out = imp.transform(df)
    assert not out["Diabetes"].isnull().any()
    assert not out["Diet"].isnull().any()


def test_missing_indicator_added():
    df = _post_engineer_df(100)
    df.loc[:6, "Age"] = np.nan  # > 5% missingness
    imp = MissingValueImputer()
    imp.fit(df)
    out = imp.transform(df)
    assert "Age_missing" in out.columns
