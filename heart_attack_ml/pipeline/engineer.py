import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Normalisation constants for Risk_Score composite index
_AGE_RANGE = (18.0, 90.0)
_BMI_RANGE = (10.0, 60.0)
_CHOL_RANGE = (120.0, 400.0)
_TRIG_RANGE = (20.0, 1000.0)
_STRESS_RANGE = (1.0, 10.0)

RISK_SCORE_WEIGHTS = {
    "Age":         0.25,
    "BMI":         0.20,
    "Cholesterol": 0.20,
    "Triglycerides": 0.20,
    "Stress_Level": 0.15,
}

SEX_MAP = {"Male": 1, "Female": 0}
DIET_MAP = {"Healthy": 0, "Average": 1, "Unhealthy": 2}

DROP_COLS = ["Patient ID", "Country", "Continent", "Hemisphere"]

RENAME_MAP = {
    "Heart Rate": "Heart_Rate",
    "Family History": "Family_History",
    "Alcohol Consumption": "Alcohol_Consumption",
    "Exercise Hours Per Week": "Exercise_Hours_Per_Week",
    "Previous Heart Problems": "Previous_Heart_Problems",
    "Medication Use": "Medication_Use",
    "Stress Level": "Stress_Level",
    "Sedentary Hours Per Day": "Sedentary_Hours_Per_Day",
    "Physical Activity Days Per Week": "Physical_Activity_Days_Per_Week",
    "Sleep Hours Per Day": "Sleep_Hours_Per_Day",
}


def _norm(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return (series.clip(lo, hi) - lo) / (hi - lo)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Step 1 of the heart-attack pipeline. Handles all raw-format conversion:
      - Renames space-containing column names to underscores
      - Encodes Sex (Male=1, Female=0) and Diet (Healthy=0, Average=1, Unhealthy=2)
      - Parses Blood Pressure string "SYS/DIA" → BP_Systolic, BP_Diastolic
      - Derives Pulse_Pressure = BP_Systolic − BP_Diastolic
      - Derives Risk_Score (normalised weighted composite: Age, BMI, Cholesterol,
        Triglycerides, Stress_Level)
      - Drops non-predictive columns (Patient ID, Country, Continent, Hemisphere,
        raw Blood Pressure string)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Parse Blood Pressure string → BP_Systolic, BP_Diastolic (must precede drop)
        if "Blood Pressure" in X.columns:
            bp_split = X["Blood Pressure"].str.split("/", expand=True)
            X["BP_Systolic"] = pd.to_numeric(bp_split[0], errors="coerce")
            X["BP_Diastolic"] = pd.to_numeric(bp_split[1], errors="coerce")
            X = X.drop(columns=["Blood Pressure"])

        # Drop non-predictive columns
        X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])

        # Rename space-containing columns
        X = X.rename(columns=RENAME_MAP)

        # Encode Sex
        if "Sex" in X.columns:
            X["Sex"] = X["Sex"].map(SEX_MAP).fillna(X["Sex"])
            X["Sex"] = pd.to_numeric(X["Sex"], errors="coerce")

        # Encode Diet
        if "Diet" in X.columns:
            X["Diet"] = X["Diet"].map(DIET_MAP).fillna(X["Diet"])
            X["Diet"] = pd.to_numeric(X["Diet"], errors="coerce")

        # Pulse Pressure
        if "BP_Systolic" in X.columns and "BP_Diastolic" in X.columns:
            X["Pulse_Pressure"] = (X["BP_Systolic"] - X["BP_Diastolic"]).astype("Int64")

        # Risk_Score: normalised weighted composite
        components = pd.Series(0.0, index=X.index)
        norm_map = {
            "Age": _AGE_RANGE,
            "BMI": _BMI_RANGE,
            "Cholesterol": _CHOL_RANGE,
            "Triglycerides": _TRIG_RANGE,
            "Stress_Level": _STRESS_RANGE,
        }
        for col, (lo, hi) in norm_map.items():
            if col in X.columns:
                weight = RISK_SCORE_WEIGHTS[col]
                components += _norm(X[col].astype(float), lo, hi) * weight
        X["Risk_Score"] = components.round(6)

        return X
