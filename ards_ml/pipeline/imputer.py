import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer

MEDIAN_FEATURES = [
    "Age", "BMI", "Oxygen_Saturation", "Lactate_Level",
    "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
    "Heart_Rate", "Respiratory_Rate",
]
MICE_FEATURES = ["PaO2_FiO2_Ratio", "CRP_Level", "D_Dimer"]
MODE_FEATURES = ["Smoking_Status", "Ventilation_Type"]
ZERO_FILL_FEATURES = [
    "Hypertension", "Diabetes", "COPD", "Cardiovascular_Disease",
    "Chronic_Kidney_Disease", "Liver_Disease",
]

MISSING_INDICATOR_THRESHOLD = 0.05
MISSING_FLAG_THRESHOLD = 0.30


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Per-feature imputation strategy as specified in Section 3.1.
    Adds missing-indicator columns for features with >5% missingness at fit time.
    Warns if any feature exceeds 30% missingness.
    """

    def __init__(self, mice_threshold: float = 0.10, random_state: int = 42):
        self.mice_threshold = mice_threshold
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        self._missing_rates = X.isnull().mean()
        self._indicator_cols = [
            c for c in X.columns
            if self._missing_rates.get(c, 0) > MISSING_INDICATOR_THRESHOLD
        ]

        high_missing = [
            c for c in X.columns
            if self._missing_rates.get(c, 0) > MISSING_FLAG_THRESHOLD
        ]
        if high_missing:
            import warnings
            warnings.warn(
                f"Features with >30% missingness (flag before training): {high_missing}",
                stacklevel=2,
            )

        # Median imputer
        median_cols = [c for c in MEDIAN_FEATURES if c in X.columns]
        self._median_imputer = SimpleImputer(strategy="median")
        if median_cols:
            self._median_imputer.fit(X[median_cols])
        self._median_cols = median_cols

        # MICE — only for columns above mice_threshold missingness, else fallback median
        mice_cols = [c for c in MICE_FEATURES if c in X.columns]
        high_miss_mice = [
            c for c in mice_cols
            if self._missing_rates.get(c, 0) > self.mice_threshold
        ]
        low_miss_mice = [c for c in mice_cols if c not in high_miss_mice]

        self._mice_cols = high_miss_mice
        self._mice_fallback_cols = low_miss_mice

        if high_miss_mice:
            self._mice_imputer = IterativeImputer(
                random_state=self.random_state, max_iter=10
            )
            self._mice_imputer.fit(X[high_miss_mice])

        if low_miss_mice:
            self._mice_fallback_imputer = SimpleImputer(strategy="median")
            self._mice_fallback_imputer.fit(X[low_miss_mice])

        # Mode imputer
        mode_cols = [c for c in MODE_FEATURES if c in X.columns]
        self._mode_imputer = SimpleImputer(strategy="most_frequent")
        if mode_cols:
            self._mode_imputer.fit(X[mode_cols])
        self._mode_cols = mode_cols

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Add missing indicators before imputation
        for col in self._indicator_cols:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isnull().astype(int)

        # Median imputation
        if self._median_cols:
            X[self._median_cols] = self._median_imputer.transform(X[self._median_cols])

        # MICE imputation
        if self._mice_cols:
            X[self._mice_cols] = self._mice_imputer.transform(X[self._mice_cols])
        if self._mice_fallback_cols:
            X[self._mice_fallback_cols] = self._mice_fallback_imputer.transform(
                X[self._mice_fallback_cols]
            )

        # Mode imputation
        if self._mode_cols:
            X[self._mode_cols] = self._mode_imputer.transform(X[self._mode_cols])

        # Zero-fill for binary comorbidity flags
        for col in ZERO_FILL_FEATURES:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        return X
