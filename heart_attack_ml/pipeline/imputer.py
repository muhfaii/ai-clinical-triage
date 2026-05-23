import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

MEDIAN_FEATURES = [
    "Age", "Cholesterol", "BP_Systolic", "BP_Diastolic", "Heart_Rate",
    "Exercise_Hours_Per_Week", "Sedentary_Hours_Per_Day", "Income",
    "BMI", "Triglycerides", "Pulse_Pressure", "Risk_Score",
]
MODE_FEATURES = [
    "Sex", "Diabetes", "Family_History", "Smoking", "Obesity",
    "Alcohol_Consumption", "Diet", "Previous_Heart_Problems",
    "Medication_Use", "Stress_Level", "Physical_Activity_Days_Per_Week",
    "Sleep_Hours_Per_Day",
]

MISSING_INDICATOR_THRESHOLD = 0.05


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Median imputation for continuous features; mode imputation for binary/ordinal.
    Adds a missing-indicator column for any feature with >5% missingness at fit time.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self._missing_rates = X.isnull().mean()
        self._indicator_cols = [
            c for c in X.columns
            if self._missing_rates.get(c, 0) > MISSING_INDICATOR_THRESHOLD
        ]

        median_cols = [c for c in MEDIAN_FEATURES if c in X.columns]
        self._median_imputer = SimpleImputer(strategy="median")
        if median_cols:
            self._median_imputer.fit(X[median_cols])
        self._median_cols = median_cols

        mode_cols = [c for c in MODE_FEATURES if c in X.columns]
        self._mode_imputer = SimpleImputer(strategy="most_frequent")
        if mode_cols:
            self._mode_imputer.fit(X[mode_cols])
        self._mode_cols = mode_cols

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self._indicator_cols:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isnull().astype(int)

        if self._median_cols:
            X[self._median_cols] = self._median_imputer.transform(X[self._median_cols])

        if self._mode_cols:
            X[self._mode_cols] = self._mode_imputer.transform(X[self._mode_cols])

        return X
