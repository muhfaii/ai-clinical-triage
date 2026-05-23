import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
GENDER_COL = ["gender"]
SMOKING_COL = ["smoking_history"]
PASSTHROUGH_COLS = ["hypertension", "heart_disease"]


class BMICapper(BaseEstimator, TransformerMixin):
    """Clips BMI at the 99th-percentile computed on training data."""

    def __init__(self, percentile: int = 99):
        self.percentile = percentile

    def fit(self, X, y=None):
        self.cap_ = float(np.percentile(X["bmi"], self.percentile))
        return self

    def transform(self, X):
        X = X.copy()
        X["bmi"] = X["bmi"].clip(upper=self.cap_)
        return X


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), NUMERIC_COLS),
            # drop="first" removes one gender level to avoid multicollinearity
            ("gender", OneHotEncoder(drop="first", sparse_output=False), GENDER_COL),
            # All 6 smoking categories retained; "No Info" is a distinct clinical signal
            ("smoking", OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"), SMOKING_COL),
            ("passthrough", "passthrough", PASSTHROUGH_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("bmi_capper", BMICapper()),
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
