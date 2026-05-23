import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SCHEMA_PATH = Path(__file__).parent.parent / "config" / "feature_schema.json"

with open(SCHEMA_PATH) as f:
    FEATURE_SCHEMA = json.load(f)

EXPECTED_FEATURES = list(FEATURE_SCHEMA.keys())


class FeatureValidator(BaseEstimator, TransformerMixin):
    """Validates incoming DataFrame against the 20-feature inference schema."""

    def __init__(self, schema: dict = None, raise_on_error: bool = False):
        self.schema = schema or FEATURE_SCHEMA
        self.raise_on_error = raise_on_error

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        issues = []

        # Check for missing required columns
        missing_cols = [c for c in self.schema if c not in X.columns]
        if missing_cols:
            msg = f"Missing required features: {missing_cols}"
            if self.raise_on_error:
                raise ValueError(msg)
            issues.append(msg)

        for col, spec in self.schema.items():
            if col not in X.columns:
                continue

            # Dtype coercion
            try:
                if spec["type"] == "int":
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                elif spec["type"] == "float":
                    X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
            except Exception as e:
                issues.append(f"{col}: dtype coercion failed — {e}")
                continue

            non_null = X[col].dropna()

            # Range check
            if "range" in spec:
                lo, hi = spec["range"]
                out_of_range = non_null[(non_null < lo) | (non_null > hi)]
                if not out_of_range.empty:
                    issues.append(
                        f"{col}: {len(out_of_range)} values outside [{lo}, {hi}]"
                    )

            # Allowed values check
            if "values" in spec:
                invalid = non_null[~non_null.isin(spec["values"])]
                if not invalid.empty:
                    issues.append(
                        f"{col}: unexpected values {invalid.unique().tolist()}"
                    )

        # ICU_Length_of_Stay must not be present as a predictor
        if "ICU_Length_of_Stay" in X.columns:
            issues.append(
                "ICU_Length_of_Stay found in input — this feature must be excluded (data leakage risk)"
            )

        if issues:
            msg = "Schema validation issues:\n" + "\n".join(f"  - {i}" for i in issues)
            if self.raise_on_error:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

        # Return only the 20 model features in schema order
        available = [c for c in EXPECTED_FEATURES if c in X.columns]
        return X[available]
