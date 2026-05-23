import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AgeDecileBMIImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing BMI using the median BMI within each age decile from training data.
    Falls back to global training median when a decile has no non-null BMI values.
    Adds a bmi_was_missing indicator column (1 = was null, 0 = present).
    """

    def fit(self, X: pd.DataFrame, y=None):
        ages = pd.to_numeric(X["age"], errors="coerce")
        bmis = pd.to_numeric(X["bmi"], errors="coerce")

        self._global_bmi_median = float(bmis.dropna().median())

        # pd.qcut with duplicates='drop' to handle ties at decile edges
        decile_labels, self._decile_edges = pd.qcut(
            ages, q=10, retbins=True, labels=False, duplicates="drop"
        )
        n_bins = len(self._decile_edges) - 1

        self._decile_medians = {}
        for bucket in range(n_bins):
            mask = decile_labels == bucket
            bucket_bmis = bmis[mask].dropna()
            self._decile_medians[bucket] = (
                float(bucket_bmis.median()) if len(bucket_bmis) > 0 else self._global_bmi_median
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        bmi = pd.to_numeric(X["bmi"], errors="coerce")
        X["bmi_was_missing"] = bmi.isnull().astype(int)

        null_mask = bmi.isnull()
        if null_mask.any():
            ages = pd.to_numeric(X["age"], errors="coerce")
            # Clip ages to the range seen at fit time to avoid out-of-bounds at inference
            clipped = ages.clip(self._decile_edges[0], self._decile_edges[-1])
            buckets = np.searchsorted(self._decile_edges[1:-1], clipped, side="right")

            fill_values = pd.Series(index=X.index, dtype=float)
            null_buckets = pd.Series(buckets, index=X.index)[null_mask]
            fill_values[null_mask] = null_buckets.map(
                lambda b: self._decile_medians.get(int(b), self._global_bmi_median)
            )
            X["bmi"] = bmi.where(~null_mask, fill_values)

        # glucose_bmi_ratio needs imputed (non-null) bmi — computed here after fill
        if "avg_glucose_level" in X.columns and "bmi" in X.columns:
            X["glucose_bmi_ratio"] = (
                pd.to_numeric(X["avg_glucose_level"], errors="coerce") /
                X["bmi"].replace(0, float("nan"))
            )

        return X
