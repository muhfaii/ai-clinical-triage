import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

GENDER_MAP = {"Other": "Female"}
MARRIED_MAP = {"Yes": 1, "No": 0}
RESIDENCE_MAP = {"Urban": 1, "Rural": 0}

GENDER_DUMMIES = ["gender_Male"]
WORK_DUMMIES = [
    "work_type_Never_worked",
    "work_type_Private",
    "work_type_Self-employed",
    "work_type_children",
]
SMOKING_DUMMIES = [
    "smoking_status_Unknown",
    "smoking_status_formerly smoked",
    "smoking_status_never smoked",
    "smoking_status_smokes",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Encodes raw stroke dataset columns into model-ready features."""

    def fit(self, X: pd.DataFrame, y=None):
        # Record expected columns from a dry-run transform so inference can reindex safely
        self._expected_cols = list(self._apply(X.copy()).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._apply(X.copy())
        if hasattr(self, "_expected_cols"):
            X = X.reindex(columns=self._expected_cols, fill_value=0)
        return X

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:
        if "id" in X.columns:
            X = X.drop(columns=["id"])

        # Coerce "N/A" strings in bmi to NaN before any numeric ops
        if "bmi" in X.columns:
            X["bmi"] = pd.to_numeric(X["bmi"], errors="coerce")

        if "gender" in X.columns:
            X["gender"] = X["gender"].replace(GENDER_MAP)
            gender_dummies = pd.get_dummies(X["gender"], prefix="gender", drop_first=True)
            # Ensure gender_Male exists even if only one gender in batch
            for col in GENDER_DUMMIES:
                if col not in gender_dummies.columns:
                    gender_dummies[col] = 0
            gender_dummies = gender_dummies[GENDER_DUMMIES].astype(int)
            X = X.drop(columns=["gender"]).join(gender_dummies)

        if "ever_married" in X.columns:
            X["ever_married"] = X["ever_married"].map(MARRIED_MAP)

        if "work_type" in X.columns:
            work_dummies = pd.get_dummies(X["work_type"], prefix="work_type", drop_first=True)
            for col in WORK_DUMMIES:
                if col not in work_dummies.columns:
                    work_dummies[col] = 0
            work_dummies = work_dummies[WORK_DUMMIES].astype(int)
            X = X.drop(columns=["work_type"]).join(work_dummies)

        if "Residence_type" in X.columns:
            X["Residence_type"] = X["Residence_type"].map(RESIDENCE_MAP)

        if "smoking_status" in X.columns:
            # drop=None: keep all 4 categories including Unknown as explicit column
            smoke_dummies = pd.get_dummies(X["smoking_status"], prefix="smoking_status", drop_first=False)
            for col in SMOKING_DUMMIES:
                if col not in smoke_dummies.columns:
                    smoke_dummies[col] = 0
            smoke_dummies = smoke_dummies[SMOKING_DUMMIES].astype(int)
            X = X.drop(columns=["smoking_status"]).join(smoke_dummies)

        # Interaction features — computed after encoding so hypertension/heart_disease are numeric
        if "age" in X.columns and "hypertension" in X.columns:
            X["age_hypertension"] = pd.to_numeric(X["age"], errors="coerce") * pd.to_numeric(X["hypertension"], errors="coerce")

        if "hypertension" in X.columns and "heart_disease" in X.columns:
            X["cardiometabolic_risk"] = (
                pd.to_numeric(X["hypertension"], errors="coerce") +
                pd.to_numeric(X["heart_disease"], errors="coerce")
            )

        return X
