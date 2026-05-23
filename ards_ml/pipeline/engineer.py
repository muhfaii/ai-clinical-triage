import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Charlson Comorbidity Index weights (simplified for available features)
CCI_WEIGHTS = {
    "Diabetes": 1,
    "COPD": 1,
    "Cardiovascular_Disease": 1,
    "Chronic_Kidney_Disease": 2,
    "Liver_Disease": 1,
    "Hypertension": 0,  # Not scored in standard CCI but included as flag
}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives clinical features as specified in Section 2.2:
      - Charlson Comorbidity Index (CCI)
      - Hypoxaemia severity tier (Berlin ARDS criteria)
      - Pulse pressure
      - Log-transformed CRP and D_Dimer
      - Tachycardia flag (HR > 100)
      - Tachypnoea flag (RR > 30)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Charlson Comorbidity Index
        cci = pd.Series(0.0, index=X.index)
        for col, weight in CCI_WEIGHTS.items():
            if col in X.columns:
                cci += X[col].fillna(0) * weight
        X["CCI"] = cci

        # Hypoxaemia severity tier (Berlin criteria)
        # mild=1 (200-300), moderate=2 (100-200), severe=3 (<100)
        if "PaO2_FiO2_Ratio" in X.columns:
            pf = X["PaO2_FiO2_Ratio"]
            X["hypoxaemia_tier"] = np.select(
                [pf >= 200, (pf >= 100) & (pf < 200), pf < 100],
                [1, 2, 3],
                default=0,
            )

        # Pulse pressure
        if "Blood_Pressure_Systolic" in X.columns and "Blood_Pressure_Diastolic" in X.columns:
            X["pulse_pressure"] = (
                X["Blood_Pressure_Systolic"] - X["Blood_Pressure_Diastolic"]
            )

        # Log-transformed CRP (normalises right skew)
        if "CRP_Level" in X.columns:
            X["log_CRP"] = np.log1p(X["CRP_Level"])

        # Log-transformed D_Dimer (inspect skewness — apply log)
        if "D_Dimer" in X.columns:
            X["log_D_Dimer"] = np.log1p(X["D_Dimer"])

        # Tachycardia flag
        if "Heart_Rate" in X.columns:
            X["tachycardia_flag"] = (X["Heart_Rate"] > 100).astype(int)

        # Tachypnoea flag
        if "Respiratory_Rate" in X.columns:
            X["tachypnoea_flag"] = (X["Respiratory_Rate"] > 30).astype(int)

        return X
