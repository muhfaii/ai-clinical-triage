from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from .engineer import FeatureEngineer
from .imputer import MissingValueImputer
from .validator import FeatureValidator

# Features after engineering step — order matters for ColumnTransformer
STANDARD_SCALE_COLS = [
    "Age", "BMI",
    "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
    "Heart_Rate", "Respiratory_Rate",
    "pulse_pressure", "CCI",
]
ROBUST_SCALE_COLS = ["CRP_Level", "D_Dimer", "Lactate_Level", "log_CRP", "log_D_Dimer"]
MINMAX_SCALE_COLS = ["PaO2_FiO2_Ratio", "Oxygen_Saturation"]
ONEHOT_COLS = ["Smoking_Status", "Ventilation_Type"]
PASSTHROUGH_COLS = [
    "Sex",
    "Hypertension", "Diabetes", "COPD", "Cardiovascular_Disease",
    "Chronic_Kidney_Disease", "Liver_Disease",
    "hypoxaemia_tier", "tachycardia_flag", "tachypnoea_flag",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("std", StandardScaler(), STANDARD_SCALE_COLS),
            ("robust", RobustScaler(), ROBUST_SCALE_COLS),
            ("minmax", MinMaxScaler(), MINMAX_SCALE_COLS),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False), ONEHOT_COLS),
            ("passthrough", "passthrough", PASSTHROUGH_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(classifier) -> Pipeline:
    """
    Assembles the full 7-step sklearn-compatible Pipeline.
    `classifier` should be an unfitted XGBoost/LightGBM estimator
    with scale_pos_weight already set.
    """
    return Pipeline(
        steps=[
            ("validator", FeatureValidator()),
            ("imputer", MissingValueImputer()),
            ("engineer", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
