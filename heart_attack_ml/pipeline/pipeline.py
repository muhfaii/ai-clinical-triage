from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .engineer import FeatureEngineer
from .imputer import MissingValueImputer
from .validator import FeatureValidator

# Continuous features scaled for LR baseline (XGBoost ignores scale)
SCALE_COLS = [
    "Age", "Cholesterol", "BP_Systolic", "BP_Diastolic", "Heart_Rate",
    "Exercise_Hours_Per_Week", "Sedentary_Hours_Per_Day", "Income",
    "BMI", "Triglycerides", "Pulse_Pressure", "Risk_Score",
]
PASSTHROUGH_COLS = [
    "Sex", "Diabetes", "Family_History", "Smoking", "Obesity",
    "Alcohol_Consumption", "Diet", "Previous_Heart_Problems",
    "Medication_Use", "Stress_Level", "Physical_Activity_Days_Per_Week",
    "Sleep_Hours_Per_Day",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), SCALE_COLS),
            ("passthrough", "passthrough", PASSTHROUGH_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("validator", FeatureValidator()),
            ("imputer", MissingValueImputer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
