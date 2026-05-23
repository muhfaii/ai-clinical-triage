from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .engineer import FeatureEngineer
from .imputer import AgeDecileBMIImputer
from .validator import FeatureValidator

CONTINUOUS_COLS = ["age", "avg_glucose_level", "bmi", "age_hypertension", "glucose_bmi_ratio"]
PASSTHROUGH_COLS = [
    "bmi_was_missing",
    "hypertension",
    "heart_disease",
    "ever_married",
    "Residence_type",
    "gender_Male",
    "work_type_Never_worked",
    "work_type_Private",
    "work_type_Self-employed",
    "work_type_children",
    "smoking_status_Unknown",
    "smoking_status_formerly smoked",
    "smoking_status_never smoked",
    "smoking_status_smokes",
    "cardiometabolic_risk",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("scale", RobustScaler(), CONTINUOUS_COLS),
            ("passthrough", "passthrough", PASSTHROUGH_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(classifier, use_smote: bool = False) -> Pipeline:
    steps = [
        ("engineer", FeatureEngineer()),
        ("imputer", AgeDecileBMIImputer()),
        ("validator", FeatureValidator()),
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ]
    if use_smote:
        # SMOTE inserted before classifier; imblearn Pipeline skips sampler on predict
        steps.insert(-1, ("smote", SMOTE(random_state=42, k_neighbors=5)))
        return ImbPipeline(steps=steps)
    return Pipeline(steps=steps)
