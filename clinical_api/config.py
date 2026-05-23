from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

ARTEFACT_PATHS = {
    "stroke": {
        "pipeline": BASE_DIR / "stroke_ml/artefacts/xgb_pipeline.pkl",
        "explainer": BASE_DIR / "stroke_ml/artefacts/shap_explainer.pkl",
        "config":   BASE_DIR / "stroke_ml/config/threshold_config.yaml",
    },
    "heart_attack": {
        "pipeline": BASE_DIR / "heart_attack_ml/artefacts/xgb_pipeline.pkl",
        "explainer": BASE_DIR / "heart_attack_ml/artefacts/shap_explainer.pkl",
        "config":   BASE_DIR / "heart_attack_ml/config/threshold_config.yaml",
    },
    "diabetes": {
        "pipeline": BASE_DIR / "diabetes_ml/artefacts/xgb_pipeline.pkl",
        "explainer": BASE_DIR / "diabetes_ml/artefacts/shap_explainer.pkl",
        "config":   BASE_DIR / "diabetes_ml/config/threshold_config.yaml",
    },
    "ards": {
        "pipeline": BASE_DIR / "ards_ml/artefacts/xgb_pipeline.pkl",
        "explainer": BASE_DIR / "ards_ml/artefacts/shap_explainer.pkl",
        "config":   BASE_DIR / "ards_ml/config/threshold_config.yaml",
    },
}

DISCLAIMER = (
    "This prediction is a clinical decision support aid "
    "and does not constitute a medical diagnosis."
)
