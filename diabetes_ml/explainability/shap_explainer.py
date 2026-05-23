"""
SHAP explainability — global feature importance and per-patient explanations.
Works with the Platt-calibrated XGBoost pipeline artefact.
"""
import joblib
import numpy as np
import pandas as pd
import shap


def _split_pipeline(pipeline):
    """Return (preprocessor_pipeline, xgb_classifier) from a fitted Pipeline."""
    preprocessor = pipeline[:-1]
    xgb_clf = pipeline.named_steps["classifier"]
    return preprocessor, xgb_clf


def build_explainer(pipeline) -> shap.TreeExplainer:
    _, xgb_clf = _split_pipeline(pipeline)
    return shap.TreeExplainer(xgb_clf)


def get_feature_names(pipeline, X_sample: pd.DataFrame) -> list:
    preprocessor, _ = _split_pipeline(pipeline)
    ct = pipeline.named_steps.get("preprocessor")
    if ct is not None and hasattr(ct, "get_feature_names_out"):
        try:
            return list(ct.get_feature_names_out())
        except Exception:
            pass
    n = preprocessor.transform(X_sample.head(1)).shape[1]
    return [f"feature_{i}" for i in range(n)]


def explain_global(pipeline, X: pd.DataFrame, explainer: shap.TreeExplainer = None) -> pd.Series:
    """Mean absolute SHAP values across the dataset, sorted descending."""
    if explainer is None:
        explainer = build_explainer(pipeline)
    preprocessor, _ = _split_pipeline(pipeline)
    X_transformed = preprocessor.transform(X)
    shap_values = explainer.shap_values(X_transformed)
    feature_names = get_feature_names(pipeline, X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)


def explain_patient(
    pipeline,
    patient_row: pd.DataFrame,
    explainer: shap.TreeExplainer = None,
    top_n: int = 3,
) -> dict:
    """
    Per-patient SHAP explanation.
    Returns {feature_name: shap_value} for the top_n most influential features,
    suitable for surfacing in the clinician-facing UI.
    """
    if explainer is None:
        explainer = build_explainer(pipeline)
    preprocessor, _ = _split_pipeline(pipeline)
    X_transformed = preprocessor.transform(patient_row)
    shap_values = explainer.shap_values(X_transformed)[0]
    feature_names = get_feature_names(pipeline, patient_row)
    shap_dict = dict(zip(feature_names, shap_values))
    top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return {k: round(float(v), 6) for k, v in top}


def save_explainer(explainer: shap.TreeExplainer, path: str):
    joblib.dump(explainer, path)


def load_explainer(path: str) -> shap.TreeExplainer:
    return joblib.load(path)
