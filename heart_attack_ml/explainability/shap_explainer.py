import joblib
import numpy as np
import pandas as pd
import shap


def _get_base_pipeline(pipeline):
    if hasattr(pipeline, "calibrated_classifiers_"):
        return pipeline.calibrated_classifiers_[0].estimator
    elif hasattr(pipeline, "estimator"):
        return pipeline.estimator
    return pipeline


def build_explainer(pipeline) -> shap.TreeExplainer:
    base = _get_base_pipeline(pipeline)
    xgb_clf = base.named_steps["classifier"]
    return shap.TreeExplainer(xgb_clf)


def get_feature_names(pipeline, X_sample: pd.DataFrame) -> list:
    base = _get_base_pipeline(pipeline)
    ct = base.named_steps.get("preprocessor")
    if ct is not None and hasattr(ct, "get_feature_names_out"):
        try:
            return list(ct.get_feature_names_out())
        except Exception:
            pass
    preprocessor = base[:-1]
    transformed = preprocessor.transform(X_sample.head(1))
    return [f"feature_{i}" for i in range(transformed.shape[1])]


def explain_global(pipeline, X: pd.DataFrame, explainer: shap.TreeExplainer = None) -> pd.Series:
    if explainer is None:
        explainer = build_explainer(pipeline)
    base = _get_base_pipeline(pipeline)
    X_transformed = base[:-1].transform(X)
    shap_values = explainer.shap_values(X_transformed)
    feature_names = get_feature_names(pipeline, X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)


def explain_patient(
    pipeline,
    patient_features: pd.DataFrame,
    explainer: shap.TreeExplainer = None,
    top_n: int = 3,
) -> dict:
    """Returns top_n SHAP factors per patient — used in the API top_factors field."""
    if explainer is None:
        explainer = build_explainer(pipeline)
    base = _get_base_pipeline(pipeline)
    X_transformed = base[:-1].transform(patient_features)
    shap_values = explainer.shap_values(X_transformed)[0]
    feature_names = get_feature_names(pipeline, patient_features)
    shap_dict = dict(zip(feature_names, shap_values))
    top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return {k: round(float(v), 6) for k, v in top}


def save_explainer(explainer: shap.TreeExplainer, path: str):
    joblib.dump(explainer, path)


def load_explainer(path: str) -> shap.TreeExplainer:
    return joblib.load(path)
