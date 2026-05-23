"""
SHAP explainability — global summary and per-patient waterfall.
Works with the calibrated XGBoost pipeline artefact.
"""
import joblib
import numpy as np
import pandas as pd
import shap


def _extract_xgb_from_pipeline(pipeline):
    """
    Navigate through CalibratedClassifierCV → Pipeline → XGBClassifier.
    Returns the raw XGBClassifier and the preprocessing sub-pipeline.
    """
    # CalibratedClassifierCV wraps the base pipeline
    if hasattr(pipeline, "calibrated_classifiers_"):
        base = pipeline.calibrated_classifiers_[0].estimator
    elif hasattr(pipeline, "estimator"):
        base = pipeline.estimator
    else:
        base = pipeline

    # base is a sklearn Pipeline with steps ending in 'classifier'
    preprocessor = base[:-1]          # all steps except classifier
    xgb_clf = base.named_steps["classifier"]
    return xgb_clf, preprocessor


def build_explainer(pipeline) -> shap.TreeExplainer:
    xgb_clf, _ = _extract_xgb_from_pipeline(pipeline)
    return shap.TreeExplainer(xgb_clf)


def get_feature_names(pipeline, X_sample: pd.DataFrame) -> list:
    """Resolve feature names after preprocessing."""
    if hasattr(pipeline, "calibrated_classifiers_"):
        base = pipeline.calibrated_classifiers_[0].estimator
    else:
        base = pipeline

    preprocessor = base[:-1]
    transformed = preprocessor.transform(X_sample.head(1))

    # Try to get feature names from the ColumnTransformer step
    ct = base.named_steps.get("preprocessor")
    if ct is not None and hasattr(ct, "get_feature_names_out"):
        try:
            return list(ct.get_feature_names_out())
        except Exception:
            pass

    return [f"feature_{i}" for i in range(transformed.shape[1])]


def explain_global(pipeline, X: pd.DataFrame, explainer: shap.TreeExplainer = None):
    """
    Compute mean absolute SHAP values across the dataset.
    Returns a Series indexed by feature name, sorted descending.
    """
    if explainer is None:
        explainer = build_explainer(pipeline)

    if hasattr(pipeline, "calibrated_classifiers_"):
        base = pipeline.calibrated_classifiers_[0].estimator
    else:
        base = pipeline

    preprocessor = base[:-1]
    X_transformed = preprocessor.transform(X)
    shap_values = explainer.shap_values(X_transformed)

    feature_names = get_feature_names(pipeline, X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)


def explain_patient(
    pipeline,
    patient_features: pd.DataFrame,
    explainer: shap.TreeExplainer = None,
    top_n: int = 5,
) -> dict:
    """
    Per-patient SHAP explanation.
    Returns {feature_name: shap_value} for the top_n most influential features.
    """
    if explainer is None:
        explainer = build_explainer(pipeline)

    if hasattr(pipeline, "calibrated_classifiers_"):
        base = pipeline.calibrated_classifiers_[0].estimator
    else:
        base = pipeline

    preprocessor = base[:-1]
    X_transformed = preprocessor.transform(patient_features)
    shap_values = explainer.shap_values(X_transformed)[0]

    feature_names = get_feature_names(pipeline, patient_features)
    shap_dict = dict(zip(feature_names, shap_values))

    # Return top_n by absolute value
    top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return {k: round(float(v), 6) for k, v in top}


def save_explainer(explainer: shap.TreeExplainer, path: str):
    joblib.dump(explainer, path)


def load_explainer(path: str) -> shap.TreeExplainer:
    return joblib.load(path)
