import joblib
import shap


def build_explainer(pipeline):
    """Build a TreeExplainer from the XGBoost classifier step of the pipeline."""
    model = pipeline.named_steps["classifier"]
    return shap.TreeExplainer(model)


def save_explainer(explainer, path: str):
    joblib.dump(explainer, path)


def load_explainer(path: str):
    return joblib.load(path)
