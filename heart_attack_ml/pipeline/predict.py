"""
Inference interface — returns { "risk": 0/1, "probability": float, "top_factors": [...] }
as specified in Section 6.3 of requirements.
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "threshold_config.yaml"


def _load_threshold_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _input_hash(patient_features: dict) -> str:
    return hashlib.sha256(
        json.dumps(patient_features, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def predict(
    patient_features: dict,
    pipeline,
    explainer=None,
    threshold: Optional[float] = None,
    model_version: str = "1.0.0",
) -> dict:
    """
    Single-patient inference.

    Args:
        patient_features: raw dict matching the dataset schema (pre-engineering).
                          Blood Pressure as "SYS/DIA" string, Sex as "Male"/"Female", etc.
        pipeline: fitted pipeline loaded from artefacts.
        explainer: fitted shap.TreeExplainer (optional).
        threshold: override operating threshold (default: from threshold_config.yaml).
        model_version: semver string for audit logging.

    Returns:
        {
            "risk": int,           # 0 or 1
            "probability": float,  # calibrated probability of risk=1
            "top_factors": list,   # top 3 SHAP factors [{"feature": str, "shap": float}]
            "operating_threshold": float,
            "missing_features": list,
            "model_version": str,
        }
    """
    config = _load_threshold_config()
    operating_threshold = threshold if threshold is not None else config["operating_threshold"]

    missing_features = [
        k for k, v in patient_features.items()
        if v is None or (isinstance(v, float) and np.isnan(v))
    ]

    X = pd.DataFrame([patient_features])
    probability = float(pipeline.predict_proba(X)[0, 1])
    if hasattr(pipeline, "_platt_calibrator"):
        probability = float(
            pipeline._platt_calibrator.predict_proba([[probability]])[0, 1]
        )
    risk = int(probability >= operating_threshold)

    top_factors = []
    if explainer is not None:
        from ..explainability.shap_explainer import explain_patient
        try:
            shap_dict = explain_patient(pipeline, X, explainer=explainer, top_n=3)
            top_factors = [{"feature": k, "shap": v} for k, v in shap_dict.items()]
        except Exception as e:
            log.warning(f"SHAP explanation failed: {e}")

    result = {
        "risk": risk,
        "probability": round(probability, 6),
        "top_factors": top_factors,
        "operating_threshold": operating_threshold,
        "missing_features": missing_features,
        "model_version": model_version,
    }

    log.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_hash": _input_hash(patient_features),
        "probability": result["probability"],
        "risk": result["risk"],
        "model_version": model_version,
    }))

    return result
