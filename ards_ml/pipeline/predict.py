"""
Inference interface — predict() as specified in Section 7.4.
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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
    Inference interface per Section 7.4.

    Args:
        patient_features: dict mapping 20 feature names to values.
                          Missing values passed as None.
        pipeline: fitted calibrated pipeline (loaded from artefacts).
        explainer: fitted shap.TreeExplainer (optional — loaded separately).
        threshold: override operating threshold (default: from threshold_config.yaml).
        model_version: semver string for audit logging.

    Returns:
        {
            "mortality_probability": float,
            "mortality_prediction": int,
            "operating_threshold": float,
            "shap_values": dict,
            "missing_features": list,
            "model_version": str,
        }
    """
    config = _load_threshold_config()
    operating_threshold = threshold if threshold is not None else config["operating_threshold"]

    # Track which features were missing
    missing_features = [k for k, v in patient_features.items() if v is None or (
        isinstance(v, float) and np.isnan(v)
    )]

    # Build single-row DataFrame
    X = pd.DataFrame([patient_features])

    # Probability
    mortality_probability = float(pipeline.predict_proba(X)[0, 1])
    mortality_prediction = int(mortality_probability >= operating_threshold)

    # SHAP
    shap_values = {}
    if explainer is not None:
        from ..explainability.shap_explainer import explain_patient
        try:
            shap_values = explain_patient(pipeline, X, explainer=explainer, top_n=5)
        except Exception as e:
            log.warning(f"SHAP explanation failed: {e}")

    result = {
        "mortality_probability": round(mortality_probability, 6),
        "mortality_prediction": mortality_prediction,
        "operating_threshold": operating_threshold,
        "shap_values": shap_values,
        "missing_features": missing_features,
        "model_version": model_version,
    }

    # Audit log
    log.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_hash": _input_hash(patient_features),
        "mortality_probability": result["mortality_probability"],
        "mortality_prediction": result["mortality_prediction"],
        "model_version": model_version,
    }))

    return result


def batch_predict(
    records: List[dict],
    pipeline,
    explainer=None,
    threshold: Optional[float] = None,
    model_version: str = "1.0.0",
) -> list[dict]:
    """Batch inference — optimised for ≥500 records/min throughput."""
    config = _load_threshold_config()
    operating_threshold = threshold if threshold is not None else config["operating_threshold"]

    X = pd.DataFrame(records)
    probas = pipeline.predict_proba(X)[:, 1]
    predictions = (probas >= operating_threshold).astype(int)

    results = []
    for i, (record, prob, pred) in enumerate(zip(records, probas, predictions)):
        missing = [k for k, v in record.items() if v is None or (
            isinstance(v, float) and np.isnan(v)
        )]
        results.append({
            "mortality_probability": round(float(prob), 6),
            "mortality_prediction": int(pred),
            "operating_threshold": operating_threshold,
            "shap_values": {},  # SHAP skipped in batch for throughput
            "missing_features": missing,
            "model_version": model_version,
        })
    return results
