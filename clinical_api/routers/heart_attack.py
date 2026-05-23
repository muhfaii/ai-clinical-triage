import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..config import DISCLAIMER
from ..models.common import PredictionResponse, RiskFactor, risk_level
from ..models.heart_attack import HeartAttackInput
from ..services import model_store

router = APIRouter()


@router.post("/heart-attack/predict", response_model=PredictionResponse)
def predict_heart_attack(body: HeartAttackInput):
    bundle = model_store.get("heart_attack")
    if bundle is None:
        raise HTTPException(503, "Heart attack model not loaded")

    X = pd.DataFrame([body.to_pipeline_record()])

    prob = float(bundle.pipeline.predict_proba(X)[:, 1][0])
    if getattr(bundle.pipeline, "_platt_calibrator", None) is not None:
        import numpy as np
        prob = float(bundle.pipeline._platt_calibrator.predict_proba(
            np.array([[prob]])
        )[:, 1][0])

    factors = model_store.get_top_shap_factors(bundle, X)

    return PredictionResponse(
        prediction=int(prob >= bundle.threshold),
        probability=round(prob, 4),
        risk_level=risk_level(prob),
        top_risk_factors=[RiskFactor(**f) for f in factors],
        model_version=bundle.model_version,
        disclaimer=DISCLAIMER,
        request_id=str(uuid.uuid4()),
    )
