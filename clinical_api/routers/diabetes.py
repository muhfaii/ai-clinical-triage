import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..config import DISCLAIMER
from ..models.common import PredictionResponse, RiskFactor, risk_level
from ..models.diabetes import DiabetesInput
from ..services import model_store

router = APIRouter()


@router.post("/diabetes/predict", response_model=PredictionResponse)
def predict_diabetes(body: DiabetesInput):
    bundle = model_store.get("diabetes")
    if bundle is None:
        raise HTTPException(503, "Diabetes model not loaded")

    record = body.model_dump()
    X = pd.DataFrame([record])

    prob = float(bundle.pipeline.predict_proba(X)[:, 1][0])

    # Clinical hard-flag overrides: WHO diagnostic thresholds take precedence over ML
    rules = bundle.clinical_rules
    if body.blood_glucose_level >= rules.get("blood_glucose_hard_flag", 9999):
        prob = max(prob, 0.85)
    if body.HbA1c_level >= rules.get("hba1c_hard_flag", 9999):
        prob = max(prob, 0.85)

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
