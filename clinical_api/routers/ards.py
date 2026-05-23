import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..config import DISCLAIMER
from ..models.common import PredictionResponse, RiskFactor, risk_level
from ..models.ards import ARDSInput
from ..services import model_store

router = APIRouter()


@router.post("/ards/predict", response_model=PredictionResponse)
def predict_ards(body: ARDSInput):
    bundle = model_store.get("ards")
    if bundle is None:
        raise HTTPException(503, "ARDS model not loaded")

    X = pd.DataFrame([body.model_dump()])

    prob = float(bundle.pipeline.predict_proba(X)[:, 1][0])

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
