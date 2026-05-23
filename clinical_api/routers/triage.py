import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from fastapi import APIRouter

from ..config import DISCLAIMER
from ..models.common import risk_level
from ..models.triage import ModelResult, TriageInput, TriageResponse
from ..services import model_store
from ..services.triage import (
    can_run_ards, can_run_diabetes, can_run_heart_attack, can_run_stroke,
    composite_triage_level,
    map_ards, map_diabetes, map_heart_attack, map_stroke,
)

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=4)


def _run_model(name: str, record: dict) -> ModelResult | None:
    bundle = model_store.get(name)
    if bundle is None:
        return None

    import numpy as np

    X = pd.DataFrame([record])
    prob = float(bundle.pipeline.predict_proba(X)[:, 1][0])
    if getattr(bundle.pipeline, "_platt_calibrator", None) is not None:
        prob = float(bundle.pipeline._platt_calibrator.predict_proba(
            np.array([[prob]])
        )[:, 1][0])

    factors = model_store.get_top_shap_factors(bundle, X)

    return ModelResult(
        model=name,
        prediction=int(prob >= bundle.threshold),
        probability=round(prob, 4),
        risk_level=risk_level(prob),
        top_risk_factors=factors,
    )


@router.post("/triage/screen", response_model=TriageResponse)
async def triage_screen(body: TriageInput):
    # Build the list of (model_name, record) pairs for models that have enough data
    candidates = []
    if can_run_stroke(body):
        candidates.append(("stroke", map_stroke(body)))
    if can_run_diabetes(body):
        candidates.append(("diabetes", map_diabetes(body)))
    if can_run_heart_attack(body):
        candidates.append(("heart_attack", map_heart_attack(body)))
    if can_run_ards(body):
        candidates.append(("ards", map_ards(body)))

    # Run models in parallel using a thread pool (sklearn predict is CPU-bound)
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(_executor, _run_model, name, record)
        for name, record in candidates
    ]
    results_raw = await asyncio.gather(*tasks)

    results = [r for r in results_raw if r is not None]
    models_run = [r.model for r in results]
    prob_by_model = {r.model: r.probability for r in results}

    triage_level, composite = composite_triage_level(prob_by_model)

    return TriageResponse(
        triage_level=triage_level,
        composite_score=composite,
        models_run=models_run,
        results=results,
        disclaimer=DISCLAIMER,
        request_id=str(uuid.uuid4()),
    )
