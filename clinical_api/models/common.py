from typing import Literal
from pydantic import BaseModel


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases", "decreases"]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: Literal["low", "moderate", "high", "uncertain"]
    top_risk_factors: list[RiskFactor]
    model_version: str
    disclaimer: str
    request_id: str


def risk_level(prob: float, low: float = 0.2, high: float = 0.5,
               uncertain_lo: float = 0.25, uncertain_hi: float = 0.45) -> str:
    """Maps a calibrated probability to a risk label.
    'uncertain' band takes precedence — signals borderline cases for clinician review.
    """
    if uncertain_lo <= prob <= uncertain_hi:
        return "uncertain"
    if prob < low:
        return "low"
    if prob >= high:
        return "high"
    return "moderate"
