from typing import Literal, Optional
from pydantic import BaseModel, Field


class TriageInput(BaseModel):
    """
    Superset of all model input fields. Only fields relevant to each model
    need to be provided — the triage service runs whichever models have
    sufficient data and skips the rest.

    Stroke + Diabetes run on standard intake fields.
    Heart Attack additionally needs cholesterol/triglycerides (lab values).
    ARDS needs ICU-level vitals (SpO2, PaO2/FiO2, CRP, etc.).
    """
    # ── Demographics (all models) ──────────────────────────────────────────
    gender: Literal["Male", "Female", "Other"]
    age: float = Field(..., ge=0, le=120)

    # ── Common clinical history ────────────────────────────────────────────
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    bmi: Optional[float] = Field(None, ge=10, le=100)

    # ── Stroke-specific intake fields ─────────────────────────────────────
    ever_married: Optional[Literal["Yes", "No"]] = None
    work_type: Optional[Literal["children", "Govt_job", "Never_worked", "Private", "Self-employed"]] = None
    Residence_type: Optional[Literal["Rural", "Urban"]] = None
    avg_glucose_level: Optional[float] = Field(None, ge=50, le=600)
    smoking_status: Optional[Literal["formerly smoked", "never smoked", "smokes", "Unknown"]] = None

    # ── Diabetes-specific intake fields ───────────────────────────────────
    smoking_history: Optional[Literal["never", "No Info", "current", "former", "not current", "ever"]] = None
    HbA1c_level: Optional[float] = Field(None, ge=3.5, le=15.0)
    blood_glucose_level: Optional[float] = Field(None, ge=60, le=500)

    # ── Heart attack lab/vitals ────────────────────────────────────────────
    cholesterol: Optional[int] = Field(None, ge=120, le=400)
    bp_systolic: Optional[int] = Field(None, ge=80, le=200)
    bp_diastolic: Optional[int] = Field(None, ge=50, le=130)
    heart_rate: Optional[int] = Field(None, ge=30, le=150)
    triglycerides: Optional[int] = Field(None, ge=20, le=1000)
    family_history: Optional[Literal[0, 1]] = None
    smoking_binary: Optional[Literal[0, 1]] = None
    obesity: Optional[Literal[0, 1]] = None
    alcohol_consumption: Optional[Literal[0, 1]] = None
    exercise_hours: Optional[float] = Field(None, ge=0, le=25)
    diet: Optional[Literal["Healthy", "Average", "Unhealthy"]] = None
    previous_heart_problems: Optional[Literal[0, 1]] = None
    medication_use: Optional[Literal[0, 1]] = None
    stress_level: Optional[int] = Field(None, ge=1, le=10)
    sedentary_hours: Optional[float] = Field(None, ge=0, le=16)
    income: Optional[int] = Field(None, ge=0, le=500000)
    physical_activity_days: Optional[int] = Field(None, ge=0, le=7)
    sleep_hours: Optional[int] = Field(None, ge=4, le=12)

    # ── ARDS ICU vitals ────────────────────────────────────────────────────
    oxygen_saturation: Optional[float] = Field(None, ge=50, le=100)
    pao2_fio2_ratio: Optional[float] = Field(None, ge=50, le=600)
    respiratory_rate: Optional[int] = Field(None, ge=6, le=60)
    copd: Optional[Literal[0, 1]] = None
    cardiovascular_disease: Optional[Literal[0, 1]] = None
    chronic_kidney_disease: Optional[Literal[0, 1]] = None
    liver_disease: Optional[Literal[0, 1]] = None
    crp_level: Optional[float] = Field(None, ge=0, le=500)
    d_dimer: Optional[float] = Field(None, ge=0, le=50)
    lactate_level: Optional[float] = Field(None, ge=0.1, le=20)
    ventilation_type: Optional[Literal[0, 1, 2]] = None
    smoking_status_ards: Optional[Literal[0, 1, 2]] = None   # 0=Never,1=Former,2=Current

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "age": 67, "hypertension": 1, "heart_disease": 0,
        "bmi": 28.0, "ever_married": "Yes", "work_type": "Private",
        "Residence_type": "Urban", "avg_glucose_level": 200.0,
        "smoking_status": "formerly smoked",
        "smoking_history": "former", "HbA1c_level": 6.1, "blood_glucose_level": 180.0,
    }}}


class ModelResult(BaseModel):
    model: str
    prediction: int
    probability: float
    risk_level: Literal["low", "moderate", "high", "uncertain"]
    top_risk_factors: list[dict]


class TriageResponse(BaseModel):
    triage_level: Literal["critical", "urgent", "semi-urgent", "non-urgent"]
    composite_score: float
    models_run: list[str]
    results: list[ModelResult]
    disclaimer: str
    request_id: str
