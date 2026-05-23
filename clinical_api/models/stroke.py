from typing import Literal, Optional
from pydantic import BaseModel, Field


class StrokeInput(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: float = Field(..., ge=0, le=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    ever_married: Literal["Yes", "No"]
    work_type: Literal["children", "Govt_job", "Never_worked", "Private", "Self-employed"]
    Residence_type: Literal["Rural", "Urban"]
    avg_glucose_level: float = Field(..., ge=50, le=600)
    bmi: Optional[float] = Field(None, ge=10, le=100)
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "age": 67, "hypertension": 0, "heart_disease": 1,
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked",
    }}}
