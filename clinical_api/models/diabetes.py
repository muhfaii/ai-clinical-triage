from typing import Literal
from pydantic import BaseModel, Field


class DiabetesInput(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: float = Field(..., ge=0, le=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    smoking_history: Literal["never", "No Info", "current", "former", "not current", "ever"]
    bmi: float = Field(..., ge=10, le=100)
    HbA1c_level: float = Field(..., ge=3.5, le=15.0)
    blood_glucose_level: float = Field(..., ge=60, le=500)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "age": 44, "hypertension": 0, "heart_disease": 0,
        "smoking_history": "never", "bmi": 27.3, "HbA1c_level": 5.7,
        "blood_glucose_level": 140,
    }}}
