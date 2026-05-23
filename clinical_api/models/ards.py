from typing import Literal
from pydantic import BaseModel, Field


class ARDSInput(BaseModel):
    Age: int = Field(..., ge=18, le=120)
    Sex: Literal[0, 1]                               # 0=Female, 1=Male
    BMI: float = Field(..., ge=10, le=80)
    Smoking_Status: Literal[0, 1, 2]                 # 0=Never, 1=Former, 2=Current
    Hypertension: Literal[0, 1]
    Diabetes: Literal[0, 1]
    COPD: Literal[0, 1]
    Cardiovascular_Disease: Literal[0, 1]
    Chronic_Kidney_Disease: Literal[0, 1]
    Liver_Disease: Literal[0, 1]
    Oxygen_Saturation: float = Field(..., ge=50, le=100)
    PaO2_FiO2_Ratio: float = Field(..., ge=50, le=600)
    Blood_Pressure_Systolic: int = Field(..., ge=60, le=250)
    Blood_Pressure_Diastolic: int = Field(..., ge=30, le=150)
    Heart_Rate: int = Field(..., ge=30, le=200)
    Respiratory_Rate: int = Field(..., ge=6, le=60)
    CRP_Level: float = Field(..., ge=0, le=500)
    D_Dimer: float = Field(..., ge=0, le=50)
    Lactate_Level: float = Field(..., ge=0.1, le=20)
    Ventilation_Type: Literal[0, 1, 2]              # 0=None, 1=Non-invasive, 2=Invasive

    model_config = {"json_schema_extra": {"example": {
        "Age": 62, "Sex": 1, "BMI": 29.1, "Smoking_Status": 1,
        "Hypertension": 1, "Diabetes": 0, "COPD": 0, "Cardiovascular_Disease": 0,
        "Chronic_Kidney_Disease": 0, "Liver_Disease": 0,
        "Oxygen_Saturation": 88.0, "PaO2_FiO2_Ratio": 180.0,
        "Blood_Pressure_Systolic": 110, "Blood_Pressure_Diastolic": 70,
        "Heart_Rate": 105, "Respiratory_Rate": 28,
        "CRP_Level": 120.0, "D_Dimer": 3.5, "Lactate_Level": 2.1,
        "Ventilation_Type": 1,
    }}}
