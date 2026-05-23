from typing import Literal
from pydantic import BaseModel, Field


class HeartAttackInput(BaseModel):
    Age: int = Field(..., ge=18, le=90)
    Sex: Literal["Male", "Female"]
    Cholesterol: int = Field(..., ge=120, le=400)
    # Accepted as separate ints; formatted to "SYS/DIA" for the pipeline internally
    BP_Systolic: int = Field(..., ge=80, le=200)
    BP_Diastolic: int = Field(..., ge=50, le=130)
    Heart_Rate: int = Field(..., ge=30, le=150)
    Diabetes: Literal[0, 1]
    Family_History: Literal[0, 1]
    Smoking: Literal[0, 1]
    Obesity: Literal[0, 1]
    Alcohol_Consumption: Literal[0, 1]
    Exercise_Hours_Per_Week: float = Field(..., ge=0, le=25)
    Diet: Literal["Healthy", "Average", "Unhealthy"]
    Previous_Heart_Problems: Literal[0, 1]
    Medication_Use: Literal[0, 1]
    Stress_Level: int = Field(..., ge=1, le=10)
    Sedentary_Hours_Per_Day: float = Field(..., ge=0, le=16)
    Income: int = Field(..., ge=0, le=500000)
    BMI: float = Field(..., ge=10, le=60)
    Triglycerides: int = Field(..., ge=20, le=1000)
    Physical_Activity_Days_Per_Week: int = Field(..., ge=0, le=7)
    Sleep_Hours_Per_Day: int = Field(..., ge=4, le=12)

    model_config = {"json_schema_extra": {"example": {
        "Age": 55, "Sex": "Male", "Cholesterol": 220, "BP_Systolic": 130,
        "BP_Diastolic": 85, "Heart_Rate": 78, "Diabetes": 0, "Family_History": 1,
        "Smoking": 0, "Obesity": 0, "Alcohol_Consumption": 0,
        "Exercise_Hours_Per_Week": 3.5, "Diet": "Average", "Previous_Heart_Problems": 0,
        "Medication_Use": 1, "Stress_Level": 6, "Sedentary_Hours_Per_Day": 6.0,
        "Income": 60000, "BMI": 27.4, "Triglycerides": 180,
        "Physical_Activity_Days_Per_Week": 3, "Sleep_Hours_Per_Day": 7,
    }}}

    def to_pipeline_record(self) -> dict:
        """Formats fields to match what heart_attack_ml's FeatureEngineer expects."""
        return {
            "Age": self.Age,
            "Sex": self.Sex,
            "Cholesterol": self.Cholesterol,
            "Blood Pressure": f"{self.BP_Systolic}/{self.BP_Diastolic}",
            "Heart Rate": self.Heart_Rate,
            "Diabetes": self.Diabetes,
            "Family History": self.Family_History,
            "Smoking": self.Smoking,
            "Obesity": self.Obesity,
            "Alcohol Consumption": self.Alcohol_Consumption,
            "Exercise Hours Per Week": self.Exercise_Hours_Per_Week,
            "Diet": self.Diet,
            "Previous Heart Problems": self.Previous_Heart_Problems,
            "Medication Use": self.Medication_Use,
            "Stress Level": self.Stress_Level,
            "Sedentary Hours Per Day": self.Sedentary_Hours_Per_Day,
            "Income": self.Income,
            "BMI": self.BMI,
            "Triglycerides": self.Triglycerides,
            "Physical Activity Days Per Week": self.Physical_Activity_Days_Per_Week,
            "Sleep Hours Per Day": self.Sleep_Hours_Per_Day,
        }
