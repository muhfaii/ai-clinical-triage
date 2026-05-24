"""
Clinical AI MCP Server

Exposes the clinical_api FastAPI endpoints as MCP tools so any MCP-compatible
AI agent (Claude Desktop, etc.) can call them during a conversation.

Configuration:
  CLINICAL_API_URL  — base URL of the deployed API
                      default: http://localhost:8000
                      production: https://clinical-ai.fly.dev

Usage (stdio, for Claude Desktop):
  python -m clinical_mcp.server

Register in Claude Desktop (~/.claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "clinical-ai": {
        "command": "python",
        "args": ["-m", "clinical_mcp.server"],
        "env": { "CLINICAL_API_URL": "https://clinical-ai.fly.dev" }
      }
    }
  }
"""
import os
from typing import Annotated, Optional

import httpx
from fastmcp import FastMCP
from pydantic import Field

API_BASE = os.getenv("CLINICAL_API_URL", "http://localhost:8000").rstrip("/")

mcp = FastMCP(
    name="Clinical AI",
    instructions=(
        "Tools for clinical decision support. Each tool returns a probability score, "
        "risk level, and top contributing risk factors. Results are advisory only — "
        "they do not constitute a diagnosis. Always present findings to the clinician "
        "alongside the disclaimer field."
    ),
)


def _post(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(f"{API_BASE}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Health ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_api_health() -> dict:
    """Check which models are loaded and their current versions."""
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{API_BASE}/health")
        resp.raise_for_status()
        return resp.json()


# ── Stroke ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def predict_stroke_risk(
    gender: Annotated[str, Field(description="'Male', 'Female', or 'Other'")],
    age: Annotated[float, Field(description="Patient age in years (0–120)")],
    hypertension: Annotated[int, Field(description="0 = No, 1 = Yes")],
    heart_disease: Annotated[int, Field(description="0 = No, 1 = Yes")],
    ever_married: Annotated[str, Field(description="'Yes' or 'No'")],
    work_type: Annotated[str, Field(description="'children', 'Govt_job', 'Never_worked', 'Private', or 'Self-employed'")],
    Residence_type: Annotated[str, Field(description="'Rural' or 'Urban'")],
    avg_glucose_level: Annotated[float, Field(description="Average blood glucose in mg/dL (50–600)")],
    smoking_status: Annotated[str, Field(description="'formerly smoked', 'never smoked', 'smokes', or 'Unknown'")],
    bmi: Annotated[Optional[float], Field(description="Body mass index (10–100). Pass null if unknown — model will impute.")] = None,
) -> dict:
    """
    Predict stroke risk for a patient.
    Returns probability score, risk level (low/moderate/high/uncertain),
    and top SHAP risk factors. Threshold is recall-biased (0.10) for clinical safety.
    """
    return _post("/api/v1/stroke/predict", {
        "gender": gender, "age": age, "hypertension": hypertension,
        "heart_disease": heart_disease, "ever_married": ever_married,
        "work_type": work_type, "Residence_type": Residence_type,
        "avg_glucose_level": avg_glucose_level, "bmi": bmi,
        "smoking_status": smoking_status,
    })


# ── Heart Attack ───────────────────────────────────────────────────────────────

@mcp.tool()
def predict_heart_attack_risk(
    Age: Annotated[int, Field(description="Patient age (18–90)")],
    Sex: Annotated[str, Field(description="'Male' or 'Female'")],
    Cholesterol: Annotated[int, Field(description="Total cholesterol in mg/dL (120–400)")],
    BP_Systolic: Annotated[int, Field(description="Systolic blood pressure mmHg (80–200)")],
    BP_Diastolic: Annotated[int, Field(description="Diastolic blood pressure mmHg (50–130)")],
    Heart_Rate: Annotated[int, Field(description="Heart rate in bpm (30–150)")],
    Diabetes: Annotated[int, Field(description="0 = No, 1 = Yes")],
    Family_History: Annotated[int, Field(description="Family history of heart disease: 0 = No, 1 = Yes")],
    Smoking: Annotated[int, Field(description="0 = No, 1 = Yes")],
    Obesity: Annotated[int, Field(description="0 = No, 1 = Yes")],
    Alcohol_Consumption: Annotated[int, Field(description="0 = No, 1 = Yes")],
    Exercise_Hours_Per_Week: Annotated[float, Field(description="Hours of exercise per week (0–25)")],
    Diet: Annotated[str, Field(description="'Healthy', 'Average', or 'Unhealthy'")],
    Previous_Heart_Problems: Annotated[int, Field(description="0 = No, 1 = Yes")],
    Medication_Use: Annotated[int, Field(description="Currently on medication: 0 = No, 1 = Yes")],
    Stress_Level: Annotated[int, Field(description="Self-reported stress level 1–10")],
    Sedentary_Hours_Per_Day: Annotated[float, Field(description="Sedentary hours per day (0–16)")],
    Income: Annotated[int, Field(description="Annual income in USD (0–500000)")],
    BMI: Annotated[float, Field(description="Body mass index (10–60)")],
    Triglycerides: Annotated[int, Field(description="Triglycerides in mg/dL (20–1000)")],
    Physical_Activity_Days_Per_Week: Annotated[int, Field(description="Days physically active per week (0–7)")],
    Sleep_Hours_Per_Day: Annotated[int, Field(description="Sleep hours per day (4–12)")],
) -> dict:
    """Predict heart attack risk. Returns probability and top contributing factors."""
    return _post("/api/v1/heart-attack/predict", {
        "Age": Age, "Sex": Sex, "Cholesterol": Cholesterol,
        "BP_Systolic": BP_Systolic, "BP_Diastolic": BP_Diastolic,
        "Heart_Rate": Heart_Rate, "Diabetes": Diabetes,
        "Family_History": Family_History, "Smoking": Smoking,
        "Obesity": Obesity, "Alcohol_Consumption": Alcohol_Consumption,
        "Exercise_Hours_Per_Week": Exercise_Hours_Per_Week, "Diet": Diet,
        "Previous_Heart_Problems": Previous_Heart_Problems,
        "Medication_Use": Medication_Use, "Stress_Level": Stress_Level,
        "Sedentary_Hours_Per_Day": Sedentary_Hours_Per_Day, "Income": Income,
        "BMI": BMI, "Triglycerides": Triglycerides,
        "Physical_Activity_Days_Per_Week": Physical_Activity_Days_Per_Week,
        "Sleep_Hours_Per_Day": Sleep_Hours_Per_Day,
    })


# ── Diabetes ───────────────────────────────────────────────────────────────────

@mcp.tool()
def predict_diabetes_risk(
    gender: Annotated[str, Field(description="'Male', 'Female', or 'Other'")],
    age: Annotated[float, Field(description="Patient age in years (0–120)")],
    hypertension: Annotated[int, Field(description="0 = No, 1 = Yes")],
    heart_disease: Annotated[int, Field(description="0 = No, 1 = Yes")],
    smoking_history: Annotated[str, Field(description="'never', 'No Info', 'current', 'former', 'not current', or 'ever'")],
    bmi: Annotated[float, Field(description="Body mass index (10–100)")],
    HbA1c_level: Annotated[float, Field(description="Glycated haemoglobin % (3.5–15.0). WHO diagnostic threshold: 6.5%")],
    blood_glucose_level: Annotated[float, Field(description="Blood glucose in mg/dL (60–500). Hard-flag threshold: 200 mg/dL")],
) -> dict:
    """
    Predict diabetes risk. Applies WHO hard-flag overrides:
    blood_glucose >= 200 or HbA1c >= 6.5 raises probability to >= 0.85
    regardless of ML score.
    """
    return _post("/api/v1/diabetes/predict", {
        "gender": gender, "age": age, "hypertension": hypertension,
        "heart_disease": heart_disease, "smoking_history": smoking_history,
        "bmi": bmi, "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level,
    })


# ── ARDS ───────────────────────────────────────────────────────────────────────

@mcp.tool()
def predict_ards_mortality(
    Age: Annotated[int, Field(description="Patient age (18–120)")],
    Sex: Annotated[int, Field(description="0 = Female, 1 = Male")],
    BMI: Annotated[float, Field(description="Body mass index (10–80)")],
    Smoking_Status: Annotated[int, Field(description="0 = Never, 1 = Former, 2 = Current")],
    Hypertension: Annotated[int, Field(description="0 = No, 1 = Yes")],
    Diabetes: Annotated[int, Field(description="0 = No, 1 = Yes")],
    COPD: Annotated[int, Field(description="0 = No, 1 = Yes")],
    Cardiovascular_Disease: Annotated[int, Field(description="0 = No, 1 = Yes")],
    Chronic_Kidney_Disease: Annotated[int, Field(description="0 = No, 1 = Yes")],
    Liver_Disease: Annotated[int, Field(description="0 = No, 1 = Yes")],
    Oxygen_Saturation: Annotated[float, Field(description="SpO2 % (50–100)")],
    PaO2_FiO2_Ratio: Annotated[float, Field(description="P/F ratio (50–600). <200 = ARDS, <100 = severe")],
    Blood_Pressure_Systolic: Annotated[int, Field(description="Systolic BP mmHg (60–250)")],
    Blood_Pressure_Diastolic: Annotated[int, Field(description="Diastolic BP mmHg (30–150)")],
    Heart_Rate: Annotated[int, Field(description="Heart rate bpm (30–200)")],
    Respiratory_Rate: Annotated[int, Field(description="Respiratory rate breaths/min (6–60)")],
    CRP_Level: Annotated[float, Field(description="C-reactive protein mg/L (0–500)")],
    D_Dimer: Annotated[float, Field(description="D-dimer mg/L (0–50)")],
    Lactate_Level: Annotated[float, Field(description="Lactate mmol/L (0.1–20)")],
    Ventilation_Type: Annotated[int, Field(description="0 = None, 1 = Non-invasive, 2 = Invasive mechanical")],
) -> dict:
    """Predict ARDS in-hospital mortality risk for ICU patients."""
    return _post("/api/v1/ards/predict", {
        "Age": Age, "Sex": Sex, "BMI": BMI, "Smoking_Status": Smoking_Status,
        "Hypertension": Hypertension, "Diabetes": Diabetes, "COPD": COPD,
        "Cardiovascular_Disease": Cardiovascular_Disease,
        "Chronic_Kidney_Disease": Chronic_Kidney_Disease,
        "Liver_Disease": Liver_Disease, "Oxygen_Saturation": Oxygen_Saturation,
        "PaO2_FiO2_Ratio": PaO2_FiO2_Ratio,
        "Blood_Pressure_Systolic": Blood_Pressure_Systolic,
        "Blood_Pressure_Diastolic": Blood_Pressure_Diastolic,
        "Heart_Rate": Heart_Rate, "Respiratory_Rate": Respiratory_Rate,
        "CRP_Level": CRP_Level, "D_Dimer": D_Dimer, "Lactate_Level": Lactate_Level,
        "Ventilation_Type": Ventilation_Type,
    })


# ── Triage ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def run_triage_screen(
    gender: Annotated[str, Field(description="'Male', 'Female', or 'Other'")],
    age: Annotated[float, Field(description="Patient age in years")],
    hypertension: Annotated[int, Field(description="0 = No, 1 = Yes")],
    heart_disease: Annotated[int, Field(description="0 = No, 1 = Yes")],
    # Stroke fields
    ever_married: Annotated[Optional[str], Field(description="'Yes' or 'No'. Required for stroke model.")] = None,
    work_type: Annotated[Optional[str], Field(description="'children','Govt_job','Never_worked','Private','Self-employed'. Required for stroke.")] = None,
    Residence_type: Annotated[Optional[str], Field(description="'Rural' or 'Urban'. Required for stroke.")] = None,
    avg_glucose_level: Annotated[Optional[float], Field(description="Average glucose mg/dL. Required for stroke.")] = None,
    smoking_status: Annotated[Optional[str], Field(description="'formerly smoked','never smoked','smokes','Unknown'. Required for stroke.")] = None,
    bmi: Annotated[Optional[float], Field(description="BMI. Used by stroke (nullable) and diabetes.")] = None,
    # Diabetes fields
    smoking_history: Annotated[Optional[str], Field(description="'never','No Info','current','former','not current','ever'. Required for diabetes.")] = None,
    HbA1c_level: Annotated[Optional[float], Field(description="HbA1c %. Required for diabetes.")] = None,
    blood_glucose_level: Annotated[Optional[float], Field(description="Blood glucose mg/dL. Required for diabetes.")] = None,
) -> dict:
    """
    Run all applicable models in parallel and return a single triage level.

    Triage levels: critical → urgent → semi-urgent → non-urgent.
    Models run automatically based on which fields are provided:
      - Stroke: needs ever_married, work_type, Residence_type, avg_glucose_level, smoking_status
      - Diabetes: needs bmi, smoking_history, HbA1c_level, blood_glucose_level
      - Heart attack / ARDS: needs specialist fields (use individual tools for those)

    Use this at patient intake to get an instant triage recommendation.
    """
    return _post("/api/v1/triage/screen", {
        "gender": gender, "age": age,
        "hypertension": hypertension, "heart_disease": heart_disease,
        "ever_married": ever_married, "work_type": work_type,
        "Residence_type": Residence_type, "avg_glucose_level": avg_glucose_level,
        "smoking_status": smoking_status, "bmi": bmi,
        "smoking_history": smoking_history,
        "HbA1c_level": HbA1c_level, "blood_glucose_level": blood_glucose_level,
    })


# ── Sit-to-Stand ───────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_sit2stand(
    frames_json: Annotated[str, Field(
        description=(
            "JSON array of pose keyframes. Each frame must have four keys — "
            "'hip', 'knee', 'ankle', 'shoulder' — each a [x, y] coordinate pair "
            "in any consistent unit (pixels, normalised 0–1, etc.). "
            "Example: [{\"hip\":[320,400],\"knee\":[320,500],\"ankle\":[320,600],\"shoulder\":[320,250]}, ...]"
        )
    )],
    frame_rate: Annotated[float, Field(description="Video frame rate in Hz (e.g. 30.0)")] = 30.0,
) -> dict:
    """
    Analyse a sit-to-stand movement sequence from pose keyframes.

    Returns rep count, mean movement time, trunk flexion angle, angular velocity,
    per-rep breakdown, and clinical quality flags (excessive_trunk_flexion,
    slow_movement, low_angular_velocity).

    Frames must be extracted from video beforehand using a pose estimation tool
    such as MediaPipe Pose or OpenPose.
    """
    import json
    try:
        frames = json.loads(frames_json)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid frames_json: {e}"}

    return _post("/api/v1/sit2stand/analyze", {"frames": frames, "frame_rate": frame_rate})


if __name__ == "__main__":
    mcp.run()
