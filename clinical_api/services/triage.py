"""
Composite triage logic: maps TriageInput fields to each model's record format,
decides which models can run, and computes the combined triage level.
"""
from ..models.triage import TriageInput

# Stroke and diabetes are front-door models (intake fields only).
# Heart attack and ARDS need specialist/lab data — only run if provided.
TRIAGE_WEIGHTS = {
    "stroke":       0.35,
    "heart_attack": 0.30,
    "diabetes":     0.15,
    "ards":         0.20,
}


# ── Field presence checks ──────────────────────────────────────────────────────

def can_run_stroke(inp: TriageInput) -> bool:
    return all([
        inp.ever_married is not None,
        inp.work_type is not None,
        inp.Residence_type is not None,
        inp.avg_glucose_level is not None,
        inp.smoking_status is not None,
    ])


def can_run_diabetes(inp: TriageInput) -> bool:
    return all([
        inp.bmi is not None,
        inp.HbA1c_level is not None,
        inp.blood_glucose_level is not None,
        inp.smoking_history is not None,
    ])


def can_run_heart_attack(inp: TriageInput) -> bool:
    return all([
        inp.cholesterol is not None,
        inp.bp_systolic is not None,
        inp.bp_diastolic is not None,
        inp.heart_rate is not None,
        inp.triglycerides is not None,
        inp.diet is not None,
        inp.exercise_hours is not None,
        inp.sedentary_hours is not None,
        inp.income is not None,
        inp.physical_activity_days is not None,
        inp.sleep_hours is not None,
        inp.stress_level is not None,
        inp.family_history is not None,
        inp.smoking_binary is not None,
        inp.obesity is not None,
        inp.alcohol_consumption is not None,
        inp.previous_heart_problems is not None,
        inp.medication_use is not None,
        inp.bmi is not None,
    ])


def can_run_ards(inp: TriageInput) -> bool:
    return all([
        inp.oxygen_saturation is not None,
        inp.pao2_fio2_ratio is not None,
        inp.bp_systolic is not None,
        inp.bp_diastolic is not None,
        inp.heart_rate is not None,
        inp.respiratory_rate is not None,
        inp.crp_level is not None,
        inp.d_dimer is not None,
        inp.lactate_level is not None,
        inp.ventilation_type is not None,
        inp.smoking_status_ards is not None,
        inp.copd is not None,
        inp.cardiovascular_disease is not None,
        inp.chronic_kidney_disease is not None,
        inp.liver_disease is not None,
        inp.bmi is not None,
    ])


# ── Field mappers ──────────────────────────────────────────────────────────────

def map_stroke(inp: TriageInput) -> dict:
    return {
        "gender": inp.gender,
        "age": inp.age,
        "hypertension": inp.hypertension,
        "heart_disease": inp.heart_disease,
        "ever_married": inp.ever_married,
        "work_type": inp.work_type,
        "Residence_type": inp.Residence_type,
        "avg_glucose_level": inp.avg_glucose_level,
        "bmi": inp.bmi,            # nullable; stroke imputer handles it
        "smoking_status": inp.smoking_status,
    }


def map_diabetes(inp: TriageInput) -> dict:
    return {
        "gender": inp.gender,
        "age": inp.age,
        "hypertension": inp.hypertension,
        "heart_disease": inp.heart_disease,
        "smoking_history": inp.smoking_history,
        "bmi": inp.bmi,
        "HbA1c_level": inp.HbA1c_level,
        "blood_glucose_level": inp.blood_glucose_level,
    }


def map_heart_attack(inp: TriageInput) -> dict:
    return {
        "Age": inp.age,
        "Sex": inp.gender if inp.gender in ("Male", "Female") else "Female",
        "Cholesterol": inp.cholesterol,
        "Blood Pressure": f"{inp.bp_systolic}/{inp.bp_diastolic}",
        "Heart Rate": inp.heart_rate,
        "Diabetes": inp.heart_disease,   # re-uses heart_disease as diabetes proxy
        "Family History": inp.family_history,
        "Smoking": inp.smoking_binary,
        "Obesity": inp.obesity,
        "Alcohol Consumption": inp.alcohol_consumption,
        "Exercise Hours Per Week": inp.exercise_hours,
        "Diet": inp.diet,
        "Previous Heart Problems": inp.previous_heart_problems,
        "Medication Use": inp.medication_use,
        "Stress Level": inp.stress_level,
        "Sedentary Hours Per Day": inp.sedentary_hours,
        "Income": inp.income,
        "BMI": inp.bmi,
        "Triglycerides": inp.triglycerides,
        "Physical Activity Days Per Week": inp.physical_activity_days,
        "Sleep Hours Per Day": inp.sleep_hours,
    }


def map_ards(inp: TriageInput) -> dict:
    return {
        "Age": inp.age,
        "Sex": 1 if inp.gender == "Male" else 0,
        "BMI": inp.bmi,
        "Smoking_Status": inp.smoking_status_ards,
        "Hypertension": inp.hypertension,
        "Diabetes": inp.heart_disease,
        "COPD": inp.copd,
        "Cardiovascular_Disease": inp.cardiovascular_disease,
        "Chronic_Kidney_Disease": inp.chronic_kidney_disease,
        "Liver_Disease": inp.liver_disease,
        "Oxygen_Saturation": inp.oxygen_saturation,
        "PaO2_FiO2_Ratio": inp.pao2_fio2_ratio,
        "Blood_Pressure_Systolic": inp.bp_systolic,
        "Blood_Pressure_Diastolic": inp.bp_diastolic,
        "Heart_Rate": inp.heart_rate,
        "Respiratory_Rate": inp.respiratory_rate,
        "CRP_Level": inp.crp_level,
        "D_Dimer": inp.d_dimer,
        "Lactate_Level": inp.lactate_level,
        "Ventilation_Type": inp.ventilation_type,
    }


# ── Composite risk ─────────────────────────────────────────────────────────────

def composite_triage_level(model_results: dict[str, float]) -> tuple[str, float]:
    """
    Returns (triage_level, composite_score).
    model_results: {model_name: probability}

    Rules (applied in order):
      critical   — any single model probability >= 0.70, or composite >= 0.55
      urgent     — any single model >= 0.45, or composite >= 0.35
      semi-urgent — any single model >= 0.25, or composite >= 0.20
      non-urgent  — otherwise
    """
    if not model_results:
        return "non-urgent", 0.0

    total_weight = sum(TRIAGE_WEIGHTS[m] for m in model_results)
    composite = sum(TRIAGE_WEIGHTS[m] * p for m, p in model_results.items()) / total_weight
    max_prob = max(model_results.values())

    if max_prob >= 0.70 or composite >= 0.55:
        level = "critical"
    elif max_prob >= 0.45 or composite >= 0.35:
        level = "urgent"
    elif max_prob >= 0.25 or composite >= 0.20:
        level = "semi-urgent"
    else:
        level = "non-urgent"

    return level, round(composite, 4)
