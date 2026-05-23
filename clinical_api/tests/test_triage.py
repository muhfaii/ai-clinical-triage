FULL_INTAKE = {
    # demographics + stroke + diabetes fields (both models run)
    "gender": "Female", "age": 67, "hypertension": 1, "heart_disease": 0,
    "bmi": 28.0,
    "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
    "avg_glucose_level": 200.0, "smoking_status": "formerly smoked",
    "smoking_history": "former", "HbA1c_level": 6.1, "blood_glucose_level": 180.0,
}

DEMOGRAPHICS_ONLY = {
    "gender": "Male", "age": 45, "hypertension": 0, "heart_disease": 0,
}


def test_triage_200(client):
    resp = client.post("/api/v1/triage/screen", json=FULL_INTAKE)
    assert resp.status_code == 200


def test_triage_schema(client):
    resp = client.post("/api/v1/triage/screen", json=FULL_INTAKE)
    data = resp.json()
    for key in ("triage_level", "composite_score", "models_run", "results", "request_id"):
        assert key in data


def test_triage_level_valid_enum(client):
    resp = client.post("/api/v1/triage/screen", json=FULL_INTAKE)
    assert resp.json()["triage_level"] in ("critical", "urgent", "semi-urgent", "non-urgent")


def test_triage_runs_stroke_and_diabetes(client):
    resp = client.post("/api/v1/triage/screen", json=FULL_INTAKE)
    models_run = resp.json()["models_run"]
    assert "stroke" in models_run
    assert "diabetes" in models_run


def test_triage_skips_ards_without_vitals(client):
    resp = client.post("/api/v1/triage/screen", json=FULL_INTAKE)
    assert "ards" not in resp.json()["models_run"]


def test_triage_no_models_run_on_minimal_input(client):
    resp = client.post("/api/v1/triage/screen", json=DEMOGRAPHICS_ONLY)
    assert resp.status_code == 200
    assert resp.json()["models_run"] == []
    assert resp.json()["triage_level"] == "non-urgent"


def test_triage_composite_score_in_unit_interval(client):
    resp = client.post("/api/v1/triage/screen", json=FULL_INTAKE)
    assert 0.0 <= resp.json()["composite_score"] <= 1.0


def test_triage_results_have_model_names(client):
    resp = client.post("/api/v1/triage/screen", json=FULL_INTAKE)
    for r in resp.json()["results"]:
        assert r["model"] in ("stroke", "heart_attack", "diabetes", "ards")
