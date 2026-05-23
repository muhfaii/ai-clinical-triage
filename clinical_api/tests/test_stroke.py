VALID_STROKE = {
    "gender": "Female", "age": 67, "hypertension": 0, "heart_disease": 1,
    "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
    "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked",
}


def test_stroke_predict_200(client):
    resp = client.post("/api/v1/stroke/predict", json=VALID_STROKE)
    assert resp.status_code == 200


def test_stroke_response_schema(client):
    resp = client.post("/api/v1/stroke/predict", json=VALID_STROKE)
    data = resp.json()
    for key in ("prediction", "probability", "risk_level", "disclaimer", "request_id"):
        assert key in data


def test_stroke_prediction_is_binary(client):
    resp = client.post("/api/v1/stroke/predict", json=VALID_STROKE)
    assert resp.json()["prediction"] in (0, 1)


def test_stroke_probability_in_unit_interval(client):
    resp = client.post("/api/v1/stroke/predict", json=VALID_STROKE)
    p = resp.json()["probability"]
    assert 0.0 <= p <= 1.0


def test_stroke_null_bmi_accepted(client):
    payload = {**VALID_STROKE, "bmi": None}
    resp = client.post("/api/v1/stroke/predict", json=payload)
    assert resp.status_code == 200


def test_stroke_invalid_gender_422(client):
    payload = {**VALID_STROKE, "gender": "Unknown"}
    resp = client.post("/api/v1/stroke/predict", json=payload)
    assert resp.status_code == 422


def test_stroke_disclaimer_present(client):
    resp = client.post("/api/v1/stroke/predict", json=VALID_STROKE)
    assert "clinical decision support" in resp.json()["disclaimer"]
