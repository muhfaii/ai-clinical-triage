def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "models" in data
    assert "stroke" in data["models"]


def test_health_includes_model_versions(client):
    resp = client.get("/health")
    models = resp.json()["models"]
    for name in ("stroke", "heart_attack", "diabetes", "ards"):
        assert "version" in models[name]
        assert "threshold" in models[name]
