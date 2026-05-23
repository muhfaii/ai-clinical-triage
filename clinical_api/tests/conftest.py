"""
Patches model_store so tests run without loading pkl files.
Each mock bundle returns a fixed probability of 0.6 for all models.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from clinical_api.services.model_store import ModelBundle


class _MockPipeline:
    _platt_calibrator = None

    def predict_proba(self, X):
        return np.array([[0.4, 0.6]] * len(X))

    def __getitem__(self, key):
        return self

    def transform(self, X):
        return np.zeros((len(X), 5))

    def get_feature_names_out(self):
        return ["f0", "f1", "f2", "f3", "f4"]


_MOCK_BUNDLE = ModelBundle(
    pipeline=_MockPipeline(),
    explainer=None,
    threshold=0.5,
    model_version="test-1.0",
    clinical_rules={"blood_glucose_hard_flag": 200, "hba1c_hard_flag": 6.5},
)


@pytest.fixture()
def client(monkeypatch):
    from clinical_api.services import model_store

    monkeypatch.setattr(model_store, "_store", {
        "stroke":       _MOCK_BUNDLE,
        "heart_attack": _MOCK_BUNDLE,
        "diabetes":     _MOCK_BUNDLE,
        "ards":         _MOCK_BUNDLE,
    })

    from clinical_api.main import app
    return TestClient(app)
