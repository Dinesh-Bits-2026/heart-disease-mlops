"""Tests for the FastAPI service (the model is replaced by a small fitted pipeline)."""
import pytest
from fastapi.testclient import TestClient

from api.app import app
from src import predict

VALID = {"age": 54, "sex": 1, "cp": 3, "trestbps": 130, "chol": 246, "fbs": 0,
         "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 2,
         "ca": 0, "thal": 3}


@pytest.fixture
def client(fitted_model, monkeypatch):
    monkeypatch.setattr(predict, "_model", fitted_model)
    return TestClient(app)


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("disease", "no disease")
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["risk_probability"] <= 1.0


def test_predict_missing_field_is_rejected(client):
    bad = {k: v for k, v in VALID.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_out_of_range_value_is_rejected(client):
    assert client.post("/predict", json={**VALID, "age": 500}).status_code == 422


def test_predict_invalid_thal_is_rejected(client):
    assert client.post("/predict", json={**VALID, "thal": 5}).status_code == 422


def test_predict_wrong_type_is_rejected(client):
    assert client.post("/predict", json={**VALID, "chol": "high"}).status_code == 422


def test_predict_unknown_field_is_rejected(client):
    assert client.post("/predict", json={**VALID, "favourite_colour": "red"}).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=VALID)
    text = client.get("/metrics").text
    assert "http_requests_total" in text
    assert "predictions_total" in text
