# backend/tests/test_api.py

import sys
from pathlib import Path
import pytest

# Add backend and backend/api to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))
sys.path.append(str(backend_dir / 'api'))

from fastapi.testclient import TestClient
from api.main import app, threshold as model_threshold

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert data["model_loaded"] is True
    assert "threshold" in data
    assert data["threshold"] == model_threshold

def test_predict_endpoint_valid_input():
    valid_payload = {
        "tenure": 12,
        "monthly_charges": 75.5,
        "contract": "Month-to-month",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "internet_service": "Fiber optic",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "services_list": ["Online Security", "Tech Support"]
    }

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["Yes", "No"]
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert "threshold" in data
    assert data["threshold"] == model_threshold

def test_predict_endpoint_invalid_input():
    invalid_payload = {
        "tenure": -5,  # Invalid negative tenure
        "monthly_charges": 75.5,
        "contract": "Month-to-month"
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # FastAPI validation error
