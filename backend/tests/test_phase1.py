# backend/tests/test_phase1.py

import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock

backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))
sys.path.append(str(backend_dir / 'api'))

from fastapi.testclient import TestClient
from main import app, threshold as model_threshold
from services.risk_service import classify_risk, get_recommendation
from database import get_db

client = TestClient(app)


def test_health_check_phase1():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["threshold"] == model_threshold
    assert "metadata" in data


def test_predict_phase1_extended():
    valid_payload = {
        "customer_id": "TEST-1234",
        "tenure": 1,
        "monthly_charges": 85.0,
        "contract": "Month-to-month",
        "gender": "Female",
        "senior_citizen": 1,
        "partner": "No",
        "dependents": "No",
        "internet_service": "Fiber optic",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "services_list": ["Streaming TV"]
    }

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["Yes", "No"]
    assert "probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "recommended_action" in data
    assert isinstance(data["recommended_action"], str)
    assert data["model_version"] == "v1.0"
    assert data["customer_id"] == "TEST-1234"


def test_risk_classification_rules():
    assert classify_risk(0.10) == "LOW"
    assert classify_risk(0.3299) == "LOW"
    assert classify_risk(0.33) == "MEDIUM"
    assert classify_risk(0.50) == "MEDIUM"
    assert classify_risk(0.6599) == "MEDIUM"
    assert classify_risk(0.66) == "HIGH"
    assert classify_risk(0.95) == "HIGH"


def test_retention_recommendation_rules():
    # HIGH risk + Month-to-month contract -> "Offer annual contract upgrade"
    rec1 = get_recommendation({"contract": "Month-to-month", "tech_support": "Yes", "monthly_charges": 50.0}, "HIGH")
    assert rec1 == "Offer annual contract upgrade"

    # HIGH risk + One year contract + no tech support -> "Offer technical support"
    rec2 = get_recommendation({"contract": "One year", "tech_support": "No", "monthly_charges": 50.0}, "HIGH")
    assert rec2 == "Offer technical support"

    # HIGH risk + One year contract + tech support + high monthly charges -> "Consider targeted retention discount"
    rec3 = get_recommendation({"contract": "One year", "tech_support": "Yes", "monthly_charges": 95.0}, "HIGH")
    assert rec3 == "Consider targeted retention discount"

    # MEDIUM risk -> "Launch customer engagement campaign"
    rec4 = get_recommendation({}, "MEDIUM")
    assert rec4 == "Launch customer engagement campaign"

    # LOW risk -> "No immediate retention action required"
    rec5 = get_recommendation({}, "LOW")
    assert rec5 == "No immediate retention action required"


def test_customers_endpoint_mock_db():
    mock_customer = SimpleNamespace(
        customer_id="7590-VHVEG",
        gender="Female",
        senior_citizen=0,
        partner="Yes",
        dependents="No",
        tenure=1,
        phone_service="No",
        multiple_lines="No phone service",
        internet_service="DSL",
        online_security="No",
        online_backup="Yes",
        device_protection="No",
        tech_support="No",
        streaming_tv="No",
        streaming_movies="No",
        contract="Month-to-month",
        paperless_billing="Yes",
        payment_method="Electronic check",
        monthly_charges=29.85,
        total_charges=29.85,
        churn="No"
    )

    mock_db = MagicMock()
    mock_query = MagicMock()

    # Configure chain methods on mock_query
    mock_query.filter.return_value = mock_query
    mock_query.count.return_value = 1
    mock_query.offset.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.all.return_value = [mock_customer]
    mock_query.first.return_value = mock_customer

    mock_db.query.return_value = mock_query

    def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    try:
        # GET /customers
        res_list = client.get("/customers?page=1&limit=20")
        assert res_list.status_code == 200
        data_list = res_list.json()
        assert data_list["total"] == 1
        assert len(data_list["customers"]) == 1
        cust = data_list["customers"][0]
        assert cust["customer_id"] == "7590-VHVEG"
        assert "risk_level" in cust

        # GET /customers/{customer_id}
        res_single = client.get("/customers/7590-VHVEG")
        assert res_single.status_code == 200
        data_single = res_single.json()
        assert data_single["customer_id"] == "7590-VHVEG"
        assert "recommended_action" in data_single

        # GET /analytics/overview
        res_overview = client.get("/analytics/overview")
        assert res_overview.status_code == 200
        data_overview = res_overview.json()
        assert data_overview["total_customers"] == 1
        assert "revenue_at_risk" in data_overview
        assert "churn_rate" in data_overview

    finally:
        app.dependency_overrides.clear()
