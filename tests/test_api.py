import sys
import os

# Add 'src' directory to sys.path to fix ModuleNotFoundError
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from fastapi.testclient import TestClient
from src.inference_api.main import app
  # changed from src.inference_api.main

client = TestClient(app)

# Sample valid payload (adjust if your schema changes)
valid_payload = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 350.5
}

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Customer Churn Prediction API is running."}

def test_predict_valid():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "Churn" in data["probability"]
    assert "No Churn" in data["probability"]

def test_predict_invalid():
    # Missing required fields
    response = client.post("/predict", json={})
    assert response.status_code == 422  # validation error

def test_predict_wrong_type():
    bad_payload = valid_payload.copy()
    bad_payload["SeniorCitizen"] = "not_an_int"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
