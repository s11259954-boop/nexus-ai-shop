"""
Unit Tests for FastAPI REST Endpoints.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "algorithms" in data


def test_get_products():
    response = client.get("/products?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "products" in data
    assert len(data["products"]) <= 10
    assert data["total"] > 0


def test_get_product_by_id():
    response = client.get("/products/PROD_ELEC_001")
    assert response.status_code == 200
    data = response.json()
    assert data["product_id"] == "PROD_ELEC_001"
    assert "Apple" in data["brand"]


def test_similar_products_endpoint():
    response = client.get("/recommendations/similar/PROD_ELEC_001?limit=4")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 4
    assert "explanation" in data["recommendations"][0]


def test_smart_alternatives_endpoint():
    response = client.get("/recommendations/smart-alternatives/PROD_ELEC_001?limit=3")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    for item in data["recommendations"]:
        assert item["price"] < 69999.0


def test_personalized_recommendations_endpoint():
    response = client.get("/recommendations/personalized/USER_TECH_001?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 5
    assert data["user_id"] == "USER_TECH_001"


def test_record_interaction():
    payload = {
        "user_id": "USER_TECH_001",
        "product_id": "PROD_ELEC_002",
        "interaction_type": "view"
    }
    response = client.post("/interactions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["interaction_type"] == "view"


def test_analytics_metrics_endpoint():
    response = client.get("/analytics/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "kpis" in data
    assert data["kpis"]["total_users"] > 0
    assert data["kpis"]["total_products"] > 0


def test_evaluation_endpoint():
    response = client.get("/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "evaluation_metrics" in data
    assert "ab_test_experiment" in data
