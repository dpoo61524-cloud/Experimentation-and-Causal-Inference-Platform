"""Automated tests for FastAPI endpoints in the Experimentation Platform."""

from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api import app

client = TestClient(app)


# --- Root and Health Endpoints ---

def test_root_endpoint():
    """Verify that the root endpoint returns application info and purpose."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "description" in data
    assert "version" in data
    assert "docs_url" in data
    assert "Experimentation and Causal Inference Platform API" in data["name"]
    assert "traceback" not in response.text.lower()


def test_health_endpoint():
    """Verify that the health endpoint returns a healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_docs_endpoint_available():
    """Verify that interactive OpenAPI documentation is generated and accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "html" in response.text.lower()


# --- Experiment Compare Endpoint ---

def test_valid_experiment_comparison():
    """Verify a valid experiment comparison calculation using two groups."""
    payload = {
        "group_a": {"conversions": 267, "visitors": 21307},
        "group_b": {"conversions": 122, "visitors": 21306},
    }
    response = client.post("/api/experiment/compare", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["conversion_rate_a"] == pytest.approx(267 / 21307)
    assert data["conversion_rate_b"] == pytest.approx(122 / 21306)
    assert data["conversion_rate_a"] > data["conversion_rate_b"]
    assert data["absolute_difference"] == pytest.approx(
        (267 / 21307) - (122 / 21306)
    )
    assert data["p_value"] < 0.05
    assert data["statistically_significant"] is True
    assert len(data["confidence_interval_95"]) == 2
    assert data["confidence_interval_95"][0] < data["confidence_interval_95"][1]
    assert "z_statistic" in data


def test_valid_experiment_comparison_with_alias():
    """Verify that comparison endpoint also accepts flat or control/treatment aliases."""
    payload_flat = {
        "conversions_a": 20,
        "visitors_a": 1000,
        "conversions_b": 10,
        "visitors_b": 1000,
    }
    response = client.post("/api/experiment/compare", json=payload_flat)
    assert response.status_code == 200
    data = response.json()
    assert data["conversion_rate_a"] == pytest.approx(0.02)
    assert data["conversion_rate_b"] == pytest.approx(0.01)

    payload_named = {
        "control": {"conversions": 10, "visitors": 1000},
        "treatment": {"conversions": 20, "visitors": 1000},
    }
    response2 = client.post("/api/experiment/compare", json=payload_named)
    assert response2.status_code == 200


def test_zero_visitors_rejected():
    """Verify that zero visitors in either group is rejected with 400 or 422."""
    payload = {
        "group_a": {"conversions": 0, "visitors": 0},
        "group_b": {"conversions": 10, "visitors": 100},
    }
    response = client.post("/api/experiment/compare", json=payload)
    assert response.status_code in (400, 422)
    assert "detail" in response.json()
    assert "traceback" not in response.text.lower()


def test_negative_visitors_rejected():
    """Verify that negative visitors count is rejected with 400 or 422."""
    payload = {
        "group_a": {"conversions": 10, "visitors": -100},
        "group_b": {"conversions": 10, "visitors": 100},
    }
    response = client.post("/api/experiment/compare", json=payload)
    assert response.status_code in (400, 422)
    assert "detail" in response.json()


def test_conversions_greater_than_visitors_rejected():
    """Verify that conversions exceeding visitors count is rejected."""
    payload = {
        "group_a": {"conversions": 150, "visitors": 100},
        "group_b": {"conversions": 10, "visitors": 100},
    }
    response = client.post("/api/experiment/compare", json=payload)
    assert response.status_code in (400, 422)
    assert "detail" in response.json()
    assert "traceback" not in response.text.lower()


def test_negative_conversions_rejected():
    """Verify that negative conversions are rejected."""
    payload = {
        "group_a": {"conversions": -5, "visitors": 100},
        "group_b": {"conversions": 10, "visitors": 100},
    }
    response = client.post("/api/experiment/compare", json=payload)
    assert response.status_code in (400, 422)
    assert "detail" in response.json()


# --- Sample Size Endpoint ---

def test_valid_sample_size_request():
    """Verify valid sample size calculation matches statistical engine output."""
    payload = {
        "baseline_rate": 0.10,
        "expected_rate": 0.12,
        "alpha": 0.05,
        "power": 0.80,
    }
    response = client.post("/api/experiment/sample-size", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sample_size_per_group"] == 3835
    assert data["total_sample_size"] == 7670
    assert data["baseline_rate"] == 0.10
    assert data["expected_rate"] == 0.12
    assert data["alpha"] == 0.05
    assert data["power"] == 0.80
    assert data["absolute_effect"] == pytest.approx(0.02)


def test_sample_size_default_alpha_and_power():
    """Verify that alpha and power have appropriate defaults when omitted."""
    payload = {
        "baseline_rate": 0.10,
        "expected_rate": 0.12,
    }
    response = client.post("/api/experiment/sample-size", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["alpha"] == 0.05
    assert data["power"] == 0.80
    assert data["sample_size_per_group"] == 3835
    assert data["total_sample_size"] == 7670


def test_sample_size_aliases_supported():
    """Verify that descriptive field aliases are supported."""
    payload = {
        "baseline_conversion_rate": 0.10,
        "expected_conversion_rate": 0.12,
        "significance_level": 0.05,
        "statistical_power": 0.80,
    }
    response = client.post("/api/experiment/sample-size", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sample_size_per_group"] == 3835


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"baseline_rate": 0.0, "expected_rate": 0.12},
        {"baseline_rate": -0.05, "expected_rate": 0.12},
        {"baseline_rate": 1.0, "expected_rate": 0.12},
        {"baseline_rate": 1.5, "expected_rate": 0.12},
        {"baseline_rate": 0.10, "expected_rate": 0.0},
        {"baseline_rate": 0.10, "expected_rate": 1.0},
        {"baseline_rate": 0.10, "expected_rate": 1.2},
        {"baseline_rate": 0.10, "expected_rate": 0.12, "alpha": 0.0},
        {"baseline_rate": 0.10, "expected_rate": 0.12, "alpha": 1.0},
        {"baseline_rate": 0.10, "expected_rate": 0.12, "power": 0.0},
        {"baseline_rate": 0.10, "expected_rate": 0.12, "power": 1.0},
        {"baseline_rate": 0.10, "expected_rate": 0.10},  # equal rates
    ],
)
def test_invalid_probabilities_rejected(invalid_payload):
    """Verify that invalid probabilities and equal rates are rejected."""
    response = client.post("/api/experiment/sample-size", json=invalid_payload)
    assert response.status_code in (400, 422)
    assert "detail" in response.json()
    assert "traceback" not in response.text.lower()


# --- Missing and Invalid Field Types ---

def test_missing_request_fields():
    """Verify that missing required fields return HTTP 422."""
    # Empty body for compare
    resp1 = client.post("/api/experiment/compare", json={})
    assert resp1.status_code == 422
    assert "detail" in resp1.json()

    # Missing visitors for group A
    resp2 = client.post(
        "/api/experiment/compare",
        json={"group_a": {"conversions": 10}, "group_b": {"conversions": 5, "visitors": 100}},
    )
    assert resp2.status_code == 422

    # Empty body for sample size
    resp3 = client.post("/api/experiment/sample-size", json={})
    assert resp3.status_code == 422

    # Missing expected_rate
    resp4 = client.post("/api/experiment/sample-size", json={"baseline_rate": 0.10})
    assert resp4.status_code == 422


def test_invalid_field_types():
    """Verify that invalid input types return HTTP 422."""
    # String instead of int
    resp1 = client.post(
        "/api/experiment/compare",
        json={
            "group_a": {"conversions": "many", "visitors": 100},
            "group_b": {"conversions": 5, "visitors": 100},
        },
    )
    assert resp1.status_code == 422

    # Boolean instead of int
    resp2 = client.post(
        "/api/experiment/compare",
        json={
            "group_a": {"conversions": True, "visitors": 100},
            "group_b": {"conversions": 5, "visitors": 100},
        },
    )
    assert resp2.status_code == 422

    # String instead of float
    resp3 = client.post(
        "/api/experiment/sample-size",
        json={"baseline_rate": "high", "expected_rate": 0.12},
    )
    assert resp3.status_code == 422

    # Boolean instead of float
    resp4 = client.post(
        "/api/experiment/sample-size",
        json={"baseline_rate": True, "expected_rate": 0.12},
    )
    assert resp4.status_code == 422
