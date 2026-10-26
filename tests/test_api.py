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


# --- Two-Sample t-Test Endpoint Tests ---

def test_ttest_valid_request():
    """Verify successful Welch's t-test calculation via API."""
    payload = {
        "group_a": [12, 15, 14, 10, 13, 14, 16],
        "group_b": [9, 10, 11, 8, 12, 10, 9],
        "alternative": "two-sided",
        "alpha": 0.05,
    }
    response = client.post("/api/experiment/ttest", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["sample_size_a"] == 7
    assert data["sample_size_b"] == 7
    assert data["mean_a"] == pytest.approx(13.42857, rel=1e-4)
    assert data["mean_b"] == pytest.approx(9.85714, rel=1e-4)
    assert data["difference_in_means"] == pytest.approx(3.57143, rel=1e-4)
    assert data["t_statistic"] == pytest.approx(3.93648, rel=1e-4)
    assert data["p_value"] == pytest.approx(0.00252, rel=1e-3)
    assert data["statistically_significant"] is True
    assert data["degrees_of_freedom"] == pytest.approx(10.5419, rel=1e-3)
    assert data["confidence_interval_95"][0] < data["confidence_interval_95"][1]
    assert data["alternative"] == "two-sided"


def test_ttest_alternative_greater():
    """Verify t-test endpoint with 'greater' alternative hypothesis."""
    payload = {
        "group_a": [15.0, 16.0, 14.5, 15.5],
        "group_b": [10.0, 11.0, 9.5, 10.5],
        "alternative": "greater",
    }
    response = client.post("/api/experiment/ttest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["p_value"] < 0.05
    assert data["statistically_significant"] is True
    assert data["alternative"] == "greater"


def test_ttest_unequal_sample_sizes():
    """Verify t-test endpoint handles unequal group sizes."""
    payload = {
        "group_a": [10.5, 11.2, 9.8, 10.1, 12.0, 11.5, 10.9, 11.1],
        "group_b": [8.2, 8.5, 9.1, 8.8],
    }
    response = client.post("/api/experiment/ttest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sample_size_a"] == 8
    assert data["sample_size_b"] == 4


def test_ttest_constant_values_where_supported():
    """Verify t-test endpoint when one group is constant and one has variance."""
    payload = {
        "group_a": [10.0, 10.0, 10.0, 10.0],
        "group_b": [5.0, 8.0, 12.0, 15.0],
    }
    response = client.post("/api/experiment/ttest", json=payload)
    assert response.status_code == 200


def test_ttest_both_constant_rejected():
    """Verify that both groups with zero variance are rejected with HTTP 400."""
    payload = {
        "group_a": [5.0, 5.0, 5.0],
        "group_b": [5.0, 5.0, 5.0],
    }
    response = client.post("/api/experiment/ttest", json=payload)
    assert response.status_code in (400, 422)
    assert "detail" in response.json()
    assert "traceback" not in response.text.lower()


def test_ttest_insufficient_samples_rejected():
    """Verify that samples with fewer than 2 elements return 422."""
    payload1 = {"group_a": [10.0], "group_b": [10.0, 12.0]}
    response1 = client.post("/api/experiment/ttest", json=payload1)
    assert response1.status_code == 422

    payload2 = {"group_a": [], "group_b": [10.0, 12.0]}
    response2 = client.post("/api/experiment/ttest", json=payload2)
    assert response2.status_code == 422


def test_ttest_invalid_element_types_rejected():
    """Verify that non-numeric or missing elements return 422."""
    # String element
    res1 = client.post(
        "/api/experiment/ttest",
        json={"group_a": [1.0, "bad", 3.0], "group_b": [4.0, 5.0]},
    )
    assert res1.status_code == 422

    # None element
    res2 = client.post(
        "/api/experiment/ttest",
        json={"group_a": [1.0, None, 3.0], "group_b": [4.0, 5.0]},
    )
    assert res2.status_code == 422

    # Boolean element
    res3 = client.post(
        "/api/experiment/ttest",
        json={"group_a": [True, 2.0, 3.0], "group_b": [4.0, 5.0]},
    )
    assert res3.status_code == 422


def test_ttest_missing_fields_and_invalid_options():
    """Verify missing fields and invalid alternatives return 422."""
    # Missing group_b
    res1 = client.post("/api/experiment/ttest", json={"group_a": [1.0, 2.0]})
    assert res1.status_code == 422

    # Invalid alternative
    res2 = client.post(
        "/api/experiment/ttest",
        json={"group_a": [1.0, 2.0], "group_b": [3.0, 4.0], "alternative": "not_valid"},
    )
    assert res2.status_code == 422

    # Invalid alpha
    res3 = client.post(
        "/api/experiment/ttest",
        json={"group_a": [1.0, 2.0], "group_b": [3.0, 4.0], "alpha": 2.0},
    )
    assert res3.status_code == 422


# --- Chi-Square Endpoint Tests ---

def test_chi_square_valid_2x2():
    """Verify chi-square endpoint with valid 2x2 table."""
    payload = {
        "table": [[50, 30], [20, 40]],
        "alpha": 0.05,
    }
    response = client.post("/api/experiment/chi-square", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["degrees_of_freedom"] == 1
    assert data["chi_square_statistic"] == pytest.approx(10.52917, rel=1e-4)
    assert data["p_value"] == pytest.approx(0.001175, rel=1e-3)
    assert data["statistically_significant"] is True
    assert data["warning"] is None
    assert data["expected_frequencies"][0] == pytest.approx([40.0, 40.0])
    assert data["expected_frequencies"][1] == pytest.approx([30.0, 30.0])


def test_chi_square_valid_larger_table():
    """Verify chi-square endpoint with a 2x3 contingency table."""
    payload = {
        "table": [[20, 30, 25], [40, 15, 35]],
    }
    response = client.post("/api/experiment/chi-square", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["degrees_of_freedom"] == 2
    assert len(data["expected_frequencies"]) == 2
    assert len(data["expected_frequencies"][0]) == 3


def test_chi_square_zero_count_cell_warning():
    """Verify chi-square endpoint warning on small expected cell counts."""
    payload = {
        "table": [[0, 20], [10, 30]],
    }
    response = client.post("/api/experiment/chi-square", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["warning"] is not None
    assert "less than 5" in data["warning"]


def test_chi_square_invalid_dimensions():
    """Verify rejection of 1D, single-row, or single-col tables."""
    # 1D list
    res1 = client.post("/api/experiment/chi-square", json={"table": [10, 20]})
    assert res1.status_code == 422

    # Single row
    res2 = client.post("/api/experiment/chi-square", json={"table": [[10, 20]]})
    assert res2.status_code == 422

    # Single column
    res3 = client.post("/api/experiment/chi-square", json={"table": [[10], [20]]})
    assert res3.status_code == 422

    # Jagged table
    res4 = client.post(
        "/api/experiment/chi-square",
        json={"table": [[10, 20], [10, 20, 30]]},
    )
    assert res4.status_code == 422


def test_chi_square_negative_counts_rejected():
    """Verify negative counts return 422."""
    payload = {"table": [[10, -5], [20, 30]]}
    response = client.post("/api/experiment/chi-square", json=payload)
    assert response.status_code == 422


def test_chi_square_all_zero_and_empty_marginals_rejected():
    """Verify all-zero tables and empty marginal rows/cols return 422."""
    # All zeros
    res1 = client.post("/api/experiment/chi-square", json={"table": [[0, 0], [0, 0]]})
    assert res1.status_code == 422

    # Row sum zero
    res2 = client.post("/api/experiment/chi-square", json={"table": [[0, 0], [10, 20]]})
    assert res2.status_code == 422

    # Column sum zero
    res3 = client.post("/api/experiment/chi-square", json={"table": [[0, 10], [0, 20]]})
    assert res3.status_code == 422


def test_chi_square_invalid_types_and_missing_fields():
    """Verify invalid cell types and missing fields return 422."""
    # String in cell
    res1 = client.post(
        "/api/experiment/chi-square",
        json={"table": [[10, "many"], [20, 30]]},
    )
    assert res1.status_code == 422

    # Boolean in cell
    res2 = client.post(
        "/api/experiment/chi-square",
        json={"table": [[True, 10], [20, 30]]},
    )
    assert res2.status_code == 422

    # None in cell
    res3 = client.post(
        "/api/experiment/chi-square",
        json={"table": [[None, 10], [20, 30]]},
    )
    assert res3.status_code == 422

    # Missing table field
    res4 = client.post("/api/experiment/chi-square", json={})
    assert res4.status_code == 422


