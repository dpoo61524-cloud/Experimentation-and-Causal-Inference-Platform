"""Automated tests for FastAPI endpoints in the Experimentation Platform."""

import math
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


# --- Dashboard and Static File Endpoints ---

def test_dashboard_endpoint_serves_html():
    """Verify that GET /dashboard serves the HTML dashboard with title and 6 methods."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "").lower()

    html = response.text
    # Verify main dashboard heading
    assert "Causal Analysis Dashboard" in html

    # Verify all six causal analysis method names
    assert "Unadjusted ATE" in html
    assert "Regression-Adjusted ATE" in html
    assert "Propensity Scores" in html
    assert "IPTW" in html
    assert "Difference-in-Differences" in html
    assert "Sensitivity Analysis" in html


def test_dashboard_static_css_asset():
    """Verify that dashboard.css is served from /static/css/dashboard.css."""
    response = client.get("/static/css/dashboard.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "").lower()
    css = response.text
    assert "--accent-navy" in css
    assert "dashboard-workspace" in css


def test_dashboard_preserves_root_and_health_json():
    """Verify that GET / and GET /health remain intact alongside GET /dashboard."""
    root_res = client.get("/")
    assert root_res.status_code == 200
    assert root_res.headers.get("content-type", "").startswith("application/json")
    root_data = root_res.json()
    assert "name" in root_data
    assert "docs_url" in root_data
    assert "Experimentation and Causal Inference Platform API" in root_data["name"]

    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json() == {"status": "healthy"}


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


# --- Sequential Testing Endpoint Tests ---

def test_sequential_endpoint_early_efficacy_stopping():
    """Verify sequential endpoint detects early efficacy boundary crossing at interim look."""
    payload = {
        "checkpoint": 1,
        "total_checkpoints": 5,
        "conversions_a": 10,
        "visitors_a": 200,
        "conversions_b": 80,
        "visitors_b": 200,
        "max_visitors_per_group": 1000,
        "alpha": 0.05,
    }
    response = client.post("/api/experiment/sequential", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["checkpoint"] == 1
    assert data["total_checkpoints"] == 5
    assert data["planned_information_fraction"] == pytest.approx(0.20)
    assert data["observed_information_fraction"] == pytest.approx(0.20)
    assert data["schedule_compliant"] is True
    assert data["efficacy_boundary_crossed"] is True
    assert data["maximum_sample_size_reached"] is False
    assert data["decision"] == "stop_for_efficacy"
    assert data["efficacy_boundary_z"] == pytest.approx(4.562, abs=0.01)
    assert "crossed the O'Brien-Fleming efficacy boundary" in data["explanation"]


def test_sequential_endpoint_continuation():
    """Verify sequential endpoint returns continue when boundary is not crossed."""
    payload = {
        "checkpoint": 2,
        "total_checkpoints": 5,
        "conversions_a": 40,
        "visitors_a": 400,
        "conversions_b": 46,
        "visitors_b": 400,
        "max_visitors_per_group": 1000,
        "alpha": 0.05,
    }
    response = client.post("/api/experiment/sequential", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["checkpoint"] == 2
    assert data["efficacy_boundary_crossed"] is False
    assert data["maximum_sample_size_reached"] is False
    assert data["decision"] == "continue"
    assert "Continue experiment to next scheduled checkpoint" in data["explanation"]


def test_sequential_endpoint_final_look_efficacy():
    """Verify final look reaches maximum sample size and detects statistical significance."""
    payload = {
        "checkpoint": 5,
        "total_checkpoints": 5,
        "conversions_a": 100,
        "visitors_a": 1000,
        "conversions_b": 140,
        "visitors_b": 1000,
        "max_visitors_per_group": 1000,
        "alpha": 0.05,
    }
    response = client.post("/api/experiment/sequential", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["checkpoint"] == 5
    assert data["maximum_sample_size_reached"] is True
    assert data["efficacy_boundary_crossed"] is True
    assert data["decision"] == "stop_max_sample_size_efficacy_achieved"
    assert "statistically significant at overall alpha=0.05" in data["explanation"]


def test_sequential_endpoint_final_look_inconclusive():
    """Verify final look non-crossing is reported as inconclusive evidence."""
    payload = {
        "checkpoint": 5,
        "total_checkpoints": 5,
        "conversions_a": 100,
        "visitors_a": 1000,
        "conversions_b": 105,
        "visitors_b": 1000,
        "max_visitors_per_group": 1000,
        "alpha": 0.05,
    }
    response = client.post("/api/experiment/sequential", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["checkpoint"] == 5
    assert data["maximum_sample_size_reached"] is True
    assert data["efficacy_boundary_crossed"] is False
    assert data["decision"] == "stop_max_sample_size_inconclusive"
    assert "inconclusive evidence, NOT proof that the treatment has no effect" in data["explanation"]


def test_sequential_endpoint_futility_recommendation():
    """Verify non-binding futility recommendation in endpoint response."""
    payload = {
        "checkpoint": 3,
        "total_checkpoints": 5,
        "conversions_a": 60,
        "visitors_a": 600,
        "conversions_b": 50,
        "visitors_b": 600,
        "max_visitors_per_group": 1000,
        "alpha": 0.05,
        "evaluate_futility": True,
    }
    response = client.post("/api/experiment/sequential", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["futility_boundary_crossed"] is True
    assert data["futility_recommendation"] == "stop_for_futility"
    assert data["decision"] == "continue"
    assert "does NOT prove absence of an effect" in data["explanation"]


def test_sequential_endpoint_schedule_deviation_rejected():
    """Verify schedule deviation beyond tolerance is rejected with 400 Bad Request."""
    payload = {
        "checkpoint": 1,
        "total_checkpoints": 5,
        "conversions_a": 90,
        "visitors_a": 900,
        "conversions_b": 90,
        "visitors_b": 900,
        "max_visitors_per_group": 1000,
        "schedule_tolerance": 0.15,
    }
    response = client.post("/api/experiment/sequential", json=payload)
    assert response.status_code == 400
    assert "does not meet the design's schedule assumptions" in response.json()["detail"]


def test_sequential_endpoint_validation_errors():
    """Verify Pydantic request validation returns 422 for invalid parameters."""
    base_valid = {
        "checkpoint": 1,
        "total_checkpoints": 5,
        "conversions_a": 20,
        "visitors_a": 200,
        "conversions_b": 25,
        "visitors_b": 200,
        "max_visitors_per_group": 1000,
        "alpha": 0.05,
    }

    # Conversions > visitors
    res1 = client.post("/api/experiment/sequential", json={**base_valid, "conversions_a": 250})
    assert res1.status_code == 422

    # Visitors > max_visitors_per_group
    res2 = client.post("/api/experiment/sequential", json={**base_valid, "visitors_a": 1500})
    assert res2.status_code == 422

    # Checkpoint > total_checkpoints
    res3 = client.post("/api/experiment/sequential", json={**base_valid, "checkpoint": 6})
    assert res3.status_code == 422

    # Negative conversions
    res4 = client.post("/api/experiment/sequential", json={**base_valid, "conversions_a": -5})
    assert res4.status_code == 422

    # Boolean count
    res5 = client.post("/api/experiment/sequential", json={**base_valid, "conversions_a": True})
    assert res5.status_code == 422

    # Unsupported total_checkpoints (e.g., 6)
    res6 = client.post("/api/experiment/sequential", json={**base_valid, "total_checkpoints": 6})
    assert res6.status_code == 422

    # Missing mandatory field
    res7 = client.post("/api/experiment/sequential", json={"checkpoint": 1})
    assert res7.status_code == 422


# --- Bayesian A/B Testing Endpoint ---

def test_bayesian_endpoint_valid_request():
    """Verify that a valid request to /api/experiment/bayesian returns HTTP 200 and expected fields."""
    payload = {
        "conversions_a": 45,
        "visitors_a": 500,
        "conversions_b": 60,
        "visitors_b": 500,
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "credible_interval": 0.95,
        "minimum_effect": 0.0,
    }
    response = client.post("/api/experiment/bayesian", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify presence of all primary fields
    expected_keys = [
        "visitors_a",
        "conversions_a",
        "conversion_rate_a",
        "observed_conversion_rate_a",
        "visitors_b",
        "conversions_b",
        "conversion_rate_b",
        "observed_conversion_rate_b",
        "observed_difference",
        "observed_absolute_difference",
        "prior_alpha",
        "prior_beta",
        "posterior_alpha_a",
        "posterior_beta_a",
        "posterior_alpha_b",
        "posterior_beta_b",
        "posterior_mean_a",
        "posterior_mean_b",
        "posterior_mean_difference",
        "credible_interval_a",
        "credible_interval_b",
        "credible_interval_difference",
        "probability_b_greater_than_a",
        "probability_b_exceeds_a_by_minimum_effect",
        "credible_interval_level",
        "credible_interval",
        "minimum_effect",
        "simulation_method",
        "num_samples",
    ]
    for key in expected_keys:
        assert key in data, f"Missing key in response: {key}"

    # Verify analytical posterior parameters and means
    assert data["posterior_alpha_a"] == 46.0
    assert data["posterior_beta_a"] == 456.0
    assert data["posterior_alpha_b"] == 61.0
    assert data["posterior_beta_b"] == 441.0
    assert data["posterior_mean_a"] == pytest.approx(46.0 / 502.0)
    assert data["posterior_mean_b"] == pytest.approx(61.0 / 502.0)
    assert data["posterior_mean_difference"] == pytest.approx((61.0 - 46.0) / 502.0)

    # Verify probability and intervals
    assert 0.0 <= data["probability_b_greater_than_a"] <= 1.0
    assert data["probability_b_greater_than_a"] > 0.85
    assert data["credible_interval_a"][0] < data["credible_interval_a"][1]
    assert data["credible_interval_b"][0] < data["credible_interval_b"][1]
    assert data["credible_interval_difference"][0] < data["credible_interval_difference"][1]


def test_bayesian_endpoint_nested_format_aliases():
    """Verify that nested group_a and group_b objects are supported via aliases."""
    payload = {
        "group_a": {"conversions": 45, "visitors": 500},
        "group_b": {"conversions": 60, "visitors": 500},
    }
    response = client.post("/api/experiment/bayesian", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["posterior_alpha_a"] == 46.0
    assert data["posterior_alpha_b"] == 61.0


def test_bayesian_endpoint_custom_priors():
    """Verify that custom informative priors (e.g. Beta(5, 95)) pass through correctly."""
    payload = {
        "conversions_a": 10,
        "visitors_a": 200,
        "conversions_b": 25,
        "visitors_b": 200,
        "prior_alpha": 5.0,
        "prior_beta": 95.0,
    }
    response = client.post("/api/experiment/bayesian", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prior_alpha"] == 5.0
    assert data["prior_beta"] == 95.0
    assert data["posterior_alpha_a"] == 15.0
    assert data["posterior_beta_a"] == 285.0
    assert data["posterior_mean_a"] == pytest.approx(15.0 / 300.0)


def test_bayesian_endpoint_custom_credible_interval():
    """Verify that configuring a custom credible interval level (e.g. 0.90) is supported."""
    payload = {
        "conversions_a": 30,
        "visitors_a": 300,
        "conversions_b": 45,
        "visitors_b": 300,
        "credible_interval": 0.90,
    }
    response = client.post("/api/experiment/bayesian", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["credible_interval_level"] == 0.90
    assert 0.0 <= data["credible_interval_a"][0] < data["credible_interval_a"][1] <= 1.0


def test_bayesian_endpoint_positive_minimum_effect():
    """Verify that specifying a positive minimum_effect returns a valid probability."""
    payload = {
        "conversions_a": 50,
        "visitors_a": 1000,
        "conversions_b": 80,
        "visitors_b": 1000,
        "minimum_effect": 0.01,
    }
    response = client.post("/api/experiment/bayesian", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["minimum_effect"] == 0.01
    prob_min = data["probability_b_exceeds_a_by_minimum_effect"]
    prob_sup = data["probability_b_greater_than_a"]
    assert 0.0 <= prob_min <= 1.0
    assert prob_min <= prob_sup


def test_bayesian_endpoint_validation_errors():
    """Verify that invalid Bayesian requests are rejected with HTTP 422 Unprocessable Entity."""
    base_valid = {
        "conversions_a": 20,
        "visitors_a": 200,
        "conversions_b": 25,
        "visitors_b": 200,
    }

    # Zero visitors
    res1 = client.post("/api/experiment/bayesian", json={**base_valid, "visitors_a": 0})
    assert res1.status_code == 422

    # Negative counts
    res2 = client.post("/api/experiment/bayesian", json={**base_valid, "conversions_a": -1})
    assert res2.status_code == 422

    # Conversions > visitors
    res3 = client.post("/api/experiment/bayesian", json={**base_valid, "conversions_a": 205})
    assert res3.status_code == 422

    # Boolean counts
    res4 = client.post("/api/experiment/bayesian", json={**base_valid, "conversions_a": True})
    assert res4.status_code == 422
    res5 = client.post("/api/experiment/bayesian", json={**base_valid, "visitors_b": False})
    assert res5.status_code == 422

    # Non-positive prior parameters
    res6 = client.post("/api/experiment/bayesian", json={**base_valid, "prior_alpha": 0.0})
    assert res6.status_code == 422
    res7 = client.post("/api/experiment/bayesian", json={**base_valid, "prior_beta": -1.0})
    assert res7.status_code == 422
    res8 = client.post("/api/experiment/bayesian", json={**base_valid, "prior_alpha": True})
    assert res8.status_code == 422

    # Out of range credible interval
    res9 = client.post("/api/experiment/bayesian", json={**base_valid, "credible_interval": 0.0})
    assert res9.status_code == 422
    res10 = client.post("/api/experiment/bayesian", json={**base_valid, "credible_interval": 1.0})
    assert res10.status_code == 422
    res11 = client.post("/api/experiment/bayesian", json={**base_valid, "credible_interval": 1.05})
    assert res11.status_code == 422

    # Out of range minimum effect
    res12 = client.post("/api/experiment/bayesian", json={**base_valid, "minimum_effect": -0.05})
    assert res12.status_code == 422
    res13 = client.post("/api/experiment/bayesian", json={**base_valid, "minimum_effect": 1.05})
    assert res13.status_code == 422
    res14 = client.post("/api/experiment/bayesian", json={**base_valid, "minimum_effect": True})
    assert res14.status_code == 422

    # Missing mandatory fields
    res15 = client.post("/api/experiment/bayesian", json={"conversions_a": 10})
    assert res15.status_code == 422


# --- Causal Inference: Unadjusted ATE Endpoint Tests ---

@pytest.fixture
def example_causal_payload():
    return {
        "data": [
            {"treatment": 1, "outcome": 12},
            {"treatment": 1, "outcome": 14},
            {"treatment": 1, "outcome": 16},
            {"treatment": 0, "outcome": 6},
            {"treatment": 0, "outcome": 10},
        ],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "treated_value": 1,
        "untreated_value": 0,
        "drop_missing": False,
        "confidence_level": 0.95,
    }


def test_unadjusted_ate_example_dataset_success(example_causal_payload):
    """Verify successful unadjusted ATE calculation on the example dataset."""
    response = client.post("/api/causal/unadjusted-ate", json=example_causal_payload)
    assert response.status_code == 200
    data = response.json()

    # Core statistical expectations
    assert data["estimator"] == "unadjusted_difference_in_means"
    assert data["ate"] == pytest.approx(6.0)
    assert data["mean_outcome_treated"] == pytest.approx(14.0)
    assert data["mean_outcome_control"] == pytest.approx(8.0)
    assert data["n_total"] == 5
    assert data["n_treated"] == 3
    assert data["n_control"] == 2
    assert data["rows_dropped"] == 0

    # Welch uncertainty and confidence interval
    assert data["std_error"] == pytest.approx(2.309401, rel=1e-4)
    assert data["degrees_of_freedom"] == pytest.approx(1.684211, rel=1e-4)
    assert data["confidence_level"] == 0.95
    assert "confidence_interval" in data
    assert data["ci_lower"] is not None and data["ci_upper"] is not None

    # Confidence interval centered on the point estimate
    ci = data["confidence_interval"]
    assert len(ci) == 2
    ci_mid = (ci[0] + ci[1]) / 2.0
    assert ci_mid == pytest.approx(data["ate"], abs=1e-6)
    assert ci[0] == pytest.approx(data["ci_lower"])
    assert ci[1] == pytest.approx(data["ci_upper"])

    # Assumptions and interpretation
    assert "assumptions" in data
    assert "interpretation" in data
    assert "traceback" not in response.text.lower()


def test_unadjusted_ate_custom_confidence_level(example_causal_payload):
    """Verify unadjusted ATE with a custom confidence level (0.90 vs 0.95)."""
    payload_90 = {**example_causal_payload, "confidence_level": 0.90}
    response_90 = client.post("/api/causal/unadjusted-ate", json=payload_90)
    assert response_90.status_code == 200
    data_90 = response_90.json()

    assert data_90["confidence_level"] == 0.90
    assert data_90["ate"] == pytest.approx(6.0)

    # 90% interval must be narrower than 95% interval
    response_95 = client.post("/api/causal/unadjusted-ate", json=example_causal_payload)
    data_95 = response_95.json()

    width_90 = data_90["ci_upper"] - data_90["ci_lower"]
    width_95 = data_95["ci_upper"] - data_95["ci_lower"]
    assert width_90 < width_95


def test_unadjusted_ate_string_labels():
    """Verify that custom string treatment labels work via the API."""
    payload = {
        "data": [
            {"arm": "treated", "y": 20.0},
            {"arm": "treated", "y": 25.0},
            {"arm": "treated", "y": 30.0},
            {"arm": "control", "y": 10.0},
            {"arm": "control", "y": 15.0},
        ],
        "treatment_col": "arm",
        "outcome_col": "y",
        "treated_value": "treated",
        "untreated_value": "control",
    }
    response = client.post("/api/causal/unadjusted-ate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ate"] == pytest.approx(12.5)
    assert data["mean_outcome_treated"] == pytest.approx(25.0)
    assert data["mean_outcome_control"] == pytest.approx(12.5)


@pytest.mark.parametrize(
    "missing_field",
    ["data", "treatment_col", "outcome_col"]
)
def test_unadjusted_ate_missing_required_fields(example_causal_payload, missing_field):
    """Verify that omitting required request fields returns HTTP 422."""
    payload = {k: v for k, v in example_causal_payload.items() if k != missing_field}
    response = client.post("/api/causal/unadjusted-ate", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "traceback" not in response.text.lower()


@pytest.mark.parametrize(
    "invalid_conf",
    [0.0, 1.0, -0.05, 1.5, True, False, "invalid"]
)
def test_unadjusted_ate_invalid_confidence_level(example_causal_payload, invalid_conf):
    """Verify that invalid confidence levels are rejected with HTTP 422."""
    payload = {**example_causal_payload, "confidence_level": invalid_conf}
    response = client.post("/api/causal/unadjusted-ate", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "traceback" not in response.text.lower()


def test_unadjusted_ate_missing_treatment_column(example_causal_payload):
    """Verify rejection when the specified treatment column is missing from data."""
    payload = {**example_causal_payload, "treatment_col": "non_existent_column"}
    response = client.post("/api/causal/unadjusted-ate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "non_existent_column" in str(data["detail"])


def test_unadjusted_ate_missing_outcome_column(example_causal_payload):
    """Verify rejection when the specified outcome column is missing from data."""
    payload = {**example_causal_payload, "outcome_col": "non_existent_column"}
    response = client.post("/api/causal/unadjusted-ate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "non_existent_column" in str(data["detail"])


def test_unadjusted_ate_invalid_treatment_labels():
    """Verify rejection when treatment column has non-binary categories or invalid labels."""
    payload = {
        "data": [
            {"treatment": 0, "outcome": 10},
            {"treatment": 1, "outcome": 15},
            {"treatment": 2, "outcome": 20},
        ],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
    }
    response = client.post("/api/causal/unadjusted-ate", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_unadjusted_ate_nonnumeric_outcomes():
    """Verify rejection when outcome column contains non-numeric strings or booleans."""
    payload_str = {
        "data": [
            {"treatment": 1, "outcome": "high"},
            {"treatment": 1, "outcome": 14},
            {"treatment": 0, "outcome": 6},
            {"treatment": 0, "outcome": 10},
        ],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
    }
    res_str = client.post("/api/causal/unadjusted-ate", json=payload_str)
    assert res_str.status_code == 422
    assert "detail" in res_str.json()

    payload_bool = {
        "data": [
            {"treatment": 1, "outcome": True},
            {"treatment": 1, "outcome": False},
            {"treatment": 0, "outcome": True},
            {"treatment": 0, "outcome": False},
        ],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
    }
    res_bool = client.post("/api/causal/unadjusted-ate", json=payload_bool)
    assert res_bool.status_code == 422
    assert "detail" in res_bool.json()


def test_unadjusted_ate_missing_outcomes_default_rejected():
    """Verify rejection when missing outcome values are present with drop_missing=False."""
    payload = {
        "data": [
            {"treatment": 1, "outcome": 12},
            {"treatment": 1, "outcome": None},
            {"treatment": 0, "outcome": 6},
            {"treatment": 0, "outcome": 10},
        ],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "drop_missing": False,
    }
    response = client.post("/api/causal/unadjusted-ate", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "missing" in str(response.json()["detail"]).lower()


def test_unadjusted_ate_explicit_drop_missing():
    """Verify that drop_missing=True drops rows and reports rows_dropped accurately."""
    payload = {
        "data": [
            {"treatment": 1, "outcome": 12},
            {"treatment": 1, "outcome": 14},
            {"treatment": 1, "outcome": 16},
            {"treatment": 0, "outcome": 6},
            {"treatment": 0, "outcome": 10},
            {"treatment": 1, "outcome": None},  # should be dropped
            {"treatment": None, "outcome": 8},   # should be dropped
        ],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "drop_missing": True,
    }
    response = client.post("/api/causal/unadjusted-ate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["rows_dropped"] == 2
    assert data["n_total"] == 5
    assert data["n_treated"] == 3
    assert data["n_control"] == 2
    assert data["ate"] == pytest.approx(6.0)


@pytest.mark.parametrize(
    "malformed_data",
    [
        [],                    # empty list
        "not_a_list",          # string
        {"a": 1},              # dict instead of list
        [1, 2, 3],             # list of non-dicts
    ]
)
def test_unadjusted_ate_empty_or_malformed_datasets(malformed_data):
    """Verify that empty or malformed dataset payloads are rejected with HTTP 422."""
    payload = {
        "data": malformed_data,
        "treatment_col": "treatment",
        "outcome_col": "outcome",
    }
    response = client.post("/api/causal/unadjusted-ate", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_unadjusted_ate_single_group_or_insufficient_sample():
    """Verify rejection when only one group is present or group size is insufficient."""
    # Single treatment group
    payload_single = {
        "data": [
            {"treatment": 1, "outcome": 10},
            {"treatment": 1, "outcome": 12},
            {"treatment": 1, "outcome": 14},
        ],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
    }
    res1 = client.post("/api/causal/unadjusted-ate", json=payload_single)
    assert res1.status_code == 422

    # n=1 in control group (insufficient for Welch variance calculation)
    payload_small = {
        "data": [
            {"treatment": 1, "outcome": 10},
            {"treatment": 1, "outcome": 12},
            {"treatment": 0, "outcome": 8},
        ],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
    }
    res2 = client.post("/api/causal/unadjusted-ate", json=payload_small)
    assert res2.status_code == 422


# --- Regression-Adjusted ATE Fixtures & Tests ---

@pytest.fixture
def example_regression_causal_payload():
    """Deterministic payload generated from: outcome = 2 + 5 * treatment + 3 * age."""
    t_vals = [1, 1, 1, 1, 0, 0, 0, 0]
    a_vals = [20, 25, 30, 35, 22, 26, 32, 38]
    data = [
        {"treatment": t, "age": a, "outcome": 2.0 + 5.0 * t + 3.0 * a}
        for t, a in zip(t_vals, a_vals)
    ]
    return {
        "data": data,
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": ["age"],
        "treated_value": 1,
        "untreated_value": 0,
        "drop_missing": False,
    }


def test_regression_adjusted_ate_success(example_regression_causal_payload):
    """Verify HTTP 200, correct response schema, recovered known ATE=5.0, counterfactual means, and raw difference."""
    response = client.post("/api/causal/regression-adjusted-ate", json=example_regression_causal_payload)
    assert response.status_code == 200
    data = response.json()

    # Core identification & estimates
    assert data["estimator"] == "regression_adjusted_ate"
    assert data["adjusted_ate"] == pytest.approx(5.0)
    assert data["ate"] == pytest.approx(5.0)
    assert data["unadjusted_ate"] == pytest.approx(-1.0)
    assert data["raw_difference"] == pytest.approx(-1.0)
    assert data["difference_in_means"] == pytest.approx(-1.0)

    # Counterfactual predictions (G-computation)
    assert data["mean_predicted_treated"] == pytest.approx(92.5)
    assert data["mean_predicted_control"] == pytest.approx(87.5)
    assert data["mean_predicted_treated"] - data["mean_predicted_control"] == pytest.approx(5.0)

    # Sample statistics
    assert data["mean_outcome_treated"] == pytest.approx(89.5)
    assert data["mean_outcome_control"] == pytest.approx(90.5)
    assert data["n_total"] == 8
    assert data["n_treated"] == 4
    assert data["n_control"] == 4
    assert data["n_untreated"] == 4

    # Model and feature metadata
    assert data["r_squared"] == pytest.approx(1.0)
    assert "const" in data["model_features"]
    assert "treatment_indicator" in data["model_features"]
    assert "age" in data["model_features"]
    assert data["confounder_cols"] == ["age"]
    assert data["numeric_confounders"] == ["age"]
    assert data["categorical_confounders"] == []
    assert data["rows_dropped"] == 0

    # Interpretation & assumptions
    assert "interpretation" in data
    assert "assumptions" in data
    assert "G-computation" in data["assumptions"]
    assert "Conditional Exchangeability" in data["assumptions"]


def test_regression_adjusted_ate_categorical_confounders():
    """Verify regression adjustment with both continuous numeric and categorical confounders."""
    # Outcome equation: Y = 5.0 + 4.0 * T + 2.0 * age + 10.0 * (city == 'NYC')
    records = [
        {"grp": 1, "age": 20, "city": "NYC", "outcome": 5.0 + 4.0 + 40.0 + 10.0},
        {"grp": 1, "age": 30, "city": "LA",  "outcome": 5.0 + 4.0 + 60.0 + 0.0},
        {"grp": 1, "age": 40, "city": "NYC", "outcome": 5.0 + 4.0 + 80.0 + 10.0},
        {"grp": 1, "age": 25, "city": "CHI", "outcome": 5.0 + 4.0 + 50.0 + 0.0},
        {"grp": 0, "age": 20, "city": "NYC", "outcome": 5.0 + 0.0 + 40.0 + 10.0},
        {"grp": 0, "age": 30, "city": "LA",  "outcome": 5.0 + 0.0 + 60.0 + 0.0},
        {"grp": 0, "age": 40, "city": "NYC", "outcome": 5.0 + 0.0 + 80.0 + 10.0},
        {"grp": 0, "age": 25, "city": "CHI", "outcome": 5.0 + 0.0 + 50.0 + 0.0},
    ]
    payload = {
        "data": records,
        "treatment_col": "grp",
        "outcome_col": "outcome",
        "confounder_cols": ["age", "city"],
    }
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["adjusted_ate"] == pytest.approx(4.0)
    assert data["numeric_confounders"] == ["age"]
    assert data["categorical_confounders"] == ["city"]
    assert any("city" in col for col in data["model_features"])
    assert data["n_total"] == 8
    assert data["n_treated"] == 4
    assert data["n_control"] == 4


@pytest.mark.parametrize(
    "missing_field",
    ["data", "treatment_col", "outcome_col", "confounder_cols"],
)
def test_regression_adjusted_ate_missing_required_fields(example_regression_causal_payload, missing_field):
    """Verify that omitting required request fields returns HTTP 422."""
    payload = {k: v for k, v in example_regression_causal_payload.items() if k != missing_field}
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "traceback" not in response.text.lower()


@pytest.mark.parametrize(
    "empty_data",
    [[], "not_a_list", {"a": 1}, [1, 2, 3]],
)
def test_regression_adjusted_ate_empty_or_malformed_datasets(empty_data):
    """Verify that empty or malformed dataset payloads return HTTP 422."""
    payload = {
        "data": empty_data,
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": ["age"],
    }
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "invalid_confounders",
    [[], [""], ["   "], "not_a_list", [123]],
)
def test_regression_adjusted_ate_invalid_confounder_list(example_regression_causal_payload, invalid_confounders):
    """Verify that empty or invalid confounder lists produce HTTP 422."""
    payload = {**example_regression_causal_payload, "confounder_cols": invalid_confounders}
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "missing_col,col_name",
    [
        ("treatment_col", "ghost_treatment"),
        ("outcome_col", "ghost_outcome"),
    ],
)
def test_regression_adjusted_ate_missing_columns(example_regression_causal_payload, missing_col, col_name):
    """Verify descriptive HTTP 422 response when treatment or outcome columns are missing from data."""
    payload = {**example_regression_causal_payload, missing_col: col_name}
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 422
    assert col_name in str(response.json()["detail"])


def test_regression_adjusted_ate_missing_confounder_column(example_regression_causal_payload):
    """Verify descriptive HTTP 422 when a specified confounder column does not exist in data."""
    payload = {**example_regression_causal_payload, "confounder_cols": ["non_existent_confounder"]}
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 422
    assert "non_existent_confounder" in str(response.json()["detail"])


@pytest.mark.parametrize(
    "bad_config",
    [
        # treatment == outcome
        {"treatment_col": "x", "outcome_col": "x", "confounder_cols": ["y"]},
        # treatment in confounders
        {"treatment_col": "t", "outcome_col": "y", "confounder_cols": ["t"]},
        # outcome in confounders
        {"treatment_col": "t", "outcome_col": "y", "confounder_cols": ["y"]},
        # duplicate confounders
        {"treatment_col": "t", "outcome_col": "y", "confounder_cols": ["x", "x"]},
    ],
)
def test_regression_adjusted_ate_duplicate_or_conflicting_columns(bad_config):
    """Verify rejection when column roles conflict or are duplicated."""
    payload = {
        "data": [{"t": 1, "y": 10, "x": 5}, {"t": 0, "y": 8, "x": 3}],
        **bad_config,
    }
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_regression_adjusted_ate_invalid_treatment_labels():
    """Verify rejection when treatment column has non-binary labels."""
    payload = {
        "data": [
            {"treatment": 0, "outcome": 10, "age": 20},
            {"treatment": 1, "outcome": 15, "age": 25},
            {"treatment": 2, "outcome": 20, "age": 30},
        ],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": ["age"],
    }
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 422


def test_regression_adjusted_ate_missing_values_rejected_by_default(example_regression_causal_payload):
    """Verify that missing values in analysis columns are rejected with HTTP 422 when drop_missing=False."""
    bad_data = [dict(row) for row in example_regression_causal_payload["data"]]
    bad_data[0]["age"] = None
    payload = {**example_regression_causal_payload, "data": bad_data, "drop_missing": False}
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 422
    assert "missing" in str(response.json()["detail"]).lower()


def test_regression_adjusted_ate_explicit_drop_missing(example_regression_causal_payload):
    """Verify that drop_missing=True drops rows with missing values and returns accurate rows_dropped."""
    dirty_data = [dict(row) for row in example_regression_causal_payload["data"]]
    # Add two corrupted rows
    dirty_data.append({"treatment": 1, "age": None, "outcome": 100})
    dirty_data.append({"treatment": None, "age": 25, "outcome": 80})

    payload = {**example_regression_causal_payload, "data": dirty_data, "drop_missing": True}
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["rows_dropped"] == 2
    assert data["n_total"] == 8
    assert data["adjusted_ate"] == pytest.approx(5.0)


def test_regression_adjusted_ate_constant_confounder_rejection():
    """Verify that a confounder with zero variance is rejected with HTTP 422."""
    data = [
        {"treatment": 1, "age": 30, "outcome": 15},
        {"treatment": 1, "age": 30, "outcome": 18},
        {"treatment": 0, "age": 30, "outcome": 10},
        {"treatment": 0, "age": 30, "outcome": 12},
    ]
    payload = {
        "data": data,
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": ["age"],
    }
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 422
    assert "zero variance" in str(response.json()["detail"]).lower() or "constant" in str(response.json()["detail"]).lower()


def test_regression_adjusted_ate_collinear_rejection(monkeypatch, example_regression_causal_payload):
    """Verify that severe collinearity or non-estimable model failure is caught and returned as HTTP 422."""
    def mock_estimator(*args, **kwargs):
        raise ValueError(
            "Regression model fitting resulted in NaN coefficients (likely due to severe collinearity or rank deficiency)."
        )

    monkeypatch.setattr("api.estimate_regression_adjusted_ate", mock_estimator)
    response = client.post("/api/causal/regression-adjusted-ate", json=example_regression_causal_payload)
    assert response.status_code == 422
    assert "collinearity" in response.json()["detail"].lower()
    assert "traceback" not in response.text.lower()


def test_regression_adjusted_ate_insufficient_observations():
    """Verify that datasets with fewer observations than model features are rejected."""
    # 2 observations, but model has 3 features (intercept, treatment, age)
    data = [
        {"treatment": 1, "age": 20, "outcome": 15},
        {"treatment": 0, "age": 25, "outcome": 10},
    ]
    payload = {
        "data": data,
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": ["age"],
    }
    response = client.post("/api/causal/regression-adjusted-ate", json=payload)
    assert response.status_code == 422
    assert "insufficient observations" in str(response.json()["detail"]).lower()


def test_causal_unadjusted_and_regression_adjusted_coexist(example_causal_payload, example_regression_causal_payload):
    """Verify that both causal endpoints work seamlessly side by side."""
    res_unadj = client.post("/api/causal/unadjusted-ate", json=example_causal_payload)
    assert res_unadj.status_code == 200
    assert res_unadj.json()["estimator"] == "unadjusted_difference_in_means"

    res_reg = client.post("/api/causal/regression-adjusted-ate", json=example_regression_causal_payload)
    assert res_reg.status_code == 200
    assert res_reg.json()["estimator"] == "regression_adjusted_ate"


# --- Propensity Score Fixtures & Tests ---

@pytest.fixture
def example_propensity_payload():
    """Deterministic payload with continuous numeric confounders."""
    data = [
        {"treatment": 1, "age": 25, "income": 45000.0},
        {"treatment": 1, "age": 30, "income": 52000.0},
        {"treatment": 1, "age": 35, "income": 48000.0},
        {"treatment": 1, "age": 40, "income": 60000.0},
        {"treatment": 0, "age": 24, "income": 42000.0},
        {"treatment": 0, "age": 29, "income": 50000.0},
        {"treatment": 0, "age": 36, "income": 46000.0},
        {"treatment": 0, "age": 42, "income": 58000.0},
    ]
    return {
        "data": data,
        "treatment_col": "treatment",
        "confounder_cols": ["age", "income"],
        "treated_value": 1,
        "untreated_value": 0,
        "drop_missing": False,
        "near_zero_threshold": 0.05,
        "near_one_threshold": 0.95,
    }


def test_propensity_scores_success(example_propensity_payload):
    """Verify HTTP 200, schema structure, observation alignment, scores in [0, 1], and diagnostics."""
    response = client.post("/api/causal/propensity-scores", json=example_propensity_payload)
    assert response.status_code == 200
    data = response.json()

    # Core metadata
    assert data["estimator"] == "logistic_propensity_score"
    assert data["n_total"] == 8
    assert data["n_treated"] == 4
    assert data["n_control"] == 4
    assert data["treatment_col"] == "treatment"
    assert data["confounder_cols"] == ["age", "income"]
    assert data["numeric_confounders"] == ["age", "income"]
    assert data["categorical_confounders"] == []
    assert data["rows_dropped"] == 0

    # Propensity score properties
    scores = data["propensity_scores"]
    assert len(scores) == 8
    assert data["scores"] == scores
    assert all(isinstance(s, float) for s in scores)
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Observation index alignment
    indices = data["observation_indices"]
    assert len(indices) == 8
    assert indices == [0, 1, 2, 3, 4, 5, 6, 7]
    scores_by_idx = data["scores_by_index"]
    for i in range(8):
        assert str(i) in scores_by_idx
        assert scores_by_idx[str(i)] == pytest.approx(scores[i])

    # Model and feature metadata
    assert "const" in data["model_features"]
    assert "age" in data["model_features"]
    assert "income" in data["model_features"]
    assert isinstance(data["model_coefficients"], dict)
    assert data["model_metadata"]["converged"] is True

    # Overlap and covariate balance diagnostics
    overlap = data["overlap_diagnostics"]
    assert "min_score" in overlap
    assert "max_score" in overlap
    assert overlap["min_score"] <= overlap["max_score"]
    assert "has_overlap" in overlap
    assert overlap["near_zero_threshold"] == 0.05
    assert overlap["near_one_threshold"] == 0.95

    balance = data["unadjusted_covariate_balance"]
    assert "age" in balance
    assert "income" in balance
    assert balance["age"]["covariate_type"] == "numeric"
    assert "smd" in balance["age"]

    # Educational notes & interpretation
    assert "interpretation" in data
    assert "assumptions" in data
    assert "Positivity" in data["assumptions"]


def test_propensity_scores_categorical_confounders():
    """Verify that categorical confounders are one-hot encoded and properly handled."""
    records = [
        {"grp": 1, "age": 25, "region": "North"},
        {"grp": 1, "age": 35, "region": "South"},
        {"grp": 1, "age": 45, "region": "East"},
        {"grp": 1, "age": 30, "region": "North"},
        {"grp": 0, "age": 20, "region": "South"},
        {"grp": 0, "age": 22, "region": "East"},
        {"grp": 0, "age": 28, "region": "North"},
        {"grp": 0, "age": 24, "region": "South"},
    ]
    payload = {
        "data": records,
        "treatment_col": "grp",
        "confounder_cols": ["age", "region"],
    }
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["numeric_confounders"] == ["age"]
    assert data["categorical_confounders"] == ["region"]
    assert any("region_" in f for f in data["model_features"])
    assert len(data["propensity_scores"]) == 8


def test_propensity_scores_custom_treatment_labels():
    """Verify explicit custom string treatment and control labels."""
    records = [
        {"arm": "Treated", "covar": 10.0},
        {"arm": "Treated", "covar": 15.0},
        {"arm": "Treated", "covar": 20.0},
        {"arm": "Control", "covar": 5.0},
        {"arm": "Control", "covar": 8.0},
        {"arm": "Control", "covar": 12.0},
    ]
    payload = {
        "data": records,
        "treatment_col": "arm",
        "confounder_cols": ["covar"],
        "treated_value": "Treated",
        "control_value": "Control",
    }
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["treated_value"] == "Treated"
    assert data["control_value"] == "Control"
    assert data["n_treated"] == 3
    assert data["n_control"] == 3


@pytest.mark.parametrize(
    "missing_field",
    ["data", "treatment_col", "confounder_cols"],
)
def test_propensity_scores_missing_required_fields(example_propensity_payload, missing_field):
    """Verify that omitting required request fields produces HTTP 422."""
    payload = {k: v for k, v in example_propensity_payload.items() if k != missing_field}
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "traceback" not in response.text.lower()


@pytest.mark.parametrize(
    "bad_data",
    [[], "not_a_list", {"a": 1}, [1, 2, 3]],
)
def test_propensity_scores_empty_or_malformed_datasets(bad_data):
    """Verify that empty or malformed dataset payloads produce HTTP 422."""
    payload = {
        "data": bad_data,
        "treatment_col": "treatment",
        "confounder_cols": ["age"],
    }
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "bad_confounders",
    [[], [""], ["   "], "not_a_list", [123]],
)
def test_propensity_scores_invalid_confounder_list(example_propensity_payload, bad_confounders):
    """Verify that empty or invalid confounder lists produce HTTP 422."""
    payload = {**example_propensity_payload, "confounder_cols": bad_confounders}
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422


def test_propensity_scores_duplicate_confounder_names(example_propensity_payload):
    """Verify that duplicate confounder names in the request produce HTTP 422."""
    payload = {**example_propensity_payload, "confounder_cols": ["age", "age"]}
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422
    assert "duplicate" in str(response.json()["detail"]).lower()


def test_propensity_scores_treatment_col_in_confounders(example_propensity_payload):
    """Verify that specifying the treatment column as a confounder produces HTTP 422."""
    payload = {**example_propensity_payload, "confounder_cols": ["age", "treatment"]}
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422
    assert "cannot also be the treatment column" in str(response.json()["detail"]).lower()


def test_propensity_scores_missing_columns_in_data(example_propensity_payload):
    """Verify rejection when treatment or confounder columns are absent from the dataset."""
    # Missing treatment column
    p1 = {**example_propensity_payload, "treatment_col": "nonexistent_treatment"}
    r1 = client.post("/api/causal/propensity-scores", json=p1)
    assert r1.status_code == 422
    assert "nonexistent_treatment" in str(r1.json()["detail"])

    # Missing confounder column
    p2 = {**example_propensity_payload, "confounder_cols": ["age", "nonexistent_confounder"]}
    r2 = client.post("/api/causal/propensity-scores", json=p2)
    assert r2.status_code == 422
    assert "nonexistent_confounder" in str(r2.json()["detail"])


def test_propensity_scores_invalid_treatment_values():
    """Verify rejection when the treatment column has non-binary labels."""
    records = [
        {"treatment": 0, "age": 20},
        {"treatment": 1, "age": 25},
        {"treatment": 2, "age": 30},
    ]
    payload = {
        "data": records,
        "treatment_col": "treatment",
        "confounder_cols": ["age"],
    }
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422
    assert "must be binary" in str(response.json()["detail"]).lower()


@pytest.mark.parametrize(
    "near_zero,near_one",
    [
        (0.5, 0.5),      # equal
        (0.8, 0.2),      # reversed
        (-0.1, 0.9),     # negative
        (0.1, 1.5),      # > 1
        (True, 0.9),     # boolean
        (0.1, "high"),   # non-numeric
    ],
)
def test_propensity_scores_invalid_thresholds(example_propensity_payload, near_zero, near_one):
    """Verify that equal, reversed, out-of-range, or non-finite thresholds produce HTTP 422."""
    payload = {
        **example_propensity_payload,
        "near_zero_threshold": near_zero,
        "near_one_threshold": near_one,
    }
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422


def test_propensity_scores_missing_values_rejected_by_default(example_propensity_payload):
    """Verify rejection when missing values are present with drop_missing=False."""
    bad_data = [dict(row) for row in example_propensity_payload["data"]]
    bad_data[0]["income"] = None
    payload = {**example_propensity_payload, "data": bad_data, "drop_missing": False}
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422
    assert "missing" in str(response.json()["detail"]).lower()


def test_propensity_scores_explicit_drop_missing(example_propensity_payload):
    """Verify that drop_missing=True drops rows and returns updated sample counts."""
    dirty_data = [dict(row) for row in example_propensity_payload["data"]]
    dirty_data.append({"treatment": 1, "age": None, "income": 50000.0})
    dirty_data.append({"treatment": None, "age": 30, "income": 40000.0})

    payload = {**example_propensity_payload, "data": dirty_data, "drop_missing": True}
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["rows_dropped"] == 2
    assert data["n_total"] == 8
    assert len(data["propensity_scores"]) == 8


def test_propensity_scores_all_rows_missing_after_drop():
    """Verify rejection with HTTP 422 when no usable observations remain after drop_missing=True."""
    dirty_data = [
        {"treatment": None, "age": 25},
        {"treatment": 1, "age": None},
    ]
    payload = {
        "data": dirty_data,
        "treatment_col": "treatment",
        "confounder_cols": ["age"],
        "drop_missing": True,
    }
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422


def test_propensity_scores_single_treatment_class():
    """Verify rejection when the dataset contains only one treatment class."""
    data = [
        {"treatment": 1, "age": 20},
        {"treatment": 1, "age": 25},
        {"treatment": 1, "age": 30},
    ]
    payload = {
        "data": data,
        "treatment_col": "treatment",
        "confounder_cols": ["age"],
    }
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422


def test_propensity_scores_constant_confounder():
    """Verify rejection when a confounder has zero variance (constant)."""
    data = [
        {"treatment": 1, "const_age": 25},
        {"treatment": 1, "const_age": 25},
        {"treatment": 0, "const_age": 25},
        {"treatment": 0, "const_age": 25},
    ]
    payload = {
        "data": data,
        "treatment_col": "treatment",
        "confounder_cols": ["const_age"],
    }
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422
    assert "zero variance" in str(response.json()["detail"]).lower()


def test_propensity_scores_insufficient_observations():
    """Verify rejection when observations are fewer than model parameters."""
    data = [
        {"treatment": 1, "x1": 1.0, "x2": 2.0},
        {"treatment": 0, "x1": 3.0, "x2": 4.0},
    ]
    payload = {
        "data": data,
        "treatment_col": "treatment",
        "confounder_cols": ["x1", "x2"],
    }
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 422
    assert "insufficient observations" in str(response.json()["detail"]).lower()


def test_propensity_scores_separation_warnings_captured():
    """Verify that complete separation emits warnings in model_metadata and overlap_diagnostics."""
    # Data with complete separation: t=0 when x in [1, 2, 3], t=1 when x in [10, 11, 12]
    data = [
        {"t": 0, "x": 1.0},
        {"t": 0, "x": 2.0},
        {"t": 0, "x": 3.0},
        {"t": 1, "x": 10.0},
        {"t": 1, "x": 11.0},
        {"t": 1, "x": 12.0},
    ]
    payload = {
        "data": data,
        "treatment_col": "t",
        "confounder_cols": ["x"],
    }
    response = client.post("/api/causal/propensity-scores", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Model metadata contains separation warnings
    warnings_list = data["model_metadata"]["warnings"]
    assert any("separation" in w.lower() for w in warnings_list)

    # Overlap diagnostics flags overlap violation
    overlap = data["overlap_diagnostics"]
    assert overlap["has_overlap"] is False
    assert overlap["warning"] is not None
    assert "disjoint" in overlap["warning"].lower() or "overlap" in overlap["warning"].lower()


def test_causal_all_endpoints_coexist(
    example_causal_payload,
    example_regression_causal_payload,
    example_propensity_payload,
):
    """Verify that unadjusted ATE, regression-adjusted ATE, and propensity-scores all coexist."""
    res1 = client.post("/api/causal/unadjusted-ate", json=example_causal_payload)
    assert res1.status_code == 200
    assert res1.json()["estimator"] == "unadjusted_difference_in_means"

    res2 = client.post("/api/causal/regression-adjusted-ate", json=example_regression_causal_payload)
    assert res2.status_code == 200
    assert res2.json()["estimator"] == "regression_adjusted_ate"

    res3 = client.post("/api/causal/propensity-scores", json=example_propensity_payload)
    assert res3.status_code == 200
    assert res3.json()["estimator"] == "logistic_propensity_score"


# --- Causal Inference: IPTW ATE Endpoint Tests ---

@pytest.fixture
def example_iptw_payload():
    """Deterministic payload with numeric outcome, binary treatment, and overlapping confounders."""
    data = [
        {"treatment": 1, "outcome": 10.0, "x": 1.0, "z": 20.0},
        {"treatment": 1, "outcome": 12.0, "x": 2.0, "z": 25.0},
        {"treatment": 1, "outcome": 14.0, "x": 3.0, "z": 30.0},
        {"treatment": 1, "outcome": 16.0, "x": 4.0, "z": 35.0},
        {"treatment": 0, "outcome": 5.0,  "x": 1.0, "z": 22.0},
        {"treatment": 0, "outcome": 7.0,  "x": 2.0, "z": 24.0},
        {"treatment": 0, "outcome": 9.0,  "x": 3.0, "z": 28.0},
        {"treatment": 0, "outcome": 11.0, "x": 4.0, "z": 32.0},
    ]
    return {
        "data": data,
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": ["x", "z"],
        "treated_value": 1,
        "untreated_value": 0,
        "drop_missing": False,
        "extreme_weight_threshold": 20.0,
        "truncate_weights": False,
        "max_weight": None,
    }


def test_iptw_ate_success(example_iptw_payload):
    """Verify HTTP 200, schema structure, point estimate, weighted means, effective sample sizes, and balance."""
    response = client.post("/api/causal/iptw-ate", json=example_iptw_payload)
    assert response.status_code == 200
    data = response.json()

    # Estimator metadata & sample counts
    assert data["estimator"] == "iptw_ate"
    assert data["n_total"] == 8
    assert data["n_treated"] == 4
    assert data["n_control"] == 4
    assert data["treatment_col"] == "treatment"
    assert data["outcome_col"] == "outcome"
    assert data["confounder_cols"] == ["x", "z"]
    assert data["treated_value"] == 1
    assert data["control_value"] == 0
    assert data["rows_dropped"] == 0

    # ATE & weighted means
    assert math.isfinite(data["iptw_ate"])
    assert data["ate"] == data["iptw_ate"]
    assert data["weighted_mean_outcome_treated"] > data["weighted_mean_outcome_control"]
    assert data["iptw_ate"] == pytest.approx(
        data["weighted_mean_outcome_treated"] - data["weighted_mean_outcome_control"],
        abs=1e-6,
    )
    assert data["weighted_mean_treated"] == pytest.approx(data["weighted_mean_outcome_treated"])
    assert data["weighted_mean_control"] == pytest.approx(data["weighted_mean_outcome_control"])
    assert data["unadjusted_ate"] == pytest.approx(5.0)

    # Effective sample sizes
    assert data["ess_treated"] > 0
    assert data["ess_control"] > 0
    assert data["ess_total"] == pytest.approx(data["ess_treated"] + data["ess_control"], abs=1e-6)
    assert data["effective_sample_size_treated"] == data["ess_treated"]
    assert 0.0 < data["ess_treated_fraction"] <= 1.0
    assert 0.0 < data["ess_control_fraction"] <= 1.0

    # Weights and summaries
    assert len(data["weights"]) == 8
    assert all(w > 0 for w in data["weights"])
    ws = data["weight_summary"]
    assert ws["min_weight"] <= ws["max_weight"]
    assert ws["is_truncated"] is False
    assert ws["rows_truncated"] == 0

    # Diagnostics
    assert "propensity_score_summary" in data
    assert "overlap_diagnostics" in data
    assert "covariate_balance_before" in data
    assert "covariate_balance_after" in data
    assert "balance_summary" in data
    assert "interpretation" in data
    assert "assumptions" in data
    assert "traceback" not in response.text.lower()


def test_iptw_ate_custom_labels_and_string_treatment():
    """Verify IPTW endpoint with string treatment labels and categorical/continuous confounders."""
    data = [
        {"arm": "drug", "bp_drop": 18.0, "age": 45, "tier": "low"},
        {"arm": "drug", "bp_drop": 22.0, "age": 52, "tier": "high"},
        {"arm": "drug", "bp_drop": 20.0, "age": 48, "tier": "medium"},
        {"arm": "drug", "bp_drop": 24.0, "age": 55, "tier": "high"},
        {"arm": "placebo", "bp_drop": 8.0, "age": 44, "tier": "low"},
        {"arm": "placebo", "bp_drop": 10.0, "age": 50, "tier": "medium"},
        {"arm": "placebo", "bp_drop": 7.0, "age": 46, "tier": "low"},
        {"arm": "placebo", "bp_drop": 12.0, "age": 56, "tier": "high"},
    ]
    payload = {
        "data": data,
        "treatment_col": "arm",
        "outcome_col": "bp_drop",
        "confounder_cols": ["age", "tier"],
        "treated_value": "drug",
        "control_value": "placebo",
    }
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["treated_value"] == "drug"
    assert res_data["control_value"] == "placebo"
    assert res_data["iptw_ate"] > 0
    assert "tier" in res_data["categorical_confounders"]
    assert "age" in res_data["numeric_confounders"]


def test_iptw_ate_truncation_applied(example_iptw_payload):
    """Verify that truncate_weights=True with max_weight caps extreme weights and reports rows_truncated."""
    payload = {
        **example_iptw_payload,
        "truncate_weights": True,
        "max_weight": 2.0,
    }
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 200
    data = response.json()
    ws = data["weight_summary"]
    assert ws["is_truncated"] is True
    assert ws["truncation_max_weight"] == 2.0
    assert ws["max_weight"] <= 2.0 + 1e-6
    assert all(w <= 2.0 + 1e-6 for w in data["weights"])


def test_iptw_ate_extreme_weights_flagged():
    """Verify that low extreme_weight_threshold flags extreme weights in warnings and weight_summary."""
    data = [
        {"t": 1, "y": 10.0, "x": 5.0},
        {"t": 1, "y": 12.0, "x": 5.2},
        {"t": 1, "y": 11.0, "x": 5.1},
        {"t": 1, "y": 15.0, "x": 1.0},
        {"t": 0, "y": 2.0,  "x": 1.2},
        {"t": 0, "y": 3.0,  "x": 1.1},
        {"t": 0, "y": 2.5,  "x": 1.3},
        {"t": 0, "y": 8.0,  "x": 5.0},
    ]
    payload = {
        "data": data,
        "treatment_col": "t",
        "outcome_col": "y",
        "confounder_cols": ["x"],
        "extreme_weight_threshold": 3.0,
    }
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["weight_summary"]["has_extreme_weights"] is True
    assert data["weight_summary"]["n_extreme_weights"] > 0
    assert any("extreme" in w.lower() for w in data["warnings"])


@pytest.mark.parametrize(
    "missing_field",
    ["data", "treatment_col", "outcome_col", "confounder_cols"],
)
def test_iptw_ate_missing_required_fields(example_iptw_payload, missing_field):
    """Verify that omitting required request fields returns HTTP 422."""
    payload = {k: v for k, v in example_iptw_payload.items() if k != missing_field}
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "traceback" not in response.text.lower()


@pytest.mark.parametrize(
    "bad_confounders",
    [
        [],                    # empty list
        [""],                  # empty string in list
        ["   "],               # whitespace string in list
        ["x", "x"],            # duplicate confounders
        ["x", "treatment"],    # confounder same as treatment
        ["x", "outcome"],      # confounder same as outcome
    ],
)
def test_iptw_ate_invalid_confounders(example_iptw_payload, bad_confounders):
    """Verify that empty, blank, duplicate, or conflicting confounder lists are rejected with HTTP 422."""
    payload = {**example_iptw_payload, "confounder_cols": bad_confounders}
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_iptw_ate_treatment_and_outcome_same(example_iptw_payload):
    """Verify rejection when treatment_col is identical to outcome_col."""
    payload = {**example_iptw_payload, "treatment_col": "outcome", "outcome_col": "outcome"}
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 422
    assert "cannot be the same column" in str(response.json()["detail"]).lower()


def test_iptw_ate_missing_columns_in_data(example_iptw_payload):
    """Verify rejection with HTTP 422 when specified columns are missing from observation records."""
    res1 = client.post("/api/causal/iptw-ate", json={**example_iptw_payload, "treatment_col": "non_existent"})
    assert res1.status_code == 422

    res2 = client.post("/api/causal/iptw-ate", json={**example_iptw_payload, "outcome_col": "non_existent"})
    assert res2.status_code == 422

    res3 = client.post("/api/causal/iptw-ate", json={**example_iptw_payload, "confounder_cols": ["non_existent"]})
    assert res3.status_code == 422


@pytest.mark.parametrize(
    "threshold",
    [-5.0, 0.0, -0.01, True, False, "invalid"],
)
def test_iptw_ate_invalid_extreme_weight_threshold(example_iptw_payload, threshold):
    """Verify that non-positive or non-numeric extreme_weight_threshold values are rejected with HTTP 422."""
    payload = {**example_iptw_payload, "extreme_weight_threshold": threshold}
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "max_w",
    [-1.0, 0.0, True, False, "invalid"],
)
def test_iptw_ate_invalid_max_weight(example_iptw_payload, max_w):
    """Verify that non-positive or non-numeric max_weight values are rejected with HTTP 422."""
    payload = {**example_iptw_payload, "truncate_weights": True, "max_weight": max_w}
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 422


def test_iptw_ate_missing_values_rejected_by_default(example_iptw_payload):
    """Verify that missing values in outcome, treatment, or confounders raise HTTP 422 when drop_missing=False."""
    bad_data = [dict(row) for row in example_iptw_payload["data"]]
    bad_data[0]["outcome"] = None
    payload = {**example_iptw_payload, "data": bad_data, "drop_missing": False}
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 422
    assert "missing" in str(response.json()["detail"]).lower()


def test_iptw_ate_explicit_drop_missing(example_iptw_payload):
    """Verify that drop_missing=True drops rows with missing values and returns updated counts."""
    dirty_data = [dict(row) for row in example_iptw_payload["data"]]
    dirty_data.append({"treatment": 1, "outcome": None, "x": 2.0, "z": 25.0})
    dirty_data.append({"treatment": None, "outcome": 8.0, "x": 3.0, "z": 28.0})
    dirty_data.append({"treatment": 0, "outcome": 7.0, "x": None, "z": 24.0})

    payload = {**example_iptw_payload, "data": dirty_data, "drop_missing": True}
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["rows_dropped"] == 3
    assert data["n_total"] == 8


def test_iptw_ate_single_treatment_class():
    """Verify rejection when the dataset contains only one treatment condition."""
    data = [
        {"treatment": 1, "outcome": 10.0, "x": 1.0},
        {"treatment": 1, "outcome": 12.0, "x": 2.0},
        {"treatment": 1, "outcome": 14.0, "x": 3.0},
    ]
    payload = {
        "data": data,
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": ["x"],
    }
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 422


def test_iptw_ate_zero_variance_confounder():
    """Verify rejection when a confounder has zero variance across all observations."""
    data = [
        {"t": 1, "y": 10.0, "const_c": 5.0},
        {"t": 1, "y": 12.0, "const_c": 5.0},
        {"t": 0, "y": 6.0,  "const_c": 5.0},
        {"t": 0, "y": 8.0,  "const_c": 5.0},
    ]
    payload = {
        "data": data,
        "treatment_col": "t",
        "outcome_col": "y",
        "confounder_cols": ["const_c"],
    }
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 422
    assert "zero variance" in str(response.json()["detail"]).lower()


def test_iptw_ate_nonnumeric_outcome():
    """Verify rejection when outcome column contains non-numeric string values."""
    data = [
        {"t": 1, "y": "improved", "x": 1.0},
        {"t": 1, "y": "improved", "x": 2.0},
        {"t": 0, "y": "constant", "x": 1.0},
        {"t": 0, "y": "constant", "x": 2.0},
    ]
    payload = {
        "data": data,
        "treatment_col": "t",
        "outcome_col": "y",
        "confounder_cols": ["x"],
    }
    response = client.post("/api/causal/iptw-ate", json=payload)
    assert response.status_code == 422


def test_iptw_ate_positivity_violation_error(monkeypatch, example_iptw_payload):
    """Verify that positivity violation in engine is returned as HTTP 422 without traceback."""
    def mock_iptw(*args, **kwargs):
        raise ValueError("Positivity violation: 2 observation(s) have an assigned-treatment probability of zero.")

    monkeypatch.setattr("api.estimate_iptw_ate", mock_iptw)
    response = client.post("/api/causal/iptw-ate", json=example_iptw_payload)
    assert response.status_code == 422
    assert "positivity violation" in str(response.json()["detail"]).lower()
    assert "traceback" not in response.text.lower()


def test_causal_all_four_endpoints_coexist(
    example_causal_payload,
    example_regression_causal_payload,
    example_propensity_payload,
    example_iptw_payload,
):
    """Verify that unadjusted ATE, regression-adjusted ATE, propensity scores, and IPTW ATE all coexist."""
    res1 = client.post("/api/causal/unadjusted-ate", json=example_causal_payload)
    assert res1.status_code == 200
    assert res1.json()["estimator"] == "unadjusted_difference_in_means"

    res2 = client.post("/api/causal/regression-adjusted-ate", json=example_regression_causal_payload)
    assert res2.status_code == 200
    assert res2.json()["estimator"] == "regression_adjusted_ate"

    res3 = client.post("/api/causal/propensity-scores", json=example_propensity_payload)
    assert res3.status_code == 200
    assert res3.json()["estimator"] == "logistic_propensity_score"

    res4 = client.post("/api/causal/iptw-ate", json=example_iptw_payload)
    assert res4.status_code == 200
    assert res4.json()["estimator"] == "iptw_ate"


# --- Causal Inference: Difference-in-Differences (DiD) Endpoint Tests ---

@pytest.fixture
def example_did_payload():
    """Deterministic payload:
    Treated: pre mean = 10.0, post mean = 16.0 (change = +6.0)
    Control: pre mean = 8.0, post mean = 11.0 (change = +3.0)
    Expected DiD = 6.0 - 3.0 = 3.0
    """
    data = [
        {"treatment": 1, "time": 0, "outcome": 9.0},
        {"treatment": 1, "time": 0, "outcome": 11.0},
        {"treatment": 1, "time": 1, "outcome": 15.0},
        {"treatment": 1, "time": 1, "outcome": 17.0},
        {"treatment": 0, "time": 0, "outcome": 7.0},
        {"treatment": 0, "time": 0, "outcome": 9.0},
        {"treatment": 0, "time": 1, "outcome": 10.0},
        {"treatment": 0, "time": 1, "outcome": 12.0},
    ]
    return {
        "data": data,
        "treatment_col": "treatment",
        "time_col": "time",
        "outcome_col": "outcome",
        "treated_value": 1,
        "control_value": 0,
        "pre_value": 0,
        "post_value": 1,
        "drop_missing": False,
    }


def test_did_endpoint_success(example_did_payload):
    """Verify HTTP 200, valid schema, exact DiD estimate, group means, cell counts, and interpretation."""
    response = client.post("/api/causal/difference-in-differences", json=example_did_payload)
    assert response.status_code == 200
    data = response.json()

    # Estimator metadata & sample counts
    assert data["estimator"] == "difference_in_differences"
    assert data["n_total"] == 8
    assert data["n_treated"] == 4
    assert data["n_control"] == 4
    assert data["n_untreated"] == 4
    assert data["treatment_col"] == "treatment"
    assert data["time_col"] == "time"
    assert data["outcome_col"] == "outcome"
    assert data["treated_value"] == 1
    assert data["control_value"] == 0
    assert data["pre_value"] == 0
    assert data["post_value"] == 1
    assert data["rows_dropped"] == 0

    # Cell means
    assert data["treatment_mean_pre"] == pytest.approx(10.0)
    assert data["treated_mean_pre"] == pytest.approx(10.0)
    assert data["treatment_mean_post"] == pytest.approx(16.0)
    assert data["treated_mean_post"] == pytest.approx(16.0)
    assert data["control_mean_pre"] == pytest.approx(8.0)
    assert data["control_mean_post"] == pytest.approx(11.0)

    # Time changes & DiD
    assert data["treatment_change"] == pytest.approx(6.0)
    assert data["delta_treatment"] == pytest.approx(6.0)
    assert data["delta_treated"] == pytest.approx(6.0)
    assert data["control_change"] == pytest.approx(3.0)
    assert data["delta_control"] == pytest.approx(3.0)
    assert data["did_estimate"] == pytest.approx(3.0)
    assert data["did"] == pytest.approx(3.0)
    assert data["ate"] == pytest.approx(3.0)

    # Cell observation counts
    assert data["cell_counts"] == {
        "treatment_pre": 2,
        "treatment_post": 2,
        "control_pre": 2,
        "control_post": 2,
    }
    assert data["n_treatment_pre"] == 2
    assert data["n_treatment_post"] == 2
    assert data["n_control_pre"] == 2
    assert data["n_control_post"] == 2

    # Narrative text
    assert "assumptions" in data
    assert "interpretation" in data
    assert "traceback" not in response.text.lower()


def test_did_endpoint_time_period_col_alias():
    """Verify that time_period_col is accepted as an alias for time_col."""
    payload = {
        "data": [
            {"arm": 1, "period": 0, "y": 10.0},
            {"arm": 1, "period": 1, "y": 18.0},
            {"arm": 0, "period": 0, "y": 5.0},
            {"arm": 0, "period": 1, "y": 8.0},
        ],
        "treatment_col": "arm",
        "time_period_col": "period",
        "outcome_col": "y",
    }
    response = client.post("/api/causal/difference-in-differences", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["did_estimate"] == pytest.approx(5.0)
    assert data["time_col"] == "period"
    assert data["time_period_col"] == "period"


def test_did_endpoint_custom_labels():
    """Verify that custom string treatment labels and time period labels work seamlessly."""
    payload = {
        "data": [
            {"group": "pilot", "wave": "before", "score": 40.0},
            {"group": "pilot", "wave": "after", "score": 65.0},
            {"group": "benchmark", "wave": "before", "score": 30.0},
            {"group": "benchmark", "wave": "after", "score": 40.0},
        ],
        "treatment_col": "group",
        "time_col": "wave",
        "outcome_col": "score",
        "treated_value": "pilot",
        "control_value": "benchmark",
        "pre_value": "before",
        "post_value": "after",
    }
    response = client.post("/api/causal/difference-in-differences", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["treated_value"] == "pilot"
    assert data["control_value"] == "benchmark"
    assert data["pre_value"] == "before"
    assert data["post_value"] == "after"
    assert data["treatment_change"] == pytest.approx(25.0)
    assert data["control_change"] == pytest.approx(10.0)
    assert data["did_estimate"] == pytest.approx(15.0)


def test_did_endpoint_zero_and_negative_estimates():
    """Verify true zero and negative treatment effects."""
    payload_zero = {
        "data": [
            {"t": 1, "p": 0, "y": 10.0},
            {"t": 1, "p": 1, "y": 14.0},
            {"t": 0, "p": 0, "y": 20.0},
            {"t": 0, "p": 1, "y": 24.0},
        ],
        "treatment_col": "t",
        "time_col": "p",
        "outcome_col": "y",
    }
    res_zero = client.post("/api/causal/difference-in-differences", json=payload_zero)
    assert res_zero.status_code == 200
    assert res_zero.json()["did_estimate"] == pytest.approx(0.0)

    payload_neg = {
        "data": [
            {"t": 1, "p": 0, "y": 20.0},
            {"t": 1, "p": 1, "y": 22.0},
            {"t": 0, "p": 0, "y": 20.0},
            {"t": 0, "p": 1, "y": 28.0},
        ],
        "treatment_col": "t",
        "time_col": "p",
        "outcome_col": "y",
    }
    res_neg = client.post("/api/causal/difference-in-differences", json=payload_neg)
    assert res_neg.status_code == 200
    assert res_neg.json()["did_estimate"] == pytest.approx(-6.0)


def test_did_endpoint_negative_outcomes():
    """Verify that negative outcome values are handled correctly without numerical distortion."""
    payload = {
        "data": [
            {"t": 1, "time": 0, "y": -10.0},
            {"t": 1, "time": 1, "y": -2.0},
            {"t": 0, "time": 0, "y": -15.0},
            {"t": 0, "time": 1, "y": -12.0},
        ],
        "treatment_col": "t",
        "time_col": "time",
        "outcome_col": "y",
    }
    response = client.post("/api/causal/difference-in-differences", json=payload)
    assert response.status_code == 200
    assert response.json()["did_estimate"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    "missing_field",
    ["data", "treatment_col", "outcome_col"],
)
def test_did_endpoint_missing_required_fields(example_did_payload, missing_field):
    """Verify that omitting required request fields returns HTTP 422."""
    payload = {k: v for k, v in example_did_payload.items() if k != missing_field}
    response = client.post("/api/causal/difference-in-differences", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "traceback" not in response.text.lower()


def test_did_endpoint_missing_time_col(example_did_payload):
    """Verify that omitting both time_col and time_period_col returns HTTP 422."""
    payload = {k: v for k, v in example_did_payload.items() if k not in ["time_col", "time_period_col"]}
    response = client.post("/api/causal/difference-in-differences", json=payload)
    assert response.status_code == 422
    assert "time_col" in str(response.json()["detail"]).lower()


@pytest.mark.parametrize(
    "col1,col2",
    [
        ("treatment_col", "outcome_col"),
        ("time_col", "outcome_col"),
        ("treatment_col", "time_col"),
    ],
)
def test_did_endpoint_conflicting_column_roles(example_did_payload, col1, col2):
    """Verify that assigning the same column to multiple roles returns HTTP 422."""
    payload = {**example_did_payload, col2: example_did_payload[col1]}
    response = client.post("/api/causal/difference-in-differences", json=payload)
    assert response.status_code == 422
    assert "cannot be the same column" in str(response.json()["detail"]).lower()


def test_did_endpoint_missing_columns_in_data(example_did_payload):
    """Verify rejection with HTTP 422 when specified columns are absent from data records."""
    res1 = client.post(
        "/api/causal/difference-in-differences",
        json={**example_did_payload, "treatment_col": "non_existent"},
    )
    assert res1.status_code == 422

    res2 = client.post(
        "/api/causal/difference-in-differences",
        json={**example_did_payload, "time_col": "non_existent"},
    )
    assert res2.status_code == 422

    res3 = client.post(
        "/api/causal/difference-in-differences",
        json={**example_did_payload, "outcome_col": "non_existent"},
    )
    assert res3.status_code == 422


def test_did_endpoint_single_treatment_class():
    """Verify rejection with HTTP 422 when only treated or only control units are present."""
    data = [
        {"t": 1, "time": 0, "y": 10.0},
        {"t": 1, "time": 1, "y": 15.0},
    ]
    payload = {"data": data, "treatment_col": "t", "time_col": "time", "outcome_col": "y"}
    response = client.post("/api/causal/difference-in-differences", json=payload)
    assert response.status_code == 422
    assert "requires both treated and control" in str(response.json()["detail"]).lower()


def test_did_endpoint_single_time_period():
    """Verify rejection with HTTP 422 when only pre-period or only post-period units are present."""
    data = [
        {"t": 1, "time": 0, "y": 10.0},
        {"t": 0, "time": 0, "y": 8.0},
    ]
    payload = {"data": data, "treatment_col": "t", "time_col": "time", "outcome_col": "y"}
    response = client.post("/api/causal/difference-in-differences", json=payload)
    assert response.status_code == 422
    assert "requires both pre- and post" in str(response.json()["detail"]).lower()


def test_did_endpoint_empty_cell_rejected():
    """Verify rejection with HTTP 422 when any of the 4 group-period cells has 0 observations."""
    data = [
        {"t": 1, "time": 0, "y": 10.0},
        {"t": 1, "time": 1, "y": 15.0},
        {"t": 0, "time": 0, "y": 8.0},
    ]
    payload = {"data": data, "treatment_col": "t", "time_col": "time", "outcome_col": "y"}
    response = client.post("/api/causal/difference-in-differences", json=payload)
    assert response.status_code == 422
    assert "four group-period cells" in str(response.json()["detail"]).lower()


def test_did_endpoint_missing_values_rejected_by_default(example_did_payload):
    """Verify rejection with HTTP 422 when missing values exist and drop_missing=False."""
    bad_data = [dict(row) for row in example_did_payload["data"]]
    bad_data[0]["outcome"] = None
    payload = {**example_did_payload, "data": bad_data, "drop_missing": False}
    response = client.post("/api/causal/difference-in-differences", json=payload)
    assert response.status_code == 422
    assert "missing" in str(response.json()["detail"]).lower()


def test_did_endpoint_explicit_drop_missing(example_did_payload):
    """Verify that drop_missing=True drops rows with missing values and returns updated counts."""
    dirty_data = [dict(row) for row in example_did_payload["data"]]
    dirty_data.append({"treatment": 1, "time": 0, "outcome": None})
    dirty_data.append({"treatment": None, "time": 1, "outcome": 20.0})
    dirty_data.append({"treatment": 0, "time": None, "outcome": 8.0})

    payload = {**example_did_payload, "data": dirty_data, "drop_missing": True}
    response = client.post("/api/causal/difference-in-differences", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["rows_dropped"] == 3
    assert data["n_total"] == 8
    assert data["did_estimate"] == pytest.approx(3.0)


def test_did_endpoint_nonnumeric_outcome():
    """Verify rejection with HTTP 422 when outcome values are non-numeric strings or booleans."""
    data_str = [
        {"t": 1, "time": 0, "y": "ten"},
        {"t": 1, "time": 1, "y": 15.0},
        {"t": 0, "time": 0, "y": 8.0},
        {"t": 0, "time": 1, "y": 10.0},
    ]
    res_str = client.post(
        "/api/causal/difference-in-differences",
        json={"data": data_str, "treatment_col": "t", "time_col": "time", "outcome_col": "y"},
    )
    assert res_str.status_code == 422

    data_bool = [
        {"t": 1, "time": 0, "y": True},
        {"t": 1, "time": 1, "y": False},
        {"t": 0, "time": 0, "y": True},
        {"t": 0, "time": 1, "y": False},
    ]
    res_bool = client.post(
        "/api/causal/difference-in-differences",
        json={"data": data_bool, "treatment_col": "t", "time_col": "time", "outcome_col": "y"},
    )
    assert res_bool.status_code == 422


def test_did_endpoint_identical_labels():
    """Verify rejection when treated_value == control_value or pre_value == post_value."""
    payload1 = {
        "data": [
            {"t": 1, "time": 0, "y": 10.0},
            {"t": 1, "time": 1, "y": 15.0},
            {"t": 0, "time": 0, "y": 8.0},
            {"t": 0, "time": 1, "y": 10.0},
        ],
        "treatment_col": "t",
        "time_col": "time",
        "outcome_col": "y",
        "treated_value": 1,
        "control_value": 1,
    }
    res1 = client.post("/api/causal/difference-in-differences", json=payload1)
    assert res1.status_code == 422

    payload2 = {
        "data": [
            {"t": 1, "time": 0, "y": 10.0},
            {"t": 1, "time": 1, "y": 15.0},
            {"t": 0, "time": 0, "y": 8.0},
            {"t": 0, "time": 1, "y": 10.0},
        ],
        "treatment_col": "t",
        "time_col": "time",
        "outcome_col": "y",
        "pre_value": 0,
        "post_value": 0,
    }
    res2 = client.post("/api/causal/difference-in-differences", json=payload2)
    assert res2.status_code == 422


def test_causal_all_five_endpoints_coexist(
    example_causal_payload,
    example_regression_causal_payload,
    example_propensity_payload,
    example_iptw_payload,
    example_did_payload,
):
    """Verify that all five causal inference API endpoints coexist and operate without regressions."""
    res1 = client.post("/api/causal/unadjusted-ate", json=example_causal_payload)
    assert res1.status_code == 200
    assert res1.json()["estimator"] == "unadjusted_difference_in_means"

    res2 = client.post("/api/causal/regression-adjusted-ate", json=example_regression_causal_payload)
    assert res2.status_code == 200
    assert res2.json()["estimator"] == "regression_adjusted_ate"

    res3 = client.post("/api/causal/propensity-scores", json=example_propensity_payload)
    assert res3.status_code == 200
    assert res3.json()["estimator"] == "logistic_propensity_score"

    res4 = client.post("/api/causal/iptw-ate", json=example_iptw_payload)
    assert res4.status_code == 200
    assert res4.json()["estimator"] == "iptw_ate"

    res5 = client.post("/api/causal/difference-in-differences", json=example_did_payload)
    assert res5.status_code == 200
    assert res5.json()["estimator"] == "difference_in_differences"


# ---------------------------------------------------------------------------
# Causal Sensitivity Analysis and Robustness Checks Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def example_sensitivity_payload():
    """Valid tabular dataset fixture suitable for multi-estimator comparison and sensitivity analysis."""
    records = [
        {"treatment": 1, "outcome": 20.0, "x": 2.0},
        {"treatment": 1, "outcome": 25.0, "x": 3.0},
        {"treatment": 1, "outcome": 30.0, "x": 4.0},
        {"treatment": 0, "outcome": 10.0, "x": 1.0},
        {"treatment": 0, "outcome": 15.0, "x": 2.0},
        {"treatment": 0, "outcome": 20.0, "x": 3.0},
    ]
    return {
        "data": records,
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": ["x"],
        "reference_estimator": "regression_adjusted",
    }


def test_sensitivity_endpoint_default_success(example_sensitivity_payload):
    """Verify successful sensitivity analysis with default parameters."""
    response = client.post("/api/causal/sensitivity-analysis", json=example_sensitivity_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["reference_estimator"] == "regression_adjusted"
    assert isinstance(data["reference_estimate"], float)
    assert data["observed_estimate"] == pytest.approx(data["reference_estimate"])

    # Estimator comparisons
    comparisons = data["estimator_comparisons"]
    assert "unadjusted" in comparisons
    assert "regression_adjusted" in comparisons
    assert "iptw" in comparisons
    assert comparisons["unadjusted"]["available"] is True
    assert comparisons["regression_adjusted"]["available"] is True
    assert comparisons["iptw"]["available"] is True

    # Weight truncation and unmeasured confounding default status
    assert data["iptw_truncation_analysis"]["status"] == "not_requested"
    assert data["unmeasured_confounding_analysis"]["status"] == "not_requested"
    assert "interpretation" in data
    assert "assumptions" in data


def test_sensitivity_endpoint_with_truncation_thresholds(example_sensitivity_payload):
    """Verify that specifying truncation thresholds returns evaluated truncation scenarios."""
    payload = {
        **example_sensitivity_payload,
        "truncation_thresholds": [10.0, 3.0],
    }
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 200
    data = response.json()

    trunc = data["iptw_truncation_analysis"]
    assert trunc["status"] == "evaluated"
    assert len(trunc["scenarios"]) == 2
    assert trunc["scenarios"][0]["threshold"] == 10.0
    assert trunc["scenarios"][1]["threshold"] == 3.0
    assert "estimate" in trunc["scenarios"][0]
    assert "weights_truncated_count" in trunc["scenarios"][0]
    assert "ess_total" in trunc["scenarios"][0]


def test_sensitivity_endpoint_with_hypothetical_biases(example_sensitivity_payload):
    """Verify that hypothetical bias scenarios are evaluated and adjusted estimates match formula."""
    payload = {
        **example_sensitivity_payload,
        "hypothetical_biases": [1.0, 2.0, -1.0],
    }
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 200
    data = response.json()

    ref_est = data["reference_estimate"]
    assert data["bias_scenarios"] is not None
    assert len(data["bias_scenarios"]) == 3

    # Check tau_adjusted = tau_observed - b
    scenarios = data["bias_scenarios"]
    assert scenarios[0]["hypothetical_bias"] == 1.0
    assert scenarios[0]["adjusted_estimate"] == pytest.approx(ref_est - 1.0)
    assert scenarios[1]["hypothetical_bias"] == 2.0
    assert scenarios[1]["adjusted_estimate"] == pytest.approx(ref_est - 2.0)
    assert scenarios[2]["hypothetical_bias"] == -1.0
    assert scenarios[2]["adjusted_estimate"] == pytest.approx(ref_est - (-1.0))


def test_sensitivity_endpoint_reference_estimator_selection(example_sensitivity_payload):
    """Verify that reference_estimator can be set to iptw or unadjusted."""
    # IPTW reference
    payload_iptw = {**example_sensitivity_payload, "reference_estimator": "iptw"}
    res_iptw = client.post("/api/causal/sensitivity-analysis", json=payload_iptw)
    assert res_iptw.status_code == 200
    data_iptw = res_iptw.json()
    assert data_iptw["reference_estimator"] == "iptw"
    assert data_iptw["estimator_comparisons"]["iptw"]["difference_from_reference"] == pytest.approx(0.0)

    # Unadjusted reference
    payload_unadj = {**example_sensitivity_payload, "reference_estimator": "unadjusted"}
    res_unadj = client.post("/api/causal/sensitivity-analysis", json=payload_unadj)
    assert res_unadj.status_code == 200
    data_unadj = res_unadj.json()
    assert data_unadj["reference_estimator"] == "unadjusted"
    assert data_unadj["estimator_comparisons"]["unadjusted"]["difference_from_reference"] == pytest.approx(0.0)


def test_sensitivity_endpoint_fallback_when_confounders_omitted(example_sensitivity_payload):
    """Verify that when confounders are omitted, reference falls back to unadjusted with warnings."""
    payload = {
        "data": example_sensitivity_payload["data"],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": None,
        "reference_estimator": "regression_adjusted",
    }
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["reference_estimator"] == "unadjusted"
    assert any("fell back to 'unadjusted'" in w for w in data["warnings"])
    assert data["estimator_comparisons"]["regression_adjusted"]["available"] is False
    assert data["estimator_comparisons"]["iptw"]["available"] is False


def test_sensitivity_endpoint_standalone_observed_estimate_success():
    """Verify standalone unmeasured confounding sensitivity analysis when data is omitted."""
    payload = {
        "observed_estimate": 5.0,
        "bias_values": [2.0, -2.0, 0.0],
        "confidence_interval": [3.0, 7.0],
    }
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["observed_estimate"] == pytest.approx(5.0)
    assert data["confidence_interval"] == [3.0, 7.0]
    assert len(data["bias_scenarios"]) == 3

    # Scenario 1: bias=2.0 -> adjusted=3.0, CI=[1.0, 5.0], zero crossing=False
    s1 = data["bias_scenarios"][0]
    assert s1["hypothetical_bias"] == 2.0
    assert s1["adjusted_estimate"] == pytest.approx(3.0)
    assert s1["adjusted_confidence_interval"] == pytest.approx([1.0, 5.0])
    assert s1["interval_includes_zero"] is False

    # Scenario 2: bias=-2.0 -> adjusted=7.0, CI=[5.0, 9.0], zero crossing=False
    s2 = data["bias_scenarios"][1]
    assert s2["hypothetical_bias"] == -2.0
    assert s2["adjusted_estimate"] == pytest.approx(7.0)
    assert s2["adjusted_confidence_interval"] == pytest.approx([5.0, 9.0])
    assert s2["interval_includes_zero"] is False

    # Scenario 3: bias=0.0 -> adjusted=5.0, CI=[3.0, 7.0], zero crossing=False
    s3 = data["bias_scenarios"][2]
    assert s3["hypothetical_bias"] == 0.0
    assert s3["adjusted_estimate"] == pytest.approx(5.0)
    assert s3["adjusted_confidence_interval"] == pytest.approx([3.0, 7.0])
    assert s3["interval_includes_zero"] is False


def test_sensitivity_endpoint_standalone_interval_crosses_zero():
    """Verify detection when adjusted confidence interval spans zero."""
    payload = {
        "observed_estimate": 2.0,
        "bias_values": [2.0],
        "confidence_interval": [1.0, 3.0],
    }
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 200
    data = response.json()

    s = data["bias_scenarios"][0]
    assert s["adjusted_estimate"] == pytest.approx(0.0)
    assert s["adjusted_confidence_interval"] == pytest.approx([-1.0, 1.0])
    assert s["interval_includes_zero"] is True


def test_sensitivity_endpoint_standalone_single_bias_value():
    """Verify that a single numeric float bias_values works in standalone mode."""
    payload = {
        "observed_estimate": 4.0,
        "bias_values": 1.5,
    }
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["bias_scenarios"]) == 1
    assert data["bias_scenarios"][0]["adjusted_estimate"] == pytest.approx(2.5)


def test_sensitivity_endpoint_custom_labels():
    """Verify that custom treated_value and control_value work in dataset mode."""
    data = [
        {"grp": "treatment", "outcome": 20.0, "x": 2.0},
        {"grp": "treatment", "outcome": 25.0, "x": 3.0},
        {"grp": "treatment", "outcome": 30.0, "x": 4.0},
        {"grp": "control", "outcome": 10.0, "x": 1.0},
        {"grp": "control", "outcome": 15.0, "x": 2.0},
        {"grp": "control", "outcome": 20.0, "x": 3.0},
    ]
    payload = {
        "data": data,
        "treatment_col": "grp",
        "outcome_col": "outcome",
        "confounder_cols": ["x"],
        "treated_value": "treatment",
        "control_value": "control",
    }
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 200
    assert response.json()["reference_estimator"] == "regression_adjusted"


# Validation and error handling tests

def test_sensitivity_endpoint_empty_body():
    """Verify rejection with HTTP 422 when an empty request body is submitted."""
    response = client.post("/api/causal/sensitivity-analysis", json={})
    assert response.status_code == 422
    assert "either 'data'" in str(response.json()["detail"]).lower()


def test_sensitivity_endpoint_empty_data():
    """Verify rejection with HTTP 422 when data is an empty list."""
    payload = {"data": [], "treatment_col": "t", "outcome_col": "y"}
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 422
    assert "cannot be empty" in str(response.json()["detail"]).lower()


def test_sensitivity_endpoint_missing_columns_when_data_provided():
    """Verify rejection with HTTP 422 when treatment_col or outcome_col is missing."""
    records = [{"t": 1, "y": 10.0}, {"t": 0, "y": 5.0}]
    # Missing treatment_col
    res1 = client.post("/api/causal/sensitivity-analysis", json={"data": records, "outcome_col": "y"})
    assert res1.status_code == 422
    assert "treatment_col" in str(res1.json()["detail"]).lower()

    # Missing outcome_col
    res2 = client.post("/api/causal/sensitivity-analysis", json={"data": records, "treatment_col": "t"})
    assert res2.status_code == 422
    assert "outcome_col" in str(res2.json()["detail"]).lower()


def test_sensitivity_endpoint_conflicting_column_roles():
    """Verify rejection with HTTP 422 when treatment_col, outcome_col, or confounder_cols conflict."""
    records = [{"t": 1, "y": 10.0, "x": 2.0}, {"t": 0, "y": 5.0, "x": 1.0}]
    # treatment_col == outcome_col
    res1 = client.post(
        "/api/causal/sensitivity-analysis",
        json={"data": records, "treatment_col": "t", "outcome_col": "t"},
    )
    assert res1.status_code == 422
    assert "cannot be the same column" in str(res1.json()["detail"]).lower()

    # treatment_col in confounder_cols
    res2 = client.post(
        "/api/causal/sensitivity-analysis",
        json={"data": records, "treatment_col": "t", "outcome_col": "y", "confounder_cols": ["t"]},
    )
    assert res2.status_code == 422
    assert "treatment_col" in str(res2.json()["detail"]).lower()


def test_sensitivity_endpoint_unsupported_reference_estimator(example_sensitivity_payload):
    """Verify rejection with HTTP 422 when an unsupported reference_estimator is requested."""
    payload = {
        **example_sensitivity_payload,
        "reference_estimator": "random_forest",
    }
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 422
    assert "unsupported reference_estimator" in str(response.json()["detail"]).lower()


@pytest.mark.parametrize(
    "invalid_thresh",
    [
        [-1.0],
        [0.0],
        [-0.01],
        [],
        [True],
        ["high"],
    ],
)
def test_sensitivity_endpoint_invalid_truncation_thresholds(example_sensitivity_payload, invalid_thresh):
    """Verify rejection with HTTP 422 for non-positive, empty, or boolean thresholds."""
    payload = {
        **example_sensitivity_payload,
        "truncation_thresholds": invalid_thresh,
    }
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 422


def test_sensitivity_endpoint_standalone_missing_biases():
    """Verify rejection with HTTP 422 in standalone mode when bias_values is omitted."""
    payload = {"observed_estimate": 5.0}
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 422
    assert "bias_values" in str(response.json()["detail"]).lower()


def test_sensitivity_endpoint_standalone_empty_biases():
    """Verify rejection with HTTP 422 in standalone mode when bias_values is empty list."""
    payload = {"observed_estimate": 5.0, "bias_values": []}
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 422
    assert "empty" in str(response.json()["detail"]).lower()


@pytest.mark.parametrize(
    "invalid_ci",
    [
        [10.0, 5.0],      # lower > upper
        [1.0, 2.0, 3.0],  # len != 2
        [1.0],            # len != 2
        ["not_a_number", 5.0],
        [True, 5.0],
    ],
)
def test_sensitivity_endpoint_standalone_invalid_ci(invalid_ci):
    """Verify rejection with HTTP 422 when confidence_interval bounds are reversed or malformed."""
    payload = {
        "observed_estimate": 5.0,
        "bias_values": [1.0],
        "confidence_interval": invalid_ci,
    }
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "invalid_est",
    [
        "five",
        "nan",
        "infinity",
        True,
    ],
)
def test_sensitivity_endpoint_standalone_invalid_observed_estimate(invalid_est):
    """Verify rejection with HTTP 422 when observed_estimate is non-numeric or boolean."""
    payload = {
        "observed_estimate": invalid_est,
        "bias_values": [1.0],
    }
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 422


def test_sensitivity_endpoint_missing_values_rejected_by_default(example_sensitivity_payload):
    """Verify rejection when missing values are present with drop_missing=False."""
    bad_data = [dict(row) for row in example_sensitivity_payload["data"]]
    bad_data[0]["outcome"] = None
    payload = {**example_sensitivity_payload, "data": bad_data, "drop_missing": False}
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 422
    assert "missing" in str(response.json()["detail"]).lower()


def test_sensitivity_endpoint_explicit_drop_missing(example_sensitivity_payload):
    """Verify that drop_missing=True drops rows and succeeds."""
    dirty_data = [dict(row) for row in example_sensitivity_payload["data"]]
    dirty_data.append({"treatment": 1, "outcome": None, "x": 3.0})
    dirty_data.append({"treatment": None, "outcome": 20.0, "x": 2.0})

    payload = {**example_sensitivity_payload, "data": dirty_data, "drop_missing": True}
    response = client.post("/api/causal/sensitivity-analysis", json=payload)
    assert response.status_code == 200
    assert response.json()["reference_estimator"] == "regression_adjusted"


def test_causal_all_six_endpoints_coexist(
    example_causal_payload,
    example_regression_causal_payload,
    example_propensity_payload,
    example_iptw_payload,
    example_did_payload,
    example_sensitivity_payload,
):
    """Verify that all six causal inference API endpoints coexist and operate without regressions."""
    res1 = client.post("/api/causal/unadjusted-ate", json=example_causal_payload)
    assert res1.status_code == 200
    assert res1.json()["estimator"] == "unadjusted_difference_in_means"

    res2 = client.post("/api/causal/regression-adjusted-ate", json=example_regression_causal_payload)
    assert res2.status_code == 200
    assert res2.json()["estimator"] == "regression_adjusted_ate"

    res3 = client.post("/api/causal/propensity-scores", json=example_propensity_payload)
    assert res3.status_code == 200
    assert res3.json()["estimator"] == "logistic_propensity_score"

    res4 = client.post("/api/causal/iptw-ate", json=example_iptw_payload)
    assert res4.status_code == 200
    assert res4.json()["estimator"] == "iptw_ate"

    res5 = client.post("/api/causal/difference-in-differences", json=example_did_payload)
    assert res5.status_code == 200
    assert res5.json()["estimator"] == "difference_in_differences"

    res6 = client.post("/api/causal/sensitivity-analysis", json=example_sensitivity_payload)
    assert res6.status_code == 200
    assert res6.json()["reference_estimator"] == "regression_adjusted"


# ===========================================================================
# Stage 6 Task 6.7: Causal API Cross-Endpoint Integration Review Tests
# ===========================================================================

from causal_engine import (
    estimate_unadjusted_ate_with_uncertainty,
    estimate_regression_adjusted_ate,
    estimate_propensity_scores,
    estimate_iptw_ate,
    estimate_difference_in_differences,
    run_causal_sensitivity_analysis,
    analyze_unmeasured_confounding_sensitivity,
)


def test_integration_all_six_causal_endpoints_registered():
    """Verify that all six causal inference API endpoints are properly registered in FastAPI routes."""
    openapi_schema = app.openapi()
    paths = openapi_schema.get("paths", {})
    expected_endpoints = [
        "/api/causal/unadjusted-ate",
        "/api/causal/regression-adjusted-ate",
        "/api/causal/propensity-scores",
        "/api/causal/iptw-ate",
        "/api/causal/difference-in-differences",
        "/api/causal/sensitivity-analysis",
    ]
    for endpoint in expected_endpoints:
        assert endpoint in paths, f"Expected endpoint {endpoint} not found in registered routes."
        assert "post" in paths[endpoint], f"Endpoint {endpoint} does not support POST method."


def test_integration_all_six_endpoints_valid_response_structures(
    example_causal_payload,
    example_regression_causal_payload,
    example_propensity_payload,
    example_iptw_payload,
    example_did_payload,
    example_sensitivity_payload,
):
    """Verify that all six causal endpoints return valid 200 responses with documented top-level fields."""
    endpoints_and_payloads = [
        ("/api/causal/unadjusted-ate", example_causal_payload, ["estimator", "ate", "confidence_interval", "interpretation", "assumptions"]),
        ("/api/causal/regression-adjusted-ate", example_regression_causal_payload, ["estimator", "adjusted_ate", "mean_predicted_treated", "model_features", "assumptions"]),
        ("/api/causal/propensity-scores", example_propensity_payload, ["estimator", "propensity_scores", "overlap_diagnostics", "unadjusted_covariate_balance"]),
        ("/api/causal/iptw-ate", example_iptw_payload, ["estimator", "iptw_ate", "weight_summary", "covariate_balance_before", "covariate_balance_after"]),
        ("/api/causal/difference-in-differences", example_did_payload, ["estimator", "did_estimate", "cell_counts", "treatment_change", "control_change"]),
        ("/api/causal/sensitivity-analysis", example_sensitivity_payload, ["reference_estimator", "reference_estimate", "estimator_comparisons", "assumptions"]),
    ]
    for route, payload, required_fields in endpoints_and_payloads:
        response = client.post(route, json=payload)
        assert response.status_code == 200, f"Endpoint {route} failed with status {response.status_code}: {response.text}"
        data = response.json()
        assert isinstance(data, dict), f"Endpoint {route} did not return a JSON dict."
        for field in required_fields:
            assert field in data, f"Required field '{field}' missing from {route} response."


def test_integration_all_six_endpoints_reject_empty_payloads():
    """Verify that sending empty JSON bodies to any of the six causal endpoints yields HTTP 422 with useful details."""
    causal_routes = [
        "/api/causal/unadjusted-ate",
        "/api/causal/regression-adjusted-ate",
        "/api/causal/propensity-scores",
        "/api/causal/iptw-ate",
        "/api/causal/difference-in-differences",
        "/api/causal/sensitivity-analysis",
    ]
    for route in causal_routes:
        response = client.post(route, json={})
        assert response.status_code == 422, f"Endpoint {route} did not return 422 for empty payload, got {response.status_code}"
        detail = response.json().get("detail")
        assert detail is not None, f"Endpoint {route} response missing 'detail' field."


def test_integration_error_responses_contain_no_stack_traces():
    """Verify that invalid inputs and domain errors do not leak internal Python tracebacks or file paths."""
    error_payloads = [
        ("/api/causal/unadjusted-ate", {"data": [], "treatment_col": "t", "outcome_col": "y"}),
        ("/api/causal/regression-adjusted-ate", {"data": [{"t": 1, "y": 10.0}], "treatment_col": "t", "outcome_col": "y", "confounder_cols": ["t"]}),
        ("/api/causal/propensity-scores", {"data": [{"t": 1, "x": 1.0}, {"t": 1, "x": 2.0}], "treatment_col": "t", "confounder_cols": ["x"]}),
        ("/api/causal/iptw-ate", {"data": [{"t": 1, "y": 1.0, "x": 1.0}], "treatment_col": "t", "outcome_col": "y", "confounder_cols": ["x"], "max_weight": -5.0}),
        ("/api/causal/difference-in-differences", {"data": [{"t": 1, "time": 0, "y": 1.0}], "treatment_col": "t", "time_col": "time", "outcome_col": "y"}),
        ("/api/causal/sensitivity-analysis", {"observed_estimate": 5.0, "bias_values": [1.0], "confidence_interval": [10.0, 5.0]}),
    ]
    for route, bad_payload in error_payloads:
        response = client.post(route, json=bad_payload)
        assert response.status_code == 422, f"Expected 422 from {route}, got {response.status_code}"
        text_lower = response.text.lower()
        assert "traceback (most recent call last)" not in text_lower, f"Stack trace leaked in {route}: {response.text}"
        assert "causal_engine.py" not in text_lower, f"Internal file path leaked in {route}: {response.text}"


def test_integration_non_interference_interleaved_calls(
    example_causal_payload,
    example_regression_causal_payload,
    example_propensity_payload,
    example_iptw_payload,
    example_did_payload,
    example_sensitivity_payload,
):
    """Verify that interleaved execution of different endpoints across multiple rounds produces identical, independent results."""
    # Round 1
    res1_r1 = client.post("/api/causal/unadjusted-ate", json=example_causal_payload).json()
    res2_r1 = client.post("/api/causal/regression-adjusted-ate", json=example_regression_causal_payload).json()
    res3_r1 = client.post("/api/causal/propensity-scores", json=example_propensity_payload).json()
    res4_r1 = client.post("/api/causal/iptw-ate", json=example_iptw_payload).json()
    res5_r1 = client.post("/api/causal/difference-in-differences", json=example_did_payload).json()
    res6_r1 = client.post("/api/causal/sensitivity-analysis", json=example_sensitivity_payload).json()

    # Round 2 (interleaved in reverse order)
    res6_r2 = client.post("/api/causal/sensitivity-analysis", json=example_sensitivity_payload).json()
    res5_r2 = client.post("/api/causal/difference-in-differences", json=example_did_payload).json()
    res4_r2 = client.post("/api/causal/iptw-ate", json=example_iptw_payload).json()
    res3_r2 = client.post("/api/causal/propensity-scores", json=example_propensity_payload).json()
    res2_r2 = client.post("/api/causal/regression-adjusted-ate", json=example_regression_causal_payload).json()
    res1_r2 = client.post("/api/causal/unadjusted-ate", json=example_causal_payload).json()

    assert res1_r1["ate"] == pytest.approx(res1_r2["ate"])
    assert res2_r1["adjusted_ate"] == pytest.approx(res2_r2["adjusted_ate"])
    assert res3_r1["propensity_scores"] == pytest.approx(res3_r2["propensity_scores"])
    assert res4_r1["iptw_ate"] == pytest.approx(res4_r2["iptw_ate"])
    assert res5_r1["did_estimate"] == pytest.approx(res5_r2["did_estimate"])
    assert res6_r1["reference_estimate"] == pytest.approx(res6_r2["reference_estimate"])


def test_integration_engine_agreement_unadjusted_and_regression():
    """Verify that unadjusted and regression-adjusted endpoints exactly match direct causal_engine outputs."""
    records = [
        {"treatment": 1, "outcome": 20.0, "x": 2.0},
        {"treatment": 1, "outcome": 25.0, "x": 3.0},
        {"treatment": 1, "outcome": 30.0, "x": 4.0},
        {"treatment": 0, "outcome": 10.0, "x": 1.0},
        {"treatment": 0, "outcome": 15.0, "x": 2.0},
        {"treatment": 0, "outcome": 20.0, "x": 3.0},
    ]
    # Direct engine calls
    engine_unadj = estimate_unadjusted_ate_with_uncertainty(
        data=records, treatment_col="treatment", outcome_col="outcome"
    )
    engine_reg = estimate_regression_adjusted_ate(
        data=records, treatment_col="treatment", outcome_col="outcome", confounder_cols=["x"]
    )

    # API calls
    api_unadj = client.post("/api/causal/unadjusted-ate", json={
        "data": records, "treatment_col": "treatment", "outcome_col": "outcome"
    }).json()
    api_reg = client.post("/api/causal/regression-adjusted-ate", json={
        "data": records, "treatment_col": "treatment", "outcome_col": "outcome", "confounder_cols": ["x"]
    }).json()

    assert api_unadj["ate"] == pytest.approx(engine_unadj["ate"])
    assert api_unadj["confidence_interval"] == pytest.approx(engine_unadj["confidence_interval"])
    assert api_reg["adjusted_ate"] == pytest.approx(engine_reg["adjusted_ate"])
    assert api_reg["mean_predicted_treated"] == pytest.approx(engine_reg["mean_predicted_treated"])


def test_integration_engine_agreement_propensity_alignment():
    """Verify that propensity scores returned by the API preserve exact element-wise observation alignment with the engine."""
    records = [
        {"treatment": 1, "x1": 2.5, "x2": 10.0},
        {"treatment": 0, "x1": 1.5, "x2": 15.0},
        {"treatment": 1, "x1": 3.0, "x2": 12.0},
        {"treatment": 0, "x1": 2.0, "x2": 14.0},
    ]
    engine_res = estimate_propensity_scores(
        data=records, treatment_col="treatment", confounder_cols=["x1", "x2"]
    )
    api_res = client.post("/api/causal/propensity-scores", json={
        "data": records, "treatment_col": "treatment", "confounder_cols": ["x1", "x2"]
    }).json()

    assert len(api_res["propensity_scores"]) == len(records)
    assert api_res["propensity_scores"] == pytest.approx(engine_res["propensity_scores"])
    assert api_res["observation_indices"] == list(range(len(records)))


def test_integration_engine_agreement_iptw_truncation():
    """Verify that IPTW ATE endpoint matches direct engine calculations for both untruncated and truncated weights."""
    records = [
        {"treatment": 1, "outcome": 20.0, "x": 2.0},
        {"treatment": 1, "outcome": 25.0, "x": 3.0},
        {"treatment": 1, "outcome": 30.0, "x": 4.0},
        {"treatment": 0, "outcome": 10.0, "x": 1.0},
        {"treatment": 0, "outcome": 15.0, "x": 2.0},
        {"treatment": 0, "outcome": 20.0, "x": 3.0},
    ]
    # Untruncated comparison
    engine_untrunc = estimate_iptw_ate(
        data=records, treatment_col="treatment", outcome_col="outcome", confounder_cols=["x"], truncate_weights=False
    )
    api_untrunc = client.post("/api/causal/iptw-ate", json={
        "data": records, "treatment_col": "treatment", "outcome_col": "outcome", "confounder_cols": ["x"], "truncate_weights": False
    }).json()

    assert api_untrunc["iptw_ate"] == pytest.approx(engine_untrunc["iptw_ate"])
    assert api_untrunc["ess_total"] == pytest.approx(engine_untrunc["ess_total"])

    # Truncated comparison
    engine_trunc = estimate_iptw_ate(
        data=records, treatment_col="treatment", outcome_col="outcome", confounder_cols=["x"], truncate_weights=True, max_weight=2.0
    )
    api_trunc = client.post("/api/causal/iptw-ate", json={
        "data": records, "treatment_col": "treatment", "outcome_col": "outcome", "confounder_cols": ["x"], "truncate_weights": True, "max_weight": 2.0
    }).json()

    assert api_trunc["iptw_ate"] == pytest.approx(engine_trunc["iptw_ate"])
    assert api_trunc["weight_summary"]["weights_truncated_count"] == engine_trunc["weight_summary"]["weights_truncated_count"]
    assert api_trunc["ess_total"] == pytest.approx(engine_trunc["ess_total"])


def test_integration_engine_agreement_difference_in_differences():
    """Verify that Difference-in-Differences endpoint matches direct engine calculations on 2x2 panel data."""
    records = [
        {"group": "treat", "period": "pre", "metric": 10.0},
        {"group": "treat", "period": "post", "metric": 16.0},
        {"group": "ctrl", "period": "pre", "metric": 8.0},
        {"group": "ctrl", "period": "post", "metric": 11.0},
    ]
    engine_did = estimate_difference_in_differences(
        data=records,
        treatment_col="group",
        time_col="period",
        outcome_col="metric",
        treated_value="treat",
        control_value="ctrl",
        pre_value="pre",
        post_value="post",
    )
    api_did = client.post("/api/causal/difference-in-differences", json={
        "data": records,
        "treatment_col": "group",
        "time_col": "period",
        "outcome_col": "metric",
        "treated_value": "treat",
        "control_value": "ctrl",
        "pre_value": "pre",
        "post_value": "post",
    }).json()

    assert api_did["did_estimate"] == pytest.approx(engine_did["did_estimate"])
    assert api_did["did_estimate"] == pytest.approx(3.0)
    assert api_did["treatment_change"] == pytest.approx(engine_did["treatment_change"])
    assert api_did["control_change"] == pytest.approx(engine_did["control_change"])
    assert api_did["cell_counts"] == engine_did["cell_counts"]


def test_integration_engine_agreement_sensitivity_all_modes():
    """Verify that sensitivity analysis endpoint matches direct engine functions across dataset, standalone, and fallback modes."""
    # 1. Dataset mode
    records = [
        {"t": 1, "y": 20.0, "x": 2.0},
        {"t": 1, "y": 25.0, "x": 3.0},
        {"t": 0, "y": 10.0, "x": 1.0},
        {"t": 0, "y": 15.0, "x": 2.0},
    ]
    engine_data_mode = run_causal_sensitivity_analysis(
        data=records, treatment_col="t", outcome_col="y", confounder_cols=["x"],
        truncation_thresholds=[4.0], hypothetical_biases=[1.0, 2.0]
    )
    api_data_mode = client.post("/api/causal/sensitivity-analysis", json={
        "data": records, "treatment_col": "t", "outcome_col": "y", "confounder_cols": ["x"],
        "truncation_thresholds": [4.0], "hypothetical_biases": [1.0, 2.0]
    }).json()

    assert api_data_mode["reference_estimate"] == pytest.approx(engine_data_mode["reference_estimate"])
    assert len(api_data_mode["bias_scenarios"]) == 2
    assert api_data_mode["bias_scenarios"][0]["adjusted_estimate"] == pytest.approx(
        engine_data_mode["unmeasured_confounding_analysis"]["bias_scenarios"][0]["adjusted_estimate"]
    )

    # 2. Standalone mode
    engine_standalone = analyze_unmeasured_confounding_sensitivity(
        observed_estimate=8.0, bias_values=[2.0, 4.0], confidence_interval=[6.0, 10.0]
    )
    api_standalone = client.post("/api/causal/sensitivity-analysis", json={
        "observed_estimate": 8.0, "bias_values": [2.0, 4.0], "confidence_interval": [6.0, 10.0]
    }).json()

    assert api_standalone["observed_estimate"] == pytest.approx(engine_standalone["observed_estimate"])
    assert api_standalone["bias_scenarios"][0]["adjusted_estimate"] == pytest.approx(
        engine_standalone["bias_scenarios"][0]["adjusted_estimate"]
    )
    assert api_standalone["bias_scenarios"][0]["interval_includes_zero"] == engine_standalone["bias_scenarios"][0]["interval_includes_zero"]

    # 3. Fallback mode (no confounders passed)
    api_fallback = client.post("/api/causal/sensitivity-analysis", json={
        "data": records, "treatment_col": "t", "outcome_col": "y", "confounder_cols": None, "reference_estimator": "regression_adjusted"
    }).json()
    assert api_fallback["reference_estimator"] == "unadjusted"
    assert any("fell back to 'unadjusted'" in w for w in api_fallback["warnings"])






