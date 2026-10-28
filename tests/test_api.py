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



