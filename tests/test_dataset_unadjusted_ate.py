"""Tests for Dataset-Based Unadjusted Average Treatment Effect (ATE) Workflow.

Covers:
- Safe stored dataset resolution and full-dataset row evaluation
- Numerical equivalence with raw causal unadjusted ATE estimator
- HTTP 404 for unknown/expired dataset IDs
- HTTP 422 for identical, missing, or non-binary treatment/outcome columns
- Missing value policies (drop_missing=False vs drop_missing=True)
- Hillstrom benchmark 64,000-row full-sample evaluation without truncation
- Contract compatibility for existing Stage 6 endpoints
- Dashboard HTML integration hooks
"""

from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api import app
from causal_engine import estimate_unadjusted_ate_with_uncertainty

client = TestClient(app)


# --- 1. Basic Ingestion & Analysis Equivalence ---

def test_stored_dataset_unadjusted_ate_success():
    """Verify that a stored CSV dataset is analyzed accurately with all rows evaluated."""
    csv_data = (
        "treatment,outcome,other_col\n"
        "1,10.0,alpha\n"
        "1,12.0,beta\n"
        "1,14.0,gamma\n"
        "0,4.0,delta\n"
        "0,6.0,epsilon\n"
        "0,8.0,zeta\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "simple_experiment.csv"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    analysis_res = client.post(
        f"/api/dataset/{dataset_id}/causal/unadjusted-ate",
        json={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "confidence_level": 0.95,
        },
    )
    assert analysis_res.status_code == 200
    res = analysis_res.json()

    # Point estimate & sample averages
    assert res["estimator"] == "unadjusted_difference_in_means"
    assert res["ate"] == pytest.approx(6.0)
    assert res["mean_outcome_treated"] == pytest.approx(12.0)
    assert res["mean_outcome_control"] == pytest.approx(6.0)

    # Full row accounting
    assert res["dataset_id"] == dataset_id
    assert res["dataset_filename"] == "simple_experiment.csv"
    assert res["total_dataset_rows"] == 6
    assert res["n_total"] == 6
    assert res["n_treated"] == 3
    assert res["n_control"] == 3
    assert res["rows_dropped"] == 0

    # Uncertainty estimates
    assert res["std_error"] is not None
    assert res["degrees_of_freedom"] is not None
    assert len(res["confidence_interval"]) == 2
    assert res["confidence_interval"][0] < res["ate"] < res["confidence_interval"][1]
    assert "Unadjusted difference in means" in res["assumptions"]
    assert "confidence interval" in res["interpretation"].lower()


def test_stored_dataset_matches_existing_raw_endpoint_exactly():
    """Verify numerical equivalence between dataset endpoint and raw causal endpoint."""
    raw_records = [
        {"t": 1, "y": 25.5, "extra": 99},
        {"t": 1, "y": 28.0, "extra": 98},
        {"t": 1, "y": 30.2, "extra": 97},
        {"t": 1, "y": 26.8, "extra": 96},
        {"t": 0, "y": 14.2, "extra": 95},
        {"t": 0, "y": 16.1, "extra": 94},
        {"t": 0, "y": 15.0, "extra": 93},
        {"t": 0, "y": 17.5, "extra": 92},
    ]

    # Call Stage 6 raw endpoint
    raw_res = client.post(
        "/api/causal/unadjusted-ate",
        json={
            "data": raw_records,
            "treatment_col": "t",
            "outcome_col": "y",
            "confidence_level": 0.90,
        },
    )
    assert raw_res.status_code == 200
    raw_data = raw_res.json()

    # Upload same data as CSV and call dataset endpoint
    csv_lines = ["t,y,extra"] + [f"{r['t']},{r['y']},{r['extra']}" for r in raw_records]
    csv_bytes = "\n".join(csv_lines).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_bytes,
        headers={"X-Filename": "comparison.csv"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    ds_res = client.post(
        f"/api/dataset/{dataset_id}/causal/unadjusted-ate",
        json={
            "treatment_col": "t",
            "outcome_col": "y",
            "confidence_level": 0.90,
        },
    )
    assert ds_res.status_code == 200
    ds_data = ds_res.json()

    # Compare statistics exactly
    assert ds_data["ate"] == pytest.approx(raw_data["ate"])
    assert ds_data["std_error"] == pytest.approx(raw_data["std_error"])
    assert ds_data["degrees_of_freedom"] == pytest.approx(raw_data["degrees_of_freedom"])
    assert ds_data["mean_outcome_treated"] == pytest.approx(raw_data["mean_outcome_treated"])
    assert ds_data["mean_outcome_control"] == pytest.approx(raw_data["mean_outcome_control"])
    assert ds_data["ci_lower"] == pytest.approx(raw_data["ci_lower"])
    assert ds_data["ci_upper"] == pytest.approx(raw_data["ci_upper"])
    assert ds_data["n_total"] == raw_data["n_total"]


# --- 2. Safe Error Handling & Edge Cases ---

def test_unknown_or_expired_dataset_id_returns_404():
    """Verify that an unknown or expired dataset ID returns HTTP 404."""
    fake_id = "00000000-0000-4000-8000-000000000000"
    res = client.post(
        f"/api/dataset/{fake_id}/causal/unadjusted-ate",
        json={"treatment_col": "t", "outcome_col": "y"},
    )
    assert res.status_code == 404
    assert f"Dataset '{fake_id}' not found or has expired" in res.json()["detail"]


def test_identical_treatment_and_outcome_columns_rejected():
    """Verify that identical treatment and outcome columns are rejected with HTTP 422."""
    csv_data = b"col_a,col_b\n1,10\n0,5\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/unadjusted-ate",
        json={"treatment_col": "col_a", "outcome_col": "col_a"},
    )
    assert res.status_code == 422
    assert "cannot be the same column" in str(res.json()["detail"]).lower()


def test_missing_treatment_or_outcome_column_rejected():
    """Verify that column names not present in the dataset schema return HTTP 422 with available columns."""
    csv_data = b"t,y\n1,10\n0,5\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    # Non-existent treatment
    res_t = client.post(
        f"/api/dataset/{dataset_id}/causal/unadjusted-ate",
        json={"treatment_col": "non_existent", "outcome_col": "y"},
    )
    assert res_t.status_code == 422
    assert "Treatment column 'non_existent' not found in dataset" in res_t.json()["detail"]
    assert "Available columns: ['t', 'y']" in res_t.json()["detail"]

    # Non-existent outcome
    res_o = client.post(
        f"/api/dataset/{dataset_id}/causal/unadjusted-ate",
        json={"treatment_col": "t", "outcome_col": "missing_outcome"},
    )
    assert res_o.status_code == 422
    assert "Outcome column 'missing_outcome' not found in dataset" in res_o.json()["detail"]


def test_invalid_confidence_levels_rejected():
    """Verify that confidence levels <= 0 or >= 1 are rejected with HTTP 422."""
    csv_data = b"t,y\n1,10\n0,5\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    for invalid_ci in [0.0, 1.0, -0.05, 1.5]:
        res = client.post(
            f"/api/dataset/{dataset_id}/causal/unadjusted-ate",
            json={"treatment_col": "t", "outcome_col": "y", "confidence_level": invalid_ci},
        )
        assert res.status_code == 422
        assert "confidence_level must be strictly between 0 and 1" in str(res.json()["detail"])


# --- 3. Missing Value Integrity Policy ---

def test_missing_values_with_drop_missing_false_rejected():
    """Verify that incomplete data rows are NOT silently dropped when drop_missing=False."""
    csv_data = (
        "t,y\n"
        "1,10\n"
        "1,NA\n"  # missing token
        "0,5\n"
        "0,6\n"
    ).encode("utf-8")

    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/unadjusted-ate",
        json={"treatment_col": "t", "outcome_col": "y", "drop_missing": False},
    )
    assert res.status_code == 422
    assert "contains missing values in required columns" in res.json()["detail"]
    assert "Missing rows are not dropped silently" in res.json()["detail"]


def test_missing_values_with_drop_missing_true_succeeds():
    """Verify that when drop_missing=True, incomplete rows are dropped and accounted for."""
    csv_data = (
        "t,y\n"
        "1,10\n"
        "1,12\n"
        "1,null\n"  # missing outcome -> drop
        "0,4\n"
        "0,6\n"
        "none,5\n"  # missing treatment -> drop
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "missing_test.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/unadjusted-ate",
        json={"treatment_col": "t", "outcome_col": "y", "drop_missing": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total_dataset_rows"] == 6
    assert data["rows_dropped"] == 2
    assert data["n_total"] == 4
    assert data["total_dataset_rows"] == data["n_total"] + data["rows_dropped"]
    assert data["n_treated"] == 2
    assert data["n_control"] == 2


# --- 4. Benchmark Dataset & Full-Row Integrity ---

def test_full_row_accounting_hillstrom_benchmark():
    """Verify that Hillstrom benchmark evaluates all 64,000 rows without sampling or truncation."""
    load_res = client.post("/api/dataset/load-benchmark")
    assert load_res.status_code == 200
    dataset_id = load_res.json()["dataset_id"]

    # In Hillstrom, 'mens' is binary (0/1) and 'spend' is continuous positive outcome
    res = client.post(
        f"/api/dataset/{dataset_id}/causal/unadjusted-ate",
        json={
            "treatment_col": "mens",
            "outcome_col": "spend",
            "confidence_level": 0.95,
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["total_dataset_rows"] == 64000
    assert data["n_total"] == 64000
    assert data["n_treated"] + data["n_control"] == 64000
    assert data["rows_dropped"] == 0
    assert data["ate"] == pytest.approx(0.25996, abs=0.01)
    assert data["std_error"] > 0
    assert data["degrees_of_freedom"] > 60000


def test_non_binary_treatment_column_behavior():
    """Verify that a column with >2 unique values fails, and custom binary string labels succeed."""
    load_res = client.post("/api/dataset/load-benchmark")
    dataset_id = load_res.json()["dataset_id"]

    # In Hillstrom, 'segment' has 3 unique values: 'Womens E-Mail', 'No E-Mail', 'Mens E-Mail'
    # Without labels: fails binary requirement
    res_no_labels = client.post(
        f"/api/dataset/{dataset_id}/causal/unadjusted-ate",
        json={"treatment_col": "segment", "outcome_col": "spend"},
    )
    assert res_no_labels.status_code == 422
    assert "must be binary, but contains 3 unique values" in res_no_labels.json()["detail"]

    # Even with two labels specified, unexpected 3rd label ('Mens E-Mail') is rejected per causal_engine rules
    res_with_labels = client.post(
        f"/api/dataset/{dataset_id}/causal/unadjusted-ate",
        json={
            "treatment_col": "segment",
            "outcome_col": "spend",
            "treated_value": "Womens E-Mail",
            "untreated_value": "No E-Mail",
        },
    )
    assert res_with_labels.status_code == 422
    assert "contains values other than specified treated" in res_with_labels.json()["detail"]

    # Upload a strictly binary custom-label dataset (active vs placebo)
    custom_csv = (
        "group,outcome\n"
        "active,15.5\n"
        "active,18.0\n"
        "placebo,10.0\n"
        "placebo,12.0\n"
    ).encode("utf-8")
    upload_res = client.post("/api/dataset/upload", content=custom_csv)
    custom_ds_id = upload_res.json()["dataset_id"]

    res_custom = client.post(
        f"/api/dataset/{custom_ds_id}/causal/unadjusted-ate",
        json={
            "treatment_col": "group",
            "outcome_col": "outcome",
            "treated_value": "active",
            "untreated_value": "placebo",
        },
    )
    assert res_custom.status_code == 200
    data = res_custom.json()
    assert data["treated_value"] == "active"
    assert data["control_value"] == "placebo"
    assert data["ate"] == pytest.approx(5.75)


# --- 5. Contract Compatibility & Dashboard HTML ---

def test_existing_raw_endpoint_contract_preserved():
    """Verify that the existing Stage 6 causal endpoint remains completely intact."""
    example_causal_payload = {
        "data": [
            {"treatment": 1, "outcome": 12.0, "covariate": 5.0},
            {"treatment": 1, "outcome": 14.0, "covariate": 6.0},
            {"treatment": 1, "outcome": 16.0, "covariate": 7.0},
            {"treatment": 0, "outcome": 6.0, "covariate": 5.0},
            {"treatment": 0, "outcome": 10.0, "covariate": 8.0},
            {"treatment": 0, "outcome": 8.0, "covariate": 6.0},
        ],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
    }
    res = client.post("/api/causal/unadjusted-ate", json=example_causal_payload)
    assert res.status_code == 200
    assert res.json()["estimator"] == "unadjusted_difference_in_means"
    assert res.json()["ate"] == pytest.approx(6.0)


def test_dashboard_html_contains_causal_analysis_form_and_elements():
    """Verify that /dashboard HTML contains all required Stage 7.4A form and diagnostic elements."""
    res = client.get("/dashboard")
    assert res.status_code == 200
    html = res.text

    # Form and section hooks
    assert 'id="causal-analysis-section"' in html
    assert 'id="ate-treatment-select"' in html
    assert 'id="ate-outcome-select"' in html
    assert 'id="ate-ci-level-input"' in html
    assert 'id="ate-drop-missing-checkbox"' in html
    assert 'id="run-ate-btn"' in html

    # Results diagnostic hooks
    assert 'id="ate-results-panel"' in html
    assert 'id="res-ate-point"' in html
    assert 'id="res-ate-se"' in html
    assert 'id="res-ate-ci"' in html
    assert 'id="res-eval-rows"' in html
    assert 'id="res-dropped-rows"' in html
