"""Tests for Dataset-Based Regression-Adjusted Average Treatment Effect (ATE) Workflow.

Covers:
- Safe stored dataset resolution and full-dataset row evaluation
- Numerical equivalence with raw causal regression-adjusted ATE estimator
- HTTP 404 for unknown/expired dataset IDs
- HTTP 422 for missing columns, duplicate confounders, and role conflicts (treatment, outcome, confounder)
- Missing value policies (drop_missing=False rejected vs drop_missing=True evaluated with full accounting)
- Categorical and numeric confounder support with model features inspection
- Hillstrom benchmark 64,000-row full-sample evaluation without truncation
- Contract compatibility for existing Stage 6 raw regression endpoint
- Dashboard HTML integration hooks
"""

from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api import app

client = TestClient(app)


# --- 1. Basic Ingestion & Analysis Equivalence ---

def test_stored_dataset_regression_adjusted_ate_success():
    """Verify that a stored CSV dataset is analyzed accurately with all rows evaluated."""
    csv_data = (
        "treatment,outcome,age,income\n"
        "1,15.0,25,50000\n"
        "1,18.0,30,55000\n"
        "1,22.0,35,62000\n"
        "1,20.0,28,58000\n"
        "0,8.0,24,48000\n"
        "0,11.0,29,51000\n"
        "0,14.0,36,60000\n"
        "0,10.0,27,49000\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "experiment_with_covariates.csv"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    analysis_res = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "confounder_cols": ["age", "income"],
        },
    )
    assert analysis_res.status_code == 200
    res = analysis_res.json()

    # Estimator type and estimates
    assert res["estimator"] == "regression_adjusted_ate"
    assert "adjusted_ate" in res
    assert "unadjusted_ate" in res
    assert res["adjusted_ate"] == pytest.approx(res["ate"])
    assert res["mean_predicted_treated"] is not None
    assert res["mean_predicted_control"] is not None
    assert res["r_squared"] is not None
    assert 0.0 <= res["r_squared"] <= 1.0

    # Full row accounting
    assert res["dataset_id"] == dataset_id
    assert res["dataset_filename"] == "experiment_with_covariates.csv"
    assert res["total_dataset_rows"] == 8
    assert res["n_total"] == 8
    assert res["n_treated"] == 4
    assert res["n_control"] == 4
    assert res["rows_dropped"] == 0

    # Confounder accounting
    assert res["confounder_cols"] == ["age", "income"]
    assert "age" in res["numeric_confounders"]
    assert "income" in res["numeric_confounders"]
    assert len(res["model_features"]) >= 3

    # Narrative explanations
    assert "interpretation" in res and len(res["interpretation"]) > 0
    assert "assumptions" in res and "confounding" in res["assumptions"].lower()


def test_stored_dataset_matches_existing_raw_endpoint_exactly():
    """Verify numerical equivalence between dataset endpoint and raw causal endpoint."""
    raw_records = [
        {"t": 1, "y": 25.5, "c1": 10.0, "c2": 100.0},
        {"t": 1, "y": 28.0, "c1": 12.0, "c2": 110.0},
        {"t": 1, "y": 30.2, "c1": 15.0, "c2": 125.0},
        {"t": 1, "y": 26.8, "c1": 11.0, "c2": 105.0},
        {"t": 0, "y": 14.2, "c1": 9.0,  "c2": 95.0},
        {"t": 0, "y": 16.1, "c1": 13.0, "c2": 115.0},
        {"t": 0, "y": 15.0, "c1": 10.5, "c2": 102.0},
        {"t": 0, "y": 17.5, "c1": 14.0, "c2": 120.0},
    ]

    # 1. Call Stage 6 raw endpoint
    raw_res = client.post(
        "/api/causal/regression-adjusted-ate",
        json={
            "data": raw_records,
            "treatment_col": "t",
            "outcome_col": "y",
            "confounder_cols": ["c1", "c2"],
        },
    )
    assert raw_res.status_code == 200
    raw_data = raw_res.json()

    # 2. Upload same data as CSV and call dataset endpoint
    csv_lines = ["t,y,c1,c2"] + [f"{r['t']},{r['y']},{r['c1']},{r['c2']}" for r in raw_records]
    csv_bytes = "\n".join(csv_lines).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_bytes,
        headers={"X-Filename": "comparison_regression.csv"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    ds_res = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={
            "treatment_col": "t",
            "outcome_col": "y",
            "confounder_cols": ["c1", "c2"],
        },
    )
    assert ds_res.status_code == 200
    ds_data = ds_res.json()

    # Compare statistics exactly
    assert ds_data["adjusted_ate"] == pytest.approx(raw_data["adjusted_ate"], abs=1e-6)
    assert ds_data["unadjusted_ate"] == pytest.approx(raw_data["unadjusted_ate"], abs=1e-6)
    assert ds_data["mean_predicted_treated"] == pytest.approx(raw_data["mean_predicted_treated"], abs=1e-6)
    assert ds_data["mean_predicted_control"] == pytest.approx(raw_data["mean_predicted_control"], abs=1e-6)
    assert ds_data["r_squared"] == pytest.approx(raw_data["r_squared"], abs=1e-6)
    assert ds_data["n_total"] == raw_data["n_total"]
    assert ds_data["n_treated"] == raw_data["n_treated"]
    assert ds_data["n_control"] == raw_data["n_control"]


# --- 2. Safe Error Handling & Edge Cases ---

def test_unknown_or_expired_dataset_id_returns_404():
    """Verify that an unknown or expired dataset ID returns HTTP 404."""
    fake_id = "00000000-0000-4000-8000-000000000000"
    res = client.post(
        f"/api/dataset/{fake_id}/causal/regression-adjusted-ate",
        json={"treatment_col": "t", "outcome_col": "y", "confounder_cols": ["c1"]},
    )
    assert res.status_code == 404
    assert f"Dataset '{fake_id}' not found or has expired" in res.json()["detail"]


def test_empty_confounders_rejected():
    """Verify that an empty confounder list is rejected with HTTP 422."""
    csv_data = b"t,y,c1\n1,10,5\n0,5,3\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={"treatment_col": "t", "outcome_col": "y", "confounder_cols": []},
    )
    assert res.status_code == 422
    assert "at least one column name" in str(res.json()["detail"]).lower()


def test_duplicate_confounders_rejected():
    """Verify that duplicate confounder column names are rejected with HTTP 422."""
    csv_data = b"t,y,c1,c2\n1,10,5,2\n0,5,3,1\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={"treatment_col": "t", "outcome_col": "y", "confounder_cols": ["c1", "c1"]},
    )
    assert res.status_code == 422
    assert "duplicate column names" in str(res.json()["detail"]).lower()


def test_role_conflicts_rejected():
    """Verify that role conflicts among treatment, outcome, and confounders are rejected."""
    csv_data = b"col_a,col_b,col_c\n1,10,5\n0,5,3\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    # 1. Treatment == Outcome
    res1 = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={"treatment_col": "col_a", "outcome_col": "col_a", "confounder_cols": ["col_c"]},
    )
    assert res1.status_code == 422
    assert "cannot be the same column" in str(res1.json()["detail"]).lower()

    # 2. Confounder == Treatment
    res2 = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={"treatment_col": "col_a", "outcome_col": "col_b", "confounder_cols": ["col_a"]},
    )
    assert res2.status_code == 422
    assert "cannot also be the treatment column" in str(res2.json()["detail"]).lower()

    # 3. Confounder == Outcome
    res3 = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={"treatment_col": "col_a", "outcome_col": "col_b", "confounder_cols": ["col_b"]},
    )
    assert res3.status_code == 422
    assert "cannot also be the outcome column" in str(res3.json()["detail"]).lower()


def test_missing_column_rejected():
    """Verify that non-existent column names in dataset return HTTP 422 with available columns."""
    csv_data = b"t,y,c1\n1,10,5\n0,5,3\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    # Missing treatment
    res_t = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={"treatment_col": "missing_t", "outcome_col": "y", "confounder_cols": ["c1"]},
    )
    assert res_t.status_code == 422
    assert "Treatment column 'missing_t' not found in dataset" in res_t.json()["detail"]

    # Missing outcome
    res_o = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={"treatment_col": "t", "outcome_col": "missing_y", "confounder_cols": ["c1"]},
    )
    assert res_o.status_code == 422
    assert "Outcome column 'missing_y' not found in dataset" in res_o.json()["detail"]

    # Missing confounder
    res_c = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={"treatment_col": "t", "outcome_col": "y", "confounder_cols": ["missing_c"]},
    )
    assert res_c.status_code == 422
    assert "Confounder column 'missing_c' not found in dataset" in res_c.json()["detail"]


# --- 3. Missing Value Integrity Policy ---

def test_missing_values_with_drop_missing_false_rejected():
    """Verify that incomplete data rows are NOT silently dropped when drop_missing=False."""
    csv_data = (
        "t,y,c1\n"
        "1,10,5.0\n"
        "1,12,NA\n"  # missing token in confounder
        "0,5,4.0\n"
        "0,6,4.5\n"
    ).encode("utf-8")

    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={"treatment_col": "t", "outcome_col": "y", "confounder_cols": ["c1"], "drop_missing": False},
    )
    assert res.status_code == 422
    assert "contains missing values in required columns" in res.json()["detail"]
    assert "Missing rows are not dropped silently" in res.json()["detail"]


def test_missing_values_with_drop_missing_true_succeeds():
    """Verify that when drop_missing=True, incomplete rows are dropped and accounted for."""
    csv_data = (
        "t,y,c1\n"
        "1,10,5.0\n"
        "1,12,6.0\n"
        "1,null,7.0\n"  # missing outcome -> dropped
        "0,4,3.0\n"
        "0,6,4.0\n"
        "none,5,3.5\n"  # missing treatment -> dropped
        "0,7,NaN\n"      # missing confounder -> dropped
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "missing_reg_test.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={"treatment_col": "t", "outcome_col": "y", "confounder_cols": ["c1"], "drop_missing": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total_dataset_rows"] == 7
    assert data["rows_dropped"] == 3
    assert data["n_total"] == 4
    assert data["total_dataset_rows"] == data["n_total"] + data["rows_dropped"]
    assert data["n_treated"] == 2
    assert data["n_control"] == 2


# --- 4. Categorical Confounders & Benchmark Dataset ---

def test_categorical_confounder_support():
    """Verify that categorical confounders are handled properly via one-hot encoding."""
    csv_data = (
        "treatment,outcome,region,score\n"
        "1,20.0,North,85\n"
        "1,22.0,North,88\n"
        "1,25.0,South,92\n"
        "1,24.0,South,90\n"
        "0,12.0,North,82\n"
        "0,14.0,South,87\n"
        "0,15.0,South,89\n"
        "0,11.0,North,80\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "categorical_confounders.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "confounder_cols": ["region", "score"],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "region" in data["categorical_confounders"]
    assert "score" in data["numeric_confounders"]
    assert data["n_total"] == 8
    assert data["r_squared"] > 0.5


def test_full_row_accounting_hillstrom_benchmark():
    """Verify that Hillstrom benchmark evaluates all 64,000 rows without sampling or truncation."""
    load_res = client.post("/api/dataset/load-benchmark")
    assert load_res.status_code == 200
    dataset_id = load_res.json()["dataset_id"]

    # In Hillstrom, 'mens' is binary, 'spend' is continuous outcome, 'recency' and 'history' are confounders
    res = client.post(
        f"/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
        json={
            "treatment_col": "mens",
            "outcome_col": "spend",
            "confounder_cols": ["recency", "history"],
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["total_dataset_rows"] == 64000
    assert data["n_total"] == 64000
    assert data["n_treated"] + data["n_control"] == 64000
    assert data["rows_dropped"] == 0
    assert data["adjusted_ate"] == pytest.approx(0.187, abs=0.05)
    assert data["unadjusted_ate"] == pytest.approx(0.260, abs=0.05)
    assert data["r_squared"] is not None
    assert data["mean_predicted_treated"] is not None
    assert data["mean_predicted_control"] is not None


# --- 5. Contract Compatibility & Dashboard HTML ---

def test_existing_raw_endpoint_contract_preserved():
    """Verify that the existing Stage 6 causal endpoint remains completely intact."""
    example_causal_payload = {
        "data": [
            {"t": 1, "y": 12.0, "cov": 5.0},
            {"t": 1, "y": 14.0, "cov": 6.0},
            {"t": 1, "y": 16.0, "cov": 7.0},
            {"t": 0, "y": 6.0, "cov": 5.0},
            {"t": 0, "y": 10.0, "cov": 8.0},
            {"t": 0, "y": 8.0, "cov": 6.0},
        ],
        "treatment_col": "t",
        "outcome_col": "y",
        "confounder_cols": ["cov"],
    }
    res = client.post("/api/causal/regression-adjusted-ate", json=example_causal_payload)
    assert res.status_code == 200
    assert res.json()["estimator"] == "regression_adjusted_ate"
    assert "adjusted_ate" in res.json()


def test_dashboard_html_contains_regression_analysis_elements():
    """Verify that /dashboard HTML contains all required Stage 7.4B form and diagnostic elements."""
    res = client.get("/dashboard")
    assert res.status_code == 200
    html = res.text

    # Section 3 form and control hooks
    assert 'id="regression-analysis-section"' in html
    assert 'id="reg-status-banner"' in html
    assert 'id="reg-unloaded-notice"' in html
    assert 'id="reg-form-container"' in html
    assert 'id="reg-treatment-select"' in html
    assert 'id="reg-outcome-select"' in html
    assert 'id="reg-confounders-container"' in html
    assert 'id="reg-drop-missing-checkbox"' in html
    assert 'id="run-reg-btn"' in html

    # Section 3 results diagnostic hooks
    assert 'id="reg-results-panel"' in html
    assert 'id="res-reg-dataset-name"' in html
    assert 'id="res-reg-total-rows"' in html
    assert 'id="res-reg-eval-rows"' in html
    assert 'id="res-reg-dropped-rows"' in html
    assert 'id="res-reg-adjusted-point"' in html
    assert 'id="res-reg-raw-point"' in html
    assert 'id="res-reg-r2"' in html
    assert 'id="res-reg-mean-treated"' in html
    assert 'id="res-reg-mean-control"' in html
    assert 'id="res-reg-num-conf"' in html
    assert 'id="res-reg-cat-conf"' in html
    assert 'id="res-reg-interpretation-text"' in html
    assert 'id="res-reg-assumptions-text"' in html
