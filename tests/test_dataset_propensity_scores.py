"""Tests for Dataset-Based Propensity Score Estimation Workflow.

Covers:
- Safe stored dataset resolution and full-dataset row evaluation
- Numerical equivalence with raw causal propensity score estimator
- Overlap diagnostics and pre-adjustment covariate balance
- HTTP 404 for unknown/expired dataset IDs
- HTTP 422 for missing columns, empty confounders, duplicate confounders, and role conflicts
- Missing value policies (drop_missing=False rejected vs drop_missing=True evaluated with full accounting)
- Categorical and numeric confounder support with model design inspection
- Hillstrom benchmark 64,000-row full-sample evaluation without truncation
- Contract compatibility for existing Stage 6 raw propensity score endpoint
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

def test_stored_dataset_propensity_scores_success():
    """Verify that a stored CSV dataset is analyzed accurately with all rows evaluated."""
    csv_data = (
        "treatment,age,income\n"
        "1,25,45000\n"
        "1,30,52000\n"
        "1,35,48000\n"
        "1,40,60000\n"
        "0,24,42000\n"
        "0,29,50000\n"
        "0,36,46000\n"
        "0,42,58000\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "propensity_dataset.csv"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    analysis_res = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={
            "treatment_col": "treatment",
            "confounder_cols": ["age", "income"],
            "near_zero_threshold": 0.05,
            "near_one_threshold": 0.95,
        },
    )
    assert analysis_res.status_code == 200
    res = analysis_res.json()

    # Estimator type and scores
    assert res["estimator"] == "logistic_propensity_score"
    assert "propensity_scores" in res
    assert len(res["propensity_scores"]) == 8
    for s in res["propensity_scores"]:
        assert 0.0 <= s <= 1.0

    # Full row accounting
    assert res["dataset_id"] == dataset_id
    assert res["dataset_filename"] == "propensity_dataset.csv"
    assert res["total_dataset_rows"] == 8
    assert res["n_total"] == 8
    assert res["n_treated"] == 4
    assert res["n_control"] == 4
    assert res["rows_dropped"] == 0

    # Overlap diagnostics
    diag = res["overlap_diagnostics"]
    assert "has_overlap" in diag
    assert "min_score" in diag
    assert "max_score" in diag
    assert "common_support_range" in diag
    assert "scores_near_zero" in diag
    assert "scores_near_one" in diag

    # Covariate balance
    balance = res["covariate_balance"]
    assert "age" in balance
    assert "income" in balance
    assert "smd" in balance["age"]
    assert "is_balanced" in balance["age"]

    # Model metadata
    meta = res["model_metadata"]
    assert meta["converged"] is True
    assert "features" in meta
    assert "coefficients" in meta

    # Narrative explanations
    assert "interpretation" in res and len(res["interpretation"]) > 0
    assert "assumptions" in res and "positivity" in res["assumptions"].lower()


def test_stored_dataset_matches_existing_raw_endpoint_exactly():
    """Verify numerical equivalence between dataset endpoint and raw causal endpoint."""
    raw_records = [
        {"t": 1, "c1": 10.0, "c2": 100.0},
        {"t": 1, "c1": 12.0, "c2": 110.0},
        {"t": 1, "c1": 15.0, "c2": 125.0},
        {"t": 1, "c1": 11.0, "c2": 105.0},
        {"t": 0, "c1": 9.0,  "c2": 95.0},
        {"t": 0, "c1": 13.0, "c2": 115.0},
        {"t": 0, "c1": 10.5, "c2": 102.0},
        {"t": 0, "c1": 14.0, "c2": 120.0},
    ]

    # 1. Call Stage 6 raw endpoint
    raw_res = client.post(
        "/api/causal/propensity-scores",
        json={
            "data": raw_records,
            "treatment_col": "t",
            "confounder_cols": ["c1", "c2"],
            "near_zero_threshold": 0.05,
            "near_one_threshold": 0.95,
        },
    )
    assert raw_res.status_code == 200
    raw_data = raw_res.json()

    # 2. Upload same data as CSV and call dataset endpoint
    csv_lines = ["t,c1,c2"] + [f"{r['t']},{r['c1']},{r['c2']}" for r in raw_records]
    csv_bytes = "\n".join(csv_lines).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_bytes,
        headers={"X-Filename": "comparison_ps.csv"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    ds_res = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={
            "treatment_col": "t",
            "confounder_cols": ["c1", "c2"],
            "near_zero_threshold": 0.05,
            "near_one_threshold": 0.95,
        },
    )
    assert ds_res.status_code == 200
    ds_data = ds_res.json()

    # Compare statistics and scores exactly
    assert len(ds_data["propensity_scores"]) == len(raw_data["propensity_scores"])
    for s_ds, s_raw in zip(ds_data["propensity_scores"], raw_data["propensity_scores"]):
        assert s_ds == pytest.approx(s_raw, abs=1e-6)

    assert ds_data["overlap_diagnostics"]["min_score"] == pytest.approx(raw_data["overlap_diagnostics"]["min_score"], abs=1e-6)
    assert ds_data["overlap_diagnostics"]["max_score"] == pytest.approx(raw_data["overlap_diagnostics"]["max_score"], abs=1e-6)
    assert ds_data["n_total"] == raw_data["n_total"]
    assert ds_data["n_treated"] == raw_data["n_treated"]
    assert ds_data["n_control"] == raw_data["n_control"]


# --- 2. Safe Error Handling & Edge Cases ---

def test_unknown_or_expired_dataset_id_returns_404():
    """Verify that an unknown or expired dataset ID returns HTTP 404."""
    fake_id = "00000000-0000-4000-8000-000000000000"
    res = client.post(
        f"/api/dataset/{fake_id}/causal/propensity-scores",
        json={"treatment_col": "t", "confounder_cols": ["c1"]},
    )
    assert res.status_code == 404
    assert f"Dataset '{fake_id}' not found or has expired" in res.json()["detail"]


def test_empty_confounders_rejected():
    """Verify that an empty confounder list is rejected with HTTP 422."""
    csv_data = b"t,c1\n1,5\n0,3\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={"treatment_col": "t", "confounder_cols": []},
    )
    assert res.status_code == 422
    assert "at least one column name" in str(res.json()["detail"]).lower()


def test_duplicate_confounders_rejected():
    """Verify that duplicate confounder column names are rejected with HTTP 422."""
    csv_data = b"t,c1,c2\n1,5,2\n0,3,1\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={"treatment_col": "t", "confounder_cols": ["c1", "c1"]},
    )
    assert res.status_code == 422
    assert "duplicate column names" in str(res.json()["detail"]).lower()


def test_role_conflicts_rejected():
    """Verify that a confounder cannot also be the treatment column."""
    csv_data = b"col_a,col_b\n1,10\n0,5\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={"treatment_col": "col_a", "confounder_cols": ["col_a"]},
    )
    assert res.status_code == 422
    assert "cannot also be the treatment column" in str(res.json()["detail"]).lower()


def test_invalid_thresholds_rejected():
    """Verify that near_zero_threshold >= near_one_threshold or out-of-range thresholds fail."""
    csv_data = b"t,c1\n1,5\n0,3\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    # Inverted thresholds
    res_inv = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={"treatment_col": "t", "confounder_cols": ["c1"], "near_zero_threshold": 0.8, "near_one_threshold": 0.2},
    )
    assert res_inv.status_code == 422
    assert "must be strictly less than" in str(res_inv.json()["detail"])

    # Out of [0, 1] range
    res_oor = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={"treatment_col": "t", "confounder_cols": ["c1"], "near_zero_threshold": -0.1},
    )
    assert res_oor.status_code == 422


def test_missing_column_rejected():
    """Verify that non-existent column names in dataset return HTTP 422 with available columns."""
    csv_data = b"t,c1\n1,5\n0,3\n"
    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    # Missing treatment
    res_t = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={"treatment_col": "missing_t", "confounder_cols": ["c1"]},
    )
    assert res_t.status_code == 422
    assert "Treatment column 'missing_t' not found in dataset" in res_t.json()["detail"]

    # Missing confounder
    res_c = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={"treatment_col": "t", "confounder_cols": ["missing_c"]},
    )
    assert res_c.status_code == 422
    assert "Confounder column 'missing_c' not found in dataset" in res_c.json()["detail"]


# --- 3. Missing Value Integrity Policy ---

def test_missing_values_with_drop_missing_false_rejected():
    """Verify that incomplete data rows are NOT silently dropped when drop_missing=False."""
    csv_data = (
        "t,c1\n"
        "1,5.0\n"
        "1,NA\n"  # missing token in confounder
        "0,4.0\n"
        "0,4.5\n"
    ).encode("utf-8")

    upload_res = client.post("/api/dataset/upload", content=csv_data)
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={"treatment_col": "t", "confounder_cols": ["c1"], "drop_missing": False},
    )
    assert res.status_code == 422
    assert "contains missing values in required columns" in res.json()["detail"]
    assert "Missing rows are not dropped silently" in res.json()["detail"]


def test_missing_values_with_drop_missing_true_succeeds():
    """Verify that when drop_missing=True, incomplete rows are dropped and accounted for."""
    csv_data = (
        "t,c1,c2\n"
        "1,5.0,10.0\n"
        "1,6.0,12.0\n"
        "1,null,14.0\n"  # missing confounder c1 -> dropped
        "0,3.0,8.0\n"
        "0,4.0,9.0\n"
        "none,3.5,7.0\n"  # missing treatment -> dropped
        "0,NaN,11.0\n"    # missing confounder c1 -> dropped
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "missing_ps_test.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={"treatment_col": "t", "confounder_cols": ["c1", "c2"], "drop_missing": True},
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
    """Verify that categorical confounders are handled properly via one-hot encoding in Logit."""
    csv_data = (
        "treatment,region,score\n"
        "1,North,85\n"
        "1,South,88\n"
        "1,North,90\n"
        "1,South,92\n"
        "0,North,84\n"
        "0,South,87\n"
        "0,North,89\n"
        "0,South,91\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "categorical_ps.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={
            "treatment_col": "treatment",
            "confounder_cols": ["region", "score"],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "region" in data["categorical_confounders"]
    assert "score" in data["numeric_confounders"]
    assert data["n_total"] == 8
    assert data["model_metadata"]["converged"] is True


def test_full_row_accounting_hillstrom_benchmark():
    """Verify that Hillstrom benchmark evaluates all 64,000 rows without sampling or truncation."""
    load_res = client.post("/api/dataset/load-benchmark")
    assert load_res.status_code == 200
    dataset_id = load_res.json()["dataset_id"]

    # In Hillstrom, 'mens' is binary treatment, 'recency' and 'history' are confounders
    res = client.post(
        f"/api/dataset/{dataset_id}/causal/propensity-scores",
        json={
            "treatment_col": "mens",
            "confounder_cols": ["recency", "history"],
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["total_dataset_rows"] == 64000
    assert data["n_total"] == 64000
    assert data["n_treated"] == 35266
    assert data["n_control"] == 28734
    assert data["n_treated"] + data["n_control"] == 64000
    assert data["rows_dropped"] == 0
    assert len(data["propensity_scores"]) == 64000

    diag = data["overlap_diagnostics"]
    assert diag["has_overlap"] is True
    assert diag["min_score"] == pytest.approx(0.4992, abs=0.01)
    assert diag["max_score"] == pytest.approx(0.9593, abs=0.01)

    # Covariate balance
    balance = data["covariate_balance"]
    assert "recency" in balance
    assert "history" in balance


# --- 5. Contract Compatibility & Dashboard HTML ---

def test_existing_raw_endpoint_contract_preserved():
    """Verify that the existing Stage 6 causal propensity score endpoint remains completely intact."""
    example_causal_payload = {
        "data": [
            {"t": 1, "cov": 5.0},
            {"t": 1, "cov": 6.0},
            {"t": 1, "cov": 7.0},
            {"t": 0, "cov": 5.0},
            {"t": 0, "cov": 8.0},
            {"t": 0, "cov": 6.0},
        ],
        "treatment_col": "t",
        "confounder_cols": ["cov"],
    }
    res = client.post("/api/causal/propensity-scores", json=example_causal_payload)
    assert res.status_code == 200
    assert res.json()["estimator"] == "logistic_propensity_score"
    assert "propensity_scores" in res.json()


def test_dashboard_html_contains_propensity_analysis_elements():
    """Verify that /dashboard HTML contains all required Stage 7.5A form and diagnostic elements."""
    res = client.get("/dashboard")
    assert res.status_code == 200
    html = res.text

    # Section 4 form and control hooks
    assert 'id="propensity-analysis-section"' in html
    assert 'id="ps-status-banner"' in html
    assert 'id="ps-unloaded-notice"' in html
    assert 'id="ps-form-container"' in html
    assert 'id="ps-treatment-select"' in html
    assert 'id="ps-confounders-container"' in html
    assert 'id="ps-near-zero-input"' in html
    assert 'id="ps-near-one-input"' in html
    assert 'id="ps-drop-missing-checkbox"' in html
    assert 'id="run-ps-btn"' in html

    # Section 4 results diagnostic hooks
    assert 'id="ps-results-panel"' in html
    assert 'id="res-ps-dataset-name"' in html
    assert 'id="res-ps-total-rows"' in html
    assert 'id="res-ps-eval-rows"' in html
    assert 'id="res-ps-dropped-rows"' in html
    assert 'id="res-ps-overlap-status"' in html
    assert 'id="res-ps-support-range"' in html
    assert 'id="res-ps-treated-range"' in html
    assert 'id="res-ps-control-range"' in html
    assert 'id="res-ps-boundary-count"' in html
    assert 'id="res-ps-pseudo-r2"' in html
    assert 'id="ps-balance-table"' in html
    assert 'id="ps-balance-tbody"' in html
    assert 'id="res-ps-interpretation-text"' in html
    assert 'id="res-ps-assumptions-text"' in html
