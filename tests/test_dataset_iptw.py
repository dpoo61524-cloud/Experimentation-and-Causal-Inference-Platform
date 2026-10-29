"""Tests for Dataset-Based Inverse Probability of Treatment Weighting (IPTW) Workflow.

Covers:
- Safe stored dataset resolution and full-dataset row evaluation
- Numerical equivalence with raw causal IPTW estimator
- Weight summaries, extreme weight detection, and Kish effective sample size (ESS)
- Optional weight truncation (clipping) and truncation accounting
- Covariate balance diagnostics (SMD) before and after weighting
- HTTP 404 for unknown/expired dataset IDs
- HTTP 422 for missing columns, empty confounders, duplicate confounders, and role conflicts
- Missing value policies (drop_missing=False rejected vs drop_missing=True evaluated with full accounting)
- Categorical and numeric confounder support with model design inspection
- Hillstrom benchmark 64,000-row full-sample evaluation without truncation
- Contract compatibility for existing Stage 6 raw IPTW endpoint
- Dashboard HTML integration hooks
"""

from pathlib import Path
import math
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api import app
from causal_engine import estimate_iptw_ate

client = TestClient(app)


# --- 1. Basic Ingestion & Analysis Equivalence ---

def test_stored_dataset_iptw_ate_success():
    """Verify that a stored CSV dataset is analyzed accurately with all rows evaluated."""
    # Data with overlapping confounders ensuring stable Logit estimation
    csv_data = (
        "treatment,outcome,x,z\n"
        "1,10.0,1.0,20.0\n"
        "1,12.0,2.0,25.0\n"
        "1,14.0,3.0,30.0\n"
        "1,16.0,4.0,35.0\n"
        "0,5.0,1.0,22.0\n"
        "0,7.0,2.0,24.0\n"
        "0,9.0,3.0,28.0\n"
        "0,11.0,4.0,32.0\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "iptw_dataset.csv"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    analysis_res = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "confounder_cols": ["x", "z"],
            "extreme_weight_threshold": 20.0,
            "truncate_weights": False,
        },
    )
    assert analysis_res.status_code == 200
    res = analysis_res.json()

    # Estimator metadata & sample counts
    assert res["estimator"] == "iptw_ate"
    assert res["dataset_id"] == dataset_id
    assert res["dataset_filename"] == "iptw_dataset.csv"
    assert res["total_dataset_rows"] == 8
    assert res["n_total"] == 8
    assert res["n_treated"] == 4
    assert res["n_control"] == 4
    assert res["rows_dropped"] == 0

    # Effect estimates
    assert "iptw_ate" in res
    assert "unadjusted_ate" in res
    assert res["ate"] == res["iptw_ate"]
    assert res["unadjusted_ate"] == pytest.approx(5.0)

    # Weighted means
    assert res["weighted_mean_outcome_treated"] == pytest.approx(res["weighted_mean_treated"])
    assert res["weighted_mean_outcome_control"] == pytest.approx(res["weighted_mean_control"])
    assert res["iptw_ate"] == pytest.approx(
        res["weighted_mean_outcome_treated"] - res["weighted_mean_outcome_control"],
        abs=1e-6,
    )

    # Effective sample sizes
    assert res["ess_treated"] > 0
    assert res["ess_control"] > 0
    assert res["ess_total"] == pytest.approx(res["ess_treated"] + res["ess_control"], abs=1e-6)
    assert 0.0 < res["ess_treated_fraction"] <= 1.0
    assert 0.0 < res["ess_control_fraction"] <= 1.0

    # Weight distribution
    ws = res["weight_summary"]
    assert ws["min_weight"] <= ws["max_weight"]
    assert ws["is_truncated"] is False
    assert ws["rows_truncated"] == 0

    # Covariate balance
    assert "covariate_balance_before" in res
    assert "covariate_balance_after" in res
    assert "balance_summary" in res
    assert "interpretation" in res
    assert "assumptions" in res


def test_stored_dataset_iptw_numerical_equivalence_with_engine():
    """Verify exact numerical agreement between dataset endpoint and raw estimate_iptw_ate."""
    csv_data = (
        "t,y,age,score\n"
        "1,15.2,25,80.0\n"
        "1,18.4,30,85.0\n"
        "1,16.8,35,78.0\n"
        "1,21.0,40,92.0\n"
        "0,10.1,24,79.0\n"
        "0,12.5,29,84.0\n"
        "0,11.8,36,77.0\n"
        "0,14.2,41,90.0\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "comparison_iptw.csv"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    req_payload = {
        "treatment_col": "t",
        "outcome_col": "y",
        "confounder_cols": ["age", "score"],
        "drop_missing": False,
        "extreme_weight_threshold": 20.0,
        "truncate_weights": False,
    }

    api_res = client.post(f"/api/dataset/{dataset_id}/causal/iptw-ate", json=req_payload)
    assert api_res.status_code == 200
    api_data = api_res.json()

    # Read into pandas and compute directly via engine
    import pandas as pd
    import io
    df = pd.read_csv(io.BytesIO(csv_data))
    engine_data = estimate_iptw_ate(
        data=df,
        treatment_col="t",
        outcome_col="y",
        confounder_cols=["age", "score"],
        drop_missing=False,
    )

    assert api_data["iptw_ate"] == pytest.approx(engine_data["iptw_ate"], abs=1e-6)
    assert api_data["unadjusted_ate"] == pytest.approx(engine_data["unadjusted_ate"], abs=1e-6)
    assert api_data["weighted_mean_outcome_treated"] == pytest.approx(engine_data["weighted_mean_outcome_treated"], abs=1e-6)
    assert api_data["weighted_mean_outcome_control"] == pytest.approx(engine_data["weighted_mean_outcome_control"], abs=1e-6)
    assert api_data["ess_treated"] == pytest.approx(engine_data["ess_treated"], abs=1e-6)
    assert api_data["ess_control"] == pytest.approx(engine_data["ess_control"], abs=1e-6)


# --- 2. Weight Truncation & Diagnostic Options ---

def test_stored_dataset_iptw_weight_truncation():
    """Verify weight truncation (clipping) behavior on a dataset."""
    csv_data = (
        "treatment,outcome,x,z\n"
        "1,10.0,1.0,20.0\n"
        "1,12.0,2.0,25.0\n"
        "1,14.0,3.0,30.0\n"
        "1,16.0,4.0,35.0\n"
        "0,5.0,1.0,22.0\n"
        "0,7.0,2.0,24.0\n"
        "0,9.0,3.0,28.0\n"
        "0,11.0,4.0,32.0\n"
    ).encode("utf-8")

    upload_res = client.post("/api/dataset/upload", content=csv_data, headers={"X-Filename": "trunc_iptw.csv"})
    dataset_id = upload_res.json()["dataset_id"]

    # First get untruncated max weight
    res_untrunc = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "confounder_cols": ["x", "z"],
            "truncate_weights": False,
        },
    ).json()

    max_untrunc = res_untrunc["weight_summary"]["max_weight"]
    cap = max_untrunc * 0.75  # Set cap below max weight to force clipping

    res_trunc = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "confounder_cols": ["x", "z"],
            "truncate_weights": True,
            "max_weight": cap,
        },
    ).json()

    ws = res_trunc["weight_summary"]
    assert ws["is_truncated"] is True
    assert ws["truncation_max_weight"] == cap
    assert ws["rows_truncated"] > 0
    assert ws["max_weight"] <= cap + 1e-9
    for w in res_trunc["weights"]:
        assert w <= cap + 1e-9


def test_stored_dataset_iptw_extreme_weights_flagged():
    """Verify that extreme weights exceeding threshold are detected and warned."""
    csv_data = (
        "treatment,outcome,x,z\n"
        "1,10.0,1.0,20.0\n"
        "1,12.0,2.0,25.0\n"
        "1,14.0,3.0,30.0\n"
        "1,16.0,4.0,35.0\n"
        "0,5.0,1.0,22.0\n"
        "0,7.0,2.0,24.0\n"
        "0,9.0,3.0,28.0\n"
        "0,11.0,4.0,32.0\n"
    ).encode("utf-8")

    upload_res = client.post("/api/dataset/upload", content=csv_data, headers={"X-Filename": "extreme_test.csv"})
    dataset_id = upload_res.json()["dataset_id"]

    # Set very low extreme threshold (e.g. 1.5) so that weights exceed it
    res = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "confounder_cols": ["x", "z"],
            "extreme_weight_threshold": 1.5,
        },
    ).json()

    ws = res["weight_summary"]
    assert ws["has_extreme_weights"] is True
    assert ws["n_extreme_weights"] > 0
    assert any("Extreme weights warning" in w for w in res["warnings"])


# --- 3. Error Handling and Validation ---

def test_stored_dataset_iptw_dataset_not_found():
    """Verify HTTP 404 when dataset_id does not exist."""
    res = client.post(
        "/api/dataset/nonexistent-id-00000000/causal/iptw-ate",
        json={"treatment_col": "treatment", "outcome_col": "outcome", "confounder_cols": ["x"]},
    )
    assert res.status_code == 404
    assert "not found or has expired" in res.json()["detail"].lower()


def test_stored_dataset_iptw_missing_columns():
    """Verify HTTP 422 when specified treatment, outcome, or confounder column is missing from dataset."""
    csv_data = "treatment,outcome,x\n1,10.0,1.0\n0,5.0,1.0\n".encode("utf-8")
    upload_res = client.post("/api/dataset/upload", content=csv_data, headers={"X-Filename": "test_cols.csv"})
    dataset_id = upload_res.json()["dataset_id"]

    # Missing outcome
    res1 = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={"treatment_col": "treatment", "outcome_col": "nonexistent_outcome", "confounder_cols": ["x"]},
    )
    assert res1.status_code == 422
    assert "nonexistent_outcome" in res1.json()["detail"]

    # Missing confounder
    res2 = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={"treatment_col": "treatment", "outcome_col": "outcome", "confounder_cols": ["missing_conf"]},
    )
    assert res2.status_code == 422
    assert "missing_conf" in res2.json()["detail"]


def test_stored_dataset_iptw_role_conflicts():
    """Verify HTTP 422 when treatment == outcome or confounders overlap with treatment/outcome."""
    csv_data = "treatment,outcome,x\n1,10.0,1.0\n0,5.0,1.0\n".encode("utf-8")
    upload_res = client.post("/api/dataset/upload", content=csv_data, headers={"X-Filename": "test_roles.csv"})
    dataset_id = upload_res.json()["dataset_id"]

    # treatment == outcome
    res1 = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={"treatment_col": "treatment", "outcome_col": "treatment", "confounder_cols": ["x"]},
    )
    assert res1.status_code == 422
    assert "cannot be the same column" in str(res1.json()["detail"]).lower()

    # confounder == treatment
    res2 = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={"treatment_col": "treatment", "outcome_col": "outcome", "confounder_cols": ["treatment"]},
    )
    assert res2.status_code == 422
    assert "cannot also be the treatment column" in str(res2.json()["detail"]).lower()

    # confounder == outcome
    res3 = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={"treatment_col": "treatment", "outcome_col": "outcome", "confounder_cols": ["outcome"]},
    )
    assert res3.status_code == 422
    assert "cannot also be the outcome column" in str(res3.json()["detail"]).lower()

    # duplicate confounders
    res4 = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={"treatment_col": "treatment", "outcome_col": "outcome", "confounder_cols": ["x", "x"]},
    )
    assert res4.status_code == 422
    assert "duplicate" in str(res4.json()["detail"]).lower()


def test_stored_dataset_iptw_invalid_truncation_settings():
    """Verify HTTP 422 when truncation parameters are invalid."""
    csv_data = "treatment,outcome,x\n1,10.0,1.0\n0,5.0,1.0\n".encode("utf-8")
    upload_res = client.post("/api/dataset/upload", content=csv_data, headers={"X-Filename": "test_params.csv"})
    dataset_id = upload_res.json()["dataset_id"]

    # Negative extreme weight threshold
    res1 = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "confounder_cols": ["x"],
            "extreme_weight_threshold": -5.0,
        },
    )
    assert res1.status_code == 422

    # Negative max_weight
    res2 = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "confounder_cols": ["x"],
            "truncate_weights": True,
            "max_weight": -1.0,
        },
    )
    assert res2.status_code == 422


# --- 4. Missing Data & Row Accounting Integrity ---

def test_stored_dataset_iptw_missing_data_rejected_by_default():
    """Verify HTTP 422 when dataset has missing values in analysis columns and drop_missing=False."""
    csv_data = (
        "treatment,outcome,x,z\n"
        "1,10.0,1.0,20.0\n"
        "1,12.0,,25.0\n"  # missing x
        "1,14.0,3.0,30.0\n"
        "1,16.0,4.0,35.0\n"
        "0,5.0,1.0,22.0\n"
        "0,7.0,2.0,24.0\n"
        "0,9.0,3.0,28.0\n"
        "0,11.0,4.0,32.0\n"
    ).encode("utf-8")

    upload_res = client.post("/api/dataset/upload", content=csv_data, headers={"X-Filename": "missing_iptw.csv"})
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "confounder_cols": ["x", "z"],
            "drop_missing": False,
        },
    )
    assert res.status_code == 422
    assert "missing" in res.json()["detail"].lower()


def test_stored_dataset_iptw_explicit_drop_missing_row_accounting():
    """Verify that drop_missing=True accurately reports total rows, dropped rows, and evaluated rows."""
    csv_data = (
        "treatment,outcome,x,z\n"
        "1,10.0,1.0,20.0\n"
        "1,12.0,,25.0\n"   # missing x (row 2)
        "1,14.0,3.0,30.0\n"
        "1,16.0,4.0,35.0\n"
        "0,5.0,1.0,22.0\n"
        "0,7.0,2.0,\n"     # missing z (row 6)
        "0,9.0,3.0,28.0\n"
        "0,11.0,4.0,32.0\n"
    ).encode("utf-8")

    upload_res = client.post("/api/dataset/upload", content=csv_data, headers={"X-Filename": "drop_iptw.csv"})
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "confounder_cols": ["x", "z"],
            "drop_missing": True,
        },
    )
    assert res.status_code == 200
    data = res.json()

    # Exact row accounting
    assert data["total_dataset_rows"] == 8
    assert data["rows_dropped"] == 2
    assert data["n_total"] == 6
    assert data["n_treated"] == 3
    assert data["n_control"] == 3
    assert len(data["weights"]) == 6


# --- 5. Categorical Confounders & String Labels ---

def test_stored_dataset_iptw_categorical_confounders_and_string_treatment():
    """Verify IPTW modeling with string treatment labels and categorical confounder columns."""
    csv_data = (
        "arm,bp_drop,age,tier\n"
        "drug,18.0,45,low\n"
        "drug,22.0,52,high\n"
        "drug,20.0,48,medium\n"
        "drug,24.0,55,high\n"
        "placebo,8.0,44,low\n"
        "placebo,10.0,50,medium\n"
        "placebo,7.0,46,low\n"
        "placebo,12.0,56,high\n"
    ).encode("utf-8")

    upload_res = client.post("/api/dataset/upload", content=csv_data, headers={"X-Filename": "cat_iptw.csv"})
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={
            "treatment_col": "arm",
            "outcome_col": "bp_drop",
            "confounder_cols": ["age", "tier"],
            "treated_value": "drug",
            "control_value": "placebo",
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["estimator"] == "iptw_ate"
    assert data["n_total"] == 8
    assert data["treated_value"] == "drug"
    assert data["control_value"] == "placebo"
    assert "age" in data["numeric_confounders"]
    assert "tier" in data["categorical_confounders"]
    assert data["iptw_ate"] > 0


# --- 6. Hillstrom Benchmark Evaluation (64,000 Rows) ---

def test_stored_dataset_iptw_hillstrom_benchmark():
    """Evaluate full 64,000-row Hillstrom benchmark dataset without row sampling or silent truncation."""
    bm_res = client.post("/api/dataset/load-benchmark")
    assert bm_res.status_code == 200
    dataset_id = bm_res.json()["dataset_id"]

    analysis_res = client.post(
        f"/api/dataset/{dataset_id}/causal/iptw-ate",
        json={
            "treatment_col": "mens",
            "outcome_col": "spend",
            "confounder_cols": ["recency", "history"],
            "drop_missing": False,
        },
    )
    assert analysis_res.status_code == 200
    res = analysis_res.json()

    # Exact full sample accounting
    assert res["dataset_filename"] == "hillstrom.csv"
    assert res["total_dataset_rows"] == 64000
    assert res["n_total"] == 64000
    assert res["n_treated"] == 35266
    assert res["n_control"] == 28734
    assert res["n_treated"] + res["n_control"] == 64000
    assert res["rows_dropped"] == 0

    # Effect estimates
    assert math.isclose(res["iptw_ate"], 0.1782, abs_tol=0.01)
    assert math.isclose(res["unadjusted_ate"], 0.2600, abs_tol=0.01)
    assert res["ess_total"] > 50000
    assert len(res["weights"]) == 64000

    # Balance improvement
    bs = res["balance_summary"]
    assert bs["balance_improved"] is True
    assert bs["max_abs_smd_after"] < bs["max_abs_smd_before"]


# --- 7. Backward Compatibility: Existing Raw IPTW Endpoint ---

def test_raw_iptw_endpoint_compatibility():
    """Confirm that the existing Stage 6 raw IPTW endpoint continues to work identically."""
    raw_payload = {
        "data": [
            {"treatment": 1, "outcome": 10.0, "x": 1.0, "z": 20.0},
            {"treatment": 1, "outcome": 12.0, "x": 2.0, "z": 25.0},
            {"treatment": 1, "outcome": 14.0, "x": 3.0, "z": 30.0},
            {"treatment": 1, "outcome": 16.0, "x": 4.0, "z": 35.0},
            {"treatment": 0, "outcome": 5.0,  "x": 1.0, "z": 22.0},
            {"treatment": 0, "outcome": 7.0,  "x": 2.0, "z": 24.0},
            {"treatment": 0, "outcome": 9.0,  "x": 3.0, "z": 28.0},
            {"treatment": 0, "outcome": 11.0, "x": 4.0, "z": 32.0},
        ],
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": ["x", "z"],
        "treated_value": 1,
        "untreated_value": 0,
        "drop_missing": False,
        "extreme_weight_threshold": 20.0,
        "truncate_weights": False,
    }

    res = client.post("/api/causal/iptw-ate", json=raw_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["estimator"] == "iptw_ate"
    assert data["n_total"] == 8
    assert data["unadjusted_ate"] == pytest.approx(5.0)


# --- 8. Frontend HTML Section 5 Verification ---

def test_dashboard_html_contains_iptw_section():
    """Verify that the dashboard HTML serves Section 5 for IPTW ATE with required controls and hooks."""
    res = client.get("/dashboard")
    assert res.status_code == 200
    html = res.text

    # Section card and headings
    assert 'id="iptw-analysis-section"' in html
    assert 'id="iptw-analysis-heading"' in html
    assert "5. Inverse Probability of Treatment Weighting (IPTW)" in html

    # Form controls
    assert 'id="iptw-treatment-select"' in html
    assert 'id="iptw-outcome-select"' in html
    assert 'id="iptw-confounders-container"' in html
    assert 'id="iptw-extreme-weight-threshold-input"' in html
    assert 'id="iptw-max-weight-input"' in html
    assert 'id="iptw-truncate-checkbox"' in html
    assert 'id="iptw-drop-missing-checkbox"' in html
    assert 'id="run-iptw-btn"' in html

    # Results panel elements
    assert 'id="iptw-results-panel"' in html
    assert 'id="res-iptw-ate"' in html
    assert 'id="res-iptw-unadjusted-ate"' in html
    assert 'id="res-iptw-mean-treated"' in html
    assert 'id="res-iptw-mean-control"' in html
    assert 'id="res-iptw-ess-total"' in html
    assert 'id="res-iptw-weight-summary"' in html
    assert 'id="iptw-balance-table"' in html
    assert 'id="iptw-balance-tbody"' in html
    assert 'id="iptw-warnings-card"' in html
    assert 'id="res-iptw-interpretation-text"' in html
    assert 'id="res-iptw-assumptions-text"' in html
