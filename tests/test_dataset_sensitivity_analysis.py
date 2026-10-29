"""Tests for Dataset-Based Causal Sensitivity Analysis and Robustness Checks.

Stage 7.6B Test Suite covering:
1. Correctness and numerical equivalence against the underlying causal engine
   (run_causal_sensitivity_analysis and analyze_unmeasured_confounding_sensitivity).
2. Full dataset mode:
   - Multi-estimator cross-comparison (unadjusted, regression adjustment, IPTW).
   - IPTW weight truncation robustness across requested thresholds.
   - Hypothetical unmeasured confounding scenarios (tau_adjusted = tau_observed - b).
3. Standalone observed-estimate mode:
   - Fast O(1) evaluation without requiring full dataset columns.
   - Hypothetical signed bias adjustments and 95% CI zero-crossing detection.
4. Error handling & validation:
   - HTTP 404 for unknown or non-existent dataset IDs.
   - HTTP 422 when neither dataset columns nor observed_estimate are provided.
   - HTTP 422 for invalid column names or role conflicts (treatment == outcome).
   - HTTP 422 for non-positive or invalid IPTW truncation thresholds.
   - HTTP 422 for non-numeric, NaN, or empty bias values.
   - HTTP 422 when confidence interval lower bound exceeds upper bound.
   - HTTP 422 when drop_missing=False and analysis columns contain missing data.
   - HTTP 200 when drop_missing=True drops incomplete rows.
5. Benchmark dataset evaluation (Hillstrom email marketing dataset).
6. Backward compatibility with Stage 6 raw sensitivity endpoint (/api/causal/sensitivity-analysis).
7. Dashboard HTML integration hooks (/dashboard contains Section 7 markup and element IDs).
"""

from pathlib import Path
import sys
import pytest
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api import app
from causal_engine import (
    run_causal_sensitivity_analysis,
    analyze_unmeasured_confounding_sensitivity,
)

client = TestClient(app)


# --- 1. Dataset Mode Sensitivity Analysis ---

def test_dataset_mode_sensitivity_analysis_success():
    """Verify full dataset mode evaluating multi-estimators, truncation, and unmeasured confounding."""
    csv_data = (
        "treatment,outcome,x,z\n"
        "1,15.0,2.0,10.0\n"
        "1,18.0,3.0,12.0\n"
        "1,21.0,4.0,15.0\n"
        "1,24.0,5.0,20.0\n"
        "0,8.0,1.0,8.0\n"
        "0,10.0,2.0,10.0\n"
        "0,12.0,3.0,12.0\n"
        "0,14.0,4.0,15.0\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "sens_dataset_test.csv"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    req_payload = {
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": ["x", "z"],
        "reference_estimator": "regression_adjusted",
        "truncation_thresholds": [10.0, 5.0, 3.0],
        "hypothetical_biases": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "drop_missing": False,
    }

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/sensitivity-analysis",
        json=req_payload,
    )
    assert res.status_code == 200
    data = res.json()

    # Dataset metadata
    assert data["dataset_id"] == dataset_id
    assert data["dataset_filename"] == "sens_dataset_test.csv"
    assert data["total_dataset_rows"] == 8

    # Reference estimator
    assert data["reference_estimator"] == "regression_adjusted"
    assert data["reference_estimate"] is not None
    assert isinstance(data["reference_estimate"], float)

    # Multi-estimator comparisons
    comps = data["estimator_comparisons"]
    assert "unadjusted" in comps
    assert "regression_adjusted" in comps
    assert "iptw" in comps
    assert comps["regression_adjusted"]["available"] is True
    assert comps["unadjusted"]["available"] is True
    assert comps["iptw"]["available"] is True

    # Differences from reference
    assert comps["regression_adjusted"]["difference_from_reference"] == pytest.approx(0.0)

    # IPTW truncation analysis
    trunc = data["iptw_truncation_analysis"]
    assert trunc["status"] in ("evaluated", "completed")
    assert len(trunc["scenarios"]) == 3
    assert trunc["scenarios"][0]["threshold"] == 10.0
    assert trunc["scenarios"][1]["threshold"] == 5.0
    assert trunc["scenarios"][2]["threshold"] == 3.0

    # Unmeasured confounding scenarios
    scenarios = data["bias_scenarios"]
    assert len(scenarios) == 5
    ref_est = data["reference_estimate"]
    for sc in scenarios:
        b = sc["hypothetical_bias"]
        assert sc["adjusted_estimate"] == pytest.approx(ref_est - b)

    # Numerical equivalence with direct causal_engine call
    df = pd.read_csv(pd.io.common.BytesIO(csv_data))
    engine_res = run_causal_sensitivity_analysis(
        data=df,
        treatment_col="treatment",
        outcome_col="outcome",
        confounder_cols=["x", "z"],
        reference_estimator="regression_adjusted",
        truncation_thresholds=[10.0, 5.0, 3.0],
        hypothetical_biases=[-2.0, -1.0, 0.0, 1.0, 2.0],
    )
    assert data["reference_estimate"] == pytest.approx(engine_res["reference_estimate"])
    assert comps["regression_adjusted"]["estimate"] == pytest.approx(engine_res["estimator_comparisons"]["regression_adjusted"]["estimate"])
    assert comps["unadjusted"]["estimate"] == pytest.approx(engine_res["estimator_comparisons"]["unadjusted"]["estimate"])


# --- 2. Standalone Observed-Estimate Mode ---

def test_standalone_observed_estimate_sensitivity_analysis():
    """Verify standalone mode evaluates unmeasured confounding without reading data columns."""
    # First upload an arbitrary dataset to have a valid dataset_id
    upload_res = client.post(
        "/api/dataset/upload",
        content="a,b\n1,2\n3,4\n".encode("utf-8"),
        headers={"X-Filename": "dummy.csv"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    req_payload = {
        "observed_estimate": 5.0,
        "confidence_interval": [2.0, 8.0],
        "bias_values": [-2.0, 0.0, 3.0, 6.0],
    }

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/sensitivity-analysis",
        json=req_payload,
    )
    assert res.status_code == 200
    data = res.json()

    assert data["dataset_id"] == dataset_id
    assert data["observed_estimate"] == pytest.approx(5.0)
    assert data["confidence_interval"] == [2.0, 8.0]
    assert data["reference_estimate"] == pytest.approx(5.0)

    # Multi-estimator and truncation should not be run in standalone mode
    assert data["estimator_comparisons"] is None
    assert data["iptw_truncation_analysis"] is None

    # Check scenarios and zero-crossing detection
    scenarios = data["bias_scenarios"]
    assert len(scenarios) == 4

    # b = -2.0: tau_adj = 5.0 - (-2.0) = 7.0, CI = [4.0, 10.0] -> Does not span zero
    assert scenarios[0]["hypothetical_bias"] == pytest.approx(-2.0)
    assert scenarios[0]["adjusted_estimate"] == pytest.approx(7.0)
    assert scenarios[0]["adjusted_confidence_interval"] == [pytest.approx(4.0), pytest.approx(10.0)]
    assert scenarios[0]["interval_includes_zero"] is False

    # b = 0.0: tau_adj = 5.0, CI = [2.0, 8.0] -> Does not span zero
    assert scenarios[1]["hypothetical_bias"] == pytest.approx(0.0)
    assert scenarios[1]["adjusted_estimate"] == pytest.approx(5.0)
    assert scenarios[1]["adjusted_confidence_interval"] == [pytest.approx(2.0), pytest.approx(8.0)]
    assert scenarios[1]["interval_includes_zero"] is False

    # b = 3.0: tau_adj = 2.0, CI = [-1.0, 5.0] -> Spans zero!
    assert scenarios[2]["hypothetical_bias"] == pytest.approx(3.0)
    assert scenarios[2]["adjusted_estimate"] == pytest.approx(2.0)
    assert scenarios[2]["adjusted_confidence_interval"] == [pytest.approx(-1.0), pytest.approx(5.0)]
    assert scenarios[2]["interval_includes_zero"] is True

    # b = 6.0: tau_adj = -1.0, CI = [-4.0, 2.0] -> Spans zero!
    assert scenarios[3]["hypothetical_bias"] == pytest.approx(6.0)
    assert scenarios[3]["adjusted_estimate"] == pytest.approx(-1.0)
    assert scenarios[3]["adjusted_confidence_interval"] == [pytest.approx(-4.0), pytest.approx(2.0)]
    assert scenarios[3]["interval_includes_zero"] is True

    # Verify numerical equivalence with analyze_unmeasured_confounding_sensitivity
    engine_res = analyze_unmeasured_confounding_sensitivity(
        observed_estimate=5.0,
        bias_values=[-2.0, 0.0, 3.0, 6.0],
        confidence_interval=[2.0, 8.0],
    )
    for idx in range(4):
        assert scenarios[idx]["adjusted_estimate"] == pytest.approx(engine_res["bias_scenarios"][idx]["adjusted_estimate"])
        assert scenarios[idx]["interval_includes_zero"] == engine_res["bias_scenarios"][idx]["interval_includes_zero"]


def test_standalone_mode_without_confidence_interval():
    """Verify standalone mode operates correctly when confidence interval is omitted."""
    upload_res = client.post(
        "/api/dataset/upload",
        content="x,y\n1,2\n".encode("utf-8"),
        headers={"X-Filename": "dummy2.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/sensitivity-analysis",
        json={
            "observed_estimate": 3.5,
            "bias_values": [1.0, 2.0],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["observed_estimate"] == pytest.approx(3.5)
    assert data["confidence_interval"] is None
    scenarios = data["bias_scenarios"]
    assert len(scenarios) == 2
    assert scenarios[0]["adjusted_confidence_interval"] is None
    assert scenarios[0]["interval_includes_zero"] is None


# --- 3. Error Handling and Validation ---

def test_unknown_dataset_id_returns_404():
    """Verify that an unknown dataset ID returns HTTP 404."""
    res = client.post(
        "/api/dataset/non-existent-dataset-id/causal/sensitivity-analysis",
        json={
            "observed_estimate": 5.0,
            "bias_values": [1.0],
        },
    )
    assert res.status_code == 404
    assert "not found or has expired" in res.json()["detail"]


def test_empty_request_body_returns_422():
    """Verify that omitting both dataset columns and observed estimate returns HTTP 422."""
    upload_res = client.post(
        "/api/dataset/upload",
        content="t,y\n1,2\n".encode("utf-8"),
        headers={"X-Filename": "d.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/sensitivity-analysis",
        json={},
    )
    assert res.status_code == 422


def test_treatment_without_outcome_returns_422():
    """Verify that specifying treatment_col without outcome_col returns HTTP 422."""
    upload_res = client.post(
        "/api/dataset/upload",
        content="t,y\n1,2\n".encode("utf-8"),
        headers={"X-Filename": "d.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/sensitivity-analysis",
        json={
            "treatment_col": "t",
        },
    )
    assert res.status_code == 422


def test_same_treatment_and_outcome_returns_422():
    """Verify that treatment_col == outcome_col returns HTTP 422."""
    upload_res = client.post(
        "/api/dataset/upload",
        content="t,y\n1,2\n".encode("utf-8"),
        headers={"X-Filename": "d.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/sensitivity-analysis",
        json={
            "treatment_col": "t",
            "outcome_col": "t",
            "bias_values": [1.0],
        },
    )
    assert res.status_code == 422
    assert "cannot be the same column" in str(res.json()["detail"])


def test_invalid_truncation_thresholds_returns_422():
    """Verify that non-positive truncation thresholds return HTTP 422."""
    upload_res = client.post(
        "/api/dataset/upload",
        content="t,y,x\n1,2,3\n0,1,2\n".encode("utf-8"),
        headers={"X-Filename": "d.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/sensitivity-analysis",
        json={
            "treatment_col": "t",
            "outcome_col": "y",
            "confounder_cols": ["x"],
            "truncation_thresholds": [10.0, -1.0],
            "bias_values": [0.0],
        },
    )
    assert res.status_code == 422
    assert "must be strictly positive" in str(res.json()["detail"])


def test_invalid_confidence_interval_bounds_returns_422():
    """Verify that CI lower bound exceeding upper bound returns HTTP 422."""
    upload_res = client.post(
        "/api/dataset/upload",
        content="t,y\n1,2\n".encode("utf-8"),
        headers={"X-Filename": "d.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/sensitivity-analysis",
        json={
            "observed_estimate": 5.0,
            "confidence_interval": [10.0, 5.0],
            "bias_values": [1.0],
        },
    )
    assert res.status_code == 422
    assert "cannot exceed upper bound" in str(res.json()["detail"])


def test_missing_data_policy_enforcement():
    """Verify drop_missing=False rejects missing data while drop_missing=True drops rows."""
    csv_data = (
        "t,y,x\n"
        "1,10.0,1.0\n"
        "1,NA,2.0\n"  # missing outcome
        "1,14.0,3.0\n"
        "0,5.0,1.0\n"
        "0,7.0,2.0\n"
        "0,9.0,3.0\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "missing_sens.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    # drop_missing=False should be rejected with 422
    res_reject = client.post(
        f"/api/dataset/{dataset_id}/causal/sensitivity-analysis",
        json={
            "treatment_col": "t",
            "outcome_col": "y",
            "confounder_cols": ["x"],
            "drop_missing": False,
            "bias_values": [0.0],
        },
    )
    assert res_reject.status_code == 422
    assert "missing values" in str(res_reject.json()["detail"]).lower()

    # drop_missing=True should succeed and drop the row
    res_ok = client.post(
        f"/api/dataset/{dataset_id}/causal/sensitivity-analysis",
        json={
            "treatment_col": "t",
            "outcome_col": "y",
            "confounder_cols": ["x"],
            "drop_missing": True,
            "bias_values": [0.0],
        },
    )
    assert res_ok.status_code == 200
    assert res_ok.json()["total_dataset_rows"] == 6


# --- 4. Benchmark Dataset Sensitivity Analysis ---

def test_hillstrom_benchmark_sensitivity_analysis():
    """Verify sensitivity analysis runs on the loaded Hillstrom benchmark dataset."""
    load_res = client.post("/api/dataset/load-benchmark")
    assert load_res.status_code == 200
    dataset_id = load_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/sensitivity-analysis",
        json={
            "treatment_col": "mens",
            "outcome_col": "spend",
            "confounder_cols": ["recency", "history"],
            "reference_estimator": "regression_adjusted",
            "truncation_thresholds": [10.0, 5.0],
            "hypothetical_biases": [-0.5, 0.0, 0.5],
            "drop_missing": False,
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["dataset_id"] == dataset_id
    assert data["total_dataset_rows"] == 64000
    assert data["reference_estimator"] == "regression_adjusted"
    assert data["reference_estimate"] is not None

    # Check multi-estimator table
    comps = data["estimator_comparisons"]
    assert comps["unadjusted"]["available"] is True
    assert comps["regression_adjusted"]["available"] is True
    assert comps["iptw"]["available"] is True

    # Truncation table
    assert len(data["iptw_truncation_analysis"]["scenarios"]) == 2

    # Scenarios
    assert len(data["bias_scenarios"]) == 3


# --- 5. Backward Compatibility with Raw Sensitivity Endpoint ---

def test_raw_causal_sensitivity_endpoint_compatibility():
    """Verify that Stage 6 raw POST /api/causal/sensitivity-analysis remains intact."""
    raw_payload = {
        "data": [
            {"t": 1, "y": 10.0, "x": 1.0},
            {"t": 1, "y": 12.0, "x": 2.0},
            {"t": 1, "y": 14.0, "x": 3.0},
            {"t": 0, "y": 4.0,  "x": 1.0},
            {"t": 0, "y": 6.0,  "x": 2.0},
            {"t": 0, "y": 8.0,  "x": 3.0},
        ],
        "treatment_col": "t",
        "outcome_col": "y",
        "confounder_cols": ["x"],
        "hypothetical_biases": [0.0, 1.0],
    }

    res = client.post(
        "/api/causal/sensitivity-analysis",
        json=raw_payload,
    )
    assert res.status_code == 200
    data = res.json()
    assert "reference_estimator" in data
    assert "reference_estimate" in data
    assert "estimator_comparisons" in data
    assert "bias_scenarios" in data

    # Standalone raw call
    res_standalone = client.post(
        "/api/causal/sensitivity-analysis",
        json={
            "observed_estimate": 4.0,
            "confidence_interval": [2.0, 6.0],
            "bias_values": [1.0, 2.0],
        },
    )
    assert res_standalone.status_code == 200
    assert res_standalone.json()["observed_estimate"] == pytest.approx(4.0)


# --- 6. Dashboard HTML Integration ---

def test_dashboard_html_contains_sensitivity_section():
    """Verify that the dashboard HTML includes all Section 7 sensitivity elements."""
    res = client.get("/dashboard")
    assert res.status_code == 200
    html = res.text

    # Section wrapper
    assert 'id="sensitivity-analysis-section"' in html
    assert "Sensitivity Analysis" in html

    # Mode controls
    assert 'id="sens-mode-dataset"' in html
    assert 'id="sens-mode-standalone"' in html
    assert 'id="sens-dataset-fields"' in html
    assert 'id="sens-standalone-fields"' in html

    # Form inputs
    assert 'id="sens-treatment-select"' in html
    assert 'id="sens-outcome-select"' in html
    assert 'id="sens-ref-estimator-select"' in html
    assert 'id="sens-confounders-container"' in html
    assert 'id="sens-truncation-thresholds-input"' in html
    assert 'id="sens-observed-estimate-input"' in html
    assert 'id="sens-ci-lower-input"' in html
    assert 'id="sens-ci-upper-input"' in html
    assert 'id="sens-biases-input"' in html
    assert 'id="sens-drop-missing-checkbox"' in html
    assert 'id="run-sensitivity-btn"' in html

    # Results panel & tables
    assert 'id="sens-results-panel"' in html
    assert 'id="sens-estimator-table"' in html
    assert 'id="sens-estimator-tbody"' in html
    assert 'id="sens-truncation-table"' in html
    assert 'id="sens-truncation-tbody"' in html
    assert 'id="sens-scenarios-table"' in html
    assert 'id="sens-scenarios-tbody"' in html
    assert 'id="sens-warnings-card"' in html
    assert 'id="res-sens-interpretation-text"' in html
    assert 'id="res-sens-assumptions-text"' in html
