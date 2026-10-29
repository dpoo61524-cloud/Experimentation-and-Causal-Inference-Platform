"""Tests for Dataset-Based Difference-in-Differences (DiD) Workflow.

Covers:
- Safe stored dataset resolution and full-dataset row evaluation
- Numerical equivalence with the underlying causal DiD engine estimator
- Verifiable 2x2 group-period sample means and treatment effect calculation
- Custom string and integer group/period label resolution
- Auto-inferred label resolution for standard 0/1 encodings
- HTTP 404 for unknown or expired dataset IDs
- HTTP 422 for missing columns, conflicting column roles, and identical labels
- Rejection of designs with empty group-period cells (all 4 cells required)
- Missing-data policy (drop_missing=False rejected vs drop_missing=True evaluated with full row accounting)
- Non-numeric outcome rejection
- Empty dataset handling
- Field aliases compatibility (time_period_col, untreated_value, etc.)
- Preservation of Stage 6 raw DiD API endpoint contract
- Dashboard HTML integration hooks
"""

from pathlib import Path
import sys
import pytest
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api import app
from causal_engine import estimate_difference_in_differences

client = TestClient(app)


# --- 1. Basic Ingestion & Analysis Equivalence ---

def test_stored_dataset_did_success():
    """Verify that a stored CSV dataset is analyzed accurately with all rows evaluated in a 2x2 design."""
    # Treated Pre: [10.0, 12.0] -> mean = 11.0 (N=2)
    # Treated Post: [18.0, 22.0] -> mean = 20.0 (N=2) -> Delta_T = +9.0
    # Control Pre: [5.0, 7.0] -> mean = 6.0 (N=2)
    # Control Post: [8.0, 10.0] -> mean = 9.0 (N=2) -> Delta_C = +3.0
    # DiD = 9.0 - 3.0 = +6.0
    csv_data = (
        "treatment,time,outcome\n"
        "1,0,10.0\n"
        "1,0,12.0\n"
        "1,1,18.0\n"
        "1,1,22.0\n"
        "0,0,5.0\n"
        "0,0,7.0\n"
        "0,1,8.0\n"
        "0,1,10.0\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "did_test_dataset.csv"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    analysis_res = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "treatment",
            "time_col": "time",
            "outcome_col": "outcome",
        },
    )
    assert analysis_res.status_code == 200
    res = analysis_res.json()

    # Estimator metadata & row accounting
    assert res["estimator"] == "difference_in_differences"
    assert res["dataset_id"] == dataset_id
    assert res["dataset_filename"] == "did_test_dataset.csv"
    assert res["total_dataset_rows"] == 8
    assert res["n_total"] == 8
    assert res["rows_dropped"] == 0

    # Resolved labels
    assert res["treated_value"] == 1
    assert res["control_value"] == 0
    assert res["pre_value"] == 0
    assert res["post_value"] == 1

    # Cell means
    assert res["treatment_mean_pre"] == pytest.approx(11.0)
    assert res["treatment_mean_post"] == pytest.approx(20.0)
    assert res["control_mean_pre"] == pytest.approx(6.0)
    assert res["control_mean_post"] == pytest.approx(9.0)

    # Time changes & DiD
    assert res["treatment_change"] == pytest.approx(9.0)
    assert res["control_change"] == pytest.approx(3.0)
    assert res["did_estimate"] == pytest.approx(6.0)
    assert res["did"] == pytest.approx(6.0)
    assert res["ate"] == pytest.approx(6.0)

    # Cell observation counts
    assert res["cell_counts"] == {
        "treatment_pre": 2,
        "treatment_post": 2,
        "control_pre": 2,
        "control_post": 2,
    }
    assert res["n_treatment_pre"] == 2
    assert res["n_treatment_post"] == 2
    assert res["n_control_pre"] == 2
    assert res["n_control_post"] == 2
    assert res["n_treated"] == 4
    assert res["n_control"] == 4

    # Narrative text
    assert "assumptions" in res
    assert "interpretation" in res
    assert "Parallel Trends" in res["assumptions"]


def test_stored_dataset_did_numerical_equivalence_with_engine():
    """Verify exact numerical agreement between dataset endpoint and engine estimate_difference_in_differences."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 1, 0, 0, 0, 0],
        "period": [0, 0, 1, 1, 0, 0, 1, 1],
        "y": [100.0, 110.0, 140.0, 160.0, 50.0, 60.0, 70.0, 80.0],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_bytes,
        headers={"X-Filename": "equivalence_test.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    # Endpoint evaluation
    endpoint_res = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "t",
            "time_col": "period",
            "outcome_col": "y",
        },
    )
    assert endpoint_res.status_code == 200
    ep_data = endpoint_res.json()

    # Direct engine evaluation
    engine_data = estimate_difference_in_differences(
        data=df,
        treatment_col="t",
        time_col="period",
        outcome_col="y",
    )

    assert ep_data["did_estimate"] == pytest.approx(engine_data["did_estimate"])
    assert ep_data["treatment_mean_pre"] == pytest.approx(engine_data["treatment_mean_pre"])
    assert ep_data["treatment_mean_post"] == pytest.approx(engine_data["treatment_mean_post"])
    assert ep_data["control_mean_pre"] == pytest.approx(engine_data["control_mean_pre"])
    assert ep_data["control_mean_post"] == pytest.approx(engine_data["control_mean_post"])
    assert ep_data["treatment_change"] == pytest.approx(engine_data["treatment_change"])
    assert ep_data["control_change"] == pytest.approx(engine_data["control_change"])
    assert ep_data["n_total"] == engine_data["n_total"]
    assert ep_data["cell_counts"] == engine_data["cell_counts"]


# --- 2. Custom Labels and Value Overrides ---

def test_stored_dataset_did_custom_labels_and_string_encoding():
    """Verify custom string labels for cohorts and time periods."""
    csv_data = (
        "cohort,wave,outcome\n"
        "Intervention,Baseline_2022,50.0\n"
        "Intervention,Followup_2023,80.0\n"
        "Standard,Baseline_2022,40.0\n"
        "Standard,Followup_2023,55.0\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "string_labels_did.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "cohort",
            "time_col": "wave",
            "outcome_col": "outcome",
            "treated_value": "Intervention",
            "control_value": "Standard",
            "pre_value": "Baseline_2022",
            "post_value": "Followup_2023",
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["treated_value"] == "Intervention"
    assert data["control_value"] == "Standard"
    assert data["pre_value"] == "Baseline_2022"
    assert data["post_value"] == "Followup_2023"

    # Intervention: 50 -> 80 (+30); Standard: 40 -> 55 (+15) -> DiD = +15
    assert data["treatment_change"] == pytest.approx(30.0)
    assert data["control_change"] == pytest.approx(15.0)
    assert data["did_estimate"] == pytest.approx(15.0)


def test_stored_dataset_did_auto_inferred_labels():
    """Verify automatic label inference when no treated/pre values are explicitly supplied."""
    csv_data = (
        "t,period,y\n"
        "1,0,10.0\n"
        "1,1,20.0\n"
        "0,0,10.0\n"
        "0,1,12.0\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "auto_labels.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "t",
            "time_col": "period",
            "outcome_col": "y",
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["treated_value"] == 1
    assert data["control_value"] == 0
    assert data["pre_value"] == 0
    assert data["post_value"] == 1
    assert data["did_estimate"] == pytest.approx(8.0)


# --- 3. Validation & Error Handling ---

def test_stored_dataset_did_unknown_dataset_id_404():
    """Verify HTTP 404 when dataset_id does not exist."""
    res = client.post(
        "/api/dataset/non_existent_dataset_id/causal/difference-in-differences",
        json={
            "treatment_col": "t",
            "time_col": "time",
            "outcome_col": "y",
        },
    )
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_stored_dataset_did_missing_columns():
    """Verify HTTP 422 when specified columns do not exist in dataset."""
    csv_data = "treatment,time,outcome\n1,0,10\n".encode("utf-8")
    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "cols_test.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    # Missing treatment column
    res1 = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={"treatment_col": "non_existent_col", "time_col": "time", "outcome_col": "outcome"},
    )
    assert res1.status_code == 422
    assert "not found in dataset" in res1.json()["detail"]

    # Missing time column
    res2 = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={"treatment_col": "treatment", "time_col": "non_existent_time", "outcome_col": "outcome"},
    )
    assert res2.status_code == 422
    assert "not found in dataset" in res2.json()["detail"]

    # Missing outcome column
    res3 = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={"treatment_col": "treatment", "time_col": "time", "outcome_col": "non_existent_y"},
    )
    assert res3.status_code == 422
    assert "not found in dataset" in res3.json()["detail"]


def test_stored_dataset_did_role_conflicts():
    """Verify HTTP 422 when treatment, time, or outcome roles overlap."""
    csv_data = "treatment,time,outcome\n1,0,10\n".encode("utf-8")
    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "roles_test.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    # Treatment == Outcome
    res1 = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={"treatment_col": "treatment", "time_col": "time", "outcome_col": "treatment"},
    )
    assert res1.status_code == 422
    assert "cannot be the same column" in str(res1.json()["detail"]).lower()

    # Time == Outcome
    res2 = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={"treatment_col": "treatment", "time_col": "time", "outcome_col": "time"},
    )
    assert res2.status_code == 422
    assert "cannot be the same column" in str(res2.json()["detail"]).lower()

    # Treatment == Time
    res3 = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={"treatment_col": "treatment", "time_col": "treatment", "outcome_col": "outcome"},
    )
    assert res3.status_code == 422
    assert "cannot be the same column" in str(res3.json()["detail"]).lower()


def test_stored_dataset_did_identical_labels_rejected():
    """Verify HTTP 422 when treated/control or pre/post labels are identical."""
    csv_data = "treatment,time,outcome\n1,0,10\n".encode("utf-8")
    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "identical_labels.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    # treated == control
    res1 = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "treatment",
            "time_col": "time",
            "outcome_col": "outcome",
            "treated_value": 1,
            "control_value": 1,
        },
    )
    assert res1.status_code == 422
    assert "identical" in str(res1.json()["detail"]).lower()

    # pre == post
    res2 = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "treatment",
            "time_col": "time",
            "outcome_col": "outcome",
            "pre_value": 0,
            "post_value": 0,
        },
    )
    assert res2.status_code == 422
    assert "identical" in str(res2.json()["detail"]).lower()


def test_stored_dataset_did_missing_cell_combination_rejected():
    """Verify HTTP 422 when any of the four 2x2 cells is empty."""
    # Has: T,pre (1,0); C,pre (0,0); C,post (0,1). Missing: T,post (1,1)!
    csv_data = (
        "treatment,time,outcome\n"
        "1,0,10.0\n"
        "0,0,5.0\n"
        "0,1,8.0\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "empty_cell.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "treatment",
            "time_col": "time",
            "outcome_col": "outcome",
        },
    )
    assert res.status_code == 422
    assert "requires at least one observation in all four group-period cells" in res.json()["detail"]


# --- 4. Missing Value Policy & Accounting ---

def test_stored_dataset_did_missing_data_rejected_by_default():
    """Verify HTTP 422 when dataset contains missing values and drop_missing=False."""
    csv_data = (
        "treatment,time,outcome\n"
        "1,0,10.0\n"
        "1,1,20.0\n"
        "0,0,5.0\n"
        "0,1,NA\n"  # Missing outcome
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "missing_data_strict.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "treatment",
            "time_col": "time",
            "outcome_col": "outcome",
            "drop_missing": False,
        },
    )
    assert res.status_code == 422
    assert "missing values" in res.json()["detail"].lower()
    assert "drop_missing=true" in res.json()["detail"].lower()


def test_stored_dataset_did_explicit_drop_missing_row_accounting():
    """Verify that drop_missing=True drops incomplete rows with accurate accounting."""
    csv_data = (
        "treatment,time,outcome\n"
        "1,0,10.0\n"
        "1,0,12.0\n"
        "1,1,18.0\n"
        "1,1,22.0\n"
        "0,0,5.0\n"
        "0,0,7.0\n"
        "0,1,8.0\n"
        "0,1,10.0\n"
        "0,1,NA\n"      # Missing outcome
        "1,NA,15.0\n"    # Missing time
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "missing_data_dropped.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "treatment",
            "time_col": "time",
            "outcome_col": "outcome",
            "drop_missing": True,
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["total_dataset_rows"] == 10
    assert data["rows_dropped"] == 2
    assert data["n_total"] == 8
    assert data["n_total"] + data["rows_dropped"] == data["total_dataset_rows"]
    assert data["did_estimate"] == pytest.approx(6.0)


# --- 5. Data Types & Edge Cases ---

def test_stored_dataset_did_non_numeric_outcome_rejected():
    """Verify HTTP 422 when outcome column contains non-numeric values."""
    csv_data = (
        "treatment,time,outcome\n"
        "1,0,high\n"
        "1,1,higher\n"
        "0,0,low\n"
        "0,1,medium\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "text_outcome.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "treatment",
            "time_col": "time",
            "outcome_col": "outcome",
        },
    )
    assert res.status_code == 422
    assert "must be numeric" in res.json()["detail"].lower() or "non-numeric" in res.json()["detail"].lower()


def test_stored_dataset_did_empty_dataset_rejected():
    """Verify HTTP 422 when dataset has 0 rows."""
    csv_data = "treatment,time,outcome\n".encode("utf-8")
    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "empty_table.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "treatment",
            "time_col": "time",
            "outcome_col": "outcome",
        },
    )
    assert res.status_code == 422
    assert "empty" in res.json()["detail"].lower()


def test_stored_dataset_did_field_aliases_support():
    """Verify that aliases like time_period_col, untreated_value, pre_period_value, and post_period_value work seamlessly."""
    csv_data = (
        "group,wave,val\n"
        "1,0,10.0\n"
        "1,1,20.0\n"
        "0,0,10.0\n"
        "0,1,12.0\n"
    ).encode("utf-8")

    upload_res = client.post(
        "/api/dataset/upload",
        content=csv_data,
        headers={"X-Filename": "aliases_did.csv"},
    )
    dataset_id = upload_res.json()["dataset_id"]

    res = client.post(
        f"/api/dataset/{dataset_id}/causal/difference-in-differences",
        json={
            "treatment_col": "group",
            "time_period_col": "wave",
            "outcome_col": "val",
            "treated_value": 1,
            "untreated_value": 0,
            "pre_period_value": 0,
            "post_period_value": 1,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["time_col"] == "wave"
    assert data["time_period_col"] == "wave"
    assert data["did_estimate"] == pytest.approx(8.0)


# --- 6. Contract Compatibility & Dashboard HTML ---

def test_raw_did_endpoint_compatibility():
    """Verify that Stage 6 raw /api/causal/difference-in-differences endpoint still operates without breakage."""
    raw_payload = {
        "data": [
            {"treatment": 1, "time": 0, "outcome": 10.0},
            {"treatment": 1, "time": 0, "outcome": 12.0},
            {"treatment": 1, "time": 1, "outcome": 18.0},
            {"treatment": 1, "time": 1, "outcome": 22.0},
            {"treatment": 0, "time": 0, "outcome": 5.0},
            {"treatment": 0, "time": 0, "outcome": 7.0},
            {"treatment": 0, "time": 1, "outcome": 8.0},
            {"treatment": 0, "time": 1, "outcome": 10.0},
        ],
        "treatment_col": "treatment",
        "time_col": "time",
        "outcome_col": "outcome",
    }
    res = client.post("/api/causal/difference-in-differences", json=raw_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["estimator"] == "difference_in_differences"
    assert data["did_estimate"] == pytest.approx(6.0)
    assert data["n_total"] == 8


def test_dashboard_html_contains_did_section():
    """Verify that the dashboard HTML includes all Section 6 Difference-in-Differences elements."""
    res = client.get("/dashboard")
    assert res.status_code == 200
    html = res.text

    assert 'id="did-analysis-section"' in html
    assert 'id="did-analysis-heading"' in html
    assert '6. Difference-in-Differences (DiD)' in html
    assert 'id="did-status-banner"' in html
    assert 'id="did-unloaded-notice"' in html
    assert 'id="did-form-container"' in html
    assert 'id="did-treatment-select"' in html
    assert 'id="did-time-select"' in html
    assert 'id="did-outcome-select"' in html
    assert 'id="did-treated-val-input"' in html
    assert 'id="did-control-val-input"' in html
    assert 'id="did-pre-val-input"' in html
    assert 'id="did-post-val-input"' in html
    assert 'id="did-drop-missing-checkbox"' in html
    assert 'id="run-did-btn"' in html
    assert 'id="did-results-panel"' in html
    assert 'id="res-did-estimate"' in html
    assert 'id="did-2x2-table"' in html
    assert 'id="res-did-interpretation-text"' in html
    assert 'id="res-did-assumptions-text"' in html
    assert 'Available Causal Analysis Methods' in html
