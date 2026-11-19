"""Unit tests for Milestones 5 and 6 additions:
- PyMC Bayesian A/B testing
- Propensity Score Matching (PSM)
- Synthetic Control Method (SCM)
- DoWhy 4-step Causal Pipeline (Model, Identify, Estimate, Refute)
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.pymc_bayesian_engine import run_pymc_bayesian_ab
from src.propensity_score_matching import estimate_propensity_score_matching
from src.synthetic_control import estimate_synthetic_control
from src.dowhy_econml_engine import run_dowhy_causal_pipeline

client = TestClient(app)


def test_pymc_bayesian_engine_execution():
    res = run_pymc_bayesian_ab(
        conversions_a=50,
        visitors_a=1000,
        conversions_b=75,
        visitors_b=1000,
    )
    assert "engine" in res
    assert res["comparative_metrics"]["probability_b_beats_a"] > 0.90
    assert len(res["posteriors"]["group_a"]["credible_interval"]) == 2
    assert res["sample_counts"]["group_a"]["rate"] == 0.05
    assert res["sample_counts"]["group_b"]["rate"] == 0.075


def test_pymc_bayesian_api_endpoint():
    resp = client.post(
        "/api/v1/bayesian-ab/pymc",
        json={
            "conversions_a": 40,
            "visitors_a": 1000,
            "conversions_b": 65,
            "visitors_b": 1000,
            "alpha_prior_a": 1.0,
            "beta_prior_a": 1.0,
            "alpha_prior_b": 1.0,
            "beta_prior_b": 1.0,
            "draws": 500,
            "tune": 100,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "comparative_metrics" in data
    assert data["comparative_metrics"]["probability_b_beats_a"] > 0.85


def test_propensity_score_matching_engine():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(10, 2, n)
    prob_t = 1.0 / (1.0 + np.exp(-(0.5 * x1 + 0.1 * (x2 - 10))))
    t = (rng.uniform(size=n) < prob_t).astype(int)
    # True ATT is 3.0
    y = 20.0 + 3.0 * t + 1.5 * x1 + 0.8 * x2 + rng.normal(0, 0.5, n)

    df = pd.DataFrame({"t": t, "y": y, "x1": x1, "x2": x2})
    res = estimate_propensity_score_matching(df, "t", "y", ["x1", "x2"], caliper=0.5)

    assert res["method"] == "propensity_score_matching_nearest_neighbor"
    assert res["sample_summary"]["matched_pairs"] > 20
    assert abs(res["effect_estimates"]["att"] - 3.0) < 1.5
    assert "covariate_balance" in res


def test_propensity_score_matching_api():
    df_data = [
        {"t": 1, "y": 15.0, "x": 1.2},
        {"t": 1, "y": 18.0, "x": 2.1},
        {"t": 1, "y": 21.0, "x": 3.0},
        {"t": 0, "y": 10.0, "x": 1.1},
        {"t": 0, "y": 12.0, "x": 2.0},
        {"t": 0, "y": 14.0, "x": 2.9},
    ]
    resp = client.post(
        "/api/v1/causal/propensity-score-matching",
        json={
            "data": df_data,
            "treatment_col": "t",
            "outcome_col": "y",
            "confounder_cols": ["x"],
            "caliper": 1.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["effect_estimates"]["att"] > 0


def test_synthetic_control_engine():
    # 4 units over 6 time periods (periods 1-4 pre-treatment, 5-6 post-treatment)
    times = [1, 2, 3, 4, 5, 6]
    rows = []
    for t in times:
        # Treated unit receives +10 boost at periods 5 and 6
        effect = 10.0 if t >= 5 else 0.0
        rows.append({"unit": "Region_A", "time": t, "sales": 100.0 + 5.0 * t + effect})
        rows.append({"unit": "Donor_1", "time": t, "sales": 80.0 + 4.0 * t})
        rows.append({"unit": "Donor_2", "time": t, "sales": 120.0 + 6.0 * t})
        rows.append({"unit": "Donor_3", "time": t, "sales": 90.0 + 5.0 * t})

    df = pd.DataFrame(rows)
    res = estimate_synthetic_control(
        df,
        unit_col="unit",
        time_col="time",
        outcome_col="sales",
        treated_unit="Region_A",
        treatment_time=5,
    )

    assert res["method"] == "synthetic_control_abadie"
    assert res["configuration"]["treated_unit"] == "Region_A"
    assert res["fit_metrics"]["pre_treatment_rmspe"] < 2.0
    assert res["fit_metrics"]["average_post_treatment_effect"] > 5.0
    assert len(res["trajectory"]) == 6


def test_synthetic_control_api():
    rows = []
    for t in range(1, 6):
        effect = 5.0 if t >= 4 else 0.0
        rows.append({"unit": "U1", "time": t, "y": 10.0 + t + effect})
        rows.append({"unit": "U2", "time": t, "y": 9.0 + t})
        rows.append({"unit": "U3", "time": t, "y": 11.0 + t})

    resp = client.post(
        "/api/v1/causal/synthetic-control",
        json={
            "data": rows,
            "unit_col": "unit",
            "time_col": "time",
            "outcome_col": "y",
            "treated_unit": "U1",
            "treatment_time": 4,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "fit_metrics" in data
    assert data["fit_metrics"]["average_post_treatment_effect"] > 0


def test_dowhy_pipeline_engine():
    rng = np.random.default_rng(123)
    n = 150
    confounder = rng.normal(0, 1, n)
    t = (rng.uniform(size=n) < 0.5).astype(int)
    y = 50.0 + 4.0 * t + 2.0 * confounder + rng.normal(0, 0.5, n)

    df = pd.DataFrame({"treat": t, "outcome": y, "conf": confounder})
    res = run_dowhy_causal_pipeline(
        df,
        treatment_col="treat",
        outcome_col="outcome",
        common_causes=["conf"],
        run_refutations=True,
    )

    assert res["framework"] == "dowhy_econml_compatible"
    assert "step_1_model" in res
    assert "step_2_identify" in res
    assert abs(res["step_3_estimate"]["average_treatment_effect"] - 4.0) < 0.5
    assert "step_4_refute" in res
    assert "placebo_treatment_refuter" in res["step_4_refute"]
    assert res["robustness_summary"]["all_refutations_passed"] is True
