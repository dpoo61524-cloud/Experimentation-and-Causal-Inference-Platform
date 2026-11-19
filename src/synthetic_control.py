"""Synthetic Control Method (SCM) Engine.

Implements Abadie, Diamond, and Hainmueller (2003, 2010, 2015) constrained
optimization to estimate counterfactual trajectories for aggregated treatment units,
computing donor weights, pre-intervention RMSPE fit, post-intervention treatment effects,
and in-space placebo permutation significance tests.
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def estimate_synthetic_control(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treated_unit: Any,
    treatment_time: Any,
) -> Dict[str, Any]:
    """Estimate Synthetic Control counterfactual trajectory and treatment gap."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")
    for c in [unit_col, time_col, outcome_col]:
        if c not in data.columns:
            raise ValueError(f"Required column '{c}' not found in data.")

    df = data[[unit_col, time_col, outcome_col]].dropna().copy()
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    if df[outcome_col].isna().any():
        raise ValueError("Outcome column must be numeric.")

    # Validate units
    units = df[unit_col].unique()
    if treated_unit not in units:
        raise ValueError(f"Treated unit '{treated_unit}' not present in {unit_col} column.")
    donor_units = [u for u in units if u != treated_unit]
    if len(donor_units) < 1:
        raise ValueError("Synthetic control requires at least one donor unit in the control pool.")

    # Pivot table: time index x units columns
    pivot_df = df.pivot_table(index=time_col, columns=unit_col, values=outcome_col, aggfunc="mean")
    times = sorted(pivot_df.index.tolist())

    pre_times = [t for t in times if t < treatment_time]
    post_times = [t for t in times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError("Synthetic control requires at least 2 pre-treatment time periods to optimize donor weights.")
    if len(post_times) < 1:
        raise ValueError("Synthetic control requires at least 1 post-treatment time period to evaluate effect.")

    # Extract target and donor matrices
    y1_pre = pivot_df.loc[pre_times, treated_unit].values
    y0_pre = pivot_df.loc[pre_times, donor_units].values

    y1_post = pivot_df.loc[post_times, treated_unit].values
    y0_post = pivot_df.loc[post_times, donor_units].values

    # Check for NaNs
    if np.isnan(y1_pre).any() or np.isnan(y0_pre).any():
        raise ValueError("Pre-treatment series contains missing values for target or donor units.")

    n_donors = len(donor_units)

    # Quadratic loss objective: || Y1_pre - Y0_pre * W ||^2
    def loss(w):
        diff = y1_pre - np.dot(y0_pre, w)
        return np.sum(diff ** 2)

    # Constraints: sum(w) = 1, w_j >= 0
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 1.0) for _ in range(n_donors)]
    w0 = np.ones(n_donors) / n_donors

    res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons)
    weights = res.x
    # Normalize small numerical negative precision artifacts
    weights = np.maximum(weights, 0.0)
    weights = weights / np.sum(weights)

    donor_weight_map = {donor_units[i]: float(weights[i]) for i in range(n_donors) if weights[i] > 1e-4}

    # Generate synthetic series across all periods
    y1_all = pivot_df.loc[times, treated_unit].values
    y0_all = pivot_df.loc[times, donor_units].values
    y_synth_all = np.dot(y0_all, weights)

    gaps = y1_all - y_synth_all

    pre_mask = np.array([t < treatment_time for t in times])
    post_mask = ~pre_mask

    pre_rmspe = float(np.sqrt(np.mean(gaps[pre_mask] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gaps[post_mask] ** 2)))
    avg_post_effect = float(np.mean(gaps[post_mask]))
    rmspe_ratio = float(post_rmspe / pre_rmspe) if pre_rmspe > 1e-9 else 0.0

    # In-Space Placebo Permutation Test (Run for each donor)
    placebo_ratios = []
    for d_idx, d_unit in enumerate(donor_units):
        placebo_donors = [u for u in units if u != d_unit]
        y_d_pre = pivot_df.loc[pre_times, d_unit].values
        y_d_donors_pre = pivot_df.loc[pre_times, placebo_donors].values
        y_d_post = pivot_df.loc[post_times, d_unit].values
        y_d_donors_post = pivot_df.loc[post_times, placebo_donors].values

        if np.isnan(y_d_pre).any() or np.isnan(y_d_donors_pre).any():
            continue

        def p_loss(w):
            return np.sum((y_d_pre - np.dot(y_d_donors_pre, w)) ** 2)

        p_res = minimize(
            p_loss,
            np.ones(len(placebo_donors)) / len(placebo_donors),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(placebo_donors),
            constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0}),
        )
        p_w = np.maximum(p_res.x, 0.0)
        p_w /= np.sum(p_w)

        p_synth_pre = np.dot(y_d_donors_pre, p_w)
        p_synth_post = np.dot(y_d_donors_post, p_w)

        d_pre_rmspe = np.sqrt(np.mean((y_d_pre - p_synth_pre) ** 2))
        d_post_rmspe = np.sqrt(np.mean((y_d_post - p_synth_post) ** 2))
        if d_pre_rmspe > 1e-9:
            placebo_ratios.append(float(d_post_rmspe / d_pre_rmspe))

    # Pseudo p-value: proportion of placebos with ratio >= treated ratio
    if placebo_ratios:
        all_ratios = [rmspe_ratio] + placebo_ratios
        pseudo_p_value = float(np.mean(np.array(all_ratios) >= rmspe_ratio))
    else:
        pseudo_p_value = None

    trajectory = []
    for i, t in enumerate(times):
        trajectory.append({
            "time": str(t),
            "actual": float(y1_all[i]),
            "synthetic": float(y_synth_all[i]),
            "gap": float(gaps[i]),
            "is_post_treatment": bool(t >= treatment_time),
        })

    return {
        "method": "synthetic_control_abadie",
        "configuration": {
            "treated_unit": str(treated_unit),
            "treatment_time": str(treatment_time),
            "donor_pool_count": n_donors,
            "pre_treatment_periods": len(pre_times),
            "post_treatment_periods": len(post_times),
        },
        "donor_weights": donor_weight_map,
        "fit_metrics": {
            "pre_treatment_rmspe": pre_rmspe,
            "post_treatment_rmspe": post_rmspe,
            "rmspe_ratio": rmspe_ratio,
            "average_post_treatment_effect": avg_post_effect,
            "placebo_pseudo_p_value": pseudo_p_value,
        },
        "trajectory": trajectory,
        "conclusion": (
            f"Synthetic control established pre-treatment RMSPE fit of {pre_rmspe:.4f}. "
            f"Average treatment effect in the post-intervention period is {avg_post_effect:.4f} "
            f"(Post/Pre RMSPE Ratio: {rmspe_ratio:.2f}"
            + (f", Placebo p-value: {pseudo_p_value:.3f})" if pseudo_p_value is not None else ")")
            + "."
        ),
    }
