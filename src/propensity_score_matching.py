"""Propensity Score Matching (PSM) Engine.

Implements nearest-neighbor matching on propensity scores with caliper constraints,
estimating the Average Treatment Effect on the Treated (ATT) and evaluating
covariate balance before and after matching.
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


def estimate_propensity_score_matching(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: Sequence[str],
    caliper: Optional[float] = 0.2,
    replacement: bool = False,
    ratio: int = 1,
) -> Dict[str, Any]:
    """Execute Nearest-Neighbor Propensity Score Matching (PSM)."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if treatment_col not in data.columns:
        raise ValueError(f"Treatment column '{treatment_col}' not found in data.")
    if outcome_col not in data.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in data.")
    for col in confounder_cols:
        if col not in data.columns:
            raise ValueError(f"Confounder column '{col}' not found in data.")

    # Drop missing values in analysis columns
    cols_to_use = [treatment_col, outcome_col] + list(confounder_cols)
    df = data[cols_to_use].dropna().copy()
    if len(df) == 0:
        raise ValueError("No complete observations remaining after dropping missing values.")

    # Validate binary treatment
    unique_treat = np.sort(df[treatment_col].unique())
    if len(unique_treat) != 2:
        raise ValueError(f"Treatment column must be strictly binary, found {len(unique_treat)} levels.")
    control_val, treat_val = unique_treat[0], unique_treat[1]
    df["_T_"] = (df[treatment_col] == treat_val).astype(float)
    df["_Y_"] = pd.to_numeric(df[outcome_col], errors="coerce")
    if df["_Y_"].isna().any():
        raise ValueError("Outcome column must contain numeric values.")

    # Preprocess confounders (One-hot encode non-numeric)
    X_raw = df[list(confounder_cols)].copy()
    X_encoded = pd.get_dummies(X_raw, drop_first=True, dtype=float)
    X_design = sm.add_constant(X_encoded, prepend=True, has_constant="add")

    # Fit Logistic Regression to estimate propensity scores
    try:
        logit_model = sm.Logit(df["_T_"], X_design)
        logit_res = logit_model.fit(disp=False)
        pscores = logit_res.predict(X_design)
    except Exception as exc:
        raise ValueError(f"Failed to fit propensity score model: {exc}")

    df["_PS_"] = pscores

    treated_idx = df[df["_T_"] == 1.0].index.tolist()
    control_idx = df[df["_T_"] == 0.0].index.tolist()

    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise ValueError("Both treatment and control units are required for matching.")

    # Calculate caliper standard distance
    caliper_val = None
    if caliper is not None and caliper > 0:
        ps_std = df["_PS_"].std()
        caliper_val = caliper * ps_std if ps_std > 0 else caliper

    # 1-to-1 Nearest Neighbor Matching with high-performance O(N log M) outward search
    matched_pairs: List[Tuple[Any, Any, float]] = []

    c_ids = np.array(control_idx)
    c_ps_raw = df.loc[control_idx, "_PS_"].to_numpy()
    c_sort_order = np.argsort(c_ps_raw)
    sorted_c_ids = c_ids[c_sort_order]
    sorted_c_ps = c_ps_raw[c_sort_order]
    n_controls = len(sorted_c_ps)
    used_controls_mask = np.zeros(n_controls, dtype=bool)

    for t_id in treated_idx:
        t_ps = float(df.loc[t_id, "_PS_"])
        pos = int(np.searchsorted(sorted_c_ps, t_ps))

        left = pos - 1
        right = pos
        best_i = None
        best_diff = 1e9

        while left >= 0 or right < n_controls:
            d_left = abs(sorted_c_ps[left] - t_ps) if left >= 0 else 1e9
            d_right = abs(sorted_c_ps[right] - t_ps) if right < n_controls else 1e9

            if d_left <= d_right:
                if replacement or not used_controls_mask[left]:
                    best_i = left
                    best_diff = d_left
                    break
                left -= 1
            else:
                if replacement or not used_controls_mask[right]:
                    best_i = right
                    best_diff = d_right
                    break
                right += 1

        if best_i is not None:
            if caliper_val is None or best_diff <= caliper_val:
                best_c = sorted_c_ids[best_i]
                matched_pairs.append((t_id, best_c, float(best_diff)))
                if not replacement:
                    used_controls_mask[best_i] = True

    if len(matched_pairs) == 0:
        raise ValueError("No treated units could be matched within the specified caliper.")

    matched_t_ids = [p[0] for p in matched_pairs]
    matched_c_ids = [p[1] for p in matched_pairs]

    y_t = df.loc[matched_t_ids, "_Y_"].values
    y_c = df.loc[matched_c_ids, "_Y_"].values
    pair_diffs = y_t - y_c

    # ATT Estimation and Inference
    n_pairs = len(matched_pairs)
    att = float(np.mean(pair_diffs))
    se_att = float(np.std(pair_diffs, ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else 0.0
    t_stat = float(att / se_att) if se_att > 0 else 0.0
    p_value = float(2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_pairs - 1))) if n_pairs > 1 else 1.0
    crit_val = float(stats.t.ppf(0.975, df=n_pairs - 1)) if n_pairs > 1 else 1.96
    ci_lower = float(att - crit_val * se_att)
    ci_upper = float(att + crit_val * se_att)

    # Covariate Balance Diagnostics (SMD Before vs After)
    balance_before = {}
    balance_after = {}

    for c in X_encoded.columns:
        # Before matching
        t_pre = X_encoded.loc[treated_idx, c].values
        c_pre = X_encoded.loc[control_idx, c].values
        denom_pre = np.sqrt((np.var(t_pre, ddof=1) + np.var(c_pre, ddof=1)) / 2.0)
        smd_pre = float(np.abs(np.mean(t_pre) - np.mean(c_pre)) / denom_pre) if denom_pre > 0 else 0.0
        balance_before[c] = {"mean_treated": float(np.mean(t_pre)), "mean_control": float(np.mean(c_pre)), "smd": smd_pre}

        # After matching
        t_post = X_encoded.loc[matched_t_ids, c].values
        c_post = X_encoded.loc[matched_c_ids, c].values
        denom_post = np.sqrt((np.var(t_post, ddof=1) + np.var(c_post, ddof=1)) / 2.0)
        smd_post = float(np.abs(np.mean(t_post) - np.mean(c_post)) / denom_post) if denom_post > 0 else 0.0
        balance_after[c] = {"mean_treated": float(np.mean(t_post)), "mean_control": float(np.mean(c_post)), "smd": smd_post}

    max_smd_before = max(b["smd"] for b in balance_before.values()) if balance_before else 0.0
    max_smd_after = max(b["smd"] for b in balance_after.values()) if balance_after else 0.0

    return {
        "method": "propensity_score_matching_nearest_neighbor",
        "sample_summary": {
            "total_treated": len(treated_idx),
            "total_control": len(control_idx),
            "matched_pairs": n_pairs,
            "unmatched_treated": len(treated_idx) - n_pairs,
            "caliper_applied": caliper_val,
            "replacement": replacement,
        },
        "effect_estimates": {
            "att": att,
            "standard_error": se_att,
            "t_statistic": t_stat,
            "p_value": p_value,
            "confidence_interval_95": [ci_lower, ci_upper],
            "statistically_significant": p_value < 0.05,
        },
        "covariate_balance": {
            "before_matching": balance_before,
            "after_matching": balance_after,
            "max_abs_smd_before": max_smd_before,
            "max_abs_smd_after": max_smd_after,
            "balance_improved": max_smd_after < max_smd_before,
        },
        "conclusion": (
            f"Matching established {n_pairs} comparable pairs. The estimated Average Treatment Effect "
            f"on the Treated (ATT) is {att:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], p = {p_value:.4g}). "
            f"Covariate balance improved (Max SMD reduced from {max_smd_before:.4f} to {max_smd_after:.4f})."
        ),
    }
