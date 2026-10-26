
import math
from typing import Any, Sequence

import numpy as np
import scipy.stats as stats
from statsmodels.stats.proportion import (
    proportions_ztest,
    confint_proportions_2indep,
    proportion_confint,
    proportion_effectsize,
)
from statsmodels.stats.power import NormalIndPower


def compare_conversion_rates(
    conversions_a: int,
    visitors_a: int,
    conversions_b: int,
    visitors_b: int,
) -> dict:
    """Compare two groups using a two-proportion z-test."""

    if visitors_a <= 0 or visitors_b <= 0:
        raise ValueError("Each group must have at least one visitor.")

    if not 0 <= conversions_a <= visitors_a:
        raise ValueError(
            "Group A conversions must be between zero and its visitors."
        )

    if not 0 <= conversions_b <= visitors_b:
        raise ValueError(
            "Group B conversions must be between zero and its visitors."
        )

    counts = [conversions_a, conversions_b]
    sample_sizes = [visitors_a, visitors_b]

    z_statistic, p_value = proportions_ztest(
        count=counts,
        nobs=sample_sizes,
        alternative="two-sided",
    )

    rate_a = conversions_a / visitors_a
    rate_b = conversions_b / visitors_b

    ci_low, ci_high = confint_proportions_2indep(
        count1=conversions_a,
        nobs1=visitors_a,
        count2=conversions_b,
        nobs2=visitors_b,
        method="newcomb",
    )

    return {
        "conversion_rate_a": rate_a,
        "conversion_rate_b": rate_b,
        "absolute_difference": rate_a - rate_b,
        "z_statistic": float(z_statistic),
        "p_value": float(p_value),
        "confidence_interval_95": [
            float(ci_low),
            float(ci_high),
        ],
        "statistically_significant": bool(p_value < 0.05),
    }


def calculate_sample_size(
    baseline_rate: float,
    expected_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """Estimate sample size required per group for an A/B test."""

    if not 0 < baseline_rate < 1:
        raise ValueError("Baseline rate must be between 0 and 1.")

    if not 0 < expected_rate < 1:
        raise ValueError("Expected rate must be between 0 and 1.")

    if baseline_rate == expected_rate:
        raise ValueError(
            "Expected rate must differ from baseline rate."
        )

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1.")

    effect_size = abs(
        proportion_effectsize(baseline_rate, expected_rate)
    )

    analysis = NormalIndPower()

    required_sample = analysis.solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=1.0,
        alternative="two-sided",
    )

    sample_size = math.ceil(required_sample)

    return {
        "baseline_rate": baseline_rate,
        "expected_rate": expected_rate,
        "absolute_effect": abs(expected_rate - baseline_rate),
        "alpha": alpha,
        "power": power,
        "sample_size_per_group": sample_size,
        "total_sample_size": sample_size * 2,
    }


def calculate_conversion_confidence_interval(
    conversions: int,
    visitors: int,
    confidence_level: float = 0.95,
) -> dict:
    """Calculate a Wilson confidence interval for one conversion rate."""

    if visitors <= 0:
        raise ValueError("Visitors must be greater than zero.")

    if not 0 <= conversions <= visitors:
        raise ValueError(
            "Conversions must be between zero and visitors."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    rate = conversions / visitors
    alpha = 1 - confidence_level

    ci_low, ci_high = proportion_confint(
        count=conversions,
        nobs=visitors,
        alpha=alpha,
        method="wilson",
    )

    return {
        "conversion_rate": rate,
        "confidence_level": confidence_level,
        "confidence_interval": [
            float(ci_low),
            float(ci_high),
        ],
    }


def two_sample_ttest(
    sample_a: Sequence[float | int] | Any,
    sample_b: Sequence[float | int] | Any,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> dict:
    """Perform Welch's independent two-sample t-test.

    Welch's t-test compares the means of two independent groups without assuming
    equal population variances (unlike Student's standard t-test).

    Assumptions & Considerations:
    1. Independent observations between and within groups.
    2. Continuous numerical measurement scale.
    3. Approximate normality in both populations, or sufficiently large sample
       sizes (n >= 30) for the Central Limit Theorem to apply.

    Parameters:
    - sample_a: Numerical observations for Group A.
    - sample_b: Numerical observations for Group B.
    - alternative: 'two-sided', 'greater', or 'less'. Default is 'two-sided'.
    - alpha: Significance threshold (default 0.05).

    Returns:
    Dictionary containing t_statistic, p_value, sample_size_a, sample_size_b,
    mean_a, mean_b, difference_in_means, degrees_of_freedom, confidence_interval_95,
    and statistically_significant.
    """
    valid_alternatives = ("two-sided", "greater", "less")
    if alternative not in valid_alternatives:
        raise ValueError(
            f"Alternative hypothesis must be one of {valid_alternatives}. Got '{alternative}'."
        )

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    if not hasattr(sample_a, "__iter__") or not hasattr(sample_b, "__iter__"):
        raise ValueError("Samples must be iterable sequences of numbers.")

    clean_a: list[float] = []
    for x in sample_a:
        if isinstance(x, bool) or x is None:
            raise ValueError("Sample A contains invalid non-numeric or missing values.")
        try:
            val = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"Sample A contains non-numeric value: {x}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError("Sample A contains NaN or infinite values.")
        clean_a.append(val)

    clean_b: list[float] = []
    for x in sample_b:
        if isinstance(x, bool) or x is None:
            raise ValueError("Sample B contains invalid non-numeric or missing values.")
        try:
            val = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"Sample B contains non-numeric value: {x}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError("Sample B contains NaN or infinite values.")
        clean_b.append(val)

    if len(clean_a) < 2:
        raise ValueError(
            "Group A must contain at least 2 observations for a two-sample t-test."
        )
    if len(clean_b) < 2:
        raise ValueError(
            "Group B must contain at least 2 observations for a two-sample t-test."
        )

    var_a = float(np.var(clean_a, ddof=1))
    var_b = float(np.var(clean_b, ddof=1))
    if var_a == 0.0 and var_b == 0.0:
        raise ValueError(
            "Both samples have zero variance (all values are identical). The t-test is undefined."
        )

    # Perform Welch's t-test (equal_var=False)
    res = stats.ttest_ind(clean_a, clean_b, equal_var=False, alternative=alternative)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)
    df = float(res.df) if res.df is not None else None

    mean_a = float(np.mean(clean_a))
    mean_b = float(np.mean(clean_b))
    diff_means = float(mean_a - mean_b)

    # 95% confidence interval for difference in means (using two-sided critical value)
    ci_res = stats.ttest_ind(clean_a, clean_b, equal_var=False, alternative="two-sided")
    ci = ci_res.confidence_interval(confidence_level=1.0 - alpha)
    ci_95 = [float(ci.low), float(ci.high)]

    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "sample_size_a": len(clean_a),
        "sample_size_b": len(clean_b),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference_in_means": diff_means,
        "degrees_of_freedom": df,
        "confidence_interval_95": ci_95,
        "statistically_significant": bool(p_val < alpha),
        "alternative": alternative,
    }


def chi_square_test(
    table: Sequence[Sequence[float | int]] | Any,
    alpha: float = 0.05,
    correction: bool = True,
) -> dict:
    """Perform Pearson's chi-square test of independence on a contingency table.

    Assumptions & Requirements:
    1. Independent observations: each unit of observation falls into exactly one cell.
    2. Categorical variables with mutually exclusive categories.
    3. Adequate expected cell counts: reliable chi-square approximations require
       expected frequencies of at least 5 in most cells (and at least 1 in all cells).
       When expected cell counts are too small, a warning is returned.

    Parameters:
    - table: 2D table of categorical counts (at least 2 rows and 2 columns).
    - alpha: Significance threshold (default 0.05).
    - correction: Whether to apply Yates' continuity correction for 2x2 tables (default True).

    Returns:
    Dictionary containing chi_square_statistic, degrees_of_freedom, p_value,
    expected_frequencies, statistically_significant, alpha, and warning (if any).
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    if not hasattr(table, "__iter__"):
        raise ValueError("Contingency table must be a 2D sequence of numerical counts.")

    rows = list(table)
    if len(rows) < 2:
        raise ValueError(
            "Contingency table must contain at least 2 rows."
        )

    clean_table: list[list[float]] = []
    expected_cols: int | None = None

    for row_idx, row in enumerate(rows):
        if not hasattr(row, "__iter__") or isinstance(row, (str, bytes, dict)):
            raise ValueError(
                f"Row {row_idx} of contingency table is not a valid sequence."
            )
        row_items = list(row)
        if len(row_items) < 2:
            raise ValueError(
                f"Row {row_idx} must contain at least 2 columns."
            )
        if expected_cols is None:
            expected_cols = len(row_items)
        elif len(row_items) != expected_cols:
            raise ValueError(
                f"Inconsistent column count at row {row_idx}: expected {expected_cols}, got {len(row_items)}."
            )

        clean_row: list[float] = []
        for col_idx, cell in enumerate(row_items):
            if isinstance(cell, bool) or cell is None:
                raise ValueError(
                    f"Cell at ({row_idx}, {col_idx}) contains invalid non-numeric or missing value."
                )
            try:
                val = float(cell)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Cell at ({row_idx}, {col_idx}) contains non-numeric value: {cell}"
                )
            if math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Cell at ({row_idx}, {col_idx}) contains NaN or infinite value."
                )
            if val < 0:
                raise ValueError(
                    f"Contingency table counts cannot be negative: cell ({row_idx}, {col_idx}) = {val}."
                )
            clean_row.append(val)
        clean_table.append(clean_row)

    matrix = np.array(clean_table, dtype=float)
    total_count = float(np.sum(matrix))
    if total_count <= 0:
        raise ValueError(
            "Contingency table must contain at least one observation (total count > 0)."
        )

    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)

    if np.any(row_sums == 0):
        raise ValueError(
            "Each row in the contingency table must contain at least one observation."
        )
    if np.any(col_sums == 0):
        raise ValueError(
            "Each column in the contingency table must contain at least one observation."
        )

    # Perform chi-square test
    res = stats.chi2_contingency(matrix, correction=correction)
    chi2_stat = float(res.statistic)
    p_val = float(res.pvalue)
    dof = int(res.dof)
    expected_freq = res.expected_freq.tolist()

    # Assess expected cell count adequacy
    min_expected = float(np.min(res.expected_freq))
    small_cells_count = int(np.sum(res.expected_freq < 5))

    warning = None
    if min_expected < 5.0:
        warning = (
            f"Warning: {small_cells_count} cell(s) have expected frequencies less than 5 "
            f"(minimum expected frequency is {min_expected:.2f}). Chi-square approximation "
            "may be unreliable. Consider combining categories or using Fisher's exact test for 2x2 tables."
        )

    return {
        "chi_square_statistic": chi2_stat,
        "degrees_of_freedom": dof,
        "p_value": p_val,
        "expected_frequencies": expected_freq,
        "statistically_significant": bool(p_val < alpha),
        "alpha": alpha,
        "warning": warning,
    }


# --- Sequential Testing Engine ---

# Precomputed exact discrete O'Brien-Fleming constants c_K(alpha)
# Verified against canonical multivariate normal joint integration with error < 1e-4.
# Reference: O'Brien & Fleming (Biometrics 1979); Jennison & Turnbull (2000, Table 2.3).
OBRIEN_FLEMING_CONSTANTS = {
    0.05: {
        2: 1.977431,
        3: 2.004026,
        4: 2.024315,
        5: 2.040062,
    },
    0.01: {
        2: 2.579570,
        3: 2.594794,
        4: 2.608938,
        5: 2.621068,
    },
    0.10: {
        2: 1.677953,
        3: 1.709602,
        4: 1.733084,
        5: 1.750862,
    },
}


def get_canonical_covariance(total_checkpoints: int) -> np.ndarray:
    """Generate the canonical covariance matrix for K equally spaced interim looks.

    Under H0, for looks at equal information fractions t_k = k/K, the sequential
    z-statistics follow a Gaussian Markov process with Cov(Z_j, Z_k) = sqrt(j / k) for j <= k.
    """
    K = total_checkpoints
    cov = np.zeros((K, K), dtype=float)
    for j in range(K):
        for k in range(K):
            cov[j, k] = math.sqrt(min(j + 1, k + 1) / max(j + 1, k + 1))
    return cov


def calculate_crossing_probability(boundaries: Sequence[float]) -> float:
    """Calculate the overall Type I error probability (boundary-crossing probability)

    using the canonical joint-normal distribution across K interim looks.
    """
    K = len(boundaries)
    cov = get_canonical_covariance(K)
    lower = [-float(b) for b in boundaries]
    upper = [float(b) for b in boundaries]

    # Probability of staying inside the continuation region across all K looks
    p_inside = stats.multivariate_normal.cdf(
        upper,
        mean=np.zeros(K),
        cov=cov,
        lower_limit=lower,
    )
    return float(1.0 - p_inside)


def solve_obrien_fleming_constant(total_checkpoints: int, alpha: float) -> float:
    """Numerically solve for the O'Brien-Fleming constant c_K(alpha) that controls

    the overall Type I error at alpha across K equally spaced looks.
    """
    from scipy.optimize import root_scalar

    K = total_checkpoints

    def objective(c: float) -> float:
        bounds = [c * math.sqrt(K / (k + 1)) for k in range(K)]
        return calculate_crossing_probability(bounds) - alpha

    result = root_scalar(objective, bracket=[1.0, 4.0], xtol=1e-8, rtol=1e-8)
    return float(result.root)


def get_obrien_fleming_design(
    total_checkpoints: int,
    alpha: float = 0.05,
) -> tuple[float, list[float]]:
    """Retrieve or numerically solve the O'Brien-Fleming constant and boundary vector.

    For K equally spaced looks, b_k = c_K * sqrt(K / k).
    """
    if total_checkpoints not in (2, 3, 4, 5):
        raise ValueError("Supported total_checkpoints are 2, 3, 4, or 5.")

    rounded_alpha = round(alpha, 4)
    if rounded_alpha in OBRIEN_FLEMING_CONSTANTS and total_checkpoints in OBRIEN_FLEMING_CONSTANTS[rounded_alpha]:
        c_k = OBRIEN_FLEMING_CONSTANTS[rounded_alpha][total_checkpoints]
    else:
        c_k = solve_obrien_fleming_constant(total_checkpoints, alpha)

    boundaries = [
        float(c_k * math.sqrt(total_checkpoints / k))
        for k in range(1, total_checkpoints + 1)
    ]
    return c_k, boundaries


def sequential_conversion_test(
    checkpoint: int,
    total_checkpoints: int,
    conversions_a: int,
    visitors_a: int,
    conversions_b: int,
    visitors_b: int,
    max_visitors_per_group: int,
    alpha: float = 0.05,
    evaluate_futility: bool = False,
    schedule_tolerance: Optional[float] = 0.15,
) -> dict:
    """Evaluate an A/B conversion-rate experiment at an interim or final checkpoint
    using a discrete O'Brien-Fleming (1979) group-sequential design.

    Statistical Methodology:
    - Overall Type I error is controlled at alpha across K equally spaced looks.
    - Test statistic Z_k is the standard pooled two-proportion z-statistic.
    - Efficacy stopping boundary at look k is Z_crit(k) = c_K * sqrt(K / k).
    - Asymptotic normal approximation assumption: requires adequate sample size
      (n*p >= 5, n*(1-p) >= 5 per group) for the CLT to apply. Does not claim
      exact finite-sample binomial permutation control.
    - Schedule enforcement: Equal-information boundaries assume information accumulates
      proportionally (I_k / I_max approx k / K). If the observed information fraction
      deviates from planned fraction by more than schedule_tolerance, the calculation is
      rejected to prevent invalid peeking or premature boundary application.
    - Non-binding futility: an optional directional heuristic (Z_k <= 0 at look k >= K/2);
      does NOT prove absence of effect and does not evaluate formal conditional power.
    - At the final look (k = K), reports that the maximum sample size was reached,
      and whether the efficacy criteria were met (a non-significant result represents
      inconclusive evidence, not proof of no effect).

    Parameters:
    - checkpoint: Current look index (1 <= checkpoint <= total_checkpoints).
    - total_checkpoints: Planned number of looks K (2, 3, 4, or 5).
    - conversions_a: Cumulative conversions in Control (Group A).
    - visitors_a: Cumulative visitors in Control (Group A).
    - conversions_b: Cumulative conversions in Treatment (Group B).
    - visitors_b: Cumulative visitors in Treatment (Group B).
    - max_visitors_per_group: Planned maximum sample size per group (N_max).
    - alpha: Overall two-sided significance level (default 0.05).
    - evaluate_futility: Whether to evaluate optional non-binding directional futility.
    - schedule_tolerance: Maximum allowable absolute difference between observed
      and planned information fraction (|I_obs - k/K|). Default is 0.15 (15%).
      Set to None to disable schedule enforcement.

    Returns:
    Structured dictionary containing design metadata, interim statistics,
    boundaries, decision, schedule audit, and educational explanation.
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    if total_checkpoints not in (2, 3, 4, 5):
        raise ValueError("total_checkpoints must be one of 2, 3, 4, or 5.")

    if not isinstance(checkpoint, int) or isinstance(checkpoint, bool):
        raise ValueError("checkpoint must be an integer.")

    if not (1 <= checkpoint <= total_checkpoints):
        raise ValueError(
            f"checkpoint must be between 1 and {total_checkpoints}, got {checkpoint}."
        )

    if max_visitors_per_group <= 0:
        raise ValueError("max_visitors_per_group must be greater than zero.")

    # Validate counts
    for name, conv, vis in [
        ("Group A (control)", conversions_a, visitors_a),
        ("Group B (treatment)", conversions_b, visitors_b),
    ]:
        if isinstance(conv, bool) or isinstance(vis, bool):
            raise ValueError(f"{name} counts must be integers, not booleans.")
        if vis <= 0:
            raise ValueError(f"{name} visitors must be greater than zero.")
        if conv < 0:
            raise ValueError(f"{name} conversions cannot be negative.")
        if conv > vis:
            raise ValueError(f"{name} conversions cannot exceed visitors.")
        if vis > max_visitors_per_group:
            raise ValueError(
                f"{name} visitors ({vis}) exceed planned maximum sample size per group ({max_visitors_per_group})."
            )

    # Schedule fractions
    planned_info_fraction = checkpoint / total_checkpoints
    observed_info_fraction = (visitors_a + visitors_b) / (2 * max_visitors_per_group)

    # Enforce checkpoint schedule assumptions
    if schedule_tolerance is not None:
        if not (0.0 <= schedule_tolerance <= 1.0):
            raise ValueError("schedule_tolerance must be between 0 and 1, or None.")
        if abs(observed_info_fraction - planned_info_fraction) > schedule_tolerance:
            raise ValueError(
                f"Observed information fraction ({observed_info_fraction:.1%}) deviates from "
                f"planned information fraction ({planned_info_fraction:.1%}) by more than "
                f"allowed schedule tolerance ({schedule_tolerance:.1%}). Equal-information O'Brien-Fleming "
                f"boundaries cannot be applied when observed checkpoint does not meet the design's schedule assumptions."
            )

    # Get verified boundaries
    c_k, boundaries = get_obrien_fleming_design(total_checkpoints, alpha)
    current_z_crit = boundaries[checkpoint - 1]
    nominal_alpha_at_look = float(2.0 * (1.0 - stats.norm.cdf(current_z_crit)))

    # Compute two-proportion z-statistic (Treatment B - Control A)
    rate_a = conversions_a / visitors_a
    rate_b = conversions_b / visitors_b
    pooled_p = (conversions_a + conversions_b) / (visitors_a + visitors_b)

    if pooled_p == 0.0 or pooled_p == 1.0:
        z_stat = 0.0
        p_val = 1.0
    else:
        se_pooled = math.sqrt(
            pooled_p * (1.0 - pooled_p) * (1.0 / visitors_a + 1.0 / visitors_b)
        )
        z_stat = (rate_b - rate_a) / se_pooled
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

    efficacy_boundary_crossed = bool(abs(z_stat) >= current_z_crit)
    is_final_look = bool(checkpoint == total_checkpoints)

    # Optional non-binding futility heuristic (directional: treatment <= control at or after halfway)
    futility_boundary_crossed = False
    futility_recommendation = None
    if evaluate_futility and not is_final_look and checkpoint >= math.ceil(total_checkpoints / 2):
        if z_stat <= 0.0:
            futility_boundary_crossed = True
            futility_recommendation = "stop_for_futility"

    # Determine structured decision and explanation
    if not is_final_look:
        if efficacy_boundary_crossed:
            decision = "stop_for_efficacy"
            explanation = (
                f"Interim checkpoint {checkpoint}/{total_checkpoints}: Two-sided test statistic "
                f"|Z|={abs(z_stat):.4f} crossed the O'Brien-Fleming efficacy boundary "
                f"Z_crit={current_z_crit:.4f} (nominal p={p_val:.6g} <= {nominal_alpha_at_look:.6g}). "
                f"The trial may stop early for efficacy while preserving the overall Type I error rate."
            )
        elif futility_boundary_crossed:
            decision = "continue"
            explanation = (
                f"Interim checkpoint {checkpoint}/{total_checkpoints}: Efficacy boundary was not crossed. "
                f"Advisory non-binding futility flag: Treatment conversion rate is not exceeding control (Z={z_stat:.4f} <= 0). "
                f"This is an operational heuristic to avoid wasting traffic and does NOT prove absence of an effect."
            )
        else:
            decision = "continue"
            explanation = (
                f"Interim checkpoint {checkpoint}/{total_checkpoints}: Test statistic |Z|={abs(z_stat):.4f} "
                f"did not cross efficacy boundary Z_crit={current_z_crit:.4f}. Continue experiment to next scheduled checkpoint."
            )
    else:
        # Final look at maximum sample size
        if efficacy_boundary_crossed:
            decision = "stop_max_sample_size_efficacy_achieved"
            explanation = (
                f"Maximum planned sample size reached at final checkpoint {total_checkpoints}/{total_checkpoints}. "
                f"Test statistic |Z|={abs(z_stat):.4f} crossed the final critical boundary "
                f"Z_crit={current_z_crit:.4f} (nominal p={p_val:.6g} <= {nominal_alpha_at_look:.6g}). "
                f"The difference between treatment and control is statistically significant at overall alpha={alpha}."
            )
        else:
            decision = "stop_max_sample_size_inconclusive"
            explanation = (
                f"Maximum planned sample size reached at final checkpoint {total_checkpoints}/{total_checkpoints}. "
                f"Test statistic |Z|={abs(z_stat):.4f} did not cross the final critical boundary "
                f"Z_crit={current_z_crit:.4f}. The experiment fails to reject the null hypothesis at overall alpha={alpha}; "
                f"this represents inconclusive evidence, NOT proof that the treatment has no effect."
            )

    return {
        "design": "O'Brien-Fleming Group Sequential Design",
        "methodology": "Discrete canonical joint-normal boundary calculation",
        "checkpoint": checkpoint,
        "total_checkpoints": total_checkpoints,
        "planned_information_fraction": float(planned_info_fraction),
        "observed_information_fraction": float(observed_info_fraction),
        "schedule_tolerance": schedule_tolerance,
        "schedule_compliant": True,
        "max_visitors_per_group": max_visitors_per_group,
        "conversions_a": conversions_a,
        "visitors_a": visitors_a,
        "conversion_rate_a": rate_a,
        "conversions_b": conversions_b,
        "visitors_b": visitors_b,
        "conversion_rate_b": rate_b,
        "absolute_difference": rate_b - rate_a,
        "z_statistic": float(z_stat),
        "nominal_p_value": p_val,
        "efficacy_boundary_z": current_z_crit,
        "nominal_alpha_threshold": nominal_alpha_at_look,
        "all_planned_boundaries_z": boundaries,
        "efficacy_boundary_crossed": efficacy_boundary_crossed,
        "maximum_sample_size_reached": is_final_look,
        "evaluate_futility": evaluate_futility,
        "futility_boundary_crossed": futility_boundary_crossed,
        "futility_recommendation": futility_recommendation,
        "decision": decision,
        "overall_alpha": alpha,
        "explanation": explanation,
        "assumptions": (
            "Independent observations; CLT-based asymptotic normality of proportion test statistic; "
            "canonical joint-normal covariance for interim score statistics."
        ),
    }

