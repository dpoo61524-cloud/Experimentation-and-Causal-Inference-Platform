
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

