
import math

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
