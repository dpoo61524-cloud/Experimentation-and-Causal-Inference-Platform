"""Unit tests for the Bayesian A/B Testing Engine.

Tests cover:
- Posterior parameter updates and posterior means against analytical Beta formulas.
- Observed conversion rates and observed difference calculations.
- Credible intervals: valid bounds [0, 1], ordering (lower < upper), width scaling with level,
  and proper directionality (B minus A).
- Posterior superiority probabilities: near 0.5 for identical groups, high for clear winners,
  reversal symmetry on group swapping, and bounded in [0, 1].
- Minimum practical effect: probability bounds, non-increasing monotonicity as threshold increases,
  and equivalence to superiority when minimum_effect is 0.0.
- Boundary conditions: zero conversions, 100% conversion rates, minimal visitor counts (N=1),
  identical observed rates with different sample sizes, and custom priors.
- Reproducibility: deterministic random seed verification and simulation stability.
- Robust input validation: rejecting negative/zero visitors, conversions exceeding visitors,
  booleans/floats as counts, non-positive/infinite/NaN priors, out-of-range credible intervals,
  and invalid minimum effect values.
"""

from pathlib import Path
import sys

# Ensure src directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import math
import numpy as np
import pytest

from bayesian_engine import bayesian_ab_test


# ============================================================================
# A. Posterior Parameter Calculations & Observed Metrics
# ============================================================================

def test_posterior_parameter_calculations_with_default_prior():
    """Verify posterior alpha and beta updating rules under default flat Beta(1, 1) prior."""
    res = bayesian_ab_test(
        conversions_a=45,
        visitors_a=500,
        conversions_b=60,
        visitors_b=600,
        prior_alpha=1.0,
        prior_beta=1.0,
    )

    # alpha_post = alpha_prior + conversions
    # beta_post = beta_prior + visitors - conversions
    assert res["posterior_alpha_a"] == 46.0
    assert res["posterior_beta_a"] == 1.0 + (500 - 45)  # 456.0
    assert res["posterior_alpha_b"] == 61.0
    assert res["posterior_beta_b"] == 1.0 + (600 - 60)  # 541.0

    # Analytical posterior means: alpha / (alpha + beta)
    expected_mean_a = 46.0 / (46.0 + 456.0)
    expected_mean_b = 61.0 / (61.0 + 541.0)
    assert res["posterior_mean_a"] == pytest.approx(expected_mean_a, rel=1e-5)
    assert res["posterior_mean_b"] == pytest.approx(expected_mean_b, rel=1e-5)
    assert res["posterior_mean_difference"] == pytest.approx(
        expected_mean_b - expected_mean_a, rel=1e-5
    )


def test_posterior_parameter_calculations_with_custom_prior():
    """Verify posterior parameters when using an informative prior (e.g. Beta(5, 95))."""
    res = bayesian_ab_test(
        conversions_a=10,
        visitors_a=200,
        conversions_b=25,
        visitors_b=200,
        prior_alpha=5.0,
        prior_beta=95.0,
    )

    assert res["prior_alpha"] == 5.0
    assert res["prior_beta"] == 95.0
    assert res["posterior_alpha_a"] == 15.0
    assert res["posterior_beta_a"] == 95.0 + 190.0  # 285.0
    assert res["posterior_alpha_b"] == 30.0
    assert res["posterior_beta_b"] == 95.0 + 175.0  # 270.0

    expected_mean_a = 15.0 / 300.0  # 0.05
    expected_mean_b = 30.0 / 300.0  # 0.10
    assert res["posterior_mean_a"] == pytest.approx(expected_mean_a, rel=1e-5)
    assert res["posterior_mean_b"] == pytest.approx(expected_mean_b, rel=1e-5)
    assert res["posterior_mean_difference"] == pytest.approx(0.05, rel=1e-5)


def test_observed_rates_and_differences():
    """Verify observed conversion rates match simple sample fractions."""
    res = bayesian_ab_test(
        conversions_a=25,
        visitors_a=250,
        conversions_b=40,
        visitors_b=200,
    )

    assert res["visitors_a"] == 250
    assert res["conversions_a"] == 25
    assert res["visitors_b"] == 200
    assert res["conversions_b"] == 40
    assert res["conversion_rate_a"] == pytest.approx(0.10)
    assert res["observed_conversion_rate_a"] == pytest.approx(0.10)
    assert res["conversion_rate_b"] == pytest.approx(0.20)
    assert res["observed_conversion_rate_b"] == pytest.approx(0.20)
    assert res["observed_difference"] == pytest.approx(0.10)
    assert res["observed_absolute_difference"] == pytest.approx(0.10)


# ============================================================================
# B. Credible Intervals
# ============================================================================

def test_credible_intervals_bounds_and_ordering():
    """Confirm credible intervals are strictly within [0, 1] and lower < upper."""
    res = bayesian_ab_test(
        conversions_a=30,
        visitors_a=300,
        conversions_b=45,
        visitors_b=300,
        credible_interval=0.95,
    )

    ci_a = res["credible_interval_a"]
    ci_b = res["credible_interval_b"]
    ci_diff = res["credible_interval_difference"]

    # Bounds check
    assert 0.0 <= ci_a[0] < ci_a[1] <= 1.0
    assert 0.0 <= ci_b[0] < ci_b[1] <= 1.0
    assert ci_diff[0] < ci_diff[1]

    # CI levels recorded
    assert res["credible_interval_level"] == 0.95
    assert res["credible_interval"] == 0.95


def test_credible_interval_level_coverage_width():
    """Higher credible interval probability (e.g. 0.99 vs 0.80) must produce wider intervals."""
    res_80 = bayesian_ab_test(50, 1000, 70, 1000, credible_interval=0.80)
    res_95 = bayesian_ab_test(50, 1000, 70, 1000, credible_interval=0.95)
    res_99 = bayesian_ab_test(50, 1000, 70, 1000, credible_interval=0.99)

    width_a_80 = res_80["credible_interval_a"][1] - res_80["credible_interval_a"][0]
    width_a_95 = res_95["credible_interval_a"][1] - res_95["credible_interval_a"][0]
    width_a_99 = res_99["credible_interval_a"][1] - res_99["credible_interval_a"][0]

    assert width_a_80 < width_a_95 < width_a_99

    width_diff_80 = res_80["credible_interval_difference"][1] - res_80["credible_interval_difference"][0]
    width_diff_95 = res_95["credible_interval_difference"][1] - res_95["credible_interval_difference"][0]
    width_diff_99 = res_99["credible_interval_difference"][1] - res_99["credible_interval_difference"][0]

    assert width_diff_80 < width_diff_95 < width_diff_99


def test_difference_credible_interval_direction_b_minus_a():
    """Verify credible interval for difference follows the direction B minus A."""
    # When Group B is much better, difference CI should be strictly positive
    res_b_better = bayesian_ab_test(conversions_a=20, visitors_a=1000, conversions_b=100, visitors_b=1000)
    assert res_b_better["credible_interval_difference"][0] > 0.0

    # When Group A is much better, difference CI should be strictly negative
    res_a_better = bayesian_ab_test(conversions_a=100, visitors_a=1000, conversions_b=20, visitors_b=1000)
    assert res_b_better["credible_interval_difference"][1] > 0.0
    assert res_a_better["credible_interval_difference"][1] < 0.0


# ============================================================================
# C. Posterior Superiority Probabilities
# ============================================================================

def test_posterior_superiority_probability_identical_groups():
    """Identical group counts should yield P(B > A) approx 0.5 within Monte Carlo tolerance."""
    res = bayesian_ab_test(
        conversions_a=100,
        visitors_a=2000,
        conversions_b=100,
        visitors_b=2000,
    )
    prob = res["probability_b_greater_than_a"]
    assert 0.0 <= prob <= 1.0
    assert prob == pytest.approx(0.50, abs=0.02)


def test_posterior_superiority_probability_clear_winner():
    """A clearly superior Treatment group B should yield P(B > A) close to 1.0."""
    res = bayesian_ab_test(
        conversions_a=50,
        visitors_a=1000,
        conversions_b=120,
        visitors_b=1000,
    )
    assert res["probability_b_greater_than_a"] > 0.999


def test_posterior_superiority_probability_group_swap_symmetry():
    """Swapping groups A and B should approximately reverse the probability: P(B>A) + P(A>B) approx 1."""
    res_ab = bayesian_ab_test(conversions_a=40, visitors_a=800, conversions_b=55, visitors_b=800)
    res_ba = bayesian_ab_test(conversions_a=55, visitors_a=800, conversions_b=40, visitors_b=800)

    p_b_greater = res_ab["probability_b_greater_than_a"]
    p_a_greater = res_ba["probability_b_greater_than_a"]

    assert p_b_greater + p_a_greater == pytest.approx(1.0, abs=0.01)


def test_posterior_probabilities_bounded_between_zero_and_one():
    """Ensure all returned probabilities strictly lie in [0, 1]."""
    test_scenarios = [
        (0, 100, 100, 100),
        (50, 100, 50, 100),
        (99, 100, 1, 100),
        (5, 500, 6, 500),
    ]
    for ca, va, cb, vb in test_scenarios:
        res = bayesian_ab_test(ca, va, cb, vb, minimum_effect=0.01)
        assert 0.0 <= res["probability_b_greater_than_a"] <= 1.0
        assert 0.0 <= res["probability_b_exceeds_a_by_minimum_effect"] <= 1.0


# ============================================================================
# D. Minimum Practical Effect
# ============================================================================

def test_minimum_practical_effect_monotonicity():
    """Increasing minimum_effect threshold must not increase the estimated probability."""
    # When Group B has rate ~8% and Group A has rate ~5%
    ca, va, cb, vb = 50, 1000, 80, 1000
    effects = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
    probs = [
        bayesian_ab_test(ca, va, cb, vb, minimum_effect=eff)["probability_b_exceeds_a_by_minimum_effect"]
        for eff in effects
    ]

    for p in probs:
        assert 0.0 <= p <= 1.0

    # Check non-increasing ordering: probs[i] >= probs[i+1]
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1], f"Monotonicity violated at effect {effects[i]} vs {effects[i+1]}"


def test_minimum_practical_effect_zero_matches_superiority():
    """When minimum_effect=0.0, probability_b_exceeds_a_by_minimum_effect matches superiority."""
    res = bayesian_ab_test(conversions_a=35, visitors_a=500, conversions_b=45, visitors_b=500, minimum_effect=0.0)
    assert res["probability_b_exceeds_a_by_minimum_effect"] == res["probability_b_greater_than_a"]


# ============================================================================
# E. Boundary Cases
# ============================================================================

def test_boundary_zero_conversions():
    """Zero conversions in one or both groups must be handled cleanly via Beta regularization."""
    res_a_zero = bayesian_ab_test(conversions_a=0, visitors_a=200, conversions_b=10, visitors_b=200)
    assert res_a_zero["conversions_a"] == 0
    assert res_a_zero["observed_conversion_rate_a"] == 0.0
    assert res_a_zero["posterior_alpha_a"] == 1.0
    assert res_a_zero["posterior_beta_a"] == 201.0
    assert res_a_zero["posterior_mean_a"] == pytest.approx(1.0 / 202.0)
    assert 0.0 < res_a_zero["credible_interval_a"][0] < res_a_zero["credible_interval_a"][1] < 1.0

    res_both_zero = bayesian_ab_test(0, 100, 0, 100)
    assert res_both_zero["observed_conversion_rate_a"] == 0.0
    assert res_both_zero["observed_conversion_rate_b"] == 0.0
    assert res_both_zero["probability_b_greater_than_a"] == pytest.approx(0.5, abs=0.02)


def test_boundary_all_conversions():
    """All visitors converting in a group must be handled without mathematical errors."""
    res = bayesian_ab_test(conversions_a=100, visitors_a=100, conversions_b=90, visitors_b=100)
    assert res["observed_conversion_rate_a"] == 1.0
    assert res["posterior_alpha_a"] == 101.0
    assert res["posterior_beta_a"] == 1.0
    assert res["posterior_mean_a"] == pytest.approx(101.0 / 102.0)
    assert res["credible_interval_a"][0] < 1.0
    assert res["probability_b_greater_than_a"] < 0.01


def test_boundary_minimal_visitor_counts():
    """Valid minimal visitor counts (N=1) must execute and return sensible posteriors."""
    res = bayesian_ab_test(conversions_a=1, visitors_a=1, conversions_b=0, visitors_b=1)
    assert res["visitors_a"] == 1
    assert res["conversions_a"] == 1
    assert res["visitors_b"] == 1
    assert res["conversions_b"] == 0
    assert res["posterior_alpha_a"] == 2.0
    assert res["posterior_beta_a"] == 1.0
    assert res["posterior_alpha_b"] == 1.0
    assert res["posterior_beta_b"] == 2.0
    assert res["posterior_mean_a"] == pytest.approx(2.0 / 3.0)
    assert res["posterior_mean_b"] == pytest.approx(1.0 / 3.0)
    assert 0.0 < res["probability_b_greater_than_a"] < 0.5


def test_boundary_identical_rates_different_sample_sizes():
    """Groups with identical rates but different sample sizes should reflect tighter CI on larger sample."""
    res = bayesian_ab_test(conversions_a=10, visitors_a=100, conversions_b=100, visitors_b=1000)
    assert res["conversion_rate_a"] == pytest.approx(0.10)
    assert res["conversion_rate_b"] == pytest.approx(0.10)

    width_a = res["credible_interval_a"][1] - res["credible_interval_a"][0]
    width_b = res["credible_interval_b"][1] - res["credible_interval_b"][0]

    # Larger sample size (N=1000) must have a narrower credible interval than small sample (N=100)
    assert width_b < width_a


def test_boundary_informative_and_jeffreys_priors():
    """Support non-standard valid priors, such as Jeffreys prior Beta(0.5, 0.5)."""
    res = bayesian_ab_test(
        conversions_a=20,
        visitors_a=200,
        conversions_b=30,
        visitors_b=200,
        prior_alpha=0.5,
        prior_beta=0.5,
    )
    assert res["posterior_alpha_a"] == 20.5
    assert res["posterior_beta_a"] == 180.5
    assert res["posterior_alpha_b"] == 30.5
    assert res["posterior_beta_b"] == 170.5
    assert res["posterior_mean_a"] == pytest.approx(20.5 / 201.0)
    assert res["posterior_mean_b"] == pytest.approx(30.5 / 201.0)


# ============================================================================
# G. Reproducibility & Simulation Behavior
# ============================================================================

def test_deterministic_reproducibility_with_seed():
    """Identical calls with default or fixed seed must return identical simulation results."""
    run1 = bayesian_ab_test(50, 1000, 65, 1000, random_seed=42)
    run2 = bayesian_ab_test(50, 1000, 65, 1000, random_seed=42)

    assert run1["probability_b_greater_than_a"] == run2["probability_b_greater_than_a"]
    assert run1["credible_interval_difference"] == run2["credible_interval_difference"]
    assert run1["probability_b_exceeds_a_by_minimum_effect"] == run2["probability_b_exceeds_a_by_minimum_effect"]


def test_different_seeds_produce_stable_estimates():
    """Different random seeds should produce slightly varied but mutually consistent estimates."""
    run_seed_1 = bayesian_ab_test(50, 1000, 65, 1000, random_seed=123)
    run_seed_2 = bayesian_ab_test(50, 1000, 65, 1000, random_seed=789)

    # Different seeds should not produce identical float bits
    assert run_seed_1["probability_b_greater_than_a"] != run_seed_2["probability_b_greater_than_a"]

    # But estimates should be extremely close (within 0.01 for 100,000 samples)
    assert run_seed_1["probability_b_greater_than_a"] == pytest.approx(
        run_seed_2["probability_b_greater_than_a"], abs=0.01
    )


# ============================================================================
# H. Input Validation & Error Handling
# ============================================================================

@pytest.mark.parametrize(
    "invalid_kwargs, err_match",
    [
        ({"visitors_a": 0}, "visitors_a must be at least 1"),
        ({"visitors_b": 0}, "visitors_b must be at least 1"),
        ({"visitors_a": -5}, "visitors_a must be at least 1"),
        ({"visitors_b": -10}, "visitors_b must be at least 1"),
        ({"conversions_a": -1}, "conversions_a cannot be negative"),
        ({"conversions_b": -4}, "conversions_b cannot be negative"),
        ({"conversions_a": 105, "visitors_a": 100}, r"conversions_a.*cannot exceed visitors_a"),
        ({"conversions_b": 51, "visitors_b": 50}, r"conversions_b.*cannot exceed visitors_b"),
    ],
)
def test_invalid_counts_and_visitors(invalid_kwargs, err_match):
    """Reject non-positive visitors, negative conversions, and conversions exceeding visitors."""
    base_kwargs = {"conversions_a": 10, "visitors_a": 100, "conversions_b": 10, "visitors_b": 100}
    base_kwargs.update(invalid_kwargs)
    with pytest.raises(ValueError, match=err_match):
        bayesian_ab_test(**base_kwargs)


@pytest.mark.parametrize(
    "invalid_kwargs, err_match",
    [
        ({"conversions_a": True}, "conversions_a must be an integer"),
        ({"conversions_b": False}, "conversions_b must be an integer"),
        ({"visitors_a": True}, "visitors_a must be an integer"),
        ({"visitors_b": False}, "visitors_b must be an integer"),
        ({"conversions_a": 10.5}, "conversions_a must be an integer"),
        ({"visitors_a": 100.0}, "conversions_a must be an integer" if False else "visitors_a must be an integer"),
        ({"conversions_b": "10"}, "conversions_b must be an integer"),
    ],
)
def test_invalid_counts_types_booleans_and_floats(invalid_kwargs, err_match):
    """Counts must be integers; booleans, floats, and strings must be rejected."""
    base_kwargs = {"conversions_a": 10, "visitors_a": 100, "conversions_b": 10, "visitors_b": 100}
    base_kwargs.update(invalid_kwargs)
    with pytest.raises(ValueError, match=err_match):
        bayesian_ab_test(**base_kwargs)


@pytest.mark.parametrize(
    "invalid_kwargs, err_match",
    [
        ({"prior_alpha": 0.0}, "prior_alpha must be a positive number"),
        ({"prior_alpha": -1.0}, "prior_alpha must be a positive number"),
        ({"prior_alpha": float("nan")}, "prior_alpha must be a finite number"),
        ({"prior_alpha": float("inf")}, "prior_alpha must be a finite number"),
        ({"prior_alpha": True}, "prior_alpha must be a positive numeric value"),
        ({"prior_beta": 0.0}, "prior_beta must be a positive number"),
        ({"prior_beta": -0.5}, "prior_beta must be a positive number"),
        ({"prior_beta": float("nan")}, "prior_beta must be a finite number"),
        ({"prior_beta": float("inf")}, "prior_beta must be a finite number"),
        ({"prior_beta": False}, "prior_beta must be a positive numeric value"),
    ],
)
def test_invalid_prior_parameters(invalid_kwargs, err_match):
    """Prior alpha and beta must be positive, finite numeric values."""
    base_kwargs = {"conversions_a": 10, "visitors_a": 100, "conversions_b": 10, "visitors_b": 100}
    base_kwargs.update(invalid_kwargs)
    with pytest.raises(ValueError, match=err_match):
        bayesian_ab_test(**base_kwargs)


@pytest.mark.parametrize(
    "invalid_ci, err_match",
    [
        (0.0, "credible_interval must be strictly between 0 and 1"),
        (1.0, "credible_interval must be strictly between 0 and 1"),
        (-0.05, "credible_interval must be strictly between 0 and 1"),
        (1.05, "credible_interval must be strictly between 0 and 1"),
        (float("nan"), "credible_interval must be a finite number"),
        (float("inf"), "credible_interval must be a finite number"),
        (True, "credible_interval must be numeric"),
        ("0.95", "credible_interval must be numeric"),
    ],
)
def test_invalid_credible_interval_values(invalid_ci, err_match):
    """Credible interval level must be strictly between 0 and 1 exclusive."""
    with pytest.raises(ValueError, match=err_match):
        bayesian_ab_test(10, 100, 10, 100, credible_interval=invalid_ci)


@pytest.mark.parametrize(
    "invalid_effect, err_match",
    [
        (-0.001, "minimum_effect must be between 0 and 1"),
        (1.001, "minimum_effect must be between 0 and 1"),
        (float("nan"), "minimum_effect must be a finite number"),
        (float("inf"), "minimum_effect must be a finite number"),
        (True, "minimum_effect must be numeric"),
        ("0.05", "minimum_effect must be numeric"),
    ],
)
def test_invalid_minimum_effect_values(invalid_effect, err_match):
    """Minimum effect must be a finite number between 0 and 1 inclusive."""
    with pytest.raises(ValueError, match=err_match):
        bayesian_ab_test(10, 100, 10, 100, minimum_effect=invalid_effect)


@pytest.mark.parametrize(
    "invalid_kwargs, err_match",
    [
        ({"num_samples": 0}, "num_samples must be a positive integer"),
        ({"num_samples": -100}, "num_samples must be a positive integer"),
        ({"num_samples": True}, "num_samples must be an integer"),
        ({"num_samples": 1000.5}, "num_samples must be an integer"),
        ({"random_seed": True}, "random_seed must be an integer or None"),
        ({"random_seed": 42.5}, "random_seed must be an integer or None"),
    ],
)
def test_invalid_simulation_settings(invalid_kwargs, err_match):
    """Simulation settings must enforce positive integers for sample count and integer seeds."""
    base_kwargs = {"conversions_a": 10, "visitors_a": 100, "conversions_b": 10, "visitors_b": 100}
    base_kwargs.update(invalid_kwargs)
    with pytest.raises(ValueError, match=err_match):
        bayesian_ab_test(**base_kwargs)
