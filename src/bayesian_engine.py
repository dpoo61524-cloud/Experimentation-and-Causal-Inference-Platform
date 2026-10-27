"""Bayesian A/B Testing Engine using a Beta-Binomial Model.

This module provides a beginner-friendly, mathematically sound Bayesian A/B testing
engine for conversion-rate experiments.

Statistical Overview:
---------------------
1. Beta-Binomial Conjugate Model:
   - In conversion experiments, each visitor either converts (1) or does not (0),
     which is modeled as a Bernoulli trial.
   - For a group with `n` visitors and `c` conversions, the likelihood is Binomial(n, p),
     where `p` is the true underlying conversion rate.
   - The Beta distribution is the conjugate prior for the Binomial likelihood.
     If the prior belief about `p` is Beta(alpha_0, beta_0), observing `c` conversions
     out of `n` visitors results in an updated posterior distribution:
         p ~ Beta(alpha_0 + c, beta_0 + n - c)
   - Default prior: Beta(1, 1), which is a uniform distribution over [0, 1],
     representing an uninformative ("flat") initial belief before observing any data.

2. Interpreting Posterior Metrics:
   - Posterior Mean: The expected conversion rate given the prior and observed data:
         E[p] = alpha / (alpha + beta)
     With Beta(1, 1), this becomes (c + 1) / (n + 2) (Laplace's rule of succession),
     which gently regularizes small-sample estimates toward 0.5.
   - Credible Interval (CI): A Bayesian credible interval [lower, upper] has a direct
     probabilistic interpretation: given the data and prior, there is a (e.g.) 95%
     probability that the true conversion rate lies within this interval.
     Calculated via the inverse CDF (percent-point function) of the Beta distribution.
   - Probability of Superiority: P(p_B > p_A) is the posterior probability that Group B's
     true conversion rate is strictly higher than Group A's.
   - Probability of Meaningful Improvement: P(p_B - p_A >= minimum_effect) is the posterior
     probability that Group B outperforms Group A by at least a predefined threshold.
   - Credible Interval for the Difference: The range within which the difference (p_B - p_A)
     falls with high posterior probability (e.g., 95%).

Units and Conventions:
----------------------
- All conversion rates, minimum effects, and differences are expressed as proportions
  in the range [0.0, 1.0], NOT percentages from 0 to 100.
  Example: A 5% conversion rate must be represented as 0.05.
- The difference is defined as Group B (Treatment) minus Group A (Control): (p_B - p_A).

Methodology & Limitations:
--------------------------
- The difference distribution (p_B - p_A) and tail probabilities are estimated via
  Monte Carlo sampling from the two independent posterior Beta distributions.
- Monte Carlo estimates are reproducible approximations using a deterministic random
  seed, not exact closed-form solutions.
- The model assumes independent, identically distributed Bernoulli trials with stationary
  conversion rates (no time-varying seasonality or network interference).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats


def _validate_count(name: str, count: Any, is_visitor: bool = False) -> int:
    """Validate visitor and conversion counts as nonnegative integers."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer, not {type(count).__name__}."
        )
    val = int(count)
    if is_visitor:
        if val <= 0:
            raise ValueError(
                f"{name} must be at least 1 (each group must have at least one visitor), got {val}."
            )
    else:
        if val < 0:
            raise ValueError(f"{name} cannot be negative, got {val}.")
    return val


def _validate_prior(name: str, val: Any) -> float:
    """Validate prior hyperparameters as finite, strictly positive floats."""
    if isinstance(val, bool) or not isinstance(val, (int, float, np.floating, np.integer)):
        raise ValueError(
            f"{name} must be a positive numeric value, not {type(val).__name__}."
        )
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be a finite number, got {val}.")
    if fval <= 0.0:
        raise ValueError(f"{name} must be a positive number (> 0), got {val}.")
    return fval


def _validate_credible_interval(val: Any) -> float:
    """Validate credible interval probability strictly between 0 and 1."""
    if isinstance(val, bool) or not isinstance(val, (int, float, np.floating, np.integer)):
        raise ValueError(
            f"credible_interval must be numeric, not {type(val).__name__}."
        )
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"credible_interval must be a finite number, got {val}.")
    if not (0.0 < fval < 1.0):
        raise ValueError(
            f"credible_interval must be strictly between 0 and 1 (exclusive), got {val}."
        )
    return fval


def _validate_minimum_effect(val: Any) -> float:
    """Validate minimum practical effect as a finite float between 0 and 1 inclusive."""
    if isinstance(val, bool) or not isinstance(val, (int, float, np.floating, np.integer)):
        raise ValueError(
            f"minimum_effect must be numeric, not {type(val).__name__}."
        )
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"minimum_effect must be a finite number, got {val}.")
    if not (0.0 <= fval <= 1.0):
        raise ValueError(
            f"minimum_effect must be between 0 and 1, inclusive, got {val}."
        )
    return fval


def _validate_simulation_settings(
    num_samples: Any, random_seed: Any
) -> Tuple[int, Optional[int]]:
    """Validate Monte Carlo simulation parameters."""
    if isinstance(num_samples, bool) or not isinstance(num_samples, (int, np.integer)):
        raise ValueError(
            f"num_samples must be an integer, not {type(num_samples).__name__}."
        )
    samples = int(num_samples)
    if samples <= 0:
        raise ValueError(f"num_samples must be a positive integer, got {samples}.")

    seed: Optional[int] = None
    if random_seed is not None:
        if isinstance(random_seed, bool) or not isinstance(random_seed, (int, np.integer)):
            raise ValueError(
                f"random_seed must be an integer or None, not {type(random_seed).__name__}."
            )
        seed = int(random_seed)

    return samples, seed


def bayesian_ab_test(
    conversions_a: int,
    visitors_a: int,
    conversions_b: int,
    visitors_b: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_interval: float = 0.95,
    minimum_effect: float = 0.0,
    num_samples: int = 100_000,
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Perform a Bayesian A/B test for conversion rates using a Beta-Binomial conjugate model.

    Parameters:
    -----------
    conversions_a : int
        Number of conversions observed in Group A (Control). Non-negative integer <= visitors_a.
    visitors_a : int
        Total number of visitors in Group A (Control). Positive integer >= 1.
    conversions_b : int
        Number of conversions observed in Group B (Treatment). Non-negative integer <= visitors_b.
    visitors_b : int
        Total number of visitors in Group B (Treatment). Positive integer >= 1.
    prior_alpha : float, default 1.0
        Prior pseudo-conversions alpha parameter for both groups. Must be finite and > 0.
        Default 1.0 corresponds to an uninformative uniform prior when paired with prior_beta=1.0.
    prior_beta : float, default 1.0
        Prior pseudo-non-conversions beta parameter for both groups. Must be finite and > 0.
        Default 1.0 corresponds to an uninformative uniform prior when paired with prior_alpha=1.0.
    credible_interval : float, default 0.95
        Desired credible interval mass (e.g., 0.95 for a 95% credible interval).
        Must be strictly between 0 and 1.
    minimum_effect : float, default 0.0
        Minimum practically meaningful absolute conversion-rate improvement (B minus A).
        Must be a finite number between 0.0 and 1.0, inclusive.
    num_samples : int, default 100,000
        Number of Monte Carlo draws used to approximate the posterior difference distribution
        and tail probabilities. Must be a positive integer.
    random_seed : int, optional, default 42
        Random seed for the Monte Carlo generator to ensure deterministic, reproducible results.

    Returns:
    --------
    dict
        A structured dictionary containing:
        - Observed sample counts and conversion rates (conversions_a, visitors_a, conversion_rate_a, etc.)
        - Prior hyperparameters (prior_alpha, prior_beta)
        - Updated posterior Beta distribution parameters (posterior_alpha_a, posterior_beta_a, etc.)
        - Posterior mean conversion rates (posterior_mean_a, posterior_mean_b)
        - Equal-tailed credible intervals for each group's conversion rate [lower, upper]
        - Posterior mean absolute difference (B minus A)
        - Credible interval for the absolute difference (B minus A) [lower, upper]
        - Probability that Group B's rate exceeds Group A's rate (probability_b_greater_than_a)
        - Probability that Group B exceeds Group A by at least minimum_effect
          (probability_b_exceeds_a_by_minimum_effect)
        - Credible interval level, simulation settings, and method metadata.

    Raises:
    -------
    ValueError:
        If any input parameters violate the domain restrictions or types.
    """
    # 1. Input Validation
    v_a = _validate_count("visitors_a", visitors_a, is_visitor=True)
    c_a = _validate_count("conversions_a", conversions_a, is_visitor=False)
    if c_a > v_a:
        raise ValueError(
            f"conversions_a ({c_a}) cannot exceed visitors_a ({v_a})."
        )

    v_b = _validate_count("visitors_b", visitors_b, is_visitor=True)
    c_b = _validate_count("conversions_b", conversions_b, is_visitor=False)
    if c_b > v_b:
        raise ValueError(
            f"conversions_b ({c_b}) cannot exceed visitors_b ({v_b})."
        )

    p_alpha = _validate_prior("prior_alpha", prior_alpha)
    p_beta = _validate_prior("prior_beta", prior_beta)
    ci_level = _validate_credible_interval(credible_interval)
    min_eff = _validate_minimum_effect(minimum_effect)
    n_samples, seed = _validate_simulation_settings(num_samples, random_seed)

    # 2. Observed Conversion Rates
    observed_rate_a = c_a / v_a
    observed_rate_b = c_b / v_b
    observed_diff = observed_rate_b - observed_rate_a

    # 3. Posterior Beta Parameters
    # For Beta-Binomial: alpha_post = alpha_prior + conversions, beta_post = beta_prior + visitors - conversions
    post_alpha_a = p_alpha + c_a
    post_beta_a = p_beta + (v_a - c_a)

    post_alpha_b = p_alpha + c_b
    post_beta_b = p_beta + (v_b - c_b)

    # 4. Posterior Mean Conversion Rates: E[p] = alpha / (alpha + beta)
    post_mean_a = post_alpha_a / (post_alpha_a + post_beta_a)
    post_mean_b = post_alpha_b / (post_alpha_b + post_beta_b)
    post_mean_diff = post_mean_b - post_mean_a

    # 5. Exact Equal-Tailed Credible Intervals for Individual Groups
    tail_prob = (1.0 - ci_level) / 2.0
    ci_a_lower = float(stats.beta.ppf(tail_prob, post_alpha_a, post_beta_a))
    ci_a_upper = float(stats.beta.ppf(1.0 - tail_prob, post_alpha_a, post_beta_a))

    ci_b_lower = float(stats.beta.ppf(tail_prob, post_alpha_b, post_beta_b))
    ci_b_upper = float(stats.beta.ppf(1.0 - tail_prob, post_alpha_b, post_beta_b))

    # 6. Monte Carlo Sampling for Difference and Posterior Probabilities
    rng = np.random.default_rng(seed)
    samples_a = rng.beta(post_alpha_a, post_beta_a, size=n_samples)
    samples_b = rng.beta(post_alpha_b, post_beta_b, size=n_samples)

    diff_samples = samples_b - samples_a

    # Credible interval for the difference (B minus A)
    diff_ci_lower = float(np.percentile(diff_samples, tail_prob * 100.0))
    diff_ci_upper = float(np.percentile(diff_samples, (1.0 - tail_prob) * 100.0))

    # Posterior probability that Group B > Group A
    prob_b_greater_than_a = float(np.mean(diff_samples > 0.0))

    # Posterior probability that Group B exceeds Group A by at least minimum_effect
    if min_eff == 0.0:
        prob_b_exceeds_min_eff = prob_b_greater_than_a
    else:
        prob_b_exceeds_min_eff = float(np.mean(diff_samples >= min_eff))

    # 7. Assemble Structured Return Dictionary
    return {
        # Observed Sample Data
        "visitors_a": v_a,
        "conversions_a": c_a,
        "conversion_rate_a": float(observed_rate_a),
        "observed_conversion_rate_a": float(observed_rate_a),
        "visitors_b": v_b,
        "conversions_b": c_b,
        "conversion_rate_b": float(observed_rate_b),
        "observed_conversion_rate_b": float(observed_rate_b),
        "observed_absolute_difference": float(observed_diff),
        "observed_difference": float(observed_diff),

        # Prior Hyperparameters
        "prior_alpha": float(p_alpha),
        "prior_beta": float(p_beta),

        # Posterior Parameters
        "posterior_alpha_a": float(post_alpha_a),
        "posterior_beta_a": float(post_beta_a),
        "posterior_alpha_b": float(post_alpha_b),
        "posterior_beta_b": float(post_beta_b),

        # Posterior Estimates for Individual Groups
        "posterior_mean_a": float(post_mean_a),
        "posterior_mean_b": float(post_mean_b),
        "credible_interval_a": [ci_a_lower, ci_a_upper],
        "credible_interval_b": [ci_b_lower, ci_b_upper],

        # Comparative Posterior Metrics (B minus A)
        "posterior_mean_difference": float(post_mean_diff),
        "credible_interval_difference": [diff_ci_lower, diff_ci_upper],
        "probability_b_greater_than_a": float(prob_b_greater_than_a),
        "probability_b_exceeds_a_by_minimum_effect": float(prob_b_exceeds_min_eff),

        # Metadata and Configuration
        "credible_interval_level": float(ci_level),
        "credible_interval": float(ci_level),
        "minimum_effect": float(min_eff),
        "simulation_method": "monte_carlo",
        "num_samples": n_samples,
        "random_seed": seed,
    }
