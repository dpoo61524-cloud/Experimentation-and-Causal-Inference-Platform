"""PyMC-Powered Bayesian A/B Testing Engine.

Provides MCMC and conjugate inference for binary A/B conversion experiments,
computing posterior distributions, probability of superiority P(B > A),
expected loss, credible intervals, and relative lift.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import scipy.stats as stats

# Check if PyMC is available
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


def run_pymc_bayesian_ab(
    conversions_a: int,
    visitors_a: int,
    conversions_b: int,
    visitors_b: int,
    alpha_prior_a: float = 1.0,
    beta_prior_a: float = 1.0,
    alpha_prior_b: float = 1.0,
    beta_prior_b: float = 1.0,
    draws: int = 2000,
    tune: int = 1000,
    random_seed: Optional[int] = 42,
    minimum_effect: float = 0.0,
    credible_level: float = 0.95,
) -> Dict[str, Any]:
    """Execute Bayesian A/B testing using PyMC MCMC sampling with automatic conjugate fallback."""
    # Input validation
    if visitors_a <= 0 or visitors_b <= 0:
        raise ValueError("Visitors count for both groups must be strictly positive.")
    if conversions_a < 0 or conversions_a > visitors_a:
        raise ValueError("Conversions A must be between 0 and visitors A.")
    if conversions_b < 0 or conversions_b > visitors_b:
        raise ValueError("Conversions B must be between 0 and visitors B.")
    if alpha_prior_a <= 0 or beta_prior_a <= 0 or alpha_prior_b <= 0 or beta_prior_b <= 0:
        raise ValueError("Beta prior hyperparameters must be strictly positive.")
    if not (0.5 < credible_level < 1.0):
        raise ValueError("Credible level must be between 0.5 and 1.0.")

    # Calculate exact posterior hyperparameters for conjugate updating
    post_alpha_a = alpha_prior_a + conversions_a
    post_beta_a = beta_prior_a + (visitors_a - conversions_a)
    post_alpha_b = alpha_prior_b + conversions_b
    post_beta_b = beta_prior_b + (visitors_b - conversions_b)

    # Attempt PyMC MCMC execution if available
    pymc_executed = False
    trace_samples = None

    if PYMC_AVAILABLE:
        try:
            with pm.Model() as model:
                p_a = pm.Beta("p_a", alpha=alpha_prior_a, beta=beta_prior_a)
                p_b = pm.Beta("p_b", alpha=alpha_prior_b, beta=beta_prior_b)

                pm.Binomial("obs_a", n=visitors_a, p=p_a, observed=conversions_a)
                pm.Binomial("obs_b", n=visitors_b, p=p_b, observed=conversions_b)

                pm.Deterministic("diff", p_b - p_a)
                pm.Deterministic("rel_lift", (p_b - p_a) / p_a)

                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    random_seed=random_seed,
                    progressbar=False,
                    return_inferencedata=True,
                )

            p_a_samples = trace.posterior["p_a"].values.flatten()
            p_b_samples = trace.posterior["p_b"].values.flatten()
            pymc_executed = True
        except Exception:
            pymc_executed = False

    # High-precision Monte Carlo fallback using exact conjugate posteriors
    if not pymc_executed:
        rng = np.random.default_rng(random_seed)
        n_samples = max(draws * 2, 50000)
        p_a_samples = rng.beta(post_alpha_a, post_beta_a, size=n_samples)
        p_b_samples = rng.beta(post_alpha_b, post_beta_b, size=n_samples)

    diff_samples = p_b_samples - p_a_samples
    rel_lift_samples = diff_samples / p_a_samples

    # Metrics
    prob_b_beats_a = float(np.mean(diff_samples > 0.0))
    prob_b_beats_a_mde = float(np.mean(diff_samples >= minimum_effect))

    alpha_tail = (1.0 - credible_level) / 2.0
    ci_a = [
        float(stats.beta.ppf(alpha_tail, post_alpha_a, post_beta_a)),
        float(stats.beta.ppf(1.0 - alpha_tail, post_alpha_a, post_beta_a)),
    ]
    ci_b = [
        float(stats.beta.ppf(alpha_tail, post_alpha_b, post_beta_b)),
        float(stats.beta.ppf(1.0 - alpha_tail, post_alpha_b, post_beta_b)),
    ]
    ci_diff = [
        float(np.percentile(diff_samples, alpha_tail * 100)),
        float(np.percentile(diff_samples, (1.0 - alpha_tail) * 100)),
    ]
    ci_rel_lift = [
        float(np.percentile(rel_lift_samples, alpha_tail * 100)),
        float(np.percentile(rel_lift_samples, (1.0 - alpha_tail) * 100)),
    ]

    # Expected Loss (Risk of choosing B if B < A, or choosing A if A < B)
    loss_b = float(np.mean(np.maximum(p_a_samples - p_b_samples, 0.0)))
    loss_a = float(np.mean(np.maximum(p_b_samples - p_a_samples, 0.0)))

    return {
        "engine": "pymc_mcmc" if pymc_executed else "beta_binomial_conjugate",
        "pymc_available": PYMC_AVAILABLE,
        "sample_counts": {
            "group_a": {"visitors": visitors_a, "conversions": conversions_a, "rate": conversions_a / visitors_a},
            "group_b": {"visitors": visitors_b, "conversions": conversions_b, "rate": conversions_b / visitors_b},
        },
        "priors": {
            "group_a": {"alpha": alpha_prior_a, "beta": beta_prior_a},
            "group_b": {"alpha": alpha_prior_b, "beta": beta_prior_b},
        },
        "posteriors": {
            "group_a": {"alpha": post_alpha_a, "beta": post_beta_a, "mean": post_alpha_a / (post_alpha_a + post_beta_a), "credible_interval": ci_a},
            "group_b": {"alpha": post_alpha_b, "beta": post_beta_b, "mean": post_alpha_b / (post_alpha_b + post_beta_b), "credible_interval": ci_b},
        },
        "comparative_metrics": {
            "probability_b_beats_a": prob_b_beats_a,
            "probability_b_beats_a_by_minimum_effect": prob_b_beats_a_mde,
            "expected_difference": float(np.mean(diff_samples)),
            "difference_credible_interval": ci_diff,
            "relative_lift_mean_pct": float(np.mean(rel_lift_samples)) * 100.0,
            "relative_lift_credible_interval_pct": [ci_rel_lift[0] * 100.0, ci_rel_lift[1] * 100.0],
            "expected_loss_choosing_b": loss_b,
            "expected_loss_choosing_a": loss_a,
            "recommendation": (
                "Adopt Group B" if prob_b_beats_a >= 0.95 and loss_b < 0.001
                else "Remain with Group A" if prob_b_beats_a <= 0.05
                else "Continue Experimentation (Inconclusive)"
            ),
        },
    }
