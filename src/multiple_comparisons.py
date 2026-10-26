
from statsmodels.stats.multitest import multipletests


def correct_multiple_p_values(p_values, alpha=0.05):
    """Apply Bonferroni and Benjamini-Hochberg corrections."""

    if len(p_values) == 0:
        raise ValueError("p_values must not be empty.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    for p_value in p_values:
        if not 0 <= p_value <= 1:
            raise ValueError("Every p-value must be between 0 and 1.")

    bonferroni = multipletests(
        p_values,
        alpha=alpha,
        method="bonferroni",
    )

    benjamini_hochberg = multipletests(
        p_values,
        alpha=alpha,
        method="fdr_bh",
    )

    results = []

    for i, p_value in enumerate(p_values):
        results.append({
            "raw_p_value": float(p_value),
            "bonferroni_adjusted_p_value": float(bonferroni[1][i]),
            "bonferroni_significant": bool(bonferroni[0][i]),
            "bh_adjusted_p_value": float(benjamini_hochberg[1][i]),
            "bh_significant": bool(benjamini_hochberg[0][i]),
        })

    return results
