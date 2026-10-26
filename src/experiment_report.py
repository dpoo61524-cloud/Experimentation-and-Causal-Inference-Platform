
from statistics_engine import compare_conversion_rates


def generate_experiment_report(
    experiment_name: str,
    conversions_a: int,
    visitors_a: int,
    conversions_b: int,
    visitors_b: int,
) -> dict:
    """Generate a readable report comparing two experiment groups."""

    results = compare_conversion_rates(
        conversions_a=conversions_a,
        visitors_a=visitors_a,
        conversions_b=conversions_b,
        visitors_b=visitors_b,
    )

    rate_a = results["conversion_rate_a"]
    rate_b = results["conversion_rate_b"]
    difference = results["absolute_difference"]
    p_value = results["p_value"]
    ci_low, ci_high = results["confidence_interval_95"]

    if rate_b != 0:
        relative_lift = ((rate_a - rate_b) / rate_b) * 100
    else:
        relative_lift = None

    if results["statistically_significant"]:
        conclusion = (
            "The difference is statistically significant "
            "at the 5% significance level."
        )
    else:
        conclusion = (
            "There is insufficient evidence of a statistically "
            "significant difference at the 5% significance level."
        )

    return {
        "experiment_name": experiment_name,
        "group_a_conversion_rate": rate_a,
        "group_b_conversion_rate": rate_b,
        "absolute_difference_percentage_points": difference * 100,
        "relative_lift_percent": relative_lift,
        "p_value": p_value,
        "confidence_interval_percentage_points": [
            ci_low * 100,
            ci_high * 100,
        ],
        "statistically_significant": results["statistically_significant"],
        "conclusion": conclusion,
    }


def print_experiment_report(report: dict) -> None:
    """Print the experiment results in a readable format."""

    print("\n" + "=" * 55)
    print(f"EXPERIMENT REPORT: {report['experiment_name']}")
    print("=" * 55)

    print(
        f"Group A conversion rate: "
        f"{report['group_a_conversion_rate']:.2%}"
    )
    print(
        f"Group B conversion rate: "
        f"{report['group_b_conversion_rate']:.2%}"
    )

    print(
        "Absolute difference: "
        f"{report['absolute_difference_percentage_points']:.3f} "
        "percentage points"
    )

    lift = report["relative_lift_percent"]
    if lift is not None:
        print(f"Relative lift: {lift:.2f}%")
    else:
        print("Relative lift: undefined (Group B conversion rate is zero)")

    print(f"P-value: {report['p_value']:.6g}")

    ci_low, ci_high = report["confidence_interval_percentage_points"]
    print(f"95% confidence interval: [{ci_low:.3f}, {ci_high:.3f}] percentage points")

    print(f"Conclusion: {report['conclusion']}")
    print("=" * 55)
