
import sys
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# Locate the project and source folders
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "hillstrom.csv"

# Allow Python to import our reusable module
sys.path.insert(0, str(SRC_PATH))

from multiple_comparisons import correct_multiple_p_values


def main():
    # Load the Hillstrom dataset
    df = pd.read_csv(DATA_PATH)

    # Select the three experiment groups
    groups = {
        "Mens E-Mail": df[df["segment"] == "Mens E-Mail"],
        "Womens E-Mail": df[df["segment"] == "Womens E-Mail"],
        "No E-Mail": df[df["segment"] == "No E-Mail"],
    }

    # Compare every pair of groups
    comparisons = [
        ("Mens E-Mail", "No E-Mail"),
        ("Womens E-Mail", "No E-Mail"),
        ("Mens E-Mail", "Womens E-Mail"),
    ]

    raw_p_values = []

    for group1_name, group2_name in comparisons:
        group1 = groups[group1_name]
        group2 = groups[group2_name]

        successes = [
            int(group1["conversion"].sum()),
            int(group2["conversion"].sum()),
        ]

        sample_sizes = [len(group1), len(group2)]

        _, p_value = proportions_ztest(
            count=successes,
            nobs=sample_sizes,
            alternative="two-sided",
        )

        raw_p_values.append(float(p_value))

    # Apply multiple-comparison corrections
    corrected_results = correct_multiple_p_values(raw_p_values)

    # Display the results
    for (group1, group2), result in zip(comparisons, corrected_results):
        print(f"\nComparison: {group1} vs {group2}")
        print(f"Raw p-value: {result['raw_p_value']:.6e}")

        print(
            "Bonferroni adjusted p-value: "
            f"{result['bonferroni_adjusted_p_value']:.6e}"
        )
        print(
            "Significant after Bonferroni: "
            f"{result['bonferroni_significant']}"
        )

        print(
            "BH adjusted p-value: "
            f"{result['bh_adjusted_p_value']:.6e}"
        )
        print(f"Significant after BH: {result['bh_significant']}")


if __name__ == "__main__":
    main()
