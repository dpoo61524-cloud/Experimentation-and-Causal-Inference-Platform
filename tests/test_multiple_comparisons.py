
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


# Locate the project folder reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "hillstrom.csv"

df = pd.read_csv(DATA_PATH)

# Identify the three experiment groups
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

    z_stat, p_value = proportions_ztest(
        count=successes,
        nobs=sample_sizes,
        alternative="two-sided",
    )

    raw_p_values.append(p_value)

# Correct for multiple comparisons
bonferroni = multipletests(
    raw_p_values, alpha=0.05, method="bonferroni"
)

benjamini_hochberg = multipletests(
    raw_p_values, alpha=0.05, method="fdr_bh"
)

# Display the results
for i, (group1, group2) in enumerate(comparisons):
    print(f"\nComparison: {group1} vs {group2}")
    print(f"Raw p-value: {raw_p_values[i]:.6e}")

    print(
        f"Bonferroni adjusted p-value: "
        f"{bonferroni[1][i]:.6e}"
    )
    print(f"Significant after Bonferroni: {bonferroni[0][i]}")

    print(
        f"BH adjusted p-value: "
        f"{benjamini_hochberg[1][i]:.6e}"
    )
    print(f"Significant after BH: {benjamini_hochberg[0][i]}")
