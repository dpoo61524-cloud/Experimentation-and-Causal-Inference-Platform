
import pandas as pd
from statsmodels.stats.proportion import confint_proportions_2indep
from statsmodels.stats.proportion import proportions_ztest
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_ROOT / "data" / "hillstrom.csv")

# Select the men's email group and the control group
mens = df[df["segment"] == "Mens E-Mail"]
control = df[df["segment"] == "No E-Mail"]

# Count purchases and customers
successes = [
    int(mens["conversion"].sum()),
    int(control["conversion"].sum())
]

sample_sizes = [len(mens), len(control)]

# Perform a two-proportion z-test
z_stat, p_value = proportions_ztest(
    count=successes,
    nobs=sample_sizes,
    alternative="two-sided"
)

print("Men's email conversions:", successes[0])
print("Control conversions:", successes[1])
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.10g}")

if p_value < 0.05:
    print("Result: Statistically significant at the 5% level.")
else:
    print("Result: Not statistically significant at the 5% level.")


ci_low, ci_high = confint_proportions_2indep(
    count1=successes[0],
    nobs1=sample_sizes[0],
    count2=successes[1],
    nobs2=sample_sizes[1],
    method="newcomb",
    compare="diff",
    alpha=0.05
)

difference = (
    successes[0] / sample_sizes[0]
    - successes[1] / sample_sizes[1]
)

print(f"Conversion-rate difference: {difference * 100:.3f} percentage points")
print(
    f"95% confidence interval: "
    f"[{ci_low * 100:.3f}, {ci_high * 100:.3f}] percentage points"
)
