
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_report import (
    generate_experiment_report,
    print_experiment_report,
)


def main():
    # Load the Hillstrom dataset.
    data_path = PROJECT_ROOT / "data" / "hillstrom.csv"
    df = pd.read_csv(data_path)

    # Select the treatment and control groups.
    treatment = df[df["segment"] == "Mens E-Mail"]
    control = df[df["segment"] == "No E-Mail"]

    if treatment.empty or control.empty:
        raise ValueError("Treatment or control group is missing from the dataset.")

    # Calculate conversions and visitor counts.
    treatment_visitors = len(treatment)
    control_visitors = len(control)

    treatment_conversions = int(treatment["conversion"].sum())
    control_conversions = int(control["conversion"].sum())

    # Generate the experiment report.
    report = generate_experiment_report(
        experiment_name="Men's E-Mail vs No E-Mail",
        conversions_a=treatment_conversions,
        visitors_a=treatment_visitors,
        conversions_b=control_conversions,
        visitors_b=control_visitors,
    )

    # Display the report in the terminal.
    print_experiment_report(report)

    # Create the reports folder if it doesn't exist.
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Format the report for saving.
    ci_low, ci_high = report["confidence_interval_percentage_points"]
    lift = report["relative_lift_percent"]

    report_text = f"""
EXPERIMENT REPORT
=================

Experiment: {report['experiment_name']}

GROUP RESULTS
-------------
Group A conversion rate: {report['group_a_conversion_rate']:.2%}
Group B conversion rate: {report['group_b_conversion_rate']:.2%}

EFFECT
------
Absolute difference: {report['absolute_difference_percentage_points']:.3f} percentage points
Relative lift: {lift:.2f}%

STATISTICAL ANALYSIS
--------------------
P-value: {report['p_value']:.6g}
95% confidence interval: [{ci_low:.3f}, {ci_high:.3f}] percentage points
Statistically significant: {report['statistically_significant']}

CONCLUSION
----------
{report['conclusion']}
"""

    # Save the report to a text file.
    report_path = reports_dir / "mens_email_experiment_report.txt"
    report_path.write_text(report_text.strip() + "\n", encoding="utf-8")

    print(f"\nReport saved successfully to:\n{report_path}")


if __name__ == "__main__":
    main()
