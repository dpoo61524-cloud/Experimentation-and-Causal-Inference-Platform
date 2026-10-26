
from pathlib import Path
import sys

import pandas as pd

# Set up project paths and import modules from src.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_report import (
    generate_experiment_report,
    print_experiment_report,
)
from report_export import export_report_to_csv


def main():
    # 1. Load the Hillstrom dataset.
    data_path = PROJECT_ROOT / "data" / "hillstrom.csv"
    df = pd.read_csv(data_path)

    # 2. Select treatment and control groups.
    treatment = df[df["segment"] == "Mens E-Mail"]
    control = df[df["segment"] == "No E-Mail"]

    if treatment.empty or control.empty:
        raise ValueError(
            "Treatment or control group is missing from the dataset."
        )

    # 3. Calculate visitor counts and conversions.
    treatment_visitors = len(treatment)
    control_visitors = len(control)

    treatment_conversions = int(treatment["conversion"].sum())
    control_conversions = int(control["conversion"].sum())

    # 4. Generate the experiment report.
    report = generate_experiment_report(
        experiment_name="Men's E-Mail vs No E-Mail",
        conversions_a=treatment_conversions,
        visitors_a=treatment_visitors,
        conversions_b=control_conversions,
        visitors_b=control_visitors,
    )

    # 5. Display results in the terminal.
    print_experiment_report(report)

    # 6. Create the reports folder if it doesn't exist.
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 7. Prepare the text report.
    ci_low, ci_high = report["confidence_interval_percentage_points"]
    lift = report["relative_lift_percent"]

    if lift is not None:
        lift_text = f"{lift:.2f}%"
    else:
        lift_text = "Undefined (control conversion rate is zero)"

    report_text = f"""
EXPERIMENT REPORT
=================

Experiment: {report['experiment_name']}

GROUP RESULTS
-------------
Treatment visitors: {treatment_visitors}
Treatment conversions: {treatment_conversions}
Control visitors: {control_visitors}
Control conversions: {control_conversions}

Group A conversion rate: {report['group_a_conversion_rate']:.2%}
Group B conversion rate: {report['group_b_conversion_rate']:.2%}

EFFECT
------
Absolute difference: {report['absolute_difference_percentage_points']:.3f} percentage points
Relative lift: {lift_text}

STATISTICAL ANALYSIS
--------------------
P-value: {report['p_value']:.6g}
95% confidence interval: [{ci_low:.3f}, {ci_high:.3f}] percentage points
Statistically significant: {report['statistically_significant']}

CONCLUSION
----------
{report['conclusion']}
"""

    # 8. Save the text report.
    report_path = reports_dir / "mens_email_experiment_report.txt"
    report_path.write_text(
        report_text.strip() + "\n",
        encoding="utf-8",
    )

    print(f"\nText report saved successfully to:\n{report_path}")

    # 9. Export the same report to CSV.
    csv_path = reports_dir / "mens_email_experiment_report.csv"
    export_report_to_csv(report, csv_path)

    print(f"\nCSV report saved successfully to:\n{csv_path}")


if __name__ == "__main__":
    main()
