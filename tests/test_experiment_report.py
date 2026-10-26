
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_report import (
    generate_experiment_report,
    print_experiment_report,
)


def test_report_contains_experiment_results():
    report = generate_experiment_report(
        experiment_name="Men's Email vs No Email",
        conversions_a=267,
        visitors_a=21307,
        conversions_b=122,
        visitors_b=21306,
    )

    assert report["experiment_name"] == "Men's Email vs No Email"
    assert report["group_a_conversion_rate"] > report["group_b_conversion_rate"]
    assert report["p_value"] < 0.05
    assert report["statistically_significant"] is True


def test_report_has_confidence_interval():
    report = generate_experiment_report(
        experiment_name="Sample Experiment",
        conversions_a=20,
        visitors_a=500,
        conversions_b=10,
        visitors_b=500,
    )

    low, high = report["confidence_interval_percentage_points"]

    assert low < high


def test_report_can_be_printed():
    report = generate_experiment_report(
        experiment_name="Sample Experiment",
        conversions_a=20,
        visitors_a=500,
        conversions_b=10,
        visitors_b=500,
    )

    print_experiment_report(report)
