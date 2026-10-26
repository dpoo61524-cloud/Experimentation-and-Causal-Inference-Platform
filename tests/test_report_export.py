
from pathlib import Path
import sys
import csv

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from report_export import export_report_to_csv


def test_export_report_creates_csv_file(tmp_path):
    report = {
        "experiment_name": "Test Experiment",
        "conversion_rate_a": 0.10,
        "conversion_rate_b": 0.05,
        "p_value": 0.02,
        "confidence_interval": [0.01, 0.09],
    }

    output_path = tmp_path / "test_report.csv"

    result = export_report_to_csv(report, output_path)

    assert result.exists()
    assert result.suffix == ".csv"


def test_exported_csv_contains_report_values(tmp_path):
    report = {
        "experiment_name": "Email Campaign",
        "conversion_rate_a": 0.10,
        "conversion_rate_b": 0.05,
        "p_value": 0.02,
        "confidence_interval": [0.01, 0.09],
    }

    output_path = tmp_path / "report.csv"
    export_report_to_csv(report, output_path)

    with output_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as csv_file:
        rows = list(csv.reader(csv_file))

    assert rows[0] == ["Metric", "Value"]
    assert ["experiment_name", "Email Campaign"] in rows
    assert ["p_value", "0.02"] in rows
    assert ["confidence_interval", "0.010000 to 0.090000"] in rows


def test_export_creates_missing_folder(tmp_path):
    report = {"experiment_name": "Folder Test"}
    output_path = tmp_path / "reports" / "nested" / "report.csv"

    export_report_to_csv(report, output_path)

    assert output_path.exists()


def test_export_returns_path_object(tmp_path):
    report = {"experiment_name": "Path Test"}
    output_path = tmp_path / "report.csv"

    result = export_report_to_csv(report, output_path)

    assert isinstance(result, Path)
    assert result == output_path
