
import csv
from pathlib import Path


def export_report_to_csv(report: dict, output_path: str | Path) -> Path:
    """Export experiment report results to a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(["Metric", "Value"])

        for key, value in report.items():
            if isinstance(value, list):
                value = " to ".join(f"{item:.6f}" for item in value)

            writer.writerow([key, value])

    return output_path
