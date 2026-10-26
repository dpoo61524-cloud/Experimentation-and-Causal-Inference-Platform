
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from statistics_engine import (
    compare_conversion_rates,
    calculate_sample_size,
)


def test_valid_conversion_data():
    result = compare_conversion_rates(
        conversions_a=20,
        visitors_a=1000,
        conversions_b=10,
        visitors_b=1000,
    )

    assert result["conversion_rate_a"] == 0.02
    assert result["conversion_rate_b"] == 0.01
    assert result["absolute_difference"] == 0.01
    assert result["p_value"] >= 0
    assert result["p_value"] <= 1


def test_zero_visitors_are_rejected():
    with pytest.raises(ValueError):
        compare_conversion_rates(0, 0, 10, 100)


def test_negative_visitors_are_rejected():
    with pytest.raises(ValueError):
        compare_conversion_rates(10, -100, 5, 100)


def test_conversions_cannot_exceed_visitors():
    with pytest.raises(ValueError):
        compare_conversion_rates(120, 100, 10, 100)


def test_negative_conversions_are_rejected():
    with pytest.raises(ValueError):
        compare_conversion_rates(-1, 100, 10, 100)

def test_sample_size_calculation():
    result = calculate_sample_size(0.10, 0.12)

    assert result["sample_size_per_group"] == 3835
    assert result["total_sample_size"] == 7670
    assert result["power"] == 0.80


def test_invalid_baseline_rate_is_rejected():
    with pytest.raises(ValueError):
        calculate_sample_size(0, 0.12)


def test_equal_conversion_rates_are_rejected():
    with pytest.raises(ValueError):
        calculate_sample_size(0.10, 0.10)
