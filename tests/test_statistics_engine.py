
from pathlib import Path
import sys

import pytest

# Make the src folder accessible to Python.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from statistics_engine import (
    compare_conversion_rates,
    calculate_sample_size,
    calculate_conversion_confidence_interval,
)


def test_compare_conversion_rates():
    """Check the conversion-rate comparison."""

    result = compare_conversion_rates(
        conversions_a=267,
        visitors_a=21307,
        conversions_b=122,
        visitors_b=21306,
    )

    assert result["conversion_rate_a"] > result["conversion_rate_b"]
    assert result["p_value"] < 0.05
    assert result["statistically_significant"] is True
    assert result["confidence_interval_95"][0] < result["confidence_interval_95"][1]


def test_conversion_rates_are_calculated_correctly():
    result = compare_conversion_rates(10, 100, 5, 100)

    assert result["conversion_rate_a"] == pytest.approx(0.10)
    assert result["conversion_rate_b"] == pytest.approx(0.05)
    assert result["absolute_difference"] == pytest.approx(0.05)


def test_zero_visitors_are_rejected():
    with pytest.raises(ValueError):
        compare_conversion_rates(0, 0, 5, 100)


def test_conversions_cannot_exceed_visitors():
    with pytest.raises(ValueError):
        compare_conversion_rates(101, 100, 5, 100)


def test_sample_size_calculation():
    result = calculate_sample_size(0.10, 0.12)

    assert result["sample_size_per_group"] == 3835
    assert result["total_sample_size"] == 7670


def test_equal_rates_are_rejected_for_sample_size():
    with pytest.raises(ValueError):
        calculate_sample_size(0.10, 0.10)


def test_invalid_baseline_rate_is_rejected():
    with pytest.raises(ValueError):
        calculate_sample_size(0, 0.12)


def test_conversion_confidence_interval():
    result = calculate_conversion_confidence_interval(
        conversions=267,
        visitors=21307,
    )

    assert result["conversion_rate"] == pytest.approx(267 / 21307)
    assert result["confidence_level"] == 0.95

    low, high = result["confidence_interval"]
    assert 0 <= low < high <= 1
    assert low < result["conversion_rate"] < high


def test_zero_visitors_rejected_for_confidence_interval():
    with pytest.raises(ValueError):
        calculate_conversion_confidence_interval(0, 0)


def test_conversions_cannot_exceed_visitors_for_confidence_interval():
    with pytest.raises(ValueError):
        calculate_conversion_confidence_interval(101, 100)


def test_invalid_confidence_level_is_rejected():
    with pytest.raises(ValueError):
        calculate_conversion_confidence_interval(
            conversions=10,
            visitors=100,
            confidence_level=1.0,
        )
