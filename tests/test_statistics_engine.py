
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
    two_sample_ttest,
    chi_square_test,
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


# --- Two-Sample t-Test Unit Tests ---

def test_two_sample_ttest_known_example():
    """Verify Welch's t-test on a known reference dataset."""
    sample_a = [12, 15, 14, 10, 13, 14, 16]
    sample_b = [9, 10, 11, 8, 12, 10, 9]

    result = two_sample_ttest(sample_a, sample_b)

    assert result["sample_size_a"] == 7
    assert result["sample_size_b"] == 7
    assert result["mean_a"] == pytest.approx(13.42857, rel=1e-4)
    assert result["mean_b"] == pytest.approx(9.85714, rel=1e-4)
    assert result["difference_in_means"] == pytest.approx(3.57143, rel=1e-4)
    assert result["t_statistic"] == pytest.approx(3.93648, rel=1e-4)
    assert result["p_value"] == pytest.approx(0.00252, rel=1e-3)
    assert result["degrees_of_freedom"] == pytest.approx(10.5419, rel=1e-3)
    assert result["statistically_significant"] is True
    assert result["confidence_interval_95"][0] < result["confidence_interval_95"][1]
    assert result["confidence_interval_95"][0] == pytest.approx(1.5639, rel=1e-3)
    assert result["confidence_interval_95"][1] == pytest.approx(5.5789, rel=1e-3)


def test_two_sample_ttest_unequal_sample_sizes():
    """Verify t-test functions correctly with unequal sample sizes."""
    sample_a = [10.5, 11.2, 9.8, 10.1, 12.0, 11.5, 10.9, 11.1]
    sample_b = [8.2, 8.5, 9.1, 8.8]

    result = two_sample_ttest(sample_a, sample_b)

    assert result["sample_size_a"] == 8
    assert result["sample_size_b"] == 4
    assert result["mean_a"] > result["mean_b"]
    assert result["t_statistic"] > 0
    assert result["p_value"] < 0.05
    assert result["statistically_significant"] is True


def test_two_sample_ttest_alternatives():
    """Verify one-sided and two-sided alternative hypotheses."""
    sample_a = [15.0, 16.0, 14.5, 15.5]
    sample_b = [10.0, 11.0, 9.5, 10.5]

    two_sided = two_sample_ttest(sample_a, sample_b, alternative="two-sided")
    greater = two_sample_ttest(sample_a, sample_b, alternative="greater")
    less = two_sample_ttest(sample_a, sample_b, alternative="less")

    assert two_sided["p_value"] == pytest.approx(greater["p_value"] * 2)
    assert less["p_value"] > 0.99
    assert greater["alternative"] == "greater"
    assert less["alternative"] == "less"


def test_two_sample_ttest_constant_values_where_supported():
    """Verify t-test when one sample has constant values and the other has variance."""
    sample_a = [10.0, 10.0, 10.0, 10.0]
    sample_b = [5.0, 8.0, 12.0, 15.0]

    result = two_sample_ttest(sample_a, sample_b)
    assert result["sample_size_a"] == 4
    assert result["sample_size_b"] == 4
    assert isinstance(result["t_statistic"], float)
    assert isinstance(result["p_value"], float)


def test_two_sample_ttest_both_constant_rejected():
    """Verify that if both samples have zero variance, a ValueError is raised."""
    with pytest.raises(ValueError, match="zero variance"):
        two_sample_ttest([5, 5, 5], [5, 5, 5])


def test_two_sample_ttest_insufficient_sample_sizes():
    """Verify that samples with fewer than 2 observations are rejected."""
    with pytest.raises(ValueError, match="at least 2 observations"):
        two_sample_ttest([10], [10, 12, 14])

    with pytest.raises(ValueError, match="at least 2 observations"):
        two_sample_ttest([], [10, 12])


def test_two_sample_ttest_invalid_inputs():
    """Verify rejection of non-numeric, missing, boolean, NaN, and invalid options."""
    with pytest.raises(ValueError, match="non-numeric"):
        two_sample_ttest([1, "bad", 3], [4, 5, 6])

    with pytest.raises(ValueError, match="invalid non-numeric or missing"):
        two_sample_ttest([1, None, 3], [4, 5, 6])

    with pytest.raises(ValueError, match="invalid non-numeric or missing"):
        two_sample_ttest([True, 2, 3], [4, 5, 6])

    with pytest.raises(ValueError, match="NaN or infinite"):
        two_sample_ttest([float("nan"), 2, 3], [4, 5, 6])

    with pytest.raises(ValueError, match="Alternative hypothesis must be one of"):
        two_sample_ttest([1, 2, 3], [4, 5, 6], alternative="invalid")

    with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
        two_sample_ttest([1, 2, 3], [4, 5, 6], alpha=1.5)


# --- Chi-Square Test Unit Tests ---

def test_chi_square_known_example():
    """Verify chi-square test on a known reference 2x2 table."""
    table = [
        [50, 30],
        [20, 40],
    ]
    result = chi_square_test(table)

    assert result["chi_square_statistic"] == pytest.approx(10.52917, rel=1e-4)
    assert result["p_value"] == pytest.approx(0.001175, rel=1e-3)
    assert result["degrees_of_freedom"] == 1
    assert result["statistically_significant"] is True
    assert result["warning"] is None
    assert result["expected_frequencies"][0] == pytest.approx([40.0, 40.0])
    assert result["expected_frequencies"][1] == pytest.approx([30.0, 30.0])


def test_chi_square_larger_table():
    """Verify chi-square test on a 2x3 contingency table."""
    table = [
        [20, 30, 25],
        [40, 15, 35],
    ]
    result = chi_square_test(table)

    assert result["degrees_of_freedom"] == 2
    assert len(result["expected_frequencies"]) == 2
    assert len(result["expected_frequencies"][0]) == 3
    assert result["chi_square_statistic"] > 0
    assert 0 <= result["p_value"] <= 1


def test_chi_square_zero_counts_valid():
    """Verify chi-square test with zero count cell where marginal sums remain positive."""
    table = [
        [0, 20],
        [10, 30],
    ]
    result = chi_square_test(table)

    assert result["degrees_of_freedom"] == 1
    assert result["chi_square_statistic"] == pytest.approx(4.335, rel=1e-3)
    assert result["p_value"] == pytest.approx(0.03734, rel=1e-3)
    assert result["warning"] is not None
    assert "less than 5" in result["warning"]


def test_chi_square_small_expected_counts_warning():
    """Verify warning is returned when expected frequencies are below 5."""
    table = [
        [2, 3],
        [4, 5],
    ]
    result = chi_square_test(table)

    assert result["warning"] is not None
    assert "Warning" in result["warning"]
    assert "less than 5" in result["warning"]


def test_chi_square_invalid_dimensions():
    """Verify rejection of 1D, single-row, single-col, or jagged tables."""
    # 1D list
    with pytest.raises(ValueError, match="not a valid sequence"):
        chi_square_test([10, 20])

    # Single row
    with pytest.raises(ValueError, match="at least 2 rows"):
        chi_square_test([[10, 20]])

    # Single column
    with pytest.raises(ValueError, match="at least 2 columns"):
        chi_square_test([[10], [20]])

    # Inconsistent column counts
    with pytest.raises(ValueError, match="Inconsistent column count"):
        chi_square_test([[10, 20], [10, 20, 30]])


def test_chi_square_negative_counts():
    """Verify rejection of negative numbers."""
    with pytest.raises(ValueError, match="cannot be negative"):
        chi_square_test([[10, -5], [20, 30]])


def test_chi_square_all_zero_and_zero_marginal_sums():
    """Verify rejection of all-zero tables and empty marginal rows/cols."""
    # All zeros
    with pytest.raises(ValueError, match="at least one observation"):
        chi_square_test([[0, 0], [0, 0]])

    # Zero row sum
    with pytest.raises(ValueError, match="Each row in the contingency table"):
        chi_square_test([[0, 0], [10, 20]])

    # Zero column sum
    with pytest.raises(ValueError, match="Each column in the contingency table"):
        chi_square_test([[0, 10], [0, 20]])


def test_chi_square_invalid_inputs():
    """Verify rejection of non-numeric, None, boolean, NaN, and invalid alpha."""
    with pytest.raises(ValueError, match="non-numeric"):
        chi_square_test([[10, "bad"], [20, 30]])

    with pytest.raises(ValueError, match="invalid non-numeric or missing"):
        chi_square_test([[10, None], [20, 30]])

    with pytest.raises(ValueError, match="invalid non-numeric or missing"):
        chi_square_test([[True, 10], [20, 30]])

    with pytest.raises(ValueError, match="NaN or infinite"):
        chi_square_test([[float("nan"), 10], [20, 30]])

    with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
        chi_square_test([[10, 20], [30, 40]], alpha=1.5)
