
from pathlib import Path
import sys

import pytest

# Add the src folder to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multiple_comparisons import correct_multiple_p_values


def test_returns_results_for_each_p_value():
    results = correct_multiple_p_values([0.01, 0.03, 0.20])

    assert len(results) == 3
    assert all("raw_p_value" in result for result in results)


def test_bonferroni_adjustment():
    results = correct_multiple_p_values([0.01, 0.03, 0.20])

    assert results[0]["bonferroni_adjusted_p_value"] == pytest.approx(0.03)
    assert results[1]["bonferroni_adjusted_p_value"] == pytest.approx(0.09)
    assert results[2]["bonferroni_adjusted_p_value"] == pytest.approx(0.60)


def test_adjusted_p_values_are_between_zero_and_one():
    results = correct_multiple_p_values([0.01, 0.03, 0.20])

    for result in results:
        assert 0 <= result["bonferroni_adjusted_p_value"] <= 1
        assert 0 <= result["bh_adjusted_p_value"] <= 1


def test_empty_p_values_are_rejected():
    with pytest.raises(ValueError):
        correct_multiple_p_values([])


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError):
        correct_multiple_p_values([0.01, 0.03], alpha=1.0)


@pytest.mark.parametrize("p_value", [-0.1, 1.1])
def test_invalid_p_values_are_rejected(p_value):
    with pytest.raises(ValueError):
        correct_multiple_p_values([p_value])
