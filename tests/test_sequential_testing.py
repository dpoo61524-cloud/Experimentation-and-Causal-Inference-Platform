"""Unit tests for the Group-Sequential Testing Statistical Engine.

Tests cover:
- Numerical boundary solving and canonical multivariate normal crossing probabilities (error <= 1e-4).
- Reference boundaries for K=5, alpha=0.05 matching [4.562, 3.226, 2.634, 2.281, 2.040].
- Schedule enforcement and distinction between planned vs observed information fractions.
- Efficacy stopping at interim looks.
- Continuation at interim looks.
- Final look distinction: efficacy achieved vs inconclusive evidence.
- Non-binding futility heuristic behavior and educational disclaimers.
- Input validation and edge cases.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import math
import pytest
from statistics_engine import (
    OBRIEN_FLEMING_CONSTANTS,
    get_canonical_covariance,
    calculate_crossing_probability,
    solve_obrien_fleming_constant,
    get_obrien_fleming_design,
    sequential_conversion_test,
)


def test_canonical_covariance_properties():
    """Verify covariance matrix structure for canonical sequential statistics."""
    cov_3 = get_canonical_covariance(3)
    assert cov_3.shape == (3, 3)
    # Diagonal should be 1.0 (variance of standardized statistic)
    for i in range(3):
        assert cov_3[i, i] == pytest.approx(1.0)
    # Off-diagonal Cov(Z_j, Z_k) = sqrt(min(j, k) / max(j, k))
    assert cov_3[0, 1] == pytest.approx(math.sqrt(1 / 2))
    assert cov_3[0, 2] == pytest.approx(math.sqrt(1 / 3))
    assert cov_3[1, 2] == pytest.approx(math.sqrt(2 / 3))
    # Symmetric
    assert cov_3[1, 0] == cov_3[0, 1]


def test_numerical_crossing_probability_and_constants():
    """Solve and verify Type I error crossing probability for all supported K and alpha pairs.

    Requirement: Resulting overall Type I error must be within 1e-4 of target alpha.
    """
    for alpha in [0.05, 0.01, 0.10]:
        for K in [2, 3, 4, 5]:
            c_k, boundaries = get_obrien_fleming_design(total_checkpoints=K, alpha=alpha)
            # Verify boundary formula b_k = c_K * sqrt(K / k)
            for k_idx, b in enumerate(boundaries, start=1):
                expected_b = c_k * math.sqrt(K / k_idx)
                assert b == pytest.approx(expected_b, abs=1e-5)

            # Compute actual crossing probability
            p_cross = calculate_crossing_probability(boundaries)
            error = abs(p_cross - alpha)
            assert error <= 1e-4, f"Failed for alpha={alpha}, K={K}: p_cross={p_cross}, error={error}"


def test_five_look_alpha_05_reference_boundaries():
    """Confirm the five-look, two-sided alpha=0.05 reference boundaries.

    Reference values: approximately 4.562, 3.226, 2.634, 2.281, and 2.040.
    """
    c_5, boundaries = get_obrien_fleming_design(total_checkpoints=5, alpha=0.05)
    assert c_5 == pytest.approx(2.040, abs=0.001)

    expected_refs = [4.562, 3.226, 2.634, 2.281, 2.040]
    assert len(boundaries) == 5
    for b_calc, b_ref in zip(boundaries, expected_refs):
        assert b_calc == pytest.approx(b_ref, abs=0.002)


def test_get_obrien_fleming_design_invalid_looks():
    """Test get_obrien_fleming_design raises ValueError for unsupported K."""
    with pytest.raises(ValueError, match="Supported total_checkpoints are 2, 3, 4, or 5"):
        get_obrien_fleming_design(total_checkpoints=1)

    with pytest.raises(ValueError, match="Supported total_checkpoints are 2, 3, 4, or 5"):
        get_obrien_fleming_design(total_checkpoints=6)


def test_sequential_interim_efficacy_stopping():
    """Test early stopping for efficacy at checkpoint 1 of 5."""
    # Look 1/5: planned info = 0.20. max_visitors_per_group = 1000.
    # 200 visitors per group (observed info = 0.20, perfectly on schedule).
    # Massive difference: Control 10/200 (5%), Treatment 80/200 (40%).
    result = sequential_conversion_test(
        checkpoint=1,
        total_checkpoints=5,
        conversions_a=10,
        visitors_a=200,
        conversions_b=80,
        visitors_b=200,
        max_visitors_per_group=1000,
        alpha=0.05,
    )

    assert result["checkpoint"] == 1
    assert result["total_checkpoints"] == 5
    assert result["planned_information_fraction"] == pytest.approx(0.20)
    assert result["observed_information_fraction"] == pytest.approx(0.20)
    assert result["schedule_compliant"] is True
    assert result["efficacy_boundary_crossed"] is True
    assert result["maximum_sample_size_reached"] is False
    assert result["decision"] == "stop_for_efficacy"
    assert "crossed the O'Brien-Fleming efficacy boundary" in result["explanation"]
    assert result["z_statistic"] > result["efficacy_boundary_z"]


def test_sequential_interim_continue():
    """Test continuation at checkpoint 2 of 5 when effect is modest."""
    # Look 2/5: planned info = 0.40. max_visitors_per_group = 1000.
    # 400 visitors per group (observed info = 0.40).
    # Small difference: Control 40/400 (10%), Treatment 46/400 (11.5%).
    result = sequential_conversion_test(
        checkpoint=2,
        total_checkpoints=5,
        conversions_a=40,
        visitors_a=400,
        conversions_b=46,
        visitors_b=400,
        max_visitors_per_group=1000,
        alpha=0.05,
    )

    assert result["checkpoint"] == 2
    assert result["efficacy_boundary_crossed"] is False
    assert result["maximum_sample_size_reached"] is False
    assert result["decision"] == "continue"
    assert "Continue experiment to next scheduled checkpoint" in result["explanation"]


def test_sequential_final_look_efficacy_achieved():
    """Test final look (checkpoint K/K) where efficacy boundary is crossed."""
    # Look 5/5: planned info = 1.0. max_visitors_per_group = 1000.
    # 1000 visitors per group (observed info = 1.0).
    # Significant difference: Control 100/1000 (10%), Treatment 140/1000 (14%).
    result = sequential_conversion_test(
        checkpoint=5,
        total_checkpoints=5,
        conversions_a=100,
        visitors_a=1000,
        conversions_b=140,
        visitors_b=1000,
        max_visitors_per_group=1000,
        alpha=0.05,
    )

    assert result["checkpoint"] == 5
    assert result["maximum_sample_size_reached"] is True
    assert result["efficacy_boundary_crossed"] is True
    assert result["decision"] == "stop_max_sample_size_efficacy_achieved"
    assert "statistically significant at overall alpha=0.05" in result["explanation"]


def test_sequential_final_look_inconclusive():
    """Test final look (checkpoint K/K) where efficacy boundary is not crossed.

    Must clearly distinguish inconclusive evidence from 'proof of no effect'.
    """
    # Look 5/5: Control 100/1000 (10%), Treatment 105/1000 (10.5%).
    result = sequential_conversion_test(
        checkpoint=5,
        total_checkpoints=5,
        conversions_a=100,
        visitors_a=1000,
        conversions_b=105,
        visitors_b=1000,
        max_visitors_per_group=1000,
        alpha=0.05,
    )

    assert result["checkpoint"] == 5
    assert result["maximum_sample_size_reached"] is True
    assert result["efficacy_boundary_crossed"] is False
    assert result["decision"] == "stop_max_sample_size_inconclusive"
    assert "inconclusive evidence, NOT proof that the treatment has no effect" in result["explanation"]


def test_sequential_futility_recommendation_and_disclaimer():
    """Test non-binding futility recommendation at halfway look with negative direction."""
    # Look 3 of 5 (>= ceil(5/2) = 3).
    # Planned info = 0.60. Visitors = 600 per group.
    # Treatment is trailing: Control 60/600 (10%), Treatment 50/600 (8.33%).
    result = sequential_conversion_test(
        checkpoint=3,
        total_checkpoints=5,
        conversions_a=60,
        visitors_a=600,
        conversions_b=50,
        visitors_b=600,
        max_visitors_per_group=1000,
        alpha=0.05,
        evaluate_futility=True,
    )

    assert result["futility_boundary_crossed"] is True
    assert result["futility_recommendation"] == "stop_for_futility"
    assert result["decision"] == "continue"  # Primary statistical decision remains continue
    assert "does NOT prove absence of an effect" in result["explanation"]


def test_schedule_enforcement_raises_on_deviation():
    """Test that equal-information boundary is rejected if observed information deviates."""
    # Checkpoint 1 of 5 (planned info = 0.20), but observed visitors is 900/1000 (observed info = 0.90).
    # Deviation is 0.70 > tolerance (0.15).
    with pytest.raises(ValueError, match="does not meet the design's schedule assumptions"):
        sequential_conversion_test(
            checkpoint=1,
            total_checkpoints=5,
            conversions_a=90,
            visitors_a=900,
            conversions_b=90,
            visitors_b=900,
            max_visitors_per_group=1000,
        )

    # Checkpoint 5 of 5 (planned info = 1.0), but observed visitors is only 200/1000 (observed info = 0.20).
    with pytest.raises(ValueError, match="does not meet the design's schedule assumptions"):
        sequential_conversion_test(
            checkpoint=5,
            total_checkpoints=5,
            conversions_a=20,
            visitors_a=200,
            conversions_b=20,
            visitors_b=200,
            max_visitors_per_group=1000,
        )


def test_schedule_tolerance_disabled():
    """Test that setting schedule_tolerance=None permits non-standard look sizes if explicit."""
    result = sequential_conversion_test(
        checkpoint=1,
        total_checkpoints=5,
        conversions_a=40,
        visitors_a=400,
        conversions_b=40,
        visitors_b=400,
        max_visitors_per_group=1000,
        schedule_tolerance=None,
    )
    assert result["checkpoint"] == 1
    assert result["schedule_tolerance"] is None


def test_input_validation_errors():
    """Test input validations for sequential_conversion_test."""
    # Invalid alpha
    with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
        sequential_conversion_test(1, 3, 10, 100, 10, 100, 300, alpha=0.0)

    with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
        sequential_conversion_test(1, 3, 10, 100, 10, 100, 300, alpha=1.0)

    # Invalid total_checkpoints
    with pytest.raises(ValueError, match="total_checkpoints must be one of 2, 3, 4, or 5"):
        sequential_conversion_test(1, 6, 10, 100, 10, 100, 300)

    # Invalid checkpoint
    with pytest.raises(ValueError, match="checkpoint must be between 1 and 3"):
        sequential_conversion_test(0, 3, 10, 100, 10, 100, 300)

    with pytest.raises(ValueError, match="checkpoint must be between 1 and 3"):
        sequential_conversion_test(4, 3, 10, 100, 10, 100, 300)

    # Negative counts
    with pytest.raises(ValueError, match="conversions cannot be negative"):
        sequential_conversion_test(1, 3, -1, 100, 10, 100, 300)

    # Conversions > visitors
    with pytest.raises(ValueError, match="conversions cannot exceed visitors"):
        sequential_conversion_test(1, 3, 150, 100, 10, 100, 300)

    # Visitors > max_visitors_per_group
    with pytest.raises(ValueError, match="exceed planned maximum sample size"):
        sequential_conversion_test(1, 3, 10, 350, 10, 300, 300)

    # Boolean counts
    with pytest.raises(ValueError, match="counts must be integers, not booleans"):
        sequential_conversion_test(1, 3, True, 100, 10, 100, 300)

    # Boolean checkpoint
    with pytest.raises(ValueError, match="checkpoint must be an integer"):
        sequential_conversion_test(True, 3, 10, 100, 10, 100, 300)

    # Zero or negative max_visitors
    with pytest.raises(ValueError, match="max_visitors_per_group must be greater than zero"):
        sequential_conversion_test(1, 3, 10, 100, 10, 100, 0)
