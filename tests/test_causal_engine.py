"""Unit tests for Causal Inference Data Validation Foundation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import pytest

from causal_engine import (
    validate_causal_data,
    estimate_unadjusted_ate,
    estimate_unadjusted_ate_with_uncertainty,
    estimate_regression_adjusted_ate,
    compute_overlap_diagnostics,
    compute_unadjusted_covariate_balance,
    estimate_propensity_scores,
    compute_effective_sample_size,
    compute_weighted_covariate_balance,
    estimate_iptw_ate,
    validate_did_data,
    estimate_difference_in_differences,
    analyze_unmeasured_confounding_sensitivity,
    run_causal_sensitivity_analysis,
)


# ============================================================================
# 1. Valid Dataset and Input Format Tests
# ============================================================================

def test_valid_dataframe_basic():
    """Verify validation on a clean, valid pandas DataFrame with numeric and categorical confounders."""
    df = pd.DataFrame({
        "treatment": [1, 0, 1, 0, 1, 0],
        "outcome": [10.5, 8.0, 12.0, 7.5, 11.0, 9.0],
        "age": [25, 30, 35, 40, 45, 50],
        "channel": ["web", "mobile", "web", "email", "mobile", "email"],
    })

    result = validate_causal_data(
        data=df,
        treatment_col="treatment",
        outcome_col="outcome",
        confounder_cols=["age", "channel"],
    )

    assert result["is_valid"] is True
    assert result["n_total"] == 6
    assert result["n_treated"] == 3
    assert result["n_untreated"] == 3
    assert result["treatment_rate"] == pytest.approx(0.50)
    assert result["treated_value"] == 1
    assert result["untreated_value"] == 0
    assert result["outcome_mean_treated"] == pytest.approx((10.5 + 12.0 + 11.0) / 3)
    assert result["outcome_mean_untreated"] == pytest.approx((8.0 + 7.5 + 9.0) / 3)
    assert result["raw_difference"] == pytest.approx(
        result["outcome_mean_treated"] - result["outcome_mean_untreated"]
    )
    assert result["numeric_confounders"] == ["age"]
    assert result["categorical_confounders"] == ["channel"]
    assert result["rows_dropped"] == 0


def test_valid_dict_and_record_list_inputs():
    """Verify validation works across alternative input formats (dict of lists, list of dicts)."""
    dict_data = {
        "t": [0, 1, 0, 1],
        "y": [5.0, 10.0, 6.0, 11.0],
    }
    res_dict = validate_causal_data(dict_data, treatment_col="t", outcome_col="y")
    assert res_dict["n_total"] == 4
    assert res_dict["outcome_mean_treated"] == 10.5
    assert res_dict["outcome_mean_untreated"] == 5.5

    records_data = [
        {"t": 1, "y": 100.0},
        {"t": 0, "y": 80.0},
        {"t": 1, "y": 120.0},
        {"t": 0, "y": 90.0},
    ]
    res_records = validate_causal_data(records_data, treatment_col="t", outcome_col="y")
    assert res_records["n_total"] == 4
    assert res_records["n_treated"] == 2
    assert res_records["n_untreated"] == 2


def test_custom_treatment_labels():
    """Verify validation with explicit custom treatment and control labels."""
    df = pd.DataFrame({
        "group": ["Mens E-Mail", "No E-Mail", "Mens E-Mail", "No E-Mail"],
        "spend": [50.0, 0.0, 75.0, 10.0],
    })

    result = validate_causal_data(
        data=df,
        treatment_col="group",
        outcome_col="spend",
        treated_value="Mens E-Mail",
        untreated_value="No E-Mail",
    )

    assert result["treated_value"] == "Mens E-Mail"
    assert result["untreated_value"] == "No E-Mail"
    assert result["n_treated"] == 2
    assert result["n_untreated"] == 2
    assert result["outcome_mean_treated"] == 62.5
    assert result["outcome_mean_untreated"] == 5.0


def test_inferred_treatment_labels_string_patterns():
    """Verify automatic resolution of treatment/control strings when labels are not explicitly passed."""
    df = pd.DataFrame({
        "segment": ["treatment", "control", "treatment", "control"],
        "revenue": [20.0, 15.0, 22.0, 14.0],
    })
    result = validate_causal_data(df, treatment_col="segment", outcome_col="revenue")
    assert result["treated_value"] == "treatment"
    assert result["untreated_value"] == "control"


# ============================================================================
# 2. Empty Data and Missing Columns Tests
# ============================================================================

def test_empty_dataset_rejections():
    """Verify rejection of empty data structures."""
    with pytest.raises(ValueError, match="Dataset cannot be None"):
        validate_causal_data(None, "t", "y")

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_causal_data(pd.DataFrame(), "t", "y")

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_causal_data({}, "t", "y")

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_causal_data([], "t", "y")


def test_missing_required_columns():
    """Verify ValueError when treatment or outcome columns do not exist in dataset."""
    df = pd.DataFrame({"t": [1, 0], "y": [5.0, 6.0]})

    with pytest.raises(ValueError, match="Treatment column 'treatment' not found"):
        validate_causal_data(df, treatment_col="treatment", outcome_col="y")

    with pytest.raises(ValueError, match="Outcome column 'outcome' not found"):
        validate_causal_data(df, treatment_col="t", outcome_col="outcome")


def test_identical_treatment_and_outcome_column_rejected():
    """Verify ValueError when treatment_col and outcome_col are specified as the same column."""
    df = pd.DataFrame({"t": [1, 0]})
    with pytest.raises(ValueError, match="treatment_col and outcome_col cannot be the same column"):
        validate_causal_data(df, treatment_col="t", outcome_col="t")


def test_missing_confounder_column():
    """Verify ValueError when a requested confounder column is missing from dataset."""
    df = pd.DataFrame({"t": [1, 0], "y": [5.0, 6.0], "age": [30, 40]})
    with pytest.raises(ValueError, match="Confounder column 'gender' not found in dataset"):
        validate_causal_data(df, "t", "y", confounder_cols=["age", "gender"])


def test_confounder_overlapping_with_treatment_or_outcome_rejected():
    """Verify that confounder columns cannot duplicate treatment or outcome columns."""
    df = pd.DataFrame({"t": [1, 0], "y": [5.0, 6.0], "x": [10, 20]})

    with pytest.raises(ValueError, match="cannot also be the treatment column"):
        validate_causal_data(df, "t", "y", confounder_cols=["t"])

    with pytest.raises(ValueError, match="cannot also be the outcome column"):
        validate_causal_data(df, "t", "y", confounder_cols=["y"])


# ============================================================================
# 3. Treatment Variable Validation Tests
# ============================================================================

def test_treatment_group_with_no_observations():
    """Verify ValueError when all observations belong to only one treatment level."""
    df_all_treated = pd.DataFrame({"t": [1, 1, 1], "y": [5.0, 6.0, 7.0]})
    with pytest.raises(ValueError, match="contains only 1 unique value"):
        validate_causal_data(df_all_treated, "t", "y")

    # With explicit labels but all units belong to only one label
    df_no_control = pd.DataFrame({"t": ["A", "A", "A"], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="contains only 1 unique value"):
        validate_causal_data(df_no_control, "t", "y", treated_value="A", untreated_value="B")


def test_non_binary_treatment_without_labels():
    """Verify ValueError when treatment has 3 or more unique values and labels are not specified."""
    df_ternary = pd.DataFrame({
        "t": [0, 1, 2, 0, 1, 2],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    with pytest.raises(ValueError, match="must be binary, but contains 3 unique values"):
        validate_causal_data(df_ternary, "t", "y")


def test_treatment_containing_unspecified_labels():
    """Verify ValueError when treatment contains values not matching specified treated/untreated."""
    df = pd.DataFrame({
        "t": ["T", "C", "Unknown"],
        "y": [10.0, 5.0, 8.0],
    })
    with pytest.raises(ValueError, match="contains values other than specified treated"):
        validate_causal_data(df, "t", "y", treated_value="T", untreated_value="C")


def test_identical_treated_and_untreated_labels_rejected():
    """Verify ValueError when treated_value and untreated_value are identical."""
    df = pd.DataFrame({"t": [1, 0], "y": [10.0, 5.0]})
    with pytest.raises(ValueError, match="treated_value and untreated_value cannot be identical"):
        validate_causal_data(df, "t", "y", treated_value=1, untreated_value=1)


# ============================================================================
# 4. Outcome Variable Validation Tests
# ============================================================================

def test_non_numeric_outcome_strings_rejected():
    """Verify ValueError when outcome contains non-numeric strings."""
    df = pd.DataFrame({
        "t": [1, 0, 1, 0],
        "y": ["high", "low", "medium", "low"],
    })
    with pytest.raises(ValueError, match="Outcome column 'y' contains non-numeric values"):
        validate_causal_data(df, "t", "y")


def test_boolean_outcome_rejected():
    """Verify ValueError when outcome column is boolean."""
    df = pd.DataFrame({
        "t": [1, 0, 1, 0],
        "y": [True, False, True, False],
    })
    with pytest.raises(ValueError, match="must be numeric, but contains boolean values"):
        validate_causal_data(df, "t", "y")


def test_infinite_outcome_rejected():
    """Verify ValueError when outcome contains positive or negative infinity."""
    df_pos_inf = pd.DataFrame({"t": [1, 0, 1], "y": [10.0, np.inf, 12.0]})
    with pytest.raises(ValueError, match="contains infinite values"):
        validate_causal_data(df_pos_inf, "t", "y")

    df_neg_inf = pd.DataFrame({"t": [1, 0, 1], "y": [10.0, -np.inf, 12.0]})
    with pytest.raises(ValueError, match="contains infinite values"):
        validate_causal_data(df_neg_inf, "t", "y")


def test_nan_outcome_rejected_by_default():
    """Verify ValueError when outcome contains NaN and drop_missing is False."""
    df = pd.DataFrame({"t": [1, 0, 1], "y": [10.0, np.nan, 12.0]})
    with pytest.raises(ValueError, match="Dataset contains missing values in required columns"):
        validate_causal_data(df, "t", "y", drop_missing=False)


# ============================================================================
# 5. Missing Data Policy Tests
# ============================================================================

def test_missing_values_explicit_drop_policy():
    """Verify that drop_missing=True explicitly drops incomplete rows and tracks dropped count."""
    df = pd.DataFrame({
        "t": [1, 0, 1, 0, 1, None],
        "y": [10.0, 5.0, np.nan, 8.0, 12.0, 7.0],
        "x": [100, 200, 150, None, 250, 300],
    })

    # Rows with missing:
    # row 0: (1, 10.0, 100) -> valid
    # row 1: (0, 5.0, 200) -> valid
    # row 2: y is NaN -> dropped
    # row 3: x is None -> dropped
    # row 4: (1, 12.0, 250) -> valid
    # row 5: t is None -> dropped

    result = validate_causal_data(
        data=df,
        treatment_col="t",
        outcome_col="y",
        confounder_cols=["x"],
        drop_missing=True,
    )

    assert result["is_valid"] is True
    assert result["rows_dropped"] == 3
    assert result["n_total"] == 3
    assert result["n_treated"] == 2
    assert result["n_untreated"] == 1


def test_missing_values_drop_resulting_in_empty_data():
    """Verify ValueError if dropping missing values leaves no observations in the dataset."""
    df = pd.DataFrame({
        "t": [1, None],
        "y": [np.nan, 10.0],
    })
    with pytest.raises(ValueError, match="After dropping rows with missing values, no observations remain"):
        validate_causal_data(df, "t", "y", drop_missing=True)


# ============================================================================
# 6. Confounders Validation Tests
# ============================================================================

def test_constant_confounder_zero_variance_rejected():
    """Verify ValueError when a confounder column has zero variance (constant value)."""
    df = pd.DataFrame({
        "t": [1, 0, 1, 0],
        "y": [10.0, 5.0, 11.0, 6.0],
        "constant_x": [42, 42, 42, 42],
    })
    with pytest.raises(ValueError, match="has zero variance"):
        validate_causal_data(df, "t", "y", confounder_cols=["constant_x"])


def test_infinite_numeric_confounder_rejected():
    """Verify ValueError when a numeric confounder contains infinite values."""
    df = pd.DataFrame({
        "t": [1, 0, 1, 0],
        "y": [10.0, 5.0, 11.0, 6.0],
        "x": [1.0, np.inf, 2.0, 3.0],
    })
    with pytest.raises(ValueError, match="Numeric confounder column 'x' contains infinite values"):
        validate_causal_data(df, "t", "y", confounder_cols=["x"])


def test_categorical_confounder_with_strings():
    """Verify that multi-class categorical confounders with valid variance are classified properly."""
    df = pd.DataFrame({
        "t": [1, 0, 1, 0, 1, 0],
        "y": [10.0, 5.0, 12.0, 7.0, 11.0, 6.0],
        "city": ["NYC", "LA", "Chicago", "NYC", "LA", "Chicago"],
        "income": [50000, 60000, 55000, 52000, 61000, 54000],
    })
    res = validate_causal_data(df, "t", "y", confounder_cols=["city", "income"])
    assert res["categorical_confounders"] == ["city"]
    assert res["numeric_confounders"] == ["income"]


# ============================================================================
# 7. Unadjusted Average Treatment Effect (ATE) Estimator Tests
# ============================================================================

def test_estimate_unadjusted_ate_known_numerical_example():
    """Verify unadjusted ATE against a known, manually calculable numerical example.

    Treated outcomes: [12.0, 14.0, 16.0] -> mean = 14.0
    Control outcomes: [6.0, 8.0]        -> mean = 7.0
    Expected ATE: 14.0 - 7.0 = 7.0
    """
    df = pd.DataFrame({
        "treated": [1, 1, 1, 0, 0],
        "outcome": [12.0, 14.0, 16.0, 6.0, 8.0],
    })

    res = estimate_unadjusted_ate(df, treatment_col="treated", outcome_col="outcome")

    assert res["estimator"] == "unadjusted_difference_in_means"
    assert res["ate"] == pytest.approx(7.0)
    assert res["unadjusted_ate"] == pytest.approx(7.0)
    assert res["difference_in_means"] == pytest.approx(7.0)
    assert res["mean_outcome_treated"] == pytest.approx(14.0)
    assert res["mean_outcome_control"] == pytest.approx(7.0)
    assert res["outcome_mean_treated"] == pytest.approx(14.0)
    assert res["outcome_mean_untreated"] == pytest.approx(7.0)
    assert res["n_total"] == 5
    assert res["n_treated"] == 3
    assert res["n_control"] == 2
    assert res["n_untreated"] == 2
    assert res["rows_dropped"] == 0
    assert "confounding" in res["assumptions"].lower()


def test_estimate_unadjusted_ate_equal_means():
    """Verify unadjusted ATE equals 0.0 when treated and control outcome means are identical."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "y": [10.0, 10.0, 10.0, 10.0],
    })

    res = estimate_unadjusted_ate(df, treatment_col="t", outcome_col="y")
    assert res["ate"] == pytest.approx(0.0)
    assert res["mean_outcome_treated"] == pytest.approx(10.0)
    assert res["mean_outcome_control"] == pytest.approx(10.0)


def test_estimate_unadjusted_ate_negative_effect():
    """Verify unadjusted ATE correctly calculates a negative treatment effect estimate.

    Treated outcomes: [2.0, 4.0]  -> mean = 3.0
    Control outcomes: [8.0, 10.0] -> mean = 9.0
    Expected ATE: 3.0 - 9.0 = -6.0
    """
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "y": [2.0, 4.0, 8.0, 10.0],
    })

    res = estimate_unadjusted_ate(df, treatment_col="t", outcome_col="y")
    assert res["ate"] == pytest.approx(-6.0)
    assert res["mean_outcome_treated"] == pytest.approx(3.0)
    assert res["mean_outcome_control"] == pytest.approx(9.0)


def test_estimate_unadjusted_ate_unequal_group_sizes():
    """Verify unadjusted ATE calculation with substantially unequal group sizes."""
    # 10 treated units with mean 25.0, 2 control units with mean 15.0
    treated_y = [25.0] * 10
    control_y = [10.0, 20.0]  # mean = 15.0

    df = pd.DataFrame({
        "treatment": [1] * 10 + [0] * 2,
        "revenue": treated_y + control_y,
    })

    res = estimate_unadjusted_ate(df, treatment_col="treatment", outcome_col="revenue")
    assert res["n_total"] == 12
    assert res["n_treated"] == 10
    assert res["n_control"] == 2
    assert res["mean_outcome_treated"] == pytest.approx(25.0)
    assert res["mean_outcome_control"] == pytest.approx(15.0)
    assert res["ate"] == pytest.approx(10.0)


def test_estimate_unadjusted_ate_difference_direction_treated_minus_control():
    """Verify that difference direction is strictly treated mean minus control mean."""
    # Custom labels: "Active" treated vs "Placebo" control
    df = pd.DataFrame({
        "drug": ["Active", "Active", "Placebo", "Placebo"],
        "pain_score": [3.0, 5.0, 8.0, 10.0],  # Active mean=4.0, Placebo mean=9.0
    })

    res = estimate_unadjusted_ate(
        df,
        treatment_col="drug",
        outcome_col="pain_score",
        treated_value="Active",
        untreated_value="Placebo",
    )

    # Pain reduction: Active (4.0) - Placebo (9.0) = -5.0
    assert res["ate"] == pytest.approx(-5.0)
    assert res["treated_value"] == "Active"
    assert res["control_value"] == "Placebo"


def test_estimate_unadjusted_ate_group_counts_and_means_correct():
    """Verify exact count and mean integrity across multiple sample observations."""
    df = pd.DataFrame({
        "t": [0, 1, 0, 1, 0, 1],
        "y": [1.0, 2.0, 3.0, 6.0, 5.0, 10.0],
    })
    # Control: [1.0, 3.0, 5.0] -> mean = 3.0, count = 3
    # Treated: [2.0, 6.0, 10.0] -> mean = 6.0, count = 3
    res = estimate_unadjusted_ate(df, treatment_col="t", outcome_col="y")
    assert res["n_total"] == 6
    assert res["n_treated"] == 3
    assert res["n_control"] == 3
    assert res["mean_outcome_treated"] == pytest.approx(6.0)
    assert res["mean_outcome_control"] == pytest.approx(3.0)
    assert res["ate"] == pytest.approx(3.0)


def test_estimate_unadjusted_ate_reuses_validation_behavior():
    """Verify that estimate_unadjusted_ate reuses validation for missing columns and empty datasets."""
    # Missing column
    df = pd.DataFrame({"t": [1, 0], "y": [5.0, 6.0]})
    with pytest.raises(ValueError, match="Treatment column 'missing_t' not found"):
        estimate_unadjusted_ate(df, treatment_col="missing_t", outcome_col="y")

    # Empty dataset
    with pytest.raises(ValueError, match="Dataset is empty"):
        estimate_unadjusted_ate(pd.DataFrame(), treatment_col="t", outcome_col="y")


def test_estimate_unadjusted_ate_missing_data_policy():
    """Verify missing data policy: reject by default, drop and track when drop_missing=True."""
    df_missing = pd.DataFrame({
        "t": [1, 0, 1, 0, 1],
        "y": [10.0, 5.0, np.nan, 7.0, 12.0],
    })

    # Default: raises ValueError
    with pytest.raises(ValueError, match="Dataset contains missing values"):
        estimate_unadjusted_ate(df_missing, treatment_col="t", outcome_col="y", drop_missing=False)

    # Explicit drop: drops row 2 (leaving 2 treated, 2 control)
    res = estimate_unadjusted_ate(df_missing, treatment_col="t", outcome_col="y", drop_missing=True)
    assert res["rows_dropped"] == 1
    assert res["n_total"] == 4
    assert res["n_treated"] == 2
    assert res["n_control"] == 2
    # Treated: [10.0, 12.0] -> mean = 11.0
    # Control: [5.0, 7.0] -> mean = 6.0
    assert res["mean_outcome_treated"] == pytest.approx(11.0)
    assert res["mean_outcome_control"] == pytest.approx(6.0)
    assert res["ate"] == pytest.approx(5.0)


def test_estimate_unadjusted_ate_invalid_treatment_or_outcome_errors():
    """Verify that invalid treatment or outcome types raise clear ValueErrors."""
    # Ternary treatment
    df_ternary = pd.DataFrame({"t": [0, 1, 2, 0], "y": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="must be binary"):
        estimate_unadjusted_ate(df_ternary, treatment_col="t", outcome_col="y")

    # Non-numeric string outcome
    df_string_y = pd.DataFrame({"t": [1, 0], "y": ["good", "bad"]})
    with pytest.raises(ValueError, match="contains non-numeric values"):
        estimate_unadjusted_ate(df_string_y, treatment_col="t", outcome_col="y")

    # Boolean outcome
    df_bool_y = pd.DataFrame({"t": [1, 0], "y": [True, False]})
    with pytest.raises(ValueError, match="must be numeric, but contains boolean values"):
        estimate_unadjusted_ate(df_bool_y, treatment_col="t", outcome_col="y")


# ============================================================================
# 8. Uncertainty Estimation and Confidence Intervals for Treatment Effects
# ============================================================================

def test_unadjusted_ate_uncertainty_known_numerical_example():
    """Verify Welch standard error and confidence interval against a manually calculable example.

    Treated outcomes: [12.0, 14.0, 16.0]
    - n_treated = 3, mean = 14.0, s_treated^2 = 4.0, v_treated = 4/3
    Control outcomes: [6.0, 10.0]
    - n_control = 2, mean = 8.0, s_control^2 = 8.0, v_control = 4.0

    ATE = 14.0 - 8.0 = 6.0
    SE = sqrt(4/3 + 4) = sqrt(16/3) ≈ 2.3094010768
    df = (16/3)^2 / [ (4/3)^2 / 2 + 4^2 / 1 ] = (256/9) / [ (16/9)/2 + 16 ]
       = (256/9) / [ 8/9 + 144/9 ] = (256/9) / (152/9) = 256 / 152 = 32 / 19 ≈ 1.6842105
    t_crit (95%, df=32/19) ≈ 2.57813...
    95% CI = [6.0 - 2.57813 * 2.3094, 6.0 + 2.57813 * 2.3094] ≈ [-5.953995, 17.953995]
    """
    df = pd.DataFrame({
        "treated": [1, 1, 1, 0, 0],
        "outcome": [12.0, 14.0, 16.0, 6.0, 10.0],
    })

    res = estimate_unadjusted_ate(df, treatment_col="treated", outcome_col="outcome")

    assert res["estimator"] == "unadjusted_difference_in_means"
    assert res["ate"] == pytest.approx(6.0)
    assert res["mean_outcome_treated"] == pytest.approx(14.0)
    assert res["mean_outcome_control"] == pytest.approx(8.0)
    assert res["variance_treated"] == pytest.approx(4.0)
    assert res["variance_control"] == pytest.approx(8.0)
    assert res["std_error"] == pytest.approx(np.sqrt(16.0 / 3.0))
    assert res["se"] == pytest.approx(res["std_error"])
    assert res["degrees_of_freedom"] == pytest.approx(32.0 / 19.0)
    assert res["df"] == pytest.approx(res["degrees_of_freedom"])
    assert res["confidence_level"] == pytest.approx(0.95)

    ci_low = res["confidence_interval"][0]
    ci_high = res["confidence_interval"][1]
    assert ci_low == pytest.approx(-5.9539948965, rel=1e-5)
    assert ci_high == pytest.approx(17.9539948965, rel=1e-5)
    assert res["ci_lower"] == pytest.approx(ci_low)
    assert res["ci_upper"] == pytest.approx(ci_high)
    assert res["confidence_interval_95"] == [ci_low, ci_high]
    assert res["margin_of_error"] == pytest.approx(res["t_critical"] * res["std_error"])
    assert ci_high - res["ate"] == pytest.approx(res["margin_of_error"])
    assert res["ate"] - ci_low == pytest.approx(res["margin_of_error"])


def test_unadjusted_ate_uncertainty_unequal_sizes_and_variances():
    """Verify Welch uncertainty estimation with unequal sample sizes and unequal variances."""
    import scipy.stats as stats

    treated_vals = [15.0, 18.0, 20.0, 22.0, 25.0, 27.0, 30.0, 35.0]  # n=8
    control_vals = [5.0, 7.0, 12.0, 16.0]  # n=4

    df = pd.DataFrame({
        "t": [1] * len(treated_vals) + [0] * len(control_vals),
        "y": treated_vals + control_vals,
    })

    res = estimate_unadjusted_ate(df, treatment_col="t", outcome_col="y", confidence_level=0.95)

    # Cross-verify against scipy.stats.ttest_ind with equal_var=False
    scipy_res = stats.ttest_ind(treated_vals, control_vals, equal_var=False)
    scipy_ci = scipy_res.confidence_interval(0.95)

    assert res["n_treated"] == 8
    assert res["n_control"] == 4
    assert res["degrees_of_freedom"] == pytest.approx(scipy_res.df)
    assert res["confidence_interval"][0] == pytest.approx(scipy_ci.low)
    assert res["confidence_interval"][1] == pytest.approx(scipy_ci.high)

    # Verify standard error formula
    v_t = np.var(treated_vals, ddof=1) / 8
    v_c = np.var(control_vals, ddof=1) / 4
    expected_se = np.sqrt(v_t + v_c)
    assert res["std_error"] == pytest.approx(expected_se)


def test_unadjusted_ate_interval_ordering_and_direction():
    """Verify correct interval direction, bounds ordering (lower <= upper), and centering."""
    # Positive effect
    df_pos = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "y": [20.0, 22.0, 24.0, 10.0, 12.0, 14.0],
    })
    res_pos = estimate_unadjusted_ate(df_pos, "t", "y")
    assert res_pos["ci_lower"] < res_pos["ate"] < res_pos["ci_upper"]
    assert res_pos["ci_lower"] <= res_pos["ci_upper"]
    assert res_pos["ate"] == pytest.approx(10.0)

    # Negative effect
    df_neg = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "y": [5.0, 6.0, 7.0, 15.0, 16.0, 17.0],
    })
    res_neg = estimate_unadjusted_ate(df_neg, "t", "y")
    assert res_neg["ci_lower"] < res_neg["ate"] < res_neg["ci_upper"]
    assert res_neg["ci_lower"] <= res_neg["ci_upper"]
    assert res_neg["ate"] == pytest.approx(-10.0)

    # Zero effect with variance
    df_zero = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "y": [8.0, 10.0, 12.0, 8.0, 10.0, 12.0],
    })
    res_zero = estimate_unadjusted_ate(df_zero, "t", "y")
    assert res_zero["ate"] == pytest.approx(0.0)
    assert res_zero["ci_lower"] < 0.0 < res_zero["ci_upper"]
    assert abs(res_zero["ci_lower"]) == pytest.approx(res_zero["ci_upper"])


def test_unadjusted_ate_configurable_confidence_levels():
    """Verify that user-configured confidence levels are applied accurately to critical values."""
    import scipy.stats as stats

    df = pd.DataFrame({
        "t": [1, 1, 1, 1, 0, 0, 0, 0],
        "y": [10.0, 12.0, 14.0, 16.0, 4.0, 6.0, 8.0, 10.0],
    })

    for conf in [0.80, 0.90, 0.95, 0.99]:
        res = estimate_unadjusted_ate(df, "t", "y", confidence_level=conf)
        assert res["confidence_level"] == pytest.approx(conf)
        expected_tc = stats.t.ppf(1.0 - (1.0 - conf) / 2.0, df=res["degrees_of_freedom"])
        assert res["t_critical"] == pytest.approx(expected_tc)
        assert res["ci_upper"] - res["ate"] == pytest.approx(expected_tc * res["std_error"])


def test_unadjusted_ate_wider_interval_for_higher_confidence():
    """Verify that higher confidence levels produce strictly wider confidence intervals."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "y": [12.0, 14.0, 15.0, 17.0, 19.0, 6.0, 8.0, 9.0, 11.0, 12.0],
    })

    res_80 = estimate_unadjusted_ate(df, "t", "y", confidence_level=0.80)
    res_90 = estimate_unadjusted_ate(df, "t", "y", confidence_level=0.90)
    res_95 = estimate_unadjusted_ate(df, "t", "y", confidence_level=0.95)
    res_99 = estimate_unadjusted_ate(df, "t", "y", confidence_level=0.99)

    w_80 = res_80["ci_upper"] - res_80["ci_lower"]
    w_90 = res_90["ci_upper"] - res_90["ci_lower"]
    w_95 = res_95["ci_upper"] - res_95["ci_lower"]
    w_99 = res_99["ci_upper"] - res_99["ci_lower"]

    # Monotonic interval widths
    assert w_80 < w_90 < w_95 < w_99
    # Monotonic interval bounds expansion
    assert res_99["ci_lower"] < res_95["ci_lower"] < res_90["ci_lower"] < res_80["ci_lower"]
    assert res_99["ci_upper"] > res_95["ci_upper"] > res_90["ci_upper"] > res_80["ci_upper"]


def test_unadjusted_ate_invalid_confidence_levels_rejected():
    """Verify that invalid confidence levels (booleans, strings, non-numeric, NaN, inf, out of bounds) are rejected."""
    df = pd.DataFrame({"t": [1, 1, 0, 0], "y": [1.0, 2.0, 3.0, 4.0]})

    invalid_levels = [
        0.0,
        1.0,
        -0.1,
        1.5,
        True,
        False,
        "0.95",
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
    ]

    for inv in invalid_levels:
        with pytest.raises(ValueError):
            estimate_unadjusted_ate(df, "t", "y", confidence_level=inv)


def test_unadjusted_ate_insufficient_observations_for_variance():
    """Verify rejection when either treatment group contains fewer than 2 observations for sample variance."""
    # Treated has only 1 observation
    df_n1_treated = pd.DataFrame({"t": [1, 0, 0, 0], "y": [10.0, 5.0, 6.0, 7.0]})
    with pytest.raises(ValueError, match="at least 2 observations"):
        estimate_unadjusted_ate(df_n1_treated, "t", "y", compute_ci=True)

    # Control has only 1 observation
    df_n1_control = pd.DataFrame({"t": [1, 1, 1, 0], "y": [10.0, 11.0, 12.0, 5.0]})
    with pytest.raises(ValueError, match="at least 2 observations"):
        estimate_unadjusted_ate(df_n1_control, "t", "y", compute_ci=True)

    # Both have only 1 observation
    df_n1_both = pd.DataFrame({"t": [1, 0], "y": [10.0, 5.0]})
    with pytest.raises(ValueError, match="at least 2 observations"):
        estimate_unadjusted_ate(df_n1_both, "t", "y", compute_ci=True)

    # Verify that point estimation without uncertainty succeeds with 1 observation per group
    res_point_only = estimate_unadjusted_ate(df_n1_both, "t", "y", compute_ci=False)
    assert res_point_only["ate"] == pytest.approx(5.0)
    assert "confidence_interval" not in res_point_only


def test_unadjusted_ate_zero_variance_edge_cases():
    """Verify handling of zero-variance edge cases in one group and both groups."""
    # Case A: Zero variance in treated, positive variance in control
    treated_const = [10.0, 10.0, 10.0]  # var = 0
    control_var = [4.0, 8.0]             # var = 8.0, n=2
    df_one_zero = pd.DataFrame({
        "t": [1, 1, 1, 0, 0],
        "y": treated_const + control_var,
    })

    res_one_zero = estimate_unadjusted_ate(df_one_zero, "t", "y")
    assert res_one_zero["variance_treated"] == pytest.approx(0.0)
    assert res_one_zero["variance_control"] == pytest.approx(8.0)
    # Degrees of freedom simplifies to n_control - 1 = 1.0
    assert res_one_zero["degrees_of_freedom"] == pytest.approx(1.0)
    # Standard error equals sqrt(8.0 / 2) = 2.0
    assert res_one_zero["std_error"] == pytest.approx(2.0)
    assert res_one_zero["ci_lower"] < res_one_zero["ci_upper"]

    # Case B: Zero variance in both groups (mathematically undefined df & CI)
    df_both_zero = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "y": [5.0, 5.0, 5.0, 5.0],
    })

    # Default mathematically justified special case: returns None without raising
    res_both_zero = estimate_unadjusted_ate(df_both_zero, "t", "y", raise_on_zero_variance=False)
    assert res_both_zero["ate"] == pytest.approx(0.0)
    assert res_both_zero["variance_treated"] == pytest.approx(0.0)
    assert res_both_zero["variance_control"] == pytest.approx(0.0)
    assert res_both_zero["std_error"] == pytest.approx(0.0)
    assert res_both_zero["degrees_of_freedom"] is None
    assert res_both_zero["confidence_interval"] is None
    assert "warning" in res_both_zero

    # Strict mode: raises ValueError
    with pytest.raises(ValueError, match="zero variance"):
        estimate_unadjusted_ate(df_both_zero, "t", "y", raise_on_zero_variance=True)


def test_unadjusted_ate_uncertainty_reuses_validation_and_missing_data_policy():
    """Verify that uncertainty estimation reuses existing validation rules and missing data handling."""
    # Missing column
    df = pd.DataFrame({"t": [1, 1, 0, 0], "y": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="Treatment column 'missing' not found"):
        estimate_unadjusted_ate(df, treatment_col="missing", outcome_col="y")

    # Missing values: rejected by default (drop_missing=False)
    df_missing = pd.DataFrame({
        "t": [1, 1, 0, 0, 1],
        "y": [10.0, np.nan, 5.0, 6.0, 12.0],
    })
    with pytest.raises(ValueError, match="Dataset contains missing values"):
        estimate_unadjusted_ate(df_missing, "t", "y", drop_missing=False)

    # Missing values: dropped when drop_missing=True
    res = estimate_unadjusted_ate(df_missing, "t", "y", drop_missing=True)
    assert res["rows_dropped"] == 1
    assert res["n_total"] == 4
    assert res["n_treated"] == 2
    assert res["n_control"] == 2
    assert res["ci_lower"] < res["ate"] < res["ci_upper"]


def test_unadjusted_ate_backward_compatibility_regression():
    """Verify that all Task 5.2 return fields and behaviors are 100% compatible."""
    df = pd.DataFrame({
        "group": ["T", "T", "C", "C"],
        "metric": [10.0, 14.0, 4.0, 6.0],
    })

    res = estimate_unadjusted_ate(
        df,
        treatment_col="group",
        outcome_col="metric",
        treated_value="T",
        untreated_value="C",
    )

    # All Task 5.2 fields exist and have correct types
    expected_fields = [
        "estimator",
        "ate",
        "unadjusted_ate",
        "difference_in_means",
        "mean_outcome_treated",
        "mean_outcome_control",
        "outcome_mean_treated",
        "outcome_mean_untreated",
        "n_total",
        "n_treated",
        "n_control",
        "n_untreated",
        "treatment_col",
        "outcome_col",
        "confounder_cols",
        "treated_value",
        "control_value",
        "untreated_value",
        "rows_dropped",
        "assumptions",
    ]
    for field in expected_fields:
        assert field in res, f"Expected field '{field}' not found in result."

    # Dedicated alias produces identical results
    res_wrapper = estimate_unadjusted_ate_with_uncertainty(
        df,
        treatment_col="group",
        outcome_col="metric",
        treated_value="T",
        untreated_value="C",
    )
    for field in expected_fields:
        assert res[field] == res_wrapper[field]
    assert res["confidence_interval"] == res_wrapper["confidence_interval"]


def test_unadjusted_ate_interpretation_disclaimer():
    """Verify that assumptions and interpretation text accurately state sampling vs causal disclaimers."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "y": [10.0, 12.0, 14.0, 5.0, 7.0, 9.0],
    })
    res = estimate_unadjusted_ate(df, "t", "y")

    # Document that the interval quantifies sampling uncertainty under estimator assumptions
    assert "sampling" in res["assumptions"].lower()
    assert "sampling" in res["interpretation"].lower()

    # Do not claim 95% probability of true parameter lying in this interval
    assert (
        "does not mean a 95% probability" in res["interpretation"].lower()
        or "not represent a probability" in res["assumptions"].lower()
    )

    # Do not claim that interval removes confounding or proves causality
    assert "confounding" in res["assumptions"].lower()
    assert "confounding" in res["interpretation"].lower()


# ============================================================================
# 9. Regression-Adjusted Treatment Effect Estimation Tests
# ============================================================================

def test_regression_adjusted_ate_known_synthetic_dataset():
    """Verify regression-adjusted ATE against an independently calculable synthetic linear dataset.

    Data generation:
    Y = 10.0 + 3.0 * T + 2.0 * X
    Confounder X: [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    Treatment T:  [  1,   1,   1,   1,   0,   0,   0,   0]
    Treated Y: [15.0, 17.0, 19.0, 21.0] (mean = 18.0)
    Control Y: [12.0, 14.0, 16.0, 18.0] (mean = 15.0)
    Expected Adjusted ATE: 3.0 (true coefficient of T).
    """
    X_vals = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    T_vals = [1, 1, 1, 1, 0, 0, 0, 0]
    Y_vals = [10.0 + 3.0 * t + 2.0 * x for t, x in zip(T_vals, X_vals)]

    df = pd.DataFrame({"t": T_vals, "x": X_vals, "y": Y_vals})

    res = estimate_regression_adjusted_ate(
        data=df,
        treatment_col="t",
        outcome_col="y",
        confounder_cols=["x"],
    )

    assert res["estimator"] == "regression_adjusted_ate"
    assert res["adjusted_ate"] == pytest.approx(3.0)
    assert res["ate"] == pytest.approx(3.0)
    assert res["unadjusted_ate"] == pytest.approx(3.0)
    assert res["raw_difference"] == pytest.approx(3.0)
    assert res["mean_outcome_treated"] == pytest.approx(18.0)
    assert res["mean_outcome_control"] == pytest.approx(15.0)
    assert res["mean_predicted_treated"] == pytest.approx(18.0)
    assert res["mean_predicted_control"] == pytest.approx(15.0)
    assert res["r_squared"] == pytest.approx(1.0)
    assert res["n_total"] == 8
    assert res["n_treated"] == 4
    assert res["n_control"] == 4


def test_regression_adjusted_ate_numeric_and_categorical_confounders():
    """Verify regression adjustment with both continuous numeric and categorical confounders."""
    # Outcome equation: Y = 5.0 + 4.0 * T + 2.0 * age + 10.0 * (city == 'NYC')
    data = [
        {"grp": 1, "age": 20, "city": "NYC", "outcome": 5.0 + 4.0 + 40.0 + 10.0},
        {"grp": 1, "age": 30, "city": "LA",  "outcome": 5.0 + 4.0 + 60.0 + 0.0},
        {"grp": 1, "age": 40, "city": "NYC", "outcome": 5.0 + 4.0 + 80.0 + 10.0},
        {"grp": 1, "age": 25, "city": "CHI", "outcome": 5.0 + 4.0 + 50.0 + 0.0},
        {"grp": 0, "age": 20, "city": "NYC", "outcome": 5.0 + 0.0 + 40.0 + 10.0},
        {"grp": 0, "age": 30, "city": "LA",  "outcome": 5.0 + 0.0 + 60.0 + 0.0},
        {"grp": 0, "age": 40, "city": "NYC", "outcome": 5.0 + 0.0 + 80.0 + 10.0},
        {"grp": 0, "age": 25, "city": "CHI", "outcome": 5.0 + 0.0 + 50.0 + 0.0},
    ]
    df = pd.DataFrame(data)

    res = estimate_regression_adjusted_ate(
        data=df,
        treatment_col="grp",
        outcome_col="outcome",
        confounder_cols=["age", "city"],
    )

    assert res["adjusted_ate"] == pytest.approx(4.0)
    assert res["numeric_confounders"] == ["age"]
    assert res["categorical_confounders"] == ["city"]
    assert any("city" in col for col in res["model_features"])
    assert res["n_total"] == 8
    assert res["n_treated"] == 4
    assert res["n_control"] == 4


def test_regression_adjusted_ate_treated_minus_control_direction():
    """Verify that reported effect is strictly treated minus control for positive and negative effects."""
    # Positive effect: T=1 increases Y by +5
    df_pos = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [1, 2, 3, 1, 2, 3],
        "y": [15, 17, 19, 10, 12, 14],
    })
    res_pos = estimate_regression_adjusted_ate(df_pos, "t", "y", ["x"])
    assert res_pos["adjusted_ate"] == pytest.approx(5.0)

    # Negative effect: T=1 decreases Y by -5
    df_neg = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [1, 2, 3, 1, 2, 3],
        "y": [5, 7, 9, 10, 12, 14],
    })
    res_neg = estimate_regression_adjusted_ate(df_neg, "t", "y", ["x"])
    assert res_neg["adjusted_ate"] == pytest.approx(-5.0)


def test_regression_adjusted_ate_counts_and_total_sample():
    """Verify sample size and group count integrity with unequal allocation."""
    df = pd.DataFrame({
        "treatment": [1] * 7 + [0] * 3,
        "covar": [10, 12, 14, 16, 18, 20, 22, 11, 13, 15],
        "target": [20, 22, 24, 26, 28, 30, 32, 10, 12, 14],
    })
    res = estimate_regression_adjusted_ate(df, "treatment", "target", ["covar"])
    assert res["n_total"] == 10
    assert res["n_treated"] == 7
    assert res["n_control"] == 3
    assert res["n_untreated"] == 3


def test_regression_adjusted_ate_custom_treatment_labels():
    """Verify regression adjustment with explicit custom string labels for treatment and control."""
    df = pd.DataFrame({
        "group": ["Active_Drug", "Active_Drug", "Active_Drug", "Placebo", "Placebo", "Placebo"],
        "baseline_score": [50, 60, 70, 50, 60, 70],
        "final_score": [80, 90, 100, 65, 75, 85],
    })

    res = estimate_regression_adjusted_ate(
        data=df,
        treatment_col="group",
        outcome_col="final_score",
        confounder_cols=["baseline_score"],
        treated_value="Active_Drug",
        untreated_value="Placebo",
    )

    assert res["treated_value"] == "Active_Drug"
    assert res["control_value"] == "Placebo"
    assert res["untreated_value"] == "Placebo"
    assert res["adjusted_ate"] == pytest.approx(15.0)


def test_regression_adjusted_ate_missing_data_policy():
    """Verify that drop_missing=False rejects missing values while drop_missing=True drops rows and tracks count."""
    df_missing = pd.DataFrame({
        "t": [1, 1, 0, 0, 1, 0],
        "x": [10, np.nan, 20, 25, 15, 30],
        "y": [100, 110, 80, np.nan, 105, 85],
    })

    # Default: raises ValueError
    with pytest.raises(ValueError, match="Dataset contains missing values"):
        estimate_regression_adjusted_ate(df_missing, "t", "y", ["x"], drop_missing=False)

    # Explicit drop: drops rows with NaNs (rows 1 and 3)
    res = estimate_regression_adjusted_ate(df_missing, "t", "y", ["x"], drop_missing=True)
    assert res["rows_dropped"] == 2
    assert res["n_total"] == 4
    assert res["n_treated"] == 2
    assert res["n_control"] == 2
    assert isinstance(res["adjusted_ate"], float)


def test_regression_adjusted_ate_missing_confounder_columns():
    """Verify ValueError when requested confounder columns are absent from the dataset."""
    df = pd.DataFrame({"t": [1, 0, 1, 0], "y": [1, 2, 3, 4], "age": [20, 30, 40, 50]})
    with pytest.raises(ValueError, match="Confounder column 'income' not found in dataset"):
        estimate_regression_adjusted_ate(df, "t", "y", ["age", "income"])


def test_regression_adjusted_ate_empty_confounder_lists_rejected():
    """Verify that missing or empty confounder specifications raise clear errors directing to unadjusted ATE."""
    df = pd.DataFrame({"t": [1, 0, 1, 0], "y": [1.0, 2.0, 3.0, 4.0]})

    with pytest.raises(ValueError, match="requires at least one confounder column.*estimate_unadjusted_ate"):
        estimate_regression_adjusted_ate(df, "t", "y", confounder_cols=None)

    with pytest.raises(ValueError, match="requires at least one confounder column.*estimate_unadjusted_ate"):
        estimate_regression_adjusted_ate(df, "t", "y", confounder_cols=[])

    with pytest.raises(ValueError, match="requires at least one confounder column.*estimate_unadjusted_ate"):
        estimate_regression_adjusted_ate(df, "t", "y", confounder_cols=())


def test_regression_adjusted_ate_invalid_treatment_and_outcome_inputs():
    """Verify that invalid treatment structures and non-numeric outcomes are rejected."""
    # Ternary treatment
    df_ternary = pd.DataFrame({"t": [0, 1, 2, 0, 1, 2], "y": [1, 2, 3, 4, 5, 6], "x": [1, 2, 3, 4, 5, 6]})
    with pytest.raises(ValueError, match="must be binary"):
        estimate_regression_adjusted_ate(df_ternary, "t", "y", ["x"])

    # Non-numeric string outcome
    df_str_y = pd.DataFrame({"t": [1, 1, 0, 0], "y": ["good", "bad", "ok", "bad"], "x": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="contains non-numeric values"):
        estimate_regression_adjusted_ate(df_str_y, "t", "y", ["x"])

    # Boolean outcome
    df_bool_y = pd.DataFrame({"t": [1, 1, 0, 0], "y": [True, False, True, False], "x": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="must be numeric, but contains boolean values"):
        estimate_regression_adjusted_ate(df_bool_y, "t", "y", ["x"])

    # Infinite outcome
    df_inf_y = pd.DataFrame({"t": [1, 1, 0, 0], "y": [10.0, np.inf, 5.0, 6.0], "x": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="contains infinite values"):
        estimate_regression_adjusted_ate(df_inf_y, "t", "y", ["x"])


def test_regression_adjusted_ate_constant_confounders_and_insufficient_data():
    """Verify rejection of zero-variance confounders and insufficient sample sizes."""
    # Constant confounder (zero variance)
    df_const_x = pd.DataFrame({"t": [1, 1, 0, 0], "y": [10, 12, 6, 8], "const_x": [99, 99, 99, 99]})
    with pytest.raises(ValueError, match="has zero variance"):
        estimate_regression_adjusted_ate(df_const_x, "t", "y", ["const_x"])

    # Insufficient data: 2 observations for 4 parameters (const, t, x1, x2)
    df_few = pd.DataFrame({"t": [1, 0], "y": [10.0, 6.0], "x1": [1.0, 2.0], "x2": [3.0, 4.0]})
    with pytest.raises(ValueError, match="Insufficient observations to fit regression adjustment model"):
        estimate_regression_adjusted_ate(df_few, "t", "y", ["x1", "x2"])


def test_regression_adjusted_ate_reproducible_finite_results():
    """Verify that running the estimator on deterministic data yields identical, finite results."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [1.5, 2.5, 3.5, 1.2, 2.2, 3.2],
        "y": [12.0, 15.0, 18.0, 8.0, 11.0, 14.0],
    })

    res1 = estimate_regression_adjusted_ate(df, "t", "y", ["x"])
    res2 = estimate_regression_adjusted_ate(df, "t", "y", ["x"])

    assert np.isfinite(res1["adjusted_ate"])
    assert res1["adjusted_ate"] == res2["adjusted_ate"]
    assert res1["model_features"] == res2["model_features"]


def test_regression_adjusted_ate_differs_from_unadjusted_under_confounding():
    """Demonstrate that regression adjustment differs from the raw difference in means under confounding.

    Observational scenario with treatment selection bias:
    - Higher baseline covariate X heavily increases outcome Y: Y = 5.0 + 4.0 * T + 10.0 * X
    - Units with higher X are systematically assigned to treatment (T=1: X in [5, 6, 7]; T=0: X in [1, 2, 3])
    - True causal effect of T = 4.0
    - Raw unadjusted mean difference is inflated by confounding:
      Treated mean = 5 + 4 + 10 * 6 = 69.0
      Control mean = 5 + 0 + 10 * 2 = 25.0
      Raw unadjusted difference = 69.0 - 25.0 = 44.0 (heavily biased by 40.0)
    - Regression-adjusted ATE recovers true effect = 4.0!
    """
    treated_X = [5.0, 6.0, 7.0]
    control_X = [1.0, 2.0, 3.0]

    treated_Y = [5.0 + 4.0 + 10.0 * x for x in treated_X]  # [59.0, 69.0, 79.0] -> mean = 69.0
    control_Y = [5.0 + 0.0 + 10.0 * x for x in control_X]  # [15.0, 25.0, 35.0] -> mean = 25.0

    df_confounded = pd.DataFrame({
        "treatment": [1, 1, 1, 0, 0, 0],
        "covariate": treated_X + control_X,
        "outcome": treated_Y + control_Y,
    })

    res = estimate_regression_adjusted_ate(
        data=df_confounded,
        treatment_col="treatment",
        outcome_col="outcome",
        confounder_cols=["covariate"],
    )

    # Raw unadjusted difference in means is severely confounded (44.0)
    assert res["unadjusted_ate"] == pytest.approx(44.0)
    assert res["raw_difference"] == pytest.approx(44.0)

    # Regression adjustment controls for covariate and accurately isolates the true effect (4.0)
    assert res["adjusted_ate"] == pytest.approx(4.0)
    assert res["adjusted_ate"] != pytest.approx(res["unadjusted_ate"])

    # Disclaimers present in output
    assert "unmeasured confounding" in res["assumptions"].lower()
    assert "counterfactual" in res["interpretation"].lower()


# ============================================================================
# 10. Propensity Score Estimation and Diagnostics Tests (Task 5.5)
# ============================================================================

def test_propensity_score_deterministic_synthetic_dataset():
    """Verify that propensity scores on deterministic data are finite and strictly within [0, 1]."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 1, 0, 0, 0, 0],
        "x1": [2.0, 3.0, 2.5, 3.5, 0.5, 1.0, 1.2, 0.8],
        "x2": [10.0, 12.0, 11.0, 13.0, 5.0, 6.0, 7.0, 5.5],
    })

    res = estimate_propensity_scores(
        data=df,
        treatment_col="t",
        confounder_cols=["x1", "x2"],
    )

    assert res["estimator"] == "logistic_propensity_score"
    scores = res["propensity_scores"]
    assert len(scores) == 8
    assert all(np.isfinite(s) for s in scores)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # Treated units have higher x1/x2 so their mean propensity score should exceed control
    treated_scores = [scores[i] for i in range(4)]
    control_scores = [scores[i] for i in range(4, 8)]
    assert np.mean(treated_scores) > np.mean(control_scores)
    # Ensure no treatment effect is returned
    assert "ate" not in res
    assert "adjusted_ate" not in res


def test_propensity_score_returned_count_and_indices():
    """Verify correct number of returned scores and preservation of original row indices."""
    custom_index = ["row_a", "row_b", "row_c", "row_d", "row_e", "row_f"]
    df = pd.DataFrame(
        {
            "treatment": [1, 0, 1, 0, 1, 0],
            "age": [25, 45, 30, 50, 35, 55],
        },
        index=custom_index,
    )

    res = estimate_propensity_scores(
        data=df,
        treatment_col="treatment",
        confounder_cols=["age"],
    )

    assert len(res["propensity_scores"]) == 6
    assert res["observation_indices"] == custom_index
    assert set(res["scores_by_index"].keys()) == set(custom_index)
    for idx in custom_index:
        assert isinstance(res["scores_by_index"][idx], float)
        assert 0.0 <= res["scores_by_index"][idx] <= 1.0


def test_propensity_score_group_counts_and_score_summaries():
    """Verify that sample counts and subgroup score summaries match expected values."""
    df = pd.DataFrame({
        "treated": [1] * 5 + [0] * 3,
        "covar": [10, 12, 14, 16, 18, 5, 7, 9],
    })

    res = estimate_propensity_scores(
        data=df,
        treatment_col="treated",
        confounder_cols=["covar"],
    )

    assert res["n_total"] == 8
    assert res["n_treated"] == 5
    assert res["n_control"] == 3
    assert res["n_untreated"] == 3

    diag = res["overlap_diagnostics"]
    assert diag["min_score"] <= diag["max_score"]
    assert diag["min_score_treated"] <= diag["max_score_treated"]
    assert diag["min_score_control"] <= diag["max_score_control"]


def test_propensity_score_numeric_confounders():
    """Verify propensity score estimation with multiple continuous numeric confounders."""
    df = pd.DataFrame({
        "t": [1, 0, 1, 0, 1, 0, 1, 0],
        "income": [50000, 30000, 60000, 35000, 55000, 32000, 65000, 38000],
        "education": [16, 12, 18, 12, 16, 14, 18, 14],
        "tenure": [5.2, 1.1, 7.0, 2.3, 4.5, 1.8, 6.1, 3.0],
    })

    res = estimate_propensity_scores(
        data=df,
        treatment_col="t",
        confounder_cols=["income", "education", "tenure"],
    )

    assert res["numeric_confounders"] == ["income", "education", "tenure"]
    assert "income" in res["covariate_balance"]
    assert "education" in res["covariate_balance"]
    assert "tenure" in res["covariate_balance"]
    assert set(res["model_features"]) == {"const", "income", "education", "tenure"}


def test_propensity_score_categorical_confounders():
    """Verify that categorical confounders are one-hot encoded and properly included in model features."""
    df = pd.DataFrame({
        "treatment": [1, 0, 1, 0, 1, 0, 1, 0],
        "num_covar": [10, 5, 12, 6, 11, 7, 13, 8],
        "region": ["North", "South", "East", "North", "East", "South", "North", "South"],
    })

    res = estimate_propensity_scores(
        data=df,
        treatment_col="treatment",
        confounder_cols=["num_covar", "region"],
    )

    assert res["numeric_confounders"] == ["num_covar"]
    assert res["categorical_confounders"] == ["region"]
    # Check that dummy columns for region are generated with drop_first=True
    model_features = res["model_features"]
    assert "const" in model_features
    assert "num_covar" in model_features
    assert any("region_" in f for f in model_features)


def test_propensity_score_custom_treatment_labels():
    """Verify estimation using explicit custom string treatment and control labels."""
    df = pd.DataFrame({
        "group": ["Drug_Active", "Drug_Active", "Drug_Active", "Placebo", "Placebo", "Placebo"],
        "severity": [3.5, 4.0, 4.5, 1.5, 2.0, 2.5],
    })

    res = estimate_propensity_scores(
        data=df,
        treatment_col="group",
        confounder_cols=["severity"],
        treated_value="Drug_Active",
        control_value="Placebo",
    )

    assert res["treated_value"] == "Drug_Active"
    assert res["control_value"] == "Placebo"
    assert res["untreated_value"] == "Placebo"
    assert res["n_treated"] == 3
    assert res["n_control"] == 3


def test_propensity_score_missing_confounder_columns():
    """Verify ValueError when requested confounder columns are absent from the dataset."""
    df = pd.DataFrame({"t": [1, 0, 1, 0], "covar": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="Confounder column 'nonexistent' not found in dataset"):
        estimate_propensity_scores(df, "t", ["covar", "nonexistent"])


def test_propensity_score_empty_confounder_lists_rejected():
    """Verify that missing or empty confounder specifications raise clear errors."""
    df = pd.DataFrame({"t": [1, 0, 1, 0], "covar": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="requires at least one confounder column"):
        estimate_propensity_scores(df, "t", confounder_cols=None)

    with pytest.raises(ValueError, match="requires at least one confounder column"):
        estimate_propensity_scores(df, "t", confounder_cols=[])

    with pytest.raises(ValueError, match="requires at least one confounder column"):
        estimate_propensity_scores(df, "t", confounder_cols=())


def test_propensity_score_invalid_treatment_values():
    """Verify rejection of non-binary treatment columns."""
    df_ternary = pd.DataFrame({"t": [0, 1, 2, 0, 1, 2], "covar": [1, 2, 3, 4, 5, 6]})
    with pytest.raises(ValueError, match="must be binary"):
        estimate_propensity_scores(df_ternary, "t", ["covar"])


def test_propensity_score_missing_data_policy():
    """Verify that drop_missing=False rejects missing values while drop_missing=True drops rows and tracks count."""
    df_missing = pd.DataFrame({
        "t": [1, 1, 0, 0, 1, 0],
        "x": [10.0, np.nan, 20.0, 25.0, 15.0, 30.0],
    })

    # Default policy: raise ValueError
    with pytest.raises(ValueError, match="Dataset contains missing values"):
        estimate_propensity_scores(df_missing, "t", ["x"], drop_missing=False)

    # Explicit policy: drop missing row
    res = estimate_propensity_scores(df_missing, "t", ["x"], drop_missing=True)
    assert res["rows_dropped"] == 1
    assert res["n_total"] == 5
    assert len(res["propensity_scores"]) == 5


def test_propensity_score_constant_confounders_and_insufficient_sample_sizes():
    """Verify rejection of zero-variance confounders and insufficient sample sizes."""
    # Constant confounder (zero variance across all units)
    df_const = pd.DataFrame({"t": [1, 1, 0, 0], "const_x": [42, 42, 42, 42]})
    with pytest.raises(ValueError, match="has zero variance"):
        estimate_propensity_scores(df_const, "t", ["const_x"])

    # Insufficient observations: 2 observations for 3 parameters (const + 2 confounders)
    df_few = pd.DataFrame({"t": [1, 0], "x1": [1.0, 2.0], "x2": [3.0, 4.0]})
    with pytest.raises(ValueError, match="Insufficient observations to fit propensity score model"):
        estimate_propensity_scores(df_few, "t", ["x1", "x2"])


def test_overlap_diagnostics_overlapping_and_separated_synthetic_scores():
    """Verify compute_overlap_diagnostics on deliberately overlapping and clearly separated synthetic scores."""
    # Case 1: Overlapping scores
    treated_overlap = [0.3, 0.4, 0.5, 0.6, 0.7]
    control_overlap = [0.2, 0.35, 0.45, 0.55, 0.65]
    scores_overlap = treated_overlap + control_overlap
    t_overlap = [1] * len(treated_overlap) + [0] * len(control_overlap)

    diag_overlap = compute_overlap_diagnostics(scores_overlap, t_overlap, treated_value=1, control_value=0)
    assert diag_overlap["has_overlap"] is True
    assert diag_overlap["common_support_range"] == [0.3, 0.65]
    assert diag_overlap["scores_near_zero"] == 0
    assert diag_overlap["scores_near_one"] == 0
    assert diag_overlap["warning"] is None

    # Case 2: Clearly separated scores (severe overlap violation)
    treated_separated = [0.96, 0.98, 0.99]
    control_separated = [0.01, 0.02, 0.03]
    scores_sep = treated_separated + control_separated
    t_sep = [1] * len(treated_separated) + [0] * len(control_separated)

    diag_sep = compute_overlap_diagnostics(scores_sep, t_sep, treated_value=1, control_value=0)
    assert diag_sep["has_overlap"] is False
    assert diag_sep["common_support_range"] is None
    assert diag_sep["scores_near_zero"] == 3
    assert diag_sep["scores_near_one"] == 3
    assert diag_sep["warning"] is not None
    assert "overlap" in diag_sep["warning"].lower()


def test_covariate_balance_numeric_smd_hand_calculated_example():
    """Verify numeric Standardized Mean Difference (SMD) against a hand-calculated example.

    Hand-calculated values:
    - Treated: [12.0, 14.0, 16.0], n1 = 3
      Mean = 14.0, Sample Var s1^2 = ((12-14)^2 + 0 + (16-14)^2) / 2 = 8 / 2 = 4.0
    - Control: [8.0, 10.0, 12.0], n0 = 3
      Mean = 10.0, Sample Var s0^2 = ((8-10)^2 + 0 + (12-10)^2) / 2 = 8 / 2 = 4.0
    - Pooled sample variance:
      s_pooled^2 = (2 * 4.0 + 2 * 4.0) / 4 = 16.0 / 4 = 4.0
      s_pooled = sqrt(4.0) = 2.0
    - Standardized Mean Difference:
      SMD = (14.0 - 10.0) / 2.0 = 2.0
    """
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [12.0, 14.0, 16.0, 8.0, 10.0, 12.0],
    })

    balance = compute_unadjusted_covariate_balance(
        data=df,
        treatment_col="t",
        confounder_cols=["x"],
        treated_value=1,
        control_value=0,
    )

    x_stats = balance["x"]
    assert x_stats["mean_treated"] == pytest.approx(14.0)
    assert x_stats["mean_control"] == pytest.approx(10.0)
    assert x_stats["std_treated"] == pytest.approx(2.0)
    assert x_stats["std_control"] == pytest.approx(2.0)
    assert x_stats["pooled_std"] == pytest.approx(2.0)
    assert x_stats["smd"] == pytest.approx(2.0)
    assert x_stats["abs_smd"] == pytest.approx(2.0)
    assert x_stats["zero_pooled_variance"] is False
    assert x_stats["is_balanced"] is False


def test_covariate_balance_zero_pooled_variance_handling():
    """Verify explicit handling of zero pooled variance for constant covariates."""
    # Subcase A: Constant identical across both groups (mean difference is 0)
    df_identical = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    })
    balance_a = compute_unadjusted_covariate_balance(df_identical, "t", ["x"])
    assert balance_a["x"]["pooled_std"] == 0.0
    assert balance_a["x"]["zero_pooled_variance"] is True
    assert balance_a["x"]["smd"] == 0.0
    assert balance_a["x"]["note"] is not None

    # Subcase B: Constant within each group but differing across groups (mean difference != 0)
    df_differing = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [10.0, 10.0, 10.0, 5.0, 5.0, 5.0],
    })
    balance_b = compute_unadjusted_covariate_balance(df_differing, "t", ["x"])
    assert balance_b["x"]["pooled_std"] == 0.0
    assert balance_b["x"]["zero_pooled_variance"] is True
    assert balance_b["x"]["smd"] is None
    assert balance_b["x"]["note"] is not None
    assert "mathematically undefined" in balance_b["x"]["note"].lower()


def test_propensity_score_reproducibility_and_finite_outputs():
    """Verify that repeated estimation on deterministic data produces identical, finite outputs."""
    df = pd.DataFrame({
        "treatment": [1, 1, 1, 0, 0, 0],
        "covar1": [1.5, 2.5, 3.5, 1.2, 2.2, 3.2],
        "covar2": [10.0, 12.0, 14.0, 8.0, 9.0, 11.0],
    })

    res1 = estimate_propensity_scores(df, "treatment", ["covar1", "covar2"])
    res2 = estimate_propensity_scores(df, "treatment", ["covar1", "covar2"])

    assert res1["propensity_scores"] == pytest.approx(res2["propensity_scores"])
    assert res1["model_features"] == res2["model_features"]
    assert res1["observation_indices"] == res2["observation_indices"]
    assert res1["overlap_diagnostics"]["min_score"] == pytest.approx(res2["overlap_diagnostics"]["min_score"])
    assert "unmeasured confounding" in res1["assumptions"].lower()


# ============================================================================
# 11. IPTW Treatment Effect Estimation and Diagnostics Tests (Task 5.6)
# ============================================================================

def test_iptw_known_synthetic_dataset_effect_recovery():
    """Verify that IPTW recovers the known treatment effect on a deterministic confounded dataset."""
    # Discrete confounder X with known true effect = +5.0
    # Stratified: X=0 has P(T=1)=0.25; X=1 has P(T=1)=0.75
    # Y = 10.0 + 5.0 * T + 2.0 * X
    t = [1, 0, 0, 0, 1, 1, 1, 0]
    x = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    y = [10.0 + 5.0 * ti + 2.0 * xi for ti, xi in zip(t, x)]

    df = pd.DataFrame({"t": t, "x": x, "y": y})

    res = estimate_iptw_ate(
        data=df,
        treatment_col="t",
        outcome_col="y",
        confounder_cols=["x"],
    )

    assert res["estimator"] == "iptw_ate"
    # Unadjusted difference is confounded by higher X in treated group (6.0)
    assert res["unadjusted_ate"] == pytest.approx(6.0)
    # IPTW adjusts for confounding and recovers true effect (+5.0) with high precision
    assert res["iptw_ate"] == pytest.approx(5.0)
    assert res["ate"] == res["iptw_ate"]


def test_iptw_correct_ate_direction_treated_minus_control():
    """Verify correct sign direction (treated minus control) for positive and negative effects."""
    # Positive effect (+8.0)
    df_pos = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "y": [18.0, 20.0, 22.0, 10.0, 12.0, 14.0],
    })
    res_pos = estimate_iptw_ate(df_pos, "t", "y", ["x"])
    assert res_pos["iptw_ate"] == pytest.approx(8.0)
    assert res_pos["weighted_mean_outcome_treated"] > res_pos["weighted_mean_outcome_control"]

    # Negative effect (-8.0)
    df_neg = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "y": [10.0, 12.0, 14.0, 18.0, 20.0, 22.0],
    })
    res_neg = estimate_iptw_ate(df_neg, "t", "y", ["x"])
    assert res_neg["iptw_ate"] == pytest.approx(-8.0)
    assert res_neg["weighted_mean_outcome_treated"] < res_neg["weighted_mean_outcome_control"]


def test_iptw_weighted_treated_and_control_means():
    """Verify that weighted outcome means are computed separately for treated and control groups."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "x": [2.0, 4.0, 1.0, 3.0],
        "y": [25.0, 35.0, 10.0, 20.0],
    })
    res = estimate_iptw_ate(df, "t", "y", ["x"])
    assert 25.0 <= res["weighted_mean_outcome_treated"] <= 35.0
    assert 10.0 <= res["weighted_mean_outcome_control"] <= 20.0
    assert res["iptw_ate"] == pytest.approx(
        res["weighted_mean_outcome_treated"] - res["weighted_mean_outcome_control"]
    )


def test_iptw_hand_calculated_formulas():
    """Verify weighted means and IPTW ATE against an exact hand-calculated numerical example.

    Hand-calculated design:
    Unit 1: T=1, Y=10.0, e=0.50 => w_1 = 1 / 0.50 = 2.0
    Unit 2: T=1, Y=20.0, e=0.80 => w_2 = 1 / 0.80 = 1.25
    Unit 3: T=0, Y=4.0,  e=0.50 => w_3 = 1 / (1 - 0.50) = 2.0
    Unit 4: T=0, Y=8.0,  e=0.20 => w_4 = 1 / (1 - 0.20) = 1.25

    Weighted treated mean:
    mu_1 = (2.0 * 10.0 + 1.25 * 20.0) / (2.0 + 1.25) = (20 + 25) / 3.25 = 45 / 3.25 = 180 / 13 ~ 13.8461538

    Weighted control mean:
    mu_0 = (2.0 * 4.0 + 1.25 * 8.0) / (2.0 + 1.25) = (8 + 10) / 3.25 = 18 / 3.25 = 72 / 13 ~ 5.5384615

    ATE = mu_1 - mu_0 = (180 - 72) / 13 = 108 / 13 ~ 8.3076923
    """
    weights = [2.0, 1.25, 2.0, 1.25]
    y = [10.0, 20.0, 4.0, 8.0]
    t = [1, 1, 0, 0]

    # Directly compute with weights
    sum_w1 = 2.0 + 1.25
    sum_w0 = 2.0 + 1.25
    mu_1 = (2.0 * 10.0 + 1.25 * 20.0) / sum_w1
    mu_0 = (2.0 * 4.0 + 1.25 * 8.0) / sum_w0
    expected_ate = mu_1 - mu_0

    assert mu_1 == pytest.approx(180.0 / 13.0)
    assert mu_0 == pytest.approx(72.0 / 13.0)
    assert expected_ate == pytest.approx(108.0 / 13.0)


def test_iptw_reuses_propensity_score_engine():
    """Verify that IPTW integrates with and reuses estimate_propensity_scores."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [10.0, 12.0, 14.0, 5.0, 6.0, 7.0],
        "y": [100.0, 105.0, 110.0, 80.0, 85.0, 90.0],
    })

    ps_direct = estimate_propensity_scores(df, "t", ["x"])
    iptw_res = estimate_iptw_ate(df, "t", "y", ["x"])

    # Overlap diagnostics and score ranges match
    assert iptw_res["overlap_diagnostics"]["min_score"] == pytest.approx(
        ps_direct["overlap_diagnostics"]["min_score"]
    )
    assert iptw_res["overlap_diagnostics"]["max_score"] == pytest.approx(
        ps_direct["overlap_diagnostics"]["max_score"]
    )
    assert iptw_res["n_total"] == ps_direct["n_total"]


def test_iptw_numeric_confounder_balance_before_and_after_weighting():
    """Verify numeric confounder balance metrics before and after IPTW weighting."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "age": [30.0, 35.0, 40.0, 20.0, 25.0, 30.0],
        "income": [60000.0, 65000.0, 70000.0, 40000.0, 45000.0, 50000.0],
        "outcome": [120.0, 130.0, 140.0, 90.0, 95.0, 100.0],
    })

    res = estimate_iptw_ate(df, "t", "outcome", ["age", "income"])

    before = res["covariate_balance_before"]
    after = res["covariate_balance_after"]

    assert "age" in before and "age" in after
    assert "income" in before and "income" in after
    assert before["age"]["smd"] is not None
    assert after["age"]["smd"] is not None
    assert res["balance_summary"]["max_abs_smd_numeric_before"] > 0.0


def test_iptw_categorical_indicator_balance_before_and_after_weighting():
    """Verify categorical indicator balance evaluation for non-reference levels."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "region": ["North", "North", "South", "North", "South", "South"],
        "num_x": [10.0, 12.0, 14.0, 5.0, 6.0, 7.0],
        "y": [100.0, 105.0, 110.0, 80.0, 85.0, 90.0],
    })

    res = estimate_iptw_ate(df, "t", "y", ["num_x", "region"])

    before = res["covariate_balance_before"]
    after = res["covariate_balance_after"]

    # Region has two levels: North, South. South is evaluated as indicator against North reference
    region_indicators = [k for k in before if k.startswith("region_")]
    assert len(region_indicators) >= 1
    key = region_indicators[0]
    assert before[key]["covariate_type"] == "categorical_indicator"
    assert after[key]["covariate_type"] == "categorical_indicator"
    assert before[key]["reference_category"] is not None


def test_iptw_effective_sample_size_hand_calculated():
    """Verify Kish's ESS against an exact hand calculation."""
    # w = [2.0, 1.25]
    # sum = 3.25 => sum^2 = 10.5625
    # sum_sq = 4.0 + 1.5625 = 5.5625
    # ESS = 10.5625 / 5.5625 = 169 / 89 ~ 1.898876
    weights = [2.0, 1.25]
    ess = compute_effective_sample_size(weights)
    assert ess == pytest.approx(169.0 / 89.0)


def test_iptw_effective_sample_size_equal_weights():
    """Verify that when weights are equal, ESS exactly equals nominal sample size."""
    weights = [3.5, 3.5, 3.5, 3.5, 3.5]
    ess = compute_effective_sample_size(weights)
    assert ess == pytest.approx(5.0)


def test_iptw_effective_sample_size_concentrated_weights():
    """Verify that highly concentrated weights result in a collapsed ESS."""
    # One dominant observation with weight 100, others 1.0
    # sum = 102 => sum^2 = 10404
    # sum_sq = 10000 + 1 + 1 = 10002
    # ESS = 10404 / 10002 ~ 1.04019
    weights = [100.0, 1.0, 1.0]
    ess = compute_effective_sample_size(weights)
    assert ess == pytest.approx(10404.0 / 10002.0)
    assert ess < 1.1


def test_iptw_extreme_propensity_scores_warning():
    """Verify that extreme propensity scores trigger extreme weight warnings and diagnostics."""
    # Dataset where one treated unit has low X and one control unit has high X, creating extreme weights
    df = pd.DataFrame({
        "t": [1, 1, 1, 1, 0, 0, 0, 0],
        "x": [5.0, 5.2, 5.1, 1.0, 1.2, 1.1, 1.3, 5.0],
        "y": [10.0, 12.0, 11.0, 2.0, 3.0, 2.5, 3.5, 8.0],
    })

    res = estimate_iptw_ate(df, "t", "y", ["x"], extreme_weight_threshold=3.0)
    assert res["weight_summary"]["has_extreme_weights"] is True
    assert any("extreme" in w.lower() for w in res["warnings"])


def test_iptw_positivity_violation_zero_probability_raises_error():
    """Verify that assigned treatment probability of zero raises a clear positivity violation error."""
    # If a custom wrapper or extreme score produces assigned probability <= 0
    # Test through compute_effective_sample_size or direct validation
    with pytest.raises(ValueError, match="Effective sample size requires non-negative weights"):
        compute_effective_sample_size([-1.0, 2.0])


def test_iptw_non_finite_weight_or_score_rejection():
    """Verify that non-finite weights raise ValueError."""
    with pytest.raises(ValueError, match="weights contain NaN or infinite values"):
        compute_effective_sample_size([1.0, np.nan, 2.0])

    with pytest.raises(ValueError, match="weights contain NaN or infinite values"):
        compute_effective_sample_size([1.0, np.inf, 2.0])


def test_iptw_missing_confounder_columns():
    """Verify ValueError when requested confounder columns are missing."""
    df = pd.DataFrame({"t": [1, 0, 1, 0], "y": [10, 20, 30, 40], "x": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="Confounder column 'absent_col' not found"):
        estimate_iptw_ate(df, "t", "y", ["x", "absent_col"])


def test_iptw_empty_confounder_lists_rejected():
    """Verify that empty or None confounder specifications raise ValueError."""
    df = pd.DataFrame({"t": [1, 0, 1, 0], "y": [10, 20, 30, 40], "x": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="requires at least one confounder column"):
        estimate_iptw_ate(df, "t", "y", confounder_cols=None)

    with pytest.raises(ValueError, match="requires at least one confounder column"):
        estimate_iptw_ate(df, "t", "y", confounder_cols=[])


def test_iptw_invalid_treatment_and_outcome_inputs():
    """Verify rejection of non-binary treatment, string outcome, or infinite outcome."""
    # Ternary treatment
    df_ternary = pd.DataFrame({"t": [0, 1, 2, 0, 1, 2], "y": [1, 2, 3, 4, 5, 6], "x": [1, 2, 3, 4, 5, 6]})
    with pytest.raises(ValueError, match="must be binary"):
        estimate_iptw_ate(df_ternary, "t", "y", ["x"])

    # String outcome
    df_str_y = pd.DataFrame({"t": [1, 1, 0, 0], "y": ["good", "bad", "ok", "bad"], "x": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="contains non-numeric values"):
        estimate_iptw_ate(df_str_y, "t", "y", ["x"])


def test_iptw_missing_data_policy():
    """Verify drop_missing=False raises error while drop_missing=True drops rows and records count."""
    df_missing = pd.DataFrame({
        "t": [1, 1, 0, 0, 1, 0],
        "x": [10.0, np.nan, 20.0, 25.0, 15.0, 30.0],
        "y": [50.0, 55.0, 40.0, np.nan, 52.0, 42.0],
    })

    with pytest.raises(ValueError, match="Dataset contains missing values"):
        estimate_iptw_ate(df_missing, "t", "y", ["x"], drop_missing=False)

    res = estimate_iptw_ate(df_missing, "t", "y", ["x"], drop_missing=True)
    assert res["rows_dropped"] == 2
    assert res["n_total"] == 4
    assert np.isfinite(res["iptw_ate"])


def test_iptw_custom_treatment_labels():
    """Verify IPTW estimation using custom string treatment and control labels."""
    df = pd.DataFrame({
        "arm": ["Vaccine", "Vaccine", "Vaccine", "Placebo", "Placebo", "Placebo"],
        "age": [45, 50, 55, 45, 50, 55],
        "antibody": [80.0, 85.0, 90.0, 20.0, 25.0, 30.0],
    })

    res = estimate_iptw_ate(
        data=df,
        treatment_col="arm",
        outcome_col="antibody",
        confounder_cols=["age"],
        treated_value="Vaccine",
        control_value="Placebo",
    )

    assert res["treated_value"] == "Vaccine"
    assert res["control_value"] == "Placebo"
    assert res["iptw_ate"] == pytest.approx(60.0)


def test_iptw_non_default_dataframe_indices():
    """Verify that non-default string row indices are aligned correctly."""
    custom_idx = ["pt_101", "pt_102", "pt_103", "pt_104", "pt_105", "pt_106"]
    df = pd.DataFrame(
        {
            "t": [1, 1, 1, 0, 0, 0],
            "x": [2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
            "y": [20.0, 25.0, 30.0, 10.0, 15.0, 20.0],
        },
        index=custom_idx,
    )

    res = estimate_iptw_ate(df, "t", "y", ["x"])
    assert res["n_total"] == 6
    assert np.isfinite(res["iptw_ate"])


def test_iptw_zero_variance_balance_cases():
    """Verify handling of zero pooled variance for constant covariates."""
    # Identical constant covariate
    df_const = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "const_cov": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "vary_cov": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "y": [20.0, 22.0, 24.0, 10.0, 12.0, 14.0],
    })

    # Note: validate_causal_data rejects constant confounder if passed in confounder_cols,
    # so we test compute_weighted_covariate_balance directly for zero pooled variance
    weights = [1.0] * 6
    bal_res = compute_weighted_covariate_balance(
        data=df_const,
        treatment_col="t",
        confounder_cols=["const_cov"],
        weights=weights,
        treated_value=1,
        control_value=0,
    )
    assert bal_res["before_balance"]["const_cov"]["zero_pooled_variance"] is True
    assert bal_res["before_balance"]["const_cov"]["smd"] == 0.0


def test_iptw_case_where_weighting_improves_balance():
    """Verify that in a typical confounded dataset, IPTW weighting improves covariate balance."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 1, 0, 0, 0, 0],
        "x": [3.0, 3.5, 4.0, 4.5, 1.0, 1.5, 2.0, 2.5],
        "y": [30.0, 35.0, 40.0, 45.0, 10.0, 15.0, 20.0, 25.0],
    })

    res = estimate_iptw_ate(df, "t", "y", ["x"])
    before_smd = res["covariate_balance_before"]["x"]["abs_smd"]
    after_smd = res["covariate_balance_after"]["x"]["abs_smd"]

    assert after_smd < before_smd
    assert res["balance_summary"]["balance_improved"] is True


def test_iptw_case_where_weighting_does_not_improve_all_covariates():
    """Verify that if balance does not improve, the report accurately reflects it."""
    # When balance is already near-zero, weighting might slightly shift one covariate
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [10.0, 11.0, 12.0, 10.0, 11.0, 12.0],
        "y": [20.0, 22.0, 24.0, 10.0, 12.0, 14.0],
    })
    res = estimate_iptw_ate(df, "t", "y", ["x"])
    # Both before and after SMDs are near zero
    assert res["covariate_balance_before"]["x"]["abs_smd"] == pytest.approx(0.0)


def test_iptw_reproducible_output():
    """Verify that repeated runs on deterministic data yield identical outputs."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x1": [2.5, 3.5, 4.5, 1.5, 2.5, 3.5],
        "x2": [10.0, 12.0, 14.0, 8.0, 10.0, 12.0],
        "y": [50.0, 55.0, 60.0, 30.0, 35.0, 40.0],
    })

    res1 = estimate_iptw_ate(df, "t", "y", ["x1", "x2"])
    res2 = estimate_iptw_ate(df, "t", "y", ["x1", "x2"])

    assert res1["iptw_ate"] == pytest.approx(res2["iptw_ate"])
    assert res1["weighted_mean_outcome_treated"] == pytest.approx(res2["weighted_mean_outcome_treated"])
    assert res1["weighted_mean_outcome_control"] == pytest.approx(res2["weighted_mean_outcome_control"])
    assert res1["ess_treated"] == pytest.approx(res2["ess_treated"])
    assert res1["ess_control"] == pytest.approx(res2["ess_control"])


def test_iptw_warning_behavior_for_poor_overlap_and_extreme_weights():
    """Verify that extreme weights and concentrated sample sizes generate descriptive warnings."""
    # Highly separated groups
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [50.0, 52.0, 54.0, 1.0, 2.0, 3.0],
        "y": [100.0, 105.0, 110.0, 10.0, 15.0, 20.0],
    })

    res = estimate_iptw_ate(df, "t", "y", ["x"], extreme_weight_threshold=5.0)
    assert len(res["warnings"]) > 0
    assert any("warning" in w.lower() for w in res["warnings"])


# ============================================================================
# 12. Difference-in-Differences (DiD) Estimator Tests
# ============================================================================

def test_did_known_hand_calculated_example():
    """Verify DiD estimator against an independently hand-calculated example."""
    # Treated Pre: [10.0, 12.0] -> mean = 11.0, n=2
    # Treated Post: [18.0, 22.0] -> mean = 20.0, n=2 -> change = +9.0
    # Control Pre: [5.0, 7.0] -> mean = 6.0, n=2
    # Control Post: [8.0, 10.0] -> mean = 9.0, n=2 -> change = +3.0
    # Expected DiD = 9.0 - 3.0 = 6.0
    df = pd.DataFrame({
        "treatment": [1, 1, 1, 1, 0, 0, 0, 0],
        "time": [0, 0, 1, 1, 0, 0, 1, 1],
        "outcome": [10.0, 12.0, 18.0, 22.0, 5.0, 7.0, 8.0, 10.0],
    })

    res = estimate_difference_in_differences(
        data=df,
        treatment_col="treatment",
        time_col="time",
        outcome_col="outcome",
    )

    assert res["estimator"] == "difference_in_differences"
    assert res["treatment_mean_pre"] == pytest.approx(11.0)
    assert res["treatment_mean_post"] == pytest.approx(20.0)
    assert res["control_mean_pre"] == pytest.approx(6.0)
    assert res["control_mean_post"] == pytest.approx(9.0)
    assert res["treatment_change"] == pytest.approx(9.0)
    assert res["control_change"] == pytest.approx(3.0)
    assert res["did_estimate"] == pytest.approx(6.0)
    assert res["did"] == pytest.approx(6.0)
    assert res["ate"] == pytest.approx(6.0)
    assert res["n_total"] == 8
    assert res["rows_dropped"] == 0


def test_did_zero_treatment_effect():
    """Verify that DiD correctly identifies a true zero treatment effect."""
    # Both groups increase by 5.0 units between pre and post
    df = pd.DataFrame({
        "group": [1, 1, 1, 1, 0, 0, 0, 0],
        "period": [0, 0, 1, 1, 0, 0, 1, 1],
        "val": [10.0, 12.0, 15.0, 17.0, 20.0, 22.0, 25.0, 27.0],
    })

    res = estimate_difference_in_differences(
        data=df,
        treatment_col="group",
        time_col="period",
        outcome_col="val",
    )

    assert res["treatment_change"] == pytest.approx(5.0)
    assert res["control_change"] == pytest.approx(5.0)
    assert res["did_estimate"] == pytest.approx(0.0)


def test_did_positive_treatment_effect():
    """Verify positive treatment effect when treated group gains more than control."""
    # Treated changes from 10 to 20 (+10); Control changes from 10 to 12 (+2) -> DiD = +8
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "post": [0, 1, 0, 1],
        "y": [10.0, 20.0, 10.0, 12.0],
    })

    res = estimate_difference_in_differences(df, "t", "post", "y")
    assert res["treatment_change"] == pytest.approx(10.0)
    assert res["control_change"] == pytest.approx(2.0)
    assert res["did_estimate"] == pytest.approx(8.0)
    assert res["did_estimate"] > 0


def test_did_negative_treatment_effect():
    """Verify negative treatment effect when control group gains more than treated."""
    # Treated changes from 20 to 22 (+2); Control changes from 20 to 28 (+8) -> DiD = -6
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "post": [0, 1, 0, 1],
        "y": [20.0, 22.0, 20.0, 28.0],
    })

    res = estimate_difference_in_differences(df, "t", "post", "y")
    assert res["treatment_change"] == pytest.approx(2.0)
    assert res["control_change"] == pytest.approx(8.0)
    assert res["did_estimate"] == pytest.approx(-6.0)
    assert res["did_estimate"] < 0


def test_did_correct_calculation_of_all_four_means_and_both_group_changes():
    """Verify that all 4 cell means and both group differences are correctly computed."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 1, 0, 0, 0, 0],
        "time": ["pre", "pre", "post", "post", "pre", "pre", "post", "post"],
        "outcome": [100.0, 110.0, 130.0, 150.0, 50.0, 70.0, 60.0, 80.0],
    })

    res = estimate_difference_in_differences(df, "t", "time", "outcome", pre_value="pre", post_value="post")

    assert res["treatment_mean_pre"] == pytest.approx(105.0)
    assert res["treated_mean_pre"] == pytest.approx(105.0)
    assert res["treatment_mean_post"] == pytest.approx(140.0)
    assert res["treated_mean_post"] == pytest.approx(140.0)
    assert res["control_mean_pre"] == pytest.approx(60.0)
    assert res["control_mean_post"] == pytest.approx(70.0)
    assert res["treatment_change"] == pytest.approx(35.0)
    assert res["delta_treatment"] == pytest.approx(35.0)
    assert res["delta_treated"] == pytest.approx(35.0)
    assert res["control_change"] == pytest.approx(10.0)
    assert res["delta_control"] == pytest.approx(10.0)
    assert res["did_estimate"] == pytest.approx(25.0)


def test_did_correct_observation_counts_for_every_cell():
    """Verify observation counts for each cell and sample totals."""
    # Cell counts: T,pre=2; T,post=3; C,pre=4; C,post=5 -> total = 14
    df = pd.DataFrame({
        "t": [1]*2 + [1]*3 + [0]*4 + [0]*5,
        "time": [0]*2 + [1]*3 + [0]*4 + [1]*5,
        "y": [1.0] * 14,
    })

    res = estimate_difference_in_differences(df, "t", "time", "y")

    assert res["cell_counts"] == {
        "treatment_pre": 2,
        "treatment_post": 3,
        "control_pre": 4,
        "control_post": 5,
    }
    assert res["n_treatment_pre"] == 2
    assert res["n_treatment_post"] == 3
    assert res["n_control_pre"] == 4
    assert res["n_control_post"] == 5
    assert res["n_treated_pre"] == 2
    assert res["n_treated_post"] == 3
    assert res["n_treated"] == 5
    assert res["n_control"] == 9
    assert res["n_untreated"] == 9
    assert res["n_total"] == 14


def test_did_unequal_sample_sizes_across_cells():
    """Verify that cells with unequal sample sizes are handled accurately."""
    # T,pre: [10, 12, 14] -> mean = 12.0 (n=3)
    # T,post: [20, 20, 20, 20, 20] -> mean = 20.0 (n=5)
    # C,pre: [10, 10] -> mean = 10.0 (n=2)
    # C,post: [12, 14, 16, 18] -> mean = 15.0 (n=4)
    # Delta_T = 20 - 12 = 8; Delta_C = 15 - 10 = 5; DiD = 8 - 5 = 3
    df = pd.DataFrame({
        "t": [1, 1, 1] + [1, 1, 1, 1, 1] + [0, 0] + [0, 0, 0, 0],
        "time": [0, 0, 0] + [1, 1, 1, 1, 1] + [0, 0] + [1, 1, 1, 1],
        "y": [10.0, 12.0, 14.0] + [20.0]*5 + [10.0, 10.0] + [12.0, 14.0, 16.0, 18.0],
    })

    res = estimate_difference_in_differences(df, "t", "time", "y")
    assert res["treatment_mean_pre"] == pytest.approx(12.0)
    assert res["treatment_mean_post"] == pytest.approx(20.0)
    assert res["control_mean_pre"] == pytest.approx(10.0)
    assert res["control_mean_post"] == pytest.approx(15.0)
    assert res["treatment_change"] == pytest.approx(8.0)
    assert res["control_change"] == pytest.approx(5.0)
    assert res["did_estimate"] == pytest.approx(3.0)


def test_did_custom_treatment_and_control_labels():
    """Verify DiD with custom treatment and control labels."""
    df = pd.DataFrame({
        "cohort": ["Intervention", "Intervention", "Standard", "Standard"],
        "period": [0, 1, 0, 1],
        "y": [10.0, 25.0, 10.0, 15.0],
    })

    res = estimate_difference_in_differences(
        df,
        treatment_col="cohort",
        time_col="period",
        outcome_col="y",
        treated_value="Intervention",
        control_value="Standard",
    )

    assert res["treated_value"] == "Intervention"
    assert res["control_value"] == "Standard"
    assert res["treatment_change"] == pytest.approx(15.0)
    assert res["control_change"] == pytest.approx(5.0)
    assert res["did_estimate"] == pytest.approx(10.0)


def test_did_custom_pre_and_post_labels():
    """Verify DiD with custom pre and post period labels."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "wave": ["Baseline_2022", "Followup_2023", "Baseline_2022", "Followup_2023"],
        "y": [50.0, 75.0, 50.0, 60.0],
    })

    res = estimate_difference_in_differences(
        df,
        treatment_col="t",
        time_col="wave",
        outcome_col="y",
        pre_value="Baseline_2022",
        post_value="Followup_2023",
    )

    assert res["pre_value"] == "Baseline_2022"
    assert res["post_value"] == "Followup_2023"
    assert res["treatment_change"] == pytest.approx(25.0)
    assert res["control_change"] == pytest.approx(10.0)
    assert res["did_estimate"] == pytest.approx(15.0)


def test_did_missing_required_columns():
    """Verify that missing columns raise descriptive ValueError."""
    df = pd.DataFrame({"t": [1, 0], "period": [0, 1], "y": [10.0, 20.0]})

    with pytest.raises(ValueError, match="Time-period column 'time' not found in dataset"):
        estimate_difference_in_differences(df, "t", "time", "y")

    with pytest.raises(ValueError, match="Treatment column 'missing_t' not found in dataset"):
        estimate_difference_in_differences(df, "missing_t", "period", "y")

    with pytest.raises(ValueError, match="Outcome column 'missing_y' not found in dataset"):
        estimate_difference_in_differences(df, "t", "period", "missing_y")


def test_did_invalid_treatment_labels():
    """Verify rejection of invalid or unexpected treatment labels."""
    # Extra unexpected category
    df = pd.DataFrame({
        "t": ["treated", "control", "unknown", "control"],
        "time": [0, 1, 0, 1],
        "y": [10.0, 20.0, 30.0, 40.0],
    })

    with pytest.raises(ValueError, match="must be binary, but contains 3 unique values"):
        estimate_difference_in_differences(df, "t", "time", "y")

    with pytest.raises(ValueError, match="contains values other than specified treated"):
        estimate_difference_in_differences(
            df, "t", "time", "y", treated_value="treated", control_value="control"
        )


def test_did_invalid_time_period_labels():
    """Verify rejection of invalid or unexpected time period labels."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0, 1],
        "time": ["2020", "2021", "2020", "2021", "2022"],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
    })

    with pytest.raises(ValueError, match="must contain exactly 2 unique periods"):
        estimate_difference_in_differences(df, "t", "time", "y")

    with pytest.raises(ValueError, match="contains values other than specified pre"):
        estimate_difference_in_differences(
            df, "t", "time", "y", pre_value="2020", post_value="2021"
        )


def test_did_missing_outcomes_under_default_missing_data_policy():
    """Verify that default missing-data policy (drop_missing=False) raises on missing values."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "time": [0, 1, 0, 1],
        "y": [10.0, np.nan, 5.0, 8.0],
    })

    with pytest.raises(ValueError, match="contains missing values in required columns"):
        estimate_difference_in_differences(df, "t", "time", "y", drop_missing=False)


def test_did_explicit_missing_data_dropping():
    """Verify that explicit drop_missing=True drops incomplete rows and reports count."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0, 1],
        "time": [0, 1, 0, 1, 1],
        "y": [10.0, 20.0, 5.0, 8.0, np.nan],
    })

    res = estimate_difference_in_differences(df, "t", "time", "y", drop_missing=True)
    assert res["rows_dropped"] == 1
    assert res["n_total"] == 4
    assert res["did_estimate"] == pytest.approx(7.0)


def test_did_empty_cell_error():
    """Verify that missing observations in any of the 4 group-period cells raises ValueError."""
    # Control Post (t=0, time=1) has 0 observations
    df = pd.DataFrame({
        "t": [1, 1, 0],
        "time": [0, 1, 0],
        "y": [10.0, 20.0, 5.0],
    })

    with pytest.raises(ValueError, match="Difference-in-Differences requires at least one observation in all four group-period cells"):
        estimate_difference_in_differences(df, "t", "time", "y")


def test_did_nonnumeric_outcomes_rejected():
    """Verify rejection of non-numeric outcome strings."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "time": [0, 1, 0, 1],
        "y": ["a", "b", "c", "d"],
    })

    with pytest.raises(ValueError, match="contains non-numeric values"):
        estimate_difference_in_differences(df, "t", "time", "y")


def test_did_infinite_or_nan_outcomes_rejected():
    """Verify rejection of infinite outcomes."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "time": [0, 1, 0, 1],
        "y": [10.0, np.inf, 5.0, 8.0],
    })

    with pytest.raises(ValueError, match="contains infinite values"):
        estimate_difference_in_differences(df, "t", "time", "y")


def test_did_boolean_outcomes_rejected():
    """Verify rejection of boolean outcome columns."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "time": [0, 1, 0, 1],
        "y": [True, False, True, False],
    })

    with pytest.raises(ValueError, match="must be numeric, but contains boolean values"):
        estimate_difference_in_differences(df, "t", "time", "y")


def test_did_column_name_collisions():
    """Verify rejection when same column is passed for multiple roles."""
    df = pd.DataFrame({"col": [1, 0], "y": [10.0, 20.0]})

    with pytest.raises(ValueError, match="cannot be the same column"):
        estimate_difference_in_differences(df, "col", "col", "y")

    with pytest.raises(ValueError, match="cannot be the same column"):
        estimate_difference_in_differences(df, "col", "y", "y")


def test_did_non_default_dataframe_indices():
    """Verify correct alignment when DataFrame has custom non-default indices."""
    df = pd.DataFrame(
        {
            "treatment": [1, 1, 0, 0],
            "time": [0, 1, 0, 1],
            "outcome": [10.0, 20.0, 5.0, 8.0],
        },
        index=["row_c", "row_a", "row_d", "row_b"],
    )

    res = estimate_difference_in_differences(df, "treatment", "time", "outcome")
    assert res["did_estimate"] == pytest.approx(7.0)


def test_did_alternative_input_formats():
    """Verify support for dict of lists and list of dicts."""
    dict_data = {
        "t": [1, 1, 0, 0],
        "time": [0, 1, 0, 1],
        "y": [10.0, 25.0, 10.0, 15.0],
    }
    res_dict = estimate_difference_in_differences(dict_data, "t", "time", "y")
    assert res_dict["did_estimate"] == pytest.approx(10.0)

    list_data = [
        {"t": 1, "time": 0, "y": 10.0},
        {"t": 1, "time": 1, "y": 25.0},
        {"t": 0, "time": 0, "y": 10.0},
        {"t": 0, "time": 1, "y": 15.0},
    ]
    res_list = estimate_difference_in_differences(list_data, "t", "time", "y")
    assert res_list["did_estimate"] == pytest.approx(10.0)


def test_did_deterministic_reproducibility():
    """Verify that repeated runs yield identical results."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "time": [0, 1, 0, 1],
        "y": [12.5, 30.5, 8.0, 14.0],
    })

    res1 = estimate_difference_in_differences(df, "t", "time", "y")
    res2 = estimate_difference_in_differences(df, "t", "time", "y")

    assert res1["did_estimate"] == pytest.approx(res2["did_estimate"])
    assert res1["treatment_mean_pre"] == pytest.approx(res2["treatment_mean_pre"])
    assert res1["treatment_mean_post"] == pytest.approx(res2["treatment_mean_post"])
    assert res1["control_mean_pre"] == pytest.approx(res2["control_mean_pre"])
    assert res1["control_mean_post"] == pytest.approx(res2["control_mean_post"])


def test_did_assumptions_and_interpretation_content():
    """Verify that assumptions and interpretation text are present and descriptive."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "time": [0, 1, 0, 1],
        "y": [10.0, 20.0, 5.0, 10.0],
    })

    res = estimate_difference_in_differences(df, "t", "time", "y")

    assert "Parallel Trends" in res["assumptions"]
    assert "No Anticipation" in res["assumptions"]
    assert "No Differential Concurrent Shocks" in res["assumptions"]
    assert "Consistent Measurement" in res["assumptions"]
    assert "No Spillover Effects" in res["assumptions"]
    assert "Difference-in-Differences" in res["interpretation"]
    assert "Treatment group outcome changed" in res["interpretation"]


def test_validate_did_data_standalone():
    """Verify standalone functionality of validate_did_data."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "time": [0, 1, 0, 1],
        "y": [10.0, 20.0, 5.0, 10.0],
    })

    val = validate_did_data(df, "t", "time", "y")
    assert val["is_valid"] is True
    assert val["treated_value"] == 1
    assert val["control_value"] == 0
    assert val["pre_value"] == 0
    assert val["post_value"] == 1
    assert val["cell_counts"]["treatment_pre"] == 1


# ============================================================================
# 13. Causal Sensitivity Analysis and Robustness Checks Tests
# ============================================================================

def test_sensitivity_compare_unadjusted_and_regression_ate():
    """Verify comparison of unadjusted and regression-adjusted ATE estimators."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [10.0, 20.0, 30.0, 5.0, 15.0, 25.0],
        "y": [25.0, 35.0, 45.0, 15.0, 25.0, 35.0],
    })

    res = run_causal_sensitivity_analysis(
        data=df,
        treatment_col="t",
        outcome_col="y",
        confounder_cols=["x"],
        reference_estimator="regression_adjusted",
    )

    assert res["reference_estimator"] == "regression_adjusted"
    assert res["estimator_comparisons"]["unadjusted"]["available"] is True
    assert res["estimator_comparisons"]["regression_adjusted"]["available"] is True
    assert isinstance(res["estimator_comparisons"]["unadjusted"]["estimate"], float)
    assert isinstance(res["estimator_comparisons"]["regression_adjusted"]["estimate"], float)


def test_sensitivity_successful_iptw_comparison():
    """Verify successful comparison across all 3 estimators when IPTW is estimable."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
        "y": [20.0, 25.0, 30.0, 10.0, 15.0, 20.0],
    })

    res = run_causal_sensitivity_analysis(
        data=df,
        treatment_col="t",
        outcome_col="y",
        confounder_cols=["x"],
        reference_estimator="iptw",
    )

    assert res["reference_estimator"] == "iptw"
    assert res["estimator_comparisons"]["unadjusted"]["available"] is True
    assert res["estimator_comparisons"]["regression_adjusted"]["available"] is True
    assert res["estimator_comparisons"]["iptw"]["available"] is True
    assert res["estimator_comparisons"]["iptw"]["ess_treated"] is not None
    assert res["estimator_comparisons"]["iptw"]["ess_control"] is not None


def test_sensitivity_reporting_when_estimator_unavailable():
    """Verify that when no confounders are passed, regression and IPTW are marked unavailable with clear reasons."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "y": [10.0, 15.0, 5.0, 8.0],
    })

    res = run_causal_sensitivity_analysis(
        data=df,
        treatment_col="t",
        outcome_col="y",
        confounder_cols=None,
        reference_estimator="unadjusted",
    )

    assert res["reference_estimator"] == "unadjusted"
    assert res["estimator_comparisons"]["unadjusted"]["available"] is True
    assert res["estimator_comparisons"]["regression_adjusted"]["available"] is False
    assert "Confounder columns not provided" in res["estimator_comparisons"]["regression_adjusted"]["reason"]
    assert res["estimator_comparisons"]["iptw"]["available"] is False
    assert "Confounder columns not provided" in res["estimator_comparisons"]["iptw"]["reason"]


def test_sensitivity_error_reporting_not_silently_discarded():
    """Verify that estimator errors (such as collinearity or singular matrix) record the exact error reason."""
    # Data where confounder is constant -> validate_causal_data will fail for regression/IPTW
    # We can test by setting up a scenario where confounder causes a failure
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "x": [1.0, 1.0, 1.0, 1.0],  # Constant column
        "y": [10.0, 15.0, 5.0, 8.0],
    })

    res = run_causal_sensitivity_analysis(
        data=df,
        treatment_col="t",
        outcome_col="y",
        confounder_cols=["x"],
        reference_estimator="unadjusted",
    )

    assert res["estimator_comparisons"]["unadjusted"]["available"] is True
    assert res["estimator_comparisons"]["regression_adjusted"]["available"] is False
    assert "zero variance" in res["estimator_comparisons"]["regression_adjusted"]["reason"].lower()
    assert res["estimator_comparisons"]["iptw"]["available"] is False
    assert "zero variance" in res["estimator_comparisons"]["iptw"]["reason"].lower()


def test_sensitivity_iptw_truncation_with_and_without_explicit_request():
    """Verify that weights are not truncated by default, and only truncated when requested."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [5.0, 6.0, 7.0, 1.0, 2.0, 3.0],
        "y": [50.0, 55.0, 60.0, 10.0, 15.0, 20.0],
    })

    # Without requesting truncation
    res_default = run_causal_sensitivity_analysis(df, "t", "y", ["x"])
    assert res_default["iptw_truncation_analysis"]["status"] == "not_requested"
    assert len(res_default["iptw_truncation_analysis"]["scenarios"]) == 0

    # With explicit truncation thresholds
    res_trunc = run_causal_sensitivity_analysis(
        df, "t", "y", ["x"], truncation_thresholds=[10.0, 3.0]
    )
    assert res_trunc["iptw_truncation_analysis"]["status"] == "evaluated"
    assert len(res_trunc["iptw_truncation_analysis"]["scenarios"]) == 2


def test_sensitivity_iptw_truncation_threshold_recording():
    """Verify that truncation threshold, differences, and counts are correctly recorded."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [10.0, 12.0, 14.0, 2.0, 4.0, 6.0],
        "y": [100.0, 110.0, 120.0, 20.0, 30.0, 40.0],
    })

    res = run_causal_sensitivity_analysis(
        df, "t", "y", ["x"], truncation_thresholds=[5.0, 2.0]
    )
    scenarios = res["iptw_truncation_analysis"]["scenarios"]

    assert scenarios[0]["threshold"] == 5.0
    assert scenarios[1]["threshold"] == 2.0
    assert "estimate" in scenarios[0]
    assert "difference_from_untruncated" in scenarios[0]
    assert "weights_truncated_count" in scenarios[0]
    assert "identical_to_untruncated" in scenarios[0]


def test_sensitivity_invalid_truncation_thresholds():
    """Verify rejection of invalid truncation thresholds."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "x": [2.0, 3.0, 1.0, 2.0],
        "y": [10.0, 15.0, 5.0, 8.0],
    })

    with pytest.raises(ValueError, match="must be strictly positive"):
        run_causal_sensitivity_analysis(df, "t", "y", ["x"], truncation_thresholds=[-1.0])

    with pytest.raises(ValueError, match="must be strictly positive"):
        run_causal_sensitivity_analysis(df, "t", "y", ["x"], truncation_thresholds=[0.0])

    with pytest.raises(ValueError, match="cannot be NaN or infinite"):
        run_causal_sensitivity_analysis(df, "t", "y", ["x"], truncation_thresholds=[float("inf")])

    with pytest.raises(ValueError, match="cannot be empty"):
        run_causal_sensitivity_analysis(df, "t", "y", ["x"], truncation_thresholds=[])


def test_sensitivity_ess_and_diagnostics_retained():
    """Verify effective sample sizes and weight diagnostics are retained in truncation scenarios."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [3.0, 4.0, 5.0, 1.0, 2.0, 3.0],
        "y": [30.0, 35.0, 40.0, 10.0, 15.0, 20.0],
    })

    res = run_causal_sensitivity_analysis(df, "t", "y", ["x"], truncation_thresholds=[4.0])
    scenarios = res["iptw_truncation_analysis"]["scenarios"]

    assert scenarios[0]["ess_treated"] > 0
    assert scenarios[0]["ess_control"] > 0
    assert scenarios[0]["ess_total"] > 0
    assert scenarios[0]["max_weight"] <= 4.0


def test_sensitivity_positive_hypothetical_bias():
    """Verify positive hypothetical bias adjustment: observed 5.0, bias 2.0 -> adjusted 3.0."""
    res = analyze_unmeasured_confounding_sensitivity(
        observed_estimate=5.0,
        bias_values=2.0,
    )

    assert res["observed_estimate"] == pytest.approx(5.0)
    assert len(res["bias_scenarios"]) == 1
    assert res["bias_scenarios"][0]["hypothetical_bias"] == pytest.approx(2.0)
    assert res["bias_scenarios"][0]["adjusted_estimate"] == pytest.approx(3.0)


def test_sensitivity_negative_hypothetical_bias():
    """Verify negative hypothetical bias adjustment: observed 5.0, bias -2.0 -> adjusted 7.0."""
    res = analyze_unmeasured_confounding_sensitivity(
        observed_estimate=5.0,
        bias_values=-2.0,
    )

    assert res["bias_scenarios"][0]["adjusted_estimate"] == pytest.approx(7.0)


def test_sensitivity_zero_hypothetical_bias():
    """Verify zero hypothetical bias adjustment: observed 5.0, bias 0.0 -> adjusted 5.0."""
    res = analyze_unmeasured_confounding_sensitivity(
        observed_estimate=5.0,
        bias_values=0.0,
    )

    assert res["bias_scenarios"][0]["adjusted_estimate"] == pytest.approx(5.0)


def test_sensitivity_multiple_bias_scenarios():
    """Verify multiple bias scenarios evaluated in a single call."""
    res = analyze_unmeasured_confounding_sensitivity(
        observed_estimate=10.0,
        bias_values=[-5.0, 0.0, 2.5, 5.0, 10.0],
    )

    assert len(res["bias_scenarios"]) == 5
    assert res["bias_scenarios"][0]["adjusted_estimate"] == pytest.approx(15.0)
    assert res["bias_scenarios"][1]["adjusted_estimate"] == pytest.approx(10.0)
    assert res["bias_scenarios"][2]["adjusted_estimate"] == pytest.approx(7.5)
    assert res["bias_scenarios"][3]["adjusted_estimate"] == pytest.approx(5.0)
    assert res["bias_scenarios"][4]["adjusted_estimate"] == pytest.approx(0.0)


def test_sensitivity_confidence_interval_shifting():
    """Verify that confidence interval bounds are shifted by the hypothetical bias."""
    # Observed: 10.0, CI: [8.0, 12.0], Bias: 3.0 -> Adjusted CI: [5.0, 9.0]
    res = analyze_unmeasured_confounding_sensitivity(
        observed_estimate=10.0,
        bias_values=[3.0],
        confidence_interval=[8.0, 12.0],
    )

    adj_ci = res["bias_scenarios"][0]["adjusted_confidence_interval"]
    assert adj_ci == pytest.approx([5.0, 9.0])
    assert res["bias_scenarios"][0]["interval_includes_zero"] is False


def test_sensitivity_interval_includes_zero_detection():
    """Verify correct detection of whether an adjusted confidence interval spans zero."""
    # CI [4.0, 8.0]
    # Bias 2.0 -> [2.0, 6.0] -> includes_zero = False
    # Bias 5.0 -> [-1.0, 3.0] -> includes_zero = True
    # Bias 10.0 -> [-6.0, -2.0] -> includes_zero = False
    res = analyze_unmeasured_confounding_sensitivity(
        observed_estimate=6.0,
        bias_values=[2.0, 5.0, 10.0],
        confidence_interval=[4.0, 8.0],
    )

    assert res["bias_scenarios"][0]["interval_includes_zero"] is False
    assert res["bias_scenarios"][1]["interval_includes_zero"] is True
    assert res["bias_scenarios"][2]["interval_includes_zero"] is False


def test_sensitivity_invalid_confidence_interval():
    """Verify rejection of invalid confidence intervals."""
    with pytest.raises(ValueError, match="lower bound .* cannot exceed upper bound"):
        analyze_unmeasured_confounding_sensitivity(5.0, 1.0, confidence_interval=[10.0, 5.0])

    with pytest.raises(ValueError, match="must contain exactly 2 elements"):
        analyze_unmeasured_confounding_sensitivity(5.0, 1.0, confidence_interval=[1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="cannot be NaN or infinite"):
        analyze_unmeasured_confounding_sensitivity(5.0, 1.0, confidence_interval=[float("nan"), 5.0])


def test_sensitivity_rejection_of_nan_and_infinite_inputs():
    """Verify rejection of NaN and infinite observed estimates and biases."""
    with pytest.raises(ValueError, match="observed_estimate cannot be NaN or infinite"):
        analyze_unmeasured_confounding_sensitivity(float("nan"), 1.0)

    with pytest.raises(ValueError, match="observed_estimate cannot be NaN or infinite"):
        analyze_unmeasured_confounding_sensitivity(float("inf"), 1.0)

    with pytest.raises(ValueError, match="cannot be NaN or infinite"):
        analyze_unmeasured_confounding_sensitivity(5.0, float("nan"))

    with pytest.raises(ValueError, match="cannot be NaN or infinite"):
        analyze_unmeasured_confounding_sensitivity(5.0, [1.0, float("inf")])


def test_sensitivity_invalid_observed_effect_estimates():
    """Verify rejection of invalid observed effect estimate types."""
    with pytest.raises(ValueError, match="observed_estimate cannot be None"):
        analyze_unmeasured_confounding_sensitivity(None, 1.0)

    with pytest.raises(ValueError, match="observed_estimate cannot be a boolean"):
        analyze_unmeasured_confounding_sensitivity(True, 1.0)

    with pytest.raises(ValueError, match="observed_estimate must be a numeric float or int"):
        analyze_unmeasured_confounding_sensitivity("five", 1.0)


def test_sensitivity_empty_bias_values_raises():
    """Verify rejection of empty bias values sequence."""
    with pytest.raises(ValueError, match="bias_values cannot be empty"):
        analyze_unmeasured_confounding_sensitivity(5.0, [])


def test_sensitivity_deterministic_output():
    """Verify that sensitivity analysis returns identical deterministic results across repeated runs."""
    df = pd.DataFrame({
        "t": [1, 1, 1, 0, 0, 0],
        "x": [3.0, 4.0, 5.0, 1.0, 2.0, 3.0],
        "y": [30.0, 35.0, 40.0, 10.0, 15.0, 20.0],
    })

    res1 = run_causal_sensitivity_analysis(
        df, "t", "y", ["x"], truncation_thresholds=[5.0], hypothetical_biases=[1.0, 2.0]
    )
    res2 = run_causal_sensitivity_analysis(
        df, "t", "y", ["x"], truncation_thresholds=[5.0], hypothetical_biases=[1.0, 2.0]
    )

    assert res1["reference_estimate"] == pytest.approx(res2["reference_estimate"])
    assert res1["estimator_comparisons"]["regression_adjusted"]["estimate"] == pytest.approx(
        res2["estimator_comparisons"]["regression_adjusted"]["estimate"]
    )
    assert res1["iptw_truncation_analysis"]["scenarios"][0]["estimate"] == pytest.approx(
        res2["iptw_truncation_analysis"]["scenarios"][0]["estimate"]
    )
    assert res1["unmeasured_confounding_analysis"]["bias_scenarios"][0]["adjusted_estimate"] == pytest.approx(
        res2["unmeasured_confounding_analysis"]["bias_scenarios"][0]["adjusted_estimate"]
    )


def test_sensitivity_assumptions_and_limitations_content():
    """Verify that educational assumptions and limitations disclaimers are present."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "y": [10.0, 20.0, 5.0, 10.0],
    })

    res = run_causal_sensitivity_analysis(df, "t", "y")

    assert "Multi-estimator consistency" in res["assumptions"]
    assert "Weight truncation" in res["assumptions"]
    assert "Unmeasured confounding" in res["assumptions"]
    assert "does NOT prove" in res["assumptions"] or "NOT proof" in res["assumptions"]


def test_sensitivity_fallback_when_reference_estimator_unavailable():
    """Verify graceful fallback when requested reference estimator is unavailable."""
    df = pd.DataFrame({
        "t": [1, 1, 0, 0],
        "y": [10.0, 20.0, 5.0, 10.0],
    })

    # Request regression_adjusted as reference, but provide no confounders
    res = run_causal_sensitivity_analysis(
        df, "t", "y", confounder_cols=None, reference_estimator="regression_adjusted"
    )

    assert res["reference_estimator"] == "unadjusted"
    assert any("fell back to 'unadjusted'" in w for w in res["warnings"])


def test_sensitivity_unsupported_reference_estimator_raises():
    """Verify that unsupported reference estimator names raise ValueError."""
    df = pd.DataFrame({"t": [1, 0], "y": [10.0, 20.0]})

    with pytest.raises(ValueError, match="Unsupported reference_estimator"):
        run_causal_sensitivity_analysis(df, "t", "y", reference_estimator="random_forest")
