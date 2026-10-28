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
