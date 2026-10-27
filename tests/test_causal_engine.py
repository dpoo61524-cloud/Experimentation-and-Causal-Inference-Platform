"""Unit tests for Causal Inference Data Validation Foundation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import pytest

from causal_engine import validate_causal_data


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
