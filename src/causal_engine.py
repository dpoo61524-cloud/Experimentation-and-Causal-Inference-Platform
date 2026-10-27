"""Causal Inference Engine: Data Validation and Foundations.

This module provides data validation foundations for observational and experimental
causal inference workflows. It verifies that tabular datasets satisfy essential
structural prerequisites before applying causal estimation techniques (e.g.,
regression adjustment, propensity score methods, or difference-in-differences).

Causal Data Model Overview:
---------------------------
1. Treatment Variable ($T$):
   - A binary indicator distinguishing treated units ($T=1$) from untreated/control units ($T=0$).
   - Can be passed as binary integers (0/1), booleans, or custom categorical labels
     (e.g., 'treated'/'control', 'Mens E-Mail'/'No E-Mail').
   - Both treatment groups must have at least one observation ($n_1 > 0$ and $n_0 > 0$).

2. Outcome Variable ($Y$):
   - A numeric, finite variable measuring the outcome of interest (continuous or binary).
   - Non-numeric strings, booleans, infinite values, and unhandled NaNs are rejected.

3. Confounders / Covariates ($X$):
   - Optional vector of observed baseline covariates.
   - May contain numeric variables (continuous, discrete) or categorical variables
     (strings, categories) for adjustment.
   - Constant (zero-variance) columns are rejected because they provide no explanatory
     variation and introduce collinearity into downstream causal models.

4. Explicit Missing Value Policy:
   - Missing data (NaN, None, pd.NA) are never dropped or imputed silently.
   - By default (`drop_missing=False`), presence of any missing values in treatment,
     outcome, or confounder columns raises a descriptive ValueError.
   - When explicitly configured (`drop_missing=True`), rows with missing values across
     the required analysis columns are dropped, and the count of dropped rows is tracked.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _to_dataframe(data: Any) -> pd.DataFrame:
    """Convert input data (DataFrame, dict of lists, list of dicts) to a pandas DataFrame."""
    if data is None:
        raise ValueError("Dataset cannot be None.")

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError(
                "Dataset is empty (0 rows). Causal inference requires observations in both treatment groups."
            )
        return data.copy()

    if isinstance(data, dict):
        if not data:
            raise ValueError("Dataset is empty (empty dictionary).")
        try:
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("Dataset is empty (0 rows).")
            return df
        except Exception as exc:
            raise ValueError(f"Failed to convert dictionary to DataFrame: {exc}")

    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("Dataset is empty (empty sequence).")
        try:
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("Dataset is empty (0 rows).")
            return df
        except Exception as exc:
            raise ValueError(f"Failed to convert records to DataFrame: {exc}")

    raise ValueError(
        f"Unsupported data format: {type(data).__name__}. "
        "Expected pandas DataFrame, dictionary of columns, or list of record dicts."
    )


def validate_causal_data(
    data: Any,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: Optional[Sequence[str]] = None,
    treated_value: Optional[Any] = None,
    untreated_value: Optional[Any] = None,
    drop_missing: bool = False,
) -> Dict[str, Any]:
    """Validate tabular causal inference data and compute preliminary group summaries.

    Parameters:
    -----------
    data : pd.DataFrame, Dict[str, Sequence], or List[Dict[str, Any]]
        The tabular dataset to validate.
    treatment_col : str
        Name of the binary treatment column.
    outcome_col : str
        Name of the numeric outcome column.
    confounder_cols : Sequence[str], optional
        Names of observed confounder/covariate columns to include in the analysis.
    treated_value : Any, optional
        Label identifying treated units. If None, inferred automatically for binary data.
    untreated_value : Any, optional
        Label identifying control/untreated units. If None, inferred automatically for binary data.
    drop_missing : bool, default False
        Explicit missing data policy:
        - If False (default): raises ValueError if any missing values are found in required columns.
        - If True: explicitly drops rows containing missing values in treatment, outcome,
          or confounder columns and records the number of dropped rows.

    Returns:
    --------
    Dict[str, Any]
        A structured dictionary containing:
        - 'is_valid': True
        - 'n_total': Total number of usable observations
        - 'n_treated': Count of observations in the treated group
        - 'n_untreated': Count of observations in the untreated group
        - 'treatment_rate': Proportion of sample receiving treatment (n_treated / n_total)
        - 'treatment_col': Treatment column name
        - 'outcome_col': Outcome column name
        - 'confounder_cols': List of validated confounder column names
        - 'numeric_confounders': List of numeric confounder column names
        - 'categorical_confounders': List of categorical confounder column names
        - 'treated_value': Resolved label for treated group
        - 'untreated_value': Resolved label for untreated group
        - 'outcome_mean_treated': Sample mean of outcome for treated group
        - 'outcome_mean_untreated': Sample mean of outcome for untreated group
        - 'raw_difference': Unadjusted difference in outcome means (treated - untreated)
        - 'rows_dropped': Number of rows dropped due to missing values (0 if drop_missing=False)
        - 'validated_data': Validated DataFrame containing the analysis variables

    Raises:
    -------
    ValueError:
        If columns are missing, data is empty, treatment is not binary, outcome is non-numeric
        or non-finite, confounders are invalid, or missing data violates the policy.
    """
    # 1. Structure and Column Name Checks
    df = _to_dataframe(data)

    if not isinstance(treatment_col, str) or not treatment_col.strip():
        raise ValueError("treatment_col must be a non-empty string.")
    if not isinstance(outcome_col, str) or not outcome_col.strip():
        raise ValueError("outcome_col must be a non-empty string.")

    if treatment_col == outcome_col:
        raise ValueError(
            f"treatment_col and outcome_col cannot be the same column ('{treatment_col}')."
        )

    available_cols = list(df.columns)
    if treatment_col not in df.columns:
        raise ValueError(
            f"Treatment column '{treatment_col}' not found in dataset. "
            f"Available columns: {available_cols}"
        )
    if outcome_col not in df.columns:
        raise ValueError(
            f"Outcome column '{outcome_col}' not found in dataset. "
            f"Available columns: {available_cols}"
        )

    # 2. Confounder Column Name Validation
    clean_confounders: List[str] = []
    if confounder_cols is not None:
        if not isinstance(confounder_cols, (list, tuple)):
            raise ValueError(
                f"confounder_cols must be a list or tuple of column names, got {type(confounder_cols).__name__}."
            )
        for col in confounder_cols:
            if not isinstance(col, str) or not col.strip():
                raise ValueError(
                    f"Confounder column names must be non-empty strings, got {col!r}."
                )
            if col == treatment_col:
                raise ValueError(
                    f"Confounder column '{col}' cannot also be the treatment column."
                )
            if col == outcome_col:
                raise ValueError(
                    f"Confounder column '{col}' cannot also be the outcome column."
                )
            if col not in df.columns:
                raise ValueError(
                    f"Confounder column '{col}' not found in dataset. "
                    f"Available columns: {available_cols}"
                )
            if col not in clean_confounders:
                clean_confounders.append(col)

    required_cols = [treatment_col, outcome_col] + clean_confounders

    # 3. Explicit Missing Values Policy
    missing_series = df[required_cols].isnull().sum()
    columns_with_missing = missing_series[missing_series > 0]

    rows_dropped = 0
    if not columns_with_missing.empty:
        if not drop_missing:
            detail_list = [f"{col}: {cnt}" for col, cnt in columns_with_missing.items()]
            raise ValueError(
                f"Dataset contains missing values in required columns ({', '.join(detail_list)}). "
                "Missing rows are not dropped silently. Set drop_missing=True to explicitly drop incomplete observations."
            )
        else:
            initial_count = len(df)
            df = df.dropna(subset=required_cols).copy()
            rows_dropped = initial_count - len(df)
            if len(df) == 0:
                raise ValueError(
                    f"After dropping rows with missing values, no observations remain (dropped {rows_dropped} rows)."
                )

    # 4. Treatment Column Validation & Binary Label Resolution
    treatment_series = df[treatment_col]
    unique_treatment_values = list(pd.unique(treatment_series))

    if len(unique_treatment_values) == 0:
        raise ValueError(f"Treatment column '{treatment_col}' contains no observations.")
    if len(unique_treatment_values) == 1:
        raise ValueError(
            f"Treatment column '{treatment_col}' contains only 1 unique value ({unique_treatment_values[0]!r}). "
            "Causal inference requires both treated and untreated observations."
        )

    # Check specified vs inferred labels
    if treated_value is not None and untreated_value is not None:
        if treated_value == untreated_value:
            raise ValueError("treated_value and untreated_value cannot be identical.")

        # Ensure no unexpected labels exist beyond treated_value and untreated_value
        is_valid_label = treatment_series.isin([treated_value, untreated_value])
        if not is_valid_label.all():
            unexpected = list(treatment_series[~is_valid_label].unique())
            raise ValueError(
                f"Treatment column '{treatment_col}' contains values other than specified treated "
                f"({treated_value!r}) and untreated ({untreated_value!r}) labels: {unexpected}."
            )
        resolved_treated = treated_value
        resolved_untreated = untreated_value
    elif treated_value is not None or untreated_value is not None:
        raise ValueError(
            "Both treated_value and untreated_value must be provided together, or neither to infer them."
        )
    else:
        # Neither specified: require strictly binary treatment
        if len(unique_treatment_values) > 2:
            raise ValueError(
                f"Treatment column '{treatment_col}' must be binary, but contains {len(unique_treatment_values)} "
                f"unique values: {unique_treatment_values}. "
                "Please specify treated_value and untreated_value or subset the dataset."
            )

        unique_set = set(unique_treatment_values)
        if unique_set == {0, 1}:
            resolved_treated, resolved_untreated = 1, 0
        elif unique_set == {False, True}:
            resolved_treated, resolved_untreated = True, False
        else:
            # Check for standard semantic substrings like 'treat' and 'control'
            s0 = str(unique_treatment_values[0]).lower()
            s1 = str(unique_treatment_values[1]).lower()
            if "treat" in s0 and ("control" in s1 or "untreat" in s1):
                resolved_treated, resolved_untreated = unique_treatment_values[0], unique_treatment_values[1]
            elif "treat" in s1 and ("control" in s0 or "untreat" in s0):
                resolved_treated, resolved_untreated = unique_treatment_values[1], unique_treatment_values[0]
            else:
                # Deterministic fallback: sort string representations
                sorted_vals = sorted(unique_treatment_values, key=lambda v: str(v))
                resolved_untreated, resolved_treated = sorted_vals[0], sorted_vals[1]

    # Verify both groups have at least one observation
    treated_mask = (treatment_series == resolved_treated)
    n_treated = int(treated_mask.sum())
    n_untreated = int((~treated_mask).sum())

    if n_treated == 0:
        raise ValueError(
            f"Treated group ({resolved_treated!r}) contains 0 observations. "
            "Causal inference requires at least one observation in each treatment group."
        )
    if n_untreated == 0:
        raise ValueError(
            f"Untreated group ({resolved_untreated!r}) contains 0 observations. "
            "Causal inference requires at least one observation in each treatment group."
        )

    # 5. Outcome Column Validation
    outcome_series = df[outcome_col]

    # Reject boolean outcomes explicitly (numeric outcomes required)
    if pd.api.types.is_bool_dtype(outcome_series) or any(isinstance(v, bool) for v in outcome_series):
        raise ValueError(
            f"Outcome column '{outcome_col}' must be numeric, but contains boolean values."
        )

    try:
        numeric_outcomes = pd.to_numeric(outcome_series, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Outcome column '{outcome_col}' contains non-numeric values: {exc}"
        )

    if np.isinf(numeric_outcomes).any():
        raise ValueError(
            f"Outcome column '{outcome_col}' contains infinite values (inf or -inf)."
        )
    if np.isnan(numeric_outcomes).any():
        raise ValueError(
            f"Outcome column '{outcome_col}' contains missing/NaN values."
        )

    # 6. Confounders Validation (Usability & Typing)
    numeric_confounders: List[str] = []
    categorical_confounders: List[str] = []

    for col in clean_confounders:
        series = df[col]
        # Check for constant (zero-variance) columns
        if series.nunique() <= 1:
            val = series.iloc[0] if len(series) > 0 else None
            raise ValueError(
                f"Confounder column '{col}' has zero variance (all values are identical: {val!r}). "
                "Constant columns cannot be used as confounders."
            )

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            if np.isinf(series).any():
                raise ValueError(
                    f"Numeric confounder column '{col}' contains infinite values."
                )
            numeric_confounders.append(col)
        else:
            categorical_confounders.append(col)

    # 7. Compute Summary Statistics
    n_total = len(df)
    treatment_rate = float(n_treated / n_total)

    treated_outcomes = numeric_outcomes[treated_mask]
    untreated_outcomes = numeric_outcomes[~treated_mask]

    mean_treated = float(treated_outcomes.mean())
    mean_untreated = float(untreated_outcomes.mean())
    raw_diff = float(mean_treated - mean_untreated)

    return {
        "is_valid": True,
        "n_total": n_total,
        "n_treated": n_treated,
        "n_untreated": n_untreated,
        "treatment_rate": treatment_rate,
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "confounder_cols": clean_confounders,
        "numeric_confounders": numeric_confounders,
        "categorical_confounders": categorical_confounders,
        "treated_value": resolved_treated,
        "untreated_value": resolved_untreated,
        "outcome_mean_treated": mean_treated,
        "outcome_mean_untreated": mean_untreated,
        "raw_difference": raw_diff,
        "rows_dropped": rows_dropped,
        "validated_data": df,
    }
