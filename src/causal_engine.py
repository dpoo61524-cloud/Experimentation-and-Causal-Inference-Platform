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

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import warnings


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


def _validate_confidence_level(confidence_level: Any) -> float:
    """Validate that confidence_level is a valid numeric probability strictly between 0 and 1.

    Parameters:
    -----------
    confidence_level : Any
        Input confidence level to validate.

    Returns:
    --------
    float
        Validated confidence level as float.

    Raises:
    -------
    ValueError:
        If confidence_level is None, boolean, string, non-numeric, NaN, infinite,
        or not strictly within (0.0, 1.0).
    """
    if confidence_level is None:
        raise ValueError(
            "confidence_level cannot be None. It must be a numeric float strictly between 0 and 1."
        )
    if isinstance(confidence_level, bool):
        raise ValueError(
            "confidence_level cannot be a boolean. It must be a numeric float strictly between 0 and 1."
        )
    if not isinstance(confidence_level, (int, float)):
        raise ValueError(
            f"confidence_level must be a numeric float strictly between 0 and 1, got {type(confidence_level).__name__}."
        )
    if math.isnan(confidence_level) or math.isinf(confidence_level):
        raise ValueError("confidence_level cannot be NaN or infinite.")

    conf = float(confidence_level)
    if not (0.0 < conf < 1.0):
        raise ValueError(
            f"confidence_level must be strictly between 0 and 1 (exclusive), got {confidence_level}."
        )
    return conf


def _calculate_welch_uncertainty(
    treated_outcomes: pd.Series,
    untreated_outcomes: pd.Series,
    confidence_level: float,
    raise_on_zero_variance: bool = False,
) -> Dict[str, Any]:
    """Calculate Welch-Satterthwaite standard error, degrees of freedom, and confidence interval.

    Parameters:
    -----------
    treated_outcomes : pd.Series
        Numeric outcome series for the treated group.
    untreated_outcomes : pd.Series
        Numeric outcome series for the control/untreated group.
    confidence_level : float
        Two-sided confidence level (must be strictly between 0 and 1).
    raise_on_zero_variance : bool, default False
        If True, raises ValueError when both treatment groups have zero sample variance.
        If False, returns None for confidence interval and degrees of freedom with a warning,
        avoiding misleading finite intervals when sampling variance is undefined.

    Returns:
    --------
    Dict[str, Any]
        Dictionary with sample variances, standard error, Welch-Satterthwaite degrees
        of freedom, critical t value, and two-sided confidence interval bounds.

    Raises:
    -------
    ValueError:
        If either group has fewer than 2 observations, confidence_level is invalid,
        or if both groups have zero variance and raise_on_zero_variance is True.
    """
    conf = _validate_confidence_level(confidence_level)

    n_treated = len(treated_outcomes)
    n_control = len(untreated_outcomes)

    if n_treated < 2 or n_control < 2:
        raise ValueError(
            f"Both treated and control groups must contain at least 2 observations to compute "
            f"sample variance and Welch confidence intervals. Got n_treated={n_treated}, n_control={n_control}."
        )

    var_treated = float(np.var(treated_outcomes, ddof=1))
    var_control = float(np.var(untreated_outcomes, ddof=1))

    # Edge Case: Zero variance in both groups
    if var_treated == 0.0 and var_control == 0.0:
        if raise_on_zero_variance:
            raise ValueError(
                "Both treatment groups have zero variance (all outcomes within each group are identical). "
                "Welch degrees of freedom and confidence intervals are undefined."
            )
        # Mathematically justified special case: sampling variance is zero,
        # but t-distribution df is undefined (0/0). Do not return misleading finite CI.
        warning_msg = (
            "Both treatment groups have zero sample variance (all outcomes within each group are identical). "
            "Welch-Satterthwaite degrees of freedom and confidence intervals are undefined."
        )
        res_zero: Dict[str, Any] = {
            "variance_treated": 0.0,
            "variance_control": 0.0,
            "std_error": 0.0,
            "se": 0.0,
            "degrees_of_freedom": None,
            "df": None,
            "confidence_level": conf,
            "t_critical": None,
            "critical_value": None,
            "margin_of_error": None,
            "confidence_interval": None,
            "ci_lower": None,
            "ci_upper": None,
            "warning": warning_msg,
        }
        if abs(conf - 0.95) < 1e-9:
            res_zero["confidence_interval_95"] = None
        return res_zero

    v_treated = var_treated / n_treated
    v_control = var_control / n_control

    # Edge Case: Zero variance in exactly one group
    if var_treated == 0.0:
        df = float(n_control - 1)
        se = float(math.sqrt(v_control))
    elif var_control == 0.0:
        df = float(n_treated - 1)
        se = float(math.sqrt(v_treated))
    else:
        se = float(math.sqrt(v_treated + v_control))
        denom = (v_treated**2 / (n_treated - 1)) + (v_control**2 / (n_control - 1))
        df = float((v_treated + v_control) ** 2 / denom)

    # Student's t critical value for two-sided confidence interval
    alpha = 1.0 - conf
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    margin_of_error = float(t_crit * se)

    mean_treated = float(treated_outcomes.mean())
    mean_control = float(untreated_outcomes.mean())
    estimate = float(mean_treated - mean_control)

    ci_lower = float(estimate - margin_of_error)
    ci_upper = float(estimate + margin_of_error)

    res: Dict[str, Any] = {
        "variance_treated": var_treated,
        "variance_control": var_control,
        "std_error": se,
        "se": se,
        "degrees_of_freedom": df,
        "df": df,
        "confidence_level": conf,
        "t_critical": t_crit,
        "critical_value": t_crit,
        "margin_of_error": margin_of_error,
        "confidence_interval": [ci_lower, ci_upper],
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

    if abs(conf - 0.95) < 1e-9:
        res["confidence_interval_95"] = [ci_lower, ci_upper]

    return res


def estimate_unadjusted_ate(
    data: Any,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: Optional[Sequence[str]] = None,
    treated_value: Optional[Any] = None,
    untreated_value: Optional[Any] = None,
    drop_missing: bool = False,
    confidence_level: float = 0.95,
    compute_ci: bool = True,
    raise_on_zero_variance: bool = False,
) -> Dict[str, Any]:
    """Estimate the unadjusted Average Treatment Effect (ATE) with Welch-style uncertainty.

    Methodology & Formula:
    ----------------------
    1. Point Estimate (Difference in Means):
       The unadjusted ATE estimator computes the difference between the mean outcome of the
       treated group and the mean outcome of the control/untreated group:

           ATE_unadjusted = mean(Y | T = 1) - mean(Y | T = 0)

    2. Welch Standard Error:
       Allowing for unequal population variances across treatment groups, the Welch standard error is:

           SE = sqrt(s_treated^2 / n_treated + s_control^2 / n_control)

       where each sample variance s^2 uses sample-variance denominator (n - 1).

    3. Welch–Satterthwaite Degrees of Freedom:
       Let v_treated = s_treated^2 / n_treated and v_control = s_control^2 / n_control.
       The effective degrees of freedom df is:

           df = (v_treated + v_control)^2 / [v_treated^2 / (n_treated - 1) + v_control^2 / (n_control - 1)]

    4. Two-Sided Confidence Interval:
       Given confidence level (1 - alpha), the critical value t_crit is the (1 - alpha / 2)
       quantile of the Student t distribution with df degrees of freedom:

           CI = estimate +/- t_crit * SE

    Statistical & Causal Interpretation:
    ------------------------------------
    - Sampling Uncertainty: The confidence interval quantifies sampling variability under repeated
      sampling from the same population under the estimator's sampling assumptions.
    - Important Distinction: A 95% confidence interval does NOT mean there is a 95% probability
      that the fixed true effect lies inside this particular realized interval.
    - Causal Confounding Disclaimer: A confidence interval measures statistical sampling precision,
      NOT causal validity. In observational studies or non-randomized settings, treatment assignment
      may be correlated with confounding variables. The confidence interval does NOT eliminate
      confounding bias or prove causality. Adjustment methods (e.g. regression adjustment,
      propensity score weighting/matching) are required for causal identification with observational data.

    Parameters:
    -----------
    data : pd.DataFrame, Dict[str, Sequence], or List[Dict[str, Any]]
        The tabular dataset to analyze.
    treatment_col : str
        Name of the binary treatment column.
    outcome_col : str
        Name of the numeric outcome column.
    confounder_cols : Sequence[str], optional
        Optional names of observed confounder/covariate columns to validate alongside data.
    treated_value : Any, optional
        Label identifying the treated group. If None, inferred automatically for binary data.
    untreated_value : Any, optional
        Label identifying the control group. If None, inferred automatically for binary data.
    drop_missing : bool, default False
        Explicit missing data policy (reused from validate_causal_data):
        - If False (default): raises ValueError if any missing values exist in analysis columns.
        - If True: drops rows containing missing values and records the count.
    confidence_level : float, default 0.95
        Two-sided confidence level for the Welch confidence interval, strictly between 0 and 1.
    compute_ci : bool, default True
        Whether to calculate Welch standard error and confidence interval. When True, both
        groups must contain at least 2 observations and cannot both have zero variance.

    Returns:
    --------
    Dict[str, Any]
        Structured dictionary containing:
        - 'estimator': 'unadjusted_difference_in_means'
        - 'ate': Unadjusted average treatment effect estimate (treated mean - control mean)
        - 'unadjusted_ate': Alias for 'ate'
        - 'difference_in_means': Alias for 'ate'
        - 'mean_outcome_treated': Mean outcome in the treated group
        - 'mean_outcome_control': Mean outcome in the control/untreated group
        - 'outcome_mean_treated': Alias for mean_outcome_treated
        - 'outcome_mean_untreated': Alias for mean_outcome_control
        - 'n_total': Total number of validated observations
        - 'n_treated': Count of treated observations
        - 'n_control': Count of control observations
        - 'n_untreated': Alias for 'n_control'
        - 'treatment_col': Treatment column name
        - 'outcome_col': Outcome column name
        - 'confounder_cols': List of validated confounder column names
        - 'treated_value': Value representing the treated condition
        - 'untreated_value': Value representing the control condition
        - 'control_value': Alias for 'untreated_value'
        - 'rows_dropped': Number of rows dropped due to missing data (0 if drop_missing=False)
        - 'assumptions': Educational explanation of identifying assumptions and confounding limitations
        When compute_ci=True, additionally includes:
        - 'variance_treated': Sample variance of treated group (ddof=1)
        - 'variance_control': Sample variance of control group (ddof=1)
        - 'std_error': Welch standard error of the difference in means
        - 'se': Alias for 'std_error'
        - 'degrees_of_freedom': Welch-Satterthwaite degrees of freedom
        - 'df': Alias for 'degrees_of_freedom'
        - 'confidence_level': Configured confidence level (e.g. 0.95)
        - 't_critical': Two-sided Student t critical value
        - 'critical_value': Alias for 't_critical'
        - 'margin_of_error': Half-width of the confidence interval (t_critical * SE)
        - 'confidence_interval': Two-element list [ci_lower, ci_upper]
        - 'ci_lower': Lower bound of the confidence interval
        - 'ci_upper': Upper bound of the confidence interval
        - 'confidence_interval_95': Two-element list when confidence_level == 0.95
        - 'interpretation': Clear statement clarifying sampling vs causal interpretation

    Raises:
    -------
    ValueError:
        If data validation fails, confidence_level is invalid, fewer than 2 observations
        exist per group when compute_ci=True, or both groups have zero variance.
    """
    # 1. Reuse validate_causal_data for comprehensive prerequisite verification
    v = validate_causal_data(
        data=data,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        confounder_cols=confounder_cols,
        treated_value=treated_value,
        untreated_value=untreated_value,
        drop_missing=drop_missing,
    )

    df_validated = v["validated_data"]
    treatment_series = df_validated[v["treatment_col"]]
    numeric_outcomes = pd.to_numeric(df_validated[v["outcome_col"]])
    treated_mask = treatment_series == v["treated_value"]
    treated_outcomes = numeric_outcomes[treated_mask]
    untreated_outcomes = numeric_outcomes[~treated_mask]

    mean_treated = float(v["outcome_mean_treated"])
    mean_control = float(v["outcome_mean_untreated"])
    ate = float(mean_treated - mean_control)

    assumptions_text = (
        "Unadjusted difference in means reflects the true Average Treatment Effect (ATE) only under "
        "randomized treatment assignment (exchangeability: (Y(1), Y(0)) ⫫ T). In observational data, "
        "confounding bias can distort this estimate; this estimator does not adjust for confounders. "
        "The confidence interval quantifies sampling variability under the estimator's sampling "
        "assumptions. It does not represent a probability that the fixed true effect lies within this "
        "particular interval, nor does it remove confounding or establish causal identification."
    )

    result: Dict[str, Any] = {
        "estimator": "unadjusted_difference_in_means",
        "ate": ate,
        "unadjusted_ate": ate,
        "difference_in_means": ate,
        "mean_outcome_treated": mean_treated,
        "mean_outcome_control": mean_control,
        "outcome_mean_treated": mean_treated,
        "outcome_mean_untreated": mean_control,
        "n_total": int(v["n_total"]),
        "n_treated": int(v["n_treated"]),
        "n_control": int(v["n_untreated"]),
        "n_untreated": int(v["n_untreated"]),
        "treatment_col": v["treatment_col"],
        "outcome_col": v["outcome_col"],
        "confounder_cols": v["confounder_cols"],
        "treated_value": v["treated_value"],
        "control_value": v["untreated_value"],
        "untreated_value": v["untreated_value"],
        "rows_dropped": int(v["rows_dropped"]),
        "assumptions": assumptions_text,
    }

    if compute_ci:
        unc = _calculate_welch_uncertainty(
            treated_outcomes=treated_outcomes,
            untreated_outcomes=untreated_outcomes,
            confidence_level=confidence_level,
            raise_on_zero_variance=raise_on_zero_variance,
        )
        result.update(unc)
        pct = (
            int(unc["confidence_level"] * 100)
            if unc["confidence_level"] * 100 == int(unc["confidence_level"] * 100)
            else unc["confidence_level"] * 100
        )
        if unc["ci_lower"] is not None and unc["ci_upper"] is not None:
            result["interpretation"] = (
                f"The {pct}% confidence interval [{unc['ci_lower']:.4f}, {unc['ci_upper']:.4f}] "
                f"quantifies sampling uncertainty around the difference in means ({ate:.4f}). "
                "Under repeated sampling, 95% of such intervals cover the true difference in means if model "
                "assumptions hold. It does not mean a 95% probability of the true parameter being inside "
                "this realized interval, nor does it remove confounding bias for observational data."
            )
        else:
            result["interpretation"] = (
                f"The difference in means is {ate:.4f}. Because both groups have zero variance, "
                "degrees of freedom and confidence intervals are undefined under the Welch t model."
            )

    return result


def estimate_unadjusted_ate_with_uncertainty(
    data: Any,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: Optional[Sequence[str]] = None,
    treated_value: Optional[Any] = None,
    untreated_value: Optional[Any] = None,
    drop_missing: bool = False,
    confidence_level: float = 0.95,
    raise_on_zero_variance: bool = False,
) -> Dict[str, Any]:
    """Estimate the unadjusted ATE with Welch-style standard error and confidence interval.

    Dedicated alias for estimate_unadjusted_ate(..., compute_ci=True).
    """
    return estimate_unadjusted_ate(
        data=data,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        confounder_cols=confounder_cols,
        treated_value=treated_value,
        untreated_value=untreated_value,
        drop_missing=drop_missing,
        confidence_level=confidence_level,
        compute_ci=True,
        raise_on_zero_variance=raise_on_zero_variance,
    )


def estimate_regression_adjusted_ate(
    data: Any,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: Sequence[str],
    treated_value: Optional[Any] = None,
    untreated_value: Optional[Any] = None,
    drop_missing: bool = False,
) -> Dict[str, Any]:
    """Estimate the Average Treatment Effect (ATE) using regression adjustment (G-computation).

    Methodology & Causal Identification:
    ------------------------------------
    Regression adjustment (also known as the parametric G-formula or standardization)
    estimates the Average Treatment Effect by modeling the conditional expectation of the
    outcome given treatment and observed baseline confounders:

        E[Y | T, X] = f(T, X; beta)

    1. Model Fitting:
       An Ordinary Least Squares (OLS) regression model is fitted on the validated observations:
           Y_i = beta_0 + beta_T * T_i + sum_j beta_j * X_{ij} + epsilon_i
       where categorical confounders are one-hot encoded with the baseline level dropped.

    2. Counterfactual Predictions:
       Using the single fitted regression model, two counterfactual predictions are generated
       for EVERY observed unit i:
       - Y_hat_i(1): predicted outcome setting unit i's treatment indicator to treated (T = 1),
         holding all their observed confounders X_i constant.
       - Y_hat_i(0): predicted outcome setting unit i's treatment indicator to control (T = 0),
         holding all their observed confounders X_i constant.

    3. Marginal Contrast (G-computation):
       The individual counterfactual treatment effects are:
           tau_hat_i = Y_hat_i(1) - Y_hat_i(0)
       The sample Average Treatment Effect (ATE) is the empirical mean of these contrasts:
           ATE_adjusted = (1 / N) * sum_{i=1}^N (Y_hat_i(1) - Y_hat_i(0))
       The contrast is strictly computed as treated minus control.

    Identification Assumptions & Limitations:
    -----------------------------------------
    - Conditional Exchangeability (No Unmeasured Confounding):
      Potential outcomes are independent of treatment assignment given observed confounders:
          (Y(1), Y(0)) ⫫ T | X
      Regression adjustment controls ONLY for the observed, included confounders. Any unmeasured
      or poorly measured confounders will leave residual confounding bias.
    - Positivity (Common Support / Overlap):
      Every subpopulation defined by X must have a non-zero probability of receiving either
      treatment status: 0 < P(T = 1 | X = x) < 1.
    - Model Specification:
      The parametric outcome model is assumed to correctly approximate the true conditional
      expectation function. Severe model misspecification (e.g., omitted non-linearities)
      can introduce bias.
    - SUTVA (Stable Unit Treatment Value Assumption):
      No interference between units (one unit's treatment assignment does not affect another's
      outcome) and no hidden variations of treatment.
    - Note on Constant Effects: This method estimates the population average contrast. While an
      additive linear model produces a uniform contrast across units, G-computation averages
      over the empirical distribution of covariates in the study sample.

    Parameters:
    -----------
    data : pd.DataFrame, Dict[str, Sequence], or List[Dict[str, Any]]
        The tabular dataset to analyze.
    treatment_col : str
        Name of the binary treatment column.
    outcome_col : str
        Name of the numeric outcome column.
    confounder_cols : Sequence[str]
        Names of observed confounder/covariate columns to adjust for. At least one confounder
        column is required.
    treated_value : Any, optional
        Label identifying the treated condition. If None, inferred automatically for binary data.
    untreated_value : Any, optional
        Label identifying the control condition. If None, inferred automatically for binary data.
    drop_missing : bool, default False
        Explicit missing data policy (reused from validate_causal_data):
        - If False (default): raises ValueError if missing values exist in analysis columns.
        - If True: drops rows with missing values and tracks the count.

    Returns:
    --------
    Dict[str, Any]
        Structured dictionary containing:
        - 'estimator': 'regression_adjusted_ate'
        - 'adjusted_ate': Estimated regression-adjusted ATE (treated minus control)
        - 'ate': Alias for 'adjusted_ate'
        - 'unadjusted_ate': Unadjusted difference in sample means (treated minus control)
        - 'raw_difference': Alias for 'unadjusted_ate'
        - 'difference_in_means': Alias for 'unadjusted_ate'
        - 'mean_outcome_treated': Observed sample mean outcome in the treated group
        - 'mean_outcome_control': Observed sample mean outcome in the control group
        - 'outcome_mean_treated': Alias for 'mean_outcome_treated'
        - 'outcome_mean_untreated': Alias for 'mean_outcome_control'
        - 'mean_predicted_treated': Average counterfactual predicted outcome under treatment
        - 'mean_predicted_control': Average counterfactual predicted outcome under control
        - 'n_total': Total number of validated observations used in model fitting
        - 'n_treated': Count of treated observations
        - 'n_control': Count of control observations
        - 'n_untreated': Alias for 'n_control'
        - 'treatment_col': Treatment column name
        - 'outcome_col': Outcome column name
        - 'confounder_cols': List of validated confounder column names
        - 'numeric_confounders': List of numeric confounder column names
        - 'categorical_confounders': List of categorical confounder column names
        - 'model_features': Feature column names included in the design matrix
        - 'treated_value': Value representing the treated condition
        - 'control_value': Value representing the control condition
        - 'untreated_value': Alias for 'control_value'
        - 'rows_dropped': Number of rows dropped due to missing data (0 if drop_missing=False)
        - 'r_squared': Coefficient of determination ($R^2$) of the fitted outcome model
        - 'interpretation': Concise interpretation of the adjusted estimate vs raw difference
        - 'assumptions': Educational explanation of identifying assumptions and confounding limitations

    Raises:
    -------
    ValueError:
        If no confounders are provided, or if data validation fails (missing columns, non-binary
        treatment, non-numeric outcome, constant confounders, missing data policy violation,
        or insufficient observations for regression model fitting).
    """
    # 1. Require at least one confounder column
    if confounder_cols is None:
        raise ValueError(
            "Regression-adjusted ATE estimation requires at least one confounder column. "
            "If no confounders are available, use estimate_unadjusted_ate(...) instead."
        )
    if not isinstance(confounder_cols, (list, tuple)):
        raise ValueError(
            f"confounder_cols must be a list or tuple of column names, got {type(confounder_cols).__name__}."
        )
    if len(confounder_cols) == 0:
        raise ValueError(
            "Regression-adjusted ATE estimation requires at least one confounder column. "
            "If no confounders are available, use estimate_unadjusted_ate(...) instead."
        )

    # 2. Reuse validate_causal_data for comprehensive validation
    v = validate_causal_data(
        data=data,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        confounder_cols=confounder_cols,
        treated_value=treated_value,
        untreated_value=untreated_value,
        drop_missing=drop_missing,
    )

    df_validated = v["validated_data"]
    treatment_series = df_validated[v["treatment_col"]]
    numeric_outcomes = pd.to_numeric(df_validated[v["outcome_col"]], errors="raise").astype(float)

    treated_label = v["treated_value"]
    control_label = v["untreated_value"]
    t_binary = (treatment_series == treated_label).astype(float)

    clean_confounders = v["confounder_cols"]
    numeric_confounders = v["numeric_confounders"]
    categorical_confounders = v["categorical_confounders"]

    confounders_df = df_validated[clean_confounders].copy()

    # 3. Preprocess Confounders (One-Hot Encoding for Categoricals with drop_first=True)
    if categorical_confounders:
        X_conf = pd.get_dummies(
            confounders_df,
            columns=categorical_confounders,
            drop_first=True,
            dtype=float,
        )
    else:
        X_conf = confounders_df.astype(float)

    # 4. Construct Design Matrix with Treatment Indicator and Intercept
    X_design = X_conf.copy()
    X_design.insert(0, "treatment_indicator", t_binary)
    X_design = sm.add_constant(X_design, prepend=True, has_constant="add")

    n_obs = len(df_validated)
    n_features = X_design.shape[1]

    if n_obs < n_features:
        raise ValueError(
            f"Insufficient observations to fit regression adjustment model: "
            f"dataset contains {n_obs} observation(s), but model requires at least {n_features} "
            f"observations for {n_features} parameters (including intercept, treatment, and confounders)."
        )

    # 5. Fit OLS Regression Model
    try:
        model = sm.OLS(numeric_outcomes, X_design).fit()
    except Exception as exc:
        raise ValueError(f"Failed to fit regression adjustment model: {exc}")

    if np.isnan(model.params).any():
        raise ValueError(
            "Regression model fitting resulted in NaN coefficients (likely due to severe collinearity or rank deficiency)."
        )

    # 6. Counterfactual Predictions (G-computation)
    # Counterfactual 1: Every unit set to treated condition (T = 1.0)
    X_treated = X_design.copy()
    X_treated["treatment_indicator"] = 1.0
    pred_treated = model.predict(X_treated)

    # Counterfactual 0: Every unit set to control condition (T = 0.0)
    X_control = X_design.copy()
    X_control["treatment_indicator"] = 0.0
    pred_control = model.predict(X_control)

    if (
        np.isnan(pred_treated).any()
        or np.isnan(pred_control).any()
        or np.isinf(pred_treated).any()
        or np.isinf(pred_control).any()
    ):
        raise ValueError("Regression model generated non-finite counterfactual predictions.")

    # 7. Compute Individual Contrasts and Adjusted ATE
    individual_contrasts = pred_treated - pred_control
    adjusted_ate = float(np.mean(individual_contrasts))
    mean_pred_treated = float(np.mean(pred_treated))
    mean_pred_control = float(np.mean(pred_control))

    raw_diff = float(v["raw_difference"])
    r_squared = (
        float(model.rsquared)
        if hasattr(model, "rsquared") and model.rsquared is not None
        else None
    )

    assumptions_text = (
        "Regression adjustment (G-computation) identifies the Average Treatment Effect (ATE) under "
        "four primary identification assumptions: (1) Conditional Exchangeability / No Unmeasured Confounding "
        "(potential outcomes are conditionally independent of treatment given observed confounders: (Y(1), Y(0)) ⫫ T | X); "
        "(2) Positivity / Overlap (every subpopulation has a non-zero probability of receiving both treatment conditions: "
        "0 < P(T=1 | X=x) < 1); (3) Correct Model Specification (the linear outcome model accurately reflects the true "
        "conditional expectation function E[Y | T, X]); and (4) SUTVA (no interference between units and well-defined "
        "treatment versions). Important: Regression adjustment controls ONLY for observed, modeled covariates. "
        "It does NOT guarantee causality if unmeasured confounding, severe model misspecification, or lack of overlap exists."
    )

    interpretation_text = (
        f"The regression-adjusted ATE is {adjusted_ate:.4f} (unadjusted difference in means: {raw_diff:.4f}). "
        "This reflects the average counterfactual contrast: the expected difference in outcome if the entire sample "
        "were assigned to the treated condition versus the control condition, adjusting for observed confounders "
        f"({', '.join(clean_confounders)}). This estimator controls for observed covariates under a linear model; "
        "it does not remove unmeasured confounding or prove causality in observational settings."
    )

    return {
        "estimator": "regression_adjusted_ate",
        "adjusted_ate": adjusted_ate,
        "ate": adjusted_ate,
        "unadjusted_ate": raw_diff,
        "raw_difference": raw_diff,
        "difference_in_means": raw_diff,
        "mean_outcome_treated": float(v["outcome_mean_treated"]),
        "mean_outcome_control": float(v["outcome_mean_untreated"]),
        "outcome_mean_treated": float(v["outcome_mean_treated"]),
        "outcome_mean_untreated": float(v["outcome_mean_untreated"]),
        "mean_predicted_treated": mean_pred_treated,
        "mean_predicted_control": mean_pred_control,
        "n_total": int(v["n_total"]),
        "n_treated": int(v["n_treated"]),
        "n_control": int(v["n_untreated"]),
        "n_untreated": int(v["n_untreated"]),
        "treatment_col": v["treatment_col"],
        "outcome_col": v["outcome_col"],
        "confounder_cols": clean_confounders,
        "numeric_confounders": numeric_confounders,
        "categorical_confounders": categorical_confounders,
        "model_features": list(X_design.columns),
        "treated_value": treated_label,
        "control_value": control_label,
        "untreated_value": control_label,
        "rows_dropped": int(v["rows_dropped"]),
        "r_squared": r_squared,
        "interpretation": interpretation_text,
        "assumptions": assumptions_text,
    }


def compute_overlap_diagnostics(
    propensity_scores: Sequence[float],
    treatment: Sequence[Any],
    treated_value: Any = 1,
    control_value: Any = 0,
    near_zero_threshold: float = 0.05,
    near_one_threshold: float = 0.95,
) -> Dict[str, Any]:
    """Compute overlap and common support diagnostics for propensity scores.

    Overlap (also known as the positivity assumption) requires that every individual in the
    target population has a non-zero probability of receiving either treatment condition:
        0 < P(Treatment = treated | X) < 1

    When treatment assignment is near-deterministic (propensity scores near 0 or 1), or when
    treated and control score ranges do not overlap, estimating counterfactual outcomes
    requires severe model extrapolation. This diagnostic evaluates the empirical score distributions
    across treatment groups to identify potential positivity violations.

    Parameters:
    -----------
    propensity_scores : Sequence[float]
        Sequence of estimated propensity scores P(Treatment = treated | X).
    treatment : Sequence[Any]
        Sequence of treatment group labels matching each propensity score.
    treated_value : Any, default 1
        Label identifying treated observations.
    control_value : Any, default 0
        Label identifying control observations.
    near_zero_threshold : float, default 0.05
        Threshold below which a propensity score is considered near zero.
    near_one_threshold : float, default 0.95
        Threshold above which a propensity score is considered near one.

    Returns:
    --------
    Dict[str, Any]
        Dictionary of overlap diagnostics containing:
        - 'min_score': Global minimum propensity score.
        - 'max_score': Global maximum propensity score.
        - 'min_score_treated': Minimum score among treated units.
        - 'max_score_treated': Maximum score among treated units.
        - 'min_score_control': Minimum score among control units.
        - 'max_score_control': Maximum score among control units.
        - 'treated_range': [min_score_treated, max_score_treated].
        - 'control_range': [min_score_control, max_score_control].
        - 'common_support_range': [common_support_min, common_support_max] if overlap exists, else None.
        - 'has_overlap': True if treated and control score ranges intersect, False otherwise.
        - 'scores_near_zero': Total count of observations with score < near_zero_threshold.
        - 'scores_near_one': Total count of observations with score > near_one_threshold.
        - 'treated_near_zero': Count of treated observations with score < near_zero_threshold.
        - 'treated_near_one': Count of treated observations with score > near_one_threshold.
        - 'control_near_zero': Count of control observations with score < near_zero_threshold.
        - 'control_near_one': Count of control observations with score > near_one_threshold.
        - 'near_zero_threshold': The threshold used for near-zero detection.
        - 'near_one_threshold': The threshold used for near-one detection.
        - 'warning': Explanatory warning string if overlap is violated or extreme scores are present, else None.
        - 'interpretation': Concise interpretation of the empirical overlap assessment.
    """
    if propensity_scores is None:
        raise ValueError("propensity_scores cannot be None.")
    if treatment is None:
        raise ValueError("treatment cannot be None.")

    try:
        scores_arr = np.asarray(propensity_scores, dtype=float)
    except Exception as exc:
        raise ValueError(f"propensity_scores must contain numeric values: {exc}")

    if scores_arr.ndim != 1 or len(scores_arr) == 0:
        raise ValueError("propensity_scores must be a non-empty 1D sequence.")

    if np.isnan(scores_arr).any() or np.isinf(scores_arr).any():
        raise ValueError("propensity_scores contains NaN or infinite values.")

    if (scores_arr < 0.0).any() or (scores_arr > 1.0).any():
        raise ValueError("propensity_scores must lie in the valid probability interval [0, 1].")

    t_arr = np.asarray(treatment)
    if len(t_arr) != len(scores_arr):
        raise ValueError(
            f"Length mismatch: propensity_scores ({len(scores_arr)}) and treatment ({len(t_arr)}) "
            "must have identical lengths."
        )

    treated_mask = (t_arr == treated_value)
    control_mask = (t_arr == control_value)

    n_treated = int(np.sum(treated_mask))
    n_control = int(np.sum(control_mask))

    if n_treated == 0:
        raise ValueError(
            f"No observations match treated_value={treated_value!r} in treatment array."
        )
    if n_control == 0:
        raise ValueError(
            f"No observations match control_value={control_value!r} in treatment array."
        )

    treated_scores = scores_arr[treated_mask]
    control_scores = scores_arr[control_mask]

    min_score = float(np.min(scores_arr))
    max_score = float(np.max(scores_arr))

    min_score_treated = float(np.min(treated_scores))
    max_score_treated = float(np.max(treated_scores))
    min_score_control = float(np.min(control_scores))
    max_score_control = float(np.max(control_scores))

    treated_range = [min_score_treated, max_score_treated]
    control_range = [min_score_control, max_score_control]

    # Check common support intersection
    common_support_min = max(min_score_treated, min_score_control)
    common_support_max = min(max_score_treated, max_score_control)
    has_overlap = bool(common_support_min <= common_support_max)
    common_support_range = (
        [float(common_support_min), float(common_support_max)] if has_overlap else None
    )

    # Near-boundary counts
    near_zero = float(near_zero_threshold)
    near_one = float(near_one_threshold)

    scores_near_zero = int(np.sum(scores_arr < near_zero))
    scores_near_one = int(np.sum(scores_arr > near_one))
    treated_near_zero = int(np.sum(treated_scores < near_zero))
    treated_near_one = int(np.sum(treated_scores > near_one))
    control_near_zero = int(np.sum(control_scores < near_zero))
    control_near_one = int(np.sum(control_scores > near_one))

    # Overlap warning formulation
    warning: Optional[str] = None
    if not has_overlap:
        warning = (
            "Severe overlap violation: Treated and control propensity score ranges are completely disjoint. "
            f"Treated range is [{min_score_treated:.4f}, {max_score_treated:.4f}] while control range is "
            f"[{min_score_control:.4f}, {max_score_control:.4f}]. Common support does not exist, so matching "
            "or weighting cannot be performed without extreme parametric extrapolation."
        )
    elif scores_near_zero > 0 or scores_near_one > 0:
        warning = (
            f"Limited overlap warning: {scores_near_zero} score(s) are near zero (< {near_zero}) and "
            f"{scores_near_one} score(s) are near one (> {near_one}). Near-deterministic treatment assignment "
            "indicates poor positivity, which can cause extreme weights or unmatchable units in downstream analyses."
        )

    interpretation = (
        f"Overlap diagnostic: {'Overlap exists' if has_overlap else 'No common support'}. "
        f"Treated score range: [{min_score_treated:.4f}, {max_score_treated:.4f}], "
        f"Control score range: [{min_score_control:.4f}, {max_score_control:.4f}]. "
        f"Scores near boundaries: {scores_near_zero} near 0 (< {near_zero}), "
        f"{scores_near_one} near 1 (> {near_one})."
    )

    return {
        "min_score": min_score,
        "max_score": max_score,
        "min_score_treated": min_score_treated,
        "max_score_treated": max_score_treated,
        "min_score_control": min_score_control,
        "max_score_control": max_score_control,
        "treated_range": treated_range,
        "control_range": control_range,
        "common_support_range": common_support_range,
        "has_overlap": has_overlap,
        "scores_near_zero": scores_near_zero,
        "scores_near_one": scores_near_one,
        "treated_near_zero": treated_near_zero,
        "treated_near_one": treated_near_one,
        "control_near_zero": control_near_zero,
        "control_near_one": control_near_one,
        "near_zero_threshold": near_zero,
        "near_one_threshold": near_one,
        "warning": warning,
        "interpretation": interpretation,
    }


def compute_unadjusted_covariate_balance(
    data: pd.DataFrame,
    treatment_col: str,
    confounder_cols: Sequence[str],
    treated_value: Any = 1,
    control_value: Any = 0,
) -> Dict[str, Dict[str, Any]]:
    """Compute unadjusted Standardized Mean Differences (SMDs) for observed confounders.

    Standardized Mean Difference (SMD) quantifies the magnitude of baseline covariate imbalance
    between treated and control groups in units of pooled standard deviation:
        SMD = (mean_treated - mean_control) / pooled_standard_deviation

    where the pooled sample standard deviation is:
        s_pooled = sqrt(((n_1 - 1)*s_1^2 + (n_0 - 1)*s_0^2) / (n_1 + n_0 - 2))

    By convention in causal inference literature (e.g. Austin 2009), an absolute SMD < 0.10
    is commonly considered indicative of acceptable balance, whereas larger values indicate
    substantial baseline confounding.

    Note: This function evaluates UNADJUSTED balance only. Post-adjustment balance (e.g. after
    matching or inverse-probability weighting) will be evaluated in subsequent tasks.

    Parameters:
    -----------
    data : pd.DataFrame
        Tabular dataset containing treatment and confounder columns.
    treatment_col : str
        Name of the binary treatment column.
    confounder_cols : Sequence[str]
        List of observed confounder column names to evaluate.
    treated_value : Any, default 1
        Label identifying treated observations.
    control_value : Any, default 0
        Label identifying control observations.

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary keyed by confounder column name, with balance statistics for each variable:
        - 'covariate': Column name
        - 'covariate_type': 'numeric' or 'categorical'
        - 'mean_treated': Sample mean of treated group (numeric only)
        - 'mean_control': Sample mean of control group (numeric only)
        - 'std_treated': Sample standard deviation of treated group (numeric only)
        - 'std_control': Sample standard deviation of control group (numeric only)
        - 'pooled_std': Pooled sample standard deviation (numeric only)
        - 'smd': Standardized Mean Difference (float or None)
        - 'abs_smd': Absolute Standardized Mean Difference (float or None)
        - 'zero_pooled_variance': Boolean indicating whether pooled variance is zero
        - 'is_balanced': True if abs(smd) <= 0.10, False otherwise (None for categorical/undefined)
        - 'note': Educational or diagnostic note (e.g. for zero pooled variance or categorical variables)
    """
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    if treatment_col not in data.columns:
        raise ValueError(f"treatment_col '{treatment_col}' not found in data.")

    t_series = data[treatment_col]
    treated_mask = (t_series == treated_value)
    control_mask = (t_series == control_value)

    n_1 = int(np.sum(treated_mask))
    n_0 = int(np.sum(control_mask))

    if n_1 == 0 or n_0 == 0:
        raise ValueError("Data must contain both treated and control observations for balance calculation.")

    balance_dict: Dict[str, Dict[str, Any]] = {}

    for col in confounder_cols:
        if col not in data.columns:
            raise ValueError(f"Confounder column '{col}' not found in data.")

        series = data[col]
        # Check if column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

        if not is_numeric:
            balance_dict[col] = {
                "covariate": col,
                "covariate_type": "categorical",
                "mean_treated": None,
                "mean_control": None,
                "std_treated": None,
                "std_control": None,
                "pooled_std": None,
                "smd": None,
                "abs_smd": None,
                "zero_pooled_variance": False,
                "is_balanced": None,
                "note": (
                    "Standardized mean difference is designed for numeric covariates. "
                    "Categorical covariate balance is deferred to the subsequent matching/weighting module."
                ),
            }
            continue

        x1 = series[treated_mask].astype(float).values
        x0 = series[control_mask].astype(float).values

        m_1 = float(np.mean(x1))
        m_0 = float(np.mean(x0))

        v_1 = float(np.var(x1, ddof=1)) if len(x1) > 1 else 0.0
        v_0 = float(np.var(x0, ddof=1)) if len(x0) > 1 else 0.0

        s_1 = float(np.sqrt(v_1))
        s_0 = float(np.sqrt(v_0))

        df_pooled = (len(x1) - 1) + (len(x0) - 1)
        if df_pooled > 0:
            pooled_var = ((len(x1) - 1) * v_1 + (len(x0) - 1) * v_0) / df_pooled
            pooled_std = float(np.sqrt(max(0.0, pooled_var)))
        else:
            pooled_std = 0.0

        if pooled_std > 0.0:
            smd = float((m_1 - m_0) / pooled_std)
            abs_smd = float(abs(smd))
            zero_pooled_var = False
            is_balanced = bool(abs_smd <= 0.10)
            note = None
        else:
            zero_pooled_var = True
            if m_1 == m_0:
                smd = 0.0
                abs_smd = 0.0
                is_balanced = True
                note = (
                    "Zero pooled variance: covariate values are identical across both treatment groups; "
                    "standardized mean difference is 0.0."
                )
            else:
                smd = None
                abs_smd = None
                is_balanced = False
                note = (
                    "Zero pooled variance with differing group means: within-group variation is zero, "
                    "so standardized mean difference is mathematically undefined."
                )

        balance_dict[col] = {
            "covariate": col,
            "covariate_type": "numeric",
            "mean_treated": m_1,
            "mean_control": m_0,
            "std_treated": s_1,
            "std_control": s_0,
            "pooled_std": pooled_std,
            "smd": smd,
            "abs_smd": abs_smd,
            "zero_pooled_variance": zero_pooled_var,
            "is_balanced": is_balanced,
            "note": note,
        }

    return balance_dict


def estimate_propensity_scores(
    data: Any,
    treatment_col: str,
    confounder_cols: Optional[Sequence[str]] = None,
    outcome_col: Optional[str] = None,
    treated_value: Optional[Any] = None,
    untreated_value: Optional[Any] = None,
    control_value: Optional[Any] = None,
    drop_missing: bool = False,
    near_zero_threshold: float = 0.05,
    near_one_threshold: float = 0.95,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Estimate propensity scores and compute overlap and balance diagnostics.

    The propensity score is defined as the conditional probability of receiving treatment
    given observed baseline confounders:
        e(X) = P(Treatment = treated | X)

    In observational studies, individuals who receive treatment typically differ systematically
    from those who do not (treatment selection bias). By summarizing multi-dimensional observed
    confounders into a single scalar probability, propensity scores facilitate assessing common
    support (positivity) and covariate balance before estimating treatment effects.

    Methodology:
    ------------
    1. Validates tabular data, binary treatment labels, and non-empty confounder lists
       by reusing the core validation foundation (`validate_causal_data`).
    2. Encodes numeric confounders as continuous floats and categorical confounders via
       one-hot encoding (`drop_first=True`) to maintain a consistent design matrix.
    3. Fits a binary logistic regression model (`statsmodels.api.Logit`) to estimate
       conditional treatment assignment probabilities.
    4. Evaluates empirical common support and overlap diagnostics across treatment groups,
       identifying near-deterministic assignment (scores near 0 or 1).
    5. Calculates baseline unadjusted Standardized Mean Differences (SMDs) for numeric
       confounders to establish pre-adjustment imbalance.

    Important Causal Limitations:
    -----------------------------
    - Propensity scores are ESTIMATED probabilities, not known true assignment mechanisms.
    - Propensity score estimation ALONE does NOT estimate a treatment effect. Treatment effect
      estimation requires subsequent methods such as matching, weighting, or stratification.
    - Propensity score methods adjust ONLY for observed, modeled confounders. They CANNOT
      correct for unmeasured confounding.
    - If overlap is poor or absent (lack of common support), treatment effect estimation
      relies heavily on parametric extrapolation and may produce unstable results.
    - Scores are NEVER silently clipped or modified in this function.

    Parameters:
    -----------
    data : Any
        Tabular dataset (pandas DataFrame, dictionary of columns, or list of dicts).
    treatment_col : str
        Name of the binary treatment column.
    confounder_cols : Sequence[str], optional
        List of observed confounder column names used to model treatment assignment.
        Must contain at least one confounder column.
    outcome_col : str, optional
        Optional outcome column name. If provided, outcome validity is verified according
        to existing validator conventions.
    treated_value : Any, optional
        Label identifying treated observations. If None, inferred automatically.
    untreated_value : Any, optional
        Label identifying control observations. If None, inferred automatically.
    control_value : Any, optional
        Alias for untreated_value.
    drop_missing : bool, default False
        Explicit missing data policy:
        - False (default): raises ValueError if any missing values are found.
        - True: explicitly drops rows containing missing values and records rows_dropped.
    near_zero_threshold : float, default 0.05
        Threshold below which a score is flagged as near-zero.
    near_one_threshold : float, default 0.95
        Threshold above which a score is flagged as near-one.

    Returns:
    --------
    Dict[str, Any]
        Structured dictionary containing:
        - 'estimator': 'logistic_propensity_score'
        - 'propensity_scores': List of estimated probabilities P(T=1 | X) for each validated observation
        - 'scores': Alias for 'propensity_scores'
        - 'observation_indices': List of original dataset row indices corresponding to each score
        - 'scores_by_index': Dict mapping original row index to estimated propensity score
        - 'n_total': Total number of validated observations
        - 'n_treated': Number of treated observations
        - 'n_control': Number of control observations
        - 'n_untreated': Alias for 'n_control'
        - 'treatment_col': Treatment column name
        - 'confounder_cols': List of confounder column names
        - 'numeric_confounders': List of numeric confounder column names
        - 'categorical_confounders': List of categorical confounder column names
        - 'treated_value': Value representing the treated condition
        - 'control_value': Value representing the control condition
        - 'untreated_value': Alias for 'control_value'
        - 'rows_dropped': Number of rows dropped due to missing data (0 if drop_missing=False)
        - 'model_features': List of features included in the logistic regression design matrix
        - 'model_coefficients': Dict mapping feature names to estimated logistic regression coefficients
        - 'model_metadata': Metadata including convergence status, pseudo-R2, and optimization warnings
        - 'overlap_diagnostics': Comprehensive overlap and common support diagnostic dictionary
        - 'unadjusted_covariate_balance': Dictionary of baseline SMDs for confounders
        - 'covariate_balance': Alias for 'unadjusted_covariate_balance'
        - 'balance_assessment_type': 'unadjusted'
        - 'post_adjustment_balance_note': Explanation that post-adjustment balance is evaluated in subsequent tasks
        - 'categorical_balance_note': Statement regarding categorical confounder balance
        - 'interpretation': Concise interpretation of score distribution and overlap
        - 'assumptions': Educational statement on causal assumptions, positivity, and limitations

    Raises:
    -------
    ValueError:
        If confounders are missing or empty, data validation fails, treatment is non-binary,
        sample size is insufficient for the number of model parameters, or the logistic regression
        model cannot be fit.
    """
    # 1. Flexible argument resolution
    actual_outcome_col = outcome_col
    actual_confounder_cols = confounder_cols

    if isinstance(confounder_cols, str) and isinstance(outcome_col, (list, tuple)):
        actual_outcome_col = confounder_cols
        actual_confounder_cols = outcome_col

    if control_value is not None and untreated_value is None:
        untreated_value = control_value
    if treated_value is None and "treatment_value" in kwargs:
        treated_value = kwargs["treatment_value"]

    # 2. Confounder list validation
    if actual_confounder_cols is None:
        raise ValueError(
            "Propensity score estimation requires at least one confounder column. "
            "Confounder list cannot be None."
        )
    if not isinstance(actual_confounder_cols, (list, tuple)):
        raise ValueError(
            f"confounder_cols must be a list or tuple of column names, got {type(actual_confounder_cols).__name__}."
        )
    if len(actual_confounder_cols) == 0:
        raise ValueError(
            "Propensity score estimation requires at least one confounder column. "
            "Confounder list cannot be empty."
        )

    # 3. Reuse validate_causal_data
    if actual_outcome_col is not None:
        v = validate_causal_data(
            data=data,
            treatment_col=treatment_col,
            outcome_col=actual_outcome_col,
            confounder_cols=actual_confounder_cols,
            treated_value=treated_value,
            untreated_value=untreated_value,
            drop_missing=drop_missing,
        )
    else:
        df_raw = _to_dataframe(data)
        dummy_col = "__dummy_outcome__"
        while dummy_col in df_raw.columns:
            dummy_col += "_"
        df_with_dummy = df_raw.copy()
        df_with_dummy[dummy_col] = 1.0

        v = validate_causal_data(
            data=df_with_dummy,
            treatment_col=treatment_col,
            outcome_col=dummy_col,
            confounder_cols=actual_confounder_cols,
            treated_value=treated_value,
            untreated_value=untreated_value,
            drop_missing=drop_missing,
        )

    df_validated = v["validated_data"]
    if actual_outcome_col is None and dummy_col in df_validated.columns:
        df_validated = df_validated.drop(columns=[dummy_col])

    treated_label = v["treated_value"]
    control_label = v["untreated_value"]
    treatment_series = df_validated[v["treatment_col"]]
    t_binary = (treatment_series == treated_label).astype(float)

    clean_confounders = v["confounder_cols"]
    numeric_confounders = v["numeric_confounders"]
    categorical_confounders = v["categorical_confounders"]

    # 4. Preprocessing confounders (One-Hot Encoding for Categoricals with drop_first=True)
    confounders_df = df_validated[clean_confounders].copy()
    if categorical_confounders:
        X_conf = pd.get_dummies(
            confounders_df,
            columns=categorical_confounders,
            drop_first=True,
            dtype=float,
        )
    else:
        X_conf = confounders_df.astype(float)

    X_design = sm.add_constant(X_conf, prepend=True, has_constant="add")

    n_obs = len(df_validated)
    n_features = X_design.shape[1]

    if n_obs < n_features:
        raise ValueError(
            f"Insufficient observations to fit propensity score model: "
            f"dataset contains {n_obs} observation(s), but model requires at least {n_features} "
            f"observations for {n_features} parameters (including intercept and confounders)."
        )

    n_t = int(np.sum(t_binary == 1.0))
    n_c = int(np.sum(t_binary == 0.0))
    if n_t == 0 or n_c == 0:
        raise ValueError(
            "Propensity score estimation requires observations in both treatment groups "
            f"(got {n_t} treated, {n_c} control)."
        )

    # 5. Fit Logistic Regression Model
    warning_msgs: List[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            logit_model = sm.Logit(t_binary, X_design)
            logit_res = logit_model.fit(disp=False)
            for cw in caught:
                cname = cw.category.__name__
                if "PerfectSeparation" in cname:
                    warning_msgs.append(
                        "Perfect separation or quasi-complete separation detected in covariate space."
                    )
                elif "Convergence" in cname:
                    warning_msgs.append(
                        "Optimization algorithm failed to achieve standard likelihood convergence."
                    )
    except Exception as exc:
        raise ValueError(f"Failed to fit propensity score model (logistic regression): {exc}")

    if np.isnan(logit_res.params).any():
        raise ValueError(
            "Propensity score model fitting resulted in NaN coefficients (likely due to severe collinearity or rank deficiency)."
        )

    # 6. Predict Propensity Scores
    pred_scores = logit_res.predict(X_design)
    scores_array = np.asarray(pred_scores, dtype=float)

    if np.isnan(scores_array).any() or np.isinf(scores_array).any():
        raise ValueError("Propensity score model produced non-finite predicted probabilities.")
    if (scores_array < 0.0).any() or (scores_array > 1.0).any():
        raise ValueError("Propensity score model produced probabilities outside the valid [0, 1] interval.")

    # 7. Model Metadata
    converged = bool(getattr(logit_res, "mle_retvals", {}).get("converged", True))
    if not converged and not any("converge" in m.lower() for m in warning_msgs):
        warning_msgs.append("Model did not converge within the maximum iterations.")

    pseudo_r2 = (
        float(logit_res.prsquared)
        if hasattr(logit_res, "prsquared") and logit_res.prsquared is not None and not np.isnan(logit_res.prsquared)
        else None
    )

    model_metadata = {
        "model_type": "LogisticRegression (sm.Logit)",
        "converged": converged,
        "n_iterations": int(getattr(logit_res, "mle_retvals", {}).get("iterations", 0)),
        "pseudo_r2": pseudo_r2,
        "features": list(X_design.columns),
        "coefficients": {str(k): float(v) for k, v in logit_res.params.items()},
        "warnings": warning_msgs,
    }

    # 8. Overlap and Balance Diagnostics
    overlap_diag = compute_overlap_diagnostics(
        propensity_scores=scores_array,
        treatment=t_binary,
        treated_value=1.0,
        control_value=0.0,
        near_zero_threshold=near_zero_threshold,
        near_one_threshold=near_one_threshold,
    )

    covar_balance = compute_unadjusted_covariate_balance(
        data=df_validated,
        treatment_col=v["treatment_col"],
        confounder_cols=clean_confounders,
        treated_value=treated_label,
        control_value=control_label,
    )

    # 9. Educational Assumptions and Limitations
    assumptions_text = (
        "Propensity score estimation models the conditional probability of treatment assignment "
        "P(Treatment = treated | X) using observed confounders. Key identification assumptions "
        "for subsequent causal inference include: (1) Positivity / Common Support (0 < P(T=1 | X=x) < 1 "
        "for all x in the target population); (2) No Unmeasured Confounding ((Y(1), Y(0)) ⫫ T | X); and "
        "(3) Correct Model Specification for the assignment mechanism. "
        "Critical limitation: Propensity scores are ESTIMATED probabilities and adjust only for "
        "observed, modeled covariates. Propensity score estimation alone does NOT estimate a treatment "
        "effect or remove unmeasured confounding."
    )

    interpretation_text = (
        f"Estimated propensity scores for {len(scores_array)} observations using {len(clean_confounders)} confounder(s). "
        f"Propensity scores range from {overlap_diag['min_score']:.4f} to {overlap_diag['max_score']:.4f} "
        f"(treated: [{overlap_diag['min_score_treated']:.4f}, {overlap_diag['max_score_treated']:.4f}], "
        f"control: [{overlap_diag['min_score_control']:.4f}, {overlap_diag['max_score_control']:.4f}]). "
        f"Common support overlap: {'Present' if overlap_diag['has_overlap'] else 'Violated'}. "
        f"Scores near extreme boundaries: {overlap_diag['scores_near_zero']} near 0 (< {near_zero_threshold}), "
        f"{overlap_diag['scores_near_one']} near 1 (> {near_one_threshold})."
    )

    scores_list = [float(s) for s in scores_array]
    indices_list = list(df_validated.index)
    scores_by_idx = {idx: score for idx, score in zip(indices_list, scores_list)}

    return {
        "estimator": "logistic_propensity_score",
        "propensity_scores": scores_list,
        "scores": scores_list,
        "observation_indices": indices_list,
        "scores_by_index": scores_by_idx,
        "n_total": int(v["n_total"]),
        "n_treated": int(v["n_treated"]),
        "n_control": int(v["n_untreated"]),
        "n_untreated": int(v["n_untreated"]),
        "treatment_col": v["treatment_col"],
        "confounder_cols": clean_confounders,
        "numeric_confounders": numeric_confounders,
        "categorical_confounders": categorical_confounders,
        "treated_value": treated_label,
        "control_value": control_label,
        "untreated_value": control_label,
        "rows_dropped": int(v["rows_dropped"]),
        "model_features": list(X_design.columns),
        "model_coefficients": {str(k): float(v) for k, v in logit_res.params.items()},
        "model_metadata": model_metadata,
        "overlap_diagnostics": overlap_diag,
        "unadjusted_covariate_balance": covar_balance,
        "covariate_balance": covar_balance,
        "balance_assessment_type": "unadjusted",
        "post_adjustment_balance_note": (
            "Post-adjustment balance is not evaluated in this task because matching and weighting "
            "are not yet applied. Balance improvement will be assessed in the subsequent matching/weighting module."
        ),
        "categorical_balance_note": (
            "Categorical confounder balance assessment (e.g. difference in proportions) is deferred "
            "to the subsequent matching/weighting module."
        ),
        "interpretation": interpretation_text,
        "assumptions": assumptions_text,
    }
