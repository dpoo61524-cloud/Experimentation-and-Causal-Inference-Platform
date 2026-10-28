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
            "Severe overlap violation warning: Treated and control propensity score ranges are completely disjoint. "
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


def compute_effective_sample_size(weights: Sequence[float]) -> float:
    """Compute Kish's effective sample size (ESS) for an array of non-negative weights.

    Kish's ESS quantifies the degree of weight concentration:
        ESS = (sum_i w_i)^2 / sum_i (w_i^2)

    When weights are uniform (w_i = c for all i), ESS equals the nominal sample size n.
    When weights are highly variable or concentrated on a few observations, ESS is
    substantially smaller than n, reflecting the reduction in statistical precision
    (variance inflation) due to weighting.

    Parameters:
    -----------
    weights : Sequence[float]
        Sequence of non-negative weighting values.

    Returns:
    --------
    float
        The calculated effective sample size. Returns 0.0 if weights are empty or sum to zero.

    Raises:
    -------
    ValueError:
        If weights contain negative, NaN, or infinite values.
    """
    if weights is None:
        raise ValueError("weights cannot be None.")

    arr = np.asarray(weights, dtype=float)
    if len(arr) == 0:
        return 0.0

    if (arr < 0.0).any():
        raise ValueError("Effective sample size requires non-negative weights.")

    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("weights contain NaN or infinite values.")

    sum_w = float(np.sum(arr))
    if sum_w == 0.0:
        return 0.0

    sum_w_sq = float(np.sum(arr ** 2))
    if sum_w_sq == 0.0:
        return 0.0

    return float((sum_w ** 2) / sum_w_sq)


def compute_weighted_covariate_balance(
    data: pd.DataFrame,
    treatment_col: str,
    confounder_cols: Sequence[str],
    weights: Sequence[float],
    treated_value: Any = 1,
    control_value: Any = 0,
) -> Dict[str, Any]:
    """Compute covariate balance diagnostics (SMDs) before and after IPTW weighting.

    Calculates Standardized Mean Differences (SMDs) for numeric covariates and indicator
    variables for non-reference categories of categorical covariates:
        SMD_unweighted = (mean_treated - mean_control) / s_reference
        SMD_weighted   = (weighted_mean_treated - weighted_mean_control) / s_reference

    Reference Standard Deviation Convention:
    ----------------------------------------
    Following standard methodological practice (Austin 2009, 2011; Stuart 2010), the denominator
    s_reference is the pooled sample standard deviation of the unweighted sample:
        s_reference = sqrt(((n_1 - 1)*s_1^2 + (n_0 - 1)*s_0^2) / (n_1 + n_0 - 2))

    Using the unweighted pooled standard deviation keeps the scale fixed and prevents the
    post-weighting SMD from appearing artificially small or large due to variance changes.

    Parameters:
    -----------
    data : pd.DataFrame
        Validated tabular dataset containing treatment and confounder columns.
    treatment_col : str
        Name of the binary treatment column.
    confounder_cols : Sequence[str]
        List of observed confounder column names to evaluate.
    weights : Sequence[float]
        Array of IPTW weights aligned row-for-row with data.
    treated_value : Any, default 1
        Label identifying treated observations.
    control_value : Any, default 0
        Label identifying control observations.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'before_balance': Dict of covariate balance statistics before weighting.
        - 'after_balance': Dict of covariate balance statistics after weighting.
        - 'balance_summary': Summary metrics including max absolute SMDs and improvement counts.
    """
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    if treatment_col not in data.columns:
        raise ValueError(f"treatment_col '{treatment_col}' not found in data.")

    w_arr = np.asarray(weights, dtype=float)
    if len(w_arr) != len(data):
        raise ValueError(
            f"Length mismatch: weights ({len(w_arr)}) and data ({len(data)}) must have identical lengths."
        )

    t_series = data[treatment_col]
    treated_mask = (t_series == treated_value).values
    control_mask = (t_series == control_value).values

    n_1 = int(np.sum(treated_mask))
    n_0 = int(np.sum(control_mask))

    if n_1 == 0 or n_0 == 0:
        raise ValueError("Data must contain both treated and control observations for balance calculation.")

    w_1 = w_arr[treated_mask]
    w_0 = w_arr[control_mask]

    sum_w1 = float(np.sum(w_1))
    sum_w0 = float(np.sum(w_0))

    if sum_w1 == 0.0 or sum_w0 == 0.0:
        raise ValueError("Sum of weights in treated or control group is zero.")

    before_balance: Dict[str, Dict[str, Any]] = {}
    after_balance: Dict[str, Dict[str, Any]] = {}

    for col in confounder_cols:
        if col not in data.columns:
            raise ValueError(f"Confounder column '{col}' not found in data.")

        series = data[col]
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

        if is_numeric:
            x1 = series[treated_mask].astype(float).values
            x0 = series[control_mask].astype(float).values

            m_1 = float(np.mean(x1))
            m_0 = float(np.mean(x0))

            v_1 = float(np.var(x1, ddof=1)) if len(x1) > 1 else 0.0
            v_0 = float(np.var(x0, ddof=1)) if len(x0) > 1 else 0.0

            df_pooled = (len(x1) - 1) + (len(x0) - 1)
            if df_pooled > 0:
                pooled_var = ((len(x1) - 1) * v_1 + (len(x0) - 1) * v_0) / df_pooled
                s_ref = float(np.sqrt(max(0.0, pooled_var)))
            else:
                s_ref = 0.0

            # Unweighted SMD
            if s_ref > 0.0:
                smd_before = float((m_1 - m_0) / s_ref)
                abs_smd_before = float(abs(smd_before))
                zero_var = False
                note_before = None
            else:
                zero_var = True
                if m_1 == m_0:
                    smd_before = 0.0
                    abs_smd_before = 0.0
                    note_before = "Zero pooled variance: covariate values are identical across groups; SMD is 0.0."
                else:
                    smd_before = None
                    abs_smd_before = None
                    note_before = "Zero pooled variance with differing group means: SMD is mathematically undefined."

            # Weighted SMD
            m_1_w = float(np.sum(w_1 * x1) / sum_w1)
            m_0_w = float(np.sum(w_0 * x0) / sum_w0)

            if s_ref > 0.0:
                smd_after = float((m_1_w - m_0_w) / s_ref)
                abs_smd_after = float(abs(smd_after))
                note_after = None
            else:
                if m_1_w == m_0_w:
                    smd_after = 0.0
                    abs_smd_after = 0.0
                    note_after = "Zero pooled variance: weighted covariate values are identical; SMD is 0.0."
                else:
                    smd_after = None
                    abs_smd_after = None
                    note_after = "Zero pooled variance with differing weighted means: SMD is mathematically undefined."

            is_bal_before = bool(abs_smd_before <= 0.10) if abs_smd_before is not None else None
            is_bal_after = bool(abs_smd_after <= 0.10) if abs_smd_after is not None else None
            smd_improved = (
                bool(abs_smd_after < abs_smd_before)
                if (abs_smd_before is not None and abs_smd_after is not None)
                else None
            )

            before_balance[col] = {
                "covariate": col,
                "covariate_type": "numeric",
                "mean_treated": m_1,
                "mean_control": m_0,
                "pooled_std": s_ref,
                "smd": smd_before,
                "abs_smd": abs_smd_before,
                "zero_pooled_variance": zero_var,
                "is_balanced": is_bal_before,
                "note": note_before,
            }

            after_balance[col] = {
                "covariate": col,
                "covariate_type": "numeric",
                "mean_treated": m_1_w,
                "mean_control": m_0_w,
                "pooled_std": s_ref,
                "smd": smd_after,
                "abs_smd": abs_smd_after,
                "zero_pooled_variance": zero_var,
                "is_balanced": is_bal_after,
                "smd_improved": smd_improved,
                "note": note_after,
            }

        else:
            # Categorical Confounder: Evaluate non-reference level indicators
            unique_levels = sorted([lvl for lvl in pd.unique(series) if pd.notna(lvl)])
            if len(unique_levels) <= 1:
                continue

            ref_level = unique_levels[0]
            non_ref_levels = unique_levels[1:]

            for lvl in non_ref_levels:
                key = f"{col}_{lvl}"
                indicator = (series == lvl).astype(float).values

                i1 = indicator[treated_mask]
                i0 = indicator[control_mask]

                p_1 = float(np.mean(i1))
                p_0 = float(np.mean(i0))

                v_1 = float(np.var(i1, ddof=1)) if len(i1) > 1 else 0.0
                v_0 = float(np.var(i0, ddof=1)) if len(i0) > 1 else 0.0

                df_pooled = (len(i1) - 1) + (len(i0) - 1)
                if df_pooled > 0:
                    pooled_var = ((len(i1) - 1) * v_1 + (len(i0) - 1) * v_0) / df_pooled
                    s_ref = float(np.sqrt(max(0.0, pooled_var)))
                else:
                    s_ref = 0.0

                # Unweighted SMD for category indicator
                if s_ref > 0.0:
                    smd_before = float((p_1 - p_0) / s_ref)
                    abs_smd_before = float(abs(smd_before))
                    zero_var = False
                    note_before = f"Indicator for category '{lvl}' (reference: '{ref_level}')"
                else:
                    zero_var = True
                    if p_1 == p_0:
                        smd_before = 0.0
                        abs_smd_before = 0.0
                        note_before = (
                            f"Zero variance: category '{lvl}' proportion identical across groups; SMD is 0.0."
                        )
                    else:
                        smd_before = None
                        abs_smd_before = None
                        note_before = f"Zero variance for category '{lvl}' with differing proportions; SMD undefined."

                # Weighted SMD for category indicator
                p_1_w = float(np.sum(w_1 * i1) / sum_w1)
                p_0_w = float(np.sum(w_0 * i0) / sum_w0)

                if s_ref > 0.0:
                    smd_after = float((p_1_w - p_0_w) / s_ref)
                    abs_smd_after = float(abs(smd_after))
                    note_after = f"Weighted indicator for category '{lvl}' (reference: '{ref_level}')"
                else:
                    if p_1_w == p_0_w:
                        smd_after = 0.0
                        abs_smd_after = 0.0
                        note_after = f"Zero variance: weighted category '{lvl}' proportion identical; SMD is 0.0."
                    else:
                        smd_after = None
                        abs_smd_after = None
                        note_after = (
                            f"Zero variance for category '{lvl}' with differing weighted proportions; SMD undefined."
                        )

                is_bal_before = bool(abs_smd_before <= 0.10) if abs_smd_before is not None else None
                is_bal_after = bool(abs_smd_after <= 0.10) if abs_smd_after is not None else None
                smd_improved = (
                    bool(abs_smd_after < abs_smd_before)
                    if (abs_smd_before is not None and abs_smd_after is not None)
                    else None
                )

                before_balance[key] = {
                    "covariate": col,
                    "category": lvl,
                    "reference_category": ref_level,
                    "covariate_type": "categorical_indicator",
                    "mean_treated": p_1,
                    "mean_control": p_0,
                    "pooled_std": s_ref,
                    "smd": smd_before,
                    "abs_smd": abs_smd_before,
                    "zero_pooled_variance": zero_var,
                    "is_balanced": is_bal_before,
                    "note": note_before,
                }

                after_balance[key] = {
                    "covariate": col,
                    "category": lvl,
                    "reference_category": ref_level,
                    "covariate_type": "categorical_indicator",
                    "mean_treated": p_1_w,
                    "mean_control": p_0_w,
                    "pooled_std": s_ref,
                    "smd": smd_after,
                    "abs_smd": abs_smd_after,
                    "zero_pooled_variance": zero_var,
                    "is_balanced": is_bal_after,
                    "smd_improved": smd_improved,
                    "note": note_after,
                }

    # Summary metrics
    num_abs_before = [
        b["abs_smd"] for b in before_balance.values()
        if b["covariate_type"] == "numeric" and b["abs_smd"] is not None
    ]
    num_abs_after = [
        a["abs_smd"] for a in after_balance.values()
        if a["covariate_type"] == "numeric" and a["abs_smd"] is not None
    ]
    cat_abs_before = [
        b["abs_smd"] for b in before_balance.values()
        if b["covariate_type"] == "categorical_indicator" and b["abs_smd"] is not None
    ]
    cat_abs_after = [
        a["abs_smd"] for a in after_balance.values()
        if a["covariate_type"] == "categorical_indicator" and a["abs_smd"] is not None
    ]

    all_abs_before = [b["abs_smd"] for b in before_balance.values() if b["abs_smd"] is not None]
    all_abs_after = [a["abs_smd"] for a in after_balance.values() if a["abs_smd"] is not None]

    max_num_before = float(max(num_abs_before)) if num_abs_before else 0.0
    max_num_after = float(max(num_abs_after)) if num_abs_after else 0.0
    max_cat_before = float(max(cat_abs_before)) if cat_abs_before else None
    max_cat_after = float(max(cat_abs_after)) if cat_abs_after else None
    max_all_before = float(max(all_abs_before)) if all_abs_before else 0.0
    max_all_after = float(max(all_abs_after)) if all_abs_after else 0.0

    improved_cnt = sum(
        1 for k in before_balance
        if before_balance[k]["abs_smd"] is not None
        and after_balance[k]["abs_smd"] is not None
        and after_balance[k]["abs_smd"] < before_balance[k]["abs_smd"]
    )
    worsened_cnt = sum(
        1 for k in before_balance
        if before_balance[k]["abs_smd"] is not None
        and after_balance[k]["abs_smd"] is not None
        and after_balance[k]["abs_smd"] > before_balance[k]["abs_smd"]
    )
    unchanged_cnt = sum(
        1 for k in before_balance
        if before_balance[k]["abs_smd"] is not None
        and after_balance[k]["abs_smd"] is not None
        and after_balance[k]["abs_smd"] == before_balance[k]["abs_smd"]
    )

    balance_improved = bool(max_all_after < max_all_before)

    summary = {
        "max_abs_smd_numeric_before": max_num_before,
        "max_abs_smd_numeric_after": max_num_after,
        "max_abs_smd_categorical_before": max_cat_before,
        "max_abs_smd_categorical_after": max_cat_after,
        "max_abs_smd_before": max_all_before,
        "max_abs_smd_after": max_all_after,
        "balance_improved": balance_improved,
        "covariates_improved_count": improved_cnt,
        "covariates_worsened_count": worsened_cnt,
        "covariates_unchanged_count": unchanged_cnt,
    }

    return {
        "before_balance": before_balance,
        "after_balance": after_balance,
        "balance_summary": summary,
    }


def estimate_iptw_ate(
    data: Any,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: Sequence[str],
    treated_value: Optional[Any] = None,
    control_value: Optional[Any] = None,
    untreated_value: Optional[Any] = None,
    drop_missing: bool = False,
    extreme_weight_threshold: float = 20.0,
    truncate_weights: bool = False,
    max_weight: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Estimate Average Treatment Effect (ATE) using Inverse Probability of Treatment Weighting (IPTW).

    Methodology:
    ------------
    1. Propensity Score Estimation:
       Reuses `estimate_propensity_scores` to fit a logistic regression model
       e_i = P(Treatment = treated | X_i) and obtain empirical overlap diagnostics.
    2. Unstabilized ATE Weights:
       For each observation i:
           p_i = e_i if T_i = 1 else (1 - e_i)
           w_i = 1 / p_i
    3. Weighted Counterfactual Means:
           mu_1 = sum_{i in Treated} (w_i * Y_i) / sum_{i in Treated} w_i
           mu_0 = sum_{i in Control} (w_i * Y_i) / sum_{i in Control} w_i
    4. IPTW ATE Estimate:
           ATE_{IPTW} = mu_1 - mu_0  (treated minus control)
    5. Kish's Effective Sample Size:
           ESS = (sum_i w_i)^2 / sum_i (w_i^2)
    6. Covariate Balance Diagnostics:
           Evaluates Standardized Mean Differences (SMDs) before and after weighting
           for numeric confounders and categorical level indicators.

    Parameters:
    -----------
    data : Any
        Tabular dataset (DataFrame, dict of lists, list of dicts).
    treatment_col : str
        Name of the binary treatment column.
    outcome_col : str
        Name of the numeric outcome column.
    confounder_cols : Sequence[str]
        List of observed confounder column names.
    treated_value : Any, optional
        Label identifying treated observations. If None, inferred automatically.
    control_value : Any, optional
        Label identifying control observations. If None, inferred automatically.
    untreated_value : Any, optional
        Alias for control_value.
    drop_missing : bool, default False
        Missing data policy:
        - False (default): raises ValueError if missing values are detected.
        - True: drops incomplete rows and tracks rows_dropped.
    extreme_weight_threshold : float, default 20.0
        Diagnostic threshold above which a weight is flagged as extreme.
    truncate_weights : bool, default False
        Explicit opt-in setting to truncate weights at max_weight.
    max_weight : float, optional
        Maximum weight threshold used if truncate_weights=True.

    Returns:
    --------
    Dict[str, Any]
        Structured results dictionary containing ATE estimate, weighted means,
        sample counts, weight summaries, effective sample sizes, overlap diagnostics,
        and before-and-after covariate balance metrics.

    Raises:
    -------
    ValueError:
        If inputs are invalid, confounders are empty, positivity is violated (assigned
        probability near zero), or sample size is insufficient.
    """
    # 1. Flexible argument resolution
    actual_outcome_col = outcome_col
    actual_confounder_cols = confounder_cols

    if isinstance(outcome_col, (list, tuple)) and isinstance(confounder_cols, str):
        actual_outcome_col = confounder_cols
        actual_confounder_cols = outcome_col

    if control_value is not None and untreated_value is None:
        untreated_value = control_value
    if treated_value is None and "treatment_value" in kwargs:
        treated_value = kwargs["treatment_value"]

    # 2. Confounder list validation
    if actual_confounder_cols is None:
        raise ValueError("IPTW ATE estimation requires at least one confounder column. Confounder list cannot be None.")
    if not isinstance(actual_confounder_cols, (list, tuple)):
        raise ValueError(
            f"confounder_cols must be a list or tuple of column names, got {type(actual_confounder_cols).__name__}."
        )
    if len(actual_confounder_cols) == 0:
        raise ValueError("IPTW ATE estimation requires at least one confounder column. Confounder list cannot be empty.")

    # 3. Propensity score estimation & data validation reuse
    ps_res = estimate_propensity_scores(
        data=data,
        treatment_col=treatment_col,
        confounder_cols=actual_confounder_cols,
        outcome_col=actual_outcome_col,
        treated_value=treated_value,
        untreated_value=untreated_value,
        drop_missing=drop_missing,
    )

    df_raw = _to_dataframe(data)
    df_validated = df_raw.loc[ps_res["observation_indices"]].copy()

    treated_label = ps_res["treated_value"]
    control_label = ps_res["control_value"]
    treatment_series = df_validated[treatment_col]
    t_binary = (treatment_series == treated_label).astype(float).values
    outcome_series = pd.to_numeric(df_validated[actual_outcome_col], errors="raise").astype(float).values

    # Align propensity scores with rows
    scores_arr = np.array([ps_res["scores_by_index"][idx] for idx in df_validated.index], dtype=float)

    if np.isnan(scores_arr).any() or np.isinf(scores_arr).any():
        raise ValueError("Propensity scores contain NaN or infinite values.")

    # 4. Assigned-treatment probabilities & positivity checks
    assigned_probs = np.where(t_binary == 1.0, scores_arr, 1.0 - scores_arr)

    # Detect zero or near-zero probabilities that would yield infinite weights
    zero_prob_mask = (assigned_probs <= 0.0) | np.isclose(assigned_probs, 0.0, atol=1e-12)
    if np.any(zero_prob_mask):
        n_violating = int(np.sum(zero_prob_mask))
        raise ValueError(
            f"Positivity violation: {n_violating} observation(s) have an assigned-treatment probability "
            f"of zero (or numerically indistinguishable from zero, <= 1e-12). "
            "IPTW weighting would produce infinite weights and division by zero. "
            "Check common support and overlap before applying IPTW."
        )

    # 5. Calculate weights
    raw_weights = 1.0 / assigned_probs
    if np.isnan(raw_weights).any() or np.isinf(raw_weights).any():
        raise ValueError("IPTW calculation resulted in non-finite weights.")

    weights = raw_weights.copy()
    rows_truncated = 0
    if truncate_weights and max_weight is not None:
        if max_weight <= 0:
            raise ValueError(f"max_weight must be positive, got {max_weight}.")
        rows_truncated = int(np.sum(weights > max_weight))
        weights = np.minimum(weights, max_weight)

    # 6. Weighted outcome means & IPTW ATE
    treated_mask = (t_binary == 1.0)
    control_mask = (t_binary == 0.0)

    w_treated = weights[treated_mask]
    w_control = weights[control_mask]

    y_treated = outcome_series[treated_mask]
    y_control = outcome_series[control_mask]

    sum_wt = float(np.sum(w_treated))
    sum_wc = float(np.sum(w_control))

    if sum_wt == 0.0 or sum_wc == 0.0:
        raise ValueError("Sum of IPTW weights in treated or control group is zero.")

    weighted_mean_treated = float(np.sum(w_treated * y_treated) / sum_wt)
    weighted_mean_control = float(np.sum(w_control * y_control) / sum_wc)

    iptw_ate = float(weighted_mean_treated - weighted_mean_control)

    # Raw unadjusted difference in means
    raw_mean_treated = float(np.mean(y_treated))
    raw_mean_control = float(np.mean(y_control))
    raw_diff = float(raw_mean_treated - raw_mean_control)

    # 7. Effective Sample Sizes
    ess_treated = compute_effective_sample_size(w_treated)
    ess_control = compute_effective_sample_size(w_control)
    ess_total = float(ess_treated + ess_control)

    n_t = int(len(w_treated))
    n_c = int(len(w_control))

    ess_treated_frac = float(ess_treated / n_t) if n_t > 0 else 0.0
    ess_control_frac = float(ess_control / n_c) if n_c > 0 else 0.0

    # 8. Covariate balance before and after weighting
    balance_res = compute_weighted_covariate_balance(
        data=df_validated,
        treatment_col=treatment_col,
        confounder_cols=actual_confounder_cols,
        weights=weights,
        treated_value=treated_label,
        control_value=control_label,
    )

    # 9. Weight summary & diagnostic checks
    ext_threshold = float(extreme_weight_threshold)
    n_extreme = int(np.sum(weights > ext_threshold))
    has_extreme = bool(n_extreme > 0)

    weight_summary = {
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
        "mean_weight": float(np.mean(weights)),
        "std_weight": float(np.std(weights, ddof=1)) if len(weights) > 1 else 0.0,
        "median_weight": float(np.median(weights)),
        "p95_weight": float(np.percentile(weights, 95)),
        "p99_weight": float(np.percentile(weights, 99)),
        "extreme_weight_threshold": ext_threshold,
        "n_extreme_weights": n_extreme,
        "has_extreme_weights": has_extreme,
        "is_truncated": truncate_weights,
        "truncation_max_weight": max_weight if truncate_weights else None,
        "rows_truncated": rows_truncated,
        "weights_truncated_count": rows_truncated,
        "max": float(np.max(weights)),
        "min": float(np.min(weights)),
        "mean": float(np.mean(weights)),
    }

    # 10. Warnings formulation
    warnings_list: List[str] = []
    if ps_res["overlap_diagnostics"]["warning"]:
        warnings_list.append(ps_res["overlap_diagnostics"]["warning"])

    if has_extreme:
        warnings_list.append(
            f"Extreme weights warning: {n_extreme} observation(s) have IPTW weights exceeding the diagnostic "
            f"threshold of {ext_threshold:.2f} (max weight: {weight_summary['max_weight']:.2f}). Extreme weights "
            "can inflate estimator variance and lead to unstable treatment effect estimates."
        )

    if ess_treated_frac < 0.50 or ess_control_frac < 0.50:
        warnings_list.append(
            f"Weight concentration warning: Effective sample size in one or both treatment arms is below 50% "
            f"of nominal sample size (treated ESS: {ess_treated:.1f}/{n_t} [{ess_treated_frac:.1%}], "
            f"control ESS: {ess_control:.1f}/{n_c} [{ess_control_frac:.1%}]), indicating substantial variance inflation."
        )

    if not balance_res["balance_summary"]["balance_improved"]:
        warnings_list.append(
            "Covariate balance warning: Weighting did not improve overall maximum standardized mean difference. "
            f"Max absolute SMD before: {balance_res['balance_summary']['max_abs_smd_before']:.4f}, "
            f"after: {balance_res['balance_summary']['max_abs_smd_after']:.4f}."
        )

    # 11. Interpretation and assumptions text
    interpretation_text = (
        f"The IPTW-estimated Average Treatment Effect (ATE) is {iptw_ate:.4f} "
        f"(weighted treated outcome mean: {weighted_mean_treated:.4f}, control: {weighted_mean_control:.4f}; "
        f"unadjusted difference in means: {raw_diff:.4f}). "
        f"Effective sample sizes: treated {ess_treated:.1f}/{n_t} ({ess_treated_frac:.1%}), "
        f"control {ess_control:.1f}/{n_c} ({ess_control_frac:.1%}). "
        f"Max absolute covariate SMD: {balance_res['balance_summary']['max_abs_smd_before']:.4f} before weighting, "
        f"{balance_res['balance_summary']['max_abs_smd_after']:.4f} after weighting "
        f"({'improved' if balance_res['balance_summary']['balance_improved'] else 'not improved'})."
    )

    assumptions_text = (
        "Inverse Probability of Treatment Weighting (IPTW) identifies the Average Treatment Effect (ATE) "
        "under five fundamental identification assumptions: (1) Conditional Exchangeability / No Unmeasured "
        "Confounding (potential outcomes are conditionally independent of treatment given observed confounders: "
        "(Y(1), Y(0)) ⫫ T | X); (2) Positivity / Common Support (0 < P(T=1 | X=x) < 1 for all relevant x); "
        "(3) Correct Propensity Score Specification (the logistic regression model accurately models the true "
        "assignment mechanism); (4) Consistency (the observed outcome under treatment T matches the corresponding "
        "potential outcome Y = Y(T)); and (5) SUTVA (no interference between units). "
        "Critical limitations: IPTW adjusts only for observed, modeled covariates. Good observed balance does "
        "NOT guarantee absence of unmeasured confounding. Near-positivity violations produce extreme weights, "
        "inflating variance and destabilizing effect estimates."
    )

    return {
        "estimator": "iptw_ate",
        "iptw_ate": iptw_ate,
        "ate": iptw_ate,
        "weighted_mean_outcome_treated": weighted_mean_treated,
        "weighted_mean_outcome_control": weighted_mean_control,
        "unadjusted_ate": raw_diff,
        "raw_difference": raw_diff,
        "difference_in_means": raw_diff,
        "n_total": int(ps_res["n_total"]),
        "n_treated": n_t,
        "n_control": n_c,
        "n_untreated": n_c,
        "treatment_col": treatment_col,
        "outcome_col": actual_outcome_col,
        "confounder_cols": ps_res["confounder_cols"],
        "numeric_confounders": ps_res["numeric_confounders"],
        "categorical_confounders": ps_res["categorical_confounders"],
        "treated_value": treated_label,
        "control_value": control_label,
        "untreated_value": control_label,
        "rows_dropped": int(ps_res["rows_dropped"]),
        "propensity_score_summary": {
            "min": float(np.min(scores_arr)),
            "max": float(np.max(scores_arr)),
            "mean": float(np.mean(scores_arr)),
            "median": float(np.median(scores_arr)),
            "std": float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0,
        },
        "weight_summary": weight_summary,
        "weights": [float(w) for w in weights],
        "ess_treated": ess_treated,
        "ess_control": ess_control,
        "ess_total": ess_total,
        "ess_treated_fraction": ess_treated_frac,
        "ess_control_fraction": ess_control_frac,
        "overlap_diagnostics": ps_res["overlap_diagnostics"],
        "covariate_balance_before": balance_res["before_balance"],
        "covariate_balance_after": balance_res["after_balance"],
        "balance_summary": balance_res["balance_summary"],
        "warnings": warnings_list,
        "interpretation": interpretation_text,
        "assumptions": assumptions_text,
    }


# ============================================================================
# Difference-in-Differences (DiD) Estimation Foundation
# ============================================================================

def validate_did_data(
    data: Any,
    treatment_col: str,
    time_col: Optional[str] = None,
    outcome_col: Optional[str] = None,
    treated_value: Optional[Any] = None,
    control_value: Optional[Any] = None,
    pre_value: Optional[Any] = None,
    post_value: Optional[Any] = None,
    drop_missing: bool = False,
    untreated_value: Optional[Any] = None,
    pre_period_value: Optional[Any] = None,
    post_period_value: Optional[Any] = None,
    time_period_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate tabular causal data for Difference-in-Differences (DiD) estimation.

    Parameters:
    -----------
    data : pd.DataFrame, Dict[str, Sequence], or List[Dict[str, Any]]
        The tabular dataset to validate.
    treatment_col : str
        Name of the treatment-group column distinguishing treated from control units.
    time_col : str, optional
        Name of the time-period column distinguishing pre-intervention from post-intervention.
    outcome_col : str, optional
        Name of the numeric outcome column.
    treated_value : Any, optional
        Label identifying the treated group.
    control_value : Any, optional
        Label identifying the control group.
    pre_value : Any, optional
        Label identifying the pre-intervention period.
    post_value : Any, optional
        Label identifying the post-intervention period.
    drop_missing : bool, default False
        Explicit missing data policy:
        - If False (default): raises ValueError if any missing values exist in analysis columns.
        - If True: explicitly drops rows containing missing values in treatment, time, or outcome columns.
    untreated_value : Any, optional
        Alias for control_value.
    pre_period_value : Any, optional
        Alias for pre_value.
    post_period_value : Any, optional
        Alias for post_value.
    time_period_col : str, optional
        Alias for time_col.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing validated DataFrame, resolved labels, masks, and cell counts.

    Raises:
    -------
    ValueError:
        If columns are missing, data is empty, labels are invalid, outcome is non-numeric,
        or any of the four group-period cells has 0 observations.
    """
    # 1. Alias Resolution
    if control_value is None and untreated_value is not None:
        control_value = untreated_value
    if pre_value is None and pre_period_value is not None:
        pre_value = pre_period_value
    if post_value is None and post_period_value is not None:
        post_value = post_period_value
    if time_col is None and time_period_col is not None:
        time_col = time_period_col

    if not isinstance(treatment_col, str) or not treatment_col.strip():
        raise ValueError("treatment_col must be a non-empty string.")
    if time_col is None or not isinstance(time_col, str) or not time_col.strip():
        raise ValueError("time_col (or time_period_col) must be a non-empty string.")
    if outcome_col is None or not isinstance(outcome_col, str) or not outcome_col.strip():
        raise ValueError("outcome_col must be a non-empty string.")

    # 2. Distinct Column Checks
    if treatment_col == outcome_col:
        raise ValueError(
            f"treatment_col and outcome_col cannot be the same column ('{treatment_col}')."
        )
    if time_col == outcome_col:
        raise ValueError(
            f"time_col and outcome_col cannot be the same column ('{time_col}')."
        )
    if treatment_col == time_col:
        raise ValueError(
            f"treatment_col and time_col cannot be the same column ('{treatment_col}')."
        )

    # 3. DataFrame Conversion & Column Existence
    df = _to_dataframe(data)
    available_cols = list(df.columns)

    if treatment_col not in df.columns:
        raise ValueError(
            f"Treatment column '{treatment_col}' not found in dataset. "
            f"Available columns: {available_cols}"
        )
    if time_col not in df.columns:
        raise ValueError(
            f"Time-period column '{time_col}' not found in dataset. "
            f"Available columns: {available_cols}"
        )
    if outcome_col not in df.columns:
        raise ValueError(
            f"Outcome column '{outcome_col}' not found in dataset. "
            f"Available columns: {available_cols}"
        )

    # 4. Explicit Missing Data Policy
    required_cols = [treatment_col, time_col, outcome_col]
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

    # 5. Outcome Column Validation
    outcome_series = df[outcome_col]
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

    # 6. Treatment Column Validation & Binary Label Resolution
    treatment_series = df[treatment_col]
    unique_treatment = list(pd.unique(treatment_series))

    if len(unique_treatment) == 0:
        raise ValueError(f"Treatment column '{treatment_col}' contains no observations.")

    if treated_value is not None and control_value is not None:
        if treated_value == control_value:
            raise ValueError("treated_value and control_value cannot be identical.")

        is_valid_t = treatment_series.isin([treated_value, control_value])
        if not is_valid_t.all():
            unexpected = list(treatment_series[~is_valid_t].unique())
            raise ValueError(
                f"Treatment column '{treatment_col}' contains values other than specified treated "
                f"({treated_value!r}) and control ({control_value!r}) labels: {unexpected}."
            )
        if (treatment_series == treated_value).sum() == 0:
            raise ValueError(
                f"Specified treated label ({treated_value!r}) not found in treatment column '{treatment_col}'. "
                f"Available values: {unique_treatment}"
            )
        if (treatment_series == control_value).sum() == 0:
            raise ValueError(
                f"Specified control label ({control_value!r}) not found in treatment column '{treatment_col}'. "
                f"Available values: {unique_treatment}"
            )
        resolved_treated = treated_value
        resolved_control = control_value
    elif treated_value is not None or control_value is not None:
        raise ValueError(
            "Both treated_value and control_value must be provided together, or neither to infer them."
        )
    else:
        if len(unique_treatment) > 2:
            raise ValueError(
                f"Treatment column '{treatment_col}' must be binary, but contains {len(unique_treatment)} "
                f"unique values: {unique_treatment}. "
                "Please specify treated_value and control_value or subset the dataset."
            )
        if len(unique_treatment) == 1:
            raise ValueError(
                f"Treatment column '{treatment_col}' contains only 1 unique value ({unique_treatment[0]!r}). "
                "Difference-in-Differences requires both treated and control observations."
            )

        unique_t_set = set(unique_treatment)
        if unique_t_set == {0, 1}:
            resolved_treated, resolved_control = 1, 0
        elif unique_t_set == {False, True}:
            resolved_treated, resolved_control = True, False
        else:
            s0 = str(unique_treatment[0]).lower()
            s1 = str(unique_treatment[1]).lower()
            if "treat" in s0 and ("control" in s1 or "untreat" in s1):
                resolved_treated, resolved_control = unique_treatment[0], unique_treatment[1]
            elif "treat" in s1 and ("control" in s0 or "untreat" in s0):
                resolved_treated, resolved_control = unique_treatment[1], unique_treatment[0]
            else:
                sorted_vals = sorted(unique_treatment, key=lambda v: str(v))
                resolved_control, resolved_treated = sorted_vals[0], sorted_vals[1]

    # 7. Time Column Validation & Binary Period Resolution
    time_series = df[time_col]
    unique_time = list(pd.unique(time_series))

    if len(unique_time) == 0:
        raise ValueError(f"Time-period column '{time_col}' contains no observations.")

    if pre_value is not None and post_value is not None:
        if pre_value == post_value:
            raise ValueError("pre_value and post_value cannot be identical.")

        is_valid_time = time_series.isin([pre_value, post_value])
        if not is_valid_time.all():
            unexpected_time = list(time_series[~is_valid_time].unique())
            raise ValueError(
                f"Time-period column '{time_col}' contains values other than specified pre "
                f"({pre_value!r}) and post ({post_value!r}) labels: {unexpected_time}."
            )
        if (time_series == pre_value).sum() == 0:
            raise ValueError(
                f"Specified pre-intervention label ({pre_value!r}) not found in time-period column '{time_col}'. "
                f"Available values: {unique_time}"
            )
        if (time_series == post_value).sum() == 0:
            raise ValueError(
                f"Specified post-intervention label ({post_value!r}) not found in time-period column '{time_col}'. "
                f"Available values: {unique_time}"
            )
        resolved_pre = pre_value
        resolved_post = post_value
    elif pre_value is not None or post_value is not None:
        raise ValueError(
            "Both pre_value and post_value must be provided together, or neither to infer them."
        )
    else:
        if len(unique_time) > 2:
            raise ValueError(
                f"Time-period column '{time_col}' must contain exactly 2 unique periods (pre and post), "
                f"but contains {len(unique_time)}: {unique_time}. "
                "Please specify pre_value and post_value or subset the dataset."
            )
        if len(unique_time) == 1:
            raise ValueError(
                f"Time-period column '{time_col}' contains only 1 unique value ({unique_time[0]!r}). "
                "Difference-in-Differences requires both pre- and post-intervention periods."
            )

        unique_time_set = set(unique_time)
        if unique_time_set == {0, 1}:
            resolved_pre, resolved_post = 0, 1
        elif unique_time_set == {False, True}:
            resolved_pre, resolved_post = False, True
        else:
            s0 = str(unique_time[0]).lower()
            s1 = str(unique_time[1]).lower()
            if "post" in s0 and "pre" in s1:
                resolved_post, resolved_pre = unique_time[0], unique_time[1]
            elif "post" in s1 and "pre" in s0:
                resolved_post, resolved_pre = unique_time[1], unique_time[0]
            elif "after" in s0 and "before" in s1:
                resolved_post, resolved_pre = unique_time[0], unique_time[1]
            elif "after" in s1 and "before" in s0:
                resolved_post, resolved_pre = unique_time[1], unique_time[0]
            else:
                sorted_time = sorted(unique_time, key=lambda v: str(v))
                resolved_pre, resolved_post = sorted_time[0], sorted_time[1]

    # 8. Four Group-Period Cells Validation
    mask_t_pre = (treatment_series == resolved_treated) & (time_series == resolved_pre)
    mask_t_post = (treatment_series == resolved_treated) & (time_series == resolved_post)
    mask_c_pre = (treatment_series == resolved_control) & (time_series == resolved_pre)
    mask_c_post = (treatment_series == resolved_control) & (time_series == resolved_post)

    n_t_pre = int(mask_t_pre.sum())
    n_t_post = int(mask_t_post.sum())
    n_c_pre = int(mask_c_pre.sum())
    n_c_post = int(mask_c_post.sum())

    empty_cells = []
    if n_t_pre == 0:
        empty_cells.append(f"Treatment Pre (treatment={resolved_treated!r}, period={resolved_pre!r})")
    if n_t_post == 0:
        empty_cells.append(f"Treatment Post (treatment={resolved_treated!r}, period={resolved_post!r})")
    if n_c_pre == 0:
        empty_cells.append(f"Control Pre (treatment={resolved_control!r}, period={resolved_pre!r})")
    if n_c_post == 0:
        empty_cells.append(f"Control Post (treatment={resolved_control!r}, period={resolved_post!r})")

    if empty_cells:
        raise ValueError(
            f"Difference-in-Differences requires at least one observation in all four group-period cells. "
            f"Empty cell(s) found: {', '.join(empty_cells)}."
        )

    return {
        "is_valid": True,
        "validated_data": df,
        "treatment_col": treatment_col,
        "time_col": time_col,
        "outcome_col": outcome_col,
        "treated_value": resolved_treated,
        "control_value": resolved_control,
        "pre_value": resolved_pre,
        "post_value": resolved_post,
        "rows_dropped": rows_dropped,
        "n_total": len(df),
        "cell_counts": {
            "treatment_pre": n_t_pre,
            "treatment_post": n_t_post,
            "control_pre": n_c_pre,
            "control_post": n_c_post,
        },
        "mask_treatment_pre": mask_t_pre,
        "mask_treatment_post": mask_t_post,
        "mask_control_pre": mask_c_pre,
        "mask_control_post": mask_c_post,
        "numeric_outcomes": numeric_outcomes,
    }


def estimate_difference_in_differences(
    data: Any,
    treatment_col: str,
    time_col: Optional[str] = None,
    outcome_col: Optional[str] = None,
    treated_value: Optional[Any] = None,
    control_value: Optional[Any] = None,
    pre_value: Optional[Any] = None,
    post_value: Optional[Any] = None,
    drop_missing: bool = False,
    untreated_value: Optional[Any] = None,
    pre_period_value: Optional[Any] = None,
    post_period_value: Optional[Any] = None,
    time_period_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Estimate treatment effect using Difference-in-Differences (DiD).

    Methodology & Formula:
    ----------------------
    Difference-in-Differences compares the before-and-after outcome change observed in a
    treatment group to the before-and-after outcome change observed in a control group:

    1. Group-Period Sample Means:
       - Treatment pre-intervention mean:  Ybar_{T,pre}  = mean(Y | T = treated, Time = pre)
       - Treatment post-intervention mean: Ybar_{T,post} = mean(Y | T = treated, Time = post)
       - Control pre-intervention mean:    Ybar_{C,pre}  = mean(Y | T = control, Time = pre)
       - Control post-intervention mean:   Ybar_{C,post} = mean(Y | T = control, Time = post)

    2. Time Differences:
       - Treatment group change: Delta_T = Ybar_{T,post} - Ybar_{T,pre}
       - Control group change:   Delta_C = Ybar_{C,post} - Ybar_{C,pre}

    3. Difference-in-Differences Estimate:
       - DiD = Delta_T - Delta_C = (Ybar_{T,post} - Ybar_{T,pre}) - (Ybar_{C,post} - Ybar_{C,pre})

    Identifying Causal Assumptions:
    -------------------------------
    Difference-in-Differences identifies the Average Treatment Effect on the Treated (ATT)
    under five core structural assumptions:
    1. Parallel Trends: In the counterfactual absence of treatment, the average outcome in the
       treatment group would have evolved along the exact same trajectory as the control group:
           E[Y(0)_post - Y(0)_pre | T = 1] = E[Y(0)_post - Y(0)_pre | T = 0]
    2. No Anticipation: Units do not alter their behavior or outcomes in anticipation of future
       treatment before the intervention occurs (pre-period outcomes are uncontaminated).
    3. No Differential Concurrent Shocks: No extraneous policy change, economic disruption, or
       confounding event coincided with the intervention that selectively affected the treated
       or control group.
    4. Consistent Measurement & Stable Group Definitions: The outcome variable is measured with
       identical methodology across both periods, and group membership does not shift dynamically.
    5. No Spillover Effects (SUTVA): The intervention applied to treated units does not indirectly
       alter outcomes among control units.

    Critical Limitations:
    ---------------------
    - These identifying assumptions cannot be proven purely from computing the DiD point estimate.
    - DiD accommodates repeated cross-sections or panel data, but observations must not be assumed
      independent when the same individuals or units are measured across multiple periods.

    Parameters:
    -----------
    data : pd.DataFrame, Dict[str, Sequence], or List[Dict[str, Any]]
        The tabular dataset to analyze.
    treatment_col : str
        Name of the treatment-group column.
    time_col : str, optional
        Name of the time-period column.
    outcome_col : str, optional
        Name of the numeric outcome column.
    treated_value : Any, optional
        Label identifying the treated group. Inferred automatically if None.
    control_value : Any, optional
        Label identifying the control group. Inferred automatically if None.
    pre_value : Any, optional
        Label identifying the pre-intervention period. Inferred automatically if None.
    post_value : Any, optional
        Label identifying the post-intervention period. Inferred automatically if None.
    drop_missing : bool, default False
        Explicit missing data policy:
        - If False (default): raises ValueError if missing values exist in analysis columns.
        - If True: drops incomplete observations and records rows_dropped.
    untreated_value : Any, optional
        Alias for control_value.
    pre_period_value : Any, optional
        Alias for pre_value.
    post_period_value : Any, optional
        Alias for post_value.
    time_period_col : str, optional
        Alias for time_col.

    Returns:
    --------
    Dict[str, Any]
        Structured dictionary containing:
        - 'estimator': 'difference_in_differences'
        - 'did_estimate': Difference-in-Differences point estimate
        - 'did': Alias for did_estimate
        - 'ate': Alias for did_estimate
        - 'treatment_mean_pre': Mean outcome for treated units before intervention
        - 'treatment_mean_post': Mean outcome for treated units after intervention
        - 'control_mean_pre': Mean outcome for control units before intervention
        - 'control_mean_post': Mean outcome for control units after intervention
        - 'treated_mean_pre': Alias for treatment_mean_pre
        - 'treated_mean_post': Alias for treatment_mean_post
        - 'treatment_change': Outcome change for treated units (post - pre)
        - 'control_change': Outcome change for control units (post - pre)
        - 'delta_treatment': Alias for treatment_change
        - 'delta_control': Alias for control_change
        - 'delta_treated': Alias for treatment_change
        - 'cell_counts': Dictionary with counts for the 4 cells
        - 'n_treatment_pre': Count of treated observations in pre-period
        - 'n_treatment_post': Count of treated observations in post-period
        - 'n_control_pre': Count of control observations in pre-period
        - 'n_control_post': Count of control observations in post-period
        - 'n_treated_pre': Alias for n_treatment_pre
        - 'n_treated_post': Alias for n_treatment_post
        - 'n_treated': Total treated observations
        - 'n_control': Total control observations
        - 'n_untreated': Alias for n_control
        - 'n_total': Total observations used
        - 'rows_dropped': Number of rows dropped due to missing data
        - 'treatment_col': Treatment column name
        - 'time_col': Time-period column name
        - 'time_period_col': Alias for time_col
        - 'outcome_col': Outcome column name
        - 'treated_value': Resolved treated label
        - 'control_value': Resolved control label
        - 'untreated_value': Alias for control_value
        - 'pre_value': Resolved pre-period label
        - 'post_value': Resolved post-period label
        - 'interpretation': Narrative summary of results
        - 'assumptions': Educational explanation of identifying assumptions and limitations
    """
    val_res = validate_did_data(
        data=data,
        treatment_col=treatment_col,
        time_col=time_col,
        outcome_col=outcome_col,
        treated_value=treated_value,
        control_value=control_value,
        pre_value=pre_value,
        post_value=post_value,
        drop_missing=drop_missing,
        untreated_value=untreated_value,
        pre_period_value=pre_period_value,
        post_period_value=post_period_value,
        time_period_col=time_period_col,
    )

    numeric_outcomes = val_res["numeric_outcomes"]
    mask_t_pre = val_res["mask_treatment_pre"]
    mask_t_post = val_res["mask_treatment_post"]
    mask_c_pre = val_res["mask_control_pre"]
    mask_c_post = val_res["mask_control_post"]

    mean_t_pre = float(numeric_outcomes[mask_t_pre].mean())
    mean_t_post = float(numeric_outcomes[mask_t_post].mean())
    mean_c_pre = float(numeric_outcomes[mask_c_pre].mean())
    mean_c_post = float(numeric_outcomes[mask_c_post].mean())

    delta_t = float(mean_t_post - mean_t_pre)
    delta_c = float(mean_c_post - mean_c_pre)
    did = float(delta_t - delta_c)

    cell_counts = val_res["cell_counts"]
    n_t_pre = cell_counts["treatment_pre"]
    n_t_post = cell_counts["treatment_post"]
    n_c_pre = cell_counts["control_pre"]
    n_c_post = cell_counts["control_post"]

    interpretation_text = (
        f"The Difference-in-Differences (DiD) treatment effect estimate is {did:.4f}. "
        f"Treatment group outcome changed by {delta_t:.4f} (from {mean_t_pre:.4f} pre-intervention "
        f"to {mean_t_post:.4f} post-intervention; n_pre={n_t_pre}, n_post={n_t_post}). "
        f"Control group outcome changed by {delta_c:.4f} (from {mean_c_pre:.4f} pre-intervention "
        f"to {mean_c_post:.4f} post-intervention; n_pre={n_c_pre}, n_post={n_c_post})."
    )

    assumptions_text = (
        "Difference-in-Differences (DiD) identifies the causal treatment effect under five core assumptions: "
        "(1) Parallel Trends: In the absence of treatment, the average outcome of the treatment group would "
        "have evolved with the same trajectory as the control group (E[Y(0)_post - Y(0)_pre | T=1] = "
        "E[Y(0)_post - Y(0)_pre | T=0]); "
        "(2) No Anticipation: Units do not change their behavior or outcomes in anticipation of future treatment "
        "prior to the post-intervention period; "
        "(3) No Differential Concurrent Shocks: No other contemporaneous event or external policy shock "
        "affected the treatment group differently from the control group at or around the intervention time; "
        "(4) Consistent Measurement and Stable Group Definitions: Outcome measurement methods and group "
        "classifications remain invariant and comparable across both pre and post periods; and "
        "(5) No Spillover Effects (SUTVA): The intervention in the treated group does not affect the outcomes "
        "of units in the control group. "
        "Critical limitations: These identifying assumptions cannot be proven solely by computing the point estimate. "
        "DiD point estimation accommodates repeated cross-sections or panel data, but observations should not "
        "be assumed independent when identical subjects are measured repeatedly across time periods."
    )

    return {
        "estimator": "difference_in_differences",
        "did_estimate": did,
        "did": did,
        "ate": did,
        "treatment_mean_pre": mean_t_pre,
        "treatment_mean_post": mean_t_post,
        "control_mean_pre": mean_c_pre,
        "control_mean_post": mean_c_post,
        "treated_mean_pre": mean_t_pre,
        "treated_mean_post": mean_t_post,
        "treatment_change": delta_t,
        "control_change": delta_c,
        "delta_treatment": delta_t,
        "delta_control": delta_c,
        "delta_treated": delta_t,
        "cell_counts": {
            "treatment_pre": n_t_pre,
            "treatment_post": n_t_post,
            "control_pre": n_c_pre,
            "control_post": n_c_post,
        },
        "n_treatment_pre": n_t_pre,
        "n_treatment_post": n_t_post,
        "n_control_pre": n_c_pre,
        "n_control_post": n_c_post,
        "n_treated_pre": n_t_pre,
        "n_treated_post": n_t_post,
        "n_treated": n_t_pre + n_t_post,
        "n_control": n_c_pre + n_c_post,
        "n_untreated": n_c_pre + n_c_post,
        "n_total": n_t_pre + n_t_post + n_c_pre + n_c_post,
        "rows_dropped": val_res["rows_dropped"],
        "treatment_col": val_res["treatment_col"],
        "time_col": val_res["time_col"],
        "time_period_col": val_res["time_col"],
        "outcome_col": val_res["outcome_col"],
        "treated_value": val_res["treated_value"],
        "control_value": val_res["control_value"],
        "untreated_value": val_res["control_value"],
        "pre_value": val_res["pre_value"],
        "post_value": val_res["post_value"],
        "interpretation": interpretation_text,
        "assumptions": assumptions_text,
    }


# ============================================================================
# Causal Sensitivity Analysis and Robustness Checks
# ============================================================================

def analyze_unmeasured_confounding_sensitivity(
    observed_estimate: float,
    bias_values: Union[float, Sequence[float]],
    confidence_interval: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Perform hypothetical bias-adjustment sensitivity analysis for unmeasured confounding.

    Methodology:
    ------------
    This function implements a transparent, scenario-based bias adjustment model:

        tau_adjusted(b) = tau_observed - b

    where b represents an assumed signed bias induced by unmeasured confounding.
    - Positive bias (b > 0): Assumes unmeasured confounding inflated the observed treatment
      effect estimate upward (so adjusted estimate is smaller).
    - Negative bias (b < 0): Assumes unmeasured confounding suppressed the observed treatment
      effect estimate downward (so adjusted estimate is larger).
    - Zero bias (b = 0): Unmeasured confounding has zero net impact (adjusted equals observed).

    When a confidence interval [L, U] is provided, the interval bounds are shifted by b:
        [L - b, U - b]
    and the scenario reports whether the shifted interval spans zero.

    Critical Assumptions & Limitations:
    -----------------------------------
    1. Hypothetical Scenarios: The bias values (b) are user-specified scenarios, NOT empirically
       estimated quantities.
    2. Additive Bias: Assumes a simple additive bias on the outcome contrast scale.
    3. No P-Value Invention: This analysis shifts point estimates and interval bounds conditional
       on hypothetical assumptions; it does not calculate or invent pseudo p-values.
    4. Explanatory Scope: Sensitivity analysis demonstrates how conclusions would change under
       specific assumptions about unmeasured confounding, but cannot prove that unmeasured
       confounding is absent or bounded by the chosen values.

    Parameters:
    -----------
    observed_estimate : float
        Observed causal effect point estimate (must be a finite numeric float or int).
    bias_values : float or Sequence[float]
        One or more hypothetical signed bias values (b).
    confidence_interval : Sequence[float], optional
        Optional two-element confidence interval [lower, upper] for the observed estimate.

    Returns:
    --------
    Dict[str, Any]
        Structured dictionary containing:
        - 'observed_estimate': float
        - 'confidence_interval': [lower, upper] or None
        - 'bias_scenarios': list of scenario dictionaries with adjusted estimates and intervals
        - 'sign_convention': explanation of bias direction
        - 'interpretation': narrative summary of scenario findings
        - 'assumptions': educational explanation of assumptions and limitations

    Raises:
    -------
    ValueError:
        If observed_estimate, bias_values, or confidence_interval are non-numeric,
        NaN, infinite, empty, or structurally invalid.
    """
    # 1. Validate observed_estimate
    if observed_estimate is None:
        raise ValueError("observed_estimate cannot be None.")
    if isinstance(observed_estimate, bool):
        raise ValueError("observed_estimate cannot be a boolean.")
    if not isinstance(observed_estimate, (int, float)):
        raise ValueError(
            f"observed_estimate must be a numeric float or int, got {type(observed_estimate).__name__}."
        )
    if math.isnan(observed_estimate) or math.isinf(observed_estimate):
        raise ValueError("observed_estimate cannot be NaN or infinite.")
    obs_est = float(observed_estimate)

    # 2. Validate bias_values
    if bias_values is None:
        raise ValueError("bias_values cannot be None.")
    if isinstance(bias_values, (int, float)) and not isinstance(bias_values, bool):
        if math.isnan(bias_values) or math.isinf(bias_values):
            raise ValueError("bias_values cannot be NaN or infinite.")
        biases = [float(bias_values)]
    elif isinstance(bias_values, (list, tuple, np.ndarray, Sequence)):
        if len(bias_values) == 0:
            raise ValueError("bias_values cannot be empty.")
        biases = []
        for idx, b in enumerate(bias_values):
            if b is None or isinstance(b, bool) or not isinstance(b, (int, float)):
                raise ValueError(
                    f"bias_values element at index {idx} ({b!r}) must be a numeric float or int."
                )
            if math.isnan(b) or math.isinf(b):
                raise ValueError(f"bias_values element at index {idx} cannot be NaN or infinite.")
            biases.append(float(b))
    else:
        raise ValueError(
            f"bias_values must be a numeric float/int or sequence of numbers, got {type(bias_values).__name__}."
        )

    # 3. Validate confidence_interval
    ci_validated: Optional[List[float]] = None
    if confidence_interval is not None:
        if not isinstance(confidence_interval, (list, tuple, np.ndarray, Sequence)):
            raise ValueError(
                f"confidence_interval must be a list or tuple of 2 numbers, got {type(confidence_interval).__name__}."
            )
        if len(confidence_interval) != 2:
            raise ValueError(
                f"confidence_interval must contain exactly 2 elements [lower, upper], got {len(confidence_interval)}."
            )
        lower, upper = confidence_interval[0], confidence_interval[1]
        for val, name in [(lower, "lower"), (upper, "upper")]:
            if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"confidence_interval {name} bound ({val!r}) must be a numeric float or int.")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"confidence_interval {name} bound cannot be NaN or infinite.")
        l_val, u_val = float(lower), float(upper)
        if l_val > u_val:
            raise ValueError(f"confidence_interval lower bound ({l_val}) cannot exceed upper bound ({u_val}).")
        ci_validated = [l_val, u_val]

    # 4. Compute scenarios
    scenarios = []
    for b in biases:
        adj_est = float(obs_est - b)
        if ci_validated is not None:
            adj_l = float(ci_validated[0] - b)
            adj_u = float(ci_validated[1] - b)
            adj_ci = [adj_l, adj_u]
            inc_zero = bool(adj_l <= 0.0 <= adj_u)
        else:
            adj_ci = None
            inc_zero = None

        scenarios.append({
            "hypothetical_bias": b,
            "adjusted_estimate": adj_est,
            "adjusted_confidence_interval": adj_ci,
            "interval_includes_zero": inc_zero,
        })

    sign_convention_desc = (
        "tau_adjusted(b) = tau_observed - b. "
        "Positive bias (b > 0) assumes unmeasured confounding inflated the observed effect; "
        "negative bias (b < 0) assumes unmeasured confounding suppressed the observed effect."
    )

    num_scenarios = len(scenarios)
    if ci_validated is not None:
        zero_crossings = [s["hypothetical_bias"] for s in scenarios if s["interval_includes_zero"]]
        if zero_crossings:
            ci_note = (
                f"Under {len(zero_crossings)} of {num_scenarios} scenario(s) (bias={zero_crossings}), "
                "the adjusted confidence interval spans zero, indicating that the null effect cannot be ruled out."
            )
        else:
            ci_note = "Under none of the evaluated scenarios does the adjusted confidence interval span zero."
    else:
        ci_note = "No confidence interval was provided for interval coverage evaluation."

    interpretation_text = (
        f"Sensitivity analysis evaluated {num_scenarios} hypothetical unmeasured confounding scenario(s) "
        f"for an observed estimate of {obs_est:.4f}. {ci_note} "
        "Note: hypothetical biases represent user-specified assumptions rather than empirical estimates."
    )

    assumptions_text = (
        "Hypothetical bias-adjustment sensitivity analysis explores whether qualitative conclusions "
        "are robust to potential unmeasured confounding. Key assumptions and limitations: "
        "(1) Scenario-based: the bias parameter (b) is an assumed hypothetical quantity, NOT an "
        "empirically identified statistic; "
        "(2) Additive structure: unmeasured confounding is assumed to act additively on the treatment effect scale; "
        "(3) Informational role: sensitivity analysis illustrates how findings change under specific violations "
        "of exchangeability, but cannot prove that unmeasured confounding is absent or bounded; "
        "(4) Inference disclaimer: shifted intervals are conditional on the truth of the assumed bias; "
        "no statistical p-values are calculated."
    )

    return {
        "observed_estimate": obs_est,
        "confidence_interval": ci_validated,
        "bias_scenarios": scenarios,
        "sign_convention": sign_convention_desc,
        "interpretation": interpretation_text,
        "assumptions": assumptions_text,
    }


def run_causal_sensitivity_analysis(
    data: Any,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: Optional[Sequence[str]] = None,
    treated_value: Optional[Any] = None,
    control_value: Optional[Any] = None,
    drop_missing: bool = False,
    reference_estimator: str = "regression_adjusted",
    truncation_thresholds: Optional[Sequence[float]] = None,
    hypothetical_biases: Optional[Union[float, Sequence[float]]] = None,
    untreated_value: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Perform causal sensitivity analysis across estimators, IPTW truncation, and unmeasured confounding.

    This function conducts a unified robustness check covering three complementary dimensions:
    1. Multi-Estimator Comparison (Part A):
       Compares point estimates across available estimators: Unadjusted Difference in Means,
       Regression-Adjusted ATE (G-computation), and IPTW ATE.
    2. Weight Truncation Sensitivity (Part B):
       Evaluates the stability and trade-offs of IPTW weights across caller-specified truncation
       thresholds, reporting point estimates, effective sample sizes, and affected counts.
    3. Unmeasured Confounding Sensitivity (Part C):
       Applies a transparent hypothetical bias-adjustment model to the reference estimate under
       user-supplied bias scenarios.

    Parameters:
    -----------
    data : pd.DataFrame, Dict[str, Sequence], or List[Dict[str, Any]]
        The tabular dataset to analyze.
    treatment_col : str
        Name of the binary treatment column.
    outcome_col : str
        Name of the numeric outcome column.
    confounder_cols : Sequence[str], optional
        Observed confounder column names. If omitted or empty, only unadjusted ATE is evaluated.
    treated_value : Any, optional
        Label identifying treated observations.
    control_value : Any, optional
        Label identifying control observations.
    drop_missing : bool, default False
        Explicit missing data policy (reused from validate_causal_data).
    reference_estimator : str, default 'regression_adjusted'
        Primary estimator used as the baseline reference point. One of:
        'regression_adjusted', 'iptw', or 'unadjusted'.
    truncation_thresholds : Sequence[float], optional
        Optional sequence of positive finite thresholds (e.g. [10.0, 5.0, 3.0]) for IPTW weight
        truncation robustness checks. Weights are never truncated unless explicitly requested.
    hypothetical_biases : float or Sequence[float], optional
        Optional hypothetical signed bias values for unmeasured confounding sensitivity analysis.
    untreated_value : Any, optional
        Alias for control_value.

    Returns:
    --------
    Dict[str, Any]
        Structured dictionary containing:
        - 'reference_estimator': Name of the reference estimator used
        - 'reference_estimate': Point estimate from the reference estimator
        - 'observed_estimate': Alias for reference_estimate
        - 'reference_confidence_interval': CI from reference estimator if available, else None
        - 'estimator_comparisons': Dictionary of results for unadjusted, regression_adjusted, and iptw
        - 'iptw_truncation_analysis': Results across requested truncation thresholds
        - 'unmeasured_confounding_analysis': Sensitivity results from analyze_unmeasured_confounding_sensitivity
        - 'treatment_col': str
        - 'outcome_col': str
        - 'confounder_cols': list of str
        - 'warnings': list of warning messages
        - 'interpretation': narrative summary of multi-estimator and sensitivity findings
        - 'assumptions': educational explanation of causal identification and limitations
    """
    if control_value is None and untreated_value is not None:
        control_value = untreated_value

    valid_ref_estimators = {"regression_adjusted", "iptw", "unadjusted"}
    ref_est_normalized = reference_estimator.lower().strip()
    if ref_est_normalized not in valid_ref_estimators:
        raise ValueError(
            f"Unsupported reference_estimator: {reference_estimator!r}. "
            f"Expected one of: {sorted(list(valid_ref_estimators))}."
        )

    warnings_list: List[str] = []

    # 1. Unadjusted Difference in Means
    unadj_info: Dict[str, Any] = {
        "name": "Unadjusted Difference in Means",
        "available": False,
        "estimate": None,
        "difference_from_reference": None,
        "confidence_interval": None,
        "reason": None,
    }
    unadj_res = None
    try:
        unadj_res = estimate_unadjusted_ate(
            data=data,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            treated_value=treated_value,
            untreated_value=control_value,
            drop_missing=drop_missing,
            compute_ci=True,
        )
        unadj_info["available"] = True
        unadj_info["estimate"] = float(unadj_res["ate"])
        unadj_info["confidence_interval"] = unadj_res.get("confidence_interval")
        unadj_info["n_total"] = unadj_res["n_total"]
        unadj_info["mean_treated"] = unadj_res["mean_outcome_treated"]
        unadj_info["mean_control"] = unadj_res["mean_outcome_control"]
    except Exception as exc:
        unadj_info["available"] = False
        unadj_info["reason"] = str(exc)

    # 2. Regression-Adjusted ATE
    reg_info: Dict[str, Any] = {
        "name": "Regression-Adjusted ATE (G-computation)",
        "available": False,
        "estimate": None,
        "difference_from_reference": None,
        "reason": None,
    }
    reg_res = None
    has_confounders = (confounder_cols is not None) and (len(confounder_cols) > 0)
    if not has_confounders:
        reg_info["reason"] = "Confounder columns not provided. Regression adjustment requires at least one confounder."
    else:
        try:
            reg_res = estimate_regression_adjusted_ate(
                data=data,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                confounder_cols=confounder_cols,
                treated_value=treated_value,
                untreated_value=control_value,
                drop_missing=drop_missing,
            )
            reg_info["available"] = True
            reg_info["estimate"] = float(reg_res["adjusted_ate"])
            reg_info["model_type"] = reg_res.get("model_type")
        except Exception as exc:
            reg_info["available"] = False
            reg_info["reason"] = str(exc)

    # 3. IPTW ATE (untruncated)
    iptw_info: Dict[str, Any] = {
        "name": "Inverse Probability of Treatment Weighting (IPTW)",
        "available": False,
        "estimate": None,
        "difference_from_reference": None,
        "ess_treated": None,
        "ess_control": None,
        "ess_total": None,
        "reason": None,
    }
    iptw_untrunc_res = None
    if not has_confounders:
        iptw_info["reason"] = "Confounder columns not provided. Propensity score modeling and IPTW require at least one confounder."
    else:
        try:
            iptw_untrunc_res = estimate_iptw_ate(
                data=data,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                confounder_cols=confounder_cols,
                treated_value=treated_value,
                control_value=control_value,
                drop_missing=drop_missing,
                truncate_weights=False,
            )
            iptw_info["available"] = True
            iptw_info["estimate"] = float(iptw_untrunc_res["iptw_ate"])
            iptw_info["ess_treated"] = iptw_untrunc_res["ess_treated"]
            iptw_info["ess_control"] = iptw_untrunc_res["ess_control"]
            iptw_info["ess_total"] = iptw_untrunc_res["ess_total"]
            iptw_info["weight_summary"] = iptw_untrunc_res["weight_summary"]
        except Exception as exc:
            iptw_info["available"] = False
            iptw_info["reason"] = str(exc)

    # 4. Determine Reference Estimator & Reference Estimate
    ref_estimate: Optional[float] = None
    ref_ci: Optional[List[float]] = None
    chosen_ref = ref_est_normalized

    if chosen_ref == "regression_adjusted":
        if reg_info["available"]:
            ref_estimate = reg_info["estimate"]
        else:
            if iptw_info["available"]:
                chosen_ref = "iptw"
                ref_estimate = iptw_info["estimate"]
                warnings_list.append(
                    f"Requested reference estimator 'regression_adjusted' was unavailable ({reg_info['reason']}); "
                    "fell back to 'iptw'."
                )
            elif unadj_info["available"]:
                chosen_ref = "unadjusted"
                ref_estimate = unadj_info["estimate"]
                ref_ci = unadj_info["confidence_interval"]
                warnings_list.append(
                    f"Requested reference estimator 'regression_adjusted' was unavailable ({reg_info['reason']}); "
                    "fell back to 'unadjusted'."
                )
            else:
                raise ValueError(
                    f"All causal estimators failed. Unadjusted error: {unadj_info['reason']}; "
                    f"Regression error: {reg_info['reason']}."
                )
    elif chosen_ref == "iptw":
        if iptw_info["available"]:
            ref_estimate = iptw_info["estimate"]
        else:
            if reg_info["available"]:
                chosen_ref = "regression_adjusted"
                ref_estimate = reg_info["estimate"]
                warnings_list.append(
                    f"Requested reference estimator 'iptw' was unavailable ({iptw_info['reason']}); "
                    "fell back to 'regression_adjusted'."
                )
            elif unadj_info["available"]:
                chosen_ref = "unadjusted"
                ref_estimate = unadj_info["estimate"]
                ref_ci = unadj_info["confidence_interval"]
                warnings_list.append(
                    f"Requested reference estimator 'iptw' was unavailable ({iptw_info['reason']}); "
                    "fell back to 'unadjusted'."
                )
            else:
                raise ValueError(
                    f"All causal estimators failed. Unadjusted error: {unadj_info['reason']}; "
                    f"IPTW error: {iptw_info['reason']}."
                )
    else:  # unadjusted
        if unadj_info["available"]:
            ref_estimate = unadj_info["estimate"]
            ref_ci = unadj_info["confidence_interval"]
        else:
            raise ValueError(f"Unadjusted estimator failed: {unadj_info['reason']}.")

    # Calculate differences from reference
    for info in [unadj_info, reg_info, iptw_info]:
        if info["available"] and ref_estimate is not None:
            info["difference_from_reference"] = float(info["estimate"] - ref_estimate)

    estimator_comparisons = {
        "unadjusted": unadj_info,
        "regression_adjusted": reg_info,
        "iptw": iptw_info,
    }

    # 5. IPTW Weight Truncation Robustness (Part B)
    truncation_analysis: Dict[str, Any] = {
        "status": "not_requested",
        "untruncated_estimate": iptw_info.get("estimate"),
        "scenarios": [],
        "note": "No truncation thresholds were requested. Weights were not truncated by default.",
    }

    if truncation_thresholds is not None:
        if not isinstance(truncation_thresholds, (list, tuple, np.ndarray, Sequence)):
            raise ValueError(
                f"truncation_thresholds must be a sequence of positive numbers, got {type(truncation_thresholds).__name__}."
            )
        if len(truncation_thresholds) == 0:
            raise ValueError("truncation_thresholds sequence cannot be empty when provided.")

        validated_thresholds = []
        for idx, thresh in enumerate(truncation_thresholds):
            if thresh is None or isinstance(thresh, bool) or not isinstance(thresh, (int, float)):
                raise ValueError(
                    f"truncation_thresholds element at index {idx} ({thresh!r}) must be a numeric float or int."
                )
            if math.isnan(thresh) or math.isinf(thresh):
                raise ValueError(f"truncation_thresholds element at index {idx} cannot be NaN or infinite.")
            if float(thresh) <= 0.0:
                raise ValueError(
                    f"truncation_thresholds element at index {idx} must be strictly positive, got {thresh}."
                )
            validated_thresholds.append(float(thresh))

        if not iptw_info["available"]:
            truncation_analysis = {
                "status": "unavailable",
                "untruncated_estimate": None,
                "scenarios": [],
                "reason": f"IPTW estimator unavailable: {iptw_info['reason']}",
                "note": "Cannot evaluate weight truncation when baseline IPTW estimation fails.",
            }
        else:
            scenarios = []
            untrunc_est = iptw_info["estimate"]
            for thresh in validated_thresholds:
                t_res = estimate_iptw_ate(
                    data=data,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    confounder_cols=confounder_cols,
                    treated_value=treated_value,
                    control_value=control_value,
                    drop_missing=drop_missing,
                    truncate_weights=True,
                    max_weight=thresh,
                )
                t_est = float(t_res["iptw_ate"])
                diff = float(t_est - untrunc_est)
                n_trunc = int(t_res["weight_summary"].get("rows_truncated", t_res["weight_summary"].get("weights_truncated_count", 0)))
                max_w = float(t_res["weight_summary"].get("max_weight", t_res["weight_summary"].get("max", thresh)))
                is_ident = bool(np.isclose(t_est, untrunc_est, atol=1e-12))

                scenarios.append({
                    "threshold": thresh,
                    "estimate": t_est,
                    "difference_from_untruncated": diff,
                    "weights_truncated_count": n_trunc,
                    "max_weight": max_w,
                    "ess_treated": t_res["ess_treated"],
                    "ess_control": t_res["ess_control"],
                    "ess_total": t_res["ess_total"],
                    "identical_to_untruncated": is_ident,
                })

            truncation_analysis = {
                "status": "evaluated",
                "untruncated_estimate": untrunc_est,
                "scenarios": scenarios,
                "note": (
                    "Weight truncation caps extreme weights at a specified upper threshold, "
                    "reducing sample variance at the potential cost of introducing finite-sample bias."
                ),
            }

    # 6. Unmeasured Confounding Sensitivity Analysis (Part C)
    unmeasured_analysis: Optional[Dict[str, Any]] = None
    if hypothetical_biases is not None:
        unmeasured_analysis = analyze_unmeasured_confounding_sensitivity(
            observed_estimate=ref_estimate,
            bias_values=hypothetical_biases,
            confidence_interval=ref_ci,
        )
    else:
        unmeasured_analysis = {
            "status": "not_requested",
            "note": "No hypothetical bias values provided. Pass hypothetical_biases to evaluate unmeasured confounding scenarios.",
        }

    # 7. Multi-Estimator Agreement Check & Warnings
    available_estimates = [
        info["estimate"] for info in estimator_comparisons.values() if info["available"]
    ]
    if len(available_estimates) > 1:
        spread = float(max(available_estimates) - min(available_estimates))
        warnings_list.append(
            f"Estimator agreement check: {len(available_estimates)} estimators available. "
            f"Point estimate spread: {spread:.4f}. Agreement between estimators reflects consistency "
            "across specified models, but does NOT prove absence of unmeasured confounding or model validity."
        )

    # 8. Synthesis & Interpretation Text
    available_summary = ", ".join(
        f"{name}: {info['estimate']:.4f}"
        for name, info in [
            ("Unadjusted", unadj_info),
            ("Regression-Adjusted", reg_info),
            ("IPTW", iptw_info),
        ]
        if info["available"]
    )
    interpretation_text = (
        f"Causal sensitivity analysis evaluated reference estimator '{chosen_ref}' with estimate {ref_estimate:.4f}. "
        f"Available estimates ({available_summary}). "
        f"Weight truncation status: {truncation_analysis['status']}. "
        f"Unmeasured confounding status: {unmeasured_analysis.get('status', 'evaluated')}."
    )

    assumptions_text = (
        "Causal sensitivity analysis explores robustness to analytical decisions and hypothetical unmeasured confounding. "
        "Core principles: (1) Multi-estimator consistency: different estimators target the same ATE under different "
        "functional forms (outcome modeling vs. propensity score weighting). Divergence indicates model misspecification "
        "or overlap issues; agreement does NOT guarantee causal validity. "
        "(2) Weight truncation: truncation trades variance reduction against bias; severe truncation alters the effective target estimand. "
        "(3) Unmeasured confounding: sensitivity parameters are hypothetical assumptions that explore robustness bounds; "
        "sensitivity analysis does NOT prove absence of unmeasured confounding or causal validity."
    )

    clean_confounders = list(confounder_cols) if confounder_cols is not None else []

    return {
        "reference_estimator": chosen_ref,
        "reference_estimate": ref_estimate,
        "observed_estimate": ref_estimate,
        "reference_confidence_interval": ref_ci,
        "estimator_comparisons": estimator_comparisons,
        "iptw_truncation_analysis": truncation_analysis,
        "unmeasured_confounding_analysis": unmeasured_analysis,
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "confounder_cols": clean_confounders,
        "warnings": warnings_list,
        "interpretation": interpretation_text,
        "assumptions": assumptions_text,
    }
