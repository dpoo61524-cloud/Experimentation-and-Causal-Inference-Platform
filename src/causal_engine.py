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
