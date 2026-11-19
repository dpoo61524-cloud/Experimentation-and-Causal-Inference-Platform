# Test Results and Verification Report

**Platform:** Experimentation and Causal Inference Platform  
**Repository:** `https://github.com/dpoo61524-cloud/Experimentation-and-Causal-Inference-Platform`  
**Branch:** `main`  
**Execution Environment:** Windows (win32), Python 3.11.9, pytest 9.2.0, anyio 4.16.0  
**Verification Date:** November 19, 2026  

---

## 1. Executive Summary

A comprehensive automated test suite execution was performed across the entire repository. The platform implements A/B testing engines, group-sequential design, Bayesian inference, and an observational causal inference suite backed by a streaming CSV dataset manager and a FastAPI service.

### Overall Execution Metrics

| Metric | Result |
| :--- | :--- |
| **Total Test Items Collected** | 612 |
| **Passed Tests** | 611 |
| **Failed Tests** | 1 |
| **Warnings** | 3 |
| **Execution Duration** | 13.66 seconds |
| **Overall Pass Rate** | 99.84% |

---

## 2. Test Suite Breakdown by Module

The test suite covers statistical routines, Bayesian modeling, causal estimators, dataset management, and HTTP API endpoints.

| Test File | Tests | Passed | Failed | Status | Primary Coverage Area |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `tests/test_api.py` | 242 | 242 | 0 | PASSED | REST endpoints, OpenAPI schemas, parameter aliases, validation error handling |
| `tests/test_bayesian_engine.py` | 68 | 68 | 0 | PASSED | Beta-Binomial conjugate updating, exact Beta intervals, Monte Carlo simulations |
| `tests/test_causal_engine.py` | 129 | 128 | 1 | FAILED | Unadjusted ATE, OLS regression adjustment, propensity scores, IPTW, DiD, sensitivity |
| `tests/test_dataset_did.py` | 16 | 16 | 0 | PASSED | Dataset-backed 2x2 Difference-in-Differences and parallel trends assumptions |
| `tests/test_dataset_ingestion.py` | 18 | 18 | 0 | PASSED | Chunked streaming CSV reader, type lattice (Bool &rarr; Int &rarr; Num &rarr; Cat), TTL, bounded storage |
| `tests/test_dataset_iptw.py` | 14 | 14 | 0 | PASSED | Dataset-backed Inverse Probability of Treatment Weighting and balance diagnostics |
| `tests/test_dataset_propensity_scores.py` | 14 | 14 | 0 | PASSED | Dataset-backed propensity score estimation and common support diagnostics |
| `tests/test_dataset_regression_adjusted_ate.py` | 13 | 13 | 0 | PASSED | Dataset-backed regression-adjusted ATE (G-computation) |
| `tests/test_dataset_sensitivity_analysis.py` | 13 | 13 | 0 | PASSED | Dataset-backed sensitivity analysis and hypothetical bias bounds |
| `tests/test_dataset_unadjusted_ate.py` | 12 | 12 | 0 | PASSED | Dataset-backed difference-in-means treatment effect estimation |
| `tests/test_engine_validation.py` | 8 | 8 | 0 | PASSED | Data validation rules, column checks, missing value policy |
| `tests/test_experiment_report.py` | 3 | 3 | 0 | PASSED | Formatted terminal text report generation |
| `tests/test_final_integration_qa.py` | 7 | 7 | 0 | PASSED | End-to-end integration, dashboard serving, multi-dataset isolation, benchmark load |
| `tests/test_multiple_comparisons_engine.py` | 7 | 7 | 0 | PASSED | Bonferroni and Benjamini-Hochberg (FDR) adjustments |
| `tests/test_report_export.py` | 4 | 4 | 0 | PASSED | CSV and text report export utilities |
| `tests/test_sequential_testing.py` | 12 | 12 | 0 | PASSED | Discrete O'Brien–Fleming (1979) boundaries, schedule enforcement, futility heuristics |
| `tests/test_statistics_engine.py` | 26 | 26 | 0 | PASSED | Two-proportion z-tests, Wilson score intervals, Welch's t-test, Chi-square independence |
| **Total** | **612** | **611** | **1** | **99.84%** | |

---

## 3. Detailed Failure Analysis

### Failed Test: `tests/test_causal_engine.py::test_iptw_reproducible_output`

- **Test Location:** `tests/test_causal_engine.py:1827`
- **Exception Raised:** `ValueError: Failed to fit propensity score model (logistic regression): Singular matrix`
- **Failure Traceback Summary:**
  ```python
  def test_iptw_reproducible_output():
      """Verify that repeated runs on deterministic data yield identical outputs."""
      df = pd.DataFrame({
          "t": [1, 1, 1, 0, 0, 0],
          "x1": [2.5, 3.5, 4.5, 1.5, 2.5, 3.5],
          "x2": [10.0, 12.0, 14.0, 8.0, 10.0, 12.0],
          "y": [50.0, 55.0, 60.0, 30.0, 35.0, 40.0],
      })
      res1 = estimate_iptw_ate(df, "t", "y", ["x1", "x2"])
      ...
  ```
- **Root Cause Analysis:**
  The test dataset specifies synthetic confounders `x1` and `x2` where $x_2 = 2 \cdot x_1 + 5.0$ across all 6 rows:
  - Row 0: $x_1 = 2.5 \implies 2(2.5) + 5 = 10.0$ ($x_2 = 10.0$)
  - Row 1: $x_1 = 3.5 \implies 2(3.5) + 5 = 12.0$ ($x_2 = 12.0$)
  - Row 2: $x_1 = 4.5 \implies 2(4.5) + 5 = 14.0$ ($x_2 = 14.0$)
  - Row 3: $x_1 = 1.5 \implies 2(1.5) + 5 = 8.0$ ($x_2 = 8.0$)
  - Row 4: $x_1 = 2.5 \implies 2(2.5) + 5 = 10.0$ ($x_2 = 10.0$)
  - Row 5: $x_1 = 3.5 \implies 2(3.5) + 5 = 12.0$ ($x_2 = 12.0$)

  Because `x2` is an exact linear function of `x1`, the design matrix is collinear (rank deficient). The underlying logistic regression fitting routine (`statsmodels.api.Logit`) fails with a singular matrix error, which `estimate_propensity_scores` accurately detects and handles by raising a descriptive `ValueError`. The engine's mathematical exception handling functions as designed; the failure is isolated to this specific synthetic test case specification.

---

## 4. Warnings Summary

The test run generated 3 runtime/deprecation warnings:

1. **Starlette Deprecation Warning:**
   - `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.`
   - Occurs during client initialization in `fastapi.testclient`.
2. **Moment Calculation Precision Loss Warnings (2 instances):**
   - `RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.`
   - Triggered in `scipy.stats._axis_nan_policy` during constant value input validation in `test_api.py::test_ttest_constant_values_where_supported` and `test_statistics_engine.py::test_two_sample_ttest_constant_values_where_supported`.

---

## 5. Scope, Methodological Boundaries, and Known Limitations

To ensure methodological and academic transparency, the following technical and statistical boundaries are verified and documented:

1. **Single-Node Storage Lifecycle:**
   - Ingested datasets are cached to temporary local disk with a 24-hour TTL and a maximum storage limit of 50 datasets / 500 MB.
   - Workloads are verified on datasets up to 64,000 rows (the Hillstrom benchmark). Distributed data stores (e.g., Spark, cloud warehouses) are not supported.
2. **Propensity Score Diagnostics vs. Causal Effects:**
   - Propensity scores $e(X)$ are diagnostic measures of covariate overlap and common support, not causal effect estimates themselves.
3. **Causal Sensitivity Bounds:**
   - Sensitivity analysis evaluates hypothetical confounding bias bounds $b$; it does not prove that unmeasured confounding is absent.
4. **Unimplemented Advanced Estimators:**
   - The platform does not implement Ding & VanderWeele E-values, Rosenbaum Wilcoxon bounds, or panel clustered standard errors.
5. **Group-Sequential Design:**
   - Group-sequential testing uses discrete O'Brien–Fleming (1979) boundaries controlling overall Type I error across planned interim checkpoints.
   - Non-binding futility is a directional decision aid, not formal conditional power or proof of null effect.

---

## 6. Verification Conclusion

The test suite thoroughly validates 611 passing scenarios covering statistical correctness, input validation, boundary condition handling, schema compliance, and end-to-end user workflows.
