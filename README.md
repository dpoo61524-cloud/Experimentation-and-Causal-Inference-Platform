# Experimentation and Causal Inference Platform

A Python platform for A/B testing, causal inference, and statistical analysis, featuring statistical comparison engines, sample size calculators, automated experiment reporting, and a FastAPI backend service.

---

## Features

- **Conversion Rate Comparison Engine**: Two-proportion z-test, Wilson confidence intervals, and p-value calculations for binary metrics.
- **Independent Two-Sample Welch's t-Test**: Hypothesis testing for continuous metrics (e.g., revenue per user, pages viewed, session duration) without assuming equal variances.
- **Chi-Square Test of Independence**: Pearson's chi-square test on $r \times c$ contingency tables for categorical outcomes, complete with expected frequency evaluation and small-sample warnings.
- **Sample Size Estimation**: Power analysis (`NormalIndPower`) to determine required sample size per group and total sample size given baseline rate, expected rate, alpha, and power.
- **Multiple Comparisons Corrections**: Family-wise error rate control using Bonferroni and Benjamini-Hochberg (FDR) adjustments.
- **Automated Experiment Reporting**: Human-readable terminal output and CSV exports of experiment findings.
- **FastAPI REST API**: High-performance backend endpoints exposing statistical testing and power analysis engines, complete with input validation and interactive OpenAPI documentation.

---

## Installation

### Prerequisites
- Python 3.10+ (tested with Python 3.14)
- `pip` package manager

### Setup

1. **Clone the repository** (if not already local):
   ```bash
   git clone https://github.com/dpoo61524-cloud/Experimentation-and-Causal-Inference-Platform.git
   cd Experimentation-and-Causal-Inference-Platform
   ```

2. **Create and activate a virtual environment**:
   - On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - On macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Starting the API

Start the FastAPI application using Uvicorn:

```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Alternatively, you can run the entrypoint script directly:
```bash
python src/api.py
```

The API will be accessible at:
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`
- **Alternative Documentation (ReDoc)**: `http://127.0.0.1:8000/redoc`

---

## API Endpoints & Example Requests

### 1. Root Endpoint
- **URL**: `GET /`
- **Description**: Returns application identity and purpose.

**Request (curl)**:
```bash
curl -X GET http://127.0.0.1:8000/
```

**Response (200 OK)**:
```json
{
  "name": "Experimentation and Causal Inference Platform API",
  "description": "API for A/B testing statistical analysis and sample size estimation.",
  "version": "1.0.0",
  "docs_url": "/docs"
}
```

---

### 2. Health Check Endpoint
- **URL**: `GET /health`
- **Description**: Returns the operational status of the service.

**Request (curl)**:
```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response (200 OK)**:
```json
{
  "status": "healthy"
}
```

---

### 3. Experiment Comparison Endpoint (Proportions)
- **URL**: `POST /api/experiment/compare`
- **Description**: Compares two experiment groups using a two-proportion z-test. Calculates conversion rates, absolute difference, p-value, 95% confidence interval, and statistical significance.
- **Validation**:
  - `visitors` must be greater than zero.
  - `conversions` must be non-negative.
  - `conversions` cannot exceed `visitors`.

**Request (curl)**:
```bash
curl -X POST http://127.0.0.1:8000/api/experiment/compare \
  -H "Content-Type: application/json" \
  -d '{
    "group_a": {
      "conversions": 267,
      "visitors": 21307
    },
    "group_b": {
      "conversions": 122,
      "visitors": 21306
    }
  }'
```

**Response (200 OK)**:
```json
{
  "conversion_rate_a": 0.01253109306800582,
  "conversion_rate_b": 0.005726086548389998,
  "absolute_difference": 0.006805006519615822,
  "z_statistic": 7.337834575827725,
  "p_value": 2.170669286708781e-13,
  "confidence_interval_95": [
    0.005005852504268612,
    0.008643807902409015
  ],
  "statistically_significant": true
}
```

---

### 4. Sample Size Estimation Endpoint
- **URL**: `POST /api/experiment/sample-size`
- **Description**: Calculates the required sample size per group and total sample size for an A/B test based on baseline rate, expected rate, significance level (`alpha`), and statistical power (`power`).
- **Validation**:
  - Probabilities (`baseline_rate`, `expected_rate`, `alpha`, `power`) must be strictly between 0 and 1.
  - `expected_rate` must differ from `baseline_rate`.

**Request (curl)**:
```bash
curl -X POST http://127.0.0.1:8000/api/experiment/sample-size \
  -H "Content-Type: application/json" \
  -d '{
    "baseline_rate": 0.10,
    "expected_rate": 0.12,
    "alpha": 0.05,
    "power": 0.80
  }'
```

**Response (200 OK)**:
```json
{
  "baseline_rate": 0.1,
  "expected_rate": 0.12,
  "absolute_effect": 0.02,
  "alpha": 0.05,
  "power": 0.8,
  "sample_size_per_group": 3835,
  "total_sample_size": 7670
}
```

---

### 5. Independent Two-Sample t-Test Endpoint
- **URL**: `POST /api/experiment/ttest`
- **Description**: Performs Welch's two-sample t-test on continuous numerical metrics across two independent groups. Welch's t-test does not assume equal population variances ($s_a^2 \ne s_b^2$).
- **Assumptions & Considerations**:
  - **Independence**: Observations within each group and between the two groups must be independent.
  - **Scale**: Data should be continuous numerical measurements (e.g. order value, revenue, session duration).
  - **Normality**: Requires approximately normally distributed populations, or sufficiently large samples ($N \ge 30$) where the Central Limit Theorem holds.
  - **Sample Size**: Each group must have at least 2 numerical observations.
- **Alternative Hypotheses**: Supported options are `"two-sided"` (default), `"greater"` ($\mu_a > \mu_b$), or `"less"` ($\mu_a < \mu_b$).

**Request (curl)**:
```bash
curl -X POST http://127.0.0.1:8000/api/experiment/ttest \
  -H "Content-Type: application/json" \
  -d '{
    "group_a": [12.0, 15.0, 14.0, 10.0, 13.0, 14.0, 16.0],
    "group_b": [9.0, 10.0, 11.0, 8.0, 12.0, 10.0, 9.0],
    "alternative": "two-sided",
    "alpha": 0.05
  }'
```

**Response (200 OK)**:
```json
{
  "t_statistic": 3.936479108111085,
  "p_value": 0.0025220582997136265,
  "sample_size_a": 7,
  "sample_size_b": 7,
  "mean_a": 13.428571428571429,
  "mean_b": 9.857142857142858,
  "difference_in_means": 3.571428571428571,
  "degrees_of_freedom": 10.541941677667106,
  "confidence_interval_95": [
    1.5639151755674603,
    5.578941967289682
  ],
  "statistically_significant": true,
  "alternative": "two-sided"
}
```

---

### 6. Chi-Square Test of Independence Endpoint
- **URL**: `POST /api/experiment/chi-square`
- **Description**: Performs Pearson's chi-square test of independence on a 2D contingency table of categorical observation counts.
- **Assumptions & Considerations**:
  - **Independence**: Each subject/observation must contribute to exactly one cell in the contingency table.
  - **Categorical Structure**: Mutually exclusive row and column classifications.
  - **Sample Size & Cell Frequencies**: Reliable chi-square approximation assumes expected frequencies of at least 5 in most cells. If any expected frequency is below 5, an explicit warning is provided.
  - **Table Dimensions**: Requires at least 2 rows and 2 columns ($r \ge 2, c \ge 2$) with non-negative counts and positive marginal totals.

**Request (curl)**:
```bash
curl -X POST http://127.0.0.1:8000/api/experiment/chi-square \
  -H "Content-Type: application/json" \
  -d '{
    "table": [
      [50, 30],
      [20, 40]
    ],
    "alpha": 0.05,
    "correction": true
  }'
```

**Response (200 OK)**:
```json
{
  "chi_square_statistic": 10.529166666666667,
  "degrees_of_freedom": 1,
  "p_value": 0.0011750518530845063,
  "expected_frequencies": [
    [40.0, 40.0],
    [30.0, 30.0]
  ],
  "statistically_significant": true,
  "alpha": 0.05,
  "warning": null
}
```

---

## Interactive Documentation

FastAPI generates interactive Swagger documentation automatically:
1. Start the API server: `uvicorn src.api:app --reload`
2. Open your browser and navigate to `http://127.0.0.1:8000/docs`.
3. Test endpoints directly with the "Try it out" button.

---

## Running Automated Tests

Run the full automated test suite (statistical engine unit tests + API integration tests) using pytest:

```bash
pytest
```

To run with detailed test names:
```bash
pytest -v
```

---

## Existing Analysis Scripts

To run the offline dataset analysis and reporting scripts:
```bash
# Analyze Hillstrom email dataset summary
python check_hillstrom.py

# Generate experiment report (text and CSV in reports/)
python main_report.py

# Run multiple comparisons correction analysis
python multiple_comparisons_analysis.py
```
