# Experimentation and Causal Inference Platform

A Python platform for A/B testing, causal inference, and statistical analysis, featuring statistical comparison engines, sample size calculators, automated experiment reporting, and a FastAPI backend service.

---

## Features

- **Conversion Rate Comparison Engine**: Two-proportion z-test, Wilson confidence intervals, and p-value calculations for binary metrics.
- **Independent Two-Sample Welch's t-Test**: Hypothesis testing for continuous metrics (e.g., revenue per user, pages viewed, session duration) without assuming equal variances.
- **Chi-Square Test of Independence**: Pearson's chi-square test on $r \times c$ contingency tables for categorical outcomes, complete with expected frequency evaluation and small-sample warnings.
- **Group-Sequential Testing Engine**: Discrete O'Brien–Fleming (1979) design with exact canonical multivariate normal boundary calculation across $K \in \{2, 3, 4, 5\}$ interim looks to control overall Type I error ($\alpha \in \{0.01, 0.05, 0.10\}$). Features schedule enforcement, planned vs observed information fraction auditing, non-binding futility heuristics, and statistically sound decisions without false "proof of no effect" claims.
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

### 7. Group-Sequential Testing Endpoint (O'Brien–Fleming Conversion Test)
- **URL**: `POST /api/experiment/sequential`
- **Description**: Evaluates an ongoing conversion-rate A/B experiment at an interim or final planned checkpoint ($k \in \{1, \dots, K\}$) using a discrete O'Brien–Fleming (1979) group-sequential design.
- **The "Peeking Problem" & Type I Error Inflation**:
  - In continuous experimentation, repeatedly testing standard unadjusted p-values as data accumulates causes severe Type I error inflation. For instance, peeking 5 times at nominal $\alpha = 0.05$ inflates the true false positive rate to $\approx 14.2\%$; continuous monitoring can inflate it beyond $25\%$.
  - Group-sequential designs solve this by pre-specifying $K$ equally spaced looks and adjusting the critical boundary at each look so that the **cumulative boundary-crossing probability across all $K$ looks under $H_0$ equals exactly $\alpha$**.
- **Canonical Joint-Normal Boundary Formulation**:
  - For $K$ equally spaced looks ($t_k = k/K$), the sequential score statistics follow a canonical Gaussian Markov process with covariance:
    $$\operatorname{Cov}(Z_j, Z_k) = \sqrt{\frac{\min(j, k)}{\max(j, k)}} \quad \text{for } 1 \le j, k \le K$$
  - The discrete O'Brien–Fleming boundary at look $k$ is given by:
    $$b_k = c_K(\alpha) \sqrt{\frac{K}{k}}$$
    where the constant $c_K(\alpha)$ is numerically solved such that the multidimensional normal continuation probability satisfies:
    $$P_{H_0}\left(\bigcap_{k=1}^K \{-b_k < Z_k < b_k\}\right) = 1 - \alpha$$
- **Independently Verified Boundary Table**:
  Every supported pair of $K \in \{2, 3, 4, 5\}$ and $\alpha \in \{0.01, 0.05, 0.10\}$ has been numerically solved using multivariate normal integration and verified to control overall Type I error within $1 \times 10^{-4}$ tolerance:

| Overall $\alpha$ | Planned Looks $K$ | Solver Constant $c_K$ | Boundary Vector $[b_1, \dots, b_K]$ | Crossing Prob $P(\text{cross})$ | Numerical Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05** | 2 | 1.977431 | `[2.797, 1.977]` | 0.050000 | $3.5 \times 10^{-10}$ |
| **0.05** | 3 | 2.004026 | `[3.471, 2.454, 2.004]` | 0.050002 | $2.0 \times 10^{-6}$ |
| **0.05** | 4 | 2.024315 | `[4.049, 2.863, 2.337, 2.024]` | 0.050002 | $2.0 \times 10^{-6}$ |
| **0.05** | **5** | **2.040062** | **`[4.562, 3.226, 2.634, 2.281, 2.040]`** | **0.050003** | **$2.9 \times 10^{-6}$** |
| **0.01** | 2 | 2.579570 | `[3.648, 2.580]` | 0.010000 | $5.3 \times 10^{-12}$ |
| **0.01** | 3 | 2.594794 | `[4.494, 3.178, 2.595]` | 0.009988 | $1.2 \times 10^{-5}$ |
| **0.01** | 4 | 2.608938 | `[5.218, 3.690, 3.013, 2.609]` | 0.010007 | $7.3 \times 10^{-6}$ |
| **0.01** | 5 | 2.621068 | `[5.861, 4.144, 3.384, 2.930, 2.621]` | 0.010010 | $1.0 \times 10^{-5}$ |
| **0.10** | 2 | 1.677953 | `[2.373, 1.678]` | 0.100000 | $4.2 \times 10^{-12}$ |
| **0.10** | 3 | 1.709602 | `[2.961, 2.094, 1.710]` | 0.100001 | $1.3 \times 10^{-6}$ |
| **0.10** | 4 | 1.733084 | `[3.466, 2.451, 2.001, 1.733]` | 0.099999 | $8.2 \times 10^{-7}$ |
| **0.10** | 5 | 1.750862 | `[3.915, 2.768, 2.260, 1.958, 1.751]` | 0.100005 | $4.9 \times 10^{-6}$ |

- **Planned Schedule vs. Observed Information Fraction**:
  - The design assumes equal increments of information ($\Delta I \approx I_{\max}/K$).
  - **Planned Information Fraction**: $t_k = k / K$.
  - **Observed Information Fraction**: $I_{\text{obs}} = \frac{n_A + n_B}{2 N_{\max}}$.
  - **Schedule Enforcement**: If $|I_{\text{obs}} - t_k| > \text{schedule\_tolerance}$ (default $0.15$), the endpoint rejects the evaluation with HTTP 400. This prevents applying an equal-information boundary to unaligned sample sizes (such as peeking with 90% of data at look 1).
- **Statistically Grounded Decisions**:
  - **Interim Look ($k < K$)**:
    - `stop_for_efficacy`: $|Z| \ge Z_{\text{crit}}(k)$. Extreme early evidence; trial may stop early with rigorous Type I error control.
    - `continue`: $|Z| < Z_{\text{crit}}(k)$. Evidence is insufficient for early stopping; accumulate data to the next look.
  - **Final Look ($k = K$)**:
    - `stop_max_sample_size_efficacy_achieved`: Maximum planned sample size reached; $|Z| \ge Z_{\text{crit}}(K)$ ($p \le \alpha_{\text{final}}$). Statistically significant.
    - `stop_max_sample_size_inconclusive`: Maximum planned sample size reached; $|Z| < Z_{\text{crit}}(K)$. Fails to reject $H_0$. **Statistically documented as inconclusive evidence, NOT proof of no effect.**
- **Non-Binding Directional Futility**:
  - An optional heuristic (`evaluate_futility: true`) at or after the halfway checkpoint ($k \ge \lceil K/2 \rceil$).
  - If the treatment conversion rate is not exceeding control ($Z \le 0$), an advisory flag (`stop_for_futility`) is raised to save traffic.
  - **Critical Rule**: Futility is non-binding and does **NOT** prove absence of an effect.

**Example Request 1: Interim Checkpoint 1 of 5 (Early Efficacy Stopping)**:
```bash
curl -X POST http://127.0.0.1:8000/api/experiment/sequential \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint": 1,
    "total_checkpoints": 5,
    "conversions_a": 10,
    "visitors_a": 200,
    "conversions_b": 80,
    "visitors_b": 200,
    "max_visitors_per_group": 1000,
    "alpha": 0.05
  }'
```

**Response (200 OK)**:
```json
{
  "design": "O'Brien-Fleming Group Sequential Design",
  "methodology": "Discrete canonical joint-normal boundary calculation",
  "checkpoint": 1,
  "total_checkpoints": 5,
  "planned_information_fraction": 0.2,
  "observed_information_fraction": 0.2,
  "schedule_tolerance": 0.15,
  "schedule_compliant": true,
  "max_visitors_per_group": 1000,
  "conversions_a": 10,
  "visitors_a": 200,
  "conversion_rate_a": 0.05,
  "conversions_b": 80,
  "visitors_b": 200,
  "conversion_rate_b": 0.4,
  "absolute_difference": 0.35,
  "z_statistic": 8.7963,
  "nominal_p_value": 0.0,
  "efficacy_boundary_z": 4.5617,
  "nominal_alpha_threshold": 5.07e-06,
  "all_planned_boundaries_z": [4.5617, 3.2256, 2.6337, 2.2809, 2.0401],
  "efficacy_boundary_crossed": true,
  "maximum_sample_size_reached": false,
  "evaluate_futility": false,
  "futility_boundary_crossed": false,
  "futility_recommendation": null,
  "decision": "stop_for_efficacy",
  "overall_alpha": 0.05,
  "explanation": "Interim checkpoint 1/5: Two-sided test statistic |Z|=8.7963 crossed the O'Brien-Fleming efficacy boundary Z_crit=4.5617 (nominal p=0 <= 5.07e-06). The trial may stop early for efficacy while preserving the overall Type I error rate.",
  "assumptions": "Independent observations; CLT-based asymptotic normality of proportion test statistic; canonical joint-normal covariance for interim score statistics."
}
```

**Example Request 2: Final Checkpoint 5 of 5 (Inconclusive Result)**:
```bash
curl -X POST http://127.0.0.1:8000/api/experiment/sequential \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint": 5,
    "total_checkpoints": 5,
    "conversions_a": 100,
    "visitors_a": 1000,
    "conversions_b": 105,
    "visitors_b": 1000,
    "max_visitors_per_group": 1000,
    "alpha": 0.05
  }'
```

**Response (200 OK)**:
```json
{
  "design": "O'Brien-Fleming Group Sequential Design",
  "checkpoint": 5,
  "total_checkpoints": 5,
  "planned_information_fraction": 1.0,
  "observed_information_fraction": 1.0,
  "efficacy_boundary_crossed": false,
  "maximum_sample_size_reached": true,
  "decision": "stop_max_sample_size_inconclusive",
  "explanation": "Maximum planned sample size reached at final checkpoint 5/5. Test statistic |Z|=0.3643 did not cross the final critical boundary Z_crit=2.0401. The experiment fails to reject the null hypothesis at overall alpha=0.05; this represents inconclusive evidence, NOT proof that the treatment has no effect."
}
```

---

## Academic Viva & Exam Notes

### 1. What is the "peeking problem" in online A/B testing?
In naive A/B testing, experimenters monitor dashboards daily and stop whenever $p < 0.05$. Because random fluctuations produce temporary spikes in test statistics, testing multiple times creates multiple opportunities to commit a Type I error (false positive). Peeking 5 times inflates the true error from $5\%$ to $\approx 14.2\%$. Group-sequential testing mathematically bounds this overall false positive rate at the pre-specified $\alpha$.

### 2. Why does O'Brien–Fleming use conservative boundaries at early looks?
The O'Brien–Fleming boundary $b_k = c_K \sqrt{K/k}$ assigns very large critical values early on (e.g., $Z_{\text{crit}} \approx 4.562$ at look 1 of 5), requiring overwhelming evidence to stop early. This preserves almost all of the alpha for the final look ($Z_{\text{crit}} \approx 2.040$, very close to the fixed-sample $1.960$), ensuring minimal loss in statistical power if the experiment runs to completion.

### 3. Why must planned information fraction and observed information be distinguished?
Group-sequential boundaries are derived under an assumed information accumulation schedule (equal information increments $t_k = k/K$). If an experimenter evaluates look 1 when $90\%$ of traffic has arrived, using the look 1 boundary ($Z_{\text{crit}} \approx 4.562$) is statistically invalid. Conversely, applying look 5 boundary ($Z_{\text{crit}} \approx 2.040$) to $20\%$ of data causes uncontrolled alpha inflation. Schedule enforcement ensures that equal-information boundaries are only applied when the observed traffic matches design assumptions within a calibrated tolerance.

### 4. Why is failing to reject $H_0$ or calling futility NOT "proof of no effect"?
In classical frequentist hypothesis testing, failing to cross a critical boundary means the data is consistent with the null hypothesis, but it does **not** prove the null is true (absence of evidence is not evidence of absence). A real effect smaller than the minimum detectable effect may exist. Similarly, non-binding futility is an operational heuristic to avoid wasting resources on unpromising treatments; it carries no formal guarantee that the treatment effect is zero.

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
