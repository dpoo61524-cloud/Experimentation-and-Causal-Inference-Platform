# Experimentation and Causal Inference Platform

A Python platform for A/B testing, causal inference, and statistical analysis, featuring statistical comparison engines, sample size calculators, automated experiment reporting, and a FastAPI backend service.

---

## Features

- **Conversion Rate Comparison Engine**: Two-proportion z-test, Wilson confidence intervals, and p-value calculations for binary metrics.
- **Independent Two-Sample Welch's t-Test**: Hypothesis testing for continuous metrics (e.g., revenue per user, pages viewed, session duration) without assuming equal variances.
- **Chi-Square Test of Independence**: Pearson's chi-square test on $r \times c$ contingency tables for categorical outcomes, complete with expected frequency evaluation and small-sample warnings.
- **Group-Sequential Testing Engine**: Discrete O'Brien–Fleming (1979) design with exact canonical multivariate normal boundary calculation across $K \in \{2, 3, 4, 5\}$ interim looks to control overall Type I error ($\alpha \in \{0.01, 0.05, 0.10\}$). Features schedule enforcement, planned vs observed information fraction auditing, non-binding futility heuristics, and statistically sound decisions without false "proof of no effect" claims.
- **Bayesian A/B Testing Engine**: Beta-Binomial conjugate model for conversion rates with independent Beta priors (default Beta(1, 1)). Computes analytical posterior means, exact Beta credible intervals, posterior probability of superiority $P(B > A)$, probability of practical improvement $P(B - A \ge \delta)$ exceeding a minimum practical effect threshold, and reproducible Monte Carlo credible intervals for conversion differences.
- **Sample Size Estimation**: Power analysis (`NormalIndPower`) to determine required sample size per group and total sample size given baseline rate, expected rate, alpha, and power.
- **Multiple Comparisons Corrections**: Family-wise error rate control using Bonferroni and Benjamini-Hochberg (FDR) adjustments.
- **Automated Experiment Reporting**: Human-readable terminal output and CSV exports of experiment findings.
- **FastAPI REST API**: High-performance backend endpoints exposing statistical testing, power analysis, dataset storage, and causal inference engines, complete with robust schema validation, error handling, and OpenAPI documentation.
- **Interactive Web Dashboard**: Clean, responsive frontend interface (`/dashboard`) styled in a Charcoal & Sage Green theme, providing live API health checks, chunked CSV uploads, schema inspection, Hillstrom benchmark loading, and execution of all six causal inference workflows without external heavy JavaScript frameworks.
- **Dataset Storage & Ingestion Engine**: Chunked streaming CSV ingestion with single-pass type inference (boolean &rarr; integer &rarr; numeric &rarr; categorical), bounded memory, 500 MB upload limit, 50-dataset registry capacity, 24-hour TTL, and explicit dataset deletion.

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
- **Interactive Web Dashboard**: `http://127.0.0.1:8000/dashboard`
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

## Bayesian A/B Testing

### 1. What Bayesian A/B Testing Does

In classical (frequentist) hypothesis testing, experiments center on $p$-values and null hypotheses ($H_0$). A frequentist $p$-value calculates how unlikely the observed data would be if there were truly zero effect. However, practitioners and decision-makers often want direct answers to practical questions:
- *What is the probability that Treatment B is actually better than Control A?*
- *What is the probability that Treatment B beats Control A by at least a meaningful business threshold (e.g., 1 percentage point)?*
- *What is the credible range of true conversion rates given our data and assumptions?*

Bayesian A/B testing answers these questions directly by treating conversion rates as random variables characterized by probability distributions. Using prior information and observed experimental data, it calculates the **posterior distribution** for each group's conversion rate.

#### Key Concepts & Distinctions
- **Observed Conversion Rate**: The raw sample fraction ($c / n$). It reflects only the empirical data collected in the current sample and can fluctuate substantially when sample size $n$ is small.
- **Posterior Mean Conversion Rate**: The expected value of the true conversion rate under the posterior distribution:
  $$\mathbb{E}[p] = \frac{\alpha_{\text{post}}}{\alpha_{\text{post}} + \beta_{\text{post}}}$$
  Under a flat $\text{Beta}(1, 1)$ prior, this becomes $(c + 1) / (n + 2)$ (Laplace's rule of succession), providing gentle regularization toward 0.5 when sample sizes are small.
- **Posterior Probability of Superiority ($P(B > A)$)**: The posterior probability that Group B's true underlying conversion rate is strictly greater than Group A's ($P(p_B > p_A)$).
- **Probability of Meaningful Improvement ($P(B - A \ge \delta)$)**: The posterior probability that Group B outperforms Group A by at least a user-specified minimum practical effect threshold $\delta$ (`minimum_effect`).

> [!IMPORTANT]
> **Probabilistic Belief vs. Future Guarantee**: Posterior probabilities represent the state of belief about true parameters under the specified model and prior assumptions. They are **not guarantees of future business performance**, nor do they guarantee financial returns.

---

### 2. Beta-Binomial Model

The platform uses the conjugate **Beta-Binomial model** for conversion rate experiments:

1. **Likelihood**: In a conversion experiment with $n$ visitors and $c$ conversions, each visitor outcome is modeled as an independent Bernoulli trial. The cumulative conversions follow a Binomial likelihood:
   $$c \sim \text{Binomial}(n, p)$$
2. **Conjugate Prior**: The Beta distribution is the conjugate prior for the Binomial likelihood:
   $$p \sim \text{Beta}(\alpha_{\text{prior}}, \beta_{\text{prior}})$$
3. **Posterior Update**: Observing $c$ conversions and $n - c$ non-conversions updates the parameters in closed form:
   $$\alpha_{\text{post}} = \alpha_{\text{prior}} + c$$
   $$\beta_{\text{post}} = \beta_{\text{prior}} + (n - c)$$
   The resulting posterior distribution is:
   $$p \sim \text{Beta}(\alpha_{\text{post}}, \beta_{\text{post}})$$

#### Prior Parameter Interpretation
- $\alpha_{\text{prior}}$ represents prior pseudo-conversions ("successes").
- $\beta_{\text{prior}}$ represents prior pseudo-non-conversions ("failures").
- **Default Prior**: $\text{Beta}(1, 1)$, a uniform distribution over $[0, 1]$ providing an uninformative, flat initial belief that assigns equal prior density across all possible conversion rates.
- **Custom Priors**: Users can supply alternative positive hyperparameters ($\alpha, \beta > 0$) reflecting historical baseline rates (e.g., $\text{Beta}(10, 90)$ for a prior centered around 10%) or Jeffreys uninformative prior $\text{Beta}(0.5, 0.5)$. Prior parameters represent model weights, not literal customer records.

---

### 3. Endpoint Details

- **Method & Route**: `POST /api/experiment/bayesian`
- **Summary**: Bayesian A/B testing for conversion rates
- **Tags**: `["Experimentation"]`

#### Supported Request Parameters
| Parameter | Type | Required / Default | Constraints & Range | Description |
|:---|:---:|:---:|:---|:---|
| `conversions_a` | `int` | **Required** | Integer $\ge 0$, $\le \text{visitors\_a}$ | Number of conversions in Group A (Control). Booleans and non-integers are rejected. |
| `visitors_a` | `int` | **Required** | Integer $\ge 1$ | Total visitors in Group A (Control). Booleans and non-integers are rejected. |
| `conversions_b` | `int` | **Required** | Integer $\ge 0$, $\le \text{visitors\_b}$ | Number of conversions in Group B (Treatment). Booleans and non-integers are rejected. |
| `visitors_b` | `int` | **Required** | Integer $\ge 1$ | Total visitors in Group B (Treatment). Booleans and non-integers are rejected. |
| `prior_alpha` | `float` | Optional (`1.0`) | Finite number $> 0$ | Beta prior pseudo-conversions $\alpha$. |
| `prior_beta` | `float` | Optional (`1.0`) | Finite number $> 0$ | Beta prior pseudo-non-conversions $\beta$. |
| `credible_interval` | `float` | Optional (`0.95`) | Finite number $\in (0, 1)$ | Credible interval probability mass (e.g., 0.95 for a 95% CI). |
| `minimum_effect` | `float` | Optional (`0.0`) | Finite number $\in [0.0, 1.0]$ | Minimum practically meaningful absolute difference ($B - A$). |

*Note: In addition to flat parameters, structured nested objects (`{"group_a": {...}, "group_b": {...}}` or `{"control": {...}, "treatment": {...}}`) are supported via model validation.*

#### Validation & Error Behavior
- Schema violations, missing required fields, boolean counts, or out-of-range floats produce an **HTTP 422 Unprocessable Entity** response.
- Statistical domain errors produce an **HTTP 400 Bad Request** response.

---

### 4. Example Request

```bash
curl -X POST http://127.0.0.1:8000/api/experiment/bayesian \
  -H "Content-Type: application/json" \
  -d '{
    "conversions_a": 45,
    "visitors_a": 500,
    "conversions_b": 60,
    "visitors_b": 500,
    "prior_alpha": 1.0,
    "prior_beta": 1.0,
    "credible_interval": 0.95,
    "minimum_effect": 0.01
  }'
```

---

### 5. Example Response (200 OK)

```json
{
  "visitors_a": 500,
  "conversions_a": 45,
  "conversion_rate_a": 0.09,
  "observed_conversion_rate_a": 0.09,
  "visitors_b": 500,
  "conversions_b": 60,
  "conversion_rate_b": 0.12,
  "observed_conversion_rate_b": 0.12,
  "observed_absolute_difference": 0.03,
  "observed_difference": 0.03,
  "prior_alpha": 1.0,
  "prior_beta": 1.0,
  "posterior_alpha_a": 46.0,
  "posterior_beta_a": 456.0,
  "posterior_alpha_b": 61.0,
  "posterior_beta_b": 441.0,
  "posterior_mean_a": 0.091633,
  "posterior_mean_b": 0.121514,
  "posterior_mean_difference": 0.02988,
  "credible_interval_a": [0.068305, 0.118672],
  "credible_interval_b": [0.094592, 0.151746],
  "credible_interval_difference": [-0.009384, 0.069486],
  "probability_b_greater_than_a": 0.93179,
  "probability_b_exceeds_a_by_minimum_effect": 0.83515,
  "credible_interval_level": 0.95,
  "credible_interval": 0.95,
  "minimum_effect": 0.01,
  "simulation_method": "monte_carlo",
  "num_samples": 100000,
  "random_seed": 42
}
```

---

### 6. Response Interpretation

- **Posterior Parameters & Regularization**:
  - Group A: $\alpha_{\text{post}} = 1 + 45 = 46$, $\beta_{\text{post}} = 1 + (500 - 45) = 456$. Posterior mean is $46 / 502 \approx 0.0916$ ($9.16\%$).
  - Group B: $\alpha_{\text{post}} = 1 + 60 = 61$, $\beta_{\text{post}} = 1 + (500 - 60) = 441$. Posterior mean is $61 / 502 \approx 0.1215$ ($12.15\%$).
  - Notice the regularizing effect of the uniform $\text{Beta}(1, 1)$ prior: observed rates ($9.00\%$ and $12.00\%$) are gently shrunk toward $9.16\%$ and $12.15\%$.
- **Credible Intervals**:
  - Individual group intervals are calculated via exact inverse cumulative distribution function (`ppf`) of each Beta distribution. For Group A, there is a 95% posterior probability that the true conversion rate lies between $6.83\%$ and $11.87\%$. For Group B, it lies between $9.46\%$ and $15.17\%$.
- **Difference Direction & Interval**:
  - The difference is defined as **Treatment ($B$) minus Control ($A$)**: $p_B - p_A$.
  - The 95% credible interval for the difference is $[-0.0094, 0.0695]$, meaning that with 95% probability, Treatment B's true conversion rate is between $-0.94$ percentage points and $+6.95$ percentage points relative to Control A.
- **Probability of Superiority**:
  - `probability_b_greater_than_a` $\approx 0.9318$ (93.18%). Given the data and prior, there is a 93.18% posterior probability that Treatment B has a higher true conversion rate than Control A.
- **Probability Exceeding Minimum Practical Effect**:
  - `probability_b_exceeds_a_by_minimum_effect` $\approx 0.8352$ (83.52%). There is an 83.52% posterior probability that Treatment B beats Control A by at least 1 percentage point ($\delta = 0.01$).
- **Units**:
  - All rates, differences, and thresholds are expressed as **proportions** between `0.0` and `1.0`. For example, `0.01` represents one percentage point ($1\%$).

---

### 7. Assumptions & Limitations

- **Independent Bernoulli Outcomes**: Assumes each visitor's conversion decision is an independent, identically distributed Bernoulli trial with no cross-user interference or multi-device duplicates.
- **Stationary Conversion Rates**: Assumes true underlying conversion rates remain constant throughout the duration of the test, without unmodeled day-of-week seasonality or trend drift.
- **Independent Beta Priors**: Assumes the prior beliefs for Group A and Group B conversion rates are statistically independent.
- **Monte Carlo Simulation for Difference Distribution**: While group posterior means and individual credible intervals are exact analytical Beta calculations, the difference distribution $(p_B - p_A)$ and tail probabilities are computed via reproducible Monte Carlo simulation ($N = 100,000$ draws, default seed `42`). Simulation results are stable numerical approximations rather than exact closed-form solutions.
- **Prior Sensitivity on Small Samples**: When sample sizes are small ($N < 50$), the choice of prior has an observable influence on the posterior mean and intervals. As sample sizes grow, observed data rapidly dominates the prior.
- **Sound Experimental Design Required for Causal Inference**: Bayesian probabilities describe parameter uncertainty under the statistical model, but **do not prove causality** unless the experiment was conducted under sound randomized assignment without confounding, selection bias, or sample ratio mismatches.

---

### 8. Running Bayesian Tests

Run the dedicated Bayesian engine unit tests and full automated test suite using the project's virtual environment:

```powershell
# Windows (PowerShell)
.venv\Scripts\python.exe -m pytest tests/test_bayesian_engine.py -q
.venv\Scripts\python.exe -m pytest tests/test_api.py -q
.venv\Scripts\python.exe -m pytest -q
```

Or using standard pytest:
```bash
pytest tests/test_bayesian_engine.py -v
pytest -v
```

---

## Causal Inference API

### Unadjusted Average Treatment Effect (ATE)
- **URL**: `POST /api/causal/unadjusted-ate`
- **Description**: Estimates the unadjusted Average Treatment Effect (difference in sample means) with two-sided Welch-style confidence intervals and Welch–Satterthwaite degrees of freedom.
- **Request Parameters**:
  - `data` (list of dicts, required): Tabular observations containing treatment and outcome values.
  - `treatment_col` (string, required): Name of the binary treatment indicator column.
  - `outcome_col` (string, required): Name of the numeric outcome column.
  - `treated_value` (any, optional): Label identifying treated units (inferred if omitted).
  - `untreated_value` (any, optional): Label identifying control units (inferred if omitted).
  - `drop_missing` (boolean, optional, default `false`): Whether to drop rows with missing values across analysis columns.
  - `confidence_level` (float, optional, default `0.95`): Two-sided confidence level strictly between 0 and 1.

**Example Request**:
```bash
curl -X POST http://127.0.0.1:8000/api/causal/unadjusted-ate \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"treatment": 1, "outcome": 12},
      {"treatment": 1, "outcome": 14},
      {"treatment": 1, "outcome": 16},
      {"treatment": 0, "outcome": 6},
      {"treatment": 0, "outcome": 10}
    ],
    "treatment_col": "treatment",
    "outcome_col": "outcome",
    "treated_value": 1,
    "untreated_value": 0,
    "drop_missing": false,
    "confidence_level": 0.95
  }'
```

**Example Response (200 OK)**:
```json
{
  "estimator": "unadjusted_difference_in_means",
  "ate": 6.0,
  "mean_outcome_treated": 14.0,
  "mean_outcome_control": 8.0,
  "n_total": 5,
  "n_treated": 3,
  "n_control": 2,
  "std_error": 2.3094,
  "degrees_of_freedom": 1.6842,
  "confidence_level": 0.95,
  "ci_lower": -5.9540,
  "ci_upper": 17.9540,
  "confidence_interval": [-5.9540, 17.9540],
  "margin_of_error": 11.9540,
  "rows_dropped": 0,
  "interpretation": "The 95% confidence interval [-5.9540, 17.9540] quantifies sampling uncertainty around the difference in means (6.0000)...",
  "assumptions": "Unadjusted difference in means reflects the true Average Treatment Effect (ATE) only under randomized treatment assignment..."
}
```

---

### Regression-Adjusted Average Treatment Effect (ATE)
- **URL**: `POST /api/causal/regression-adjusted-ate`
- **Method**: `POST`
- **Description**: Estimates the Average Treatment Effect (ATE) adjusting for observed baseline confounders using regression adjustment (parametric G-computation). An ordinary least squares (OLS) model is fitted, potential outcomes $\hat{Y}_i(1)$ and $\hat{Y}_i(0)$ are imputed for all units, and the average individual contrast is calculated.
- **Request Parameters**:
  - `data` (list of dicts, required): Non-empty list of observation records containing treatment, outcome, and confounders.
  - `treatment_col` (string, required): Name of the binary treatment indicator column.
  - `outcome_col` (string, required): Name of the numeric outcome column.
  - `confounder_cols` (list of strings, required): Non-empty list of continuous or categorical confounder column names to adjust for.
  - `treated_value` (any, optional): Label identifying treated units (inferred automatically if omitted).
  - `untreated_value` (any, optional): Label identifying control units (inferred automatically if omitted).
  - `drop_missing` (boolean, optional, default `false`): Whether to drop rows containing missing values in analysis columns.
- **Methodological & Causal Considerations**:
  - **Adjustment for Observed Confounders**: The endpoint models the conditional expectation $E[Y \mid T, X]$ and controls for the supplied observed confounders.
  - **No Unmeasured Confounding**: Regression adjustment does **not** automatically eliminate unmeasured confounding. If relevant confounders are omitted or unobserved, residual confounding bias remains.
  - **Causal Assumptions**: Causal interpretation strictly depends on key identification assumptions:
    1. *Conditional Exchangeability* ($(Y(1), Y(0)) \perp\!\!\perp T \mid X$): Potential outcomes are independent of treatment conditional on observed covariates.
    2. *Positivity / Common Support* ($0 < P(T = 1 \mid X = x) < 1$): Every covariate subpopulation has a non-zero probability of receiving either treatment.
    3. *Model Specification*: The parametric outcome regression correctly approximates the conditional expectation function.
    4. *SUTVA*: No interference between units and no hidden variations of treatment.

**Example Request**:
```bash
curl -X POST http://127.0.0.1:8000/api/causal/regression-adjusted-ate \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"treatment": 1, "outcome": 67.0, "age": 20},
      {"treatment": 1, "outcome": 82.0, "age": 25},
      {"treatment": 1, "outcome": 97.0, "age": 30},
      {"treatment": 1, "outcome": 112.0, "age": 35},
      {"treatment": 0, "outcome": 68.0, "age": 22},
      {"treatment": 0, "outcome": 80.0, "age": 26},
      {"treatment": 0, "outcome": 98.0, "age": 32},
      {"treatment": 0, "outcome": 116.0, "age": 38}
    ],
    "treatment_col": "treatment",
    "outcome_col": "outcome",
    "confounder_cols": ["age"],
    "treated_value": 1,
    "untreated_value": 0,
    "drop_missing": false
  }'
```

**Example Response (200 OK)**:
```json
{
  "estimator": "regression_adjusted_ate",
  "adjusted_ate": 5.0,
  "ate": 5.0,
  "unadjusted_ate": -1.0,
  "raw_difference": -1.0,
  "difference_in_means": -1.0,
  "mean_outcome_treated": 89.5,
  "mean_outcome_control": 90.5,
  "outcome_mean_treated": 89.5,
  "outcome_mean_untreated": 90.5,
  "mean_predicted_treated": 92.5,
  "mean_predicted_control": 87.5,
  "n_total": 8,
  "n_treated": 4,
  "n_control": 4,
  "n_untreated": 4,
  "treatment_col": "treatment",
  "outcome_col": "outcome",
  "confounder_cols": ["age"],
  "numeric_confounders": ["age"],
  "categorical_confounders": [],
  "model_features": ["const", "treatment_indicator", "age"],
  "treated_value": 1,
  "control_value": 0,
  "untreated_value": 0,
  "rows_dropped": 0,
  "r_squared": 1.0,
  "regression_coefficients": null,
  "f_statistic": null,
  "interpretation": "The regression-adjusted ATE is 5.0000 (unadjusted difference in means: -1.0000). This reflects the average counterfactual contrast: the expected difference in outcome if the entire sample were assigned to the treated condition versus the control condition, adjusting for observed confounders (age)...",
  "assumptions": "Regression adjustment (G-computation) identifies the Average Treatment Effect (ATE) under four primary identification assumptions: (1) Conditional Exchangeability / No Unmeasured Confounding ((Y(1), Y(0)) ⫫ T | X); (2) Positivity / Overlap (0 < P(T=1 | X=x) < 1); (3) Correct Model Specification; and (4) SUTVA..."
}
```

---

### Propensity Score Estimation and Diagnostics
- **URL**: `POST /api/causal/propensity-scores`
- **Method**: `POST`
- **Description**: Fits a binary logistic regression model predicting treatment assignment conditional on observed baseline confounders ($e(X) = P(T = \text{treated} \mid X)$). Returns individual propensity scores aligned with dataset observation rows, empirical common support (overlap) diagnostics, and baseline pre-adjustment standardized mean differences (SMDs).
- **Understanding Propensity Scores (Viva Concept)**:
  - **What it is**: A propensity score estimates an observation's probability of being assigned to the **treatment group** based on their observed characteristics $X$.
  - **What it is NOT**: It is **not** the probability of a positive outcome (e.g. conversion, recovery, or click).
  - **Purpose**: In observational studies, treated and control subjects often differ systematically. Propensity scores reduce high-dimensional confounders into a single scalar probability, allowing researchers to evaluate overlap (common support) and balance before estimating treatment effects.
- **Request Parameters**:
  - `data` (list of dicts, required): Non-empty list of observation records containing treatment and confounders.
  - `treatment_col` (string, required): Name of the binary treatment assignment column.
  - `confounder_cols` (list of strings, required): Non-empty list of continuous or categorical confounder column names.
  - `treated_value` (any, optional): Label identifying treated units (inferred automatically if omitted).
  - `untreated_value` (any, optional): Label identifying control units (inferred automatically if omitted).
  - `control_value` (any, optional): Alias for `untreated_value`.
  - `drop_missing` (boolean, optional, default `false`): Whether to drop rows containing missing values in analysis columns.
  - `near_zero_threshold` (float, optional, default `0.05`): Diagnostic threshold below which a score is flagged as near-zero.
  - `near_one_threshold` (float, optional, default `0.95`): Diagnostic threshold above which a score is flagged as near-one.

**Example Request**:
```bash
curl -X POST http://127.0.0.1:8000/api/causal/propensity-scores \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"treatment": 1, "age": 25, "income": 45000.0},
      {"treatment": 1, "age": 30, "income": 52000.0},
      {"treatment": 1, "age": 35, "income": 48000.0},
      {"treatment": 1, "age": 40, "income": 60000.0},
      {"treatment": 0, "age": 24, "income": 42000.0},
      {"treatment": 0, "age": 29, "income": 50000.0},
      {"treatment": 0, "age": 36, "income": 46000.0},
      {"treatment": 0, "age": 42, "income": 58000.0}
    ],
    "treatment_col": "treatment",
    "confounder_cols": ["age", "income"],
    "treated_value": 1,
    "untreated_value": 0,
    "drop_missing": false,
    "near_zero_threshold": 0.05,
    "near_one_threshold": 0.95
  }'
```

**Example Response (200 OK)**:
```json
{
  "estimator": "logistic_propensity_score",
  "propensity_scores": [
    0.539678,
    0.688146,
    0.301595,
    0.694579,
    0.426367,
    0.632164,
    0.195779,
    0.521693
  ],
  "scores": [
    0.539678,
    0.688146,
    0.301595,
    0.694579,
    0.426367,
    0.632164,
    0.195779,
    0.521693
  ],
  "observation_indices": [0, 1, 2, 3, 4, 5, 6, 7],
  "scores_by_index": {
    "0": 0.539678,
    "1": 0.688146,
    "2": 0.301595,
    "3": 0.694579,
    "4": 0.426367,
    "5": 0.632164,
    "6": 0.195779,
    "7": 0.521693
  },
  "n_total": 8,
  "n_treated": 4,
  "n_control": 4,
  "n_untreated": 4,
  "treatment_col": "treatment",
  "confounder_cols": ["age", "income"],
  "numeric_confounders": ["age", "income"],
  "categorical_confounders": [],
  "treated_value": 1,
  "control_value": 0,
  "untreated_value": 0,
  "rows_dropped": 0,
  "model_features": ["const", "age", "income"],
  "model_coefficients": {
    "const": -4.2185,
    "age": -0.0632,
    "income": 0.00013
  },
  "model_metadata": {
    "model_type": "LogisticRegression (sm.Logit)",
    "converged": true,
    "n_iterations": 6,
    "pseudo_r2": 0.1582,
    "features": ["const", "age", "income"],
    "coefficients": {
      "const": -4.2185,
      "age": -0.0632,
      "income": 0.00013
    },
    "warnings": []
  },
  "overlap_diagnostics": {
    "min_score": 0.195779,
    "max_score": 0.694579,
    "min_score_treated": 0.301595,
    "max_score_treated": 0.694579,
    "min_score_control": 0.195779,
    "max_score_control": 0.632164,
    "treated_range": [0.301595, 0.694579],
    "control_range": [0.195779, 0.632164],
    "common_support_range": [0.301595, 0.632164],
    "has_overlap": true,
    "scores_near_zero": 0,
    "scores_near_one": 0,
    "treated_near_zero": 0,
    "treated_near_one": 0,
    "control_near_zero": 0,
    "control_near_one": 0,
    "near_zero_threshold": 0.05,
    "near_one_threshold": 0.95,
    "warning": null,
    "interpretation": "Overlap check indicates common support is present across groups."
  },
  "unadjusted_covariate_balance": {
    "age": {
      "covariate": "age",
      "covariate_type": "numeric",
      "mean_treated": 32.5,
      "mean_control": 32.75,
      "std_treated": 6.4549,
      "std_control": 8.0570,
      "pooled_std": 7.3006,
      "smd": -0.0342,
      "abs_smd": 0.0342,
      "zero_pooled_variance": false,
      "is_balanced": true,
      "note": null
    },
    "income": {
      "covariate": "income",
      "covariate_type": "numeric",
      "mean_treated": 53750.0,
      "mean_control": 49000.0,
      "std_treated": 6396.6,
      "std_control": 6976.1,
      "pooled_std": 6692.6,
      "smd": 0.7097,
      "abs_smd": 0.7097,
      "zero_pooled_variance": false,
      "is_balanced": false,
      "note": null
    }
  },
  "covariate_balance": { "...": "alias for unadjusted_covariate_balance" },
  "balance_assessment_type": "unadjusted",
  "post_adjustment_balance_note": "Post-adjustment balance is evaluated in subsequent matching/weighting modules.",
  "categorical_balance_note": "Categorical confounder balance assessment is deferred to subsequent modules.",
  "interpretation": "Estimated propensity scores for 8 observations using 2 confounder(s)...",
  "assumptions": "Propensity score estimation models the conditional probability of treatment assignment P(Treatment = treated | X) using observed confounders..."
}
```

- **Important Methodological Limitations**:
  - **No Treatment Effect**: Propensity score estimation alone does **not** estimate an average treatment effect (ATE). It is a diagnostic and design phase prerequisite for IPTW, matching, or subclassification.
  - **Extreme Propensity Scores**: Scores very close to 0 or 1 indicate near-deterministic treatment assignment and severe lack of common support (positivity violations). In such regions, causal effects cannot be identified without extreme extrapolation.
  - **Unmeasured Confounding**: Propensity scores balance only observed, modeled confounders. They do **not** protect against hidden or unmeasured confounders.

---

### Inverse Probability of Treatment Weighting (IPTW) ATE
- **URL**: `POST /api/causal/iptw-ate`
- **Method**: `POST`
- **Description**: Estimates the Average Treatment Effect (ATE) using Inverse Probability of Treatment Weighting (IPTW). Propensity scores are estimated via logistic regression to construct unstabilized observation weights ($w_i = 1 / p_i$), counterfactual group means are estimated via normalized Hájek-style weights, effective sample sizes (ESS) quantify variance inflation, and covariate balance (SMDs) is evaluated before and after weighting.
- **Viva Concept: The Four Steps of IPTW**:
  1. *Estimate Propensity Scores*: A logistic regression model estimates each observation's conditional probability of receiving treatment given observed baseline covariates: $e(X) = P(T = \text{treated} \mid X)$.
  2. *Calculate Inverse Probability Weights*: For each observation, calculate the probability of receiving their observed treatment condition: $p_i = e(X_i)$ if $T_i = 1$, and $(1 - e(X_i))$ if $T_i = 0$. The unstabilized IPTW weight is $w_i = 1 / p_i$.
  3. *Calculate Weighted Treatment Effect (Hájek Estimator)*: Compute normalized weighted outcome means for treated and control groups:
     $$\mu_1 = \frac{\sum_{i \in \text{Treated}} w_i Y_i}{\sum_{i \in \text{Treated}} w_i}, \quad \mu_0 = \frac{\sum_{i \in \text{Control}} w_i Y_i}{\sum_{i \in \text{Control}} w_i}$$
     The IPTW ATE is $\text{ATE}_{\text{IPTW}} = \mu_1 - \mu_0$.
  4. *Examine Diagnostics*:
     - *Overlap & Positivity*: Check that treated and control propensity score ranges overlap adequately ($0 < P(T = 1 \mid X) < 1$).
     - *Kish's Effective Sample Size (ESS)*: $\text{ESS} = (\sum w_i)^2 / \sum (w_i^2)$. Compares effective statistical sample size against nominal sample size to detect weight concentration.
     - *Extreme Weights*: Diagnoses whether any weight exceeds `extreme_weight_threshold` (default `20.0`), which would inflate estimator variance.
     - *Covariate Balance*: Compares Standardized Mean Differences (SMDs) before and after weighting to ensure pseudo-population balance ($|\text{SMD}| \le 0.10$).
- **Weight Truncation / Clipping**:
  - Setting `truncate_weights: true` and specifying `max_weight` (e.g. `10.0`) truncates weights exceeding `max_weight` ($w_i \leftarrow \min(w_i, \text{max\_weight})$).
  - *Bias-Variance Tradeoff*: Truncating weights reduces estimator variance and protects against erratic swings caused by near-positivity violations, at the potential cost of introducing small finite-sample bias.
- **Causal Assumptions & Limitations**:
  - *Conditional Exchangeability / No Unmeasured Confounding*: $(Y(1), Y(0)) \perp\!\!\perp T \mid X$. Weighting balances only observed, modeled covariates; it does **not** protect against hidden or unmeasured confounders.
  - *Positivity / Common Support*: $0 < P(T = 1 \mid X = x) < 1$ for all units. If assigned-treatment probability is near zero ($\le 10^{-12}$), weights diverge toward infinity, raising a positivity violation.
  - *Correct Propensity Model Specification*: Misspecifying the logistic regression functional form biases the estimated weights.
  - *SUTVA & Consistency*: No interference between units and observed outcome matches potential outcome under the received treatment.
- **Request Parameters**:
  - `data` (list of dicts, required): Non-empty list of observation records containing treatment, outcome, and confounders.
  - `treatment_col` (string, required): Name of binary treatment indicator column.
  - `outcome_col` (string, required): Name of numeric outcome column.
  - `confounder_cols` (list of strings, required): Non-empty list of observed confounder column names.
  - `treated_value` (any, optional): Label identifying treated condition (inferred automatically if omitted).
  - `untreated_value` / `control_value` (any, optional): Label identifying control condition (inferred automatically if omitted).
  - `drop_missing` (boolean, optional, default `false`): Whether to drop incomplete rows across analysis columns.
  - `extreme_weight_threshold` (float, optional, default `20.0`): Threshold above which weights are flagged as extreme in diagnostics.
  - `truncate_weights` (boolean, optional, default `false`): Whether to truncate (clip) weights above `max_weight`.
  - `max_weight` (float, optional): Maximum weight threshold when `truncate_weights=true`.

**Example Request**:
```bash
curl -X POST http://127.0.0.1:8000/api/causal/iptw-ate \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"treatment": 1, "outcome": 10.0, "x": 1.0, "z": 20.0},
      {"treatment": 1, "outcome": 12.0, "x": 2.0, "z": 25.0},
      {"treatment": 1, "outcome": 14.0, "x": 3.0, "z": 30.0},
      {"treatment": 1, "outcome": 16.0, "x": 4.0, "z": 35.0},
      {"treatment": 0, "outcome": 5.0,  "x": 1.0, "z": 22.0},
      {"treatment": 0, "outcome": 7.0,  "x": 2.0, "z": 24.0},
      {"treatment": 0, "outcome": 9.0,  "x": 3.0, "z": 28.0},
      {"treatment": 0, "outcome": 11.0, "x": 4.0, "z": 32.0}
    ],
    "treatment_col": "treatment",
    "outcome_col": "outcome",
    "confounder_cols": ["x", "z"],
    "treated_value": 1,
    "untreated_value": 0,
    "drop_missing": false,
    "extreme_weight_threshold": 20.0,
    "truncate_weights": false
  }'
```

**Example Response (200 OK)**:
```json
{
  "estimator": "iptw_ate",
  "iptw_ate": 5.3491,
  "ate": 5.3491,
  "weighted_mean_outcome_treated": 13.1491,
  "weighted_mean_outcome_control": 7.8000,
  "weighted_mean_treated": 13.1491,
  "weighted_mean_control": 7.8000,
  "unadjusted_ate": 5.0,
  "raw_difference": 5.0,
  "difference_in_means": 5.0,
  "n_total": 8,
  "n_treated": 4,
  "n_control": 4,
  "n_untreated": 4,
  "treatment_col": "treatment",
  "outcome_col": "outcome",
  "confounder_cols": ["x", "z"],
  "numeric_confounders": ["x", "z"],
  "categorical_confounders": [],
  "treated_value": 1,
  "control_value": 0,
  "untreated_value": 0,
  "rows_dropped": 0,
  "propensity_score_summary": {
    "min": 0.2810,
    "max": 0.7291,
    "mean": 0.5000,
    "median": 0.4962,
    "std": 0.1610
  },
  "weight_summary": {
    "min_weight": 1.3715,
    "max_weight": 3.5587,
    "mean_weight": 2.1834,
    "std_weight": 0.7265,
    "extreme_weight_threshold": 20.0,
    "n_extreme_weights": 0,
    "has_extreme_weights": false,
    "is_truncated": false,
    "truncation_max_weight": null,
    "rows_truncated": 0
  },
  "weights": [1.9841, 1.4820, 1.3715, 1.3912, 1.3912, 1.5420, 2.7482, 3.5587],
  "ess_treated": 3.7650,
  "ess_control": 3.4210,
  "ess_total": 7.1860,
  "effective_sample_size_treated": 3.7650,
  "effective_sample_size_control": 3.4210,
  "effective_sample_size_total": 7.1860,
  "ess_treated_fraction": 0.9412,
  "ess_control_fraction": 0.8553,
  "overlap_diagnostics": {
    "has_overlap": true,
    "common_support_range": [0.2810, 0.7190],
    "warning": null
  },
  "covariate_balance_before": {
    "x": {"smd": 0.0, "abs_smd": 0.0, "is_balanced": true},
    "z": {"smd": 0.4520, "abs_smd": 0.4520, "is_balanced": false}
  },
  "covariate_balance_after": {
    "x": {"smd": 0.0421, "abs_smd": 0.0421, "is_balanced": true},
    "z": {"smd": 0.0812, "abs_smd": 0.0812, "is_balanced": true}
  },
  "balance_summary": {
    "max_abs_smd_before": 0.4520,
    "max_abs_smd_after": 0.0812,
    "balance_improved": true
  },
  "warnings": [],
  "interpretation": "The IPTW-estimated Average Treatment Effect (ATE) is 5.3491 (weighted treated outcome mean: 13.1491, control: 7.8000; unadjusted difference in means: 5.0000)...",
  "assumptions": "Inverse Probability of Treatment Weighting (IPTW) identifies the Average Treatment Effect (ATE) under five fundamental identification assumptions..."
}
```

- **Example Validation Errors (422 Unprocessable Content)**:
  - Empty dataset: `"data cannot be empty. Causal inference requires observations."`
  - Missing column: `"Confounder column 'x' not found in dataset."`
  - Duplicate confounders: `"confounder_cols contains duplicate column names: ['x', 'x']."`
  - Column role conflict: `"Confounder column 'x' cannot also be the treatment column."`
  - Invalid threshold: `"extreme_weight_threshold must be a positive finite number (> 0)."`
  - Missing values without opt-in: `"Missing values detected ... Set drop_missing=True to drop incomplete rows."`
  - Positivity violation: `"Positivity violation: observation(s) have an assigned-treatment probability of zero ... IPTW weighting would produce infinite weights."`

- **Running Focused API Tests**:
  ```powershell
  # Windows PowerShell
  .\.venv\Scripts\python.exe -m pytest tests/test_api.py -q
  ```

---

### Difference-in-Differences (DiD)
- **URL**: `POST /api/causal/difference-in-differences`
- **Method**: `POST`
- **Description**: Estimates the Average Treatment Effect on the Treated (ATT) using the canonical two-group, two-period Difference-in-Differences (DiD) estimator. Compares the observed before-and-after change in outcome for a treatment group against the contemporaneous before-and-after change observed in an untreated comparison group.
- **Viva Concept: Understanding Difference-in-Differences**:
  - **The Problem**: A simple before-and-after comparison in the treated group ($\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}}$) confuses treatment effects with general time trends, seasonality, or macroeconomic changes. A simple post-period cross-sectional comparison ($\bar{Y}_{T,\text{post}} - \bar{Y}_{C,\text{post}}$) confuses treatment effects with pre-existing baseline differences between groups.
  - **The Solution**: DiD subtracts the control group's temporal trend from the treated group's temporal change:
    $$\text{DiD} = (\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}}) - (\bar{Y}_{C,\text{post}} - \bar{Y}_{C,\text{pre}})$$
  - **Four Group-Period Means**:
    1. Treatment Pre ($\bar{Y}_{T,\text{pre}}$): Baseline mean outcome for the treatment cohort.
    2. Treatment Post ($\bar{Y}_{T,\text{post}}$): Follow-up mean outcome for the treatment cohort.
    3. Control Pre ($\bar{Y}_{C,\text{pre}}$): Baseline mean outcome for the comparison cohort.
    4. Control Post ($\bar{Y}_{C,\text{post}}$): Follow-up mean outcome for the comparison cohort.
  - **Two Group Trends**:
    - Treatment Change ($\Delta_T = \bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}}$).
    - Control Change ($\Delta_C = \bar{Y}_{C,\text{post}} - \bar{Y}_{C,\text{pre}}$).
    - $\text{DiD Estimate} = \Delta_T - \Delta_C$.
- **Identifying Causal Assumptions**:
  1. *Parallel Trends Assumption*: **Without the treatment, the treated group and comparison group would have followed similar outcome trends over the same period**:
     $$E[Y(0)_{\text{post}} - Y(0)_{\text{pre}} \mid T = 1] = E[Y(0)_{\text{post}} - Y(0)_{\text{pre}} \mid T = 0]$$
     *Crucial Viva Note*: A positive DiD estimate does **not**, by itself, prove that the treatment caused the change. The validity of the estimate rests entirely on the unobservable counterfactual that both groups would have evolved in parallel absent treatment. Because this implementation evaluates a single pre-period and single post-period, pre-intervention trends cannot be formally tested without multiple historical periods.
  2. *No Anticipation of Treatment*: Units do not alter their behavior or outcomes in anticipation of the upcoming intervention before the post-treatment period begins.
  3. *No Differential Concurrent Shocks*: No other contemporaneous external shock, marketing campaign, local regulation, or economic event affected the treatment group differently from the control group around the intervention period.
  4. *Consistent Outcome Measurement & Stable Group Definitions*: The outcome is measured identically in both periods, and group membership does not shift dynamically across waves.
  5. *No Spillover Effects (SUTVA)*: The intervention applied to treated units does not indirectly influence outcomes in the control group.
  6. *Repeated Observations / Panel Caution*: DiD accommodates repeated cross-sections or longitudinal panel data, but individual observations must not be treated as independent when identical units are repeatedly measured across time.
- **Request Parameters**:
  - `data` (list of dicts, required): Non-empty list of observation records containing treatment, time period, and outcome.
  - `treatment_col` (string, required): Treatment-group indicator column distinguishing treated from control units.
  - `time_col` / `time_period_col` (string, required): Time-period indicator column distinguishing pre- from post-intervention.
  - `outcome_col` (string, required): Numeric outcome column.
  - `treated_value` (any, optional): Label identifying treated condition (inferred automatically if omitted).
  - `control_value` / `untreated_value` (any, optional): Label identifying control condition (inferred automatically if omitted).
  - `pre_value` / `pre_period_value` (any, optional): Label identifying pre-intervention period (inferred automatically if omitted).
  - `post_value` / `post_period_value` (any, optional): Label identifying post-intervention period (inferred automatically if omitted).
  - `drop_missing` (boolean, optional, default `false`): Whether to drop incomplete rows across treatment, time, or outcome columns.

**Example Request**:
```bash
curl -X POST http://127.0.0.1:8000/api/causal/difference-in-differences \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"treatment": 1, "time": 0, "outcome": 9.0},
      {"treatment": 1, "time": 0, "outcome": 11.0},
      {"treatment": 1, "time": 1, "outcome": 15.0},
      {"treatment": 1, "time": 1, "outcome": 17.0},
      {"treatment": 0, "time": 0, "outcome": 7.0},
      {"treatment": 0, "time": 0, "outcome": 9.0},
      {"treatment": 0, "time": 1, "outcome": 10.0},
      {"treatment": 0, "time": 1, "outcome": 12.0}
    ],
    "treatment_col": "treatment",
    "time_col": "time",
    "outcome_col": "outcome",
    "treated_value": 1,
    "control_value": 0,
    "pre_value": 0,
    "post_value": 1,
    "drop_missing": false
  }'
```

**Example Response (200 OK)**:
```json
{
  "estimator": "difference_in_differences",
  "did_estimate": 3.0,
  "did": 3.0,
  "ate": 3.0,
  "treatment_mean_pre": 10.0,
  "treatment_mean_post": 16.0,
  "control_mean_pre": 8.0,
  "control_mean_post": 11.0,
  "treated_mean_pre": 10.0,
  "treated_mean_post": 16.0,
  "treatment_change": 6.0,
  "control_change": 3.0,
  "delta_treatment": 6.0,
  "delta_control": 3.0,
  "delta_treated": 6.0,
  "cell_counts": {
    "treatment_pre": 2,
    "treatment_post": 2,
    "control_pre": 2,
    "control_post": 2
  },
  "n_treatment_pre": 2,
  "n_treatment_post": 2,
  "n_control_pre": 2,
  "n_control_post": 2,
  "n_treated_pre": 2,
  "n_treated_post": 2,
  "n_treated": 4,
  "n_control": 4,
  "n_untreated": 4,
  "n_total": 8,
  "rows_dropped": 0,
  "treatment_col": "treatment",
  "time_col": "time",
  "time_period_col": "time",
  "outcome_col": "outcome",
  "treated_value": 1,
  "control_value": 0,
  "untreated_value": 0,
  "pre_value": 0,
  "post_value": 1,
  "interpretation": "The Difference-in-Differences (DiD) treatment effect estimate is 3.0000. Treatment group outcome changed by 6.0000 (from 10.0000 pre-intervention to 16.0000 post-intervention; n_pre=2, n_post=2). Control group outcome changed by 3.0000 (from 8.0000 pre-intervention to 11.0000 post-intervention; n_pre=2, n_post=2).",
  "assumptions": "Difference-in-Differences (DiD) identifies the causal treatment effect under five core assumptions: (1) Parallel Trends ... (2) No Anticipation ... (3) No Differential Concurrent Shocks ... (4) Consistent Measurement ... (5) No Spillover Effects (SUTVA)..."
}
```

- **Example Validation Errors (422 Unprocessable Content)**:
  - Empty dataset: `"data cannot be empty. Causal inference requires observations."`
  - Missing time column: `"time_col (or time_period_col) must be provided as a non-empty string."`
  - Column role conflict: `"treatment_col and time_col cannot be the same column ('treatment')."`
  - Single treatment condition: `"Treatment column 'treatment' contains only 1 unique value ... Difference-in-Differences requires both treated and control observations."`
  - Single time period: `"Time-period column 'time' contains only 1 unique value ... Difference-in-Differences requires both pre- and post-intervention periods."`
  - Empty cell in 2x2 design: `"Difference-in-Differences requires at least one observation in all four group-period cells. Empty cell(s) found: Control Post (treatment=0, period=1)."`
  - Non-numeric outcome: `"Outcome column 'outcome' must be numeric, but contains boolean values."`
  - Missing values without opt-in: `"Dataset contains missing values in required columns ... Set drop_missing=True to explicitly drop incomplete observations."`

- **Running Focused API Tests**:
  ```powershell
  # Windows PowerShell
  .\.venv\Scripts\python.exe -m pytest tests/test_api.py -q
  ```


---

### Causal Sensitivity Analysis and Robustness Checks
- **URL**: `POST /api/causal/sensitivity-analysis`
- **Description**: Conducts systematic sensitivity analysis and robustness checks for causal inference across three complementary dimensions:
  1. **Multi-Estimator Comparison (Part A)**: Evaluates whether conclusions are consistent across distinct functional specifications: Unadjusted Difference in Means, Regression-Adjusted ATE (G-computation), and Inverse Probability of Treatment Weighting (IPTW).
  2. **IPTW Weight Truncation Sensitivity (Part B)**: Assesses weight stability, effective sample sizes (ESS), and point estimate shifts across caller-specified truncation thresholds (e.g. $[10.0, 5.0, 3.0]$).
  3. **Unmeasured Confounding Sensitivity (Part C)**: Applies a transparent hypothetical bias-adjustment model ($\tau_{\text{adjusted}}(b) = \tau_{\text{observed}} - b$) to assess whether qualitative conclusions (and confidence intervals) would hold under plausible unmeasured confounding scenarios.
  4. **Dual Execution Modes**:
     - *Dataset Mode*: Accepts raw observation records and coordinates full multi-estimator comparisons, optional IPTW truncation, and bias scenarios.
     - *Standalone Mode*: Accepts a pre-calculated point estimate (`observed_estimate`), hypothetical bias scenarios (`bias_values`), and optional confidence interval (`confidence_interval`) without requiring raw microdata.

- **Request Parameters**:
  - `data` (list of dicts, optional): Observation records (required in dataset mode).
  - `treatment_col` (string, optional): Binary treatment assignment column (required if `data` is provided).
  - `outcome_col` (string, optional): Numeric outcome column (required if `data` is provided).
  - `confounder_cols` (list of strings, optional): Confounders for regression adjustment and IPTW.
  - `reference_estimator` (string, optional, default `"regression_adjusted"`): Baseline reference estimator (`"regression_adjusted"`, `"iptw"`, or `"unadjusted"`).
  - `truncation_thresholds` (list of positive floats, optional): Sequence of upper bounds for IPTW weight truncation checks (weights are never truncated unless explicitly requested).
  - `hypothetical_biases` / `bias_values` (float or list of floats, optional): Hypothetical signed bias values $b$ for unmeasured confounding scenarios.
  - `observed_estimate` (float, optional): Observed effect estimate (required in standalone mode when `data` is omitted).
  - `confidence_interval` (list of 2 floats, optional): Two-element confidence interval $[L, U]$ of the observed estimate in standalone mode.
  - `drop_missing` (boolean, optional, default `false`): Whether to drop incomplete rows across analysis columns.

**Example 1: Dataset Mode with Truncation and Bias Scenarios**:
```bash
curl -X POST http://127.0.0.1:8000/api/causal/sensitivity-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"treatment": 1, "outcome": 20.0, "x": 2.0},
      {"treatment": 1, "outcome": 25.0, "x": 3.0},
      {"treatment": 1, "outcome": 30.0, "x": 4.0},
      {"treatment": 0, "outcome": 10.0, "x": 1.0},
      {"treatment": 0, "outcome": 15.0, "x": 2.0},
      {"treatment": 0, "outcome": 20.0, "x": 3.0}
    ],
    "treatment_col": "treatment",
    "outcome_col": "outcome",
    "confounder_cols": ["x"],
    "reference_estimator": "regression_adjusted",
    "truncation_thresholds": [10.0, 3.0],
    "hypothetical_biases": [1.0, 2.0, -1.0]
  }'
```

**Example 1 Response (200 OK)**:
```json
{
  "reference_estimator": "regression_adjusted",
  "reference_estimate": 10.0,
  "observed_estimate": 10.0,
  "reference_confidence_interval": null,
  "estimator_comparisons": {
    "unadjusted": {
      "name": "Unadjusted Difference in Means",
      "available": true,
      "estimate": 10.0,
      "difference_from_reference": 0.0,
      "confidence_interval": [2.6397, 17.3603],
      "n_total": 6,
      "mean_treated": 25.0,
      "mean_control": 15.0
    },
    "regression_adjusted": {
      "name": "Regression-Adjusted ATE (G-computation)",
      "available": true,
      "estimate": 10.0,
      "difference_from_reference": 0.0,
      "model_type": "ols_linear"
    },
    "iptw": {
      "name": "Inverse Probability of Treatment Weighting (IPTW)",
      "available": true,
      "estimate": 10.0,
      "difference_from_reference": 0.0,
      "ess_treated": 2.94,
      "ess_control": 2.94,
      "ess_total": 5.88
    }
  },
  "iptw_truncation_analysis": {
    "status": "evaluated",
    "untruncated_estimate": 10.0,
    "scenarios": [
      {
        "threshold": 10.0,
        "estimate": 10.0,
        "difference_from_untruncated": 0.0,
        "weights_truncated_count": 0,
        "max_weight": 2.5,
        "ess_total": 5.88,
        "identical_to_untruncated": true
      },
      {
        "threshold": 3.0,
        "estimate": 10.0,
        "difference_from_untruncated": 0.0,
        "weights_truncated_count": 0,
        "max_weight": 2.5,
        "ess_total": 5.88,
        "identical_to_untruncated": true
      }
    ],
    "note": "Weight truncation caps extreme weights at a specified upper threshold, reducing sample variance at the potential cost of introducing finite-sample bias."
  },
  "unmeasured_confounding_analysis": {
    "observed_estimate": 10.0,
    "confidence_interval": null,
    "bias_scenarios": [
      {
        "hypothetical_bias": 1.0,
        "adjusted_estimate": 9.0,
        "adjusted_confidence_interval": null,
        "interval_includes_zero": null
      },
      {
        "hypothetical_bias": 2.0,
        "adjusted_estimate": 8.0,
        "adjusted_confidence_interval": null,
        "interval_includes_zero": null
      },
      {
        "hypothetical_bias": -1.0,
        "adjusted_estimate": 11.0,
        "adjusted_confidence_interval": null,
        "interval_includes_zero": null
      }
    ],
    "sign_convention": "tau_adjusted(b) = tau_observed - b. Positive bias (b > 0) assumes unmeasured confounding inflated the observed effect; negative bias (b < 0) assumes unmeasured confounding suppressed the observed effect."
  },
  "bias_scenarios": [
    {"hypothetical_bias": 1.0, "adjusted_estimate": 9.0},
    {"hypothetical_bias": 2.0, "adjusted_estimate": 8.0},
    {"hypothetical_bias": -1.0, "adjusted_estimate": 11.0}
  ],
  "sign_convention": "tau_adjusted(b) = tau_observed - b.",
  "treatment_col": "treatment",
  "outcome_col": "outcome",
  "confounder_cols": ["x"],
  "warnings": [],
  "interpretation": "Causal sensitivity analysis evaluated reference estimator 'regression_adjusted' with estimate 10.0000. Available estimates (Unadjusted: 10.0000, Regression-Adjusted: 10.0000, IPTW: 10.0000). Weight truncation status: evaluated. Unmeasured confounding status: evaluated.",
  "assumptions": "Causal sensitivity analysis explores robustness to analytical decisions and hypothetical unmeasured confounding. Core principles: (1) Multi-estimator consistency: different estimators target the same ATE under different functional forms... Divergence indicates model misspecification; agreement does NOT guarantee causal validity. (2) Weight truncation: trades variance reduction against bias. (3) Unmeasured confounding: sensitivity parameters are hypothetical assumptions that explore robustness bounds; sensitivity analysis does NOT prove absence of unmeasured confounding or causal validity."
}
```

**Example 2: Standalone Mode (Observed Estimate with Interval Shifting)**:
```bash
curl -X POST http://127.0.0.1:8000/api/causal/sensitivity-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "observed_estimate": 5.0,
    "bias_values": [2.0, 5.0, -2.0],
    "confidence_interval": [3.0, 7.0]
  }'
```

**Example 2 Response (200 OK)**:
```json
{
  "observed_estimate": 5.0,
  "confidence_interval": [3.0, 7.0],
  "bias_scenarios": [
    {
      "hypothetical_bias": 2.0,
      "adjusted_estimate": 3.0,
      "adjusted_confidence_interval": [1.0, 5.0],
      "interval_includes_zero": false
    },
    {
      "hypothetical_bias": 5.0,
      "adjusted_estimate": 0.0,
      "adjusted_confidence_interval": [-2.0, 2.0],
      "interval_includes_zero": true
    },
    {
      "hypothetical_bias": -2.0,
      "adjusted_estimate": 7.0,
      "adjusted_confidence_interval": [5.0, 9.0],
      "interval_includes_zero": false
    }
  ],
  "sign_convention": "tau_adjusted(b) = tau_observed - b. Positive bias (b > 0) assumes unmeasured confounding inflated the observed effect; negative bias (b < 0) assumes unmeasured confounding suppressed the observed effect.",
  "interpretation": "Sensitivity analysis evaluated 3 hypothetical unmeasured confounding scenario(s) for an observed estimate of 5.0000. Under 1 of 3 scenario(s) (bias=[5.0]), the adjusted confidence interval spans zero, indicating that the null effect cannot be ruled out.",
  "assumptions": "Hypothetical bias-adjustment sensitivity analysis explores whether qualitative conclusions are robust to potential unmeasured confounding. Key assumptions and limitations: (1) Scenario-based: the bias parameter (b) is an assumed hypothetical quantity, NOT an empirically identified statistic; (2) Additive structure; (3) Informational role: sensitivity analysis illustrates how findings change under specific violations of exchangeability, but cannot prove that unmeasured confounding is absent or bounded; (4) Inference disclaimer: shifted intervals are conditional on the truth of the assumed bias; no statistical p-values are calculated."
}
```

- **Example Validation Errors (422 Unprocessable Content)**:
  - Missing all inputs: `"Either 'data' (with 'treatment_col' and 'outcome_col') or 'observed_estimate' (with 'hypothetical_biases' or 'bias_values') must be provided."`
  - Empty dataset: `"data cannot be empty. Causal inference requires observations."`
  - Missing column in dataset mode: `"treatment_col must be provided as a non-empty string when data is provided."`
  - Conflicting column roles: `"treatment_col and outcome_col cannot be the same column ('t')."`
  - Confounder collision: `"treatment_col ('t') cannot also appear in confounder_cols."`
  - Unsupported reference estimator: `"Unsupported reference_estimator: 'random_forest'. Expected one of: ['iptw', 'regression_adjusted', 'unadjusted']."`
  - Non-positive truncation threshold: `"truncation_thresholds element at index 0 must be strictly positive, got -1.0."`
  - Standalone mode missing bias scenarios: `"bias_values (or hypothetical_biases) must be provided when analyzing unmeasured confounding for an observed estimate."`
  - Reversed confidence interval bounds: `"confidence_interval lower bound (10.0) cannot exceed upper bound (5.0)."`
  - Non-numeric estimate: `"observed_estimate must be a numeric float or int, got str."`

- **Viva Concepts on Causal Sensitivity Analysis**:
  1. *Why doesn't multi-estimator agreement prove causality?* If an unmeasured confounder $U$ affects both treatment assignment and the outcome, all three estimators (unadjusted, G-computation, and IPTW) will be equally biased because none of them adjust for $U$. Agreement shows that the functional form (linear outcome model vs. logistic propensity model) does not drastically distort the result, but cannot substitute for unconfoundedness.
  2. *What is the bias-variance trade-off in IPTW weight truncation?* Weight truncation caps large weights at a fixed threshold $M$. While this eliminates huge variance and stabilizes the effective sample size (ESS), it introduces finite-sample bias and alters the target estimand from the true population ATE to a pseudo-population where extreme units receive downweighted influence.
  3. *Why can't sensitivity analysis produce a "robustness p-value"?* Sensitivity parameters (hypothetical bias $b$) are subjective assumptions exploring "what-if" scenarios. Inventing a pseudo p-value would falsely imply that the unmeasured confounder's distribution was known and empirically tested. The adjusted interval $[L - b, U - b]$ is purely conditional on the assumption that the true unmeasured bias is exactly $b$.

---

## Dataset Management & Interactive Web Dashboard

The platform includes an interactive frontend dashboard (`http://127.0.0.1:8000/dashboard`) built with semantic HTML, vanilla CSS, and vanilla JavaScript. It adheres to an academic Charcoal & Sage Green visual theme, avoiding flashy animations, gradients, or heavy external framework dependencies.

### 1. Ingestion Architecture & Storage Lifecycle

Datasets uploaded to the platform are managed by `DatasetManager` (`src/dataset_manager.py`):
- **Streaming Ingestion**: Files uploaded via `POST /api/dataset/upload` are streamed to disk in chunks with bounded memory consumption rather than loaded entirely into RAM.
- **Temporary Storage Directory**: Datasets are stored in a managed local temporary directory (`tempfile.mkdtemp(prefix="exp_platform_datasets_")`).
- **Upload Size Limit**: Strictly enforced at **500 MB** (`max_upload_size_bytes = 500 * 1024 * 1024`). Files exceeding this limit are aborted and rejected with HTTP 413.
- **Registry Capacity**: The in-memory catalog tracks up to **50 datasets** (`max_registry_size = 50`). When capacity is exceeded, least-recently-used non-benchmark datasets are automatically unlinked from disk and removed.
- **Time-to-Live (TTL)**: Datasets expire after **24 hours** (`ttl_seconds = 86400`) from their creation time. Expired datasets are cleaned up and subsequent requests return HTTP 404.
- **Explicit Deletion**: Clients can immediately delete an active dataset via `DELETE /api/dataset/{dataset_id}`, freeing on-disk storage and evicting the dataset from the registry.
- **Benchmark Dataset**: The Kevin Hillstrom MineThatData email campaign dataset (64,000 customers, 12 attributes) is stored server-side at `data/hillstrom.csv` and loaded only upon explicit client request via `POST /api/dataset/load-benchmark`.

### 2. Type Inference Lattice & Missing-Value Policies

- **Single-Pass Type Inference**: During ingestion, column data types are inferred via a single-pass streaming lattice:
  $$\text{boolean} \longrightarrow \text{integer} \longrightarrow \text{numeric} \longrightarrow \text{categorical}$$
- **Recognized Missing Values**: The following tokens (case-insensitive) are recognized as missing: `""` (empty string), `NA`, `N/A`, `null`, `none`, `nan`, `.`, and whitespace-only cells.
- **Strict Missing Data Policy**: By default (`drop_missing=false`), causal estimation endpoints reject datasets containing missing values in the requested analysis columns with HTTP 422, preserving data integrity. When `drop_missing=true` is explicitly provided, incomplete records across the selected analysis columns are dropped, and precise row accounting is returned:
  - `total_dataset_rows`: Total records in the stored dataset on disk.
  - `n_total`: Number of complete records evaluated by the statistical engine.
  - `rows_dropped`: Number of records excluded due to missing values.

---

## Dataset-Based Causal Inference Workflows

The platform provides six dataset-backed causal estimation endpoints operating directly on cataloged datasets on disk:

| Endpoint | Method | Key Request Fields | Statistical Estimator / Engine |
| :--- | :--- | :--- | :--- |
| `/api/dataset/{id}/causal/unadjusted-ate` | `POST` | `treatment_col`, `outcome_col`, `drop_missing`, `confidence_level` | Difference in sample means with Welch robust standard errors and Welch–Satterthwaite degrees of freedom |
| `/api/dataset/{id}/causal/regression-adjusted-ate` | `POST` | `treatment_col`, `outcome_col`, `confounder_cols`, `drop_missing` | G-computation (standardization) over an OLS outcome regression model adjusting for continuous & categorical covariates |
| `/api/dataset/{id}/causal/propensity-scores` | `POST` | `treatment_col`, `confounder_cols`, `near_zero_threshold`, `near_one_threshold` | Logistic regression propensity model $e(X) = P(T=1 \mid X)$, common support interval, boundary counts, and baseline SMD covariate balance |
| `/api/dataset/{id}/causal/iptw-ate` | `POST` | `treatment_col`, `outcome_col`, `confounder_cols`, `truncate_weights`, `max_weight` | Hájek normalized inverse probability weighting, Kish effective sample size (ESS), and post-weighting covariate balance checks |
| `/api/dataset/{id}/causal/difference-in-differences` | `POST` | `treatment_col`, `time_col`, `outcome_col`, `drop_missing` | Canonical $2 \times 2$ group-period Difference-in-Differences $(\bar{Y}_{\text{treat, post}} - \bar{Y}_{\text{treat, pre}}) - (\bar{Y}_{\text{ctrl, post}} - \bar{Y}_{\text{ctrl, pre}})$ |
| `/api/dataset/{id}/causal/sensitivity-analysis` | `POST` | `treatment_col`, `outcome_col`, `confounder_cols`, `bias_values`, `truncation_thresholds` | Tri-estimator robustness comparison (Unadjusted, Regression, IPTW), weight truncation bounds, and hypothetical signed additive confounding scenarios |

---

## Identification Assumptions, Methodological Scope, and Known Limitations

### 1. Fundamental Causal Identification Assumptions

Any causal claim generated by the platform relies on foundational identification assumptions that cannot be verified by data alone:
1. **Conditional Exchangeability / Unconfoundedness**:
   $$(Y(1), Y(0)) \perp T \mid X$$
   All common causes of treatment assignment and the outcome must be measured and included in $X$. Unmeasured confounding creates omitted-variable bias across all observational estimators.
2. **Positivity / Common Support**:
   $$0 < P(T = 1 \mid X = x) < 1 \quad \forall x \text{ with } P(X = x) > 0$$
   Every unit must have a non-zero probability of receiving either treatment condition. Positivity violations produce near-boundary propensity scores and unstable inverse probability weights.
3. **Stable Unit Treatment Value Assumption (SUTVA)**:
   No interference between units (one unit's treatment does not affect another's outcome) and no hidden variations of treatment.
4. **Parallel Trends (Difference-in-Differences)**:
   In the absence of treatment, the average outcome of the treated group would have followed the same trajectory over time as the control group. Requires observations across all four $2 \times 2$ group-period cells and absence of differential shocks.

### 2. Diagnostic Role of Propensity Scores

Propensity scores $e(X)$ are conditional probabilities of treatment assignment used to diagnose common support and covariate overlap. **Propensity scores are diagnostic tools, not treatment-effect estimates.** They indicate whether observational groups are comparable, but do not quantify treatment impact on the outcome.

### 3. Purpose and Interpretation of Sensitivity Analysis

Sensitivity analysis evaluates whether qualitative conclusions are vulnerable to potential violations of exchangeability or weighting instability:
- **Hypothetical Parameter**: The bias parameter $b$ in $\tau_{\text{adj}} = \tau_{\text{obs}} - b$ is an assumed hypothetical quantity, **not** an empirically identified statistic.
- **Informational Bound**: Sensitivity scenarios illustrate what magnitude of confounding would nullify an estimate, but **do not prove that unmeasured confounding is absent**.

### 4. Honest Scope & Unimplemented Features

To maintain academic and professional integrity, the following boundaries and limitations are explicitly noted:
- **No E-Values**: The platform does not implement Ding & VanderWeele E-values.
- **No Rosenbaum Bounds**: Non-parametric sensitivity bounds for Wilcoxon signed-rank tests are not implemented.
- **No Clustered Standard Errors**: Clustered standard errors, panel fixed-effects regressions, and generalized estimating equations (GEE) are not implemented.
- **Single-Node Storage**: Dataset storage is temporary and single-node local disk (not a distributed data warehouse or cloud-scale object store).
- **Tested Scale**: The platform is tested and verified on datasets up to **64,000 rows** (the Hillstrom benchmark, executing in approximately 15–20 seconds). Workloads of massive scale (e.g., 10 million rows or distributed big data) are **not supported or guaranteed**.

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
