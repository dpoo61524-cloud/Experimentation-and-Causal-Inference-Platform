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
