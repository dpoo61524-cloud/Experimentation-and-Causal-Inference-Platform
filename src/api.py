"""FastAPI Backend for Experimentation and Causal Inference Platform.

Exposes statistical engine functionality for A/B testing and sample size estimation.
"""

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
import uvicorn

# Ensure the src directory is in sys.path for direct imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from statistics_engine import (
    compare_conversion_rates,
    calculate_sample_size,
    two_sample_ttest,
    chi_square_test,
)

app = FastAPI(
    title="Experimentation and Causal Inference Platform API",
    description=(
        "Backend API exposing statistical analysis and sample size estimation "
        "engines for A/B testing and experimentation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle ValueErrors cleanly without exposing internal tracebacks."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


# --- Request and Response Schemas ---

class ExperimentGroup(BaseModel):
    conversions: int = Field(
        ...,
        description="Number of conversions (non-negative integer, <= visitors)",
        examples=[267],
    )
    visitors: int = Field(
        ...,
        description="Number of visitors (positive integer > 0)",
        examples=[21307],
    )

    @field_validator("conversions", "visitors", mode="before")
    @classmethod
    def check_strictly_integer(cls, value: Any, info: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be an integer, not a boolean.")
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{info.field_name} must be a whole integer.")
            value = int(value)
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> "ExperimentGroup":
        if self.visitors <= 0:
            raise ValueError("Visitors must be greater than zero.")
        if self.conversions < 0:
            raise ValueError("Conversions cannot be negative.")
        if self.conversions > self.visitors:
            raise ValueError("Conversions cannot exceed visitors.")
        return self


class ExperimentCompareRequest(BaseModel):
    group_a: ExperimentGroup = Field(
        ...,
        description="Group A / Control group data (conversions and visitors)",
    )
    group_b: ExperimentGroup = Field(
        ...,
        description="Group B / Treatment group data (conversions and visitors)",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_input_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support control/treatment naming
            if "group_a" not in data and "control" in data and "treatment" in data:
                return {"group_a": data["control"], "group_b": data["treatment"]}
            # Support flat conversions_a, visitors_a, etc.
            if "group_a" not in data and "conversions_a" in data and "visitors_a" in data:
                return {
                    "group_a": {
                        "conversions": data.get("conversions_a"),
                        "visitors": data.get("visitors_a"),
                    },
                    "group_b": {
                        "conversions": data.get("conversions_b"),
                        "visitors": data.get("visitors_b"),
                    },
                }
        return data


class ExperimentCompareResponse(BaseModel):
    conversion_rate_a: float = Field(..., description="Conversion rate of Group A")
    conversion_rate_b: float = Field(..., description="Conversion rate of Group B")
    absolute_difference: float = Field(..., description="Absolute difference (rate_a - rate_b)")
    z_statistic: float = Field(..., description="Calculated Z-statistic")
    p_value: float = Field(..., description="Two-sided p-value")
    confidence_interval_95: List[float] = Field(
        ...,
        description="95% confidence interval for the difference",
    )
    statistically_significant: bool = Field(
        ...,
        description="Whether difference is statistically significant (p < 0.05)",
    )


class SampleSizeRequest(BaseModel):
    baseline_rate: float = Field(
        ...,
        description="Baseline conversion rate, between 0 and 1 (exclusive)",
        examples=[0.10],
    )
    expected_rate: float = Field(
        ...,
        description="Expected conversion rate, between 0 and 1 (exclusive)",
        examples=[0.12],
    )
    alpha: float = Field(
        default=0.05,
        description="Significance level (alpha), between 0 and 1 (exclusive). Default is 0.05.",
        examples=[0.05],
    )
    power: float = Field(
        default=0.80,
        description="Statistical power (1 - beta), between 0 and 1 (exclusive). Default is 0.80.",
        examples=[0.80],
    )

    @field_validator("baseline_rate", "expected_rate", "alpha", "power", mode="before")
    @classmethod
    def check_strictly_number(cls, value: Any, info: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a number, not a boolean.")
        return value

    @model_validator(mode="before")
    @classmethod
    def handle_parameter_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support parameter name aliases
            if "baseline_rate" not in data and "baseline_conversion_rate" in data:
                data["baseline_rate"] = data["baseline_conversion_rate"]
            if "expected_rate" not in data and "expected_conversion_rate" in data:
                data["expected_rate"] = data["expected_conversion_rate"]
            if "alpha" not in data and "significance_level" in data:
                data["alpha"] = data["significance_level"]
            if "power" not in data and "statistical_power" in data:
                data["power"] = data["statistical_power"]
        return data

    @model_validator(mode="after")
    def validate_probabilities(self) -> "SampleSizeRequest":
        if not (0 < self.baseline_rate < 1):
            raise ValueError("Baseline conversion rate must be between 0 and 1 (exclusive).")
        if not (0 < self.expected_rate < 1):
            raise ValueError("Expected conversion rate must be between 0 and 1 (exclusive).")
        if self.baseline_rate == self.expected_rate:
            raise ValueError("Expected conversion rate must differ from baseline rate.")
        if not (0 < self.alpha < 1):
            raise ValueError("Significance level (alpha) must be between 0 and 1 (exclusive).")
        if not (0 < self.power < 1):
            raise ValueError("Statistical power must be between 0 and 1 (exclusive).")
        return self


class SampleSizeResponse(BaseModel):
    baseline_rate: float = Field(..., description="Baseline conversion rate")
    expected_rate: float = Field(..., description="Expected conversion rate")
    absolute_effect: float = Field(..., description="Absolute effect size (|expected - baseline|)")
    alpha: float = Field(..., description="Significance level")
    power: float = Field(..., description="Statistical power")
    sample_size_per_group: int = Field(..., description="Required sample size per group")
    total_sample_size: int = Field(..., description="Total required sample size across both groups")


# --- Endpoints ---

@app.get("/", tags=["General"])
async def root() -> Dict[str, Any]:
    """Identify the application and its purpose."""
    return {
        "name": "Experimentation and Causal Inference Platform API",
        "description": "API for A/B testing statistical analysis and sample size estimation.",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Return the health status of the API."""
    return {"status": "healthy"}


@app.post(
    "/api/experiment/compare",
    response_model=ExperimentCompareResponse,
    tags=["Experimentation"],
    summary="Compare two experiment groups",
)
async def compare_experiment_groups(
    payload: ExperimentCompareRequest,
) -> Dict[str, Any]:
    """Compare two experiment groups using a two-proportion z-test.

    Calls the existing statistical engine to calculate:
    - Conversion rates for both groups
    - Absolute difference between rates
    - Z-statistic and p-value
    - 95% confidence interval for difference
    - Statistical significance at alpha = 0.05
    """
    try:
        results = compare_conversion_rates(
            conversions_a=payload.group_a.conversions,
            visitors_a=payload.group_a.visitors,
            conversions_b=payload.group_b.conversions,
            visitors_b=payload.group_b.visitors,
        )
        return results
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@app.post(
    "/api/experiment/sample-size",
    response_model=SampleSizeResponse,
    tags=["Experimentation"],
    summary="Calculate required sample size for A/B test",
)
async def estimate_sample_size(
    payload: SampleSizeRequest,
) -> Dict[str, Any]:
    """Estimate required sample size per group and total sample size for an A/B test.

    Uses the existing statistical engine's NormalIndPower calculation.
    """
    try:
        results = calculate_sample_size(
            baseline_rate=payload.baseline_rate,
            expected_rate=payload.expected_rate,
            alpha=payload.alpha,
            power=payload.power,
        )
        return results
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


class TTestRequest(BaseModel):
    group_a: List[float] = Field(
        ...,
        description="Numerical sample observations for Group A (at least 2 observations)",
        examples=[[12, 15, 14, 10, 13, 14, 16]],
    )
    group_b: List[float] = Field(
        ...,
        description="Numerical sample observations for Group B (at least 2 observations)",
        examples=[[9, 10, 11, 8, 12, 10, 9]],
    )
    alternative: Literal["two-sided", "greater", "less"] = Field(
        default="two-sided",
        description="Alternative hypothesis: 'two-sided', 'greater', or 'less'. Default is 'two-sided'.",
        examples=["two-sided"],
    )
    alpha: float = Field(
        default=0.05,
        description="Significance level (alpha), between 0 and 1 (exclusive). Default is 0.05.",
        examples=[0.05],
    )

    @field_validator("group_a", "group_b", mode="before")
    @classmethod
    def check_strictly_numbers(cls, values: Any, info: Any) -> Any:
        if not isinstance(values, list):
            raise ValueError(f"{info.field_name} must be a list of numbers.")
        for v in values:
            if isinstance(v, bool) or v is None:
                raise ValueError(
                    f"{info.field_name} contains invalid non-numeric or missing values."
                )
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{info.field_name} contains non-numeric value: {v}")
            if math.isnan(fv) or math.isinf(fv):
                raise ValueError(f"{info.field_name} contains NaN or infinite values.")
        return values

    @model_validator(mode="after")
    def validate_ttest_inputs(self) -> "TTestRequest":
        if len(self.group_a) < 2:
            raise ValueError("group_a must contain at least 2 numerical observations.")
        if len(self.group_b) < 2:
            raise ValueError("group_b must contain at least 2 numerical observations.")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be strictly between 0 and 1.")
        return self


class TTestResponse(BaseModel):
    t_statistic: float = Field(..., description="Welch's t-statistic")
    p_value: float = Field(..., description="Two-sided or one-sided p-value")
    sample_size_a: int = Field(..., description="Number of observations in Group A")
    sample_size_b: int = Field(..., description="Number of observations in Group B")
    mean_a: float = Field(..., description="Sample mean of Group A")
    mean_b: float = Field(..., description="Sample mean of Group B")
    difference_in_means: float = Field(..., description="Difference in sample means (mean_a - mean_b)")
    degrees_of_freedom: float = Field(..., description="Welch-Satterthwaite degrees of freedom")
    confidence_interval_95: List[float] = Field(..., description="95% confidence interval for difference in means")
    statistically_significant: bool = Field(..., description="Whether p_value < alpha")
    alternative: str = Field(..., description="Direction of alternative hypothesis tested")


@app.post(
    "/api/experiment/ttest",
    response_model=TTestResponse,
    tags=["Experimentation"],
    summary="Independent two-sample Welch's t-test",
)
async def run_two_sample_ttest(
    payload: TTestRequest,
) -> Dict[str, Any]:
    """Perform Welch's independent two-sample t-test on two continuous numerical samples.

    Welch's t-test compares the means of two independent groups without assuming equal variances.
    """
    try:
        results = two_sample_ttest(
            sample_a=payload.group_a,
            sample_b=payload.group_b,
            alternative=payload.alternative,
            alpha=payload.alpha,
        )
        return results
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


class ChiSquareRequest(BaseModel):
    table: List[List[float]] = Field(
        ...,
        description=(
            "2D contingency table of non-negative categorical observation counts "
            "(at least 2 rows and 2 columns). For example: [[50, 30], [20, 40]]."
        ),
        examples=[[[50, 30], [20, 40]]],
    )
    alpha: float = Field(
        default=0.05,
        description="Significance level (alpha), strictly between 0 and 1. Default is 0.05.",
        examples=[0.05],
    )
    correction: bool = Field(
        default=True,
        description="Whether to apply Yates' continuity correction for 2x2 tables. Default is True.",
        examples=[True],
    )

    @field_validator("table", mode="before")
    @classmethod
    def validate_table_structure(cls, value: Any) -> Any:
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError("table must be a 2D list with at least 2 rows.")
        expected_cols = None
        for row_idx, row in enumerate(value):
            if not isinstance(row, list) or len(row) < 2:
                raise ValueError(
                    f"Row {row_idx} must be a list with at least 2 columns."
                )
            if expected_cols is None:
                expected_cols = len(row)
            elif len(row) != expected_cols:
                raise ValueError(
                    f"All rows must have the same number of columns. Row {row_idx} has {len(row)} columns, expected {expected_cols}."
                )
            for col_idx, cell in enumerate(row):
                if isinstance(cell, bool) or cell is None:
                    raise ValueError(
                        f"Cell ({row_idx}, {col_idx}) contains invalid non-numeric or missing value."
                    )
                try:
                    val = float(cell)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Cell ({row_idx}, {col_idx}) must be a number: {cell}"
                    )
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(
                        f"Cell ({row_idx}, {col_idx}) cannot be NaN or infinite."
                    )
                if val < 0:
                    raise ValueError(
                        f"Cell ({row_idx}, {col_idx}) cannot be negative: {val}"
                    )
        return value

    @model_validator(mode="after")
    def validate_totals(self) -> "ChiSquareRequest":
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be strictly between 0 and 1.")
        total = sum(sum(row) for row in self.table)
        if total <= 0:
            raise ValueError(
                "Contingency table must contain at least one observation (total count > 0)."
            )
        for r_idx, row in enumerate(self.table):
            if sum(row) == 0:
                raise ValueError(f"Row {r_idx} has a sum of zero observations.")
        col_count = len(self.table[0])
        for c_idx in range(col_count):
            col_sum = sum(self.table[r][c_idx] for r in range(len(self.table)))
            if col_sum == 0:
                raise ValueError(f"Column {c_idx} has a sum of zero observations.")
        return self


class ChiSquareResponse(BaseModel):
    chi_square_statistic: float = Field(..., description="Pearson's chi-square test statistic")
    degrees_of_freedom: int = Field(..., description="Degrees of freedom ((r-1) * (c-1))")
    p_value: float = Field(..., description="Asymptotic p-value")
    expected_frequencies: List[List[float]] = Field(..., description="Expected frequencies under null hypothesis")
    statistically_significant: bool = Field(..., description="Whether p_value < alpha")
    alpha: float = Field(..., description="Significance threshold used")
    warning: str | None = Field(None, description="Warning if any expected cell frequency is less than 5")


@app.post(
    "/api/experiment/chi-square",
    response_model=ChiSquareResponse,
    tags=["Experimentation"],
    summary="Chi-square test of independence",
)
async def run_chi_square_endpoint(
    payload: ChiSquareRequest,
) -> Dict[str, Any]:
    """Perform Pearson's chi-square test of independence on a contingency table.

    Assesses whether an association exists between two categorical variables.
    """
    try:
        results = chi_square_test(
            table=payload.table,
            alpha=payload.alpha,
            correction=payload.correction,
        )
        return results
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

