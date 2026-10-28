"""FastAPI Backend for Experimentation and Causal Inference Platform.

Exposes statistical engine functionality for A/B testing and sample size estimation.
"""

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
import uvicorn

# Ensure the src directory is in sys.path for direct imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

STATIC_DIR = SRC_DIR / "static"

from statistics_engine import (
    compare_conversion_rates,
    calculate_sample_size,
    two_sample_ttest,
    chi_square_test,
    sequential_conversion_test,
)
from bayesian_engine import bayesian_ab_test
from causal_engine import (
    estimate_unadjusted_ate_with_uncertainty,
    estimate_regression_adjusted_ate,
    estimate_propensity_scores,
    estimate_iptw_ate,
    estimate_difference_in_differences,
    run_causal_sensitivity_analysis,
    analyze_unmeasured_confounding_sensitivity,
)
from dataset_manager import (
    DatasetIngestionResponse,
    default_dataset_manager,
    MISSING_TOKENS,
)
from urllib.parse import unquote
import numpy as np
import pandas as pd

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

# Mount static files directory for the frontend dashboard
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle ValueErrors cleanly without exposing internal tracebacks."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


def _sanitize_for_json(val: Any) -> Any:
    """Recursively convert NumPy scalars and non-native types for clean JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_for_json(item) for item in val]
    return val


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


@app.get(
    "/dashboard",
    response_class=FileResponse,
    tags=["Dashboard"],
    summary="Serve the causal inference web dashboard",
)
async def dashboard() -> FileResponse:
    """Serve the academic experimentation and causal inference web dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard frontend files not found.",
        )
    return FileResponse(str(index_path), media_type="text/html")


# --- Dataset Ingestion Endpoints ---

@app.post(
    "/api/dataset/upload",
    response_model=DatasetIngestionResponse,
    tags=["Datasets"],
    summary="Upload and inspect CSV dataset with bounded streaming",
)
async def upload_dataset_endpoint(request: Request) -> DatasetIngestionResponse:
    """Stream an uploaded CSV dataset to managed storage, enforcing bounded memory and size limits.

    Parses headers, validates row consistency, infers column data types,
    computes missing value statistics, and generates a bounded 5-row preview.
    """
    raw_filename = request.headers.get("X-Filename") or request.query_params.get("filename") or "uploaded_dataset.csv"
    clean_filename = unquote(raw_filename)

    try:
        response = await default_dataset_manager.ingest_stream(
            request.stream(),
            filename=clean_filename,
        )
        return response
    except ValueError as exc:
        err_msg = str(exc)
        if "PAYLOAD_TOO_LARGE" in err_msg:
            raise HTTPException(
                status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                detail=err_msg.replace("PAYLOAD_TOO_LARGE: ", ""),
            )
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=err_msg,
        )


@app.post(
    "/api/dataset/load-benchmark",
    response_model=DatasetIngestionResponse,
    tags=["Datasets"],
    summary="Load Hillstrom benchmark dataset safely from project data directory",
)
async def load_benchmark_endpoint() -> DatasetIngestionResponse:
    """Load the Hillstrom email marketing benchmark dataset safely from project-relative data/hillstrom.csv."""
    try:
        project_root = SRC_DIR.parent
        response = default_dataset_manager.load_benchmark(project_root)
        return response
    except ValueError as exc:
        if "not found" in str(exc).lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )


@app.get(
    "/api/dataset/{dataset_id}",
    response_model=DatasetIngestionResponse,
    tags=["Datasets"],
    summary="Retrieve dataset schema and preview by opaque ID",
)
async def get_dataset_metadata(dataset_id: str) -> DatasetIngestionResponse:
    """Retrieve metadata and bounded preview for an active dataset by its opaque UUID4 identifier."""
    dataset = default_dataset_manager.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found or has expired.",
        )
    return dataset


@app.delete(
    "/api/dataset/{dataset_id}",
    tags=["Datasets"],
    summary="Delete dataset and release temporary storage",
)
async def delete_dataset_endpoint(dataset_id: str) -> Dict[str, Any]:
    """Explicitly delete a stored dataset and free managed disk space."""
    deleted = default_dataset_manager.delete_dataset(dataset_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found or already deleted.",
        )
    return {"status": "deleted", "dataset_id": dataset_id}


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


class SequentialTestRequest(BaseModel):
    checkpoint: int = Field(
        ...,
        description="Current checkpoint / interim look index (1 <= checkpoint <= total_checkpoints)",
        examples=[1],
    )
    total_checkpoints: int = Field(
        ...,
        description="Total planned checkpoints / interim looks K (2, 3, 4, or 5)",
        examples=[5],
    )
    conversions_a: int = Field(
        ...,
        description="Cumulative conversions in Control (Group A)",
        examples=[20],
    )
    visitors_a: int = Field(
        ...,
        description="Cumulative visitors in Control (Group A)",
        examples=[200],
    )
    conversions_b: int = Field(
        ...,
        description="Cumulative conversions in Treatment (Group B)",
        examples=[45],
    )
    visitors_b: int = Field(
        ...,
        description="Cumulative visitors in Treatment (Group B)",
        examples=[200],
    )
    max_visitors_per_group: int = Field(
        ...,
        description="Planned maximum visitors per group (N_max)",
        examples=[1000],
    )
    alpha: float = Field(
        0.05,
        description="Overall two-sided significance level (e.g., 0.01, 0.05, 0.10)",
        examples=[0.05],
    )
    evaluate_futility: bool = Field(
        False,
        description="Whether to evaluate non-binding directional futility heuristic",
        examples=[False],
    )
    schedule_tolerance: float | None = Field(
        0.15,
        description="Allowable absolute deviation between observed and planned information fractions",
        examples=[0.15],
    )

    @field_validator(
        "checkpoint",
        "total_checkpoints",
        "conversions_a",
        "visitors_a",
        "conversions_b",
        "visitors_b",
        "max_visitors_per_group",
        mode="before",
    )
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
    def validate_sequential_params(self) -> "SequentialTestRequest":
        if self.total_checkpoints not in (2, 3, 4, 5):
            raise ValueError("total_checkpoints must be one of 2, 3, 4, or 5.")
        if not (1 <= self.checkpoint <= self.total_checkpoints):
            raise ValueError(
                f"checkpoint must be between 1 and {self.total_checkpoints}, got {self.checkpoint}."
            )
        if self.max_visitors_per_group <= 0:
            raise ValueError("max_visitors_per_group must be greater than zero.")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be strictly between 0 and 1.")
        if self.visitors_a <= 0 or self.visitors_b <= 0:
            raise ValueError("visitors must be greater than zero.")
        if self.conversions_a < 0 or self.conversions_b < 0:
            raise ValueError("conversions cannot be negative.")
        if self.conversions_a > self.visitors_a:
            raise ValueError("Group A conversions cannot exceed visitors.")
        if self.conversions_b > self.visitors_b:
            raise ValueError("Group B conversions cannot exceed visitors.")
        if self.visitors_a > self.max_visitors_per_group:
            raise ValueError(
                f"Group A visitors ({self.visitors_a}) exceed planned max sample size ({self.max_visitors_per_group})."
            )
        if self.visitors_b > self.max_visitors_per_group:
            raise ValueError(
                f"Group B visitors ({self.visitors_b}) exceed planned max sample size ({self.max_visitors_per_group})."
            )
        if self.schedule_tolerance is not None and not (0.0 <= self.schedule_tolerance <= 1.0):
            raise ValueError("schedule_tolerance must be between 0 and 1, or None.")
        return self


class SequentialTestResponse(BaseModel):
    design: str = Field(..., description="Group-sequential design used")
    methodology: str = Field(..., description="Underlying mathematical computation methodology")
    checkpoint: int = Field(..., description="Current look index evaluated")
    total_checkpoints: int = Field(..., description="Total planned looks K")
    planned_information_fraction: float = Field(..., description="Planned information fraction t_k = k/K")
    observed_information_fraction: float = Field(..., description="Observed information fraction (nA + nB)/(2*N_max)")
    schedule_tolerance: float | None = Field(None, description="Tolerance applied to information fraction schedule")
    schedule_compliant: bool = Field(..., description="Whether observed sample size meets schedule assumptions")
    max_visitors_per_group: int = Field(..., description="Planned maximum sample size per group")
    conversions_a: int = Field(..., description="Cumulative conversions in Control")
    visitors_a: int = Field(..., description="Cumulative visitors in Control")
    conversion_rate_a: float = Field(..., description="Control conversion rate")
    conversions_b: int = Field(..., description="Cumulative conversions in Treatment")
    visitors_b: int = Field(..., description="Cumulative visitors in Treatment")
    conversion_rate_b: float = Field(..., description="Treatment conversion rate")
    absolute_difference: float = Field(..., description="Treatment conversion rate - Control conversion rate")
    z_statistic: float = Field(..., description="Pooled two-proportion z-statistic")
    nominal_p_value: float = Field(..., description="Two-sided unadjusted nominal p-value")
    efficacy_boundary_z: float = Field(..., description="O'Brien-Fleming critical boundary at this look")
    nominal_alpha_threshold: float = Field(..., description="Nominal alpha spent at this look")
    all_planned_boundaries_z: List[float] = Field(..., description="Full vector of planned boundaries across all K looks")
    efficacy_boundary_crossed: bool = Field(..., description="Whether |Z| >= Z_crit at this look")
    maximum_sample_size_reached: bool = Field(..., description="Whether this is the final planned look (k = K)")
    evaluate_futility: bool = Field(..., description="Whether futility heuristic was evaluated")
    futility_boundary_crossed: bool = Field(..., description="Whether directional futility condition was met")
    futility_recommendation: str | None = Field(None, description="Advisory non-binding futility recommendation")
    decision: str = Field(..., description="Statistical decision: stop_for_efficacy, continue, stop_max_sample_size_efficacy_achieved, or stop_max_sample_size_inconclusive")
    overall_alpha: float = Field(..., description="Overall targeted Type I error rate")
    explanation: str = Field(..., description="Educational explanation of the result and next steps")
    assumptions: str = Field(..., description="Key statistical assumptions governing this test")


@app.post(
    "/api/experiment/sequential",
    response_model=SequentialTestResponse,
    tags=["Experimentation"],
    summary="Group-sequential testing for conversion rates",
)
async def run_sequential_endpoint(
    payload: SequentialTestRequest,
) -> Dict[str, Any]:
    """Evaluate an ongoing A/B conversion experiment at an interim or final checkpoint.

    Uses discrete O'Brien-Fleming canonical boundaries to control overall Type I error.
    """
    try:
        results = sequential_conversion_test(
            checkpoint=payload.checkpoint,
            total_checkpoints=payload.total_checkpoints,
            conversions_a=payload.conversions_a,
            visitors_a=payload.visitors_a,
            conversions_b=payload.conversions_b,
            visitors_b=payload.visitors_b,
            max_visitors_per_group=payload.max_visitors_per_group,
            alpha=payload.alpha,
            evaluate_futility=payload.evaluate_futility,
            schedule_tolerance=payload.schedule_tolerance,
        )
        return results
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


class BayesianTestRequest(BaseModel):
    conversions_a: int = Field(
        ...,
        description="Number of conversions in Group A (non-negative integer <= visitors_a)",
        examples=[45],
    )
    visitors_a: int = Field(
        ...,
        description="Total number of visitors in Group A (positive integer >= 1)",
        examples=[500],
    )
    conversions_b: int = Field(
        ...,
        description="Number of conversions in Group B (non-negative integer <= visitors_b)",
        examples=[60],
    )
    visitors_b: int = Field(
        ...,
        description="Total number of visitors in Group B (positive integer >= 1)",
        examples=[500],
    )
    prior_alpha: float = Field(
        default=1.0,
        description="Positive Beta prior alpha parameter (pseudo-conversions). Default is 1.0 (flat prior).",
        examples=[1.0],
    )
    prior_beta: float = Field(
        default=1.0,
        description="Positive Beta prior beta parameter (pseudo-non-conversions). Default is 1.0 (flat prior).",
        examples=[1.0],
    )
    credible_interval: float = Field(
        default=0.95,
        description="Credible interval mass strictly between 0 and 1. Default is 0.95 (95% CI).",
        examples=[0.95],
    )
    minimum_effect: float = Field(
        default=0.0,
        description="Minimum practically meaningful absolute improvement (B minus A), between 0 and 1 inclusive. Default is 0.0.",
        examples=[0.0],
    )

    @field_validator("conversions_a", "visitors_a", "conversions_b", "visitors_b", mode="before")
    @classmethod
    def check_strictly_integer(cls, value: Any, info: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be an integer, not a boolean.")
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{info.field_name} must be a whole integer.")
            value = int(value)
        return value

    @field_validator("prior_alpha", "prior_beta", "credible_interval", "minimum_effect", mode="before")
    @classmethod
    def check_strictly_number(cls, value: Any, info: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a number, not a boolean.")
        return value

    @model_validator(mode="before")
    @classmethod
    def handle_input_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support nested group_a / group_b format
            if "group_a" in data and isinstance(data["group_a"], dict):
                if "conversions_a" not in data:
                    data["conversions_a"] = data["group_a"].get("conversions")
                if "visitors_a" not in data:
                    data["visitors_a"] = data["group_a"].get("visitors")
            if "group_b" in data and isinstance(data["group_b"], dict):
                if "conversions_b" not in data:
                    data["conversions_b"] = data["group_b"].get("conversions")
                if "visitors_b" not in data:
                    data["visitors_b"] = data["group_b"].get("visitors")
            # Support control / treatment format
            if "control" in data and isinstance(data["control"], dict):
                if "conversions_a" not in data:
                    data["conversions_a"] = data["control"].get("conversions")
                if "visitors_a" not in data:
                    data["visitors_a"] = data["control"].get("visitors")
            if "treatment" in data and isinstance(data["treatment"], dict):
                if "conversions_b" not in data:
                    data["conversions_b"] = data["treatment"].get("conversions")
                if "visitors_b" not in data:
                    data["visitors_b"] = data["treatment"].get("visitors")
        return data

    @model_validator(mode="after")
    def validate_bayesian_inputs(self) -> "BayesianTestRequest":
        if self.visitors_a <= 0:
            raise ValueError("visitors_a must be at least 1 (each group must have at least one visitor).")
        if self.visitors_b <= 0:
            raise ValueError("visitors_b must be at least 1 (each group must have at least one visitor).")
        if self.conversions_a < 0:
            raise ValueError("conversions_a cannot be negative.")
        if self.conversions_b < 0:
            raise ValueError("conversions_b cannot be negative.")
        if self.conversions_a > self.visitors_a:
            raise ValueError(f"conversions_a ({self.conversions_a}) cannot exceed visitors_a ({self.visitors_a}).")
        if self.conversions_b > self.visitors_b:
            raise ValueError(f"conversions_b ({self.conversions_b}) cannot exceed visitors_b ({self.visitors_b}).")
        if not math.isfinite(self.prior_alpha) or self.prior_alpha <= 0:
            raise ValueError("prior_alpha must be a finite positive number (> 0).")
        if not math.isfinite(self.prior_beta) or self.prior_beta <= 0:
            raise ValueError("prior_beta must be a finite positive number (> 0).")
        if not math.isfinite(self.credible_interval) or not (0 < self.credible_interval < 1):
            raise ValueError("credible_interval must be strictly between 0 and 1 (exclusive).")
        if not math.isfinite(self.minimum_effect) or not (0 <= self.minimum_effect <= 1):
            raise ValueError("minimum_effect must be between 0 and 1, inclusive.")
        return self


class BayesianTestResponse(BaseModel):
    visitors_a: int = Field(..., description="Total visitors in Group A")
    conversions_a: int = Field(..., description="Conversions in Group A")
    conversion_rate_a: float = Field(..., description="Observed conversion rate of Group A")
    observed_conversion_rate_a: float = Field(..., description="Observed conversion rate of Group A")
    visitors_b: int = Field(..., description="Total visitors in Group B")
    conversions_b: int = Field(..., description="Conversions in Group B")
    conversion_rate_b: float = Field(..., description="Observed conversion rate of Group B")
    observed_conversion_rate_b: float = Field(..., description="Observed conversion rate of Group B")
    observed_absolute_difference: float = Field(..., description="Observed difference (B minus A)")
    observed_difference: float = Field(..., description="Observed difference (B minus A)")
    prior_alpha: float = Field(..., description="Prior alpha parameter used")
    prior_beta: float = Field(..., description="Prior beta parameter used")
    posterior_alpha_a: float = Field(..., description="Updated posterior alpha for Group A")
    posterior_beta_a: float = Field(..., description="Updated posterior beta for Group A")
    posterior_alpha_b: float = Field(..., description="Updated posterior alpha for Group B")
    posterior_beta_b: float = Field(..., description="Updated posterior beta for Group B")
    posterior_mean_a: float = Field(..., description="Posterior mean conversion rate for Group A")
    posterior_mean_b: float = Field(..., description="Posterior mean conversion rate for Group B")
    credible_interval_a: List[float] = Field(..., description="Equal-tailed credible interval for Group A rate")
    credible_interval_b: List[float] = Field(..., description="Equal-tailed credible interval for Group B rate")
    posterior_mean_difference: float = Field(..., description="Posterior mean difference (B minus A)")
    credible_interval_difference: List[float] = Field(..., description="Credible interval for difference (B minus A)")
    probability_b_greater_than_a: float = Field(..., description="Posterior probability that B > A")
    probability_b_exceeds_a_by_minimum_effect: float = Field(
        ..., description="Posterior probability that B exceeds A by at least minimum_effect"
    )
    credible_interval_level: float = Field(..., description="Configured credible interval probability mass")
    credible_interval: float = Field(..., description="Configured credible interval probability mass")
    minimum_effect: float = Field(..., description="Configured minimum practical effect threshold")
    simulation_method: str = Field(..., description="Method used for difference distribution estimation")
    num_samples: int = Field(..., description="Number of Monte Carlo samples drawn")
    random_seed: Optional[int] = Field(None, description="Random seed used for reproducibility")


@app.post(
    "/api/experiment/bayesian",
    response_model=BayesianTestResponse,
    tags=["Experimentation"],
    summary="Bayesian A/B testing for conversion rates",
)
async def run_bayesian_endpoint(
    payload: BayesianTestRequest,
) -> Dict[str, Any]:
    """Perform a Bayesian A/B test for conversion rates using a Beta-Binomial conjugate model.

    Updates Beta priors with observed conversion counts and estimates posterior means,
    credible intervals, and probability of superiority via Monte Carlo simulation.
    """
    try:
        results = bayesian_ab_test(
            conversions_a=payload.conversions_a,
            visitors_a=payload.visitors_a,
            conversions_b=payload.conversions_b,
            visitors_b=payload.visitors_b,
            prior_alpha=payload.prior_alpha,
            prior_beta=payload.prior_beta,
            credible_interval=payload.credible_interval,
            minimum_effect=payload.minimum_effect,
        )
        return results
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


# --- Causal Inference Models & Endpoints ---

class CausalUnadjustedATERequest(BaseModel):
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Tabular dataset as a list of observation record dictionaries.",
        examples=[
            [
                {"treatment": 1, "outcome": 12},
                {"treatment": 1, "outcome": 14},
                {"treatment": 1, "outcome": 16},
                {"treatment": 0, "outcome": 6},
                {"treatment": 0, "outcome": 10},
            ]
        ],
    )
    treatment_col: str = Field(
        ...,
        description="Name of the binary treatment indicator column in the dataset.",
        examples=["treatment"],
    )
    outcome_col: str = Field(
        ...,
        description="Name of the numeric outcome column in the dataset.",
        examples=["outcome"],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying treated observations. Inferred automatically if omitted.",
        examples=[1],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying control/untreated observations. Inferred automatically if omitted.",
        examples=[0],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to explicitly drop rows with missing values across analysis columns. Default is false.",
        examples=[False],
    )
    confidence_level: float = Field(
        default=0.95,
        description="Two-sided confidence level strictly between 0 and 1 (exclusive). Default is 0.95.",
        examples=[0.95],
    )

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence_level(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("confidence_level must be a numeric float strictly between 0 and 1.")
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv) or not (0.0 < fv < 1.0):
            raise ValueError("confidence_level must be strictly between 0 and 1 (exclusive).")
        return fv

    @field_validator("treatment_col", "outcome_col")
    @classmethod
    def validate_non_empty_column_name(cls, v: Any, info: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return v.strip()

    @field_validator("data")
    @classmethod
    def validate_data_list(cls, v: Any) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            raise ValueError("data must be a list of observation dictionaries.")
        if len(v) == 0:
            raise ValueError("data cannot be empty. Causal inference requires observations.")
        for idx, row in enumerate(v):
            if not isinstance(row, dict):
                raise ValueError(f"Observation at index {idx} must be a dictionary, got {type(row).__name__}.")
        return v


class CausalUnadjustedATEResponse(BaseModel):
    estimator: str = Field(..., description="Name of the estimator used")
    ate: float = Field(..., description="Estimated Average Treatment Effect (difference in sample means)")
    unadjusted_ate: float = Field(..., description="Alias for ate")
    difference_in_means: float = Field(..., description="Alias for ate")
    mean_outcome_treated: float = Field(..., description="Sample mean outcome of the treated group")
    mean_outcome_control: float = Field(..., description="Sample mean outcome of the control/untreated group")
    outcome_mean_treated: float = Field(..., description="Alias for mean_outcome_treated")
    outcome_mean_untreated: float = Field(..., description="Alias for mean_outcome_control")
    n_total: int = Field(..., description="Total number of observations evaluated")
    n_treated: int = Field(..., description="Number of treated observations")
    n_control: int = Field(..., description="Number of control/untreated observations")
    n_untreated: int = Field(..., description="Alias for n_control")
    treatment_col: str = Field(..., description="Name of the treatment column")
    outcome_col: str = Field(..., description="Name of the outcome column")
    confounder_cols: List[str] = Field(default_factory=list, description="Confounder column names (empty for unadjusted)")
    treated_value: Any = Field(..., description="Resolved value representing the treated condition")
    control_value: Any = Field(..., description="Resolved value representing the control condition")
    untreated_value: Any = Field(..., description="Alias for control_value")
    rows_dropped: int = Field(0, description="Number of rows dropped due to missing values")
    assumptions: str = Field(..., description="Educational explanation of identifying assumptions and confounding limitations")
    variance_treated: Optional[float] = Field(None, description="Sample variance of the treated group (ddof=1)")
    variance_control: Optional[float] = Field(None, description="Sample variance of the control group (ddof=1)")
    std_error: Optional[float] = Field(None, description="Welch standard error of the difference in means")
    se: Optional[float] = Field(None, description="Alias for std_error")
    degrees_of_freedom: Optional[float] = Field(None, description="Welch–Satterthwaite effective degrees of freedom")
    df: Optional[float] = Field(None, description="Alias for degrees_of_freedom")
    confidence_level: float = Field(..., description="Two-sided confidence level used")
    t_critical: Optional[float] = Field(None, description="Student's t critical value")
    critical_value: Optional[float] = Field(None, description="Alias for t_critical")
    margin_of_error: Optional[float] = Field(None, description="Margin of error (t_critical * std_error)")
    confidence_interval: Optional[List[float]] = Field(None, description="Confidence interval [ci_lower, ci_upper]")
    ci_lower: Optional[float] = Field(None, description="Lower bound of the confidence interval")
    ci_upper: Optional[float] = Field(None, description="Upper bound of the confidence interval")
    confidence_interval_95: Optional[List[float]] = Field(None, description="95% confidence interval if confidence_level == 0.95")
    interpretation: str = Field(..., description="Interpretation clarifying sampling variability vs causal claims")


@app.post(
    "/api/causal/unadjusted-ate",
    response_model=CausalUnadjustedATEResponse,
    tags=["Causal Inference"],
    summary="Estimate unadjusted Average Treatment Effect (ATE)",
)
async def unadjusted_ate_endpoint(
    payload: CausalUnadjustedATERequest,
) -> Dict[str, Any]:
    """Estimate the unadjusted Average Treatment Effect (ATE) with Welch-style confidence intervals.

    Calculates the unadjusted difference in sample means between treated and control groups,
    allowing for unequal population variances via Welch's standard error and Welch–Satterthwaite
    degrees of freedom.
    """
    try:
        results = estimate_unadjusted_ate_with_uncertainty(
            data=payload.data,
            treatment_col=payload.treatment_col,
            outcome_col=payload.outcome_col,
            treated_value=payload.treated_value,
            untreated_value=payload.untreated_value,
            drop_missing=payload.drop_missing,
            confidence_level=payload.confidence_level,
        )
        return _sanitize_for_json(results)
    except ValueError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )


class DatasetUnadjustedATERequest(BaseModel):
    treatment_col: str = Field(
        ...,
        description="Name of the binary treatment indicator column in the dataset.",
        examples=["treatment"],
    )
    outcome_col: str = Field(
        ...,
        description="Name of the numeric outcome column in the dataset.",
        examples=["outcome"],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying treated observations. Inferred automatically if omitted.",
        examples=[1],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying control/untreated observations. Inferred automatically if omitted.",
        examples=[0],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to explicitly drop rows with missing values across analysis columns. Default is false.",
        examples=[False],
    )
    confidence_level: float = Field(
        default=0.95,
        description="Two-sided confidence level strictly between 0 and 1 (exclusive). Default is 0.95.",
        examples=[0.95],
    )

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence_level(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("confidence_level must be a numeric float strictly between 0 and 1.")
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv) or not (0.0 < fv < 1.0):
            raise ValueError("confidence_level must be strictly between 0 and 1 (exclusive).")
        return fv

    @field_validator("treatment_col", "outcome_col")
    @classmethod
    def validate_non_empty_column_name(cls, v: Any, info: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return v.strip()

    @model_validator(mode="after")
    def validate_distinct_columns(self) -> "DatasetUnadjustedATERequest":
        if self.treatment_col and self.outcome_col:
            if self.treatment_col.strip() == self.outcome_col.strip():
                raise ValueError(
                    f"treatment_col and outcome_col cannot be the same column ('{self.treatment_col}')."
                )
        return self


class DatasetCausalUnadjustedATEResponse(CausalUnadjustedATEResponse):
    dataset_id: str = Field(..., description="Unique opaque dataset identifier")
    dataset_filename: str = Field(..., description="Original filename of the stored dataset evaluated")
    total_dataset_rows: int = Field(..., description="Total rows in the stored dataset before dropping missing values")


def _coerce_label_to_series_dtype(label: Any, series: pd.Series) -> Any:
    """Coerce user-specified treatment/control label to match series dtype if compatible."""
    if label is None:
        return None
    try:
        if (series == label).any():
            return label
    except Exception:
        pass

    if pd.api.types.is_integer_dtype(series):
        try:
            val = int(label)
            if (series == val).any():
                return val
        except (ValueError, TypeError):
            pass

    if pd.api.types.is_float_dtype(series):
        try:
            val = float(label)
            if (series == val).any():
                return val
        except (ValueError, TypeError):
            pass

    if pd.api.types.is_bool_dtype(series):
        str_val = str(label).strip().lower()
        if str_val in ("true", "1", "t", "yes"):
            return True
        if str_val in ("false", "0", "f", "no"):
            return False

    return label


@app.post(
    "/api/dataset/{dataset_id}/causal/unadjusted-ate",
    response_model=DatasetCausalUnadjustedATEResponse,
    tags=["Causal Inference", "Datasets"],
    summary="Estimate unadjusted ATE for a stored dataset",
)
async def dataset_unadjusted_ate_endpoint(
    dataset_id: str,
    payload: DatasetUnadjustedATERequest,
) -> Dict[str, Any]:
    """Estimate the unadjusted Average Treatment Effect (ATE) for a cataloged dataset on disk.

    Validates dataset access and column existence, loads only the relevant treatment
    and outcome columns into memory, enforces missing-data policy, and computes
    Welch-style uncertainty estimates.
    """
    dataset_meta = default_dataset_manager.get_dataset(dataset_id)
    if dataset_meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found or has expired.",
        )

    file_path = default_dataset_manager.get_dataset_file_path(dataset_id)
    if file_path is None or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file for '{dataset_id}' is not available on disk.",
        )

    available_cols = [c.name for c in dataset_meta.columns]
    if payload.treatment_col not in available_cols:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=(
                f"Treatment column '{payload.treatment_col}' not found in dataset. "
                f"Available columns: {available_cols}"
            ),
        )
    if payload.outcome_col not in available_cols:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=(
                f"Outcome column '{payload.outcome_col}' not found in dataset. "
                f"Available columns: {available_cols}"
            ),
        )

    if dataset_meta.row_count == 0:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Dataset is empty (0 rows). Causal inference requires observations in both treatment groups.",
        )

    try:
        df = pd.read_csv(
            file_path,
            usecols=[payload.treatment_col, payload.outcome_col],
            na_values=list(MISSING_TOKENS),
            keep_default_na=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=f"Failed to read dataset columns: {exc}",
        )

    if df.empty:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Dataset is empty (0 rows). Causal inference requires observations in both treatment groups.",
        )

    resolved_treated = _coerce_label_to_series_dtype(payload.treated_value, df[payload.treatment_col])
    resolved_untreated = _coerce_label_to_series_dtype(payload.untreated_value, df[payload.treatment_col])

    try:
        results = estimate_unadjusted_ate_with_uncertainty(
            data=df,
            treatment_col=payload.treatment_col,
            outcome_col=payload.outcome_col,
            treated_value=resolved_treated,
            untreated_value=resolved_untreated,
            drop_missing=payload.drop_missing,
            confidence_level=payload.confidence_level,
        )
        results["dataset_id"] = dataset_id
        results["dataset_filename"] = dataset_meta.filename
        results["total_dataset_rows"] = dataset_meta.row_count
        return _sanitize_for_json(results)
    except ValueError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )



class CausalRegressionAdjustedATERequest(BaseModel):
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Tabular dataset as a non-empty list of observation record dictionaries.",
        examples=[
            [
                {"treatment": 1, "outcome": 12, "age": 21},
                {"treatment": 1, "outcome": 14, "age": 22},
                {"treatment": 1, "outcome": 16, "age": 23},
                {"treatment": 0, "outcome": 6, "age": 21},
                {"treatment": 0, "outcome": 10, "age": 24},
                {"treatment": 0, "outcome": 8, "age": 22},
            ]
        ],
    )
    treatment_col: str = Field(
        ...,
        description="Name of the binary treatment indicator column in the dataset.",
        examples=["treatment"],
    )
    outcome_col: str = Field(
        ...,
        description="Name of the numeric outcome column in the dataset.",
        examples=["outcome"],
    )
    confounder_cols: List[str] = Field(
        ...,
        description="Non-empty list of confounder/covariate column names to adjust for.",
        examples=[["age"]],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying treated observations. Inferred automatically if omitted.",
        examples=[1],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying control/untreated observations. Inferred automatically if omitted.",
        examples=[0],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to explicitly drop rows with missing values across analysis columns. Default is false.",
        examples=[False],
    )

    @field_validator("data")
    @classmethod
    def validate_data_list(cls, v: Any) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            raise ValueError("data must be a list of observation dictionaries.")
        if len(v) == 0:
            raise ValueError("data cannot be empty. Causal inference requires observations.")
        for idx, row in enumerate(v):
            if not isinstance(row, dict):
                raise ValueError(f"Observation at index {idx} must be a dictionary, got {type(row).__name__}.")
        return v

    @field_validator("treatment_col", "outcome_col")
    @classmethod
    def validate_non_empty_column_name(cls, v: Any, info: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return v.strip()

    @field_validator("confounder_cols")
    @classmethod
    def validate_confounders(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("confounder_cols must be a list of column names.")
        if len(v) == 0:
            raise ValueError("confounder_cols must contain at least one column name.")
        clean_cols = []
        for idx, col in enumerate(v):
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"Confounder column at index {idx} must be a non-empty string.")
            clean_cols.append(col.strip())
        return clean_cols

    @field_validator("drop_missing")
    @classmethod
    def validate_drop_missing(cls, v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError("drop_missing must be a boolean.")
        return v

    @model_validator(mode="after")
    def validate_distinct_roles(self) -> "CausalRegressionAdjustedATERequest":
        if self.treatment_col == self.outcome_col:
            raise ValueError(
                f"treatment_col and outcome_col cannot be the same column ('{self.treatment_col}')."
            )
        for col in self.confounder_cols:
            if col == self.treatment_col:
                raise ValueError(
                    f"Confounder column '{col}' cannot also be the treatment column."
                )
            if col == self.outcome_col:
                raise ValueError(
                    f"Confounder column '{col}' cannot also be the outcome column."
                )
        if len(self.confounder_cols) != len(set(self.confounder_cols)):
            raise ValueError(
                f"confounder_cols contains duplicate column names: {self.confounder_cols}."
            )
        return self


class CausalRegressionAdjustedATEResponse(BaseModel):
    estimator: str = Field(..., description="Name of the estimator used (regression_adjusted_ate)")
    adjusted_ate: float = Field(..., description="Estimated regression-adjusted ATE (treated minus control)")
    ate: float = Field(..., description="Alias for adjusted_ate")
    unadjusted_ate: float = Field(..., description="Unadjusted difference in sample means (treated minus control)")
    raw_difference: float = Field(..., description="Alias for unadjusted_ate")
    difference_in_means: float = Field(..., description="Alias for unadjusted_ate")
    mean_outcome_treated: float = Field(..., description="Sample mean outcome of the treated group")
    mean_outcome_control: float = Field(..., description="Sample mean outcome of the control/untreated group")
    outcome_mean_treated: float = Field(..., description="Alias for mean_outcome_treated")
    outcome_mean_untreated: float = Field(..., description="Alias for mean_outcome_control")
    mean_predicted_treated: float = Field(..., description="Average counterfactual predicted outcome under treatment")
    mean_predicted_control: float = Field(..., description="Average counterfactual predicted outcome under control")
    n_total: int = Field(..., description="Total number of observations used in model fitting")
    n_treated: int = Field(..., description="Number of treated observations")
    n_control: int = Field(..., description="Number of control/untreated observations")
    n_untreated: int = Field(..., description="Alias for n_control")
    treatment_col: str = Field(..., description="Name of the treatment column")
    outcome_col: str = Field(..., description="Name of the outcome column")
    confounder_cols: List[str] = Field(..., description="List of confounder column names")
    numeric_confounders: List[str] = Field(..., description="List of continuous numeric confounder columns")
    categorical_confounders: List[str] = Field(..., description="List of categorical confounder columns")
    model_features: List[str] = Field(..., description="Feature column names included in the design matrix")
    treated_value: Any = Field(..., description="Resolved value representing the treated condition")
    control_value: Any = Field(..., description="Resolved value representing the control condition")
    untreated_value: Any = Field(..., description="Alias for control_value")
    rows_dropped: int = Field(0, description="Number of rows dropped due to missing data")
    r_squared: Optional[float] = Field(None, description="Coefficient of determination (R^2) of the outcome model")
    regression_coefficients: Optional[Dict[str, float]] = Field(None, description="Regression coefficients if available")
    f_statistic: Optional[float] = Field(None, description="Model F-statistic if available")
    interpretation: str = Field(..., description="Concise interpretation of adjusted vs raw estimates")
    assumptions: str = Field(..., description="Educational explanation of identifying assumptions and confounding limitations")


@app.post(
    "/api/causal/regression-adjusted-ate",
    response_model=CausalRegressionAdjustedATEResponse,
    tags=["Causal Inference"],
    summary="Estimate regression-adjusted Average Treatment Effect (ATE)",
)
async def regression_adjusted_ate_endpoint(
    payload: CausalRegressionAdjustedATERequest,
) -> Dict[str, Any]:
    """Estimate the regression-adjusted Average Treatment Effect (ATE) using G-computation.

    Fits an OLS regression model of the outcome on treatment and confounders, generates
    counterfactual predictions under treatment and control for each unit, and computes the
    sample average contrast.
    """
    try:
        results = estimate_regression_adjusted_ate(
            data=payload.data,
            treatment_col=payload.treatment_col,
            outcome_col=payload.outcome_col,
            confounder_cols=payload.confounder_cols,
            treated_value=payload.treated_value,
            untreated_value=payload.untreated_value,
            drop_missing=payload.drop_missing,
        )
        return _sanitize_for_json(results)
    except ValueError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )


class DatasetRegressionAdjustedATERequest(BaseModel):
    treatment_col: str = Field(
        ...,
        description="Name of the binary treatment indicator column in the dataset.",
        examples=["treatment"],
    )
    outcome_col: str = Field(
        ...,
        description="Name of the numeric outcome column in the dataset.",
        examples=["outcome"],
    )
    confounder_cols: List[str] = Field(
        ...,
        description="Non-empty list of confounder/covariate column names to adjust for.",
        examples=[["age"]],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying treated observations. Inferred automatically if omitted.",
        examples=[1],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying control/untreated observations. Inferred automatically if omitted.",
        examples=[0],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to explicitly drop rows with missing values across analysis columns. Default is false.",
        examples=[False],
    )

    @field_validator("treatment_col", "outcome_col")
    @classmethod
    def validate_non_empty_column_name(cls, v: Any, info: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return v.strip()

    @field_validator("confounder_cols")
    @classmethod
    def validate_confounders(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("confounder_cols must be a list of column names.")
        if len(v) == 0:
            raise ValueError("confounder_cols must contain at least one column name.")
        clean_cols = []
        for idx, col in enumerate(v):
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"Confounder column at index {idx} must be a non-empty string.")
            clean_cols.append(col.strip())
        return clean_cols

    @field_validator("drop_missing")
    @classmethod
    def validate_drop_missing(cls, v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError("drop_missing must be a boolean.")
        return v

    @model_validator(mode="after")
    def validate_distinct_roles(self) -> "DatasetRegressionAdjustedATERequest":
        if self.treatment_col and self.outcome_col:
            if self.treatment_col.strip() == self.outcome_col.strip():
                raise ValueError(
                    f"treatment_col and outcome_col cannot be the same column ('{self.treatment_col}')."
                )
        if self.confounder_cols:
            if len(self.confounder_cols) != len(set(self.confounder_cols)):
                raise ValueError(
                    f"confounder_cols contains duplicate column names: {self.confounder_cols}."
                )
            for col in self.confounder_cols:
                if self.treatment_col and col == self.treatment_col.strip():
                    raise ValueError(
                        f"Confounder column '{col}' cannot also be the treatment column."
                    )
                if self.outcome_col and col == self.outcome_col.strip():
                    raise ValueError(
                        f"Confounder column '{col}' cannot also be the outcome column."
                    )
        return self


class DatasetCausalRegressionAdjustedATEResponse(CausalRegressionAdjustedATEResponse):
    dataset_id: str = Field(..., description="Unique opaque dataset identifier")
    dataset_filename: str = Field(..., description="Original filename of the stored dataset evaluated")
    total_dataset_rows: int = Field(..., description="Total rows in the stored dataset before dropping missing values")


@app.post(
    "/api/dataset/{dataset_id}/causal/regression-adjusted-ate",
    response_model=DatasetCausalRegressionAdjustedATEResponse,
    tags=["Causal Inference", "Datasets"],
    summary="Estimate regression-adjusted ATE for a stored dataset",
)
async def dataset_regression_adjusted_ate_endpoint(
    dataset_id: str,
    payload: DatasetRegressionAdjustedATERequest,
) -> Dict[str, Any]:
    """Estimate the regression-adjusted Average Treatment Effect (ATE) for a cataloged dataset on disk.

    Validates dataset access and column existence, loads only the relevant treatment,
    outcome, and confounder columns into memory, enforces missing-data policy, and computes
    G-computation counterfactual contrasts under an OLS outcome model.
    """
    dataset_meta = default_dataset_manager.get_dataset(dataset_id)
    if dataset_meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found or has expired.",
        )

    file_path = default_dataset_manager.get_dataset_file_path(dataset_id)
    if file_path is None or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file for '{dataset_id}' is not available on disk.",
        )

    available_cols = [c.name for c in dataset_meta.columns]
    if payload.treatment_col not in available_cols:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=(
                f"Treatment column '{payload.treatment_col}' not found in dataset. "
                f"Available columns: {available_cols}"
            ),
        )
    if payload.outcome_col not in available_cols:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=(
                f"Outcome column '{payload.outcome_col}' not found in dataset. "
                f"Available columns: {available_cols}"
            ),
        )
    for c_col in payload.confounder_cols:
        if c_col not in available_cols:
            raise HTTPException(
                status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                detail=(
                    f"Confounder column '{c_col}' not found in dataset. "
                    f"Available columns: {available_cols}"
                ),
            )

    if dataset_meta.row_count == 0:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Dataset is empty (0 rows). Causal inference requires observations in both treatment groups.",
        )

    cols_to_read = [payload.treatment_col, payload.outcome_col] + payload.confounder_cols

    try:
        df = pd.read_csv(
            file_path,
            usecols=cols_to_read,
            na_values=list(MISSING_TOKENS),
            keep_default_na=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=f"Failed to read dataset columns: {exc}",
        )

    if df.empty:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Dataset is empty (0 rows). Causal inference requires observations in both treatment groups.",
        )

    resolved_treated = _coerce_label_to_series_dtype(payload.treated_value, df[payload.treatment_col])
    resolved_untreated = _coerce_label_to_series_dtype(payload.untreated_value, df[payload.treatment_col])

    try:
        results = estimate_regression_adjusted_ate(
            data=df,
            treatment_col=payload.treatment_col,
            outcome_col=payload.outcome_col,
            confounder_cols=payload.confounder_cols,
            treated_value=resolved_treated,
            untreated_value=resolved_untreated,
            drop_missing=payload.drop_missing,
        )
        results["dataset_id"] = dataset_id
        results["dataset_filename"] = dataset_meta.filename
        results["total_dataset_rows"] = dataset_meta.row_count
        return _sanitize_for_json(results)
    except ValueError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )



class CausalPropensityScoreRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Tabular dataset as a non-empty list of observation record dictionaries.",
        examples=[
            [
                {"treatment": 1, "age": 25, "income": 50000},
                {"treatment": 1, "age": 30, "income": 60000},
                {"treatment": 1, "age": 35, "income": 55000},
                {"treatment": 0, "age": 22, "income": 30000},
                {"treatment": 0, "age": 28, "income": 35000},
                {"treatment": 0, "age": 32, "income": 32000},
            ]
        ],
    )
    treatment_col: str = Field(
        ...,
        description="Name of the binary treatment indicator column in the dataset.",
        examples=["treatment"],
    )
    confounder_cols: List[str] = Field(
        ...,
        description="Non-empty list of confounder/covariate column names to model treatment assignment.",
        examples=[["age", "income"]],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying treated observations. Inferred automatically if omitted.",
        examples=[1],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying control/untreated observations. Inferred automatically if omitted.",
        examples=[0],
    )
    control_value: Optional[Any] = Field(
        None,
        description="Alias for untreated_value.",
        examples=[0],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to explicitly drop rows with missing values across analysis columns. Default is false.",
        examples=[False],
    )
    near_zero_threshold: float = Field(
        default=0.05,
        description="Diagnostic threshold below which a score is flagged as near-zero.",
        examples=[0.05],
    )
    near_one_threshold: float = Field(
        default=0.95,
        description="Diagnostic threshold above which a score is flagged as near-one.",
        examples=[0.95],
    )

    @field_validator("data")
    @classmethod
    def validate_data_list(cls, v: Any) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            raise ValueError("data must be a list of observation dictionaries.")
        if len(v) == 0:
            raise ValueError("data cannot be empty. Propensity score estimation requires observations.")
        for idx, row in enumerate(v):
            if not isinstance(row, dict):
                raise ValueError(f"Observation at index {idx} must be a dictionary, got {type(row).__name__}.")
        return v

    @field_validator("treatment_col")
    @classmethod
    def validate_treatment_col(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("treatment_col must be a non-empty string.")
        return v.strip()

    @field_validator("confounder_cols")
    @classmethod
    def validate_confounders(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("confounder_cols must be a list of column names.")
        if len(v) == 0:
            raise ValueError("confounder_cols must contain at least one column name.")
        clean_cols = []
        for idx, col in enumerate(v):
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"Confounder column at index {idx} must be a non-empty string.")
            clean_cols.append(col.strip())
        return clean_cols

    @field_validator("drop_missing")
    @classmethod
    def validate_drop_missing(cls, v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError("drop_missing must be a boolean.")
        return v

    @field_validator("near_zero_threshold", "near_one_threshold")
    @classmethod
    def validate_thresholds(cls, v: Any, info: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{info.field_name} must be a numeric float.")
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            raise ValueError(f"{info.field_name} must be a finite number.")
        if not (0.0 <= fv <= 1.0):
            raise ValueError(f"{info.field_name} must be within the probability range [0, 1].")
        return fv

    @model_validator(mode="after")
    def validate_model_consistency(self) -> "CausalPropensityScoreRequest":
        for col in self.confounder_cols:
            if col == self.treatment_col:
                raise ValueError(
                    f"Confounder column '{col}' cannot also be the treatment column."
                )
        if len(self.confounder_cols) != len(set(self.confounder_cols)):
            raise ValueError(
                f"confounder_cols contains duplicate column names: {self.confounder_cols}."
            )
        if self.near_zero_threshold >= self.near_one_threshold:
            raise ValueError(
                f"near_zero_threshold ({self.near_zero_threshold}) must be strictly less than "
                f"near_one_threshold ({self.near_one_threshold})."
            )
        return self


class CausalPropensityScoreResponse(BaseModel):
    estimator: str = Field(..., description="Name of the estimator used (logistic_propensity_score)")
    propensity_scores: List[float] = Field(..., description="Estimated propensity scores P(Treatment = treated | X) in observation order")
    scores: List[float] = Field(..., description="Alias for propensity_scores")
    observation_indices: List[Any] = Field(..., description="Original dataset row indices corresponding to each score")
    scores_by_index: Dict[Any, float] = Field(..., description="Mapping of observation index to estimated propensity score")
    n_total: int = Field(..., description="Total number of observations used in model fitting")
    n_treated: int = Field(..., description="Number of treated observations")
    n_control: int = Field(..., description="Number of control/untreated observations")
    n_untreated: int = Field(..., description="Alias for n_control")
    treatment_col: str = Field(..., description="Name of the treatment column")
    confounder_cols: List[str] = Field(..., description="List of confounder column names")
    numeric_confounders: List[str] = Field(..., description="List of continuous numeric confounder columns")
    categorical_confounders: List[str] = Field(..., description="List of categorical confounder columns")
    treated_value: Any = Field(..., description="Resolved value representing the treated condition")
    control_value: Any = Field(..., description="Resolved value representing the control condition")
    untreated_value: Any = Field(..., description="Alias for control_value")
    rows_dropped: int = Field(0, description="Number of rows dropped due to missing data")
    model_features: List[str] = Field(..., description="Feature column names included in the design matrix")
    model_coefficients: Dict[str, float] = Field(..., description="Estimated logistic regression coefficients")
    model_metadata: Dict[str, Any] = Field(..., description="Model optimization metadata, convergence status, and warnings")
    overlap_diagnostics: Dict[str, Any] = Field(..., description="Comprehensive overlap and common support diagnostics")
    unadjusted_covariate_balance: Dict[str, Any] = Field(..., description="Baseline unadjusted standardized mean differences")
    covariate_balance: Dict[str, Any] = Field(..., description="Alias for unadjusted_covariate_balance")
    balance_assessment_type: str = Field(..., description="Type of balance assessment (unadjusted)")
    post_adjustment_balance_note: Optional[str] = Field(None, description="Note regarding post-adjustment balance evaluation")
    categorical_balance_note: Optional[str] = Field(None, description="Note regarding categorical confounder balance")
    interpretation: str = Field(..., description="Interpretation of propensity score distribution and overlap")
    assumptions: str = Field(..., description="Educational explanation of identifying assumptions and confounding limitations")


@app.post(
    "/api/causal/propensity-scores",
    response_model=CausalPropensityScoreResponse,
    tags=["Causal Inference"],
    summary="Estimate propensity scores and overlap diagnostics",
)
async def propensity_scores_endpoint(
    payload: CausalPropensityScoreRequest,
) -> Dict[str, Any]:
    """Estimate propensity scores and compute overlap and balance diagnostics.

    Fits a binary logistic regression model of treatment assignment on observed confounders,
    returns the predicted conditional probability P(T = treated | X) for each observation,
    evaluates common support and positivity overlap diagnostics, and computes baseline
    unadjusted covariate balance (SMDs).
    """
    try:
        untreated_val = payload.untreated_value if payload.untreated_value is not None else payload.control_value
        results = estimate_propensity_scores(
            data=payload.data,
            treatment_col=payload.treatment_col,
            confounder_cols=payload.confounder_cols,
            treated_value=payload.treated_value,
            untreated_value=untreated_val,
            control_value=untreated_val,
            drop_missing=payload.drop_missing,
            near_zero_threshold=payload.near_zero_threshold,
            near_one_threshold=payload.near_one_threshold,
        )
        return _sanitize_for_json(results)
    except ValueError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )


class DatasetPropensityScoreRequest(BaseModel):
    treatment_col: str = Field(
        ...,
        description="Name of the binary treatment indicator column in the dataset.",
        examples=["treatment"],
    )
    confounder_cols: List[str] = Field(
        ...,
        description="Non-empty list of confounder/covariate column names to model treatment assignment.",
        examples=[["age", "income"]],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying treated observations. Inferred automatically if omitted.",
        examples=[1],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying control/untreated observations. Inferred automatically if omitted.",
        examples=[0],
    )
    control_value: Optional[Any] = Field(
        None,
        description="Alias for untreated_value.",
        examples=[0],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to explicitly drop rows with missing values across analysis columns. Default is false.",
        examples=[False],
    )
    near_zero_threshold: float = Field(
        default=0.05,
        description="Diagnostic threshold below which a score is flagged as near-zero.",
        examples=[0.05],
    )
    near_one_threshold: float = Field(
        default=0.95,
        description="Diagnostic threshold above which a score is flagged as near-one.",
        examples=[0.95],
    )

    @field_validator("treatment_col")
    @classmethod
    def validate_treatment_col(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("treatment_col must be a non-empty string.")
        return v.strip()

    @field_validator("confounder_cols")
    @classmethod
    def validate_confounders(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("confounder_cols must be a list of column names.")
        if len(v) == 0:
            raise ValueError("confounder_cols must contain at least one column name.")
        clean_cols = []
        for idx, col in enumerate(v):
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"Confounder column at index {idx} must be a non-empty string.")
            clean_cols.append(col.strip())
        return clean_cols

    @field_validator("drop_missing")
    @classmethod
    def validate_drop_missing(cls, v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError("drop_missing must be a boolean.")
        return v

    @field_validator("near_zero_threshold", "near_one_threshold")
    @classmethod
    def validate_thresholds(cls, v: Any, info: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{info.field_name} must be a numeric float.")
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            raise ValueError(f"{info.field_name} must be a finite number.")
        if not (0.0 <= fv <= 1.0):
            raise ValueError(f"{info.field_name} must be within the probability range [0, 1].")
        return fv

    @model_validator(mode="after")
    def validate_model_consistency(self) -> "DatasetPropensityScoreRequest":
        for col in self.confounder_cols:
            if col == self.treatment_col:
                raise ValueError(
                    f"Confounder column '{col}' cannot also be the treatment column."
                )
        if len(self.confounder_cols) != len(set(self.confounder_cols)):
            raise ValueError(
                f"confounder_cols contains duplicate column names: {self.confounder_cols}."
            )
        if self.near_zero_threshold >= self.near_one_threshold:
            raise ValueError(
                f"near_zero_threshold ({self.near_zero_threshold}) must be strictly less than "
                f"near_one_threshold ({self.near_one_threshold})."
            )
        return self


class DatasetCausalPropensityScoreResponse(CausalPropensityScoreResponse):
    dataset_id: str = Field(..., description="Unique opaque dataset identifier")
    dataset_filename: str = Field(..., description="Original filename of the stored dataset evaluated")
    total_dataset_rows: int = Field(..., description="Total rows in the stored dataset before dropping missing values")


@app.post(
    "/api/dataset/{dataset_id}/causal/propensity-scores",
    response_model=DatasetCausalPropensityScoreResponse,
    tags=["Causal Inference", "Datasets"],
    summary="Estimate propensity scores for a stored dataset",
)
async def dataset_propensity_scores_endpoint(
    dataset_id: str,
    payload: DatasetPropensityScoreRequest,
) -> Dict[str, Any]:
    """Estimate propensity scores and overlap diagnostics for a cataloged dataset on disk.

    Validates dataset access and column existence, loads only the relevant treatment
    and confounder columns into memory, enforces missing-data policy, fits a logistic
    regression assignment model, and computes overlap and baseline covariate balance diagnostics.
    """
    dataset_meta = default_dataset_manager.get_dataset(dataset_id)
    if dataset_meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found or has expired.",
        )

    file_path = default_dataset_manager.get_dataset_file_path(dataset_id)
    if file_path is None or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file for '{dataset_id}' is not available on disk.",
        )

    available_cols = [c.name for c in dataset_meta.columns]
    if payload.treatment_col not in available_cols:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=(
                f"Treatment column '{payload.treatment_col}' not found in dataset. "
                f"Available columns: {available_cols}"
            ),
        )
    for c_col in payload.confounder_cols:
        if c_col not in available_cols:
            raise HTTPException(
                status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                detail=(
                    f"Confounder column '{c_col}' not found in dataset. "
                    f"Available columns: {available_cols}"
                ),
            )

    if dataset_meta.row_count == 0:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Dataset is empty (0 rows). Causal inference requires observations in both treatment groups.",
        )

    cols_to_read = [payload.treatment_col] + payload.confounder_cols

    try:
        df = pd.read_csv(
            file_path,
            usecols=cols_to_read,
            na_values=list(MISSING_TOKENS),
            keep_default_na=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=f"Failed to read dataset columns: {exc}",
        )

    if df.empty:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Dataset is empty (0 rows). Causal inference requires observations in both treatment groups.",
        )

    resolved_treated = _coerce_label_to_series_dtype(payload.treated_value, df[payload.treatment_col])
    resolved_untreated_val = payload.untreated_value if payload.untreated_value is not None else payload.control_value
    resolved_untreated = _coerce_label_to_series_dtype(resolved_untreated_val, df[payload.treatment_col])

    try:
        results = estimate_propensity_scores(
            data=df,
            treatment_col=payload.treatment_col,
            confounder_cols=payload.confounder_cols,
            treated_value=resolved_treated,
            untreated_value=resolved_untreated,
            control_value=resolved_untreated,
            drop_missing=payload.drop_missing,
            near_zero_threshold=payload.near_zero_threshold,
            near_one_threshold=payload.near_one_threshold,
        )
        results["dataset_id"] = dataset_id
        results["dataset_filename"] = dataset_meta.filename
        results["total_dataset_rows"] = dataset_meta.row_count
        return _sanitize_for_json(results)
    except ValueError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )



class CausalIPTWATERequest(BaseModel):
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Tabular dataset as a non-empty list of observation record dictionaries.",
        examples=[
            [
                {"treatment": 1, "outcome": 10.0, "x": 1.0, "z": 20.0},
                {"treatment": 1, "outcome": 12.0, "x": 2.0, "z": 25.0},
                {"treatment": 1, "outcome": 14.0, "x": 3.0, "z": 30.0},
                {"treatment": 1, "outcome": 16.0, "x": 4.0, "z": 35.0},
                {"treatment": 0, "outcome": 5.0,  "x": 1.0, "z": 22.0},
                {"treatment": 0, "outcome": 7.0,  "x": 2.0, "z": 24.0},
                {"treatment": 0, "outcome": 9.0,  "x": 3.0, "z": 28.0},
                {"treatment": 0, "outcome": 11.0, "x": 4.0, "z": 32.0},
            ]
        ],
    )
    treatment_col: str = Field(
        ...,
        description="Name of the binary treatment indicator column in the dataset.",
        examples=["treatment"],
    )
    outcome_col: str = Field(
        ...,
        description="Name of the numeric outcome column in the dataset.",
        examples=["outcome"],
    )
    confounder_cols: List[str] = Field(
        ...,
        description="Non-empty list of observed confounder column names.",
        examples=[["x", "z"]],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying treated observations. Inferred automatically if omitted.",
        examples=[1],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying control/untreated observations. Inferred automatically if omitted.",
        examples=[0],
    )
    control_value: Optional[Any] = Field(
        None,
        description="Alias for untreated_value.",
        examples=[0],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to explicitly drop rows with missing values across analysis columns. Default is false.",
        examples=[False],
    )
    extreme_weight_threshold: float = Field(
        default=20.0,
        description="Diagnostic threshold above which a weight is flagged as extreme.",
        examples=[20.0],
    )
    truncate_weights: bool = Field(
        default=False,
        description="Whether to truncate (clip) weights above max_weight. Default is false.",
        examples=[False],
    )
    max_weight: Optional[float] = Field(
        None,
        description="Maximum weight threshold used if truncate_weights is true.",
        examples=[10.0],
    )

    @field_validator("data")
    @classmethod
    def validate_data_list(cls, v: Any) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            raise ValueError("data must be a list of observation dictionaries.")
        if len(v) == 0:
            raise ValueError("data cannot be empty. Causal inference requires observations.")
        for idx, row in enumerate(v):
            if not isinstance(row, dict):
                raise ValueError(f"Observation at index {idx} must be a dictionary, got {type(row).__name__}.")
        return v

    @field_validator("treatment_col", "outcome_col")
    @classmethod
    def validate_column_names(cls, v: Any, info: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return v.strip()

    @field_validator("confounder_cols")
    @classmethod
    def validate_confounders(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("confounder_cols must be a list of column names.")
        if len(v) == 0:
            raise ValueError("confounder_cols must contain at least one column name.")
        clean_cols = []
        for idx, col in enumerate(v):
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"Confounder column at index {idx} must be a non-empty string.")
            clean_cols.append(col.strip())
        return clean_cols

    @field_validator("drop_missing", "truncate_weights")
    @classmethod
    def validate_booleans(cls, v: Any, info: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError(f"{info.field_name} must be a boolean.")
        return v

    @field_validator("extreme_weight_threshold", mode="before")
    @classmethod
    def validate_extreme_weight_threshold(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("extreme_weight_threshold must be a numeric float.")
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv) or fv <= 0:
            raise ValueError("extreme_weight_threshold must be a positive finite number (> 0).")
        return fv

    @field_validator("max_weight", mode="before")
    @classmethod
    def validate_max_weight(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("max_weight must be a numeric float.")
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv) or fv <= 0:
            raise ValueError("max_weight must be a positive finite number (> 0).")
        return fv

    @model_validator(mode="after")
    def validate_model_consistency(self) -> "CausalIPTWATERequest":
        if self.treatment_col == self.outcome_col:
            raise ValueError(
                f"treatment_col and outcome_col cannot be the same column ('{self.treatment_col}')."
            )
        for col in self.confounder_cols:
            if col == self.treatment_col:
                raise ValueError(
                    f"Confounder column '{col}' cannot also be the treatment column."
                )
            if col == self.outcome_col:
                raise ValueError(
                    f"Confounder column '{col}' cannot also be the outcome column."
                )
        if len(self.confounder_cols) != len(set(self.confounder_cols)):
            raise ValueError(
                f"confounder_cols contains duplicate column names: {self.confounder_cols}."
            )
        if self.truncate_weights and self.max_weight is not None and self.max_weight <= 0:
            raise ValueError(
                f"max_weight must be positive when truncate_weights is enabled, got {self.max_weight}."
            )
        return self


class CausalIPTWATEResponse(BaseModel):
    estimator: str = Field(..., description="Name of the estimator used (iptw_ate)")
    iptw_ate: float = Field(..., description="Estimated IPTW Average Treatment Effect (treated minus control)")
    ate: float = Field(..., description="Alias for iptw_ate")
    weighted_mean_outcome_treated: float = Field(..., description="Weighted mean outcome of the treated group")
    weighted_mean_outcome_control: float = Field(..., description="Weighted mean outcome of the control/untreated group")
    weighted_mean_treated: Optional[float] = Field(None, description="Alias for weighted_mean_outcome_treated")
    weighted_mean_control: Optional[float] = Field(None, description="Alias for weighted_mean_outcome_control")
    unadjusted_ate: float = Field(..., description="Unadjusted difference in sample means (treated minus control)")
    raw_difference: float = Field(..., description="Alias for unadjusted_ate")
    difference_in_means: float = Field(..., description="Alias for unadjusted_ate")
    n_total: int = Field(..., description="Total number of observations used in estimation")
    n_treated: int = Field(..., description="Number of treated observations")
    n_control: int = Field(..., description="Number of control/untreated observations")
    n_untreated: int = Field(..., description="Alias for n_control")
    treatment_col: str = Field(..., description="Name of the treatment column")
    outcome_col: str = Field(..., description="Name of the outcome column")
    confounder_cols: List[str] = Field(..., description="List of confounder column names")
    numeric_confounders: List[str] = Field(..., description="List of continuous numeric confounder columns")
    categorical_confounders: List[str] = Field(..., description="List of categorical confounder columns")
    treated_value: Any = Field(..., description="Resolved value representing the treated condition")
    control_value: Any = Field(..., description="Resolved value representing the control condition")
    untreated_value: Any = Field(..., description="Alias for control_value")
    rows_dropped: int = Field(0, description="Number of rows dropped due to missing data")
    propensity_score_summary: Dict[str, float] = Field(..., description="Summary statistics of estimated propensity scores")
    weight_summary: Dict[str, Any] = Field(..., description="Summary statistics and diagnostic flags for IPTW weights")
    weights: List[float] = Field(..., description="IPTW weights assigned to each observation")
    ess_treated: float = Field(..., description="Kish's effective sample size for the treated group")
    ess_control: float = Field(..., description="Kish's effective sample size for the control group")
    ess_total: float = Field(..., description="Total effective sample size (ess_treated + ess_control)")
    effective_sample_size_treated: Optional[float] = Field(None, description="Alias for ess_treated")
    effective_sample_size_control: Optional[float] = Field(None, description="Alias for ess_control")
    effective_sample_size_total: Optional[float] = Field(None, description="Alias for ess_total")
    ess_treated_fraction: float = Field(..., description="Ratio of effective sample size to nominal size for treated group")
    ess_control_fraction: float = Field(..., description="Ratio of effective sample size to nominal size for control group")
    overlap_diagnostics: Dict[str, Any] = Field(..., description="Comprehensive overlap and common support diagnostics")
    covariate_balance_before: Dict[str, Any] = Field(..., description="Covariate standardized mean differences before weighting")
    covariate_balance_after: Dict[str, Any] = Field(..., description="Covariate standardized mean differences after weighting")
    balance_summary: Dict[str, Any] = Field(..., description="Summary of covariate balance improvement after weighting")
    covariate_balance: Optional[Dict[str, Any]] = Field(None, description="Convenience mapping containing before, after, and summary balance")
    warnings: List[str] = Field(..., description="Diagnostic warnings regarding extreme weights, overlap, or sample size")
    interpretation: str = Field(..., description="Concise interpretation of IPTW effect, effective sample sizes, and balance")
    assumptions: str = Field(..., description="Educational explanation of identifying assumptions and confounding limitations")


@app.post(
    "/api/causal/iptw-ate",
    response_model=CausalIPTWATEResponse,
    tags=["Causal Inference"],
    summary="Estimate Average Treatment Effect (ATE) using IPTW",
)
async def iptw_ate_endpoint(
    payload: CausalIPTWATERequest,
) -> Dict[str, Any]:
    """Estimate the Average Treatment Effect (ATE) using Inverse Probability of Treatment Weighting (IPTW).

    Fits a propensity score model to estimate observation assignment probabilities, computes
    unstabilized IPTW weights (1 / p_i), calculates Hájek-style weighted outcome means for
    treated and control groups, computes effective sample sizes (ESS), and evaluates covariate
    balance improvements via standardized mean differences (SMDs) before and after weighting.
    """
    try:
        untreated_val = payload.untreated_value if payload.untreated_value is not None else payload.control_value
        results = estimate_iptw_ate(
            data=payload.data,
            treatment_col=payload.treatment_col,
            outcome_col=payload.outcome_col,
            confounder_cols=payload.confounder_cols,
            treated_value=payload.treated_value,
            untreated_value=untreated_val,
            control_value=untreated_val,
            drop_missing=payload.drop_missing,
            extreme_weight_threshold=payload.extreme_weight_threshold,
            truncate_weights=payload.truncate_weights,
            max_weight=payload.max_weight,
        )
        # Populate convenient aliases for response consistency
        results["weighted_mean_treated"] = results["weighted_mean_outcome_treated"]
        results["weighted_mean_control"] = results["weighted_mean_outcome_control"]
        results["effective_sample_size_treated"] = results["ess_treated"]
        results["effective_sample_size_control"] = results["ess_control"]
        results["effective_sample_size_total"] = results["ess_total"]
        results["covariate_balance"] = {
            "before": results["covariate_balance_before"],
            "after": results["covariate_balance_after"],
            "summary": results["balance_summary"],
        }
        return _sanitize_for_json(results)
    except ValueError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )


class DatasetIPTWATERequest(BaseModel):
    treatment_col: str = Field(
        ...,
        description="Name of the binary treatment indicator column in the dataset.",
        examples=["treatment"],
    )
    outcome_col: str = Field(
        ...,
        description="Name of the numeric outcome column in the dataset.",
        examples=["outcome"],
    )
    confounder_cols: List[str] = Field(
        ...,
        description="Non-empty list of observed confounder column names.",
        examples=[["x", "z"]],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying treated observations. Inferred automatically if omitted.",
        examples=[1],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Optional label identifying control/untreated observations. Inferred automatically if omitted.",
        examples=[0],
    )
    control_value: Optional[Any] = Field(
        None,
        description="Alias for untreated_value.",
        examples=[0],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to explicitly drop rows with missing values across analysis columns. Default is false.",
        examples=[False],
    )
    extreme_weight_threshold: float = Field(
        default=20.0,
        description="Diagnostic threshold above which a weight is flagged as extreme.",
        examples=[20.0],
    )
    truncate_weights: bool = Field(
        default=False,
        description="Whether to truncate (clip) weights above max_weight. Default is false.",
        examples=[False],
    )
    max_weight: Optional[float] = Field(
        None,
        description="Maximum weight threshold used if truncate_weights is true.",
        examples=[10.0],
    )

    @field_validator("treatment_col", "outcome_col")
    @classmethod
    def validate_column_names(cls, v: Any, info: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return v.strip()

    @field_validator("confounder_cols")
    @classmethod
    def validate_confounders(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("confounder_cols must be a list of column names.")
        if len(v) == 0:
            raise ValueError("confounder_cols must contain at least one column name.")
        clean_cols = []
        for idx, col in enumerate(v):
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"Confounder column at index {idx} must be a non-empty string.")
            clean_cols.append(col.strip())
        return clean_cols

    @field_validator("drop_missing", "truncate_weights")
    @classmethod
    def validate_booleans(cls, v: Any, info: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError(f"{info.field_name} must be a boolean.")
        return v

    @field_validator("extreme_weight_threshold", mode="before")
    @classmethod
    def validate_extreme_weight_threshold(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("extreme_weight_threshold must be a numeric float.")
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv) or fv <= 0:
            raise ValueError("extreme_weight_threshold must be a positive finite number (> 0).")
        return fv

    @field_validator("max_weight", mode="before")
    @classmethod
    def validate_max_weight(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("max_weight must be a numeric float.")
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv) or fv <= 0:
            raise ValueError("max_weight must be a positive finite number (> 0).")
        return fv

    @model_validator(mode="after")
    def validate_model_consistency(self) -> "DatasetIPTWATERequest":
        if self.treatment_col == self.outcome_col:
            raise ValueError(
                f"treatment_col and outcome_col cannot be the same column ('{self.treatment_col}')."
            )
        for col in self.confounder_cols:
            if col == self.treatment_col:
                raise ValueError(
                    f"Confounder column '{col}' cannot also be the treatment column."
                )
            if col == self.outcome_col:
                raise ValueError(
                    f"Confounder column '{col}' cannot also be the outcome column."
                )
        if len(self.confounder_cols) != len(set(self.confounder_cols)):
            raise ValueError(
                f"confounder_cols contains duplicate column names: {self.confounder_cols}."
            )
        if self.truncate_weights and self.max_weight is not None and self.max_weight <= 0:
            raise ValueError(
                f"max_weight must be positive when truncate_weights is enabled, got {self.max_weight}."
            )
        return self


class DatasetCausalIPTWATEResponse(CausalIPTWATEResponse):
    dataset_id: str = Field(..., description="Unique opaque dataset identifier")
    dataset_filename: str = Field(..., description="Original filename of the stored dataset evaluated")
    total_dataset_rows: int = Field(..., description="Total rows in the stored dataset before dropping missing values")


@app.post(
    "/api/dataset/{dataset_id}/causal/iptw-ate",
    response_model=DatasetCausalIPTWATEResponse,
    tags=["Causal Inference", "Datasets"],
    summary="Estimate IPTW ATE for a stored dataset",
)
async def dataset_iptw_ate_endpoint(
    dataset_id: str,
    payload: DatasetIPTWATERequest,
) -> Dict[str, Any]:
    """Estimate the Average Treatment Effect (ATE) via Inverse Probability of Treatment Weighting (IPTW) for a stored dataset.

    Validates dataset access and column existence, loads only the relevant treatment,
    outcome, and confounder columns into memory, enforces missing-data policy,
    fits a logistic propensity score model, calculates unstabilized (and optionally truncated)
    weights, evaluates Hájek weighted outcome means and effective sample sizes, and measures
    covariate balance improvements across pre- and post-weighting SMDs.
    """
    dataset_meta = default_dataset_manager.get_dataset(dataset_id)
    if dataset_meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found or has expired.",
        )

    file_path = default_dataset_manager.get_dataset_file_path(dataset_id)
    if file_path is None or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file for '{dataset_id}' is not available on disk.",
        )

    available_cols = [c.name for c in dataset_meta.columns]
    if payload.treatment_col not in available_cols:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=(
                f"Treatment column '{payload.treatment_col}' not found in dataset. "
                f"Available columns: {available_cols}"
            ),
        )
    if payload.outcome_col not in available_cols:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=(
                f"Outcome column '{payload.outcome_col}' not found in dataset. "
                f"Available columns: {available_cols}"
            ),
        )
    for c_col in payload.confounder_cols:
        if c_col not in available_cols:
            raise HTTPException(
                status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                detail=(
                    f"Confounder column '{c_col}' not found in dataset. "
                    f"Available columns: {available_cols}"
                ),
            )

    if dataset_meta.row_count == 0:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Dataset is empty (0 rows). Causal inference requires observations in both treatment groups.",
        )

    cols_to_read = [payload.treatment_col, payload.outcome_col] + payload.confounder_cols

    try:
        df = pd.read_csv(
            file_path,
            usecols=cols_to_read,
            na_values=list(MISSING_TOKENS),
            keep_default_na=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=f"Failed to read dataset columns: {exc}",
        )

    if df.empty:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Dataset is empty (0 rows). Causal inference requires observations in both treatment groups.",
        )

    resolved_treated = _coerce_label_to_series_dtype(payload.treated_value, df[payload.treatment_col])
    untreated_val = payload.untreated_value if payload.untreated_value is not None else payload.control_value
    resolved_untreated = _coerce_label_to_series_dtype(untreated_val, df[payload.treatment_col])

    try:
        results = estimate_iptw_ate(
            data=df,
            treatment_col=payload.treatment_col,
            outcome_col=payload.outcome_col,
            confounder_cols=payload.confounder_cols,
            treated_value=resolved_treated,
            untreated_value=resolved_untreated,
            control_value=resolved_untreated,
            drop_missing=payload.drop_missing,
            extreme_weight_threshold=payload.extreme_weight_threshold,
            truncate_weights=payload.truncate_weights,
            max_weight=payload.max_weight,
        )
        results["dataset_id"] = dataset_id
        results["dataset_filename"] = dataset_meta.filename
        results["total_dataset_rows"] = dataset_meta.row_count
        results["weighted_mean_treated"] = results["weighted_mean_outcome_treated"]
        results["weighted_mean_control"] = results["weighted_mean_outcome_control"]
        results["effective_sample_size_treated"] = results["ess_treated"]
        results["effective_sample_size_control"] = results["ess_control"]
        results["effective_sample_size_total"] = results["ess_total"]
        results["covariate_balance"] = {
            "before": results["covariate_balance_before"],
            "after": results["covariate_balance_after"],
            "summary": results["balance_summary"],
        }
        return _sanitize_for_json(results)
    except ValueError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )


class CausalDifferenceInDifferencesRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Tabular dataset as a non-empty list of observation record dictionaries.",
        examples=[
            [
                {"treatment": 1, "time": 0, "outcome": 10.0},
                {"treatment": 1, "time": 0, "outcome": 12.0},
                {"treatment": 1, "time": 1, "outcome": 18.0},
                {"treatment": 1, "time": 1, "outcome": 22.0},
                {"treatment": 0, "time": 0, "outcome": 5.0},
                {"treatment": 0, "time": 0, "outcome": 7.0},
                {"treatment": 0, "time": 1, "outcome": 8.0},
                {"treatment": 0, "time": 1, "outcome": 10.0},
            ]
        ],
    )
    treatment_col: str = Field(
        ...,
        description="Name of the treatment group indicator column distinguishing treated from control units.",
        examples=["treatment"],
    )
    time_col: Optional[str] = Field(
        None,
        description="Name of the time-period indicator column distinguishing pre-intervention from post-intervention.",
        examples=["time"],
    )
    time_period_col: Optional[str] = Field(
        None,
        description="Alias for time_col.",
        examples=["time"],
    )
    outcome_col: str = Field(
        ...,
        description="Name of the numeric outcome column.",
        examples=["outcome"],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Label identifying the treated group. Inferred automatically if omitted.",
        examples=[1],
    )
    control_value: Optional[Any] = Field(
        None,
        description="Label identifying the control group. Inferred automatically if omitted.",
        examples=[0],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Alias for control_value.",
        examples=[0],
    )
    pre_value: Optional[Any] = Field(
        None,
        description="Label identifying the pre-intervention period. Inferred automatically if omitted.",
        examples=[0],
    )
    post_value: Optional[Any] = Field(
        None,
        description="Label identifying the post-intervention period. Inferred automatically if omitted.",
        examples=[1],
    )
    pre_period_value: Optional[Any] = Field(
        None,
        description="Alias for pre_value.",
        examples=[0],
    )
    post_period_value: Optional[Any] = Field(
        None,
        description="Alias for post_value.",
        examples=[1],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to drop rows with missing values in treatment, time, or outcome columns. Default is false.",
        examples=[False],
    )

    @field_validator("data")
    @classmethod
    def validate_data_list(cls, v: Any) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            raise ValueError("data must be a list of observation dictionaries.")
        if len(v) == 0:
            raise ValueError("data cannot be empty. Causal inference requires observations.")
        for idx, row in enumerate(v):
            if not isinstance(row, dict):
                raise ValueError(f"Observation at index {idx} must be a dictionary, got {type(row).__name__}.")
        return v

    @field_validator("treatment_col", "outcome_col")
    @classmethod
    def validate_required_column_names(cls, v: Any, info: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return v.strip()

    @field_validator("time_col", "time_period_col")
    @classmethod
    def validate_optional_time_column_name(cls, v: Any, info: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string when provided.")
        return v.strip()

    @field_validator("drop_missing")
    @classmethod
    def validate_drop_missing(cls, v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError("drop_missing must be a boolean.")
        return v

    @model_validator(mode="after")
    def validate_model_consistency(self) -> "CausalDifferenceInDifferencesRequest":
        # Resolve time column alias
        actual_time_col = self.time_col or self.time_period_col
        if actual_time_col is None:
            raise ValueError("time_col (or time_period_col) must be provided as a non-empty string.")
        self.time_col = actual_time_col

        # Role conflicts
        if self.treatment_col == self.outcome_col:
            raise ValueError(
                f"treatment_col and outcome_col cannot be the same column ('{self.treatment_col}')."
            )
        if self.time_col == self.outcome_col:
            raise ValueError(
                f"time_col and outcome_col cannot be the same column ('{self.time_col}')."
            )
        if self.treatment_col == self.time_col:
            raise ValueError(
                f"treatment_col and time_col cannot be the same column ('{self.treatment_col}')."
            )

        # Label equality checks
        actual_control = self.control_value if self.control_value is not None else self.untreated_value
        if self.treated_value is not None and actual_control is not None and self.treated_value == actual_control:
            raise ValueError("treated_value and control_value cannot be identical.")

        actual_pre = self.pre_value if self.pre_value is not None else self.pre_period_value
        actual_post = self.post_value if self.post_value is not None else self.post_period_value
        if actual_pre is not None and actual_post is not None and actual_pre == actual_post:
            raise ValueError("pre_value and post_value cannot be identical.")

        return self


class CausalDifferenceInDifferencesResponse(BaseModel):
    estimator: str = Field(..., description="Name of the estimator used (difference_in_differences)")
    did_estimate: float = Field(..., description="Difference-in-Differences treatment effect estimate: (T_post - T_pre) - (C_post - C_pre)")
    did: float = Field(..., description="Alias for did_estimate")
    ate: float = Field(..., description="Alias for did_estimate")
    treatment_mean_pre: float = Field(..., description="Mean outcome for treated units before intervention")
    treatment_mean_post: float = Field(..., description="Mean outcome for treated units after intervention")
    control_mean_pre: float = Field(..., description="Mean outcome for control units before intervention")
    control_mean_post: float = Field(..., description="Mean outcome for control units after intervention")
    treated_mean_pre: float = Field(..., description="Alias for treatment_mean_pre")
    treated_mean_post: float = Field(..., description="Alias for treatment_mean_post")
    treatment_change: float = Field(..., description="Mean outcome change for treated units (post minus pre)")
    control_change: float = Field(..., description="Mean outcome change for control units (post minus pre)")
    delta_treatment: float = Field(..., description="Alias for treatment_change")
    delta_control: float = Field(..., description="Alias for control_change")
    delta_treated: float = Field(..., description="Alias for treatment_change")
    cell_counts: Dict[str, int] = Field(..., description="Observation counts across the four group-period cells")
    n_treatment_pre: int = Field(..., description="Count of treated observations in pre-period")
    n_treatment_post: int = Field(..., description="Count of treated observations in post-period")
    n_control_pre: int = Field(..., description="Count of control observations in pre-period")
    n_control_post: int = Field(..., description="Count of control observations in post-period")
    n_treated_pre: int = Field(..., description="Alias for n_treatment_pre")
    n_treated_post: int = Field(..., description="Alias for n_treatment_post")
    n_treated: int = Field(..., description="Total treated observations across both periods")
    n_control: int = Field(..., description="Total control observations across both periods")
    n_untreated: int = Field(..., description="Alias for n_control")
    n_total: int = Field(..., description="Total number of observations used in estimation")
    rows_dropped: int = Field(0, description="Number of rows dropped due to missing data")
    treatment_col: str = Field(..., description="Name of the treatment column")
    time_col: str = Field(..., description="Name of the time-period column")
    time_period_col: str = Field(..., description="Alias for time_col")
    outcome_col: str = Field(..., description="Name of the outcome column")
    treated_value: Any = Field(..., description="Resolved label representing the treated condition")
    control_value: Any = Field(..., description="Resolved label representing the control condition")
    untreated_value: Any = Field(..., description="Alias for control_value")
    pre_value: Any = Field(..., description="Resolved label representing the pre-intervention period")
    post_value: Any = Field(..., description="Resolved label representing the post-intervention period")
    interpretation: str = Field(..., description="Interpretation summarizing group changes and DiD effect")
    assumptions: str = Field(..., description="Educational explanation of identifying assumptions (parallel trends, no anticipation, SUTVA)")


@app.post(
    "/api/causal/difference-in-differences",
    response_model=CausalDifferenceInDifferencesResponse,
    tags=["Causal Inference"],
    summary="Estimate treatment effect using Difference-in-Differences (DiD)",
)
async def difference_in_differences_endpoint(
    payload: CausalDifferenceInDifferencesRequest,
) -> Dict[str, Any]:
    """Estimate treatment effect using the canonical two-group, two-period Difference-in-Differences estimator.

    Computes the change over time in the treatment group minus the change over time in the control group:
    DiD = (Ybar_{T,post} - Ybar_{T,pre}) - (Ybar_{C,post} - Ybar_{C,pre}).
    """
    try:
        resolved_time_col = payload.time_col or payload.time_period_col
        resolved_control = payload.control_value if payload.control_value is not None else payload.untreated_value
        resolved_pre = payload.pre_value if payload.pre_value is not None else payload.pre_period_value
        resolved_post = payload.post_value if payload.post_value is not None else payload.post_period_value

        results = estimate_difference_in_differences(
            data=payload.data,
            treatment_col=payload.treatment_col,
            time_col=resolved_time_col,
            outcome_col=payload.outcome_col,
            treated_value=payload.treated_value,
            control_value=resolved_control,
            pre_value=resolved_pre,
            post_value=resolved_post,
            drop_missing=payload.drop_missing,
        )
        return _sanitize_for_json(results)
    except ValueError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )


class DatasetDifferenceInDifferencesRequest(BaseModel):
    treatment_col: str = Field(
        ...,
        description="Name of the binary treatment-group column in the dataset.",
        examples=["treatment"],
    )
    time_col: Optional[str] = Field(
        None,
        description="Name of the two-period time indicator column in the dataset (pre/post).",
        examples=["time"],
    )
    time_period_col: Optional[str] = Field(
        None,
        description="Alias for time_col.",
        examples=["time"],
    )
    outcome_col: str = Field(
        ...,
        description="Name of the numeric outcome column in the dataset.",
        examples=["outcome"],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Label identifying the treated group. Inferred automatically if omitted.",
        examples=[1],
    )
    control_value: Optional[Any] = Field(
        None,
        description="Label identifying the control group. Inferred automatically if omitted.",
        examples=[0],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Alias for control_value.",
        examples=[0],
    )
    pre_value: Optional[Any] = Field(
        None,
        description="Label identifying the pre-intervention period. Inferred automatically if omitted.",
        examples=[0],
    )
    post_value: Optional[Any] = Field(
        None,
        description="Label identifying the post-intervention period. Inferred automatically if omitted.",
        examples=[1],
    )
    pre_period_value: Optional[Any] = Field(
        None,
        description="Alias for pre_value.",
        examples=[0],
    )
    post_period_value: Optional[Any] = Field(
        None,
        description="Alias for post_value.",
        examples=[1],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to explicitly drop rows with missing values in treatment, time, or outcome columns. Default is false.",
        examples=[False],
    )

    @field_validator("treatment_col", "outcome_col")
    @classmethod
    def validate_required_column_names(cls, v: Any, info: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return v.strip()

    @field_validator("time_col", "time_period_col")
    @classmethod
    def validate_optional_time_column_name(cls, v: Any, info: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string when provided.")
        return v.strip()

    @field_validator("drop_missing")
    @classmethod
    def validate_drop_missing(cls, v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError("drop_missing must be a boolean.")
        return v

    @model_validator(mode="after")
    def validate_model_consistency(self) -> "DatasetDifferenceInDifferencesRequest":
        # Resolve time column alias
        actual_time_col = self.time_col or self.time_period_col
        if actual_time_col is None:
            raise ValueError("time_col (or time_period_col) must be provided as a non-empty string.")
        self.time_col = actual_time_col

        # Role conflicts
        if self.treatment_col == self.outcome_col:
            raise ValueError(
                f"treatment_col and outcome_col cannot be the same column ('{self.treatment_col}')."
            )
        if self.time_col == self.outcome_col:
            raise ValueError(
                f"time_col and outcome_col cannot be the same column ('{self.time_col}')."
            )
        if self.treatment_col == self.time_col:
            raise ValueError(
                f"treatment_col and time_col cannot be the same column ('{self.treatment_col}')."
            )

        # Label equality checks
        actual_control = self.control_value if self.control_value is not None else self.untreated_value
        if self.treated_value is not None and actual_control is not None and self.treated_value == actual_control:
            raise ValueError("treated_value and control_value cannot be identical.")

        actual_pre = self.pre_value if self.pre_value is not None else self.pre_period_value
        actual_post = self.post_value if self.post_value is not None else self.post_period_value
        if actual_pre is not None and actual_post is not None and actual_pre == actual_post:
            raise ValueError("pre_value and post_value cannot be identical.")

        return self


class DatasetCausalDifferenceInDifferencesResponse(CausalDifferenceInDifferencesResponse):
    dataset_id: str = Field(..., description="Unique opaque dataset identifier")
    dataset_filename: str = Field(..., description="Original filename of the stored dataset evaluated")
    total_dataset_rows: int = Field(..., description="Total rows in the stored dataset before dropping missing values")


@app.post(
    "/api/dataset/{dataset_id}/causal/difference-in-differences",
    response_model=DatasetCausalDifferenceInDifferencesResponse,
    tags=["Causal Inference", "Datasets"],
    summary="Estimate Difference-in-Differences treatment effect for a stored dataset",
)
async def dataset_difference_in_differences_endpoint(
    dataset_id: str,
    payload: DatasetDifferenceInDifferencesRequest,
) -> Dict[str, Any]:
    """Estimate treatment effect using Difference-in-Differences (DiD) on a stored dataset.

    Validates dataset access and column existence, loads only the treatment, time, and outcome
    columns from disk, enforces missing data policy, resolves group and period labels,
    verifies that all four group-period cells contain observations, and computes the canonical
    2x2 Difference-in-Differences point estimate:
    DiD = (Ybar_{T,post} - Ybar_{T,pre}) - (Ybar_{C,post} - Ybar_{C,pre}).
    """
    dataset_meta = default_dataset_manager.get_dataset(dataset_id)
    if dataset_meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found or has expired.",
        )

    file_path = default_dataset_manager.get_dataset_file_path(dataset_id)
    if file_path is None or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file for '{dataset_id}' is not available on disk.",
        )

    available_cols = [c.name for c in dataset_meta.columns]
    actual_time_col = payload.time_col or payload.time_period_col

    if payload.treatment_col not in available_cols:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=(
                f"Treatment column '{payload.treatment_col}' not found in dataset. "
                f"Available columns: {available_cols}"
            ),
        )
    if actual_time_col not in available_cols:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=(
                f"Time column '{actual_time_col}' not found in dataset. "
                f"Available columns: {available_cols}"
            ),
        )
    if payload.outcome_col not in available_cols:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=(
                f"Outcome column '{payload.outcome_col}' not found in dataset. "
                f"Available columns: {available_cols}"
            ),
        )

    if dataset_meta.row_count == 0:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Dataset is empty (0 rows). Difference-in-Differences requires observations in all four cells.",
        )

    cols_to_read = [payload.treatment_col, actual_time_col, payload.outcome_col]

    try:
        df = pd.read_csv(
            file_path,
            usecols=cols_to_read,
            na_values=list(MISSING_TOKENS),
            keep_default_na=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=f"Failed to read dataset columns: {exc}",
        )

    if df.empty:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Dataset is empty (0 rows). Difference-in-Differences requires observations in all four cells.",
        )

    actual_control = payload.control_value if payload.control_value is not None else payload.untreated_value
    actual_pre = payload.pre_value if payload.pre_value is not None else payload.pre_period_value
    actual_post = payload.post_value if payload.post_value is not None else payload.post_period_value

    resolved_treated = _coerce_label_to_series_dtype(payload.treated_value, df[payload.treatment_col])
    resolved_control = _coerce_label_to_series_dtype(actual_control, df[payload.treatment_col])
    resolved_pre = _coerce_label_to_series_dtype(actual_pre, df[actual_time_col])
    resolved_post = _coerce_label_to_series_dtype(actual_post, df[actual_time_col])

    try:
        results = estimate_difference_in_differences(
            data=df,
            treatment_col=payload.treatment_col,
            time_col=actual_time_col,
            outcome_col=payload.outcome_col,
            treated_value=resolved_treated,
            control_value=resolved_control,
            pre_value=resolved_pre,
            post_value=resolved_post,
            drop_missing=payload.drop_missing,
        )
        results["dataset_id"] = dataset_id
        results["dataset_filename"] = dataset_meta.filename
        results["total_dataset_rows"] = dataset_meta.row_count
        return _sanitize_for_json(results)
    except ValueError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )



class CausalSensitivityAnalysisRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Tabular dataset records for multi-estimator sensitivity analysis.",
        examples=[
            [
                {"treatment": 1, "outcome": 25.0, "age": 30.0},
                {"treatment": 1, "outcome": 35.0, "age": 40.0},
                {"treatment": 0, "outcome": 15.0, "age": 25.0},
                {"treatment": 0, "outcome": 25.0, "age": 35.0},
            ]
        ],
    )
    treatment_col: Optional[str] = Field(
        None,
        description="Name of the binary treatment assignment column (required if data is provided).",
        examples=["treatment"],
    )
    outcome_col: Optional[str] = Field(
        None,
        description="Name of the numeric outcome column (required if data is provided).",
        examples=["outcome"],
    )
    confounder_cols: Optional[List[str]] = Field(
        None,
        description="Optional list of observed confounder column names for regression adjustment and IPTW.",
        examples=[["age"]],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Label identifying the treated group. Inferred automatically if omitted.",
        examples=[1],
    )
    control_value: Optional[Any] = Field(
        None,
        description="Label identifying the control group. Inferred automatically if omitted.",
        examples=[0],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Alias for control_value.",
        examples=[0],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to drop rows with missing values in required analysis columns. Default is False.",
        examples=[False],
    )
    reference_estimator: Optional[str] = Field(
        default="regression_adjusted",
        description="Primary reference estimator ('regression_adjusted', 'iptw', or 'unadjusted').",
        examples=["regression_adjusted"],
    )
    truncation_thresholds: Optional[List[float]] = Field(
        None,
        description="Optional sequence of strictly positive upper weight bounds for IPTW truncation sensitivity checks.",
        examples=[[10.0, 5.0, 3.0]],
    )
    hypothetical_biases: Optional[Union[float, List[float]]] = Field(
        None,
        description="Optional hypothetical signed bias values (b) for unmeasured confounding sensitivity scenarios: tau_adjusted = tau_observed - b.",
        examples=[[1.0, 2.0, -1.0]],
    )
    bias_values: Optional[Union[float, List[float]]] = Field(
        None,
        description="Alias for hypothetical_biases, or required bias values in standalone observed-estimate mode.",
        examples=[[1.0, 2.0, -1.0]],
    )
    observed_estimate: Optional[float] = Field(
        None,
        description="Observed effect estimate for standalone unmeasured confounding sensitivity analysis (used when data is omitted).",
        examples=[5.0],
    )
    confidence_interval: Optional[List[float]] = Field(
        None,
        description="Optional two-element confidence interval [lower, upper] for the observed estimate in standalone mode.",
        examples=[[3.0, 7.0]],
    )

    @field_validator("data")
    @classmethod
    def validate_data_list(cls, v: Any) -> Optional[List[Dict[str, Any]]]:
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("data must be a list of observation dictionaries.")
        if len(v) == 0:
            raise ValueError("data cannot be empty. Causal inference requires observations.")
        for idx, row in enumerate(v):
            if not isinstance(row, dict):
                raise ValueError(f"Observation at index {idx} must be a dictionary, got {type(row).__name__}.")
        return v

    @field_validator("treatment_col", "outcome_col")
    @classmethod
    def validate_column_name(cls, v: Any, info: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string when provided.")
        return v.strip()

    @field_validator("confounder_cols")
    @classmethod
    def validate_confounder_names(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if not isinstance(v, (list, tuple)):
            raise ValueError("confounder_cols must be a list of column names.")
        cleaned = []
        for idx, col in enumerate(v):
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"confounder_cols element at index {idx} must be a non-empty string.")
            cleaned.append(col.strip())
        return cleaned

    @field_validator("truncation_thresholds", mode="before")
    @classmethod
    def validate_truncation_thresholds(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (list, tuple)):
            raise ValueError("truncation_thresholds must be a list of strictly positive numbers.")
        if len(v) == 0:
            raise ValueError("truncation_thresholds cannot be empty when provided.")
        validated = []
        for idx, thresh in enumerate(v):
            if thresh is None or isinstance(thresh, bool) or not isinstance(thresh, (int, float)):
                raise ValueError(f"truncation_thresholds element at index {idx} must be a numeric float or int.")
            if math.isnan(thresh) or math.isinf(thresh):
                raise ValueError(f"truncation_thresholds element at index {idx} cannot be NaN or infinite.")
            if float(thresh) <= 0.0:
                raise ValueError(f"truncation_thresholds element at index {idx} must be strictly positive, got {thresh}.")
            validated.append(float(thresh))
        return validated

    @field_validator("observed_estimate", mode="before")
    @classmethod
    def validate_observed_estimate(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("observed_estimate cannot be a boolean.")
        if not isinstance(v, (int, float)):
            raise ValueError(f"observed_estimate must be a numeric float or int, got {type(v).__name__}.")
        if math.isnan(v) or math.isinf(v):
            raise ValueError("observed_estimate cannot be NaN or infinite.")
        return float(v)

    @field_validator("hypothetical_biases", "bias_values", mode="before")
    @classmethod
    def validate_biases(cls, v: Any, info: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError(f"{info.field_name} cannot be a boolean.")
        if isinstance(v, (int, float)):
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"{info.field_name} cannot be NaN or infinite.")
            return float(v)
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                raise ValueError(f"{info.field_name} cannot be empty when provided.")
            validated = []
            for idx, b in enumerate(v):
                if b is None or isinstance(b, bool) or not isinstance(b, (int, float)):
                    raise ValueError(f"{info.field_name} element at index {idx} must be a numeric float or int.")
                if math.isnan(b) or math.isinf(b):
                    raise ValueError(f"{info.field_name} element at index {idx} cannot be NaN or infinite.")
                validated.append(float(b))
            return validated
        raise ValueError(f"{info.field_name} must be a number or list of numbers.")

    @field_validator("confidence_interval", mode="before")
    @classmethod
    def validate_ci(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (list, tuple)):
            raise ValueError("confidence_interval must be a list of 2 numbers.")
        if len(v) != 2:
            raise ValueError(f"confidence_interval must contain exactly 2 elements [lower, upper], got {len(v)}.")
        for idx, bound in enumerate(v):
            if bound is None or isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise ValueError(f"confidence_interval bound at index {idx} must be a numeric float or int.")
            if math.isnan(bound) or math.isinf(bound):
                raise ValueError(f"confidence_interval bound at index {idx} cannot be NaN or infinite.")
        l_val, u_val = float(v[0]), float(v[1])
        if l_val > u_val:
            raise ValueError(f"confidence_interval lower bound ({l_val}) cannot exceed upper bound ({u_val}).")
        return [l_val, u_val]

    @field_validator("drop_missing")
    @classmethod
    def validate_drop_missing(cls, v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError("drop_missing must be a boolean.")
        return v

    @model_validator(mode="after")
    def validate_sensitivity_request(self) -> "CausalSensitivityAnalysisRequest":
        if self.data is not None:
            if not self.treatment_col:
                raise ValueError("treatment_col must be provided as a non-empty string when data is provided.")
            if not self.outcome_col:
                raise ValueError("outcome_col must be provided as a non-empty string when data is provided.")
            if self.treatment_col == self.outcome_col:
                raise ValueError(f"treatment_col and outcome_col cannot be the same column ('{self.treatment_col}').")
            if self.confounder_cols:
                if self.treatment_col in self.confounder_cols:
                    raise ValueError(f"treatment_col ('{self.treatment_col}') cannot also appear in confounder_cols.")
                if self.outcome_col in self.confounder_cols:
                    raise ValueError(f"outcome_col ('{self.outcome_col}') cannot also appear in confounder_cols.")
            if self.reference_estimator:
                norm_ref = self.reference_estimator.lower().strip()
                valid_refs = {"regression_adjusted", "iptw", "unadjusted"}
                if norm_ref not in valid_refs:
                    raise ValueError(
                        f"Unsupported reference_estimator: {self.reference_estimator!r}. "
                        f"Expected one of: {sorted(list(valid_refs))}."
                    )
                self.reference_estimator = norm_ref
            # Label checks
            actual_ctrl = self.control_value if self.control_value is not None else self.untreated_value
            if self.treated_value is not None and actual_ctrl is not None and self.treated_value == actual_ctrl:
                raise ValueError("treated_value and control_value cannot be identical.")
        else:
            if self.observed_estimate is None:
                raise ValueError(
                    "Either 'data' (with 'treatment_col' and 'outcome_col') or 'observed_estimate' "
                    "(with 'hypothetical_biases' or 'bias_values') must be provided."
                )
            resolved_biases = self.hypothetical_biases if self.hypothetical_biases is not None else self.bias_values
            if resolved_biases is None:
                raise ValueError(
                    "bias_values (or hypothetical_biases) must be provided when analyzing unmeasured confounding for an observed estimate."
                )
        return self


class CausalSensitivityAnalysisResponse(BaseModel):
    reference_estimator: Optional[str] = Field(
        None,
        description="Name of the baseline reference estimator used ('regression_adjusted', 'iptw', or 'unadjusted').",
    )
    reference_estimate: Optional[float] = Field(
        None,
        description="Point estimate of the causal treatment effect from the reference estimator.",
    )
    observed_estimate: Optional[float] = Field(
        None,
        description="Observed effect estimate analyzed (alias for reference_estimate in dataset mode).",
    )
    reference_confidence_interval: Optional[List[float]] = Field(
        None,
        description="Confidence interval [lower, upper] from reference estimator if available.",
    )
    confidence_interval: Optional[List[float]] = Field(
        None,
        description="Confidence interval [lower, upper] of observed estimate in standalone mode.",
    )
    estimator_comparisons: Optional[Dict[str, Any]] = Field(
        None,
        description="Cross-estimator point estimates and differences from reference (unadjusted, regression_adjusted, iptw).",
    )
    iptw_truncation_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="IPTW weight truncation robustness analysis across requested thresholds.",
    )
    unmeasured_confounding_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Sensitivity results and scenarios for hypothetical unmeasured confounding.",
    )
    bias_scenarios: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of evaluated hypothetical bias scenarios with adjusted estimates and intervals.",
    )
    sign_convention: Optional[str] = Field(
        None,
        description="Explanation of the bias adjustment sign convention: tau_adjusted = tau_observed - b.",
    )
    treatment_col: Optional[str] = Field(
        None,
        description="Name of the treatment column analyzed.",
    )
    outcome_col: Optional[str] = Field(
        None,
        description="Name of the outcome column analyzed.",
    )
    confounder_cols: Optional[List[str]] = Field(
        None,
        description="List of observed confounder columns included in multi-estimator models.",
    )
    warnings: Optional[List[str]] = Field(
        None,
        description="List of analytical warnings or model divergence notices.",
    )
    interpretation: str = Field(
        ...,
        description="Summary narrative explaining sensitivity findings and robustness checks.",
    )
    assumptions: str = Field(
        ...,
        description="Educational explanation of sensitivity analysis assumptions, limitations, and caveats.",
    )


@app.post(
    "/api/causal/sensitivity-analysis",
    response_model=CausalSensitivityAnalysisResponse,
    tags=["Causal Inference"],
    summary="Conduct causal sensitivity analysis across estimators and unmeasured confounding",
)
async def causal_sensitivity_analysis_endpoint(
    payload: CausalSensitivityAnalysisRequest,
) -> Dict[str, Any]:
    """Perform causal sensitivity analysis and robustness checks.

    Evaluates:
    1. Multi-estimator comparison (unadjusted difference in means, regression adjustment, IPTW).
    2. Weight truncation robustness checks for IPTW across user-specified thresholds.
    3. Hypothetical bias-adjustment sensitivity analysis for potential unmeasured confounding.
    """
    try:
        biases = (
            payload.hypothetical_biases
            if payload.hypothetical_biases is not None
            else payload.bias_values
        )
        resolved_control = (
            payload.control_value
            if payload.control_value is not None
            else payload.untreated_value
        )

        if payload.data is not None:
            results = run_causal_sensitivity_analysis(
                data=payload.data,
                treatment_col=payload.treatment_col,
                outcome_col=payload.outcome_col,
                confounder_cols=payload.confounder_cols,
                treated_value=payload.treated_value,
                control_value=resolved_control,
                drop_missing=payload.drop_missing,
                reference_estimator=payload.reference_estimator or "regression_adjusted",
                truncation_thresholds=payload.truncation_thresholds,
                hypothetical_biases=biases,
            )
            unmeas = results.get("unmeasured_confounding_analysis")
            if isinstance(unmeas, dict) and "bias_scenarios" in unmeas:
                results["bias_scenarios"] = unmeas.get("bias_scenarios")
                results["sign_convention"] = unmeas.get("sign_convention")
        else:
            results = analyze_unmeasured_confounding_sensitivity(
                observed_estimate=payload.observed_estimate,
                bias_values=biases,
                confidence_interval=payload.confidence_interval,
            )
            results["reference_estimate"] = results["observed_estimate"]
            results["unmeasured_confounding_analysis"] = dict(results)

        return _sanitize_for_json(results)
    except ValueError as exc:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail=str(exc),
        )


class DatasetCausalSensitivityAnalysisRequest(BaseModel):
    treatment_col: Optional[str] = Field(
        None,
        description="Name of the binary treatment assignment column in the dataset (required for dataset evaluation mode).",
        examples=["treatment"],
    )
    outcome_col: Optional[str] = Field(
        None,
        description="Name of the numeric outcome column in the dataset (required for dataset evaluation mode).",
        examples=["outcome"],
    )
    confounder_cols: Optional[List[str]] = Field(
        None,
        description="Optional list of observed confounder column names for regression adjustment and IPTW.",
        examples=[["age"]],
    )
    treated_value: Optional[Any] = Field(
        None,
        description="Label identifying the treated group. Inferred automatically if omitted.",
        examples=[1],
    )
    control_value: Optional[Any] = Field(
        None,
        description="Label identifying the control group. Inferred automatically if omitted.",
        examples=[0],
    )
    untreated_value: Optional[Any] = Field(
        None,
        description="Alias for control_value.",
        examples=[0],
    )
    drop_missing: bool = Field(
        default=False,
        description="Whether to drop rows with missing values in required analysis columns. Default is False.",
        examples=[False],
    )
    reference_estimator: Optional[str] = Field(
        default="regression_adjusted",
        description="Primary reference estimator ('regression_adjusted', 'iptw', or 'unadjusted').",
        examples=["regression_adjusted"],
    )
    truncation_thresholds: Optional[List[float]] = Field(
        None,
        description="Optional sequence of strictly positive upper weight bounds for IPTW truncation sensitivity checks.",
        examples=[[10.0, 5.0, 3.0]],
    )
    hypothetical_biases: Optional[Union[float, List[float]]] = Field(
        None,
        description="Optional hypothetical signed bias values (b) for unmeasured confounding sensitivity scenarios: tau_adjusted = tau_observed - b.",
        examples=[[1.0, 2.0, -1.0]],
    )
    bias_values: Optional[Union[float, List[float]]] = Field(
        None,
        description="Alias for hypothetical_biases, or required bias values in standalone observed-estimate mode.",
        examples=[[1.0, 2.0, -1.0]],
    )
    observed_estimate: Optional[float] = Field(
        None,
        description="Observed effect estimate for standalone unmeasured confounding sensitivity analysis (used when treatment_col/outcome_col are omitted).",
        examples=[5.0],
    )
    confidence_interval: Optional[List[float]] = Field(
        None,
        description="Optional two-element confidence interval [lower, upper] for the observed estimate in standalone mode.",
        examples=[[3.0, 7.0]],
    )

    @field_validator("treatment_col", "outcome_col")
    @classmethod
    def validate_column_name(cls, v: Any, info: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string when provided.")
        return v.strip()

    @field_validator("confounder_cols")
    @classmethod
    def validate_confounder_names(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if not isinstance(v, (list, tuple)):
            raise ValueError("confounder_cols must be a list of column names.")
        cleaned = []
        for idx, col in enumerate(v):
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"confounder_cols element at index {idx} must be a non-empty string.")
            cleaned.append(col.strip())
        return cleaned

    @field_validator("truncation_thresholds", mode="before")
    @classmethod
    def validate_truncation_thresholds(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (list, tuple)):
            raise ValueError("truncation_thresholds must be a list of strictly positive numbers.")
        if len(v) == 0:
            raise ValueError("truncation_thresholds cannot be empty when provided.")
        validated = []
        for idx, thresh in enumerate(v):
            if thresh is None or isinstance(thresh, bool) or not isinstance(thresh, (int, float)):
                raise ValueError(f"truncation_thresholds element at index {idx} must be a numeric float or int.")
            if math.isnan(thresh) or math.isinf(thresh):
                raise ValueError(f"truncation_thresholds element at index {idx} cannot be NaN or infinite.")
            if float(thresh) <= 0.0:
                raise ValueError(f"truncation_thresholds element at index {idx} must be strictly positive, got {thresh}.")
            validated.append(float(thresh))
        return validated

    @field_validator("observed_estimate", mode="before")
    @classmethod
    def validate_observed_estimate(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("observed_estimate cannot be a boolean.")
        if not isinstance(v, (int, float)):
            raise ValueError(f"observed_estimate must be a numeric float or int, got {type(v).__name__}.")
        if math.isnan(v) or math.isinf(v):
            raise ValueError("observed_estimate cannot be NaN or infinite.")
        return float(v)

    @field_validator("hypothetical_biases", "bias_values", mode="before")
    @classmethod
    def validate_biases(cls, v: Any, info: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError(f"{info.field_name} cannot be a boolean.")
        if isinstance(v, (int, float)):
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"{info.field_name} cannot be NaN or infinite.")
            return float(v)
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                raise ValueError(f"{info.field_name} cannot be empty when provided.")
            validated = []
            for idx, b in enumerate(v):
                if b is None or isinstance(b, bool) or not isinstance(b, (int, float)):
                    raise ValueError(f"{info.field_name} element at index {idx} must be a numeric float or int.")
                if math.isnan(b) or math.isinf(b):
                    raise ValueError(f"{info.field_name} element at index {idx} cannot be NaN or infinite.")
                validated.append(float(b))
            return validated
        raise ValueError(f"{info.field_name} must be a number or list of numbers.")

    @field_validator("confidence_interval", mode="before")
    @classmethod
    def validate_ci(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (list, tuple)):
            raise ValueError("confidence_interval must be a list of 2 numbers.")
        if len(v) != 2:
            raise ValueError(f"confidence_interval must contain exactly 2 elements [lower, upper], got {len(v)}.")
        for idx, bound in enumerate(v):
            if bound is None or isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise ValueError(f"confidence_interval bound at index {idx} must be a numeric float or int.")
            if math.isnan(bound) or math.isinf(bound):
                raise ValueError(f"confidence_interval bound at index {idx} cannot be NaN or infinite.")
        l_val, u_val = float(v[0]), float(v[1])
        if l_val > u_val:
            raise ValueError(f"confidence_interval lower bound ({l_val}) cannot exceed upper bound ({u_val}).")
        return [l_val, u_val]

    @field_validator("drop_missing")
    @classmethod
    def validate_drop_missing(cls, v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError("drop_missing must be a boolean.")
        return v

    @model_validator(mode="after")
    def validate_dataset_sensitivity_request(self) -> "DatasetCausalSensitivityAnalysisRequest":
        if self.treatment_col or self.outcome_col:
            if not self.treatment_col:
                raise ValueError("treatment_col must be provided as a non-empty string when outcome_col is provided.")
            if not self.outcome_col:
                raise ValueError("outcome_col must be provided as a non-empty string when treatment_col is provided.")
            if self.treatment_col == self.outcome_col:
                raise ValueError(f"treatment_col and outcome_col cannot be the same column ('{self.treatment_col}').")
            if self.confounder_cols:
                if self.treatment_col in self.confounder_cols:
                    raise ValueError(f"treatment_col ('{self.treatment_col}') cannot also appear in confounder_cols.")
                if self.outcome_col in self.confounder_cols:
                    raise ValueError(f"outcome_col ('{self.outcome_col}') cannot also appear in confounder_cols.")
                if len(self.confounder_cols) != len(set(self.confounder_cols)):
                    raise ValueError(f"confounder_cols contains duplicate column names: {self.confounder_cols}.")
            if self.reference_estimator:
                norm_ref = self.reference_estimator.lower().strip()
                valid_refs = {"regression_adjusted", "iptw", "unadjusted"}
                if norm_ref not in valid_refs:
                    raise ValueError(
                        f"Unsupported reference_estimator: {self.reference_estimator!r}. "
                        f"Expected one of: {sorted(list(valid_refs))}."
                    )
                self.reference_estimator = norm_ref
            actual_ctrl = self.control_value if self.control_value is not None else self.untreated_value
            if self.treated_value is not None and actual_ctrl is not None and self.treated_value == actual_ctrl:
                raise ValueError("treated_value and control_value cannot be identical.")
        else:
            if self.observed_estimate is None:
                raise ValueError(
                    "Either ('treatment_col' and 'outcome_col') for dataset analysis or ('observed_estimate' and 'bias_values'/'hypothetical_biases') for standalone sensitivity analysis must be provided."
                )
            resolved_biases = self.hypothetical_biases if self.hypothetical_biases is not None else self.bias_values
            if resolved_biases is None:
                raise ValueError(
                    "bias_values (or hypothetical_biases) must be provided when analyzing unmeasured confounding for an observed estimate."
                )
        return self


class DatasetCausalSensitivityAnalysisResponse(CausalSensitivityAnalysisResponse):
    dataset_id: str = Field(..., description="Unique opaque dataset identifier")
    dataset_filename: str = Field(..., description="Original filename of the stored dataset evaluated")
    total_dataset_rows: int = Field(..., description="Total rows in the stored dataset")


@app.post(
    "/api/dataset/{dataset_id}/causal/sensitivity-analysis",
    response_model=DatasetCausalSensitivityAnalysisResponse,
    tags=["Causal Inference", "Datasets"],
    summary="Conduct causal sensitivity analysis for a stored dataset",
)
async def dataset_causal_sensitivity_analysis_endpoint(
    dataset_id: str,
    payload: DatasetCausalSensitivityAnalysisRequest,
) -> Dict[str, Any]:
    """Perform causal sensitivity analysis and robustness checks for a stored dataset.

    Supports two evaluation modes:
    1. Dataset Mode: When treatment_col and outcome_col are specified, loads the required columns
       from disk, evaluates multi-estimator consistency (unadjusted, regression adjustment, IPTW),
       tests IPTW weight truncation stability across thresholds, and evaluates unmeasured confounding
       hypothetical bias scenarios.
    2. Observed Estimate Mode: When treatment_col is omitted and observed_estimate is provided,
       evaluates unmeasured confounding hypothetical bias scenarios and confidence interval coverage
       without requiring full dataset re-ingestion.
    """
    dataset_meta = default_dataset_manager.get_dataset(dataset_id)
    if dataset_meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found or has expired.",
        )

    file_path = default_dataset_manager.get_dataset_file_path(dataset_id)
    if file_path is None or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file for '{dataset_id}' is not available on disk.",
        )

    biases = (
        payload.hypothetical_biases
        if payload.hypothetical_biases is not None
        else payload.bias_values
    )
    resolved_control = (
        payload.control_value
        if payload.control_value is not None
        else payload.untreated_value
    )

    if payload.treatment_col and payload.outcome_col:
        available_cols = [c.name for c in dataset_meta.columns]
        if payload.treatment_col not in available_cols:
            raise HTTPException(
                status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                detail=(
                    f"Treatment column '{payload.treatment_col}' not found in dataset. "
                    f"Available columns: {available_cols}"
                ),
            )
        if payload.outcome_col not in available_cols:
            raise HTTPException(
                status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                detail=(
                    f"Outcome column '{payload.outcome_col}' not found in dataset. "
                    f"Available columns: {available_cols}"
                ),
            )
        if payload.confounder_cols:
            for c_col in payload.confounder_cols:
                if c_col not in available_cols:
                    raise HTTPException(
                        status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                        detail=(
                            f"Confounder column '{c_col}' not found in dataset. "
                            f"Available columns: {available_cols}"
                        ),
                    )

        if dataset_meta.row_count == 0:
            raise HTTPException(
                status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                detail="Dataset is empty (0 rows). Causal inference requires observations.",
            )

        cols_to_read = [payload.treatment_col, payload.outcome_col]
        if payload.confounder_cols:
            for c in payload.confounder_cols:
                if c not in cols_to_read:
                    cols_to_read.append(c)

        try:
            df = pd.read_csv(
                file_path,
                usecols=cols_to_read,
                na_values=list(MISSING_TOKENS),
                keep_default_na=True,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                detail=f"Failed to read dataset columns: {exc}",
            )

        if df.empty:
            raise HTTPException(
                status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                detail="Dataset is empty (0 rows). Causal inference requires observations.",
            )

        resolved_treated = _coerce_label_to_series_dtype(payload.treated_value, df[payload.treatment_col])
        resolved_untreated = _coerce_label_to_series_dtype(resolved_control, df[payload.treatment_col])

        try:
            results = run_causal_sensitivity_analysis(
                data=df,
                treatment_col=payload.treatment_col,
                outcome_col=payload.outcome_col,
                confounder_cols=payload.confounder_cols,
                treated_value=resolved_treated,
                control_value=resolved_untreated,
                drop_missing=payload.drop_missing,
                reference_estimator=payload.reference_estimator or "regression_adjusted",
                truncation_thresholds=payload.truncation_thresholds,
                hypothetical_biases=biases,
            )
            unmeas = results.get("unmeasured_confounding_analysis")
            if isinstance(unmeas, dict) and "bias_scenarios" in unmeas:
                results["bias_scenarios"] = unmeas.get("bias_scenarios")
                results["sign_convention"] = unmeas.get("sign_convention")
            results["dataset_id"] = dataset_id
            results["dataset_filename"] = dataset_meta.filename
            results["total_dataset_rows"] = dataset_meta.row_count
            return _sanitize_for_json(results)
        except ValueError as exc:
            raise HTTPException(
                status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                detail=str(exc),
            )
    else:
        try:
            results = analyze_unmeasured_confounding_sensitivity(
                observed_estimate=payload.observed_estimate,
                bias_values=biases,
                confidence_interval=payload.confidence_interval,
            )
            results["reference_estimator"] = payload.reference_estimator or "user_supplied"
            results["reference_estimate"] = results["observed_estimate"]
            results["unmeasured_confounding_analysis"] = dict(results)
            results["dataset_id"] = dataset_id
            results["dataset_filename"] = dataset_meta.filename
            results["total_dataset_rows"] = dataset_meta.row_count
            return _sanitize_for_json(results)
        except ValueError as exc:
            raise HTTPException(
                status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
                detail=str(exc),
            )


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)




