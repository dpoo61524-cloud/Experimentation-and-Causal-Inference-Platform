"""Dataset Ingestion and Schema Inspection Service.

Provides scalable, bounded-memory CSV parsing, type inference, missing-value
auditing, and lifecycle-managed temporary dataset storage for the Experimentation
and Causal Inference Platform.
"""

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple
import uuid

from pydantic import BaseModel, Field

# Ensure maximum field size supported by platform without C long overflow
try:
    csv.field_size_limit(2147483647)
except (OverflowError, ValueError):
    csv.field_size_limit(sys.maxsize)

# Missing value tokens according to explicit platform policy
MISSING_TOKENS: Set[str] = {
    "",
    "na",
    "n/a",
    "null",
    "none",
    "nan",
    ".",
}

# Boolean tokens recognized for treatment indicators and binary flags
BOOLEAN_TOKENS: Set[str] = {
    "true",
    "false",
    "0",
    "1",
    "t",
    "f",
    "yes",
    "no",
}

UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class ColumnMetadata(BaseModel):
    """Metadata summary for a single column in an ingested dataset."""

    name: str = Field(..., description="Column header name")
    inferred_type: str = Field(
        ...,
        description="Inferred data type: boolean, integer, numeric, categorical, or unknown",
    )
    missing_count: int = Field(..., description="Number of missing values detected")
    missing_percentage: float = Field(
        ...,
        description="Percentage of missing values (0.0 to 100.0)",
    )
    total_valid_count: int = Field(
        ...,
        description="Count of non-missing, valid values",
    )


class DatasetIngestionResponse(BaseModel):
    """Response returned upon successful CSV ingestion and schema inspection."""

    dataset_id: str = Field(..., description="Unique opaque dataset identifier")
    filename: str = Field(..., description="Original filename of the dataset")
    file_size_bytes: int = Field(..., description="Total size in bytes on disk")
    row_count: int = Field(..., description="Total number of data rows (excluding header)")
    column_count: int = Field(..., description="Number of columns in dataset")
    columns: List[ColumnMetadata] = Field(..., description="Column metadata and types")
    preview_rows: List[Dict[str, Any]] = Field(
        ...,
        description="Bounded preview of the first 5 rows",
    )
    is_benchmark: bool = Field(
        default=False,
        description="Whether this dataset is the Hillstrom benchmark dataset",
    )
    status: str = Field(default="loaded", description="Dataset ingestion status")
    created_at: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of dataset ingestion",
    )
    missing_value_policy: str = Field(
        default=(
            "Empty string, whitespace-only, and case-insensitive tokens "
            "('na', 'n/a', 'null', 'none', 'nan', '.') are counted as missing."
        ),
        description="Documented missing-value identification rule",
    )


class DatasetManager:
    """Manages streaming CSV ingestion, schema inspection, and temporary dataset lifecycle."""

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        max_upload_size_bytes: int = 500 * 1024 * 1024,  # 500 MB default
        ttl_seconds: int = 14400,  # 4 hours
        max_registry_size: int = 50,
    ) -> None:
        self.storage_dir = (
            storage_dir
            if storage_dir is not None
            else Path(tempfile.gettempdir()) / "experimentation_platform_datasets"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_upload_size_bytes = max_upload_size_bytes
        self.ttl_seconds = ttl_seconds
        self.max_registry_size = max_registry_size
        self._registry: Dict[str, Tuple[DatasetIngestionResponse, float]] = {}

    def _sanitize_filename(self, filename: Optional[str]) -> str:
        """Sanitize client-provided filename to prevent directory traversal or malformed names."""
        if not filename or not filename.strip():
            return "uploaded_dataset.csv"
        # Take basename only
        clean = Path(filename).name.strip()
        # Remove any non-whitelisted characters for safety
        clean = re.sub(r"[^a-zA-Z0-9_\-\. ]", "", clean)
        if not clean:
            return "uploaded_dataset.csv"
        if not clean.lower().endswith(".csv"):
            clean += ".csv"
        return clean

    def cleanup_expired(self) -> int:
        """Evict expired datasets from in-memory registry and delete corresponding disk files."""
        now = datetime.now(timezone.utc).timestamp()
        expired_ids = []
        for d_id, (_, created_timestamp) in list(self._registry.items()):
            if now - created_timestamp > self.ttl_seconds:
                expired_ids.append(d_id)

        # Also purge if registry exceeds capacity (FIFO by oldest)
        if len(self._registry) > self.max_registry_size:
            sorted_by_age = sorted(
                self._registry.items(),
                key=lambda item: item[1][1],
            )
            excess_count = len(self._registry) - self.max_registry_size
            for item in sorted_by_age[:excess_count]:
                if item[0] not in expired_ids:
                    expired_ids.append(item[0])

        removed_count = 0
        for d_id in expired_ids:
            if self.delete_dataset(d_id):
                removed_count += 1
        return removed_count

    def get_dataset(self, dataset_id: str) -> Optional[DatasetIngestionResponse]:
        """Retrieve stored dataset metadata by opaque ID."""
        if not isinstance(dataset_id, str) or not UUID4_PATTERN.match(dataset_id):
            return None
        self.cleanup_expired()
        entry = self._registry.get(dataset_id)
        if entry is None:
            return None
        return entry[0]

    def get_dataset_file_path(self, dataset_id: str) -> Optional[Path]:
        """Retrieve the verified on-disk path for a stored dataset."""
        if not isinstance(dataset_id, str) or not UUID4_PATTERN.match(dataset_id):
            return None
        file_path = self.storage_dir / f"{dataset_id}.csv"
        if not file_path.is_file():
            return None
        return file_path

    def delete_dataset(self, dataset_id: str) -> bool:
        """Explicitly delete a dataset from storage and registry."""
        if not isinstance(dataset_id, str) or not UUID4_PATTERN.match(dataset_id):
            return False
        file_path = self.storage_dir / f"{dataset_id}.csv"
        try:
            if file_path.is_file():
                file_path.unlink()
        except OSError:
            pass
        removed = self._registry.pop(dataset_id, None)
        return removed is not None

    def inspect_and_catalog(
        self,
        file_path: Path,
        original_filename: str,
        is_benchmark: bool = False,
    ) -> DatasetIngestionResponse:
        """Perform streaming incremental validation, schema inspection, and cataloging.
        
        Reads the CSV file strictly line-by-line without loading all rows into RAM.
        Maintains O(1) memory complexity relative to dataset row count.
        """
        if not file_path.is_file():
            raise ValueError(f"Target dataset file does not exist: {file_path}")

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError("Uploaded CSV file is empty (0 bytes).")

        # Open file with UTF-8 decoding, handling decode errors cleanly
        try:
            with open(file_path, "r", encoding="utf-8", errors="strict") as f:
                reader = csv.reader(f)

                # 1. Validate Header
                try:
                    header = next(reader)
                except StopIteration:
                    raise ValueError("Uploaded CSV file is empty.")

                if not header or all(col.strip() == "" for col in header):
                    raise ValueError("CSV header row is empty or contains only whitespace.")

                # Strip UTF-8 BOM from first column if present (e.g. from Excel exports)
                if header and header[0].startswith("\ufeff"):
                    header[0] = header[0][1:]

                seen_names: Set[str] = set()
                clean_headers: List[str] = []
                for i, raw_col in enumerate(header):
                    col_name = raw_col.strip()
                    if not col_name:
                        raise ValueError(
                            f"CSV header contains empty or whitespace-only column name at column index {i + 1}."
                        )
                    if col_name in seen_names:
                        raise ValueError(
                            f"CSV header contains duplicate column name '{col_name}' at column index {i + 1}."
                        )
                    seen_names.add(col_name)
                    clean_headers.append(col_name)

                column_count = len(clean_headers)

                # 2. Incremental single-pass scanning state
                # Type lattice states: 'boolean' -> 'integer' -> 'numeric' -> 'categorical'
                column_types = ["boolean"] * column_count
                missing_counts = [0] * column_count
                valid_counts = [0] * column_count
                preview_rows: List[Dict[str, Any]] = []
                row_count = 0

                for row_idx, row in enumerate(reader, start=1):
                    # Skip completely empty lines (e.g. accidental blank lines or trailing newlines)
                    if not row:
                        continue

                    # Check row width consistency
                    if len(row) != column_count:
                        raise ValueError(
                            f"Inconsistent row width at data row {row_idx}: "
                            f"expected {column_count} columns but found {len(row)}."
                        )

                    row_count += 1

                    # Bounded preview: capture first 5 data rows only
                    if row_count <= 5:
                        preview_rows.append(dict(zip(clean_headers, row)))

                    # Audit types and missing values column by column
                    for col_idx, cell_value in enumerate(row):
                        stripped = cell_value.strip()
                        lower_val = stripped.lower()

                        # Missing value detection
                        if lower_val in MISSING_TOKENS:
                            missing_counts[col_idx] += 1
                            continue

                        valid_counts[col_idx] += 1
                        current_type = column_types[col_idx]

                        # If already at widest type (categorical), skip parsing
                        if current_type == "categorical":
                            continue

                        # Evaluate boolean compatibility
                        if current_type == "boolean":
                            if lower_val in BOOLEAN_TOKENS:
                                continue
                            current_type = "integer"

                        # Evaluate integer compatibility
                        if current_type == "integer":
                            try:
                                int(stripped)
                                column_types[col_idx] = "integer"
                                continue
                            except ValueError:
                                current_type = "numeric"

                        # Evaluate numeric / float compatibility
                        if current_type == "numeric":
                            try:
                                f_val = float(stripped)
                                # Exclude infinite/NaN from normal numeric
                                if not (f_val == float("inf") or f_val == float("-inf") or f_val != f_val):
                                    column_types[col_idx] = "numeric"
                                    continue
                                else:
                                    column_types[col_idx] = "categorical"
                            except ValueError:
                                column_types[col_idx] = "categorical"

        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Invalid UTF-8 encoding in CSV file at byte offset {exc.start}: {exc.reason}. "
                "Please ensure the file is valid UTF-8."
            )

        # Assemble column metadata
        columns: List[ColumnMetadata] = []
        for i, col_name in enumerate(clean_headers):
            # If all values were missing, type is 'unknown'
            inferred = column_types[i] if valid_counts[i] > 0 else "unknown"
            miss_cnt = missing_counts[i]
            pct = round((miss_cnt / row_count * 100.0), 2) if row_count > 0 else 0.0

            columns.append(
                ColumnMetadata(
                    name=col_name,
                    inferred_type=inferred,
                    missing_count=miss_cnt,
                    missing_percentage=pct,
                    total_valid_count=valid_counts[i],
                )
            )

        # Catalog dataset
        dataset_id = str(uuid.uuid4())
        # If the file was temporary and needs to match its dataset ID
        final_path = self.storage_dir / f"{dataset_id}.csv"
        if file_path != final_path:
            # Atomic rename or copy
            if file_path.parent == self.storage_dir:
                file_path.rename(final_path)
            else:
                # Copy from source (e.g. benchmark data)
                import shutil
                shutil.copy2(file_path, final_path)

        now_utc = datetime.now(timezone.utc)
        response = DatasetIngestionResponse(
            dataset_id=dataset_id,
            filename=original_filename,
            file_size_bytes=file_size,
            row_count=row_count,
            column_count=column_count,
            columns=columns,
            preview_rows=preview_rows,
            is_benchmark=is_benchmark,
            status="loaded",
            created_at=now_utc.isoformat(),
        )

        self.cleanup_expired()
        self._registry[dataset_id] = (response, now_utc.timestamp())
        return response

    async def ingest_stream(
        self,
        stream_generator: AsyncGenerator[bytes, None],
        filename: Optional[str] = None,
    ) -> DatasetIngestionResponse:
        """Stream an uploaded file directly to disk, enforcing size limits incrementally.
        
        Avoids reading the entire payload into RAM at any stage.
        """
        clean_filename = self._sanitize_filename(filename)
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=str(self.storage_dir),
            prefix="upload_stream_",
            suffix=".tmp",
        )
        temp_path = Path(temp_path_str)

        total_bytes = 0
        try:
            with os.fdopen(temp_fd, "wb") as f:
                async for chunk in stream_generator:
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > self.max_upload_size_bytes:
                        max_mb = self.max_upload_size_bytes // (1024 * 1024)
                        raise ValueError(
                            f"PAYLOAD_TOO_LARGE: File size exceeds the maximum limit of {max_mb} MB "
                            f"(received {total_bytes} bytes so far)."
                        )
                    f.write(chunk)

            if total_bytes == 0:
                raise ValueError("Uploaded file is empty (0 bytes received).")

            # Validate and catalog
            return self.inspect_and_catalog(temp_path, clean_filename, is_benchmark=False)
        except Exception:
            # Clean up partial temporary file on failure
            if temp_path.is_file():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise

    def load_benchmark(self, project_root: Path) -> DatasetIngestionResponse:
        """Load the Hillstrom benchmark dataset safely from the project's data directory."""
        benchmark_path = project_root / "data" / "hillstrom.csv"
        if not benchmark_path.is_file():
            raise ValueError(
                f"Benchmark dataset not found at expected location: {benchmark_path}. "
                "Please run download_hillstrom.py to obtain the benchmark dataset."
            )

        return self.inspect_and_catalog(
            benchmark_path,
            original_filename="hillstrom.csv",
            is_benchmark=True,
        )


# Global singleton instance for the platform
default_dataset_manager = DatasetManager()
