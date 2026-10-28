"""Automated tests for Dataset Ingestion, CSV Validation, Schema Inspection, and Benchmark Loading."""

import io
from pathlib import Path
import sys
import tempfile
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api import app
from dataset_manager import DatasetManager, default_dataset_manager

client = TestClient(app)


# --- 1. CSV Header and Structure Validation Tests ---

def test_upload_empty_csv_rejected():
    """Verify that an empty file (0 bytes) is rejected with HTTP 422."""
    res = client.post("/api/dataset/upload", content=b"")
    assert res.status_code == 422
    assert "empty" in res.json()["detail"].lower()


def test_upload_whitespace_only_rejected():
    """Verify that a file containing only whitespace is rejected with HTTP 422."""
    res = client.post("/api/dataset/upload", content=b"   \n\n  \t  \n")
    assert res.status_code == 422
    assert "empty" in res.json()["detail"].lower()


def test_upload_empty_header_name_rejected():
    """Verify that empty or whitespace-only header names are rejected with HTTP 422."""
    csv_content = b"col1,,col3\n1,2,3\n"
    res = client.post("/api/dataset/upload", content=csv_content)
    assert res.status_code == 422
    assert "empty or whitespace-only column name" in res.json()["detail"]


def test_upload_duplicate_header_names_rejected():
    """Verify that duplicate column names in the header are rejected with HTTP 422."""
    csv_content = b"treatment,outcome,treatment\n1,10,0\n"
    res = client.post("/api/dataset/upload", content=csv_content)
    assert res.status_code == 422
    assert "duplicate column name 'treatment'" in res.json()["detail"]


def test_upload_inconsistent_row_widths_rejected():
    """Verify that inconsistent column counts across rows are caught and reported with row index."""
    csv_content = b"col1,col2,col3\n1,2,3\n4,5\n"
    res = client.post("/api/dataset/upload", content=csv_content)
    assert res.status_code == 422
    assert "Inconsistent row width at data row 2" in res.json()["detail"]
    assert "expected 3 columns but found 2" in res.json()["detail"]


def test_upload_invalid_utf8_rejected():
    """Verify that non-UTF-8 bytes trigger a clear decoding error with byte position."""
    invalid_bytes = b"col1,col2\n1,\xff\xfe\n"
    res = client.post("/api/dataset/upload", content=invalid_bytes)
    assert res.status_code == 422
    assert "invalid utf-8 encoding" in res.json()["detail"].lower()


def test_upload_header_only_csv():
    """Verify that a header-only CSV with zero data rows is processed safely."""
    csv_content = b"user_id,revenue,converted\n"
    res = client.post(
        "/api/dataset/upload",
        content=csv_content,
        headers={"X-Filename": "header_only.csv"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["row_count"] == 0
    assert data["column_count"] == 3
    assert len(data["preview_rows"]) == 0
    assert len(data["columns"]) == 3
    for col in data["columns"]:
        assert col["missing_count"] == 0
        assert col["missing_percentage"] == 0.0
        assert col["inferred_type"] == "unknown"


# --- 2. Advanced CSV Syntax & Quoting Handling Tests ---

def test_upload_quoted_commas_and_quotes():
    """Verify RFC 4180 handling for quoted fields, commas inside quotes, and escaped quotes."""
    csv_content = (
        'id,description,tags\n'
        '1,"Product A, Deluxe edition","tag1,tag2"\n'
        '2,"He said, ""Hello World!""","greeting"\n'
    ).encode("utf-8")

    res = client.post(
        "/api/dataset/upload",
        content=csv_content,
        headers={"X-Filename": "quoted_test.csv"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["row_count"] == 2
    assert data["column_count"] == 3
    preview = data["preview_rows"]
    assert preview[0]["description"] == "Product A, Deluxe edition"
    assert preview[1]["description"] == 'He said, "Hello World!"'


def test_upload_quoted_multiline_fields():
    """Verify that quoted fields spanning multiple lines are correctly parsed without row splitting."""
    csv_content = (
        'id,notes,value\n'
        '1,"First line\nSecond line\nThird line",100\n'
        '2,"Single line",200\n'
    ).encode("utf-8")

    res = client.post("/api/dataset/upload", content=csv_content)
    assert res.status_code == 200
    data = res.json()
    assert data["row_count"] == 2
    preview = data["preview_rows"]
    assert "\n" in preview[0]["notes"]
    assert preview[0]["value"] == "100"


def test_upload_very_long_field_values():
    """Verify that long field strings do not crash parser limits."""
    long_text = "x" * 50000
    csv_content = f"id,payload\n1,{long_text}\n".encode("utf-8")
    res = client.post("/api/dataset/upload", content=csv_content)
    assert res.status_code == 200
    data = res.json()
    assert data["row_count"] == 1
    assert data["preview_rows"][0]["payload"] == long_text


# --- 3. Type Inference and Missing-Value Policy Tests ---

def test_type_inference_lattice():
    """Verify type inference across boolean, integer, numeric, and categorical columns."""
    csv_content = (
        "binary_flag,int_val,float_val,text_val,all_null\n"
        "1,10,3.14,Urban,NA\n"
        "0,25,-12.50,Suburban,null\n"
        "true,-40,100.0,Rural,\n"
        "false,0,0.001,Urban,None\n"
    ).encode("utf-8")

    res = client.post("/api/dataset/upload", content=csv_content)
    assert res.status_code == 200
    data = res.json()

    type_map = {c["name"]: c["inferred_type"] for c in data["columns"]}
    assert type_map["binary_flag"] == "boolean"
    assert type_map["int_val"] == "integer"
    assert type_map["float_val"] == "numeric"
    assert type_map["text_val"] == "categorical"
    assert type_map["all_null"] == "unknown"


def test_missing_value_policy_counts():
    """Verify that empty, whitespace, and documented tokens (na, n/a, null, none, nan, .) are counted as missing."""
    csv_content = (
        "feature\n"
        "valid_val\n"
        '""\n'         # empty string
        "   \n"       # whitespace only
        "NA\n"        # NA
        "n/a\n"       # n/a
        "NULL\n"      # null
        "None\n"      # none
        "nan\n"       # nan
        ".\n"         # dot
        "another_val\n"
    ).encode("utf-8")

    res = client.post("/api/dataset/upload", content=csv_content)
    assert res.status_code == 200
    data = res.json()
    col = data["columns"][0]

    assert col["name"] == "feature"
    assert col["missing_count"] == 8
    assert col["total_valid_count"] == 2
    assert data["row_count"] == 10
    assert col["missing_percentage"] == 80.0


# --- 4. Bounded Preview and Large Count Integrity ---

def test_preview_bounded_to_five_rows():
    """Verify that the preview contains exactly 5 rows even when the dataset has more rows."""
    lines = ["id,value"] + [f"{i},{i*10}" for i in range(1, 21)]
    csv_content = "\n".join(lines).encode("utf-8")

    res = client.post("/api/dataset/upload", content=csv_content)
    assert res.status_code == 200
    data = res.json()
    assert data["row_count"] == 20
    assert len(data["preview_rows"]) == 5
    assert data["preview_rows"][0]["id"] == "1"
    assert data["preview_rows"][4]["id"] == "5"


# --- 5. Upload Size Limits and HTTP 413 Enforcement ---

def test_upload_exceeds_size_limit():
    """Verify that exceeding the configured upload limit raises HTTP 413 without reading the entire stream."""
    small_manager = DatasetManager(max_upload_size_bytes=500)
    # Monkey-patch default_dataset_manager temporarily to test limit
    from api import default_dataset_manager as original_mgr
    import api

    api.default_dataset_manager = small_manager
    try:
        big_content = b"col1,col2\n" + (b"12345,67890\n" * 50)  # > 500 bytes
        res = client.post("/api/dataset/upload", content=big_content)
        assert res.status_code == 413
        assert "exceeds the maximum limit" in res.json()["detail"]
    finally:
        api.default_dataset_manager = original_mgr


# --- 6. Dataset ID Retrieval, Deletion, and Path Safety ---

def test_get_and_delete_dataset():
    """Verify retrieval via opaque ID and subsequent deletion."""
    csv_content = b"a,b\n1,2\n3,4\n"
    upload_res = client.post("/api/dataset/upload", content=csv_content)
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    # Retrieve dataset
    get_res = client.get(f"/api/dataset/{dataset_id}")
    assert get_res.status_code == 200
    assert get_res.json()["dataset_id"] == dataset_id
    assert get_res.json()["row_count"] == 2

    # Delete dataset
    del_res = client.delete(f"/api/dataset/{dataset_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Confirm it is no longer retrievable
    get_after = client.get(f"/api/dataset/{dataset_id}")
    assert get_after.status_code == 404


def test_reject_arbitrary_paths():
    """Verify that path traversal attempts in dataset_id return HTTP 404."""
    bad_ids = [
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "random-string-not-uuid",
        "12345",
    ]
    for bad_id in bad_ids:
        res = client.get(f"/api/dataset/{bad_id}")
        assert res.status_code == 404


# --- 7. Hillstrom Benchmark Dataset Loader Tests ---

def test_load_hillstrom_benchmark_dataset():
    """Verify explicit loading of the Hillstrom benchmark dataset from data/hillstrom.csv."""
    res = client.post("/api/dataset/load-benchmark")
    assert res.status_code == 200
    data = res.json()

    assert data["is_benchmark"] is True
    assert data["filename"] == "hillstrom.csv"
    assert data["row_count"] == 64000
    assert data["column_count"] == 12
    assert len(data["preview_rows"]) == 5

    # Check key columns
    col_names = [c["name"] for c in data["columns"]]
    expected_cols = [
        "recency", "history_segment", "history", "mens", "womens",
        "zip_code", "newbie", "channel", "segment", "visit", "conversion", "spend",
    ]
    assert col_names == expected_cols

    # Check that spend and history are numeric
    type_map = {c["name"]: c["inferred_type"] for c in data["columns"]}
    assert type_map["history"] == "numeric"
    assert type_map["spend"] == "numeric"
    assert type_map["segment"] == "categorical"
    assert type_map["recency"] == "integer"


def test_load_benchmark_missing_file_handling(monkeypatch):
    """Verify HTTP 404 if data/hillstrom.csv is absent."""
    non_existent_root = Path(tempfile.gettempdir()) / "empty_proj_root"
    non_existent_root.mkdir(exist_ok=True)

    from dataset_manager import DatasetManager
    mgr = DatasetManager()
    with pytest.raises(ValueError) as exc:
        mgr.load_benchmark(non_existent_root)
    assert "not found" in str(exc.value).lower()
