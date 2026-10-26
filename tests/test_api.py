from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api import app

client = TestClient(app)


def test_root_endpoint():
    """Verify that root endpoint returns application metadata and purpose."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "description" in data
    assert "version" in data
    assert "docs_url" in data
    assert "Experimentation" in data["name"]


def test_health_endpoint():
    """Verify that health check endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_docs_endpoint():
    """Verify that interactive OpenAPI docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
