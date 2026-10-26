"""FastAPI Backend for Experimentation and Causal Inference Platform.

Exposes statistical engine functionality for A/B testing and sample size estimation.
"""

from pathlib import Path
import sys
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
import uvicorn

# Ensure the src directory is in sys.path for direct imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
