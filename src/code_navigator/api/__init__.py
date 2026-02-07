"""
API Module - REST endpoints for code navigation.

This module provides:
- FastAPI application
- Request/response schemas
- Health checks and monitoring

Why FastAPI?
- Automatic OpenAPI documentation
- Pydantic integration for validation
- Async support for concurrent requests
- Modern Python typing

Usage:
    uvicorn code_navigator.api.main:app --reload
"""

from .main import app
from .schemas import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    SourceReference,
    StatsResponse,
)

__all__ = [
    "app",
    "AskRequest",
    "AskResponse",
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    "HealthResponse",
    "SourceReference",
    "StatsResponse",
]
