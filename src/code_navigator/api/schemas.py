"""
Pydantic schemas for API request/response models.

These define the contract for the REST API.
Using Pydantic ensures validation and serialization.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for /api/ask endpoint."""

    question: str = Field(
        ..., description="Natural language question about the codebase", min_length=1
    )
    num_chunks: int = Field(
        default=5, description="Number of code chunks to retrieve", ge=1, le=20
    )


class ChatRequest(BaseModel):
    """Request body for /api/chat endpoint."""

    message: str = Field(
        ..., description="User message for the conversation", min_length=1
    )
    session_id: str | None = Field(
        default=None, description="Session ID for conversation continuity"
    )


class SourceReference(BaseModel):
    """A reference to source code."""

    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str


class AskResponse(BaseModel):
    """Response body for /api/ask endpoint."""

    answer: str = Field(..., description="The generated answer")
    sources: list[SourceReference] = Field(
        default_factory=list, description="Referenced code chunks"
    )
    tokens_used: int = Field(default=0, description="Tokens used for this request")


class ChatResponse(BaseModel):
    """Response body for /api/chat endpoint."""

    response: str = Field(..., description="The assistant's response")
    session_id: str = Field(..., description="Session ID for follow-up messages")
    sources: list[SourceReference] = Field(default_factory=list)
    tokens_used: int = Field(default=0)


class HealthResponse(BaseModel):
    """Response body for /api/health endpoint."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Application version")
    index_size: int = Field(..., description="Number of indexed chunks")


class StatsResponse(BaseModel):
    """Response body for /api/stats endpoint."""

    total_requests: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    index_size: int


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")


class IndexRequest(BaseModel):
    """Request body for /api/index endpoint."""

    url: str = Field(..., description="Git repository URL to clone and index")
    branch: str | None = Field(default=None, description="Branch to clone")
    reset: bool = Field(
        default=False, description="Clear existing data before indexing"
    )


class IndexResponse(BaseModel):
    """Response body for /api/index endpoint."""

    status: str = Field(..., description="Indexing status")
    repo_path: str = Field(..., description="Local path to cloned repo")
    chunks_indexed: int = Field(..., description="Number of chunks indexed")


class FlushResponse(BaseModel):
    """Response body for /api/flush endpoint."""

    status: str = Field(..., description="Flush status")
    chunks_deleted: int = Field(..., description="Number of chunks deleted")
