"""
FastAPI REST API for the Code Navigator.

Endpoints:
- POST /api/ask     - Single Q&A
- POST /api/chat    - Conversational with memory
- GET  /api/health  - Health check
- GET  /api/stats   - Usage statistics

Run with:
    uvicorn code_navigator.api.main:app --reload
"""

import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from code_navigator import __version__
from code_navigator.agents import CodeNavigator
from code_navigator.core.logging import get_logger

from .schemas import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    SourceReference,
    StatsResponse,
)

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Global state
navigator: CodeNavigator | None = None
sessions: dict[str, CodeNavigator] = {}
request_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global navigator

    logger.info("Starting Code Navigator API")

    # Initialize the navigator
    try:
        navigator = CodeNavigator()
        logger.info("Navigator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize navigator: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down Code Navigator API")


app = FastAPI(
    title="Code Navigator API",
    description="RAG-powered code understanding API",
    version=__version__,
    lifespan=lifespan,
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_sources(nav: CodeNavigator, question: str) -> list[SourceReference]:
    """Get source references for a question."""
    results = nav.retriever.search(question, k=5)
    return [
        SourceReference(
            file_path=r.chunk.file_path,
            start_line=r.chunk.start_line,
            end_line=r.chunk.end_line,
            chunk_type=r.chunk.chunk_type.value,
            name=r.chunk.name,
        )
        for r in results
    ]


@app.post(
    "/api/ask",
    response_model=AskResponse,
    responses={500: {"model": ErrorResponse}},
)
async def ask_question(request: AskRequest) -> AskResponse:
    """Ask a single question about the codebase.

    Stateless - no conversation memory.
    """
    global request_count

    if navigator is None:
        raise HTTPException(status_code=503, detail="Navigator not initialized")

    request_count += 1
    logger.info(f"Processing ask request: {request.question[:50]}...")

    try:
        # Get answer
        answer = navigator.ask(request.question, stream=False)

        # Get sources
        sources = _get_sources(navigator, request.question)

        # Get token usage
        usage = navigator.llm.get_usage_stats()

        return AskResponse(
            answer=answer,
            sources=sources,
            tokens_used=usage["total_tokens"],
        )

    except Exception as e:
        logger.error(f"Error processing ask request: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat with conversation memory.

    Use session_id for follow-up questions.
    """
    global request_count

    request_count += 1

    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        logger.info(f"Creating new session: {session_id}")
        sessions[session_id] = CodeNavigator()

    nav = sessions[session_id]

    logger.info(
        f"Processing chat request in session {session_id}: {request.message[:50]}..."
    )

    try:
        # Get response (with memory)
        response = nav.chat(request.message, stream=False)

        # Get sources
        sources = _get_sources(nav, request.message)

        # Get token usage
        usage = nav.llm.get_usage_stats()

        return ChatResponse(
            response=response,
            session_id=session_id,
            sources=sources,
            tokens_used=usage["total_tokens"],
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring."""
    if navigator is None:
        raise HTTPException(status_code=503, detail="Navigator not initialized")

    index_size = navigator.retriever.bm25_index.size

    return HealthResponse(
        status="healthy",
        version=__version__,
        index_size=index_size,
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get usage statistics."""
    if navigator is None:
        raise HTTPException(status_code=503, detail="Navigator not initialized")

    usage = navigator.llm.get_usage_stats()
    index_size = navigator.retriever.bm25_index.size

    return StatsResponse(
        total_requests=request_count,
        total_tokens=usage["total_tokens"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        index_size=index_size,
    )


@app.post(
    "/api/index",
    response_model=IndexResponse,
    responses={500: {"model": ErrorResponse}},
)
async def index_repo(request: IndexRequest) -> IndexResponse:
    """Index a remote Git repository.

    Clones (or updates) the repo and indexes it into the vector store.
    """
    global navigator

    logger.info(f"Indexing repository: {request.url}")

    try:
        from code_navigator.ingestion import CodeChunker, RepoManager
        from code_navigator.vectorstore import VectorStore

        # Clone or update the repo
        repo_manager = RepoManager()
        repo_path = repo_manager.clone_or_update(request.url, branch=request.branch)

        # Initialize indexing
        chunker = CodeChunker(verbose=False)
        store = VectorStore()

        # Clear if requested
        if request.reset:
            store.clear()

        # Chunk and index
        chunks = chunker.chunk_repository(repo_path)
        store.add_chunks(chunks)

        # Reinitialize navigator with new data
        navigator = CodeNavigator()

        return IndexResponse(
            status="success",
            repo_path=str(repo_path),
            chunks_indexed=len(chunks),
        )

    except Exception as e:
        logger.error(f"Error indexing repository: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
