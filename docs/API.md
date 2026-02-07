# Code Navigator API Documentation

## Overview

The Code Navigator API provides REST endpoints for querying and understanding codebases using RAG (Retrieval Augmented Generation).

**Base URL:** `http://localhost:8000`

---

## Authentication

Currently, the API does not require authentication. For production deployments, add API key authentication via headers.

---

## Endpoints

### POST `/api/ask`

Ask a single question about the codebase. Stateless — no conversation memory.

**Request Body:**

```json
{
  "question": "What does the parse_file method do?",
  "num_chunks": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | — | Natural language question |
| `num_chunks` | integer | No | 5 | Number of code chunks to retrieve (1-20) |

**Response:**

```json
{
  "answer": "The parse_file method parses a Python file...",
  "sources": [
    {
      "file_path": "/path/to/parser.py",
      "start_line": 38,
      "end_line": 52,
      "chunk_type": "method",
      "name": "parse_file"
    }
  ],
  "tokens_used": 1250
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does BM25 scoring work?"}'
```

---

### POST `/api/chat`

Chat with conversation memory. Use `session_id` for follow-up questions.

**Request Body:**

```json
{
  "message": "What does CodeChunker do?",
  "session_id": "abc-123-xyz"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | User message |
| `session_id` | string | No | Session ID for conversation continuity |

**Response:**

```json
{
  "response": "The CodeChunker class orchestrates...",
  "session_id": "abc-123-xyz",
  "sources": [...],
  "tokens_used": 1450
}
```

**Example:**

```bash
# Start a conversation
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the VectorStore?"}'

# Follow up (use returned session_id)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does it store embeddings?", "session_id": "abc-123-xyz"}'
```

---

### GET `/api/health`

Health check endpoint for monitoring and load balancers.

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "index_size": 53
}
```

**Example:**

```bash
curl http://localhost:8000/api/health
```

---

### GET `/api/stats`

Get usage statistics for monitoring and cost tracking.

**Response:**

```json
{
  "total_requests": 42,
  "total_tokens": 15000,
  "prompt_tokens": 12000,
  "completion_tokens": 3000,
  "index_size": 53
}
```

**Example:**

```bash
curl http://localhost:8000/api/stats
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": "Error message",
  "detail": "Detailed information (optional)"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad request (invalid input) |
| 500 | Internal server error |
| 503 | Service unavailable (navigator not initialized) |

---

## Running the API

```bash
# Development (with auto-reload)
uvicorn code_navigator.api.main:app --reload

# Production
uvicorn code_navigator.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## OpenAPI Documentation

FastAPI automatically generates interactive API docs:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
