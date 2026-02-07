"""
Data models for the ingestion pipeline.

Why Pydantic?
- Type validation at runtime (catches bugs early)
- Automatic JSON serialization (needed for vector store metadata)
- Self-documenting schemas
- Works seamlessly with FastAPI later
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class ChunkType(StrEnum):
    """Types of code chunks we extract.

    Why explicit types?
    - Enables filtering in vector search ("show me only classes")
    - Different chunk types may need different embedding strategies
    - Useful for analytics (what parts of codebase are most queried)
    """

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"  # Top-level code, imports, constants


class Language(StrEnum):
    """Supported programming languages.

    Why enum instead of string?
    - Prevents typos ("pythn" vs "python")
    - Easy to iterate for multi-language support
    - Maps to tree-sitter grammar names
    """

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    # Add more as needed


# Map file extensions to languages
EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".pyw": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
}


class CodeChunk(BaseModel):
    """A semantic unit of code extracted from a source file.

    This is the core data structure that flows through the entire RAG pipeline:
    1. Created by the parser from AST nodes
    2. Stored in vector database with embeddings
    3. Retrieved during search
    4. Passed to LLM as context

    Design decisions:
    - `content` is the actual code (what gets embedded)
    - `metadata` is everything else (used for filtering, not embedding)
    """

    # Core content
    content: str = Field(..., description="The actual source code of this chunk")
    chunk_type: ChunkType = Field(
        ..., description="The type of code element (function, class, etc.)"
    )
    name: str = Field(..., description="Name of the function/class/method")

    # Source location
    file_path: str = Field(..., description="Absolute path to the source file")
    start_line: int = Field(..., ge=1, description="Starting line number (1-indexed)")
    end_line: int = Field(
        ..., ge=1, description="Ending line number (1-indexed, inclusive)"
    )

    # Language info
    language: Language = Field(..., description="Programming language")

    # Optional enrichment
    docstring: str | None = Field(
        default=None, description="Extracted docstring if present"
    )
    parent_name: str | None = Field(
        default=None, description="Parent class name for methods"
    )
    decorators: list[str] = Field(
        default_factory=list,
        description="List of decorator names (e.g., ['staticmethod', 'lru_cache'])",
    )

    @computed_field
    @property
    def chunk_id(self) -> str:
        """Generate a unique ID for this chunk.

        Format: {file_path}::{name}::{start_line}

        Why this format?
        - Deterministic: same code = same ID (important for updates)
        - Human-readable: can debug by looking at IDs
        - Unique: file + name + line is unique within a repo
        """
        return f"{self.file_path}::{self.name}::{self.start_line}"

    @computed_field
    @property
    def relative_path(self) -> str:
        """Get just the filename for display purposes."""
        return Path(self.file_path).name

    def to_embedding_text(self) -> str:
        """Convert chunk to text for embedding.

        Why not just use `content`?
        - We want to include the name and docstring prominently
        - Prepending context improves retrieval for natural language queries
        - E.g., "calculate tax function" should match even if function name
          is only in the def line
        """
        parts = [f"{self.chunk_type.value}: {self.name}"]

        if self.parent_name:
            parts.append(f"(in class {self.parent_name})")

        if self.docstring:
            parts.append(f"\n{self.docstring}")

        parts.append(f"\n{self.content}")

        return " ".join(parts)

    def to_metadata(self) -> dict:
        """Convert to metadata dict for vector store.

        ChromaDB stores metadata alongside embeddings for filtering.
        We exclude `content` because it's stored separately.
        """
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.value,
            "name": self.name,
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language.value,
            "has_docstring": self.docstring is not None,
            "parent_name": self.parent_name or "",
            "decorators": ",".join(self.decorators),
        }


class FileInfo(BaseModel):
    """Information about a source file to be processed."""

    path: Path = Field(..., description="Absolute path to the file")
    language: Language = Field(..., description="Detected programming language")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")

    @computed_field
    @property
    def relative_name(self) -> str:
        """Just the filename."""
        return self.path.name
