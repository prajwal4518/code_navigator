"""
Ingestion Module - Transform raw code into structured chunks.

This module handles:
- Repository traversal and file discovery
- AST-based code parsing (using tree-sitter)
- Intelligent chunking that respects code boundaries
- Metadata extraction (function names, classes, docstrings)

Why AST parsing instead of simple text splitting?
- Text splitting breaks code mid-function, losing semantic meaning
- AST parsing preserves complete functions, classes, and logical units
- Better embeddings = better retrieval quality

Usage:
    >>> from code_navigator.ingestion import chunk_repository, CodeChunk
    >>> chunks = chunk_repository("./my-repo")
    >>> for chunk in chunks:
    ...     print(chunk.name, chunk.chunk_type)
"""

from .chunker import CodeChunker, chunk_repository
from .file_discovery import detect_language, discover_files
from .models import (
    EXTENSION_TO_LANGUAGE,
    ChunkType,
    CodeChunk,
    FileInfo,
    Language,
)
from .parser import PythonParser

__all__ = [
    # Models
    "ChunkType",
    "CodeChunk",
    "FileInfo",
    "Language",
    "EXTENSION_TO_LANGUAGE",
    # File discovery
    "discover_files",
    "detect_language",
    # Parser
    "PythonParser",
    # Chunker
    "CodeChunker",
    "chunk_repository",
]
