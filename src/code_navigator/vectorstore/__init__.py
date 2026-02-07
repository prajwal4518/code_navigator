"""
Vector Store Module - Persistent storage for code embeddings.

This module handles:
- ChromaDB collection management
- Embedding storage and retrieval
- Metadata filtering (by file, language, type)
- Collection versioning for updates

Why ChromaDB?
- Local-first: No external dependencies for development
- Persistent: Survives restarts without re-indexing
- Metadata filtering: Query by file type, function name, etc.
- Easy migration to cloud (Pinecone, Weaviate) when needed

Usage:
    >>> from code_navigator.vectorstore import VectorStore
    >>> store = VectorStore()
    >>> store.add_chunks(chunks)
    >>> results = store.search("parse Python code", k=5)
"""

from .config import VectorStoreSettings, get_settings
from .embeddings import LocalEmbeddingService, get_embedding_service
from .store import VectorStore, get_vector_store

__all__ = [
    # Config
    "VectorStoreSettings",
    "get_settings",
    # Embeddings
    "LocalEmbeddingService",
    "get_embedding_service",
    # Store
    "VectorStore",
    "get_vector_store",
]
