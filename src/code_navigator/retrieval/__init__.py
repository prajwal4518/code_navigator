"""
Retrieval Module - Find relevant code using hybrid search.

This module handles:
- Semantic search (embedding similarity)
- Keyword search (BM25, exact matches)
- Hybrid ranking (combining both strategies)
- Re-ranking for precision

Why Hybrid Search?
- Semantic alone misses exact variable/function names
- Keyword alone misses conceptual similarity ("authentication" vs "login")
- Hybrid captures both: find code by meaning AND by name

Usage:
    >>> from code_navigator.retrieval import HybridRetriever, SearchMode
    >>> retriever = HybridRetriever()
    >>> results = retriever.search("parse Python code", mode=SearchMode.HYBRID)
"""

from .bm25 import BM25Index
from .hybrid import HybridRetriever, SearchMode, SearchResult, get_hybrid_retriever

__all__ = [
    # BM25
    "BM25Index",
    # Hybrid
    "HybridRetriever",
    "SearchMode",
    "SearchResult",
    "get_hybrid_retriever",
]
