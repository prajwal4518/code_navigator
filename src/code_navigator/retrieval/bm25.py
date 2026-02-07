"""
BM25 keyword search implementation.

BM25 (Best Matching 25) is a ranking function used for keyword-based search.
It's the algorithm behind Elasticsearch's default scoring.

Why BM25 instead of simple TF-IDF?
- Better handling of term frequency saturation
- Document length normalization
- Tunable parameters (k1, b)

Why in-memory instead of external service?
- No additional infrastructure
- Fast for small-medium codebases
- Easy to rebuild from ChromaDB metadata
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from code_navigator.ingestion.models import CodeChunk


@dataclass
class BM25Index:
    """In-memory BM25 index for keyword search.

    Usage:
        >>> index = BM25Index()
        >>> index.add_documents(chunks)
        >>> results = index.search("parse_file", k=5)
    """

    # BM25 parameters
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Document length normalization

    # Index data
    documents: list[CodeChunk] = field(default_factory=list)
    doc_freqs: dict[str, int] = field(default_factory=dict)
    doc_lengths: list[int] = field(default_factory=list)
    avg_doc_length: float = 0.0
    doc_term_freqs: list[dict[str, int]] = field(default_factory=list)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms.

        Simple tokenization that:
        - Splits on non-alphanumeric characters
        - Lowercases everything
        - Handles snake_case and camelCase
        """
        # Split camelCase
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Split on non-alphanumeric
        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def _get_searchable_text(self, chunk: CodeChunk) -> str:
        """Get all searchable text from a chunk.

        Combines multiple fields for comprehensive keyword matching:
        - Name (most important for exact matches)
        - Docstring (natural language descriptions)
        - Content (actual code)
        """
        parts = [
            chunk.name,
            chunk.name,  # Double-weight the name
            chunk.chunk_type.value,
        ]

        if chunk.docstring:
            parts.append(chunk.docstring)

        if chunk.parent_name:
            parts.append(chunk.parent_name)

        # Add decorators
        parts.extend(chunk.decorators)

        # Add content (but tokenized, so we get identifier names)
        parts.append(chunk.content)

        return " ".join(parts)

    def add_documents(self, chunks: list[CodeChunk]) -> None:
        """Add chunks to the BM25 index.

        Args:
            chunks: List of CodeChunk objects to index
        """
        self.documents = chunks
        self.doc_term_freqs = []
        self.doc_lengths = []

        # First pass: compute term frequencies per document
        for chunk in chunks:
            text = self._get_searchable_text(chunk)
            tokens = self.tokenize(text)
            term_freqs = Counter(tokens)
            self.doc_term_freqs.append(dict(term_freqs))
            self.doc_lengths.append(len(tokens))

        # Compute average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 0

        # Compute document frequencies (how many docs contain each term)
        self.doc_freqs = {}
        for term_freqs in self.doc_term_freqs:
            for term in term_freqs:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def _score_document(self, query_terms: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document.

        The BM25 formula:
        score = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))

        Where:
        - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
        - f(qi, D) = frequency of term qi in document D
        - |D| = document length
        - avgdl = average document length
        """
        if not self.documents:
            return 0.0

        N = len(self.documents)
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]

        score = 0.0
        for term in query_terms:
            if term not in self.doc_freqs:
                continue

            # Document frequency
            df = self.doc_freqs[term]

            # Inverse document frequency
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            # Term frequency in this document
            tf = term_freqs.get(term, 0)

            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_length
            )
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, k: int = 10) -> list[tuple[CodeChunk, float]]:
        """Search for chunks matching the query.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (CodeChunk, score) tuples, sorted by score descending
        """
        if not self.documents:
            return []

        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        # Score all documents
        scores: list[tuple[int, float]] = []
        for doc_idx in range(len(self.documents)):
            score = self._score_document(query_terms, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results: list[tuple[CodeChunk, float]] = []
        for doc_idx, score in scores[:k]:
            results.append((self.documents[doc_idx], score))

        return results

    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return len(self.documents)
