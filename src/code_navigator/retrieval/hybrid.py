"""
Hybrid retrieval combining semantic and keyword search.

Why hybrid?
- Semantic search finds conceptually similar code
- Keyword search finds exact term matches
- Hybrid gets the best of both worlds

Reciprocal Rank Fusion (RRF):
- Combines rankings from multiple retrieval methods
- Simple but effective: score = Σ 1/(k + rank)
- k=60 is widely used (reduces impact of outliers)
"""

from dataclasses import dataclass
from enum import StrEnum

from rich.console import Console

from code_navigator.ingestion.models import ChunkType, CodeChunk, Language
from code_navigator.vectorstore import VectorStore

from .bm25 import BM25Index

console = Console()


class SearchMode(StrEnum):
    """Search mode options."""

    SEMANTIC = "semantic"  # Vector similarity only
    KEYWORD = "keyword"  # BM25 only
    HYBRID = "hybrid"  # Combined with RRF


@dataclass
class SearchResult:
    """A search result with scores from different methods."""

    chunk: CodeChunk
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    semantic_rank: int | None = None
    keyword_rank: int | None = None


class HybridRetriever:
    """Hybrid retriever combining semantic and keyword search.

    Usage:
        >>> retriever = HybridRetriever()
        >>> results = retriever.search("parse Python code", k=5)
        >>> for result in results:
        ...     print(result.chunk.name, result.combined_score)
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        rrf_k: int = 60,
    ):
        """Initialize the hybrid retriever.

        Args:
            vector_store: ChromaDB vector store (will create default if None)
            rrf_k: Reciprocal Rank Fusion constant (default 60)
        """
        self.vector_store = vector_store or VectorStore()
        self.rrf_k = rrf_k
        self._bm25_index: BM25Index | None = None

    @property
    def bm25_index(self) -> BM25Index:
        """Lazy-load BM25 index from vector store.

        Why lazy loading?
        - Don't build index if only doing semantic search
        - Rebuild when store contents change
        """
        if self._bm25_index is None:
            self._build_bm25_index()
        return self._bm25_index

    def _build_bm25_index(self) -> None:
        """Build BM25 index from vector store contents."""
        console.print("[dim]Building BM25 index from vector store...[/]")

        # Get all chunks from vector store
        # We need to query with a dummy to get all documents
        # This is a workaround - ideally ChromaDB would have a get_all method
        collection = self.vector_store._collection
        all_data = collection.get(include=["documents", "metadatas"])

        if not all_data["ids"]:
            self._bm25_index = BM25Index()
            return

        # Reconstruct CodeChunks from metadata
        chunks: list[CodeChunk] = []
        for i, _chunk_id in enumerate(all_data["ids"]):
            metadata = all_data["metadatas"][i]
            document = all_data["documents"][i]

            chunk = CodeChunk(
                content=document,
                chunk_type=ChunkType(metadata["chunk_type"]),
                name=metadata["name"],
                file_path=metadata["file_path"],
                start_line=metadata["start_line"],
                end_line=metadata["end_line"],
                language=Language(metadata["language"]),
                docstring=None,
                parent_name=metadata.get("parent_name") or None,
                decorators=(
                    metadata.get("decorators", "").split(",")
                    if metadata.get("decorators")
                    else []
                ),
            )
            chunks.append(chunk)

        # Build index
        self._bm25_index = BM25Index()
        self._bm25_index.add_documents(chunks)
        console.print(f"[green]✓ BM25 index built with {len(chunks)} chunks[/]")

    def invalidate_bm25_cache(self) -> None:
        """Invalidate the BM25 index cache.

        Call this after adding/removing chunks from vector store.
        """
        self._bm25_index = None

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[tuple[CodeChunk, float]],
        keyword_results: list[tuple[CodeChunk, float]],
    ) -> list[SearchResult]:
        """Combine results using Reciprocal Rank Fusion.

        RRF score = Σ 1/(k + rank)

        Where k=60 reduces the impact of high ranks from a single method.
        """
        results_by_id: dict[str, SearchResult] = {}

        # Process semantic results
        for rank, (chunk, score) in enumerate(semantic_results, start=1):
            chunk_id = chunk.chunk_id
            if chunk_id not in results_by_id:
                results_by_id[chunk_id] = SearchResult(chunk=chunk)

            results_by_id[chunk_id].semantic_score = score
            results_by_id[chunk_id].semantic_rank = rank
            results_by_id[chunk_id].combined_score += 1 / (self.rrf_k + rank)

        # Process keyword results
        for rank, (chunk, score) in enumerate(keyword_results, start=1):
            chunk_id = chunk.chunk_id
            if chunk_id not in results_by_id:
                results_by_id[chunk_id] = SearchResult(chunk=chunk)

            results_by_id[chunk_id].keyword_score = score
            results_by_id[chunk_id].keyword_rank = rank
            results_by_id[chunk_id].combined_score += 1 / (self.rrf_k + rank)

        # Sort by combined score
        results = list(results_by_id.values())
        results.sort(key=lambda r: r.combined_score, reverse=True)

        return results

    def search(
        self,
        query: str,
        k: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
        chunk_type: ChunkType | None = None,
        language: Language | None = None,
    ) -> list[SearchResult]:
        """Search for relevant code chunks.

        Args:
            query: Natural language or keyword query
            k: Number of results to return
            mode: Search mode (semantic, keyword, or hybrid)
            chunk_type: Filter by chunk type
            language: Filter by programming language

        Returns:
            List of SearchResult objects with scores
        """
        results: list[SearchResult] = []

        if mode == SearchMode.SEMANTIC:
            # Semantic only
            semantic_results = self.vector_store.search(
                query, k=k, chunk_type=chunk_type, language=language
            )
            for chunk, score in semantic_results:
                results.append(
                    SearchResult(
                        chunk=chunk,
                        semantic_score=score,
                        combined_score=score,
                    )
                )

        elif mode == SearchMode.KEYWORD:
            # Keyword only
            keyword_results = self.bm25_index.search(query, k=k)

            # Apply filters manually (BM25 doesn't support filtering)
            filtered: list[tuple[CodeChunk, float]] = []
            for chunk, score in keyword_results:
                if chunk_type and chunk.chunk_type != chunk_type:
                    continue
                if language and chunk.language != language:
                    continue
                filtered.append((chunk, score))

            for chunk, score in filtered[:k]:
                results.append(
                    SearchResult(
                        chunk=chunk,
                        keyword_score=score,
                        combined_score=score,
                    )
                )

        else:  # HYBRID
            # Get more results from each method for better fusion
            fetch_k = k * 3

            semantic_results = self.vector_store.search(
                query, k=fetch_k, chunk_type=chunk_type, language=language
            )
            keyword_results = self.bm25_index.search(query, k=fetch_k)

            # Apply filters to keyword results
            if chunk_type or language:
                filtered: list[tuple[CodeChunk, float]] = []
                for chunk, score in keyword_results:
                    if chunk_type and chunk.chunk_type != chunk_type:
                        continue
                    if language and chunk.language != language:
                        continue
                    filtered.append((chunk, score))
                keyword_results = filtered

            # Combine with RRF
            results = self._reciprocal_rank_fusion(semantic_results, keyword_results)[
                :k
            ]

        return results


def get_hybrid_retriever() -> HybridRetriever:
    """Get the default hybrid retriever."""
    return HybridRetriever()
