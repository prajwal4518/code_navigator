"""
ChromaDB vector store for code chunks.

Why ChromaDB?
- Local-first: No external dependencies for development
- Persistent: Survives restarts without re-indexing
- Metadata filtering: Query by file type, function name, etc.
- Easy migration to cloud (Pinecone, Weaviate) when needed

This module wraps ChromaDB with our CodeChunk model
for seamless integration with the ingestion pipeline.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from rich.console import Console

from code_navigator.ingestion.models import ChunkType, CodeChunk, Language

from .config import get_settings
from .embeddings import LocalEmbeddingService, get_embedding_service

console = Console()


class VectorStore:
    """ChromaDB-based vector store for code chunks.

    Features:
    - Persistent storage
    - Batch upsert with deduplication
    - Metadata filtering
    - Semantic search

    Usage:
        >>> store = VectorStore()
        >>> store.add_chunks(chunks)
        >>> results = store.search("parse Python code", k=5)
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embedding_service: LocalEmbeddingService | None = None,
    ):
        """Initialize the vector store.

        Args:
            persist_dir: Override ChromaDB persistence directory
            collection_name: Override collection name
            embedding_service: Custom embedding service
        """
        settings = get_settings()

        self.persist_dir = persist_dir or str(settings.chroma_persist_dir)
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedding_service = embedding_service or get_embedding_service()

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        console.print(
            f"[dim]VectorStore initialized: {self.collection_name} "
            f"({self._collection.count()} existing chunks)[/]"
        )

    def add_chunks(self, chunks: list[CodeChunk]) -> int:
        """Add code chunks to the store.

        Uses upsert for deduplication - chunks with same ID are updated.

        Args:
            chunks: List of CodeChunk objects to add

        Returns:
            Number of chunks added/updated
        """
        if not chunks:
            return 0

        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.to_embedding_text() for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]

        # Generate embeddings in batch
        console.print(f"[dim]Generating embeddings for {len(chunks)} chunks...[/]")
        embeddings = self.embedding_service.embed_batch(documents)

        # Upsert to ChromaDB
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        console.print(f"[green]✓ Added {len(chunks)} chunks to store[/]")
        return len(chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        chunk_type: ChunkType | None = None,
        file_path: str | None = None,
        language: Language | None = None,
    ) -> list[tuple[CodeChunk, float]]:
        """Search for similar code chunks.

        Args:
            query: Natural language search query
            k: Number of results to return
            chunk_type: Filter by chunk type (function, class, method)
            file_path: Filter by file path (partial match)
            language: Filter by programming language

        Returns:
            List of (CodeChunk, similarity_score) tuples
        """
        # Build where clause for filtering
        where: dict | None = None
        where_clauses: list[dict] = []

        if chunk_type:
            where_clauses.append({"chunk_type": chunk_type.value})

        if language:
            where_clauses.append({"language": language.value})

        if file_path:
            where_clauses.append({"file_path": {"$contains": file_path}})

        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert results to CodeChunks
        chunks_with_scores: list[tuple[CodeChunk, float]] = []

        if results["ids"] and results["ids"][0]:
            for i, _chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                document = results["documents"][0][i]
                distance = results["distances"][0][i]

                # Convert distance to similarity (cosine distance -> similarity)
                similarity = 1 - distance

                # Reconstruct CodeChunk from metadata
                chunk = CodeChunk(
                    content=document,  # Note: We stored embedding text, not raw content
                    chunk_type=ChunkType(metadata["chunk_type"]),
                    name=metadata["name"],
                    file_path=metadata["file_path"],
                    start_line=metadata["start_line"],
                    end_line=metadata["end_line"],
                    language=Language(metadata["language"]),
                    docstring=None,  # Not stored in metadata
                    parent_name=metadata.get("parent_name") or None,
                    decorators=metadata.get("decorators", "").split(",")
                    if metadata.get("decorators")
                    else [],
                )

                chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks from a specific file.

        Useful for incremental updates - delete old chunks before re-indexing.

        Args:
            file_path: Absolute path to the file

        Returns:
            Number of chunks deleted
        """
        # Get chunks to delete
        results = self._collection.get(
            where={"file_path": file_path},
            include=["metadatas"],
        )

        if not results["ids"]:
            return 0

        count = len(results["ids"])
        self._collection.delete(ids=results["ids"])

        console.print(f"[yellow]Deleted {count} chunks from {file_path}[/]")
        return count

    def clear(self) -> int:
        """Clear all chunks from the collection.

        Returns:
            Number of chunks deleted
        """
        count = self._collection.count()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        console.print(f"[yellow]Cleared {count} chunks from store[/]")
        return count

    def get_stats(self) -> dict:
        """Get statistics about the store.

        Returns:
            Dict with count, embedding dimension, etc.
        """
        return {
            "collection_name": self.collection_name,
            "chunk_count": self._collection.count(),
            "embedding_dimension": self.embedding_service.dimension,
            "persist_dir": self.persist_dir,
        }


def get_vector_store() -> VectorStore:
    """Get the default vector store instance."""
    return VectorStore()
