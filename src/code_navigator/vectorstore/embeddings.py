"""
Embedding service for converting code to vectors.

Why local embeddings first?
- No API costs during development
- Faster iteration (no network latency)
- Privacy (code stays local)
- Easy to switch to Gemini embeddings for production

Why sentence-transformers?
- State-of-the-art embedding models
- Optimized for semantic similarity
- GPU acceleration when available
- Well-tested with code-related queries
"""

from typing import Protocol

from rich.console import Console
from sentence_transformers import SentenceTransformer

from .config import get_settings

console = Console()


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Allows swapping between local and API embeddings.
    """

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class LocalEmbeddingService:
    """Local embedding service using sentence-transformers.

    Uses the all-MiniLM-L6-v2 model by default:
    - 384 dimensions
    - Fast inference
    - Good balance of quality and speed
    """

    def __init__(self, model_name: str | None = None):
        """Initialize the embedding service.

        Args:
            model_name: Override the default model from settings
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model.

        Why lazy loading?
        - Model download can be slow on first run
        - Don't load until actually needed
        - Allows checking settings before loading
        """
        if self._model is None:
            console.print(f"[dim]Loading embedding model: {self.model_name}[/]")
            self._model = SentenceTransformer(self.model_name)
            console.print(f"[green]✓ Loaded model with {self.dimension} dimensions[/]")
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently.

        Why batch processing?
        - 10-100x faster than one-at-a-time
        - Better GPU utilization
        - Reduced overhead per embedding

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        settings = get_settings()
        embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
        )
        return embeddings.tolist()


def get_embedding_service() -> LocalEmbeddingService:
    """Get the default embedding service.

    Currently returns local embeddings.
    In future, can check settings to return API embeddings.
    """
    return LocalEmbeddingService()
