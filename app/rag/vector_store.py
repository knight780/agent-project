"""VectorStore interface skeleton; no provider is initialized."""
from collections.abc import Sequence
from .models import EmbeddedChunk, Evidence


class VectorStore:
    def add(self, paper_id: str, chunks: Sequence[EmbeddedChunk]) -> None:
        """Persist embeddings and metadata in a paper-scoped Chroma collection."""
        raise NotImplementedError("T01: VectorStore is not implemented")

    def search(self, paper_id: str, query_vector: Sequence[float], top_k: int) -> list[Evidence]:
        """Return ranked evidence for one paper; score semantics await adapter design."""
        raise NotImplementedError("T01: vector search is not implemented")
