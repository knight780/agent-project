"""Embedding interface skeleton; no provider is initialized."""
from collections.abc import Sequence


class Embedding:
    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Return one embedding per input text in input order."""
        raise NotImplementedError("T01: Embedding is not implemented")

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Return a query vector using the same model as document embeddings."""
        raise NotImplementedError("T01: query embedding is not implemented")
