"""Retriever interface skeleton; no provider is initialized."""
from .models import Evidence


class Retriever:
    def retrieve(self, paper_id: str, question: str, top_k: int) -> list[Evidence]:
        """Embed a question and retrieve evidence restricted to the selected paper."""
        raise NotImplementedError("T01: Retriever is not implemented")
