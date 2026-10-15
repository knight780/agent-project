"""Generator interface skeleton; no provider is initialized."""
from collections.abc import Sequence
from .models import Answer, Evidence


class Generator:
    def generate(self, question: str, evidence: Sequence[Evidence]) -> Answer:
        """Answer in Chinese with grounded citations, or explicitly refuse for insufficient evidence."""
        raise NotImplementedError("T01: Generator is not implemented")
