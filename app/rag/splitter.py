"""Splitter interface skeleton; no provider is initialized."""
from collections.abc import Sequence
from .models import Chunk, PaperPage


class Splitter:
    def split(self, pages: Sequence[PaperPage]) -> list[Chunk]:
        """Split within page boundaries and retain source metadata on every chunk."""
        raise NotImplementedError("T01: Splitter is not implemented")
