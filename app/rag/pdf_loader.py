"""PDFLoader interface skeleton; no provider is initialized."""
from pathlib import Path
from .models import PaperPage


class PDFLoader:
    def load(self, path: Path) -> list[PaperPage]:
        """Read one PDF into pages, preserving real title, path and one-based page numbers."""
        raise NotImplementedError("T01: PDFLoader is not implemented")
