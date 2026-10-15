"""Application-facing orchestration skeleton."""
from pathlib import Path
from .embedding import Embedding
from .generator import Generator
from .models import Answer
from .pdf_loader import PDFLoader
from .retriever import Retriever
from .splitter import Splitter
from .vector_store import VectorStore


class RAGService:
    def __init__(
        self, loader: PDFLoader, splitter: Splitter, embedding: Embedding,
        vector_store: VectorStore, retriever: Retriever, generator: Generator,
    ) -> None:
        self.loader = loader
        self.splitter = splitter
        self.embedding = embedding
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator

    def ingest(self, path: Path) -> str:
        """Index a single PDF and return its stable paper_id (not implemented)."""
        raise NotImplementedError("T01: ingestion is not implemented")

    def ask(self, paper_id: str, question: str) -> Answer:
        """Return a Chinese grounded answer or refusal (not implemented)."""
        raise NotImplementedError("T01: question answering is not implemented")
