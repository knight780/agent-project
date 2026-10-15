"""Provider-independent input/output contracts. Page numbers are one-based."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PaperPage:
    paper_id: str
    paper_title: str
    source_path: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: PaperPage
    text: str


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    vector: tuple[float, ...]


@dataclass(frozen=True)
class Evidence:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    paper_title: str
    page_number: int
    source_path: str
    excerpt: str
    chunk_id: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    refused: bool
    refusal_reason: str | None = None
