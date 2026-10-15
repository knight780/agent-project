"""Offline configuration values; creating settings never contacts a provider."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Provider selection and credentials will be added when adapters are implemented."""

    chroma_directory: Path = Path("data/chroma")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 4
    answer_language: str = "zh-CN"
