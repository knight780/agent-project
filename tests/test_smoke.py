"""Offline checks for imports and the public skeleton boundary."""
import importlib
import socket
from pathlib import Path

import pytest


def test_modules_import_without_network(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden in smoke tests")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    for name in (
        "app", "app.config", "app.config.settings", "app.rag", "app.rag.models",
        "app.rag.pdf_loader", "app.rag.splitter", "app.rag.embedding",
        "app.rag.vector_store", "app.rag.retriever", "app.rag.generator",
        "app.rag.service",
    ):
        importlib.import_module(name)


def test_service_is_explicitly_unimplemented():
    from app.rag.embedding import Embedding
    from app.rag.generator import Generator
    from app.rag.pdf_loader import PDFLoader
    from app.rag.retriever import Retriever
    from app.rag.service import RAGService
    from app.rag.splitter import Splitter
    from app.rag.vector_store import VectorStore

    service = RAGService(PDFLoader(), Splitter(), Embedding(), VectorStore(), Retriever(), Generator())
    with pytest.raises(NotImplementedError, match="T01"):
        service.ingest(Path("example.pdf"))
    with pytest.raises(NotImplementedError, match="T01"):
        service.ask("paper-1", "What is the conclusion?")


def test_settings_construct_without_credentials(monkeypatch):
    from app.config import Settings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Settings().answer_language == "zh-CN"
