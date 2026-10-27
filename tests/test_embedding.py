"""Unit tests use a local fake backend, without downloads or cloud calls."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from app.rag import embedding


def test_model_load_is_local_only_on_cpu(monkeypatch):
    factory = Mock()
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=factory),
    )

    model = embedding._load_model(embedding.DEFAULT_MODEL)

    assert model is factory.return_value
    factory.assert_called_once_with(
        embedding.DEFAULT_MODEL,
        device="cpu",
        local_files_only=True,
        trust_remote_code=False,
    )


@pytest.fixture
def backend(monkeypatch):
    model = Mock()
    model.max_seq_length = 256
    model.tokenizer.side_effect = lambda texts, **kwargs: {
        "input_ids": [[0, *range(len(text.split())), 1] for text in texts]
    }
    model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    factory = Mock(return_value=model)
    monkeypatch.setattr(embedding, "_load_model", factory)
    return model, factory


def test_documents_keep_input_order_and_use_normalized_vectors(backend):
    model, factory = backend
    encoder = embedding.LocalEmbeddings()
    assert isinstance(encoder, Embeddings)

    vectors = encoder.embed_documents(["First chunk", "Second chunk"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    factory.assert_called_once_with(embedding.DEFAULT_MODEL)
    model.encode.assert_called_once_with(
        ["First chunk", "Second chunk"],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def test_query_reuses_the_same_model_and_encoding_configuration(backend):
    model, factory = backend
    encoder = embedding.LocalEmbeddings()
    model.encode.return_value = np.array([[0.6, 0.8]])

    document_vector = encoder.embed_documents(["Attention"])[0]
    query_vector = encoder.embed_query("Attention")

    assert query_vector == document_vector == [0.6, 0.8]
    factory.assert_called_once()
    assert model.encode.call_args_list[0] == model.encode.call_args_list[1]


def test_empty_batch_returns_empty_without_encoding(backend):
    model, _ = backend
    assert embedding.LocalEmbeddings().embed_documents([]) == []
    model.encode.assert_not_called()


@pytest.mark.parametrize("text", ["", " \n\t", None, 42])
def test_invalid_text_is_rejected_before_encoding(backend, text):
    model, _ = backend
    with pytest.raises(ValueError, match="非空字符串"):
        embedding.LocalEmbeddings().embed_documents(["Valid", text])
    model.encode.assert_not_called()


def test_blank_query_is_rejected(backend):
    with pytest.raises(ValueError, match="非空字符串"):
        embedding.LocalEmbeddings().embed_query("  ")


def test_overlong_input_is_rejected_instead_of_truncated(backend):
    model, _ = backend
    model.tokenizer.return_value = {"input_ids": [list(range(257))]}
    model.tokenizer.side_effect = None

    with pytest.raises(ValueError, match="文本索引 0 有 257 token"):
        embedding.LocalEmbeddings().embed_documents(["Long input"])

    model.tokenizer.assert_called_once_with(
        ["Long input"], truncation=False, add_special_tokens=True
    )
    model.encode.assert_not_called()


def test_input_at_token_limit_is_allowed(backend):
    model, _ = backend
    model.tokenizer.side_effect = None
    model.tokenizer.return_value = {"input_ids": [list(range(256))]}
    model.encode.return_value = np.array([[1.0, 0.0]])

    assert embedding.LocalEmbeddings().embed_documents(["At limit"]) == [[1.0, 0.0]]
