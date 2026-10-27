"""Retrieval regressions use synthetic text/vectors, never a remote model."""

from unittest.mock import Mock

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
import pytest

from app.rag.retrieval import BM25, Hit, PaperRetriever, index_chunks, infer_sections
from try_chroma_real import metrics


def doc(text, page=1, source="paper.pdf"):
    return Document(page_content=text, metadata={"source": source, "page": page - 1, "page_number": page})


def fake_retriever(chunks, order):
    collection = Mock()

    def query(**kwargs):
        selected = order[:kwargs["n_results"]]
        return {"metadatas": [[{"chunk_index": i} for i in selected]],
                "distances": [[rank / 10 for rank in range(len(selected))]]}

    collection.query.side_effect = query
    return PaperRetriever(collection, chunks)


@pytest.fixture
def sample():
    chunks = [
        doc("<pad>\nFigure 3: Attention mechanism example.", 13),
        doc("3.2 Attention\nAn attention function maps queries and keys to a weighted sum of values.", 3),
        doc("The weights determine the output values.", 4),
        doc("5 Training\nTraining uses the Adam optimizer.", 7),
    ]
    return fake_retriever(chunks, [0, 1, 3, 2])


def test_hybrid_promotes_definition_over_illustration(sample):
    baseline = sample.search("What is the attention mechanism?", [1.0], hybrid=False)
    improved = sample.search("What is the attention mechanism?", [1.0])
    assert baseline[0].index == 0
    assert improved[0].index == 1
    assert len({hit.index for hit in improved}) == len(improved)
    assert improved[0].document.metadata["page_number"] == 3


def test_explicit_figure_query_can_still_retrieve_figures(sample):
    assert sample.search("Figure 3", [1.0])[0].index == 0


def test_neighbor_expansion_is_traced_and_does_not_recurse():
    chunks = [doc("3 Attention\nDefinition")] + [doc("Continuation") for _ in range(4)]
    retriever = fake_retriever(chunks, [0, 4, 3, 2, 1])
    retriever.lexical.scores = lambda question: [20.0, 0.0, 0.0, 0.0, 0.0]

    hits = {hit.index: hit for hit in retriever.search("attention", [1.0])}

    assert hits[1].neighbor_from == 0
    assert hits[1].score == pytest.approx(hits[0].score * 0.85)
    assert hits[2].neighbor_from is None


def test_baseline_keeps_dense_ranking_and_distances(sample):
    hits = sample.search("attention", [1.0], hybrid=False, k=2)
    assert [hit.index for hit in hits] == [0, 1]
    assert [hit.distance for hit in hits] == [0.0, 0.1]


def test_small_corpus_caps_requested_result_count(sample):
    sample.search("attention", [1.0], k=10)
    assert sample.collection.query.call_args.kwargs["n_results"] == 4


@pytest.mark.parametrize("question,k", [(" ", 5), ("attention", 0), ("attention", -1)])
def test_invalid_query_parameters_are_rejected(sample, question, k):
    with pytest.raises(ValueError):
        sample.search(question, [1.0], k=k)
    sample.collection.query.assert_not_called()


def test_empty_corpus_has_no_hits():
    retriever = fake_retriever([], [])
    assert retriever.search("attention", [1.0]) == []
    retriever.collection.query.assert_not_called()


def test_section_inference_ignores_years_and_carries_late_heading_forward():
    chunks = [
        doc("3.2 Attention\n" + "Definition text. " * 10 + "\n3.2.1 Scaled Attention"),
        doc("2014 English-French dataset\nMore text"),
        doc("No heading in a different paper", source="another.pdf"),
    ]
    assert infer_sections(chunks) == ["Attention", "Scaled Attention", ""]


def test_context_continues_across_pages_but_not_sections_or_sources(sample):
    hit = Hit(1, sample.chunks[1], "Attention", 1.0, 0.0)
    assert sample.continuation(hit).metadata["page_number"] == 4
    hit = Hit(2, sample.chunks[2], "Attention", 1.0, 0.0)
    assert sample.continuation(hit) is None
    sample.chunks[2].metadata["source"] = "another.pdf"
    hit = Hit(1, sample.chunks[1], "Attention", 1.0, 0.0)
    assert sample.continuation(hit) is None


def test_bm25_prefers_specific_terms_and_handles_unknown_queries():
    index = BM25(["attention attention", "learning rate warmup optimizer", ""])
    scores = index.scores("warmup optimizer")
    assert scores[1] > scores[0] == scores[2] == 0
    assert index.scores("unknown") == [0, 0, 0]


def test_index_reuse_and_changed_content_do_not_pollute_existing_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path), settings=Settings(anonymized_telemetry=False))
    old = client.create_collection("existing_user_data", embedding_function=None)
    old.add(ids=["original"], documents=["Keep me"], embeddings=[[1.0, 0.0, 0.0]])
    encoder = Mock(model_name="test-model")
    encoder.embed_documents.side_effect = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    chunks = [doc("3.2 Attention\nDefinition"), doc("Continuation", 2)]
    first = index_chunks(client, chunks, encoder)
    second = index_chunks(client, chunks, encoder)
    assert first.name == second.name
    assert first.count() == 2
    encoder.embed_documents.assert_called_once()
    metadata = first.get(ids=["chunk_1"])["metadatas"][0]
    assert metadata == {"source": "paper.pdf", "page": 1, "page_number": 2, "chunk_index": 1}

    changed = index_chunks(client, [doc("Changed corpus")], encoder)
    assert changed.name != first.name
    assert changed.count() == 1
    assert first.count() == 2
    assert old.get()["documents"] == ["Keep me"]


def test_metrics_distinguish_related_from_direct_answer():
    result = metrics([0, 1, 2, 3, 4], {0: 1, 1: 0, 2: 2, 3: 1, 4: 0})
    assert result["direct_precision_at_5"] == 0.2
    assert result["related_precision_at_5"] == 0.6
    assert result["first_direct_rank"] == 3
    assert 0 < result["ndcg_at_5"] < 1
    assert metrics([], {})["ndcg_at_5"] == 0
