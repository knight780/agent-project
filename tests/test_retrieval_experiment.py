"""Paired experiment contracts and isolated real-PDF/Chroma integration, offline."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pymupdf
import pytest
from langchain_core.documents import Document

from app.rag.retrieval import Hit
from app.retrieval_evaluation import EvaluationError, load_json, text_sha256
from app.retrieval_experiment import (chunk_uid, compare_retrieval, embedding_identity, fingerprint, new_run_directory,
                                      prepare_corpus, retrievers_from_manifest, save_new, scan_pdfs)


@pytest.fixture
def experiment():
    papers = {}
    for letter in ("a", "b", "c"):
        texts = [f"Evidence about topic {letter} on page {i + 1}. " * 4 for i in range(9)]
        papers[letter * 64] = {"pdf_sha256": letter * 64, "source": f"data/{letter}.pdf", "page_count": 9,
                              "chunk_text_sha256": text_sha256(json.dumps(texts, ensure_ascii=False)),
                              "chunks": {f"chunk_{i}": {"text": text, "text_sha256": text_sha256(text),
                                                        "page_number": i + 1, "section": f"Section {i}"}
                                         for i, text in enumerate(texts)}}
    corpus = {"schema_version": 2, "dataset_version": "corpus-only", "cases": [], "papers": papers,
              "chunk_config": {"chunk_size": 600, "chunk_overlap": 120},
              "corpus_version": "synthetic-corpus-v1", "index_config": {"embedding_model": "test-model"}}
    pid = "a" * 64
    case = {"question_id": "q1", "question": "What is topic alpha?", "paper_id": pid,
            "origin": "human", "answerability": "answerable",
            "review": {"status": "approved", "reviewer": "test", "reviewed_at": "2026-10-26T12:00:00Z"},
            "review_history": [{"action": "approve", "reviewer": "test", "timestamp": "2026-10-26T12:00:00Z"}],
            "labels": {"completeness": "complete_reviewed", "judgments": {f"chunk_{i}": 2 if i == 1 else 0 for i in range(9)}},
            "target_evidence": [{"chunk_id": "chunk_1", "source": papers[pid]["source"], "page_number": 2,
                                 "text": papers[pid]["chunks"]["chunk_1"]["text"]}]}
    pending = deepcopy(case)
    pending.update(question_id="q2", question="What is topic beta?", origin="llm_generated", review={"status": "pending"})
    pending["labels"]["completeness"] = "anchor_only"
    dataset = {key: deepcopy(corpus[key]) for key in ("schema_version", "dataset_version", "papers", "chunk_config")}
    dataset["dataset_version"] = "synthetic-questions-v1"
    dataset["cases"] = [case, pending]
    encoder = Mock(model_name="test-model")
    encoder.embed_query.return_value = [0.1, 0.2, 0.3]
    retriever = Mock()

    def search(query, vector, *, hybrid, k):
        indices = [1, 0] if hybrid else [0, 1]
        return [Hit(i, Document(page_content=papers[pid]["chunks"][f"chunk_{i}"]["text"],
                                metadata={"source": papers[pid]["source"], "page_number": i + 1, "page": i}),
                    "Section", 0.8, 0.2) for i in indices[:k]]

    retriever.search.side_effect = search
    return corpus, dataset, {pid: retriever}, encoder


def test_paired_queries_vectors_config_and_real_ids(experiment):
    corpus, dataset, retrievers, encoder = experiment
    report = compare_retrieval(dataset, corpus, retrievers, encoder, k=5)
    assert encoder.embed_query.call_count == 2
    calls = retrievers["a" * 64].search.call_args_list
    for baseline, hybrid in zip(calls[::2], calls[1::2]):
        assert baseline.args == hybrid.args
        assert baseline.kwargs == {"hybrid": False, "k": 5}
        assert hybrid.kwargs == {"hybrid": True, "k": 5}
    first = report["cases"][0]
    dense, mixed = first["modes"]["baseline"], first["modes"]["hybrid"]
    assert dense["config_version"] == mixed["config_version"]
    assert dense["known_target_ranks"] == {"chunk_1": 2}
    assert mixed["known_target_ranks"] == {"chunk_1": 1}
    assert dense["hits"][1]["page_number"] == 2 and dense["hits"][1]["page"] == 1
    assert dense["hits"][1]["valid"] is True
    assert dense["hits"][1]["chunk_uid"] == chunk_uid("a" * 64, corpus["papers"]["a" * 64], "chunk_1")
    assert report["evaluations"]["baseline"]["summary"]["mean_metrics"]["mrr_at_k"] == 0.5
    assert report["evaluations"]["hybrid"]["summary"]["mean_metrics"]["mrr_at_k"] == 1
    assert report["cases"][1]["experiment_kind"] == "exploratory"
    assert report["evaluations"]["baseline"]["summary"]["formal_sample_count"] == 1


def test_all_pending_runs_real_ranking_but_no_formal_metrics(experiment):
    corpus, dataset, retrievers, encoder = experiment
    dataset["cases"] = [dataset["cases"][1]]
    report = compare_retrieval(dataset, corpus, retrievers, encoder)
    assert retrievers["a" * 64].search.call_count == 2
    for mode in ("baseline", "hybrid"):
        assert report["evaluations"][mode]["summary"]["formal_sample_count"] == 0
        assert report["evaluations"][mode]["summary"]["mean_metrics"]["recall_at_k"] is None
    assert report["comparison"]["formal_metric_deltas"]["mrr_at_k"] is None


def test_empty_results_are_valid_and_zero(experiment):
    corpus, dataset, retrievers, encoder = experiment
    retrievers["a" * 64].search.side_effect = lambda *a, **kw: []
    report = compare_retrieval(dataset, corpus, retrievers, encoder)
    assert report["comparison"]["paired_error_count"] == 0
    assert report["evaluations"]["baseline"]["summary"]["mean_metrics"]["recall_at_k"] == 0


@pytest.mark.parametrize("fault,expected", [("duplicate", "duplicate_retrieved_chunk"),
                                            ("source", "retrieval_source_mismatch"),
                                            ("text", "retrieval_text_hash_mismatch"),
                                            ("page", "retrieval_page_index_mismatch")])
def test_bad_results_are_explicit_and_exclude_both_modes(experiment, fault, expected):
    corpus, dataset, retrievers, encoder = experiment
    retriever = retrievers["a" * 64]
    original = retriever.search.side_effect

    def search(*args, **kwargs):
        results = original(*args, **kwargs)
        if kwargs["hybrid"]:
            if fault == "duplicate":
                results.append(results[0])
            elif fault == "source":
                results[0].document.metadata["source"] = "other-paper.pdf"
            elif fault == "text":
                results[0].document.page_content = "wrong content"
            else:
                results[0].document.metadata["page"] = 999
        return results

    retriever.search.side_effect = search
    report = compare_retrieval(dataset, corpus, retrievers, encoder)
    assert report["cases"][0]["modes"]["hybrid"]["error"] == expected
    assert report["cases"][0]["modes"]["baseline"]["valid"]
    for mode in ("baseline", "hybrid"):
        assert report["evaluations"][mode]["summary"]["invalid_sample_count"] == 1
        assert report["evaluations"][mode]["summary"]["mean_metrics"]["recall_at_k"] is None


def test_embedding_failure_is_redacted_and_not_a_zero_score(experiment):
    corpus, dataset, retrievers, encoder = experiment
    encoder.embed_query.side_effect = RuntimeError("secret-key private-path")
    report = compare_retrieval(dataset, corpus, retrievers, encoder)
    assert "secret-key" not in json.dumps(report)
    assert report["cases"][0]["paired_error"] == "RuntimeError"
    assert report["comparison"]["paired_error_count"] == 2


@pytest.mark.parametrize("field", ["source", "chunk_text_sha256", "page_count"])
def test_live_corpus_mismatch_prevents_query(experiment, field):
    corpus, dataset, retrievers, encoder = experiment
    corpus["papers"]["a" * 64][field] = 99 if field == "page_count" else "wrong"
    with pytest.raises(EvaluationError):
        compare_retrieval(dataset, corpus, retrievers, encoder)
    encoder.embed_query.assert_not_called()


def test_cross_paper_and_chunk_versions_have_different_uids(experiment):
    corpus = experiment[0]
    a = chunk_uid("a" * 64, corpus["papers"]["a" * 64], "chunk_0")
    b = chunk_uid("b" * 64, corpus["papers"]["b" * 64], "chunk_0")
    assert a != b
    changed = deepcopy(corpus["papers"]["a" * 64])
    changed["chunk_text_sha256"] = "d" * 64
    assert a != chunk_uid("a" * 64, changed, "chunk_0")


def test_invalid_case_is_reported_without_crashing_or_embedding(experiment):
    corpus, dataset, retrievers, encoder = experiment
    dataset["cases"] = [dataset["cases"][0]]
    dataset["cases"][0]["review"] = None
    report = compare_retrieval(dataset, corpus, retrievers, encoder)
    assert report["cases"][0]["paired_error"] == "invalid_review_status"
    encoder.embed_query.assert_not_called()


def test_wrong_embedding_model_is_rejected(experiment):
    corpus, dataset, retrievers, encoder = experiment
    encoder.model_name = "changed-model"
    with pytest.raises(EvaluationError, match="embedding_model_mismatch"):
        compare_retrieval(dataset, corpus, retrievers, encoder)


def test_rejected_questions_are_explicitly_skipped_without_query(experiment):
    corpus, dataset, retrievers, encoder = experiment
    dataset["cases"] = [dataset["cases"][1]]
    dataset["cases"][0]["review"]["status"] = "rejected"
    report = compare_retrieval(dataset, corpus, retrievers, encoder)
    assert report["comparison"]["paired_skipped_count"] == 1
    assert report["comparison"]["paired_success_count"] == 0
    assert report["comparison"]["paired_error_count"] == 0
    encoder.embed_query.assert_not_called()


def test_changed_weights_or_tokenizer_invalidate_same_name_cache():
    import torch

    weights = {"weight": torch.tensor([1.0, 2.0])}
    backend = Mock()
    backend.to_str.return_value = "tokenizer-v1"
    model = SimpleNamespace(state_dict=lambda: weights, tokenizer=SimpleNamespace(backend_tokenizer=backend), max_seq_length=256)
    encoder = SimpleNamespace(model_name="same-name", _model=model)
    first = embedding_identity(encoder)
    weights["weight"][0] = 2.0
    second = embedding_identity(encoder)
    assert first["weights_sha256"] != second["weights_sha256"]
    backend.to_str.return_value = "tokenizer-v2"
    assert second["tokenizer_sha256"] != embedding_identity(encoder)["tokenizer_sha256"]


def test_query_vector_mutation_does_not_leak_between_modes(experiment):
    corpus, dataset, retrievers, encoder = experiment
    dataset["cases"] = [dataset["cases"][0]]
    observed = []

    def search(query, vector, **kwargs):
        observed.append(list(vector))
        vector[0] = 999
        return []

    retrievers["a" * 64].search.side_effect = search
    report = compare_retrieval(dataset, corpus, retrievers, encoder)
    assert observed == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert report["cases"][0]["query_vector"] == [0.1, 0.2, 0.3]


def _pdf(path, text):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 50), text)
        path.write_bytes(pdf.tobytes())


def test_inventory_distinguishes_duplicates_versions_and_upload_cache(tmp_path):
    _pdf(tmp_path / "1234.56789v1.pdf", "First version evidence")
    _pdf(tmp_path / "1234.56789v2.pdf", "Second version evidence")
    (tmp_path / "copy.pdf").write_bytes((tmp_path / "1234.56789v1.pdf").read_bytes())
    (tmp_path / "web").mkdir()
    (tmp_path / "web/paper.pdf").write_bytes((tmp_path / "1234.56789v1.pdf").read_bytes())
    rows = {r["path"]: r for r in scan_pdfs(tmp_path)}
    assert rows["copy.pdf"]["status"] == "byte_duplicate"
    assert rows["web/paper.pdf"]["status"] == "excluded_upload_cache"
    assert rows["1234.56789v1.pdf"]["arxiv_id"] == rows["1234.56789v2.pdf"]["arxiv_id"]
    assert rows["1234.56789v1.pdf"]["paper_id"] != rows["1234.56789v2.pdf"]["paper_id"]


def test_real_pdf_and_isolated_index_reuse_with_mock_embedding(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _pdf(root / "alpha.pdf", "Alpha mechanism evidence is stored here.")
    _pdf(root / "beta.pdf", "Beta training evidence is stored here.")
    encoder = Mock(model_name="test-model")
    encoder.embed_documents.side_effect = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    kwargs = {"max_papers": 2, "model_name": "test-model", "encoder": encoder}
    first, client, _ = prepare_corpus(root, tmp_path / "evaluation", **kwargs)
    second, _, _ = prepare_corpus(root, tmp_path / "evaluation", **kwargs)
    assert len(first["papers"]) == 2
    assert first["corpus_version"] == second["corpus_version"]
    assert encoder.embed_documents.call_count == 2  # Once per PDF, none on second import.
    assert all("evaluation_indexes" in p["database_path"] for p in first["papers"].values())
    real = retrievers_from_manifest(first, client)
    for retriever in real.values():
        result = retriever.search("mechanism", [1.0, 0.0, 0.0], hybrid=False, k=5)
        assert len(result) == 1 and result[0].index == 0


def test_save_and_reload_never_overwrites_existing_run(experiment, tmp_path):
    run1, run2 = new_run_directory(tmp_path), new_run_directory(tmp_path)
    assert run1 != run2
    report = compare_retrieval(experiment[1], experiment[0], experiment[2], experiment[3])
    save_new(report, run1 / "report.json")
    assert load_json(run1 / "report.json") == report
    with pytest.raises(EvaluationError, match="refuse_to_overwrite"):
        save_new({}, run1 / "report.json")
