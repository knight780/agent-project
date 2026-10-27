"""Fixed, hand-calculated rankings only; no model, Chroma or cloud requests."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.retrieval_evaluation import (EvaluationError, evaluate_dataset, load_json, main,
                                      ranking_metrics, text_sha256, validate_dataset, write_report)


ROOT = Path(__file__).resolve().parents[1]
PAPER = "a" * 64


@pytest.fixture
def dataset():
    texts = [f"Original evidence number {i}." for i in range(6)]
    chunks = {f"chunk_{i}": {"page_number": i + 1, "text": text,
                             "text_sha256": text_sha256(text)} for i, text in enumerate(texts)}
    review = {"status": "approved", "reviewer": "offline-test-reviewer", "reviewed_at": "2026-10-26T12:00:00+08:00"}
    return {"schema_version": 2, "dataset_version": "synthetic-v1",
            "chunk_config": {"chunk_size": 600, "chunk_overlap": 120},
            "papers": {PAPER: {"pdf_sha256": PAPER, "source": "fixture.pdf", "page_count": 6,
                               "chunk_text_sha256": text_sha256(json.dumps(texts, ensure_ascii=False)),
                               "chunks": chunks}},
            "cases": [{"question_id": "q1", "question": "Which facts answer the test question?",
                       "paper_id": PAPER, "origin": "human", "answerability": "answerable",
                       "review": review,
                       "review_history": [{"action": "approve", "reviewer": review["reviewer"],
                                           "timestamp": review["reviewed_at"]}],
                       "labels": {"completeness": "complete_reviewed",
                                  "judgments": {f"chunk_{i}": 2 if i in (1, 3, 5) else 0 for i in range(6)}},
                       "target_evidence": [{"chunk_id": f"chunk_{i}", "source": "fixture.pdf",
                                            "page_number": i + 1, "text": texts[i]} for i in (1, 3, 5)]}]}


def hits(dataset, indices):
    paper = dataset["papers"][PAPER]
    return [{"paper_id": PAPER, "chunk_id": f"chunk_{i}", "source": paper["source"],
             "page_number": paper["chunks"][f"chunk_{i}"]["page_number"],
             "text_sha256": paper["chunks"][f"chunk_{i}"]["text_sha256"],
             "score": 1 / (rank + 1), "score_type": "synthetic"} for rank, i in enumerate(indices)]


def evaluate(dataset, indices, k=5):
    retriever = Mock(return_value=hits(dataset, indices))
    return evaluate_dataset(dataset, {"q1": retriever()}, k=k)


def test_hand_calculated_recall_and_reciprocal_rank(dataset):
    # Relevant = {1,3,5}; top 5 = {0,1,2,3,4}: recall 2/3, first relevant rank 2.
    report = evaluate(dataset, [0, 1, 2, 3, 4])
    assert report["cases"][0]["metrics"] == {"recall_at_k": 2 / 3, "mrr_at_k": 1 / 2}
    assert report["summary"]["formal_sample_count"] == 1
    assert report["summary"]["metric_sample_counts"]["recall_at_k"] == 1
    assert report["cases"][0]["hits"][1]["rank"] == 2
    assert report["cases"][0]["hits"][1]["score_type"] == "synthetic"


@pytest.mark.parametrize("indices,k,recall,rr,warning", [
    ([], 5, 0, 0, "empty_results"),
    ([1], 5, 1 / 3, 1, "fewer_than_k_results"),
    ([0, 2, 4], 5, 0, 0, "fewer_than_k_results"),
    ([0, 2, 1], 2, 0, 0, "metrics_use_first_k_only"),
    ([5, 3, 1], 3, 1, 1, None),
])
def test_empty_short_unrelated_and_cutoff(dataset, indices, k, recall, rr, warning):
    row = evaluate(dataset, indices, k)["cases"][0]
    assert row["status"] == "valid"
    assert row["metrics"] == {"recall_at_k": recall, "mrr_at_k": rr}
    assert row["warnings"] == ([warning] if warning else [])


def test_duplicate_results_invalidate_sample_without_rank_compression(dataset):
    report = evaluate(dataset, [0, 0, 1])
    assert report["cases"][0]["reason"] == "duplicate_retrieved_chunk"
    assert report["cases"][0]["metrics"] is None
    assert len(report["cases"][0]["hits"]) == 3
    assert report["summary"]["invalid_sample_count"] == 1
    assert report["summary"]["mean_metrics"]["recall_at_k"] is None


def test_missing_retrieval_is_not_empty_retrieval(dataset):
    row = evaluate_dataset(dataset, {})["cases"][0]
    assert row["reason"] == "missing_retrieval_output" and row["metrics"] is None


@pytest.mark.parametrize("field,value,reason", [
    ("paper_id", "b" * 64, "retrieval_paper_mismatch"),
    ("chunk_id", "chunk_99", "unknown_retrieved_chunk"),
    ("source", "other.pdf", "retrieval_source_mismatch"),
    ("page_number", 99, "retrieval_page_mismatch"),
    ("page_number", True, "retrieval_page_mismatch"),
    ("text_sha256", "b" * 64, "retrieval_text_hash_mismatch"),
    ("text", "fabricated original text", "retrieval_text_mismatch"),
    ("score", "0.99", "invalid_retrieval_score"),
])
def test_result_evidence_mismatches(dataset, field, value, reason):
    ranking = hits(dataset, [1])
    ranking[0][field] = value
    report = evaluate_dataset(dataset, {"q1": ranking})
    assert report["cases"][0]["reason"] == reason
    assert report["summary"]["valid_sample_count"] == 0


@pytest.mark.parametrize("status", ["pending", "rejected"])
def test_unreviewed_or_rejected_never_produces_metrics(dataset, status):
    case = dataset["cases"][0]
    case["origin"] = "llm_generated"
    case["review"] = {"status": status}
    case["review_history"] = []
    case["labels"]["completeness"] = "partial"
    report = evaluate(dataset, [1, 3, 5])
    assert report["cases"][0]["reason"] == f"review_{status}"
    assert report["cases"][0]["metrics"] is None
    assert report["summary"]["excluded_sample_count"] == 1
    assert all(value is None for value in report["summary"]["mean_metrics"].values())


@pytest.mark.parametrize("completeness", ["partial", "anchor_only"])
def test_reviewed_incomplete_labels_only_have_anchor_metrics(dataset, completeness):
    case = dataset["cases"][0]
    case["labels"]["completeness"] = completeness
    case["labels"]["judgments"] = {cid: 2 for cid in ("chunk_1", "chunk_3", "chunk_5")}
    report = evaluate(dataset, [0, 1, 3])
    assert report["cases"][0]["metrics"] == {
        "recall_at_k": None, "mrr_at_k": None, "anchor_hit_at_k": 1, "anchor_mrr_at_k": 0.5}
    assert report["summary"]["formal_sample_count"] == 0
    assert report["summary"]["anchor_sample_count"] == 1
    assert report["summary"]["metric_sample_counts"]["recall_at_k"] == 0


@pytest.mark.parametrize("mutation,reason", [
    (lambda c: c["labels"]["judgments"].pop("chunk_0"), "incomplete_judgment_coverage"),
    (lambda c: c["target_evidence"].pop(), "target_evidence_label_mismatch"),
    (lambda c: c["target_evidence"][0].update(page_number=6), "evidence_page_mismatch"),
    (lambda c: c["target_evidence"][0].update(source="other.pdf"), "evidence_source_mismatch"),
    (lambda c: c["target_evidence"][0].update(text="wrong"), "evidence_text_mismatch"),
    (lambda c: c["target_evidence"].append(deepcopy(c["target_evidence"][0])), "duplicate_target_evidence"),
    (lambda c: c["labels"]["judgments"].update(chunk_99=2), "unknown_labeled_chunk"),
    (lambda c: c["labels"]["judgments"].update(chunk_1=True), "invalid_relevance_grade"),
    (lambda c: c["review"].update(reviewer=None), "missing_review_provenance"),
    (lambda c: c["review"].update(reviewed_at="yesterday"), "invalid_review_timestamp"),
    (lambda c: c.update(review_history=[]), "missing_approval_record"),
    (lambda c: c["review"].update(status="pending"), "complete_labels_not_reviewed"),
    (lambda c: c.update(labels=None), "invalid_labels"),
    (lambda c: c.update(review=[]), "invalid_review_status"),
    (lambda c: c.update(origin=[]), "invalid_origin"),
    (lambda c: c["labels"].update(completeness=[]), "invalid_label_completeness"),
])
def test_invalid_labels_evidence_and_review_are_counted(dataset, mutation, reason):
    mutation(dataset["cases"][0])
    report = evaluate_dataset(dataset, {})
    assert report["cases"][0]["status"] == "invalid"
    assert report["cases"][0]["reason"] == reason
    assert report["summary"]["invalid_sample_count"] == 1


def test_no_relevant_labels_cannot_silently_become_zero_recall(dataset):
    case = dataset["cases"][0]
    case["labels"]["judgments"] = {f"chunk_{i}": 0 for i in range(6)}
    case["target_evidence"] = []
    assert evaluate(dataset, [0])["cases"][0]["reason"] == "missing_positive_evidence"
    case["answerability"] = "unanswerable"
    row = evaluate(dataset, [0])["cases"][0]
    assert row["status"] == "excluded"
    assert row["reason"] == "unanswerable_separate_evaluation" and row["metrics"] is None


def test_macro_average_and_denominators(dataset):
    q2 = deepcopy(dataset["cases"][0])
    q2["question_id"] = "q2"
    q3 = deepcopy(q2)
    q3["question_id"] = "q3"
    q3["review"]["status"] = "pending"
    q3["labels"]["completeness"] = "partial"
    dataset["cases"].extend([q2, q3])
    report = evaluate_dataset(dataset, {"q1": hits(dataset, [0, 1]), "q2": []})
    summary = report["summary"]
    assert summary["mean_metrics"]["recall_at_k"] == pytest.approx(1 / 6)
    assert summary["mean_metrics"]["mrr_at_k"] == pytest.approx(1 / 4)
    assert summary["total_sample_count"] == 3
    assert summary["valid_sample_count"] == 2
    assert summary["excluded_sample_count"] == 1
    assert summary["metric_sample_counts"]["mrr_at_k"] == 2
    assert summary["label_completeness_counts"] == {"complete_reviewed": 2, "partial": 1}


@pytest.mark.parametrize("mutation,reason", [
    (lambda d: d.update(schema_version=1), "unsupported_schema_version"),
    (lambda d: d.update(dataset_version=""), "missing_dataset_version"),
    (lambda d: d["chunk_config"].update(chunk_overlap=600), "invalid_chunk_config"),
    (lambda d: d["cases"].append(deepcopy(d["cases"][0])), "duplicate_question_id"),
    (lambda d: d["papers"][PAPER]["chunks"]["chunk_0"].update(text="changed"), "chunk_text_hash_mismatch"),
    (lambda d: d["papers"][PAPER].update(chunk_text_sha256="bad"), "chunk_version_mismatch"),
])
def test_invalid_manifests_fail_explicitly(dataset, mutation, reason):
    mutation(dataset)
    with pytest.raises(EvaluationError, match=reason):
        validate_dataset(dataset)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(dataset, k):
    with pytest.raises(EvaluationError, match="invalid_top_k"):
        evaluate_dataset(dataset, {}, k=k)


def test_math_primitive_rejects_empty_truth_and_duplicates():
    with pytest.raises(EvaluationError, match="no_relevant_chunks"):
        ranking_metrics([], [])
    with pytest.raises(EvaluationError, match="duplicate_retrieved_chunk"):
        ranking_metrics(["a", "a"], ["a"])
    with pytest.raises(EvaluationError, match="duplicate_relevance_label"):
        ranking_metrics(["a"], ["a", "a"])


def test_unknown_question_and_nonfinite_score_fail(dataset):
    with pytest.raises(EvaluationError, match="unknown_ranking_question_id"):
        evaluate_dataset(dataset, {"typo": []})
    ranking = hits(dataset, [1])
    ranking[0]["score"] = float("nan")
    with pytest.raises(EvaluationError, match="not_finite_json"):
        evaluate_dataset(dataset, {"q1": ranking})


def test_empty_dataset_means_are_null_not_fabricated_zero(dataset):
    dataset["cases"] = []
    summary = evaluate_dataset(dataset, {})["summary"]
    assert summary["total_sample_count"] == 0
    assert all(value is None for value in summary["mean_metrics"].values())


def test_json_report_round_trip_preserves_dataset_and_history(dataset, tmp_path):
    before = deepcopy(dataset)
    report = evaluate_dataset(dataset, {"q1": hits(dataset, [1])}, run_metadata={"mode": "baseline"})
    path = tmp_path / "reports" / "report.json"
    write_report(report, path)
    assert load_json(path) == report
    assert dataset == before
    assert report["dataset_version"] == "synthetic-v1"
    assert report["run_metadata"] == {"mode": "baseline"}
    assert not path.with_name(path.name + ".tmp").exists()


def test_duplicate_json_keys_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"question_id":"one","question_id":"two"}', encoding="utf-8")
    with pytest.raises(EvaluationError, match="duplicate_json_key"):
        load_json(path)


def test_legacy_annotations_preserved_and_not_promoted():
    dataset = load_json(ROOT / "tests/fixtures/retrieval_gold.json")
    original = {key: dataset[key] for key in (
        "pdf_sha256", "chunk_count", "rubric", "scope", "chunk_text_sha256")}
    original["cases"] = [{key: case[key] for key in ("question", "annotation", "grades")}
                         for case in dataset["cases"]]
    assert text_sha256(json.dumps(original, sort_keys=True, ensure_ascii=False)) == (
        "026b620896fd97df15da3cd418b9877b60b7624698c06bcde475c7f0f9291f12")
    assert "Agent-authored" in dataset["rubric"]
    assert len(dataset["cases"]) == 5
    assert [len(c["grades"]) for c in dataset["cases"]] == [32, 14, 6, 7, 21]
    for case in dataset["cases"]:
        assert case["labels"]["judgments"] == {f"chunk_{cid}": grade for cid, grade in case["grades"].items()}
        assert case["review_history"] == []
    report = evaluate_dataset(dataset, {})
    assert report["summary"]["excluded_sample_count"] == 5
    assert report["summary"]["invalid_sample_count"] == 0
    assert report["summary"]["formal_sample_count"] == 0
    assert report["summary"]["mean_metrics"]["recall_at_k"] is None


def test_cli_validates_and_writes_reports_without_loading_models(dataset, tmp_path, monkeypatch, capsys):
    dataset_path, rankings_path, output = (tmp_path / name for name in ("data.json", "rankings.json", "out.json"))
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    rankings_path.write_text(json.dumps({"q1": hits(dataset, [1])}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["evaluation", "--dataset", str(dataset_path)])
    assert main() == 0
    monkeypatch.setattr("sys.argv", ["evaluation", "--dataset", str(dataset_path),
                                    "--rankings", str(rankings_path), "--output", str(output)])
    assert main() == 0
    assert load_json(output)["summary"]["formal_sample_count"] == 1
    rankings_path.write_text("{}", encoding="utf-8")
    assert main() == 1  # Report is still written, with the missing-output reason.
    assert load_json(output)["cases"][0]["reason"] == "missing_retrieval_output"
    capsys.readouterr()
