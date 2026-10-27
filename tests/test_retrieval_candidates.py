"""Candidate preview, bounded mocked generation, evidence checks and resume."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.rag.retrieval import Hit
from app.retrieval_candidates import candidate_plan, generate_candidates
from app.retrieval_evaluation import EvaluationError, evaluate_dataset, load_json
from app.retrieval_experiment import compare_retrieval
from test_retrieval_experiment import experiment


def candidate_caller(prompt, **kwargs):
    if prompt.startswith("Check"):
        return json.dumps({"supported": True, "unambiguous": True, "single_hop": True, "reason": "Synthetic support"})
    data = json.loads(prompt.split("\n", 1)[1])
    return json.dumps({"question": f"Which fact does evidence {data['task_id'][:24]} establish?",
                       "candidate_answer": data["evidence"][:50], "support_quote": data["evidence"][:50],
                       "question_type": data["question_type"], "chunk_id": data["chunk_id"],
                       "page_number": data["page_number"]})


def test_preview_is_deterministic_diverse_and_never_calls_api(experiment, tmp_path):
    corpus = experiment[0]
    plan = candidate_plan(corpus)
    assert plan == candidate_plan(corpus)
    reordered = deepcopy(corpus)
    for paper in reordered["papers"].values():
        paper["chunks"] = dict(reversed(list(paper["chunks"].items())))
    assert plan == candidate_plan(reordered)
    assert len(plan["tasks"]) == 9
    for pid in corpus["papers"]:
        tasks = [task for task in plan["tasks"] if task["paper_id"] == pid]
        assert len(tasks) == 3
        assert len({task["page_number"] for task in tasks}) == 3
        assert {task["question_type"] for task in tasks} == {"fact", "concept", "method"}
        indices = sorted(int(task["chunk_id"].split("_")[1]) for task in tasks)
        assert all(b - a > 1 for a, b in zip(indices, indices[1:]))
    caller = Mock(side_effect=AssertionError("No API in preview"))
    output = tmp_path / "candidates.json"
    preview = generate_candidates(corpus, plan, output, caller=caller)
    assert preview["planned_questions"] == 9 and not preview["execute"]
    assert not output.exists()
    caller.assert_not_called()


def test_generated_candidates_remain_pending_with_original_raw_history(experiment, tmp_path):
    corpus = experiment[0]
    before = deepcopy(experiment[1])
    caller = Mock(side_effect=candidate_caller)
    output = tmp_path / "candidates.json"
    state = generate_candidates(corpus, candidate_plan(corpus), output, execute=True, caller=caller, sleeper=lambda _: None)
    assert len(state["cases"]) == 9 and caller.call_count == 18
    assert load_json(output) == state
    assert experiment[1] == before
    assert all(case["candidate_status"] == "pending_review" and case["review"]["status"] == "pending" for case in state["cases"])
    assert all(len(task["events"]) == 2 and task["events"][0]["raw"] for task in state["tasks"].values())
    formal = evaluate_dataset(state, {})
    assert formal["summary"]["formal_sample_count"] == 0
    assert formal["summary"]["excluded_sample_count"] == 9
    assert formal["summary"]["mean_metrics"]["mrr_at_k"] is None
    again = generate_candidates(corpus, candidate_plan(corpus), output, execute=True, caller=caller, sleeper=lambda _: None)
    assert again == state and caller.call_count == 18


def test_budget_and_resume_preserve_completed_candidates_without_reembedding(experiment, tmp_path):
    corpus, _, _, encoder = experiment
    plan = candidate_plan(corpus)
    caller = Mock(side_effect=candidate_caller)
    output = tmp_path / "candidates.json"
    partial = generate_candidates(corpus, plan, output, execute=True, caller=caller, max_requests=3, sleeper=lambda _: None)
    assert len(partial["cases"]) == 1 and caller.call_count == 3
    first = deepcopy(partial["cases"][0])
    complete = generate_candidates(corpus, plan, output, execute=True, caller=caller, max_requests=40, sleeper=lambda _: None)
    assert len(complete["cases"]) == 9 and caller.call_count == 18
    assert complete["cases"][0] == first
    encoder.embed_documents.assert_not_called()
    encoder.embed_query.assert_not_called()


def test_api_failure_is_bounded_redacted_and_does_not_change_formal_dataset(experiment, tmp_path):
    corpus, formal, _, _ = experiment
    before = deepcopy(formal)
    caller = Mock(side_effect=RuntimeError("secret-api-key private-path"))
    state = generate_candidates(corpus, candidate_plan(corpus), tmp_path / "candidates.json",
                                execute=True, caller=caller, max_requests=4, sleeper=lambda _: None)
    assert caller.call_count == 4 and not state["cases"]
    assert formal == before
    assert "secret-api-key" not in json.dumps(state)
    assert any(t["status"] == "generation_failed" for t in state["tasks"].values())


@pytest.mark.parametrize("fault", ["invalid_json", "wrong_page", "wrong_chunk", "missing_answer", "unsupported_quote", "unsupported_answer"])
def test_invalid_generated_evidence_is_isolated(experiment, tmp_path, fault):
    corpus = experiment[0]
    plan = candidate_plan(corpus, paper_count=1)

    def caller(prompt, **kwargs):
        if fault == "invalid_json":
            return "not JSON"
        result = json.loads(candidate_caller(prompt, **kwargs))
        if prompt.startswith("Check"):
            if fault == "unsupported_answer":
                result["supported"] = False
        elif fault == "wrong_page":
            result["page_number"] = 999
        elif fault == "wrong_chunk":
            result["chunk_id"] = "chunk_999"
        elif fault == "missing_answer":
            del result["candidate_answer"]
        elif fault == "unsupported_quote":
            result["support_quote"] = "Evidence that does not exist"
        return json.dumps(result)

    state = generate_candidates(corpus, plan, tmp_path / "candidates.json", execute=True, caller=caller, sleeper=lambda _: None)
    assert not state["cases"]
    assert all(task["status"] in {"generation_rejected", "validation_rejected"} for task in state["tasks"].values())
    assert all(task["events"][0]["raw"] for task in state["tasks"].values())


def test_duplicate_questions_are_flagged_not_counted_twice(experiment, tmp_path):
    corpus = experiment[0]

    def caller(prompt, **kwargs):
        result = json.loads(candidate_caller(prompt, **kwargs))
        if not prompt.startswith("Check"):
            result["question"] = "What is the same duplicated question?"
        return json.dumps(result)

    state = generate_candidates(corpus, candidate_plan(corpus, paper_count=1), tmp_path / "candidates.json",
                                execute=True, caller=caller, sleeper=lambda _: None)
    assert len(state["cases"]) == 1
    assert sum(task.get("error") == "duplicate_or_near_duplicate_question" for task in state["tasks"].values()) == 2


def test_existing_formal_dataset_cannot_be_overwritten(experiment, tmp_path):
    corpus, dataset, _, _ = experiment
    output = tmp_path / "formal.json"
    output.write_text(json.dumps(dataset), encoding="utf-8")
    caller = Mock()
    with pytest.raises(EvaluationError, match="refuse_to_overwrite_unrelated_dataset"):
        generate_candidates(corpus, candidate_plan(corpus), output, execute=True, caller=caller)
    assert load_json(output) == dataset
    caller.assert_not_called()


def test_plan_cannot_be_used_with_changed_corpus(experiment, tmp_path):
    corpus = experiment[0]
    plan = candidate_plan(corpus)
    corpus["corpus_version"] = "changed"
    with pytest.raises(EvaluationError, match="candidate_corpus_version_mismatch"):
        generate_candidates(corpus, plan, tmp_path / "out.json")


def test_candidate_dataset_feeds_paired_experiment_without_formal_promotion(experiment, tmp_path):
    corpus, _, _, encoder = experiment
    state = generate_candidates(corpus, candidate_plan(corpus), tmp_path / "candidates.json",
                                execute=True, caller=candidate_caller, sleeper=lambda _: None)
    retrievers = {}
    for pid, paper in corpus["papers"].items():
        chunk = paper["chunks"]["chunk_0"]
        doc = Document(page_content=chunk["text"], metadata={"source": paper["source"],
                       "page_number": chunk["page_number"], "page": chunk["page_number"] - 1})
        retriever = Mock()
        retriever.search.return_value = [Hit(0, doc, "Section", 0.5, 0.5)]
        retrievers[pid] = retriever
    report = compare_retrieval(state, corpus, retrievers, encoder)
    assert report["comparison"]["paired_success_count"] == 9
    assert all(c["experiment_kind"] == "exploratory" for c in report["cases"])
    assert report["evaluations"]["hybrid"]["summary"]["formal_sample_count"] == 0
    assert report["evaluations"]["baseline"]["summary"]["excluded_sample_count"] == 9
