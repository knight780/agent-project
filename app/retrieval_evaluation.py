"""Offline, review-gated retrieval evaluation. No model or database imports.

Corpus evidence is a frozen dataset snapshot, not a claim that the current PDF
has been reloaded. Experiment callers must bind rankings to this snapshot.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re


class EvaluationError(ValueError):
    """Malformed dataset, experiment input, or report."""


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition, code):
    if not condition:
        raise EvaluationError(code)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(value, minimum=1):
    return type(value) is int and value >= minimum


def _one_of(value, choices):
    return isinstance(value, str) and value in choices


def _status(value, field):
    result = value.get(field) if isinstance(value, dict) else None
    return result if isinstance(result, str) else "missing_or_invalid"


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EvaluationError("not_finite_json") from exc


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def load_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"),
                           object_pairs_hook=_unique_object)
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"cannot_load_json: {exc}") from exc
    _json(value)
    return value


def _validate_manifest(dataset):
    _require(isinstance(dataset, dict), "dataset_must_be_object")
    _require(type(dataset.get("schema_version")) is int and dataset["schema_version"] == 2,
             "unsupported_schema_version")
    _require(_text(dataset.get("dataset_version")), "missing_dataset_version")
    config = dataset.get("chunk_config", {})
    _require(isinstance(config, dict) and _integer(config.get("chunk_size"))
             and _integer(config.get("chunk_overlap"), 0)
             and config["chunk_overlap"] < config["chunk_size"], "invalid_chunk_config")
    papers = dataset.get("papers")
    _require(isinstance(papers, dict) and papers, "missing_paper_manifest")
    for paper_id, paper in papers.items():
        _require(isinstance(paper_id, str) and re.fullmatch(r"[0-9a-f]{64}", paper_id)
                 and isinstance(paper, dict) and paper.get("pdf_sha256") == paper_id,
                 "invalid_paper_identity")
        _require(_text(paper.get("source")) and _integer(paper.get("page_count")),
                 "invalid_paper_source_or_pages")
        chunks = paper.get("chunks")
        _require(isinstance(chunks, dict) and chunks, "missing_chunk_manifest")
        _require(set(chunks) == {f"chunk_{i}" for i in range(len(chunks))},
                 "invalid_chunk_ids")
        for chunk in chunks.values():
            _require(isinstance(chunk, dict) and _text(chunk.get("text")), "missing_chunk_text")
            _require(_integer(chunk.get("page_number"))
                     and chunk["page_number"] <= paper["page_count"], "invalid_chunk_page")
            _require(chunk.get("text_sha256") == text_sha256(chunk["text"]),
                     "chunk_text_hash_mismatch")
        texts = [chunks[f"chunk_{i}"]["text"] for i in range(len(chunks))]
        fingerprint = text_sha256(json.dumps(texts, ensure_ascii=False))
        _require(paper.get("chunk_text_sha256") == fingerprint, "chunk_version_mismatch")
    _require(isinstance(dataset.get("cases"), list), "cases_must_be_list")
    ids = [case.get("question_id") if isinstance(case, dict) else None for case in dataset["cases"]]
    _require(all(_text(qid) for qid in ids), "missing_question_id")
    _require(len(ids) == len(set(ids)), "duplicate_question_id")


def _validate_case(case, papers):
    _require(_text(case.get("question")), "missing_question")
    _require(_one_of(case.get("origin"), {"human", "llm_generated", "legacy_agent"}), "invalid_origin")
    paper_id = case.get("paper_id")
    _require(isinstance(paper_id, str) and paper_id in papers, "unknown_paper")
    paper = papers[paper_id]
    labels = case.get("labels", {})
    _require(isinstance(labels, dict), "invalid_labels")
    completeness = labels.get("completeness")
    _require(_one_of(completeness, {"anchor_only", "partial", "complete_reviewed"}),
             "invalid_label_completeness")
    judgments = labels.get("judgments")
    _require(isinstance(judgments, dict), "missing_judgments")
    _require(set(judgments) <= set(paper["chunks"]), "unknown_labeled_chunk")
    _require(all(type(grade) is int and grade in {0, 1, 2} for grade in judgments.values()),
             "invalid_relevance_grade")
    positives = {cid for cid, grade in judgments.items() if grade == 2}
    kind = case.get("answerability")
    _require(_one_of(kind, {"answerable", "unanswerable"}), "invalid_answerability")
    _require(bool(positives) if kind == "answerable" else not positives,
             "missing_positive_evidence" if kind == "answerable" else "contradictory_unanswerable_labels")
    evidence = case.get("target_evidence")
    _require(isinstance(evidence, list), "missing_target_evidence")
    evidence_ids = []
    for entry in evidence:
        _require(isinstance(entry, dict), "invalid_target_evidence")
        cid = entry.get("chunk_id")
        _require(isinstance(cid, str) and cid in paper["chunks"], "unknown_evidence_chunk")
        canonical = paper["chunks"][cid]
        _require(entry.get("source") == paper["source"], "evidence_source_mismatch")
        _require(type(entry.get("page_number")) is int
                 and entry["page_number"] == canonical["page_number"], "evidence_page_mismatch")
        _require(entry.get("text") == canonical["text"], "evidence_text_mismatch")
        evidence_ids.append(cid)
    _require(len(evidence_ids) == len(set(evidence_ids)), "duplicate_target_evidence")
    _require(set(evidence_ids) == positives, "target_evidence_label_mismatch")
    review = case.get("review", {})
    _require(isinstance(review, dict) and _one_of(review.get("status"), {"pending", "approved", "rejected"}),
             "invalid_review_status")
    _require(isinstance(case.get("review_history"), list), "missing_review_history")
    if review["status"] == "approved":
        _require(_text(review.get("reviewer")) and _text(review.get("reviewed_at")),
                 "missing_review_provenance")
        try:
            datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise EvaluationError("invalid_review_timestamp") from exc
        _require(any(isinstance(event, dict) and event.get("action") == "approve"
                     and event.get("reviewer") == review["reviewer"]
                     and event.get("timestamp") == review["reviewed_at"]
                     for event in case["review_history"]), "missing_approval_record")
    if completeness == "complete_reviewed":
        _require(review["status"] == "approved", "complete_labels_not_reviewed")
        _require(set(judgments) == set(paper["chunks"]), "incomplete_judgment_coverage")
    if review["status"] != "approved":
        return "excluded", f"review_{review['status']}"
    if kind == "unanswerable":
        return "excluded", "unanswerable_separate_evaluation"
    return ("valid", None) if completeness == "complete_reviewed" else (
        "anchor_only", "incomplete_labels_anchor_metrics_only")


def validate_dataset(dataset):
    """Manifest errors raise; case errors are returned explicitly, never dropped."""
    _json(dataset)
    _validate_manifest(dataset)
    cases = []
    for case in dataset["cases"]:
        try:
            status, reason = _validate_case(case, dataset["papers"])
        except EvaluationError as exc:
            status, reason = "invalid", str(exc)
        cases.append({"question_id": case["question_id"], "status": status, "reason": reason})
    return {"dataset_version": dataset["dataset_version"], "cases": cases}


def ranking_metrics(chunk_ids, relevant_chunk_ids, *, k=5):
    """Label-agnostic math only; official callers MUST use evaluate_dataset's gates.

    No rank compression: duplicate IDs are an error, not silently deduplicated.
    """
    _require(_integer(k), "invalid_top_k")
    _require(isinstance(chunk_ids, (list, tuple)) and all(_text(cid) for cid in chunk_ids),
             "invalid_ranked_chunk_ids")
    _require(len(chunk_ids) == len(set(chunk_ids)), "duplicate_retrieved_chunk")
    _require(isinstance(relevant_chunk_ids, (list, tuple, set, frozenset))
             and all(_text(cid) for cid in relevant_chunk_ids), "invalid_relevant_chunk_ids")
    _require(len(relevant_chunk_ids) == len(set(relevant_chunk_ids)), "duplicate_relevance_label")
    relevant = set(relevant_chunk_ids)
    _require(bool(relevant), "no_relevant_chunks")
    top = chunk_ids[:k]
    first = next((rank for rank, cid in enumerate(top, 1) if cid in relevant), None)
    return {"recall_at_k": len(set(top) & relevant) / len(relevant),
            "mrr_at_k": 1 / first if first else 0.0,
            "hit_at_k": float(first is not None), "first_relevant_rank": first}


def _validate_ranking(hits, paper_id, paper):
    _require(isinstance(hits, list), "ranking_must_be_list")
    seen = set()
    for hit in hits:
        _require(isinstance(hit, dict), "invalid_retrieval_record")
        _require(hit.get("paper_id") == paper_id, "retrieval_paper_mismatch")
        cid = hit.get("chunk_id")
        _require(isinstance(cid, str) and cid in paper["chunks"], "unknown_retrieved_chunk")
        _require(cid not in seen, "duplicate_retrieved_chunk")
        seen.add(cid)
        chunk = paper["chunks"][cid]
        _require(hit.get("source") == paper["source"], "retrieval_source_mismatch")
        _require(type(hit.get("page_number")) is int
                 and hit["page_number"] == chunk["page_number"], "retrieval_page_mismatch")
        _require(hit.get("text_sha256") == chunk["text_sha256"], "retrieval_text_hash_mismatch")
        if "text" in hit:
            _require(hit["text"] == chunk["text"], "retrieval_text_mismatch")
        for field in ("score", "distance", "cosine_distance"):
            value = hit.get(field)
            _require(value is None or (type(value) in {int, float} and math.isfinite(value)),
                     "invalid_retrieval_score")


def validate_ranking(hits, paper_id, paper):
    """Public adapter for validating real results, including exploratory runs."""
    _validate_ranking(hits, paper_id, paper)


def evaluate_dataset(dataset, rankings, *, k=5, run_metadata=None):
    """Evaluate one frozen ranking run; no retrieval, generation or cloud calls.

    rankings maps question_id to ordered evidence records. Missing output is an
    error, whereas an explicit [] is a valid empty retrieval (zero metrics).
    """
    _require(_integer(k), "invalid_top_k")
    validation = validate_dataset(dataset)
    _json(rankings)
    _require(isinstance(rankings, dict), "rankings_must_be_object")
    ids = {case["question_id"] for case in dataset["cases"]}
    _require(set(rankings) <= ids, "unknown_ranking_question_id")
    rows = []
    for case, checked in zip(dataset["cases"], validation["cases"]):
        row = {**checked, "question": case.get("question"), "paper_id": case.get("paper_id"),
               "review_status": _status(case.get("review"), "status"),
               "label_completeness": _status(case.get("labels"), "completeness"),
               "metrics": None, "warnings": [], "hits": None}
        if row["status"] in {"valid", "anchor_only"}:
            try:
                _require(case["question_id"] in rankings, "missing_retrieval_output")
                hits = rankings[case["question_id"]]
                row["hits"] = hits
                _validate_ranking(hits, case["paper_id"], dataset["papers"][case["paper_id"]])
                row["hits"] = [{**hit, "rank": i} for i, hit in enumerate(hits, 1)]
                if len(hits) < k:
                    row["warnings"].append("empty_results" if not hits else "fewer_than_k_results")
                if len(hits) > k:
                    row["warnings"].append("metrics_use_first_k_only")
                positives = [cid for cid, grade in case["labels"]["judgments"].items() if grade == 2]
                result = ranking_metrics([hit["chunk_id"] for hit in hits], positives, k=k)
                if row["status"] == "valid":
                    row["metrics"] = {key: result[key] for key in ("recall_at_k", "mrr_at_k")}
                else:
                    row["metrics"] = {"recall_at_k": None, "mrr_at_k": None,
                                      "anchor_hit_at_k": result["hit_at_k"],
                                      "anchor_mrr_at_k": result["mrr_at_k"]}
            except EvaluationError as exc:
                row.update(status="invalid", reason=str(exc))
        rows.append(row)
    counts = Counter(row["status"] for row in rows)
    means, sample_counts = {}, {}
    for metric in ("recall_at_k", "mrr_at_k", "anchor_hit_at_k", "anchor_mrr_at_k"):
        values = [row["metrics"][metric] for row in rows
                  if row["metrics"] is not None and row["metrics"].get(metric) is not None]
        sample_counts[metric] = len(values)
        means[metric] = sum(values) / len(values) if values else None
    report = {"report_schema_version": 1, "dataset_version": dataset["dataset_version"],
              "dataset_sha256": text_sha256(_json(dataset)), "top_k": k,
              "chunk_config": dataset["chunk_config"], "run_metadata": run_metadata or {},
              "corpus_versions": {pid: {key: paper[key] for key in (
                  "source", "pdf_sha256", "chunk_text_sha256")} for pid, paper in dataset["papers"].items()},
              "summary": {"total_sample_count": len(rows),
                          "valid_sample_count": counts["valid"] + counts["anchor_only"],
                          "formal_sample_count": counts["valid"], "anchor_sample_count": counts["anchor_only"],
                          "invalid_sample_count": counts["invalid"], "excluded_sample_count": counts["excluded"],
                          "review_status_counts": dict(Counter(row["review_status"] for row in rows)),
                          "label_completeness_counts": dict(Counter(row["label_completeness"] for row in rows)),
                          "reason_counts": dict(Counter(row["reason"] for row in rows if row["reason"])),
                          "metric_sample_counts": sample_counts, "mean_metrics": means},
              "cases": rows,
              "limitations": ["Evidence is checked against the frozen corpus manifest, not live PDF files.",
                              "Anchor metrics do not measure recall of all relevant evidence.",
                              "Review provenance is recorded, not independently authenticated."]}
    _json(report)
    return report


def write_report(report, path):
    """Write UTF-8 JSON atomically; reject non-finite numbers before touching output."""
    serialized = _json(report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, help="JSON question_id -> ordered evidence records")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/p3_retrieval_evaluation.json"))
    args = parser.parse_args()
    try:
        dataset = load_json(args.dataset)
        if args.rankings is None:
            checked = validate_dataset(dataset)
            print(json.dumps(checked, ensure_ascii=True, indent=2))
            return int(any(case["status"] == "invalid" for case in checked["cases"]))
        report = evaluate_dataset(dataset, load_json(args.rankings), k=args.top_k)
        write_report(report, args.output)
        print(json.dumps(report["summary"], ensure_ascii=True, indent=2))
        return int(report["summary"]["invalid_sample_count"] > 0)
    except EvaluationError as exc:
        parser.exit(2, f"evaluation error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
