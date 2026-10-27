"""Versioned PDF corpus and paired real-retrieval experiments, outside the UI."""
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from difflib import SequenceMatcher
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
from uuid import uuid4

from langchain_core.documents import Document

from app.retrieval_evaluation import (EvaluationError, evaluate_dataset,
                                      text_sha256, validate_dataset, validate_ranking, write_report)


def fingerprint(value):
    return text_sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def save_new(value, path):
    path = Path(path)
    if path.exists():
        raise EvaluationError("refuse_to_overwrite_existing_artifact")
    write_report(value, path)


def new_run_directory(output):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(output) / "runs" / f"{stamp}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def chunk_uid(paper_id, paper, chunk_id):
    return f"{paper_id}:{paper['chunk_text_sha256']}:{chunk_id}"


def scan_pdfs(root):
    """Inventory recursively, but only top-level originals are import candidates."""
    import hashlib

    root = Path(root).resolve()
    rows, seen = [], {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() != ".pdf":
            continue
        relative = path.relative_to(root)
        row = {"path": relative.as_posix(), "status": "not_selected", "paper_id": None}
        if not path.resolve().is_relative_to(root):
            row["status"] = "unsafe_path"
        elif relative.parts[0].lower() == "web":
            row["status"] = "excluded_upload_cache"
        elif len(relative.parts) > 1:
            row["status"] = "excluded_subdirectory"
        else:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                row.update(paper_id=digest, pdf_sha256=digest, size_bytes=path.stat().st_size)
                match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", path.name)
                row["arxiv_id"] = match[1] if match else None
                row["file_version"] = match[2] if match else None
                row["work_hint"] = row["arxiv_id"]
                row["version_identity"] = digest
                # Metadata is only a relationship hint, never a deduplication label.
                try:
                    import pymupdf
                    with pymupdf.open(path) as pdf:
                        row["metadata_title"] = str((pdf.metadata or {}).get("title") or "").strip()
                except Exception as exc:
                    row["metadata_error"] = type(exc).__name__
                if digest in seen:
                    row.update(status="byte_duplicate", duplicate_of=seen[digest])
                else:
                    seen[digest] = row["path"]
            except OSError as exc:
                row.update(status="read_error", error=type(exc).__name__)
        rows.append(row)
    return rows


def _implementation_versions():
    base = Path(__file__).parent / "rag"
    return {name: text_sha256((base / name).read_text(encoding="utf-8"))
            for name in ("pdf_loader.py", "splitter.py", "embedding.py", "retrieval.py")}


def embedding_identity(encoder):
    """Bind the isolated cache to loaded weights, even when HF revision is absent."""
    import hashlib

    model = getattr(encoder, "_model", None)
    identity = {"weights_sha256": None, "tokenizer_sha256": None, "max_sequence_length": None}
    if model is None:
        return identity
    state = model.state_dict()
    if not isinstance(state, Mapping):  # Injected offline test encoders need not expose a model.
        return identity
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    identity["weights_sha256"] = digest.hexdigest()
    tokenizer = getattr(getattr(model, "tokenizer", None), "backend_tokenizer", None)
    if tokenizer is not None:
        identity["tokenizer_sha256"] = text_sha256(tokenizer.to_str())
    limit = getattr(model, "max_seq_length", None)
    identity["max_sequence_length"] = limit if type(limit) is int else None
    return identity


def prepare_corpus(root, output, *, dataset=None, max_papers=3, include=None,
                   chunk_size=600, chunk_overlap=120, model_name=None, encoder=None):
    """Load/split only selected PDFs, reuse index_chunks in an isolated database."""
    import chromadb
    from chromadb.config import Settings
    from app.rag.embedding import DEFAULT_MODEL, LocalEmbeddings
    from app.rag.pdf_loader import load_pdf
    from app.rag.retrieval import index_chunks, infer_sections
    from app.rag.splitter import split_documents

    if type(max_papers) is not int or max_papers < 1:
        raise EvaluationError("invalid_paper_limit")
    if type(chunk_size) is not int or type(chunk_overlap) is not int or not 0 <= chunk_overlap < chunk_size:
        raise EvaluationError("invalid_chunk_config")
    if dataset is not None:
        validate_dataset(dataset)
        if dataset["chunk_config"] != {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}:
            raise EvaluationError("dataset_chunk_config_mismatch")
    root, output = Path(root).resolve(), Path(output).resolve()
    # index_chunks must never be pointed at a pre-existing demonstration database.
    isolated = output / "evaluation_indexes"
    for protected in (root / "chroma_real", root / "chroma_demo", root / "web"):
        if isolated.is_relative_to(protected.resolve()):
            raise EvaluationError("evaluation_database_overlaps_existing_data")
    rows = scan_pdfs(root)
    available = [row for row in rows if row["status"] == "not_selected"]
    requested = set(include or [])
    if requested - {row["path"] for row in available}:
        raise EvaluationError("requested_pdf_missing_or_duplicate")
    required = set(dataset["papers"]) if dataset else set()
    selected = [row for row in available if row["paper_id"] in required]
    if required - {row["paper_id"] for row in selected}:
        raise EvaluationError("dataset_pdf_hash_not_found")
    if requested:
        selected += [row for row in available if row["path"] in requested and row not in selected]
    else:
        selected += [row for row in available if row not in selected][:max(0, max_papers - len(selected))]
    if len(selected) > max_papers:
        raise EvaluationError("paper_limit_smaller_than_required_selection")
    if not selected:
        raise EvaluationError("no_original_pdfs")
    model_name = model_name or DEFAULT_MODEL
    encoder = encoder or LocalEmbeddings(model_name)
    if encoder.model_name != model_name:
        raise EvaluationError("embedding_model_mismatch")
    # SentenceTransformer exposes the local HF revision when available; never invent one.
    revision = None
    model = getattr(encoder, "_model", None)
    if model is not None:
        first = model._first_module()
        revision = getattr(getattr(getattr(first, "auto_model", None), "config", None), "_commit_hash", None)
        if not isinstance(revision, str):
            revision = None
    config = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    environment = {name: version(name) for name in ("chromadb", "sentence-transformers", "pypdf", "langchain-text-splitters")}
    index_config = {"embedding_model": model_name, "embedding_revision": revision,
                    "embedding_identity": embedding_identity(encoder),
                    "chunk_config": config, "implementation": _implementation_versions(), "packages": environment}
    db = isolated / fingerprint(index_config)
    client = chromadb.PersistentClient(path=str(db), settings=Settings(anonymized_telemetry=False))
    papers, text_versions = {}, {}
    for row in selected:
        stage = "load_pdf"
        try:
            path = root / row["path"]
            # Recheck bytes immediately before parsing; abort a changed selection.
            import hashlib
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["paper_id"]:
                raise EvaluationError("pdf_changed_during_import")
            documents = load_pdf(path)
            source = f"{root.name}/{row['path']}"
            for document in documents:
                document.metadata["source"] = source
            stage = "split_pdf"
            chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            texts = [chunk.page_content for chunk in chunks]
            text_version = text_sha256(json.dumps(texts, ensure_ascii=False))
            normalized_text_hash = text_sha256(" ".join(" ".join(texts).split()).casefold())
            title = str(documents[0].metadata.get("title") or row.get("metadata_title") or "").strip()
            if not row["arxiv_id"]:
                match = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", " ".join(texts[:8]))
                if match:
                    row.update(arxiv_id=match[1], file_version=match[2])
            work_hint = row["arxiv_id"] or (re.sub(r"\W+", "", title.casefold()) if len(title) > 10 else None)
            row.update(title=title, work_hint=work_hint, normalized_text_sha256=normalized_text_hash)
            if normalized_text_hash in text_versions and row["paper_id"] not in required:
                row.update(status="text_duplicate", duplicate_of=text_versions[normalized_text_hash])
                continue
            text_versions[normalized_text_hash] = row["path"]
            paper = {"pdf_sha256": row["paper_id"], "source": source, "page_count": len(documents),
                     "chunk_text_sha256": text_version, "title": title, "path": row["path"],
                     "chunks": {f"chunk_{i}": {"page_number": chunk.metadata["page_number"],
                                               "text": chunk.page_content, "text_sha256": text_sha256(chunk.page_content),
                                               "section": section}
                                for i, (chunk, section) in enumerate(zip(chunks, infer_sections(chunks)))}}
            if dataset and row["paper_id"] in dataset["papers"]:
                _assert_paper_matches(dataset["papers"][row["paper_id"]], paper)
            stage = "index_embedding"
            collection = index_chunks(client, chunks, encoder)
            paper.update(collection_name=collection.name, database_path=str(db))
            for cid, chunk in paper["chunks"].items():
                chunk["chunk_uid"] = chunk_uid(row["paper_id"], paper, cid)
            papers[row["paper_id"]] = paper
            row.update(status="imported", chunk_count=len(chunks), page_count=len(documents),
                       collection_name=collection.name)
        except Exception as exc:
            error = ("embedding_token_limit" if stage == "index_embedding" and isinstance(exc, ValueError)
                     and "token" in str(exc).casefold() else type(exc).__name__)
            row.update(status="import_error", error=str(exc) if isinstance(exc, EvaluationError) else error,
                       error_stage=stage)
    # Similar titles/identifiers are hints, not an assertion that versions are interchangeable.
    work_groups = defaultdict(list)
    for row in rows:
        if row.get("work_hint"):
            work_groups[row["work_hint"]].append(row["path"])
    for row in rows:
        row["possible_same_work_files"] = list(work_groups.get(row.get("work_hint"), []))
        title = re.sub(r"\W+", "", str(row.get("title") or row.get("metadata_title") or "").casefold())
        for other in rows:
            other_title = re.sub(r"\W+", "", str(other.get("title") or other.get("metadata_title") or "").casefold())
            if len(title) >= 15 and len(other_title) >= 15 and SequenceMatcher(None, title, other_title).ratio() >= 0.92:
                if other["path"] not in row["possible_same_work_files"]:
                    row["possible_same_work_files"].append(other["path"])
        row["work_relationship_status"] = "possible_versions_or_formats" if len(row["possible_same_work_files"]) > 1 else "unresolved_or_unique"
    manifest = {"schema_version": 2, "dataset_version": "corpus-only", "cases": [],
                "chunk_config": config, "papers": papers, "inventory": rows, "index_config": index_config,
                "retrieval_scope": "per_paper", "database_path": str(db)}
    manifest["corpus_version"] = fingerprint({"index_config": index_config, "papers": {
        pid: {"source": p["source"], "chunk_text_sha256": p["chunk_text_sha256"],
              "pages": [p["chunks"][f"chunk_{i}"]["page_number"] for i in range(len(p["chunks"]))]}
        for pid, p in papers.items()}})
    if papers:
        validate_dataset(manifest)
    return manifest, client, encoder


def _assert_paper_matches(expected, actual):
    for field in ("pdf_sha256", "source", "page_count", "chunk_text_sha256"):
        if expected[field] != actual[field]:
            raise EvaluationError(f"live_corpus_mismatch:{field}")
    if set(expected["chunks"]) != set(actual["chunks"]):
        raise EvaluationError("live_corpus_mismatch:chunk_ids")
    for cid, chunk in expected["chunks"].items():
        if any(chunk[key] != actual["chunks"][cid][key] for key in ("text", "text_sha256", "page_number")):
            raise EvaluationError("live_corpus_mismatch:chunk_evidence")


def retrievers_from_manifest(manifest, client):
    from app.rag.retrieval import PaperRetriever

    result = {}
    for pid, paper in manifest["papers"].items():
        chunks = [Document(page_content=paper["chunks"][f"chunk_{i}"]["text"], metadata={
            "source": paper["source"], "page_number": paper["chunks"][f"chunk_{i}"]["page_number"],
            "page": paper["chunks"][f"chunk_{i}"]["page_number"] - 1}) for i in range(len(paper["chunks"]))]
        collection = client.get_collection(paper["collection_name"], embedding_function=None)
        result[pid] = PaperRetriever(collection, chunks)
    return result


def compare_retrieval(dataset, manifest, retrievers, encoder, *, k=5):
    if type(k) is not int or k < 1:
        raise EvaluationError("invalid_top_k")
    checked = validate_dataset(dataset)
    validate_dataset(manifest)
    if encoder.model_name != manifest["index_config"]["embedding_model"]:
        raise EvaluationError("embedding_model_mismatch")
    if manifest["chunk_config"] != dataset["chunk_config"]:
        raise EvaluationError("dataset_chunk_config_mismatch")
    for pid, paper in dataset["papers"].items():
        if pid not in manifest["papers"]:
            raise EvaluationError("dataset_paper_not_imported")
        _assert_paper_matches(paper, manifest["papers"][pid])
    config = {"corpus_version": manifest["corpus_version"], "index_config": manifest["index_config"],
              "dataset_version": dataset["dataset_version"], "top_k": k, "retrieval_scope": "per_paper"}
    config_version = fingerprint(config)
    rankings = {"baseline": {}, "hybrid": {}}
    records = []
    for case, validation in zip(dataset["cases"], checked["cases"]):
        pid, qid = case.get("paper_id"), case["question_id"]
        review_status = case.get("review", {}).get("status") if isinstance(case.get("review"), dict) else None
        row = {"question_id": qid, "question": case.get("question"), "review_status": review_status,
               "experiment_kind": "formal" if validation["status"] == "valid" else "exploratory",
               "label_status": validation, "modes": {}, "query_vector": None, "paired_error": None,
               "execution_status": "failed"}
        if validation["status"] == "invalid" or review_status == "rejected":
            if validation["status"] == "invalid":
                row["paired_error"] = validation["reason"]
            else:
                row.update(execution_status="skipped", skipped_reason=validation["reason"])
            records.append(row)
            continue
        paper = manifest["papers"][pid]
        try:
            # Exactly one embedding per question; immutable values copied to both calls.
            vector = list(encoder.embed_query(case["question"]))
            fingerprint(vector)  # Reject non-finite values before calling the retriever.
            row.update(query_vector=vector, query_vector_sha256=fingerprint(vector))
            for mode in ("baseline", "hybrid"):
                mode_row = {"mode": mode, "hybrid": mode == "hybrid", "config_version": config_version,
                            "valid": False, "error": None, "hits": []}
                row["modes"][mode] = mode_row
                try:
                    hits = retrievers[pid].search(case["question"], list(vector), hybrid=mode == "hybrid", k=k)
                    seen = set()
                    for rank, hit in enumerate(hits, 1):
                        if type(hit.index) is not int or not 0 <= hit.index < len(paper["chunks"]):
                            raise EvaluationError("unknown_retrieved_chunk_index")
                        cid = f"chunk_{hit.index}"
                        entry = {"question_id": qid, "mode": mode, "config_version": config_version,
                                 "rank": rank, "paper_id": pid, "chunk_id": cid,
                                 "chunk_uid": chunk_uid(pid, paper, cid), "source": hit.document.metadata.get("source"),
                                 "page_number": hit.document.metadata.get("page_number"),
                                 "page": hit.document.metadata.get("page"), "text": hit.document.page_content,
                                 "text_sha256": text_sha256(hit.document.page_content), "section": hit.section,
                                 "score": hit.score, "distance": hit.distance, "neighbor_from": hit.neighbor_from,
                                 "score_type": "weighted_rrf" if mode == "hybrid" else "one_minus_cosine_distance",
                                 "valid": True, "error": None}
                        try:
                            validate_ranking([entry], pid, paper)
                            if cid in seen:
                                raise EvaluationError("duplicate_retrieved_chunk")
                            if type(entry["page"]) is not int or entry["page"] != entry["page_number"] - 1:
                                raise EvaluationError("retrieval_page_index_mismatch")
                        except EvaluationError as exc:
                            entry.update(valid=False, error=str(exc))
                            for field in ("score", "distance"):
                                value = entry[field]
                                if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
                                    entry[f"invalid_{field}_type"] = type(value).__name__
                                    entry[field] = None
                        seen.add(cid)
                        mode_row["hits"].append(entry)
                    invalid = [hit["error"] for hit in mode_row["hits"] if not hit["valid"]]
                    if invalid:
                        raise EvaluationError(invalid[0])
                    mode_row["valid"] = True
                    target_ids = {e["chunk_id"] for e in case["target_evidence"]}
                    mode_row["known_target_ranks"] = {cid: next((h["rank"] for h in mode_row["hits"]
                                                                 if h["chunk_id"] == cid), None) for cid in sorted(target_ids)}
                except Exception as exc:
                    mode_row["error"] = str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__
            if all(item["valid"] for item in row["modes"].values()):
                row["execution_status"] = "completed"
                for mode, item in row["modes"].items():
                    rankings[mode][qid] = item["hits"]
            else:
                row["paired_error"] = "paired_retrieval_failed_metrics_excluded_for_both"
        except Exception as exc:
            row["paired_error"] = str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__
        records.append(row)
    reports = {mode: evaluate_dataset(dataset, ranking, k=k, run_metadata={
        **config, "hybrid": mode == "hybrid", "live_pdf_bound": True}) for mode, ranking in rankings.items()}
    return {"report_type": "paired_live_retrieval", "config_version": config_version, "config": config,
            "dataset_version": dataset["dataset_version"], "corpus_version": manifest["corpus_version"],
            "dataset_sha256": fingerprint(dataset), "cases": records, "evaluations": reports,
            "comparison": {"baseline": reports["baseline"]["summary"], "hybrid": reports["hybrid"]["summary"],
                           "paired_success_count": sum(row["execution_status"] == "completed" for row in records),
                           "paired_error_count": sum(row["execution_status"] == "failed" for row in records),
                           "paired_skipped_count": sum(row["execution_status"] == "skipped" for row in records),
                           "formal_metric_deltas": {key: (reports["hybrid"]["summary"]["mean_metrics"][key]
                                                         - reports["baseline"]["summary"]["mean_metrics"][key])
                               if reports["baseline"]["summary"]["mean_metrics"][key] is not None
                               and reports["hybrid"]["summary"]["mean_metrics"][key] is not None else None
                               for key in ("recall_at_k", "mrr_at_k")}},
            "limitations": ["Per-paper retrieval, not a cross-paper search benchmark.",
                            "Pending questions produce exploratory rankings only, no formal metrics.",
                            "Same-work/version hints require review; different bytes are not silently merged."]}
