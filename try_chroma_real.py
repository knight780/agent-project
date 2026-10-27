"""Compare dense and hybrid Top-5 retrieval using cached local embeddings."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import chromadb
from chromadb.config import Settings

from app.rag.pdf_loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.embedding import LocalEmbeddings
from app.rag.retrieval import PaperRetriever, index_chunks


ROOT = Path(__file__).resolve().parent


def metrics(indices: list[int], grades: dict[int, int], k=5):
    """Direct evidence has grade 2; related context has grade 1; others 0."""
    values = [grades.get(i, 0) for i in indices[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k]
    dcg = lambda scores: sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(scores))
    first = next((rank for rank, grade in enumerate(values, 1) if grade == 2), None)
    return {
        "direct_precision_at_5": sum(grade == 2 for grade in values) / k,
        "related_precision_at_5": sum(grade > 0 for grade in values) / k,
        "first_direct_rank": first,
        "ndcg_at_5": dcg(values) / dcg(ideal) if dcg(ideal) else 0.0,
    }


def serialize(hits):
    return [
        {"chunk_index": hit.index, "page_number": hit.document.metadata["page_number"],
         "source": hit.document.metadata["source"], "section": hit.section,
         "text": hit.document.page_content, "score": hit.score,
         "cosine_distance": hit.distance, "neighbor_from": hit.neighbor_from}
        for hit in hits
    ]


def print_hits(label, hits, retriever, show_continuation=False):
    print(f"\n{label}")
    for rank, hit in enumerate(hits, 1):
        print(f"Top-{rank}: chunk_{hit.index}, 页码 {hit.document.metadata['page_number']}, 章节 {hit.section}")
        if label == "纯向量基线":
            print(f"余弦距离: {hit.distance:.4f}（越小越近）")
        else:
            print(f"混合排序分: {hit.score:.6f}（越大越优先，不是准确率）")
        print("source:", hit.document.metadata["source"])
        if hit.neighbor_from is not None:
            print(f"邻近补召回，锚点: chunk_{hit.neighbor_from}")
        print(hit.document.page_content)
        continuation = retriever.continuation(hit) if show_continuation else None
        if continuation:
            print(f"[续接上下文，页码 {continuation.metadata['page_number']}；不计入 TOP5 命中]")
            print(continuation.page_content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="What is the attention mechanism?")
    parser.add_argument("--mode", choices=["baseline", "hybrid", "compare"], default="compare")
    parser.add_argument("--evaluate", action="store_true", help="Run the annotated sample-paper regression set")
    parser.add_argument("--db", type=Path, default=ROOT / "data/chroma_real")
    parser.add_argument("--report", type=Path, default=ROOT / "data/retrieval_evaluation.json")
    args = parser.parse_args()

    pdf = ROOT / "data/sample.pdf"
    chunks = split_documents(load_pdf(pdf), chunk_size=600, chunk_overlap=120)
    encoder = LocalEmbeddings()
    client = chromadb.PersistentClient(path=str(args.db), settings=Settings(anonymized_telemetry=False))
    collection = index_chunks(client, chunks, encoder)
    retriever = PaperRetriever(collection, chunks)
    print("Collection:", collection.name, "记录数:", collection.count())

    if args.evaluate:
        gold = json.loads((ROOT / "tests/fixtures/retrieval_gold.json").read_text(encoding="utf-8"))
        text_hash = hashlib.sha256(json.dumps([chunk.page_content for chunk in chunks], ensure_ascii=False).encode("utf-8")).hexdigest()
        if (hashlib.sha256(pdf.read_bytes()).hexdigest() != gold["pdf_sha256"]
                or len(chunks) != gold["chunk_count"] or text_hash != gold["chunk_text_sha256"]):
            raise ValueError("论文或切分配置已变化，请重新人工标注评测集。")
        report = {"rubric": gold["rubric"], "scope": gold["scope"],
                  "model": encoder.model_name, "collection": collection.name,
                  "pdf_sha256": gold["pdf_sha256"], "chunk_text_sha256": text_hash,
                  "chunk_size": 600, "chunk_overlap": 120, "chunk_count": len(chunks), "cases": []}
        for case in gold["cases"]:
            query_vector = encoder.embed_query(case["question"])
            grades = {int(index): grade for index, grade in case["grades"].items()}
            entry = {"question": case["question"], "annotation": case["annotation"]}
            for mode in ("baseline", "hybrid"):
                hits = retriever.search(case["question"], query_vector, hybrid=mode == "hybrid")
                entry[mode] = {"metrics": metrics([hit.index for hit in hits], grades), "hits": serialize(hits)}
            report["cases"].append(entry)
            print(json.dumps({"question": case["question"], **{mode: entry[mode]["metrics"] for mode in ("baseline", "hybrid")}}, ensure_ascii=False))
        report["mean_metrics"] = {
            mode: {key: sum(case[mode]["metrics"][key] for case in report["cases"]) / len(report["cases"])
                   for key in ("direct_precision_at_5", "related_precision_at_5", "ndcg_at_5")}
            for mode in ("baseline", "hybrid")
        }
        print("开发集均值:", json.dumps(report["mean_metrics"], ensure_ascii=False))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("评测原文与指标:", args.report)
        return

    vector = encoder.embed_query(args.question)
    for mode, label in (("baseline", "纯向量基线"), ("hybrid", "混合检索")):
        if args.mode in (mode, "compare"):
            hits = retriever.search(args.question, vector, hybrid=mode == "hybrid")
            print_hits(label, hits, retriever, show_continuation=mode == "hybrid")


if __name__ == "__main__":
    main()
