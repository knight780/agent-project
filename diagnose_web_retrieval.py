"""Offline bilingual dense retrieval diagnostic; never calls a cloud API."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from threading import RLock
from unittest.mock import patch

from app.rag.embedding import LocalEmbeddings
from app.web_service import answer_question, prepare_paper
from app.query_translation import retrieval_query

ROOT = Path(__file__).resolve().parent
QUESTIONS = [
    "Why did the authors choose sinusoidal positional encoding?",
    "论文为什么选择正弦位置编码？",
    "What results did Transformer achieve in machine translation?",
    "Transformer 在机器翻译任务中取得了什么结果？",
]
TRANSLATION_FIXTURE = {QUESTIONS[1]: QUESTIONS[0], QUESTIONS[3]: QUESTIONS[2]}


def run_diagnostic(paper, encoder, retrieval_queries=None):
    cases = []
    for index, question in enumerate(QUESTIONS):
        query = retrieval_queries[index] if retrieval_queries else question
        hits = paper.retriever.search(query, encoder.embed_query(query), hybrid=False, k=5)
        cases.append({"question": question, "retrieval_query": query, "hits": [
            {"chunk_id": f"chunk_{hit.index}", "page": hit.document.metadata["page_number"],
             "distance": hit.distance, "summary": " ".join(hit.document.page_content.split())[:300],
             "text": hit.document.page_content} for hit in hits]})
    return cases


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/web_v1_baseline.json")
    parser.add_argument("--compare", action="store_true",
                        help="Compare using explicitly mocked English translations and answers, no API")
    args = parser.parse_args()
    content = (ROOT / "data/sample.pdf").read_bytes()
    encoder = LocalEmbeddings()
    paper = prepare_paper(content, ROOT / "data/web", lambda: (encoder, RLock()))
    report = {"pdf_sha256": hashlib.sha256(content).hexdigest(), "model": encoder.model_name,
              "chunk_size": 600, "chunk_overlap": 120, "chunk_count": paper.chunk_count,
              "generation": "not run; no cloud calls", "baseline": run_diagnostic(paper, encoder)}
    if args.compare:
        queries = [retrieval_query(question, True, TRANSLATION_FIXTURE.__getitem__)
                   for question in QUESTIONS]
        report["translated"] = run_diagnostic(paper, encoder, queries)
        report["generation"] = "MOCK ONLY: fixture translations; evidence-based canned answers; not real DeepSeek quality"
        for mode in ("baseline", "translated"):
            for case in report[mode]:
                # Deliberately simple fixture: test the real orchestration/citation gate,
                # not whether a language model can infer semantic support.
                if case["question"] in QUESTIONS[:2]:
                    relevant = [(i, h) for i, h in enumerate(case["hits"], 1)
                                if "extrapolate" in h["text"] or "We chose this function" in " ".join(h["text"].split())]
                    mock_answer = ("作者选择正弦位置编码以支持相对位置学习，并可能外推到更长序列。"
                                   + "".join(f"[S{i}]" for i, _ in relevant)) if relevant else "论文证据不足，无法确认。"
                else:
                    relevant = [(i, h) for i, h in enumerate(case["hits"], 1)
                                if "BLEU score of 41.0" in h["text"]]
                    mock_answer = ("英法翻译取得 41.0 BLEU。"
                                   + "".join(f"[S{i}]" for i, _ in relevant)) if relevant else "论文证据不足，无法确认。"
                with patch("app.query_translation.translate_query_to_english", side_effect=TRANSLATION_FIXTURE.__getitem__), \
                     patch("app.web_service.call_deepseek", return_value=mock_answer):
                    result = answer_question(paper, case["question"], encoder, "sample.pdf",
                                             translate_chinese=mode == "translated")
                case["mock_generation"] = {"status": result.status, "answer": result.answer,
                                           "sources": result.sources}
        sample_answer = "作者选择正弦位置编码以支持相对位置学习。[S1] 论文未提供实现代码。"
        old_rule = r"无法.{0,16}(?:确认|回答|确定)|(?:证据|信息)不足|未(?:提及|提供)|没有.{0,8}(?:提及|相关信息)"
        with patch("app.web_service.call_deepseek", return_value=sample_answer):
            result = answer_question(paper, QUESTIONS[0], encoder, "sample.pdf")
        report["refusal_gate_reproduction"] = {
            "scope": "MOCK response on actual English positional-encoding retrieval",
            "response": sample_answer, "old_rule_rejects": bool(re.search(old_rule, sample_answer)),
            "fixed_status": result.status, "sources": result.sources}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode in ("baseline", "translated"):
        for case in report.get(mode, []):
            print(f"{mode}: {case['question']}\n  retrieval_query={case['retrieval_query']}")
            for hit in case["hits"]:
                print(f"  {hit['chunk_id']} page={hit['page']} distance={hit['distance']:.6f} {hit['summary']}")
            if "mock_generation" in case:
                print("  MOCK:", case["mock_generation"])


if __name__ == "__main__":
    main()
