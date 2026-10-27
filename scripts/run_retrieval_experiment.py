"""Build an isolated evaluation corpus and run a paired dense/hybrid experiment."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval_evaluation import EvaluationError, load_json
from app.retrieval_experiment import (compare_retrieval, new_run_directory, prepare_corpus,
                                      retrievers_from_manifest, save_new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--dataset", type=Path, default=ROOT / "tests/fixtures/retrieval_gold.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/p3_phase2")
    parser.add_argument("--max-papers", type=int, default=3)
    parser.add_argument("--all-papers", action="store_true")
    parser.add_argument("--corpus-only", action="store_true", help="Import a new corpus version without reusing legacy question labels")
    parser.add_argument("--include", nargs="*")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run = new_run_directory(args.output_dir)
    try:
        dataset = None if args.corpus_only else load_json(args.dataset)
        corpus, client, encoder = prepare_corpus(
            args.pdf_dir, args.output_dir, dataset=dataset,
            max_papers=1000000 if args.all_papers else args.max_papers, include=args.include,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, model_name=args.model)
        save_new(corpus, run / "corpus.json")
        if dataset is None:
            dataset = {key: corpus[key] for key in ("schema_version", "dataset_version", "cases", "papers", "chunk_config")}
        report = compare_retrieval(dataset, corpus, retrievers_from_manifest(corpus, client), encoder, k=args.top_k)
        save_new(report, run / "report.json")
        summary = {"run_directory": str(run), "imported_papers": len(corpus["papers"]),
                   "inventory": [{"path": r["path"], "status": r["status"]} for r in corpus["inventory"]],
                   "comparison": report["comparison"]}
        save_new(summary, run / "summary.json")
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return int(report["comparison"]["paired_error_count"] > 0 or
                   any(r["status"] == "import_error" for r in corpus["inventory"]))
    except Exception as exc:
        error = str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__
        save_new({"error": error, "run_directory": str(run)}, run / "failure.json")
        print(json.dumps({"error": error, "run_directory": str(run)}, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
