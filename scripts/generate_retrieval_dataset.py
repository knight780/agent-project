"""Preview or explicitly execute candidate generation from an existing corpus snapshot."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from app.retrieval_candidates import candidate_plan, generate_candidates
from app.retrieval_evaluation import EvaluationError, load_json
from app.retrieval_experiment import save_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-count", type=int, default=3)
    parser.add_argument("--per-paper", type=int, choices=[3, 4, 5], default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-requests", type=int, default=40)
    parser.add_argument("--attempts-per-stage", type=int, default=2)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true", help="Explicitly enable paid DeepSeek API calls")
    args = parser.parse_args()
    try:
        load_dotenv(ROOT / ".env")
        corpus = load_json(args.corpus)
        plan = candidate_plan(corpus, paper_count=args.paper_count, per_paper=args.per_paper,
                              seed=args.seed, model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
        output = args.output_dir / "candidates.json"
        result = generate_candidates(corpus, plan, output, execute=args.execute,
                                     max_requests=args.max_requests, attempts_per_stage=args.attempts_per_stage,
                                     interval=args.interval)
        if not args.execute:
            path = args.output_dir / f"plan-{plan['plan_id'][:16]}.json"
            if not path.exists():
                save_new(result, path)
            print(json.dumps({"execute": False, "plan": str(path),
                              "planned_questions": len(plan["tasks"]), "normal_requests": 2 * len(plan["tasks"])}, indent=2))
        else:
            print(json.dumps({"candidate_dataset": str(output), "pending_review_count": len(result["cases"]),
                              "task_statuses": {key: task["status"] for key, task in result["tasks"].items()}}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
