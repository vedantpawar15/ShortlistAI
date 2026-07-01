"""CLI entrypoint for RecruitAI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recruit_ai.api import RecruitAIAPI
from recruit_ai.data.sample_data import sample_candidates, sample_job
from recruit_ai.pipeline import RecruitAIRanker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank candidates for an AI recruiting workflow.")
    parser.add_argument("--serve-api", action="store_true", help="Start the local WSGI API server.")
    parser.add_argument("--host", default="127.0.0.1", help="API bind host.")
    parser.add_argument("--port", default=8000, type=int, help="API bind port.")
    parser.add_argument("--json", action="store_true", help="Render output as JSON.")
    return parser


def run_demo(as_json: bool = False) -> int:
    ranker = RecruitAIRanker()
    results = ranker.rank(sample_job(), sample_candidates())
    if as_json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return 0

    print("RecruitAI candidate ranking demo")
    print("=" * 32)
    for index, result in enumerate(results, start=1):
        candidate = result.candidate
        print(f"{index}. {candidate.name} | score={result.total_score:.4f}")
        print(f"   Headline: {candidate.headline}")
        print(f"   Required skill match: {', '.join(result.matched_required_skills) or 'None'}")
        print(f"   Gaps: {', '.join(result.missing_required_skills) or 'None'}")
        print(f"   Why: {result.explanation[0]}")
    return 0


def run_api(host: str, port: int) -> int:
    from wsgiref.simple_server import make_server

    app = RecruitAIAPI()
    print(f"RecruitAI API listening on http://{host}:{port}")
    with make_server(host, port, app) as server:
        server.serve_forever()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.serve_api:
        return run_api(args.host, args.port)
    return run_demo(as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
