"""Pipeline runner.

Executes the extraction pipeline over the configured corpus and writes the run
summary, the evaluation-ready JSON, and the reviewer dashboard.

    legiswatch-run                                   # recorded fixtures, no network
    legiswatch-run --provider anthropic              # live inference
    legiswatch-run --provider anthropic --record     # live, refresh fixtures
    legiswatch-run --provider ollama --model llama3.1:8b   # fully local
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .audit import AuditJournal
from .corpus import corpus_metadata, load_corpus
from .dashboard import render_dashboard
from .graph import run_pipeline
from .llm import LLMClient
from .logging_setup import configure
from .schemas import RouteDecision

OUT = Path(os.getenv("LEGISWATCH_OUT_DIR", Path.cwd() / "out"))


def main() -> int:
    p = argparse.ArgumentParser(
        prog="legiswatch-run", description="Run the LegisWatch extraction pipeline."
    )
    p.add_argument(
        "--provider",
        default="replay",
        choices=["replay", "anthropic", "openai", "ollama"],
        help="replay (default) serves recorded fixtures; no key or network required.",
    )
    p.add_argument("--model", default=None)
    p.add_argument(
        "--record", action="store_true", help="Persist live responses as replay fixtures."
    )
    p.add_argument(
        "--no-negatives", action="store_true", help="Exclude negative-control documents."
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    configure(args.log_level)
    OUT.mkdir(exist_ok=True)

    docs = load_corpus(include_negative_controls=not args.no_negatives)
    meta = corpus_metadata()

    client = LLMClient(provider=args.provider, model=args.model, record=args.record)

    print(
        f"\n  LegisWatch  ·  {len(docs)} documents  ·  provider={client.provider}  model={client.model}\n"
    )

    summary = run_pipeline(docs, client)

    # --- artifacts --------------------------------------------------------
    run_json = OUT / "run_summary.json"
    run_json.write_text(json.dumps(summary.model_dump(mode="json"), indent=2))

    # The journal is passed in so each card renders its full decision history
    # and current provenance. Absent a journal the page still renders; it just
    # shows the agent's decisions with no human layer.
    dashboard = OUT / "dashboard.html"
    render_dashboard(summary, meta, dashboard, journal=AuditJournal())

    # --- console report ---------------------------------------------------
    print("\n" + "=" * 74)
    print(f"  RUN {summary.run_id}")
    print("=" * 74)
    print(f"  Documents ingested        {summary.documents_ingested}")
    print(f"    relevant                {summary.documents_relevant}")
    print(f"    filtered out by triage  {summary.documents_ingested - summary.documents_relevant}")
    print(f"  Obligations extracted     {summary.obligations_extracted}")
    print(f"    auto-filed              {summary.obligations_auto_filed}")
    print(f"    queued for human review {summary.obligations_needing_review}")
    print(f"    rejected by guardrails  {summary.obligations_rejected}")
    print(f"  Wall clock                {summary.total_latency_ms / 1000:.2f}s")
    print(f"  LLM calls                 {client.call_count}")
    if client.provider != "replay":
        print(f"  Estimated cost            ${summary.estimated_cost_usd:.4f}")
    print("=" * 74)

    rejected = [
        r
        for res in summary.results
        for r in res.routed_obligations
        if r.route == RouteDecision.REJECTED
    ]
    if rejected:
        print("\n  GUARDRAILS FIRED\n")
        for r in rejected:
            print(f"  ✗ {r.obligation.obligation_id}")
            print(f"      {r.route_reason}")
            print(
                f"      exact_match={r.groundedness.exact_match} "
                f"fuzzy={r.groundedness.fuzzy_score:.1f}\n"
            )

    print(f"  Dashboard  ->  {dashboard}")
    print(f"  Run JSON   ->  {run_json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
