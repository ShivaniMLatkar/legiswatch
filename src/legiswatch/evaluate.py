"""Evaluation harness.

Scores pipeline output against a hand-labelled gold set built independently of
the output being measured (`data/gold/gold_set.json`).

Four things get measured:

  1. **Extraction quality** -- precision / recall / F1 against the gold set,
     measured twice: on raw model output, and on what survives the guardrails.
     The gap between those two numbers *is* the value of the guardrail layer.
  2. **Groundedness** -- what fraction of quotes actually appear in the source,
     and what fraction of ungrounded ones the deterministic check caught.
  3. **Attribution accuracy** -- of correctly extracted obligations, how many
     were routed to the right office. A right duty on the wrong desk is still a
     miss operationally.
  4. **Routing behaviour** -- did anything get auto-filed that the gold set says
     is wrong? This is the number that decides whether auto-filing is safe to
     turn on at all, and it is the one to lead with in a governance conversation.

Run:  legiswatch-eval [--provider replay]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rapidfuzz import fuzz

from .corpus import load_corpus
from .graph import run_pipeline
from .llm import LLMClient
from .logging_setup import configure
from .paths import DATA_DIR
from .schemas import IUUnit, RouteDecision

GOLD_PATH = DATA_DIR / "gold" / "gold_set.json"
OUT = Path(os.getenv("LEGISWATCH_OUT_DIR", Path.cwd() / "out"))
SUMMARY_MATCH_THRESHOLD = 55.0


def normalize_citation(c: str) -> str:
    return "".join(ch for ch in c.lower() if ch.isalnum())


def match_prediction(pred, gold_entries: list[dict], used: set) -> dict | None:
    """Greedy best-match of one prediction to an unused gold entry.

    Citation must align (one containing the other after normalisation, so that a
    prediction citing the section matches a gold entry citing the subsection),
    and summaries must clear a token-set similarity floor. Deliberately lenient
    on wording and strict on citation: two obligations from the same subsection
    are the ones worth distinguishing.
    """
    pc = normalize_citation(pred.obligation.citation)
    best, best_score = None, 0.0

    for g in gold_entries:
        if g["gold_id"] in used:
            continue
        gc = normalize_citation(g["citation"])
        if not (pc.startswith(gc) or gc.startswith(pc) or pc == gc):
            continue
        score = fuzz.token_set_ratio(pred.obligation.summary, g["canonical_summary"])
        if score >= SUMMARY_MATCH_THRESHOLD and score > best_score:
            best, best_score = g, score

    return best


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def bar(label: str, value: float, width: int = 34) -> str:
    filled = round(value * width)
    return f"  {label:<34} {'█' * filled}{'·' * (width - filled)}  {value * 100:6.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="legiswatch-eval", description="Score pipeline output against the gold set."
    )
    ap.add_argument(
        "--provider", default="replay", choices=["replay", "anthropic", "openai", "ollama"]
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--json-out", default=None, help="Where to write the machine-readable report.")
    args = ap.parse_args()
    json_out = Path(args.json_out) if args.json_out else OUT / "eval_report.json"

    configure("WARNING")  # quiet the pipeline; the report is the output

    gold = json.loads(GOLD_PATH.read_text())
    gold_entries: list[dict] = gold["obligations"]
    gold_by_id = {g["gold_id"]: g for g in gold_entries}

    docs = load_corpus()
    client = LLMClient(provider=args.provider, model=args.model)
    summary = run_pipeline(docs, client)

    predictions = [r for res in summary.results for r in res.routed_obligations]

    # ---- match predictions to gold ---------------------------------------
    used: set = set()
    matches: dict[str, str] = {}  # obligation_id -> gold_id
    false_positives = []

    # Match higher-confidence predictions first so they claim their gold entry.
    for pred in sorted(predictions, key=lambda r: -r.combined_confidence):
        g = match_prediction(pred, gold_entries, used)
        if g:
            used.add(g["gold_id"])
            matches[pred.obligation.obligation_id] = g["gold_id"]
        else:
            false_positives.append(pred)

    missed = [g for g in gold_entries if g["gold_id"] not in used]

    tp, fp, fn = len(matches), len(false_positives), len(missed)
    p_raw, r_raw, f_raw = prf(tp, fp, fn)

    # ---- post-guardrail: what actually reaches a human --------------------
    kept = [r for r in predictions if r.route != RouteDecision.REJECTED]
    kept_tp = sum(1 for r in kept if r.obligation.obligation_id in matches)
    kept_fp = len(kept) - kept_tp
    kept_fn = len(gold_entries) - kept_tp
    p_g, r_g, f_g = prf(kept_tp, kept_fp, kept_fn)

    # ---- groundedness -----------------------------------------------------
    ungrounded = [r for r in predictions if not r.groundedness.passed]
    caught = [r for r in ungrounded if r.route == RouteDecision.REJECTED]
    grounded_rate = 1 - len(ungrounded) / len(predictions) if predictions else 1.0
    catch_rate = len(caught) / len(ungrounded) if ungrounded else 1.0

    # ---- attribution ------------------------------------------------------
    unit_correct = sum(
        1
        for r in predictions
        if r.obligation.obligation_id in matches
        and r.obligation.responsible_unit
        == IUUnit[gold_by_id[matches[r.obligation.obligation_id]]["expected_unit"]]
    )
    unit_acc = unit_correct / tp if tp else 0.0

    type_correct = sum(
        1
        for r in predictions
        if r.obligation.obligation_id in matches
        and r.obligation.obligation_type.value
        == gold_by_id[matches[r.obligation.obligation_id]]["expected_type"]
    )
    type_acc = type_correct / tp if tp else 0.0

    # ---- the safety number ------------------------------------------------
    auto = [r for r in predictions if r.route == RouteDecision.AUTO_FILE]
    bad_auto = [r for r in auto if r.obligation.obligation_id not in matches]
    auto_precision = 1 - len(bad_auto) / len(auto) if auto else 1.0

    # ---- triage -----------------------------------------------------------
    triage_tp = sum(
        1
        for res in summary.results
        if res.triage and res.triage.is_relevant and not _is_neg(res.doc_id, docs)
    )
    triage_fp = sum(
        1
        for res in summary.results
        if res.triage and res.triage.is_relevant and _is_neg(res.doc_id, docs)
    )
    triage_fn = sum(
        1
        for res in summary.results
        if res.triage and not res.triage.is_relevant and not _is_neg(res.doc_id, docs)
    )
    tp_p, tp_r, tp_f = prf(triage_tp, triage_fp, triage_fn)

    # ---- report -----------------------------------------------------------
    width = 74
    print("\n" + "=" * width)
    print("  LEGISWATCH EVALUATION")
    print(f"  corpus: {len(docs)} documents   gold set: {len(gold_entries)} obligations")
    print(f"  provider: {client.provider} / {client.model}")
    print("=" * width)

    print("\n  STAGE 1 — RELEVANCE TRIAGE")
    print(bar("precision", tp_p))
    print(bar("recall", tp_r))
    print(
        f"  {triage_fp} irrelevant document(s) passed to extraction "
        f"(cost leak), {triage_fn} relevant document(s) dropped (compliance risk)"
    )

    print("\n  STAGE 2 — OBLIGATION EXTRACTION, RAW MODEL OUTPUT")
    print(bar("precision", p_raw))
    print(bar("recall", r_raw))
    print(bar("F1", f_raw))
    print(f"  true positives {tp}   false positives {fp}   missed {fn}")

    print("\n  STAGE 3 — AFTER GUARDRAILS (what reaches a human)")
    print(bar("precision", p_g))
    print(bar("recall", r_g))
    print(bar("F1", f_g))
    delta = (p_g - p_raw) * 100
    print(f"  precision delta from guardrail layer: {delta:+.1f} points")
    print(f"  true obligations lost to guardrails:  {kept_fn - fn}")

    print("\n  GROUNDEDNESS")
    print(bar("quotes verified in source", grounded_rate))
    print(bar("ungrounded quotes intercepted", catch_rate))
    print(
        f"  {len(ungrounded)} ungrounded quote(s) produced, {len(caught)} stopped "
        f"before reaching a queue"
    )

    print("\n  ATTRIBUTION (on correctly extracted obligations)")
    print(bar("responsible office correct", unit_acc))
    print(bar("obligation type correct", type_acc))

    print("\n  AUTO-FILE SAFETY")
    print(bar("auto-filed entries that are correct", auto_precision))
    print(f"  {len(auto)} auto-filed, {len(bad_auto)} of them wrong")

    if missed:
        print("\n  MISSED OBLIGATIONS (recall failures)")
        for g in missed:
            print(f"    · {g['gold_id']}  {g['citation']}")
            print(f"        {g['canonical_summary'][:88]}")

    if false_positives:
        print("\n  FALSE POSITIVES (and what happened to them)")
        for r in false_positives:
            print(f"    · {r.obligation.obligation_id}")
            print(f"        route: {r.route.value.upper()} — {r.route_reason[:78]}")

    print("\n" + "=" * width + "\n")

    report = {
        "corpus_documents": len(docs),
        "gold_obligations": len(gold_entries),
        "provider": client.provider,
        "model": client.model,
        "triage": {"precision": tp_p, "recall": tp_r, "f1": tp_f},
        "extraction_raw": {
            "precision": p_raw,
            "recall": r_raw,
            "f1": f_raw,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "extraction_post_guardrail": {"precision": p_g, "recall": r_g, "f1": f_g},
        "groundedness": {
            "verified_rate": grounded_rate,
            "intercept_rate": catch_rate,
            "ungrounded_count": len(ungrounded),
        },
        "attribution": {"unit_accuracy": unit_acc, "type_accuracy": type_acc},
        "auto_file": {"count": len(auto), "precision": auto_precision, "incorrect": len(bad_auto)},
        "missed": [g["gold_id"] for g in missed],
        "false_positives": [
            {"id": r.obligation.obligation_id, "route": r.route.value} for r in false_positives
        ],
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2))
    print(f"  Report -> {json_out}\n")

    return 0 if not bad_auto else 1  # non-zero exit if anything unsafe was auto-filed


def _is_neg(doc_id: str, docs: list[dict]) -> bool:
    return any(d["doc_id"] == doc_id and d.get("is_negative_control") for d in docs)


if __name__ == "__main__":
    raise SystemExit(main())
