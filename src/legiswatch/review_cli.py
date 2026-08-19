"""Reviewer console.

The dashboard is for reading; this is for deciding. Operates directly on the run
summary and the decision journal without requiring the HTTP service:

    legiswatch-review list --route rejected
    legiswatch-review show <obligation-id>
    legiswatch-review decide <obligation-id> \\
        --action reinstate --reviewer j.doe \\
        --comment "Quote verified against source; summary corrected." \\
        --set summary="Corrected statement of the duty."
    legiswatch-review history <obligation-id>
    legiswatch-review stats

Current state is always derived by replaying the journal over the original run.
Nothing is written back into the run summary, so the journal remains the single
source of truth and no cached status field can drift from the record that
justifies it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .audit import AuditJournal
from .corpus import load_corpus
from .logging_setup import configure
from .review import ReviewError, apply_decision
from .schemas import (
    AnalystAction,
    IUUnit,
    RouteDecision,
    RoutedObligation,
    RunSummary,
)

OUT = Path(os.getenv("LEGISWATCH_OUT_DIR", Path.cwd() / "out"))
RUN_PATH = OUT / "run_summary.json"

BADGE = {
    RouteDecision.AUTO_FILE: "\033[32m● AUTO-FILED\033[0m",
    RouteDecision.HUMAN_REVIEW: "\033[33m▲ NEEDS REVIEW\033[0m",
    RouteDecision.REJECTED: "\033[31m✕ REJECTED\033[0m",
}
DIM, BOLD, OFF = "\033[2m", "\033[1m", "\033[0m"


def load_run() -> RunSummary:
    if not RUN_PATH.exists():
        sys.exit("No run found. Run `legiswatch-run` first.")
    return RunSummary.model_validate_json(RUN_PATH.read_text())


def find(run: RunSummary, oid: str) -> tuple[RoutedObligation, str]:
    corpus = {d["doc_id"]: d for d in load_corpus()}
    for res in run.results:
        for r in res.routed_obligations:
            if r.obligation.obligation_id == oid:
                return r, corpus[r.source_doc_id]["text"]
    sys.exit(f"Unknown obligation_id: {oid}")


def parse_sets(pairs) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            sys.exit(f"--set expects field=value, got: {p}")
        k, v = p.split("=", 1)
        out[k.strip()] = v
    return out


def rule(char: str = "─", width: int = 78) -> str:
    return DIM + char * width + OFF


# ---------------------------------------------------------------- commands


def cmd_list(args, run: RunSummary, journal: AuditJournal) -> int:
    rows = []
    for res in run.results:
        for r in res.routed_obligations:
            lc = journal.lifecycle(r)
            if args.route and lc.current_route.value != args.route:
                continue
            if args.unit and args.unit.lower() not in lc.current_unit.value.lower():
                continue
            rows.append(lc)

    print(f"\n  {len(rows)} obligation(s)\n")
    for lc in rows:
        mark = f" {BOLD}[reviewed ×{lc.review_count}]{OFF}" if lc.review_count else ""
        print(f"  {BADGE[lc.current_route]}{mark}")
        print(f"  {BOLD}{lc.obligation_id}{OFF}")
        print(f"  {lc.current_obligation.summary[:96]}")
        print(f"  {DIM}{lc.current_unit.value} · {lc.decided_by.value}{OFF}\n")
    return 0


def cmd_show(args, run: RunSummary, journal: AuditJournal) -> int:
    routed, _ = find(run, args.obligation_id)
    lc = journal.lifecycle(routed)
    o, a = lc.current_obligation, lc.agent_decision

    print("\n" + rule("═"))
    print(f"  {BOLD}{lc.obligation_id}{OFF}")
    print(rule("═"))
    print(f"\n  {o.summary}\n")
    print(f"  {DIM}citation{OFF}   {o.citation}")
    print(f"  {DIM}type{OFF}       {o.obligation_type.value} · {o.recurrence.value}")
    print(f"  {DIM}deadline{OFF}   {o.deadline_text or '—'}")
    print(f"  {DIM}owner{OFF}      {lc.current_unit.value}")
    print(f'\n  {DIM}quote{OFF}      "{o.verbatim_quote}"')

    print(f"\n{rule()}\n  {BOLD}WHAT THE AGENT DECIDED{OFF}\n")
    print(f"  route              {a.route.value}")
    print(f"  reason             {a.route_reason}")
    print(f"  combined conf      {a.combined_confidence:.4f}")
    print(f"  extractor conf     {a.obligation.confidence:.2f}")
    print(
        f"  groundedness       exact={a.groundedness.exact_match} "
        f"fuzzy={a.groundedness.fuzzy_score:.1f} passed={a.groundedness.passed}"
    )
    if a.verdict:
        v = a.verdict
        print(f"  verifier conf      {v.verifier_confidence:.2f}")
        print(
            f"  verifier checks    grounded={v.quote_is_grounded} "
            f"faithful={v.summary_is_faithful} deadline={v.deadline_is_correct} "
            f"unit={v.unit_is_defensible}"
        )
        if v.objection:
            print(f"  objection          {v.objection}")
    else:
        print("  verifier           not run (groundedness failed first)")

    print(f"\n{rule()}\n  {BOLD}WHAT HUMANS DID{OFF}\n")
    if not lc.analyst_decisions:
        print(f"  {DIM}No analyst decisions recorded.{OFF}")
    for d in lc.analyst_decisions:
        print(f"  {BOLD}{d.timestamp}{OFF}  {d.reviewer}  →  {d.action.value.upper()}")
        print(
            f"    {d.from_route.value} → {d.to_route.value}"
            + (
                f"   ({d.from_unit.name} → {d.to_unit.name})"
                if d.to_unit and d.from_unit and d.to_unit != d.from_unit
                else ""
            )
        )
        print(f"    comment:   {d.comment}")
        if d.field_edits:
            for k, v in d.field_edits.items():
                print(f"    edited {k}: {v[:70] if v else '(cleared)'}")
        if d.reprocess_trace:
            print(f"    {DIM}{d.reprocess_trace}{OFF}")
        print(
            f"    {DIM}decision_id {d.decision_id}"
            + (f" · supersedes {d.supersedes}" if d.supersedes else "")
            + OFF
        )
        print()

    print(rule())
    print(
        f"  {BOLD}CURRENT STATE{OFF}   {BADGE[lc.current_route]}   "
        f"provenance: {lc.decided_by.value}\n"
    )
    return 0


def cmd_decide(args, run: RunSummary, journal: AuditJournal) -> int:
    routed, source_text = find(run, args.obligation_id)
    try:
        updated, decision = apply_decision(
            routed,
            action=AnalystAction(args.action),
            reviewer=args.reviewer,
            comment=args.comment,
            source_text=source_text,
            journal=journal,
            run_id=run.run_id,
            to_unit=IUUnit[args.to_unit] if args.to_unit else None,
            to_route=RouteDecision(args.to_route) if args.to_route else None,
            field_edits=parse_sets(args.set),
        )
    except (ReviewError, KeyError, ValueError) as e:
        sys.exit(f"\n  Rejected: {e}\n")

    print(f"\n  Recorded {BOLD}{decision.decision_id}{OFF}")
    print(f"  {decision.from_route.value} → {BADGE[updated.route]}")
    if decision.reprocessed:
        print(f"\n  {BOLD}Reprocessed.{OFF} {decision.reprocess_trace}")
        if updated.route == RouteDecision.REJECTED:
            print(
                f"\n  {DIM}The corrected obligation still fails its checks. Fix the quote "
                f"against\n  the source, or use --action reroute to override deliberately — "
                f"that\n  override will be recorded against your name.{OFF}"
            )
    else:
        print(f"\n  {DIM}Human override. The pipeline's opinion is preserved in the journal.{OFF}")
    print(f"\n  Journal: {journal.path}\n")
    return 0


def cmd_history(args, run: RunSummary, journal: AuditJournal) -> int:
    for d in journal.history_for(args.obligation_id):
        print(json.dumps(d.model_dump(mode="json"), indent=2))
    return 0


def cmd_trail(args, run: RunSummary, journal: AuditJournal) -> int:
    decisions = journal.all()
    print(f"\n  {len(decisions)} decision(s) in {journal.path}\n")
    for d in decisions:
        print(
            f"  {d.timestamp}  {d.reviewer:<12} {d.action.value:<10} "
            f"{d.from_route.value} → {d.to_route.value}   {d.obligation_id}"
        )
        print(f"  {DIM}  {d.comment}{OFF}")
    print()
    return 0


def cmd_stats(args, run: RunSummary, journal: AuditJournal) -> int:
    s = journal.stats()
    print("\n" + rule("═"))
    print(f"  {BOLD}PIPELINE / ANALYST AGREEMENT{OFF}")
    print(rule("═") + "\n")
    print(f"  decisions recorded      {s.total_decisions}")
    print(f"  obligations reviewed    {s.obligations_reviewed}")
    print(f"  confirmed as-is         {s.confirmed}")
    print(f"  overridden              {s.overridden}")
    print(f"  reinstated from reject  {s.reinstated}")
    print(f"  override rate           {s.override_rate * 100:.1f}%")
    if s.by_action:
        print("\n  by action")
        for k, v in sorted(s.by_action.items(), key=lambda x: -x[1]):
            print(f"    {k:<12} {v}")
    if s.by_original_route:
        print("\n  by the route the agent chose")
        for k, v in sorted(s.by_original_route.items(), key=lambda x: -x[1]):
            print(f"    {k:<14} {v}")
    if s.unit_corrections:
        print("\n  office corrections")
        for k, v in sorted(s.unit_corrections.items(), key=lambda x: -x[1]):
            print(f"    {k}  ×{v}")
    if s.reviewers:
        print("\n  reviewers")
        for k, v in sorted(s.reviewers.items(), key=lambda x: -x[1]):
            print(f"    {k:<14} {v}")
    print(f"\n  {DIM}A high override rate on auto-filed items means the threshold is too low.")
    print("  Repeated corrections for one office mean the routing map is wrong there.")
    print(f"  Frequent reinstatements mean the guardrails are too aggressive.{OFF}\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="legiswatch-review", description="LegisWatch reviewer console."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    lst = sub.add_parser("list", help="list obligations with their current state")
    lst.add_argument("--route", choices=[r.value for r in RouteDecision])
    lst.add_argument("--unit")

    show = sub.add_parser("show", help="full decision record for one obligation")
    show.add_argument("obligation_id")

    dec = sub.add_parser("decide", help="record an analyst decision")
    dec.add_argument("obligation_id")
    dec.add_argument("--action", required=True, choices=[a.value for a in AnalystAction])
    dec.add_argument("--reviewer", required=True)
    dec.add_argument(
        "--comment",
        required=True,
        help="Mandatory. An override without a stated reason is not accepted.",
    )
    dec.add_argument("--to-unit", choices=[u.name for u in IUUnit])
    dec.add_argument("--to-route", choices=[r.value for r in RouteDecision])
    dec.add_argument(
        "--set", action="append", metavar="FIELD=VALUE", help="Correct a field. Repeatable."
    )

    hist = sub.add_parser("history", help="raw journal entries for one obligation")
    hist.add_argument("obligation_id")

    sub.add_parser("trail", help="the complete decision journal")
    sub.add_parser("stats", help="pipeline/analyst agreement metrics")

    args = p.parse_args()
    configure("WARNING")

    run = load_run()
    journal = AuditJournal()

    return {
        "list": cmd_list,
        "show": cmd_show,
        "decide": cmd_decide,
        "history": cmd_history,
        "trail": cmd_trail,
        "stats": cmd_stats,
    }[args.cmd](args, run, journal)


if __name__ == "__main__":
    raise SystemExit(main())
