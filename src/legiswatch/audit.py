"""Append-only decision journal.

Every human action taken on an obligation lands here as one JSON line. Nothing
is ever updated in place and nothing is ever deleted -- reversing a decision
writes a *new* record that points at the one it supersedes.

That constraint is the whole design. A mutable `status` field on an obligation
tells you where something ended up; it cannot tell you that an analyst rejected
it in March, a different analyst reinstated it in April, and why either of them
did. In a compliance system the second question is the one that gets asked.

Storage is JSONL rather than a database on purpose. It is append-only by nature
(a crash mid-write costs one line, not the file), it is greppable, it diffs
cleanly in version control, and it survives being copied to someone's laptop as
evidence. A real deployment would put this in Postgres with the same append-only
discipline; the interface below would not change.

The other reason this file exists: **the journal is the training set.** Every
override is a labelled disagreement between the pipeline and a domain expert.
`stats()` turns that into the numbers that tell you whether the auto-file
threshold is set correctly and whether the routing map is wrong for a specific
office.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .logging_setup import get_logger
from .schemas import (
    AnalystAction,
    AnalystDecision,
    DecidedBy,
    ObligationLifecycle,
    OverrideStats,
    RoutedObligation,
)

log = get_logger(__name__)

DEFAULT_JOURNAL = (
    Path(os.getenv("LEGISWATCH_OUT_DIR", Path.cwd() / "out")) / "decision_journal.jsonl"
)


def new_decision_id() -> str:
    return f"dec-{uuid.uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AuditJournal:
    """Append-only store of analyst decisions."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_JOURNAL
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- writing -----------------------------------------------------------

    def append(self, decision: AnalystDecision) -> AnalystDecision:
        """Write one record. Never overwrites, never reorders.

        Opened in append mode with an explicit flush and fsync: if the process
        dies immediately after a reviewer clicks confirm, the record is on disk.
        Losing an analyst's decision silently is worse than failing loudly.
        """
        line = json.dumps(decision.model_dump(mode="json"), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        log.info(
            "analyst_decision_recorded",
            extra={
                "decision_id": decision.decision_id,
                "obligation_id": decision.obligation_id,
                "reviewer": decision.reviewer,
                "action": decision.action.value,
                "from_route": decision.from_route.value,
                "to_route": decision.to_route.value,
                "reprocessed": decision.reprocessed,
                "supersedes": decision.supersedes,
            },
        )
        return decision

    # -- reading -----------------------------------------------------------

    def all(self) -> list[AnalystDecision]:
        """Every decision ever recorded, in write order.

        A malformed line is logged and skipped rather than killing the read --
        one corrupt record must not make the entire audit trail unreadable.
        """
        if not self.path.exists():
            return []

        out: list[AnalystDecision] = []
        for i, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                out.append(AnalystDecision.model_validate_json(raw))
            except Exception as e:
                log.error(
                    "journal_line_unreadable",
                    extra={"line_number": i, "error": str(e)[:200]},
                )
        return out

    def history_for(self, obligation_id: str) -> list[AnalystDecision]:
        """Every decision touching one obligation, oldest first."""
        return [d for d in self.all() if d.obligation_id == obligation_id]

    def latest_for(self, obligation_id: str) -> AnalystDecision | None:
        h = self.history_for(obligation_id)
        return h[-1] if h else None

    def touched_obligation_ids(self) -> list[str]:
        seen: list[str] = []
        for d in self.all():
            if d.obligation_id not in seen:
                seen.append(d.obligation_id)
        return seen

    # -- derived views -----------------------------------------------------

    def lifecycle(self, routed: RoutedObligation) -> ObligationLifecycle:
        """Replay the journal over one agent decision to get its current state.

        Derived, never stored. The journal stays the single source of truth, so
        there is no possibility of a cached state field drifting out of sync with
        the record that justifies it.
        """
        oid = routed.obligation.obligation_id
        decisions = self.history_for(oid)

        if not decisions:
            return ObligationLifecycle(
                obligation_id=oid,
                agent_decision=routed,
                analyst_decisions=[],
                current_route=routed.route,
                current_unit=routed.obligation.responsible_unit,
                current_obligation=routed.obligation,
                decided_by=DecidedBy.AGENT,
                is_overridden=False,
            )

        latest = decisions[-1]
        current = routed.obligation.model_copy(deep=True)

        # Replay every amendment in order so the current text reflects all edits,
        # not just the most recent one.
        for d in decisions:
            for field, value in d.field_edits.items():
                if hasattr(current, field):
                    setattr(current, field, value)
            if d.to_unit is not None:
                current.responsible_unit = d.to_unit

        return ObligationLifecycle(
            obligation_id=oid,
            agent_decision=routed,
            analyst_decisions=decisions,
            current_route=latest.to_route,
            current_unit=latest.to_unit or current.responsible_unit,
            current_obligation=current,
            decided_by=latest.decided_by,
            is_overridden=any(d.action != AnalystAction.CONFIRM for d in decisions),
        )

    def stats(self) -> OverrideStats:
        """Agreement metrics. The reason the journal is worth keeping."""
        decisions = self.all()
        if not decisions:
            return OverrideStats(
                total_decisions=0,
                obligations_reviewed=0,
                confirmed=0,
                overridden=0,
                reinstated=0,
                override_rate=0.0,
            )

        by_action = Counter(d.action.value for d in decisions)
        by_route = Counter(d.from_route.value for d in decisions)
        reviewers = Counter(d.reviewer for d in decisions)

        unit_corrections: Counter = Counter()
        for d in decisions:
            if d.to_unit is not None and d.from_unit is not None and d.to_unit != d.from_unit:
                unit_corrections[f"{d.from_unit.name} -> {d.to_unit.name}"] += 1

        confirmed = by_action.get(AnalystAction.CONFIRM.value, 0)
        reinstated = by_action.get(AnalystAction.REINSTATE.value, 0)
        overridden = len(decisions) - confirmed

        return OverrideStats(
            total_decisions=len(decisions),
            obligations_reviewed=len({d.obligation_id for d in decisions}),
            confirmed=confirmed,
            overridden=overridden,
            reinstated=reinstated,
            override_rate=round(overridden / len(decisions), 4),
            by_action=dict(by_action),
            by_original_route=dict(by_route),
            unit_corrections=dict(unit_corrections),
            reviewers=dict(reviewers),
        )

    def export_training_pairs(self) -> list[dict]:
        """Every disagreement, as a labelled example.

        The practical use: a regression set. When you change an extraction prompt
        you re-run these and check you have not reintroduced an error a human
        already corrected once.
        """
        pairs = []
        for d in self.all():
            if d.action == AnalystAction.CONFIRM:
                continue
            pairs.append(
                {
                    "obligation_id": d.obligation_id,
                    "agent_route": d.from_route.value,
                    "human_route": d.to_route.value,
                    "agent_unit": d.from_unit.value if d.from_unit else None,
                    "human_unit": d.to_unit.value if d.to_unit else None,
                    "corrections": d.field_edits,
                    "human_reasoning": d.comment,
                    "action": d.action.value,
                }
            )
        return pairs
