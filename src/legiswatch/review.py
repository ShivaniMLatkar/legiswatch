"""The review engine -- what happens when a human disagrees with the pipeline.

There are two fundamentally different things an analyst can do, and collapsing
them into one "override" concept loses the distinction that matters most when
someone audits this later:

**Override.** The human substitutes their judgement for the machine's. The route
is set by hand and the machine's opinion is recorded but not re-consulted.
Provenance becomes ANALYST. Use when the pipeline reached a defensible
conclusion and a person with more context disagrees.

**Amend and reprocess.** The human corrects the *input* -- a mis-transcribed
quote, a wrong office -- and the machine re-derives the routing from the
corrected data. Provenance becomes ANALYST_CORRECTED_AGENT_ROUTED. Use when the
pipeline reasoned correctly from bad input.

The second is the more valuable path, because the deterministic guarantees still
hold afterwards. If an analyst fixes a quote, groundedness re-runs against the
real source text -- so a "corrected" quote that still is not in the document
gets rejected again. A human can be wrong too, and reprocessing is what stops a
well-meaning correction from smuggling an ungrounded obligation into the tracker.

Reinstatement is the case that motivated all of this. A REJECTED obligation is
not a dead end: guardrails have false positives (a quote can fail groundedness
because the source PDF was badly OCR'd, not because the model invented it), so
an analyst must be able to correct and push it back through. Note that
reinstating does not *force* acceptance -- it re-runs the checks, and if they
still fail, the honest answer is recorded.
"""

from __future__ import annotations

from .audit import AuditJournal, new_decision_id, utc_now
from .llm import LLMClient
from .logging_setup import get_logger
from .nodes.route import route_obligation
from .nodes.verify import check_groundedness, verify_obligation
from .schemas import (
    AnalystAction,
    AnalystDecision,
    DecidedBy,
    IUUnit,
    RouteDecision,
    RoutedObligation,
)

log = get_logger(__name__)

# Fields an analyst is allowed to correct. Deliberately narrow: an analyst can
# fix what the statute says and who owns it, but cannot hand-edit confidence
# scores or groundedness results -- those are derived, and letting a human write
# them directly would make every downstream number meaningless.
EDITABLE_FIELDS = {
    "summary",
    "verbatim_quote",
    "citation",
    "deadline_text",
    "obligation_type",
    "recurrence",
    "unit_rationale",
}

# Editing any of these invalidates the previous adversarial verdict, because the
# verdict is a set of judgements about *specific text* -- "is this summary
# faithful to that quote," "is this deadline supported." Change the text and the
# judgement no longer describes the object it is attached to.
#
# This matters in exactly the case the review layer exists for: an obligation was
# rejected because its summary invented a deadline, an analyst removes the
# invented deadline, and the pipeline must not then re-reject it by consulting a
# stale verdict about the summary that no longer exists. A discarded verdict
# routes the corrected obligation to human review rather than auto-filing it,
# which is the conservative and correct outcome.
VERDICT_INVALIDATING_FIELDS = {"summary", "verbatim_quote", "deadline_text"}


class ReviewError(ValueError):
    pass


def _validate_edits(field_edits: dict[str, str]) -> None:
    illegal = set(field_edits) - EDITABLE_FIELDS
    if illegal:
        raise ReviewError(
            f"Fields not editable by an analyst: {sorted(illegal)}. "
            f"Editable: {sorted(EDITABLE_FIELDS)}"
        )


def apply_decision(
    routed: RoutedObligation,
    *,
    action: AnalystAction,
    reviewer: str,
    comment: str,
    source_text: str,
    journal: AuditJournal,
    run_id: str,
    to_unit: IUUnit | None = None,
    to_route: RouteDecision | None = None,
    field_edits: dict[str, str] | None = None,
    llm_client: LLMClient | None = None,
    reverify: bool = False,
) -> tuple[RoutedObligation, AnalystDecision]:
    """Apply one analyst decision, journal it, and return the new state.

    `source_text` is required for every action, not just the reprocessing ones,
    because any path that re-runs groundedness needs the real document to check
    against. Making it mandatory removes the possibility of a caller reprocessing
    against nothing and getting a free pass.
    """
    field_edits = dict(field_edits or {})
    _validate_edits(field_edits)

    original = routed.obligation
    from_route = routed.route
    from_unit = original.responsible_unit

    prior = journal.latest_for(original.obligation_id)
    supersedes = prior.decision_id if prior else None

    updated = routed
    reprocessed = False
    trace: str | None = None

    # ---------------------------------------------------------------- CONFIRM
    if action == AnalystAction.CONFIRM:
        if from_route == RouteDecision.REJECTED:
            raise ReviewError(
                "Cannot CONFIRM a rejected obligation -- use REINSTATE, which "
                "re-runs the checks rather than bypassing them."
            )
        resolved_route = RouteDecision.AUTO_FILE
        decided_by = DecidedBy.ANALYST
        updated = routed.model_copy(
            update={
                "route": resolved_route,
                "route_reason": f"Confirmed by {reviewer}: {comment}",
            }
        )

    # ----------------------------------------------------------------- REJECT
    elif action == AnalystAction.REJECT:
        resolved_route = RouteDecision.REJECTED
        decided_by = DecidedBy.ANALYST
        updated = routed.model_copy(
            update={
                "route": resolved_route,
                "route_reason": f"Rejected by {reviewer}: {comment}",
            }
        )

    # ---------------------------------------------------------------- REROUTE
    elif action == AnalystAction.REROUTE:
        if to_route is None:
            raise ReviewError("REROUTE requires an explicit to_route.")
        resolved_route = to_route
        decided_by = DecidedBy.ANALYST
        updated = routed.model_copy(
            update={
                "route": resolved_route,
                "route_reason": (
                    f"Route forced to {to_route.value} by {reviewer}, overriding "
                    f"the pipeline's {from_route.value}: {comment}"
                ),
            }
        )

    # ----------------------------------- REASSIGN / AMEND / REINSTATE (re-run)
    elif action in (AnalystAction.REASSIGN, AnalystAction.AMEND, AnalystAction.REINSTATE):
        if action == AnalystAction.REASSIGN and to_unit is None:
            raise ReviewError("REASSIGN requires to_unit.")
        if action == AnalystAction.AMEND and not field_edits:
            raise ReviewError("AMEND requires at least one field edit.")
        if action == AnalystAction.REINSTATE and from_route != RouteDecision.REJECTED:
            raise ReviewError(
                f"REINSTATE only applies to rejected obligations; this one is {from_route.value}."
            )

        corrected = original.model_copy(deep=True)
        for field, value in field_edits.items():
            setattr(corrected, field, value)
        if to_unit is not None:
            corrected.responsible_unit = to_unit

        # Re-run the deterministic gate against the REAL source. A corrected
        # quote that still is not in the document must still fail.
        grounded = check_groundedness(corrected.verbatim_quote, source_text)

        # Decide what to do with the previous verdict before routing.
        verdict = routed.verdict
        stale = bool(set(field_edits) & VERDICT_INVALIDATING_FIELDS) or (
            to_unit is not None and to_unit != from_unit
        )
        verdict_note = ""

        if not grounded.passed:
            verdict = None
            verdict_note = "quote failed groundedness, prior verdict discarded"
        elif reverify and llm_client is not None:
            verdict = verify_obligation(llm_client, corrected, source_text, corrected.citation)
            verdict_note = "adversarial verification re-run on corrected text"
        elif stale:
            verdict = None
            verdict_note = (
                "prior verdict discarded as stale (it judged text the analyst has "
                "since changed); routed without a semantic verdict, which cannot "
                "auto-file"
            )
        else:
            verdict_note = "prior verdict still applies (no assessed field changed)"

        updated = route_obligation(
            obligation=corrected,
            groundedness=grounded,
            verdict=verdict,
            source_doc_id=routed.source_doc_id,
            source_url=routed.source_url,
        )
        resolved_route = updated.route
        decided_by = DecidedBy.ANALYST_CORRECTED_AGENT_ROUTED
        reprocessed = True

        trace = (
            f"Re-ran deterministic checks on analyst-corrected input: groundedness "
            f"exact={grounded.exact_match} fuzzy={grounded.fuzzy_score:.1f} "
            f"passed={grounded.passed}; {verdict_note}; routing returned "
            f"{resolved_route.value} (combined {updated.combined_confidence:.2f})."
        )

        updated = updated.model_copy(
            update={
                "route_reason": (
                    f"{action.value.upper()} by {reviewer}, reprocessed. {updated.route_reason}"
                )
            }
        )

        if action == AnalystAction.REINSTATE and resolved_route == RouteDecision.REJECTED:
            log.warning(
                "reinstatement_still_rejected",
                extra={
                    "obligation_id": original.obligation_id,
                    "reviewer": reviewer,
                    "fuzzy_score": grounded.fuzzy_score,
                },
            )

    else:  # pragma: no cover -- enum is exhaustive
        raise ReviewError(f"Unhandled action: {action}")

    decision = AnalystDecision(
        decision_id=new_decision_id(),
        timestamp=utc_now(),
        run_id=run_id,
        obligation_id=original.obligation_id,
        reviewer=reviewer,
        action=action,
        comment=comment,
        from_route=from_route,
        to_route=resolved_route,
        from_unit=from_unit,
        to_unit=to_unit or updated.obligation.responsible_unit,
        field_edits=field_edits,
        reprocessed=reprocessed,
        reprocess_trace=trace,
        decided_by=decided_by,
        supersedes=supersedes,
    )
    journal.append(decision)

    return updated, decision
