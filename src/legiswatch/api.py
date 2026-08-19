"""FastAPI service wrapping the pipeline.

Exposes the pipeline and the review layer over HTTP so other systems can
integrate with it. Three integration shapes:

  POST /extract        one ad-hoc document -> obligations (paste text, get duties)
  POST /runs           run the whole corpus, return the full summary
  GET  /obligations    query the last run's results, filtered by route and office

The last one is the endpoint that matters organisationally. Once obligations are
queryable by owning office, the compliance tracker stops being a spreadsheet
someone maintains and becomes something a Slate workflow, a Stellic task, or a
scheduled digest can read from.

    uvicorn legiswatch.api:app --reload --app-dir src
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from .audit import AuditJournal
from .corpus import load_corpus
from .graph import run_pipeline
from .llm import LLMClient
from .logging_setup import get_logger
from .nodes.extract import extract_obligations
from .nodes.route import route_obligation
from .nodes.triage import triage_document
from .nodes.verify import check_groundedness, verify_obligation
from .review import ReviewError, apply_decision
from .schemas import (
    AnalystAction,
    AnalystDecision,
    DocumentResult,
    IUUnit,
    ObligationLifecycle,
    OverrideStats,
    RouteDecision,
    RoutedObligation,
    RunSummary,
    coerce_unit,
)

log = get_logger(__name__)

app = FastAPI(
    title="LegisWatch",
    version="1.0.0",
    description="Statutory obligation extraction, verification and routing.",
)

DEFAULT_PROVIDER = os.getenv("LEGISWATCH_PROVIDER", "replay")
_last_run: RunSummary | None = None


def _client(provider: str | None = None) -> LLMClient:
    return LLMClient(provider=provider or DEFAULT_PROVIDER)


class ExtractRequest(BaseModel):
    text: str = Field(min_length=40, description="Raw statutory or regulatory text.")
    citation: str = Field(default="ad-hoc", description="Citation for the pasted text.")
    title: str = Field(default="Ad-hoc submission")
    provider: str | None = None
    skip_triage: bool = Field(
        default=False,
        description="Extract even if triage says the document is irrelevant. Useful "
        "when a human already knows the document is in scope.",
    )


class ExtractResponse(BaseModel):
    citation: str
    triaged_relevant: bool
    triage_rationale: str
    obligations: list[RoutedObligation]
    counts: dict


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": DEFAULT_PROVIDER,
        "corpus_documents": len(load_corpus()),
        "has_cached_run": _last_run is not None,
    }


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    """Run one document through the full pipeline synchronously."""
    client = _client(req.provider)
    doc = {
        "doc_id": f"adhoc-{abs(hash(req.text)) % 10**8}",
        "citation": req.citation,
        "title": req.title,
        "text": req.text,
        "source_url": None,
    }

    triage = triage_document(client, doc)
    if not triage.is_relevant and not req.skip_triage:
        return ExtractResponse(
            citation=req.citation,
            triaged_relevant=False,
            triage_rationale=triage.rationale,
            obligations=[],
            counts={"extracted": 0, "auto_filed": 0, "review": 0, "rejected": 0},
        )

    extraction = extract_obligations(client, doc)
    routed: list[RoutedObligation] = []
    for ob in extraction.obligations:
        grounded = check_groundedness(ob.verbatim_quote, doc["text"])
        verdict = (
            verify_obligation(client, ob, doc["text"], doc["citation"]) if grounded.passed else None
        )
        routed.append(route_obligation(ob, grounded, verdict, doc["doc_id"], None))

    return ExtractResponse(
        citation=req.citation,
        triaged_relevant=triage.is_relevant,
        triage_rationale=triage.rationale,
        obligations=routed,
        counts={
            "extracted": len(routed),
            "auto_filed": sum(1 for r in routed if r.route == RouteDecision.AUTO_FILE),
            "review": sum(1 for r in routed if r.route == RouteDecision.HUMAN_REVIEW),
            "rejected": sum(1 for r in routed if r.route == RouteDecision.REJECTED),
        },
    )


@app.post("/runs", response_model=RunSummary)
def create_run(provider: str | None = None, include_negatives: bool = True) -> RunSummary:
    """Run the full corpus and cache the result for /obligations."""
    global _last_run
    client = _client(provider)
    _last_run = run_pipeline(load_corpus(include_negative_controls=include_negatives), client)
    return _last_run


@app.get("/runs/latest", response_model=RunSummary)
def latest_run() -> RunSummary:
    if _last_run is None:
        raise HTTPException(404, "No run has been executed yet. POST /runs first.")
    return _last_run


@app.get("/obligations", response_model=list[RoutedObligation])
def list_obligations(
    route: RouteDecision | None = Query(None, description="Filter by routing decision."),
    unit: str | None = Query(None, description="Filter by responsible office (substring match)."),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
) -> list[RoutedObligation]:
    """Query the last run. This is the endpoint downstream systems integrate against."""
    if _last_run is None:
        raise HTTPException(404, "No run has been executed yet. POST /runs first.")

    out = [r for res in _last_run.results for r in res.routed_obligations]
    if route:
        out = [r for r in out if r.route == route]
    if unit:
        out = [r for r in out if unit.lower() in r.obligation.responsible_unit.value.lower()]
    return [r for r in out if r.combined_confidence >= min_confidence]


@app.get("/documents", response_model=list[DocumentResult])
def list_documents() -> list[DocumentResult]:
    if _last_run is None:
        raise HTTPException(404, "No run has been executed yet. POST /runs first.")
    return _last_run.results


# ==========================================================================
# The human layer
#
# Everything above serves the pipeline's output. Everything below serves the
# analyst working the queue: record a decision, read an obligation's full
# history, read the whole audit trail, and see where the pipeline and the
# people reviewing it disagree.
# ==========================================================================


_journal = AuditJournal()


def _find(obligation_id: str) -> tuple[RoutedObligation, str]:
    """Locate an obligation in the last run and return it with its source text.

    The source text is returned alongside deliberately: any reprocessing path
    must re-check the corrected quote against the real document, so the caller
    is never able to reprocess against nothing.
    """
    if _last_run is None:
        raise HTTPException(404, "No run has been executed yet. POST /runs first.")

    corpus = {d["doc_id"]: d for d in load_corpus()}
    for res in _last_run.results:
        for r in res.routed_obligations:
            if r.obligation.obligation_id == obligation_id:
                doc = corpus.get(r.source_doc_id)
                if doc is None:
                    raise HTTPException(
                        500, f"Source document {r.source_doc_id} missing from corpus."
                    )
                return r, doc["text"]
    raise HTTPException(404, f"Unknown obligation_id: {obligation_id}")


class DecisionRequest(BaseModel):
    """An analyst's decision. `comment` is mandatory by schema, not by convention."""

    action: AnalystAction
    reviewer: str = Field(min_length=2, description="Who is making this call.")
    comment: str = Field(
        min_length=10,
        description="Why. Required on every action, including plain confirmations.",
    )
    to_unit: IUUnit | None = None
    to_route: RouteDecision | None = None
    field_edits: dict[str, str] = Field(default_factory=dict)

    # The dashboard sends enum names (its HTML option values); the API should
    # accept those as readily as full office titles.
    _coerce_unit = field_validator("to_unit", mode="before")(
        classmethod(lambda cls, v: coerce_unit(v))
    )
    reverify: bool = Field(
        default=False,
        description="Re-run the adversarial LLM verification on corrected input. "
        "Costs an inference call; the deterministic checks always re-run regardless.",
    )


class DecisionResponse(BaseModel):
    decision: AnalystDecision
    resulting_state: RoutedObligation
    reprocessed: bool
    note: str


@app.post("/obligations/{obligation_id}/decisions", response_model=DecisionResponse)
def record_decision(obligation_id: str, req: DecisionRequest) -> DecisionResponse:
    """Record an analyst decision and return the obligation's new state.

    This is the endpoint that makes the audit trail two-sided. Every call appends
    an immutable record; nothing is ever edited in place.
    """
    routed, source_text = _find(obligation_id)

    try:
        updated, decision = apply_decision(
            routed,
            action=req.action,
            reviewer=req.reviewer,
            comment=req.comment,
            source_text=source_text,
            journal=_journal,
            run_id=_last_run.run_id if _last_run else "unknown",
            to_unit=req.to_unit,
            to_route=req.to_route,
            field_edits=req.field_edits,
            llm_client=_client() if req.reverify else None,
            reverify=req.reverify,
        )
    except ReviewError as e:
        raise HTTPException(400, str(e)) from e

    # Write the new state back into the cached run so subsequent reads reflect it.
    for res in _last_run.results:  # type: ignore[union-attr]
        for i, r in enumerate(res.routed_obligations):
            if r.obligation.obligation_id == obligation_id:
                res.routed_obligations[i] = updated

    if decision.reprocessed and updated.route == RouteDecision.REJECTED:
        note = (
            "Reprocessed, but the corrected obligation still fails its checks and "
            "remains rejected. Correct the quote against the source, or use REROUTE "
            "to override deliberately -- that override will be recorded as yours."
        )
    elif decision.reprocessed:
        note = "Corrected input was pushed back through the checks; the route below is machine-derived."
    else:
        note = "Human override recorded. The pipeline's opinion is preserved in the journal."

    return DecisionResponse(
        decision=decision,
        resulting_state=updated,
        reprocessed=decision.reprocessed,
        note=note,
    )


@app.get("/obligations/{obligation_id}/history", response_model=ObligationLifecycle)
def obligation_history(obligation_id: str) -> ObligationLifecycle:
    """Full lifecycle: what the agent decided, then everything humans did to it."""
    routed, _ = _find(obligation_id)
    return _journal.lifecycle(routed)


@app.get("/audit", response_model=list[AnalystDecision])
def audit_trail(
    obligation_id: str | None = Query(None),
    reviewer: str | None = Query(None),
    action: AnalystAction | None = Query(None),
) -> list[AnalystDecision]:
    """The complete decision journal, oldest first. Filterable, never truncated."""
    out = _journal.all()
    if obligation_id:
        out = [d for d in out if d.obligation_id == obligation_id]
    if reviewer:
        out = [d for d in out if d.reviewer.lower() == reviewer.lower()]
    if action:
        out = [d for d in out if d.action == action]
    return out


@app.get("/audit/stats", response_model=OverrideStats)
def audit_stats() -> OverrideStats:
    """Where the pipeline and the people reviewing it disagree.

    Override rate by original route tells you whether the auto-file threshold is
    set correctly. Unit corrections tell you which parts of the routing map are
    wrong. Reinstatements tell you whether the guardrails are too aggressive.
    """
    return _journal.stats()


@app.get("/audit/training-pairs")
def training_pairs() -> list[dict]:
    """Every human/machine disagreement as a labelled example.

    Use as a regression set: after changing an extraction prompt, re-run these
    and confirm you have not reintroduced an error someone already corrected.
    """
    return _journal.export_training_pairs()
