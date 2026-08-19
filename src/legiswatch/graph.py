"""LangGraph orchestration.

The per-document graph:

    ingest ─→ triage ─┬─(not relevant)─→ skip ─→ END
                      │
                      └─(relevant)─────→ extract ─┬─(0 obligations)─→ skip ─→ END
                                                  │
                                                  └─→ verify ─→ route ─→ END

Modelled as an explicit state machine rather than a linear script for three
reasons that matter operationally:

* **Conditional short-circuiting.** Irrelevant documents never reach the
  expensive extraction node, and documents that yield nothing never reach the
  verifier. On a real legislative session that is the difference between a
  tractable bill and an untenable one.
* **Inspectable state.** Every node reads and writes one typed dict, so a failed
  run can be replayed from the state at any node instead of from the top.
* **Checkpointable.** A LangGraph state machine takes a checkpointer without
  restructuring, which is what you need when a run spans a thousand documents
  and the process dies at document 700.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .llm import LLMClient
from .logging_setup import get_logger
from .nodes.extract import extract_obligations
from .nodes.route import route_obligation
from .nodes.triage import triage_document
from .nodes.verify import check_groundedness, verify_obligation
from .schemas import (
    DocumentResult,
    Obligation,
    RouteDecision,
    RunSummary,
    TriageResult,
)

log = get_logger(__name__)


class DocState(TypedDict, total=False):
    """State threaded through the per-document graph."""

    doc: dict[str, Any]
    triage: TriageResult | None
    obligations: list[Obligation]
    extraction_notes: str | None
    routed: list[Any]
    skipped_reason: str | None
    timings: dict[str, float]
    _client: LLMClient


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def node_ingest(state: DocState) -> DocState:
    doc = state["doc"]
    log.info("ingest", extra={"doc_id": doc["doc_id"], "chars": len(doc["text"])})
    return {"timings": {}, "obligations": [], "routed": []}


def node_triage(state: DocState) -> DocState:
    t0 = time.perf_counter()
    result = triage_document(state["_client"], state["doc"])
    timings = dict(state.get("timings", {}))
    timings["triage_ms"] = (time.perf_counter() - t0) * 1000
    return {"triage": result, "timings": timings}


def node_extract(state: DocState) -> DocState:
    t0 = time.perf_counter()
    result = extract_obligations(state["_client"], state["doc"])
    timings = dict(state.get("timings", {}))
    timings["extract_ms"] = (time.perf_counter() - t0) * 1000
    return {
        "obligations": result.obligations,
        "extraction_notes": result.extraction_notes,
        "timings": timings,
    }


def node_verify_and_route(state: DocState) -> DocState:
    """Verification and routing share a node because they are one decision.

    Groundedness is checked deterministically first; the adversarial LLM audit
    only runs on quotes that survive, so a hallucinated quote never costs a
    second inference call.
    """
    t0 = time.perf_counter()
    doc = state["doc"]
    client = state["_client"]
    routed = []

    for ob in state.get("obligations", []):
        grounded = check_groundedness(ob.verbatim_quote, doc["text"])

        verdict = None
        if grounded.passed:
            verdict = verify_obligation(client, ob, doc["text"], doc["citation"])
        else:
            log.warning(
                "groundedness_failed_skipping_llm_verify",
                extra={
                    "obligation_id": ob.obligation_id,
                    "fuzzy_score": grounded.fuzzy_score,
                },
            )

        routed.append(
            route_obligation(
                obligation=ob,
                groundedness=grounded,
                verdict=verdict,
                source_doc_id=doc["doc_id"],
                source_url=doc.get("source_url"),
            )
        )

    timings = dict(state.get("timings", {}))
    timings["verify_route_ms"] = (time.perf_counter() - t0) * 1000
    return {"routed": routed, "timings": timings}


def node_skip(state: DocState) -> DocState:
    triage = state.get("triage")
    if triage is not None and not triage.is_relevant:
        reason = f"Triaged as not relevant (confidence {triage.confidence:.2f}): {triage.rationale}"
    else:
        reason = "Relevant, but no discrete obligations were extracted."
    log.info("document_skipped", extra={"doc_id": state["doc"]["doc_id"], "reason": reason[:200]})
    return {"skipped_reason": reason}


# --------------------------------------------------------------------------
# Conditional edges
# --------------------------------------------------------------------------


def after_triage(state: DocState) -> str:
    triage = state.get("triage")
    return "extract" if (triage and triage.is_relevant) else "skip"


def after_extract(state: DocState) -> str:
    return "verify_and_route" if state.get("obligations") else "skip"


def build_graph():
    g = StateGraph(DocState)
    g.add_node("ingest", node_ingest)
    g.add_node("triage", node_triage)
    g.add_node("extract", node_extract)
    g.add_node("verify_and_route", node_verify_and_route)
    g.add_node("skip", node_skip)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "triage")
    g.add_conditional_edges("triage", after_triage, {"extract": "extract", "skip": "skip"})
    g.add_conditional_edges(
        "extract", after_extract, {"verify_and_route": "verify_and_route", "skip": "skip"}
    )
    g.add_edge("verify_and_route", END)
    g.add_edge("skip", END)
    return g.compile()


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_pipeline(documents: list[dict], client: LLMClient) -> RunSummary:
    """Execute the graph across a corpus and assemble the run summary."""
    graph = build_graph()
    run_id = f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    results: list[DocumentResult] = []

    for doc in documents:
        final = graph.invoke({"doc": doc, "_client": client})
        results.append(
            DocumentResult(
                doc_id=doc["doc_id"],
                citation=doc["citation"],
                title=doc["title"],
                source_url=doc.get("source_url"),
                triage=final.get("triage"),
                routed_obligations=final.get("routed", []),
                skipped_reason=final.get("skipped_reason"),
                stage_timings_ms={k: round(v, 1) for k, v in final.get("timings", {}).items()},
            )
        )

    total_ms = (time.perf_counter() - t0) * 1000
    finished = datetime.now(timezone.utc)

    all_routed = [r for res in results for r in res.routed_obligations]
    relevant = sum(1 for r in results if r.triage and r.triage.is_relevant)

    summary = RunSummary(
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=finished.isoformat(timespec="seconds"),
        provider=client.provider,
        model=client.model,
        documents_ingested=len(documents),
        documents_relevant=relevant,
        documents_skipped=sum(1 for r in results if r.skipped_reason),
        obligations_extracted=len(all_routed),
        obligations_auto_filed=sum(1 for r in all_routed if r.route == RouteDecision.AUTO_FILE),
        obligations_needing_review=sum(
            1 for r in all_routed if r.route == RouteDecision.HUMAN_REVIEW
        ),
        obligations_rejected=sum(1 for r in all_routed if r.route == RouteDecision.REJECTED),
        total_latency_ms=round(total_ms, 1),
        estimated_cost_usd=round(client.estimated_cost_usd(), 6),
        results=results,
    )

    log.info(
        "run_complete",
        extra={
            "run_id": run_id,
            "documents": summary.documents_ingested,
            "obligations": summary.obligations_extracted,
            "auto_filed": summary.obligations_auto_filed,
            "review": summary.obligations_needing_review,
            "rejected": summary.obligations_rejected,
            "latency_ms": summary.total_latency_ms,
            "llm_calls": client.call_count,
        },
    )
    return summary
