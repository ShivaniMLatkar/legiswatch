"""Stage 1 -- relevance triage.

Runs on every ingested document. Deliberately cheap: one pass over the text
producing a single boolean plus a calibrated confidence, so the expensive
extraction call only fires on documents that actually impose a duty.

A legislative session produces well over a thousand bills, the large majority
of which are irrelevant to any given institution. Placing a short classifier in
front of a long extraction call is where the cost of a full-session run is
determined.
"""

from __future__ import annotations

from ..config import settings
from ..llm import LLMClient
from ..logging_setup import get_logger
from ..schemas import TriageResult

log = get_logger(__name__)

SYSTEM = f"""You are a legislative analyst for a large public university system.

Decide whether a legal document creates, modifies, or removes a duty that \
applies to {settings.institution_name} in its capacity as a \
{settings.institution_class}.

Treat as RELEVANT:
- duties imposed on "{settings.institution_class}s", "institutions", boards of \
trustees, or a state higher education commission where the commission's action \
requires an institutional submission
- requirements about faculty, students, admissions, transfer credit, degree \
programs, accreditation, reporting, or campus policy
- anything naming {settings.institution_name} explicitly

Treat as NOT RELEVANT:
- duties on unrelated agencies, industries, local government, or private parties
- documents that mention education in passing without imposing a duty

Be decisive. If a document imposes no duty on the institution, say so plainly \
and assign low relevance: a false positive costs an expensive extraction call \
and reviewer attention. Report calibrated confidence, using values below 0.7 \
where the document is genuinely ambiguous rather than defaulting to certainty."""

USER_TEMPLATE = """Document citation: {citation}
Title: {title}

--- BEGIN DOCUMENT TEXT ---
{text}
--- END DOCUMENT TEXT ---

Decide whether this document is relevant as defined above."""


def triage_document(client: LLMClient, doc: dict) -> TriageResult:
    result = client.structured(
        system=SYSTEM,
        user=USER_TEMPLATE.format(citation=doc["citation"], title=doc["title"], text=doc["text"]),
        response_model=TriageResult,
        cache_tag=f"triage__{doc['doc_id']}",
    )
    log.info(
        "triage_complete",
        extra={
            "doc_id": doc["doc_id"],
            "is_relevant": result.is_relevant,
            "confidence": result.confidence,
            "topics": result.matched_topics,
        },
    )
    return result
