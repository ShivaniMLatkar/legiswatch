"""Stage 2 -- obligation extraction.

Converts statutory prose into discrete, individually trackable duties. The
difficulty is decomposition rather than summarisation: a single subsection can
contain several duties with different owners and different recurrence patterns,
and a tracker that stores them as one row cannot represent partial completion.

Two prompt-level constraints carry most of the reliability:

1. Every obligation must include a `verbatim_quote` copied character for
   character from the source. This is verified programmatically downstream, and
   the prompt states that the check is enforced.
2. `responsible_unit` is drawn from a closed enum, so the model cannot emit an
   office that does not exist.
"""

from __future__ import annotations

from ..config import settings
from ..llm import LLMClient
from ..logging_setup import get_logger
from ..schemas import ExtractionResult, IUUnit

log = get_logger(__name__)

_UNIT_MENU = "\n".join(f"- {u.name}: {u.value}" for u in IUUnit)

SYSTEM = f"""You are a compliance analyst at {settings.institution_name}. You \
convert statutory text into a tracked list of discrete institutional obligations.

DECOMPOSITION RULES
- One obligation per distinct duty. If a subsection lists five things the \
institution "shall" do, that is five obligations, not one.
- Split duties that have different owners, different deadlines, or different \
recurrence patterns, even when they sit in the same sentence.
- Capture duties imposed ON the institution. A duty imposed solely on the \
commission for higher education is only in scope if it requires an institutional \
submission -- note that dependency in the summary.
- Do not invent duties that are implied but not stated. If the statute does not \
say it, it is not an obligation.

QUOTE RULES -- STRICTLY ENFORCED
- `verbatim_quote` must be an EXACT contiguous span copied from the document \
text you were given. Character for character. Do not paraphrase, do not fix \
typos, do not join separated fragments with ellipses.
- Every quote is checked programmatically against the source. A quote that does \
not appear verbatim causes the obligation to be discarded, so copying carefully \
matters more than covering everything.

OWNERSHIP RULES
- `responsible_unit` must be one of the following exactly:
{_UNIT_MENU}
- Base the choice on the actor the statute names. If the statute says "the board \
of trustees", route to BOARD_OF_TRUSTEES. If it says "the institution" and the \
duty is about reporting to an external body, prefer the office that owns that \
relationship.
- If the correct owner is genuinely unclear, use UNASSIGNED. Guessing an owner \
is worse than flagging one, because a wrongly-owned obligation is silently \
ignored by the office that receives it.

CONFIDENCE
- Report calibrated confidence per obligation. Reserve values above 0.9 for \
duties stated in unambiguous mandatory language with a clear actor. Use values \
below 0.7 where the statute is vague about who must act or by when."""

USER_TEMPLATE = """Document citation: {citation}
Title: {title}
Enacting act: {enacting_act}
Effective date: {effective_date}

--- BEGIN DOCUMENT TEXT ---
{text}
--- END DOCUMENT TEXT ---

Extract every discrete obligation this document places on the institution.

For obligation_id, use a stable slug combining the citation and a short \
descriptor, for example: "IC-21-39.5-2-4-annual-complaint-report".

For citation, include the subsection path where identifiable, for example \
"IC 21-39.5-2-4(a)(5)".

For first_due_date, give an ISO date (YYYY-MM-DD) only when the statute states \
or clearly implies a specific first due date. Otherwise null."""


def extract_obligations(client: LLMClient, doc: dict) -> ExtractionResult:
    result = client.structured(
        system=SYSTEM,
        user=USER_TEMPLATE.format(
            citation=doc["citation"],
            title=doc["title"],
            enacting_act=doc.get("enacting_act") or "unknown",
            effective_date=doc.get("effective_date") or "unknown",
            text=doc["text"],
        ),
        response_model=ExtractionResult,
        cache_tag=f"extract__{doc['doc_id']}",
    )
    log.info(
        "extraction_complete",
        extra={
            "doc_id": doc["doc_id"],
            "obligations_found": len(result.obligations),
            "has_notes": bool(result.extraction_notes),
        },
    )
    return result
