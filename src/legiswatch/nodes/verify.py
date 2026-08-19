"""Stage 3 -- verification.

Two independent checks, in deliberate order.

**Check 1 is deterministic and runs first.** `verbatim_quote` is normalised and
matched against the source text with exact containment, falling back to a fuzzy
partial ratio. No model is consulted. This catches the failure mode that matters
most in compliance work -- a fluent, plausible obligation that the statute does
not actually contain -- and it catches it with string matching rather than by
asking a language model to grade its own homework.

**Check 2 is an adversarial LLM verifier.** It runs only on quotes that survived
check 1, and it is prompted to look for reasons the extraction is wrong rather
than to confirm it. Separating "is this text real" from "is this reading of the
text fair" keeps a cheap deterministic gate in front of an expensive judgement
call.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from ..config import settings
from ..llm import LLMClient
from ..logging_setup import get_logger
from ..schemas import GroundednessCheck, Obligation, VerificationVerdict

log = get_logger(__name__)

# A quote scoring at or above this against its source counts as grounded.
FUZZY_THRESHOLD = settings.fuzzy_threshold

_WS = re.compile(r"\s+")
_SMART_QUOTES = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        " ": " ",
    }
)


def normalize(text: str) -> str:
    """Collapse the differences that are noise, preserve the ones that are signal.

    Whitespace, smart quotes and dash variants are normalised because models
    routinely re-typeset them. Wording is left strictly alone.
    """
    return _WS.sub(" ", text.translate(_SMART_QUOTES)).strip().lower()


def check_groundedness(quote: str, source_text: str) -> GroundednessCheck:
    """Deterministic provenance check. No LLM involved."""
    n_quote = normalize(quote)
    n_source = normalize(source_text)

    exact = n_quote in n_source
    if exact:
        return GroundednessCheck(exact_match=True, fuzzy_score=100.0, passed=True)

    score = float(fuzz.partial_ratio(n_quote, n_source))
    return GroundednessCheck(
        exact_match=False,
        fuzzy_score=round(score, 2),
        passed=score >= FUZZY_THRESHOLD,
    )


SYSTEM = """You are an adversarial verifier auditing another analyst's work on a \
university compliance tracker.

Your job is NOT to agree. Assume the extraction contains an error and go looking \
for it. You are the last check before an obligation is filed against a named \
office, and a wrong entry is worse than a missing one: it sends a real person to \
do work the statute never required, and it creates false confidence that a real \
duty is covered.

Check each of these independently:

1. quote_is_grounded -- does the quoted span genuinely appear in the source text?
2. summary_is_faithful -- does the plain-language summary describe exactly the \
duty in the quote? Mark FALSE if the summary broadens the duty, adds a deadline \
the quote does not contain, changes who must act, or converts a conditional \
duty into an unconditional one.
3. deadline_is_correct -- is the extracted deadline supported by the source? \
Mark TRUE if no deadline was claimed and none exists. Mark FALSE if a deadline \
was claimed that the source does not state, or if a stated deadline was misread.
4. unit_is_defensible -- is the assigned office a reasonable reading of the \
actor the statute names? A defensible-but-arguable assignment passes. An \
assignment contradicting the named actor fails.

If every check passes, set objection to null. Otherwise state the specific \
problem in one or two sentences. Be concrete: name the word or phrase that is \
wrong."""

USER_TEMPLATE = """SOURCE DOCUMENT ({citation}):
--- BEGIN SOURCE ---
{source_text}
--- END SOURCE ---

EXTRACTED OBLIGATION UNDER AUDIT:
  obligation_id:    {obligation_id}
  summary:          {summary}
  type:             {obligation_type}
  recurrence:       {recurrence}
  verbatim_quote:   "{verbatim_quote}"
  citation:         {ob_citation}
  deadline_text:    {deadline_text}
  first_due_date:   {first_due_date}
  responsible_unit: {responsible_unit}
  unit_rationale:   {unit_rationale}
  self-reported confidence: {confidence}

Audit this extraction against the source."""


def verify_obligation(
    client: LLMClient,
    obligation: Obligation,
    source_text: str,
    citation: str,
) -> VerificationVerdict | None:
    """Second-pass adversarial audit. Returns None if the LLM check is skipped."""
    verdict = client.structured(
        system=SYSTEM,
        user=USER_TEMPLATE.format(
            citation=citation,
            source_text=source_text,
            obligation_id=obligation.obligation_id,
            summary=obligation.summary,
            obligation_type=obligation.obligation_type.value,
            recurrence=obligation.recurrence.value,
            verbatim_quote=obligation.verbatim_quote,
            ob_citation=obligation.citation,
            deadline_text=obligation.deadline_text or "(none claimed)",
            first_due_date=obligation.first_due_date or "(none claimed)",
            responsible_unit=obligation.responsible_unit.value,
            unit_rationale=obligation.unit_rationale,
            confidence=obligation.confidence,
        ),
        response_model=VerificationVerdict,
        cache_tag=f"verify__{obligation.obligation_id}",
    )
    log.info(
        "verification_complete",
        extra={
            "obligation_id": obligation.obligation_id,
            "all_passed": verdict.all_checks_passed,
            "verifier_confidence": verdict.verifier_confidence,
            "objection": (verdict.objection or "")[:200],
        },
    )
    return verdict
