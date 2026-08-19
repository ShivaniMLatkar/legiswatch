"""Stage 4 -- confidence-gated routing.

This is the governance layer, and it is deliberately boring: pure arithmetic and
explicit thresholds, no model call. Anyone in the department can read this file
and understand exactly why a given obligation landed where it did, which is the
whole point. "The AI decided" is not an answer that survives an audit.

Three outcomes:

  REJECTED      -- failed the deterministic groundedness check, or the adversarial
                   verifier found a substantive objection. Never reaches a human
                   queue; lands in a rejects log for pipeline debugging.
  HUMAN_REVIEW  -- passed the checks but combined confidence is below the
                   auto-file bar, or the obligation is high-stakes by type.
  AUTO_FILE     -- passed everything with margin. Still fully audit-logged and
                   still reversible.

The thresholds below are tuned conservatively on purpose. In a first production
deployment you want the human-review queue to feel slightly too full, not
slightly too empty -- you can always raise the bar once you have measured
agreement between the pipeline and the analysts.
"""

from __future__ import annotations

from .. import config
from ..config import settings
from ..logging_setup import get_logger
from ..schemas import (
    GroundednessCheck,
    IUUnit,
    Obligation,
    ObligationType,
    RouteDecision,
    RoutedObligation,
    VerificationVerdict,
)

log = get_logger(__name__)

AUTO_FILE_THRESHOLD = settings.auto_file_threshold
MIN_VERIFIER_CONFIDENCE = settings.min_verifier_confidence

# Obligation types that always require human confirmation regardless of model
# confidence. See config.ALWAYS_REVIEW_TYPES for the rationale.
ALWAYS_REVIEW_TYPES = {t for t in ObligationType if t.value in config.ALWAYS_REVIEW_TYPES}

# Multiplier applied to verifier confidence when any of its four checks failed.
# Large enough that a failed check cannot be outweighed by confidence elsewhere.
FAILED_CHECK_PENALTY = 0.4


def combine_confidence(
    extractor_confidence: float,
    verdict: VerificationVerdict | None,
    groundedness: GroundednessCheck,
) -> float:
    """Blend the three signals into one score.

    Weighted toward the verifier and the deterministic check rather than the
    extractor's own self-report, because a model's confidence in its own output
    is the least trustworthy of the three signals.
    """
    ground_component = 1.0 if groundedness.exact_match else groundedness.fuzzy_score / 100.0
    if verdict is None:
        return round(0.5 * extractor_confidence + 0.5 * ground_component, 4)

    verifier_component = verdict.verifier_confidence
    if not verdict.all_checks_passed:
        verifier_component *= FAILED_CHECK_PENALTY

    return round(
        settings.weight_extractor * extractor_confidence
        + settings.weight_verifier * verifier_component
        + settings.weight_groundedness * ground_component,
        4,
    )


def route_obligation(
    obligation: Obligation,
    groundedness: GroundednessCheck,
    verdict: VerificationVerdict | None,
    source_doc_id: str,
    source_url: str | None = None,
) -> RoutedObligation:
    combined = combine_confidence(obligation.confidence, verdict, groundedness)

    # --- rejection gates, evaluated first -------------------------------
    if not groundedness.passed:
        decision, reason = (
            RouteDecision.REJECTED,
            f"Quote not found in source (fuzzy score {groundedness.fuzzy_score:.1f} "
            f"below threshold). Treated as hallucination.",
        )
    elif verdict is not None and not verdict.quote_is_grounded:
        decision, reason = (
            RouteDecision.REJECTED,
            f"Verifier rejected quote provenance: {verdict.objection or 'no detail given'}",
        )
    elif verdict is not None and not verdict.summary_is_faithful:
        decision, reason = (
            RouteDecision.REJECTED,
            f"Summary misstates the quoted duty: {verdict.objection or 'no detail given'}",
        )

    # --- mandatory-review gates ------------------------------------------
    elif verdict is None:
        # No semantic verification exists for this obligation. In a normal run
        # that only happens when groundedness failed, which is already handled
        # above. It also happens on the review path: when an analyst edits the
        # text a verdict was judging, the old verdict is discarded as stale.
        #
        # Without this gate, discarding a verdict would make auto-filing EASIER,
        # because every verdict-based gate below is skipped -- so a corrected
        # obligation could sail past checks the original had to satisfy. An
        # absent verdict must be treated as an unmet requirement, not a waived one.
        decision, reason = (
            RouteDecision.HUMAN_REVIEW,
            "No semantic verification available for this version of the obligation. "
            "Human confirmation required before filing.",
        )
    elif obligation.responsible_unit == IUUnit.UNASSIGNED:
        decision, reason = (
            RouteDecision.HUMAN_REVIEW,
            "No responsible unit could be determined from the statutory text.",
        )
    elif obligation.obligation_type in ALWAYS_REVIEW_TYPES:
        decision, reason = (
            RouteDecision.HUMAN_REVIEW,
            f"Policy gate: all '{obligation.obligation_type.value}' obligations "
            f"receive human confirmation before filing.",
        )
    elif verdict is not None and not verdict.all_checks_passed:
        decision, reason = (
            RouteDecision.HUMAN_REVIEW,
            f"Verifier raised an objection: {verdict.objection or 'unspecified'}",
        )
    elif verdict is not None and verdict.verifier_confidence < MIN_VERIFIER_CONFIDENCE:
        decision, reason = (
            RouteDecision.HUMAN_REVIEW,
            f"Verifier confidence {verdict.verifier_confidence:.2f} below "
            f"{MIN_VERIFIER_CONFIDENCE:.2f}.",
        )
    elif combined < AUTO_FILE_THRESHOLD:
        decision, reason = (
            RouteDecision.HUMAN_REVIEW,
            f"Combined confidence {combined:.2f} below auto-file threshold "
            f"{AUTO_FILE_THRESHOLD:.2f}.",
        )
    else:
        decision, reason = (
            RouteDecision.AUTO_FILE,
            f"All checks passed; combined confidence {combined:.2f}. "
            f"Filed to {obligation.responsible_unit.value}.",
        )

    log.info(
        "routing_decision",
        extra={
            "obligation_id": obligation.obligation_id,
            "route": decision.value,
            "combined_confidence": combined,
            "unit": obligation.responsible_unit.value,
        },
    )

    return RoutedObligation(
        obligation=obligation,
        verdict=verdict,
        groundedness=groundedness,
        route=decision,
        route_reason=reason,
        combined_confidence=combined,
        source_doc_id=source_doc_id,
        source_url=source_url,
    )
