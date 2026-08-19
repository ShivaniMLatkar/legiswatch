"""Tests for the parts that must not silently break.

The priority here is the guardrail layer. An extraction prompt that gets a bit
worse degrades quality; a groundedness check that silently stops working lets
invented obligations through to a real compliance queue. The tests are weighted
accordingly.
"""

from __future__ import annotations

import pytest

from legiswatch.corpus import load_corpus
from legiswatch.graph import run_pipeline
from legiswatch.llm import LLMClient
from legiswatch.nodes.route import combine_confidence, route_obligation
from legiswatch.nodes.verify import check_groundedness, normalize
from legiswatch.schemas import (
    GroundednessCheck,
    IUUnit,
    Obligation,
    ObligationType,
    Recurrence,
    RouteDecision,
    VerificationVerdict,
)

SOURCE = (
    "Sec. 4. (a) Each institution shall do the following: (1) Establish a procedure "
    "that allows both students and employees to submit complaints. (5) Submit a report "
    "to the commission for higher education not later than April 1, 2025, and not later "
    "than April 1 each year thereafter."
)


# ---------------------------------------------------------------------------
# Groundedness -- the load-bearing check
# ---------------------------------------------------------------------------


class TestGroundedness:
    def test_exact_quote_passes(self):
        r = check_groundedness("Each institution shall do the following", SOURCE)
        assert r.exact_match and r.passed and r.fuzzy_score == 100.0

    def test_whitespace_and_case_differences_are_tolerated(self):
        r = check_groundedness("each   institution\nshall do the FOLLOWING", SOURCE)
        assert r.exact_match, "normalisation should absorb whitespace and case"

    def test_smart_quotes_and_dashes_are_tolerated(self):
        src = 'The board shall review the faculty member\'s "criteria" - annually.'
        r = check_groundedness(
            "The board shall review the faculty member’s “criteria” — annually.", src
        )
        assert r.passed

    def test_fabricated_quote_is_rejected(self):
        r = check_groundedness(
            "the institution shall publish a quarterly dashboard summarizing complaint "
            "volumes disaggregated by campus",
            SOURCE,
        )
        assert not r.passed and not r.exact_match

    def test_paraphrase_is_rejected_not_merely_downscored(self):
        """A close paraphrase is the dangerous case: plausible, but not the text."""
        r = check_groundedness(
            "Every institution is required to carry out the items listed below", SOURCE
        )
        assert not r.passed

    def test_normalize_is_idempotent(self):
        once = normalize(SOURCE)
        assert normalize(once) == once


# ---------------------------------------------------------------------------
# Routing -- the governance layer
# ---------------------------------------------------------------------------


def _ob(**kw) -> Obligation:
    base = {
        "obligation_id": "test-ob",
        "summary": "Do the thing the statute requires.",
        "obligation_type": ObligationType.PROCESS,
        "recurrence": Recurrence.CONTINUOUS,
        "verbatim_quote": "Each institution shall do the following",
        "citation": "IC 21-39.5-2-4(a)",
        "responsible_unit": IUUnit.HUMAN_RESOURCES,
        "unit_rationale": "The statute names the institution; HR administers this process.",
        "confidence": 0.95,
    }
    base.update(kw)
    return Obligation(**base)


def _verdict(**kw) -> VerificationVerdict:
    base = {
        "obligation_id": "test-ob",
        "quote_is_grounded": True,
        "summary_is_faithful": True,
        "deadline_is_correct": True,
        "unit_is_defensible": True,
        "verifier_confidence": 0.95,
        "objection": None,
    }
    base.update(kw)
    return VerificationVerdict(**base)


GOOD_GROUND = GroundednessCheck(exact_match=True, fuzzy_score=100.0, passed=True)
BAD_GROUND = GroundednessCheck(exact_match=False, fuzzy_score=41.0, passed=False)


class TestRouting:
    def test_clean_high_confidence_process_auto_files(self):
        r = route_obligation(_ob(), GOOD_GROUND, _verdict(), "doc-1")
        assert r.route == RouteDecision.AUTO_FILE

    def test_ungrounded_quote_is_rejected_regardless_of_confidence(self):
        r = route_obligation(_ob(confidence=1.0), BAD_GROUND, None, "doc-1")
        assert r.route == RouteDecision.REJECTED

    def test_unfaithful_summary_is_rejected(self):
        r = route_obligation(
            _ob(),
            GOOD_GROUND,
            _verdict(summary_is_faithful=False, objection="adds a deadline"),
            "doc-1",
        )
        assert r.route == RouteDecision.REJECTED

    @pytest.mark.parametrize("otype", [ObligationType.REPORT, ObligationType.PROHIBITION])
    def test_high_stakes_types_never_auto_file(self, otype):
        """Even at maximum confidence with a clean verdict, these go to a human."""
        r = route_obligation(
            _ob(obligation_type=otype, confidence=1.0),
            GOOD_GROUND,
            _verdict(verifier_confidence=1.0),
            "doc-1",
        )
        assert r.route == RouteDecision.HUMAN_REVIEW

    def test_unassigned_unit_goes_to_human(self):
        r = route_obligation(
            _ob(responsible_unit=IUUnit.UNASSIGNED), GOOD_GROUND, _verdict(), "doc-1"
        )
        assert r.route == RouteDecision.HUMAN_REVIEW

    def test_low_verifier_confidence_goes_to_human(self):
        r = route_obligation(_ob(), GOOD_GROUND, _verdict(verifier_confidence=0.55), "doc-1")
        assert r.route == RouteDecision.HUMAN_REVIEW

    def test_indefensible_unit_goes_to_human_not_auto_file(self):
        r = route_obligation(
            _ob(),
            GOOD_GROUND,
            _verdict(unit_is_defensible=False, objection="wrong office"),
            "doc-1",
        )
        assert r.route == RouteDecision.HUMAN_REVIEW

    def test_every_route_carries_a_reason(self):
        for ground, verdict in [
            (GOOD_GROUND, _verdict()),
            (BAD_GROUND, None),
            (GOOD_GROUND, _verdict(verifier_confidence=0.4)),
        ]:
            r = route_obligation(_ob(), ground, verdict, "doc-1")
            assert r.route_reason.strip(), "a routing decision without a reason is unauditable"


class TestConfidenceBlending:
    def test_failed_checks_drag_the_score_down(self):
        clean = combine_confidence(0.9, _verdict(), GOOD_GROUND)
        dirty = combine_confidence(0.9, _verdict(unit_is_defensible=False), GOOD_GROUND)
        assert dirty < clean

    def test_weak_groundedness_drags_the_score_down(self):
        strong = combine_confidence(0.9, _verdict(), GOOD_GROUND)
        weak = combine_confidence(
            0.9, _verdict(), GroundednessCheck(exact_match=False, fuzzy_score=93.0, passed=True)
        )
        assert weak < strong

    def test_extractor_self_report_cannot_dominate(self):
        """A model claiming 1.0 with a doubtful verifier must not reach auto-file."""
        score = combine_confidence(1.0, _verdict(verifier_confidence=0.3), GOOD_GROUND)
        assert score < 0.85

    def test_score_stays_in_range(self):
        for e in (0.0, 0.5, 1.0):
            for v in (0.0, 0.5, 1.0):
                s = combine_confidence(e, _verdict(verifier_confidence=v), GOOD_GROUND)
                assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Schema contracts
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_trivial_quote_is_refused(self):
        with pytest.raises(ValueError):
            _ob(verbatim_quote="shall do so")

    def test_enum_name_is_accepted_for_unit(self):
        ob = Obligation.model_validate(
            {
                "obligation_id": "x",
                "summary": "s",
                "obligation_type": "process",
                "recurrence": "annual",
                "verbatim_quote": "Each institution shall do the following",
                "citation": "IC 1-1-1-1",
                "responsible_unit": "REGISTRAR",
                "unit_rationale": "r",
                "confidence": 0.9,
            }
        )
        assert ob.responsible_unit == IUUnit.REGISTRAR

    def test_invented_unit_is_refused(self):
        with pytest.raises(ValueError):
            _ob(responsible_unit="Office of Made Up Things")

    def test_confidence_bounds_enforced(self):
        with pytest.raises(ValueError):
            _ob(confidence=1.4)


# ---------------------------------------------------------------------------
# End to end, on the replay cache
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def summary():
    """One pipeline run shared by every end-to-end assertion."""
    return run_pipeline(load_corpus(), LLMClient(provider="replay"))


class TestEndToEnd:
    def test_negative_controls_are_filtered_before_extraction(self, summary):
        negatives = [r for r in summary.results if r.doc_id.startswith("NEG-CONTROL")]
        assert negatives, "corpus should contain negative controls"
        for r in negatives:
            assert r.triage is not None and not r.triage.is_relevant
            assert not r.routed_obligations, "an irrelevant document must not be extracted from"

    def test_obligations_were_found_in_real_statutes(self, summary):
        assert summary.obligations_extracted >= 20

    def test_both_guardrail_fixtures_are_caught(self, summary):
        rejected = {
            r.obligation.obligation_id
            for res in summary.results
            for r in res.routed_obligations
            if r.route == RouteDecision.REJECTED
        }
        assert "IC-21-39.5-2-4-quarterly-complaint-dashboard" in rejected, (
            "fabricated-quote fixture must be caught by the deterministic check"
        )
        assert "CHE-HEA1001-preliminary-list-response" in rejected, (
            "false-deadline fixture must be caught by the adversarial verifier"
        )

    def test_nothing_auto_filed_without_passing_groundedness(self, summary):
        for res in summary.results:
            for r in res.routed_obligations:
                if r.route == RouteDecision.AUTO_FILE:
                    assert r.groundedness.passed
                    assert r.verdict is not None and r.verdict.all_checks_passed

    def test_no_report_or_prohibition_was_auto_filed(self, summary):
        for res in summary.results:
            for r in res.routed_obligations:
                if r.route == RouteDecision.AUTO_FILE:
                    assert r.obligation.obligation_type not in (
                        ObligationType.REPORT,
                        ObligationType.PROHIBITION,
                    )

    def test_every_kept_obligation_traces_to_a_source_document(self, summary):
        doc_ids = {d["doc_id"] for d in load_corpus()}
        for res in summary.results:
            for r in res.routed_obligations:
                assert r.source_doc_id in doc_ids

    def test_counts_are_internally_consistent(self, summary):
        assert (
            summary.obligations_auto_filed
            + summary.obligations_needing_review
            + summary.obligations_rejected
            == summary.obligations_extracted
        )
