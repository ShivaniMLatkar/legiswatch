"""Tests for the human review layer.

Weighted toward the properties that make the trail trustworthy rather than the
happy path. Three things must never silently break:

  * the journal is append-only -- a decision, once recorded, cannot be edited away
  * a comment is mandatory -- an override with no stated reason is not accepted
  * reinstatement re-runs the checks rather than bypassing them -- a human can be
    wrong too, and "the analyst said so" must not smuggle an ungrounded
    obligation into a compliance tracker
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from legiswatch.audit import AuditJournal
from legiswatch.nodes.route import route_obligation
from legiswatch.review import ReviewError, apply_decision
from legiswatch.schemas import (
    AnalystAction,
    DecidedBy,
    GroundednessCheck,
    IUUnit,
    Obligation,
    ObligationType,
    Recurrence,
    RouteDecision,
    RoutedObligation,
    VerificationVerdict,
)

SOURCE = (
    "Sec. 4. (a) Each institution shall do the following: (1) Establish a procedure "
    "that allows both students and employees to submit complaints. (5) Submit a report "
    "to the commission for higher education not later than April 1, 2025."
)
GOOD_QUOTE = "Establish a procedure that allows both students and employees to submit complaints"
FAKE_QUOTE = "the institution shall publish a quarterly dashboard of complaint volumes by campus"
LONG_COMMENT = "Checked against the statutory text and this reading is correct."


@pytest.fixture
def journal(tmp_path) -> AuditJournal:
    return AuditJournal(tmp_path / "journal.jsonl")


def make_routed(
    *,
    quote: str = GOOD_QUOTE,
    route=RouteDecision.HUMAN_REVIEW,
    unit=IUUnit.HUMAN_RESOURCES,
    otype=ObligationType.POLICY,
    grounded: bool = True,
    with_verdict: bool = True,
    faithful: bool = True,
) -> RoutedObligation:
    ob = Obligation(
        obligation_id="test-ob",
        summary="Establish a complaint procedure for students and employees.",
        obligation_type=otype,
        recurrence=Recurrence.ONE_TIME,
        verbatim_quote=quote,
        citation="IC 21-39.5-2-4(a)(1)",
        responsible_unit=unit,
        unit_rationale="HR administers complaint intake.",
        confidence=0.9,
    )
    g = GroundednessCheck(
        exact_match=grounded, fuzzy_score=100.0 if grounded else 41.0, passed=grounded
    )
    v = (
        VerificationVerdict(
            obligation_id="test-ob",
            quote_is_grounded=True,
            summary_is_faithful=faithful,
            deadline_is_correct=True,
            unit_is_defensible=True,
            verifier_confidence=0.92,
        )
        if with_verdict
        else None
    )
    return RoutedObligation(
        obligation=ob,
        verdict=v,
        groundedness=g,
        route=route,
        route_reason="seeded for test",
        combined_confidence=0.9,
        source_doc_id="doc-1",
    )


def decide(routed, journal, **kw):
    kw.setdefault("reviewer", "k.adams")
    kw.setdefault("comment", LONG_COMMENT)
    kw.setdefault("source_text", SOURCE)
    kw.setdefault("run_id", "run-test")
    return apply_decision(routed, journal=journal, **kw)


# ---------------------------------------------------------------------------
# The journal itself
# ---------------------------------------------------------------------------


class TestJournalIsAppendOnly:
    def test_decisions_accumulate_and_keep_order(self, journal):
        r = make_routed()
        decide(r, journal, action=AnalystAction.REASSIGN, to_unit=IUUnit.REGISTRAR)
        decide(r, journal, action=AnalystAction.CONFIRM)

        all_ = journal.all()
        assert len(all_) == 2
        assert all_[0].action == AnalystAction.REASSIGN
        assert all_[1].action == AnalystAction.CONFIRM

    def test_reversal_is_a_new_record_not_an_edit(self, journal):
        r = make_routed()
        _, first = decide(r, journal, action=AnalystAction.REJECT)
        _, second = decide(
            make_routed(route=RouteDecision.REJECTED),
            journal,
            action=AnalystAction.REINSTATE,
        )
        assert len(journal.all()) == 2
        assert second.supersedes == first.decision_id
        assert journal.all()[0].to_route == RouteDecision.REJECTED

    def test_history_is_scoped_to_one_obligation(self, journal):
        decide(make_routed(), journal, action=AnalystAction.CONFIRM)
        other = make_routed()
        other.obligation.obligation_id = "other-ob"
        decide(other, journal, action=AnalystAction.CONFIRM)

        assert len(journal.history_for("test-ob")) == 1
        assert len(journal.history_for("other-ob")) == 1
        assert len(journal.all()) == 2

    def test_unreadable_line_does_not_break_the_trail(self, journal):
        decide(make_routed(), journal, action=AnalystAction.CONFIRM)
        with journal.path.open("a") as fh:
            fh.write("{ this is not valid json\n")
        decide(make_routed(), journal, action=AnalystAction.REJECT)

        assert len(journal.all()) == 2, "one corrupt line must not hide the others"

    def test_survives_a_fresh_reader(self, journal):
        decide(make_routed(), journal, action=AnalystAction.CONFIRM)
        assert len(AuditJournal(journal.path).all()) == 1


# ---------------------------------------------------------------------------
# Comments are mandatory
# ---------------------------------------------------------------------------


class TestCommentIsMandatory:
    def test_short_comment_rejected(self, journal):
        with pytest.raises(ValidationError):
            decide(make_routed(), journal, action=AnalystAction.CONFIRM, comment="ok")

    def test_placeholder_comment_rejected(self, journal):
        with pytest.raises(ValidationError):
            decide(
                make_routed(),
                journal,
                action=AnalystAction.CONFIRM,
                comment="fine fine",
            )

    def test_comment_is_preserved_verbatim(self, journal):
        note = "Statute names the board directly; General Counsel was our assumption."
        decide(make_routed(), journal, action=AnalystAction.CONFIRM, comment=note)
        assert journal.all()[0].comment == note


# ---------------------------------------------------------------------------
# Reinstatement re-runs the checks
# ---------------------------------------------------------------------------


class TestReinstatement:
    def test_rejected_is_not_terminal(self, journal):
        rejected = make_routed(route=RouteDecision.REJECTED)
        updated, d = decide(rejected, journal, action=AnalystAction.REINSTATE)
        assert d.action == AnalystAction.REINSTATE
        assert updated.route != RouteDecision.REJECTED

    def test_reinstating_a_fabricated_quote_still_fails(self, journal):
        """The case that matters: a human cannot wave an invented quote through."""
        bogus = make_routed(quote=FAKE_QUOTE, grounded=False, route=RouteDecision.REJECTED)
        updated, d = decide(bogus, journal, action=AnalystAction.REINSTATE)

        assert updated.route == RouteDecision.REJECTED
        assert d.reprocessed is True
        assert not updated.groundedness.passed

    def test_correcting_the_quote_lets_it_through(self, journal):
        bogus = make_routed(quote=FAKE_QUOTE, grounded=False, route=RouteDecision.REJECTED)
        updated, _ = decide(
            bogus,
            journal,
            action=AnalystAction.REINSTATE,
            field_edits={"verbatim_quote": GOOD_QUOTE},
        )
        assert updated.groundedness.passed
        assert updated.route != RouteDecision.REJECTED

    def test_reinstate_only_applies_to_rejected(self, journal):
        with pytest.raises(ReviewError):
            decide(
                make_routed(route=RouteDecision.HUMAN_REVIEW),
                journal,
                action=AnalystAction.REINSTATE,
            )

    def test_confirm_cannot_bypass_a_rejection(self, journal):
        with pytest.raises(ReviewError):
            decide(make_routed(route=RouteDecision.REJECTED), journal, action=AnalystAction.CONFIRM)


# ---------------------------------------------------------------------------
# Override vs amend-and-reprocess
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_reroute_is_a_human_override(self, journal):
        _, d = decide(
            make_routed(), journal, action=AnalystAction.REROUTE, to_route=RouteDecision.AUTO_FILE
        )
        assert d.decided_by == DecidedBy.ANALYST
        assert d.reprocessed is False

    def test_amend_is_machine_routed_from_corrected_input(self, journal):
        _, d = decide(
            make_routed(),
            journal,
            action=AnalystAction.AMEND,
            field_edits={"summary": "Corrected statement of the duty."},
        )
        assert d.decided_by == DecidedBy.ANALYST_CORRECTED_AGENT_ROUTED
        assert d.reprocessed is True
        assert d.reprocess_trace

    def test_reroute_requires_a_target(self, journal):
        with pytest.raises(ReviewError):
            decide(make_routed(), journal, action=AnalystAction.REROUTE)

    def test_amend_requires_an_edit(self, journal):
        with pytest.raises(ReviewError):
            decide(make_routed(), journal, action=AnalystAction.AMEND)

    def test_derived_fields_are_not_analyst_editable(self, journal):
        """Letting a human write confidence directly would void every downstream number."""
        for field in ("confidence", "obligation_id"):
            with pytest.raises(ReviewError):
                decide(
                    make_routed(), journal, action=AnalystAction.AMEND, field_edits={field: "1.0"}
                )


# ---------------------------------------------------------------------------
# A discarded verdict must not become a free pass
# ---------------------------------------------------------------------------


class TestStaleVerdictHandling:
    def test_editing_judged_text_discards_the_verdict(self, journal):
        updated, _ = decide(
            make_routed(),
            journal,
            action=AnalystAction.AMEND,
            field_edits={"summary": "A materially different summary."},
        )
        assert updated.verdict is None

    def test_no_verdict_can_never_auto_file(self):
        """The bug this guards: discarding a verdict skips every verdict-based gate,
        which would make a corrected obligation EASIER to auto-file than the original."""
        ob = make_routed().obligation
        result = route_obligation(
            obligation=ob,
            groundedness=GroundednessCheck(exact_match=True, fuzzy_score=100.0, passed=True),
            verdict=None,
            source_doc_id="doc-1",
        )
        assert result.route == RouteDecision.HUMAN_REVIEW

    def test_editing_untouched_fields_keeps_the_verdict(self, journal):
        updated, _ = decide(
            make_routed(),
            journal,
            action=AnalystAction.AMEND,
            field_edits={"unit_rationale": "Clearer explanation of ownership."},
        )
        assert updated.verdict is not None


# ---------------------------------------------------------------------------
# Lifecycle replay and stats
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_clean_obligation_reports_agent_provenance(self, journal):
        lc = journal.lifecycle(make_routed())
        assert lc.decided_by == DecidedBy.AGENT
        assert lc.is_overridden is False
        assert lc.review_count == 0

    def test_edits_replay_in_order(self, journal):
        r = make_routed()
        decide(r, journal, action=AnalystAction.AMEND, field_edits={"summary": "First correction."})
        decide(
            r, journal, action=AnalystAction.AMEND, field_edits={"summary": "Second correction."}
        )

        lc = journal.lifecycle(r)
        assert lc.current_obligation.summary == "Second correction."
        assert lc.review_count == 2

    def test_confirmation_alone_is_not_an_override(self, journal):
        r = make_routed()
        decide(r, journal, action=AnalystAction.CONFIRM)
        assert journal.lifecycle(r).is_overridden is False

    def test_reassignment_shows_in_current_unit(self, journal):
        r = make_routed(unit=IUUnit.GENERAL_COUNSEL)
        decide(r, journal, action=AnalystAction.REASSIGN, to_unit=IUUnit.BOARD_OF_TRUSTEES)
        assert journal.lifecycle(r).current_unit == IUUnit.BOARD_OF_TRUSTEES


class TestStats:
    def test_override_rate_excludes_confirmations(self, journal):
        r = make_routed()
        decide(r, journal, action=AnalystAction.CONFIRM)
        decide(r, journal, action=AnalystAction.REROUTE, to_route=RouteDecision.REJECTED)

        s = journal.stats()
        assert s.total_decisions == 2
        assert s.confirmed == 1
        assert s.overridden == 1
        assert s.override_rate == 0.5

    def test_unit_corrections_are_tracked(self, journal):
        r = make_routed(unit=IUUnit.GENERAL_COUNSEL)
        decide(r, journal, action=AnalystAction.REASSIGN, to_unit=IUUnit.BOARD_OF_TRUSTEES)
        assert "GENERAL_COUNSEL -> BOARD_OF_TRUSTEES" in journal.stats().unit_corrections

    def test_training_pairs_capture_disagreements_only(self, journal):
        r = make_routed()
        decide(r, journal, action=AnalystAction.CONFIRM)
        decide(
            r,
            journal,
            action=AnalystAction.REASSIGN,
            to_unit=IUUnit.REGISTRAR,
            comment="Registrar owns credit application, not HR.",
        )

        pairs = journal.export_training_pairs()
        assert len(pairs) == 1
        assert pairs[0]["human_reasoning"].startswith("Registrar owns")

    def test_empty_journal_is_safe(self, journal):
        assert journal.stats().total_decisions == 0
