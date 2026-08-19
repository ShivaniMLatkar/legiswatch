"""Structured output contracts for the LegisWatch pipeline.

Every LLM call in this system returns one of these models. Nothing downstream
ever parses free text -- if the model cannot produce a valid instance, the call
is retried with the validation error fed back in (error-driven refinement).
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------
# Enums -- closed vocabularies keep routing deterministic and auditable.
# --------------------------------------------------------------------------


class ObligationType(str, Enum):
    """What kind of duty the statute creates."""

    REPORT = "report"  # must submit something to an external body
    POLICY = "policy"  # must adopt/maintain a policy or procedure
    PROCESS = "process"  # must run a recurring review or workflow
    DISCLOSURE = "disclosure"  # must publish or inform
    PROHIBITION = "prohibition"  # must not do something
    RECORDKEEPING = "recordkeeping"  # must retain/produce records


class Recurrence(str, Enum):
    ONE_TIME = "one_time"
    ANNUAL = "annual"
    BIENNIAL = "biennial"
    MULTI_YEAR = "multi_year"  # e.g. every five years
    CONTINUOUS = "continuous"  # ongoing standing duty


class RouteDecision(str, Enum):
    AUTO_FILE = "auto_file"  # high confidence + grounded -> straight to tracker
    HUMAN_REVIEW = "human_review"  # needs an analyst to confirm
    REJECTED = "rejected"  # failed groundedness -- never reaches a human queue


class IUUnit(str, Enum):
    """Closed set of IU offices an obligation can be routed to.

    Deliberately an enum rather than free text: an LLM inventing a plausible
    office name is the single most damaging failure mode in a compliance
    tracker, because it produces an obligation nobody owns.
    """

    GENERAL_COUNSEL = "Office of the Vice President and General Counsel"
    ACADEMIC_AFFAIRS = "Office of the Executive Vice President for University Academic Affairs"
    STUDENT_SUCCESS = "Office of the Vice President for Student Success, Enrollment and Institutional Effectiveness"
    ENROLLMENT_SERVICES = "University Enrollment Services"
    INSTITUTIONAL_ANALYTICS = "Institutional Analytics"
    ACCREDITATION = "University Effectiveness and Accreditation"
    HUMAN_RESOURCES = "IU Human Resources"
    BOARD_OF_TRUSTEES = "Board of Trustees"
    REGISTRAR = "Office of the University Registrar"
    STRATEGIC_INITIATIVES = "University Strategic Initiatives"
    UNASSIGNED = "Unassigned - needs triage"


def coerce_unit(v):
    """Accept either the enum NAME or its full value.

    Two callers send names rather than values, for good reasons: the extraction
    prompt offers the model short identifiers like ACADEMIC_AFFAIRS because a
    full office title is unwieldy in a menu, and the dashboard uses names as
    HTML option values. Rather than making each caller translate -- and getting
    it wrong somewhere -- the coercion lives here, once.
    """
    if isinstance(v, str) and v in IUUnit.__members__:
        return IUUnit[v].value
    return v


# --------------------------------------------------------------------------
# Stage 1: relevance triage
# --------------------------------------------------------------------------


class TriageResult(BaseModel):
    """Cheap first-pass filter. Runs on every document before extraction.

    Rationale: extraction is the expensive call. Most bills in a session are
    irrelevant to the university, so a cheap classifier in front of an
    expensive extractor is where the cost savings live.
    """

    is_relevant: bool = Field(
        description="True if this document creates or modifies a duty applying to "
        "Indiana University as a state educational institution."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(
        max_length=600,
        description="One or two sentences explaining the relevance call.",
    )
    matched_topics: list[str] = Field(
        default_factory=list,
        description="Short topic tags, e.g. 'faculty governance', 'transfer credit'.",
    )

    @field_validator("rationale")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rationale must not be empty")
        return v


# --------------------------------------------------------------------------
# Stage 2: obligation extraction
# --------------------------------------------------------------------------


class Obligation(BaseModel):
    """A single discrete duty the university must satisfy.

    `verbatim_quote` is the load-bearing field. It is checked programmatically
    against the source text -- see nodes/verify.py. An obligation whose quote
    is not present in the source is treated as a hallucination and dropped,
    regardless of how confident the model claims to be.
    """

    obligation_id: str = Field(
        description="Stable slug, e.g. 'IC-21-39.5-2-4-annual-complaint-report'."
    )
    summary: str = Field(
        max_length=300,
        description="Plain-language statement of what IU must do. No legalese.",
    )
    obligation_type: ObligationType
    recurrence: Recurrence

    verbatim_quote: str = Field(
        min_length=20,
        description="EXACT contiguous span copied from the source document that "
        "creates this duty. Must be character-for-character from the source.",
    )
    citation: str = Field(
        description="Statutory citation including subsection, e.g. 'IC 21-39.5-2-4(a)(5)'."
    )

    deadline_text: str | None = Field(
        default=None,
        description="Deadline exactly as written in the statute, e.g. 'not later than April 1, 2025'.",
    )
    first_due_date: date | None = Field(
        default=None,
        description="First concrete due date if one can be determined, else null.",
    )

    responsible_unit: IUUnit
    unit_rationale: str = Field(
        max_length=400,
        description="Why this office owns it. Cite the actor named in the statute.",
    )

    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("responsible_unit", mode="before")
    @classmethod
    def _accept_enum_name_or_value(cls, v):
        """Models reliably emit the short enum NAME (e.g. 'ACADEMIC_AFFAIRS') because
        that is what the prompt offers them; the schema's canonical form is the full
        office title. Accept either rather than burning a retry on a cosmetic mismatch.
        """
        return coerce_unit(v)

    @field_validator("verbatim_quote")
    @classmethod
    def _quote_looks_like_a_quote(cls, v: str) -> str:
        if len(v.split()) < 5:
            raise ValueError("verbatim_quote must be a substantive span (>= 5 words)")
        return v


class ExtractionResult(BaseModel):
    """All obligations found in one document."""

    obligations: list[Obligation] = Field(default_factory=list)
    extraction_notes: str | None = Field(
        default=None,
        max_length=600,
        description="Ambiguities worth flagging to a human reviewer.",
    )


# --------------------------------------------------------------------------
# Stage 3: verification (second-pass, adversarial)
# --------------------------------------------------------------------------


class VerificationVerdict(BaseModel):
    """Independent check of one extracted obligation.

    Run by a separate agent with a deliberately adversarial prompt: its job is
    to find the reason the extraction is wrong, not to agree with it.
    """

    obligation_id: str
    quote_is_grounded: bool = Field(
        description="Does verbatim_quote actually appear in the source document?"
    )
    summary_is_faithful: bool = Field(
        description="Does the summary accurately reflect the quoted text without "
        "adding duties the statute does not impose?"
    )
    deadline_is_correct: bool = Field(
        description="Is the extracted deadline supported by the source? True if no "
        "deadline was claimed."
    )
    unit_is_defensible: bool = Field(
        description="Is the responsible unit a reasonable reading of the actor named "
        "in the statute?"
    )
    verifier_confidence: float = Field(ge=0.0, le=1.0)
    objection: str | None = Field(
        default=None,
        max_length=500,
        description="If any check failed, the specific problem.",
    )

    @property
    def all_checks_passed(self) -> bool:
        return (
            self.quote_is_grounded
            and self.summary_is_faithful
            and self.deadline_is_correct
            and self.unit_is_defensible
        )


# --------------------------------------------------------------------------
# Stage 4: routing output
# --------------------------------------------------------------------------


class GroundednessCheck(BaseModel):
    """Deterministic, non-LLM check of quote provenance."""

    exact_match: bool
    fuzzy_score: float = Field(ge=0.0, le=100.0)
    passed: bool


class RoutedObligation(BaseModel):
    """An obligation after verification and routing -- the pipeline's unit of output."""

    obligation: Obligation
    verdict: VerificationVerdict | None = None
    groundedness: GroundednessCheck
    route: RouteDecision
    route_reason: str
    combined_confidence: float = Field(ge=0.0, le=1.0)
    source_doc_id: str
    source_url: str | None = None


class DocumentResult(BaseModel):
    """Everything the pipeline learned about one source document."""

    doc_id: str
    citation: str
    title: str
    source_url: str | None = None
    triage: TriageResult | None = None
    routed_obligations: list[RoutedObligation] = Field(default_factory=list)
    skipped_reason: str | None = None
    stage_timings_ms: dict = Field(default_factory=dict)


class RunSummary(BaseModel):
    """Top-level run artifact. This is what gets rendered to the dashboard."""

    run_id: str
    started_at: str
    finished_at: str
    provider: str
    model: str
    documents_ingested: int
    documents_relevant: int
    documents_skipped: int
    obligations_extracted: int
    obligations_auto_filed: int
    obligations_needing_review: int
    obligations_rejected: int
    total_latency_ms: float
    estimated_cost_usd: float
    results: list[DocumentResult] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 5: the human layer
#
# Everything above records what the machine did. Nothing above records what a
# person decided afterwards, which makes the audit trail one-sided: you can
# prove what the pipeline concluded and nothing about whether anyone agreed.
#
# These models close that. The governing rules:
#
#   * The journal is APPEND-ONLY. A reversal is a new record, never an edit to
#     an old one. That is the difference between an audit log and a status field.
#   * A comment is MANDATORY on every decision. A silent override tells you what
#     changed but not why, which destroys the only useful thing about the record.
#   * REJECTED is not terminal. Guardrails have false positives too -- a quote can
#     fail groundedness because the source was badly OCR'd, not because the model
#     invented it. An analyst must be able to reinstate.
# --------------------------------------------------------------------------


class AnalystAction(str, Enum):
    """What a human did to an obligation the pipeline produced."""

    CONFIRM = "confirm"  # agree with the agent; file it
    REINSTATE = "reinstate"  # reverse a REJECTED decision and push it back through
    REASSIGN = "reassign"  # wrong office -- change it and re-run routing
    AMEND = "amend"  # correct field(s), then re-run the checks
    REROUTE = "reroute"  # force a route; the human decision stands as-is
    REJECT = "reject"  # analyst rejects something the agent kept


class DecidedBy(str, Enum):
    """Provenance of an obligation's CURRENT state.

    The third value is the interesting one: the human corrected the input and
    the machine re-derived the routing from it. That is neither a pure machine
    decision nor a pure human override, and conflating it with either loses the
    thing you most want to know when auditing later.
    """

    AGENT = "agent"
    ANALYST = "analyst"
    ANALYST_CORRECTED_AGENT_ROUTED = "analyst_corrected_agent_routed"


class AnalystDecision(BaseModel):
    """One immutable entry in the decision journal."""

    decision_id: str
    timestamp: str
    run_id: str
    obligation_id: str
    reviewer: str = Field(min_length=2)

    action: AnalystAction
    comment: str = Field(
        min_length=10,
        max_length=2000,
        description="Why. Mandatory and substantive -- an override without a "
        "stated reason is unusable as a training signal and indefensible in an audit.",
    )

    from_route: RouteDecision
    to_route: RouteDecision
    from_unit: IUUnit | None = None
    to_unit: IUUnit | None = None

    _coerce_units = field_validator("from_unit", "to_unit", mode="before")(
        classmethod(lambda cls, v: coerce_unit(v))
    )

    field_edits: dict[str, str] = Field(
        default_factory=dict,
        description="Field name -> new value, for AMEND. Old values stay recoverable "
        "from the prior journal entry and the original run summary.",
    )

    reprocessed: bool = Field(
        default=False,
        description="True when the corrected obligation was pushed back through "
        "groundedness and routing rather than the route being forced by hand.",
    )
    reprocess_trace: str | None = Field(
        default=None,
        description="What the re-run concluded, in one line.",
    )

    decided_by: DecidedBy
    supersedes: str | None = Field(
        default=None, description="decision_id this entry reverses, if any."
    )

    @field_validator("comment")
    @classmethod
    def _comment_is_substantive(cls, v: str) -> str:
        if len(v.strip().split()) < 3:
            raise ValueError("comment must be a real sentence, not a placeholder")
        return v


class ObligationLifecycle(BaseModel):
    """Assembled view: what the agent decided, then everything humans did to it.

    Built by replaying the journal over the original routed obligation. Nothing
    here is stored -- it is derived, so the journal stays the single source of truth.
    """

    obligation_id: str
    agent_decision: RoutedObligation
    analyst_decisions: list[AnalystDecision] = Field(default_factory=list)

    current_route: RouteDecision
    current_unit: IUUnit
    current_obligation: Obligation
    decided_by: DecidedBy
    is_overridden: bool

    @property
    def review_count(self) -> int:
        return len(self.analyst_decisions)


class OverrideStats(BaseModel):
    """Agreement between the pipeline and the people reviewing it.

    This is the payoff for keeping the journal. Override rate by route tells you
    whether the auto-file threshold is set right; override rate by office tells
    you whether the routing map is wrong for a specific unit; the reinstatement
    count tells you whether the guardrails are too aggressive.
    """

    total_decisions: int
    obligations_reviewed: int
    confirmed: int
    overridden: int
    reinstated: int
    override_rate: float = Field(ge=0.0, le=1.0)
    by_action: dict[str, int] = Field(default_factory=dict)
    by_original_route: dict[str, int] = Field(default_factory=dict)
    unit_corrections: dict[str, int] = Field(default_factory=dict)
    reviewers: dict[str, int] = Field(default_factory=dict)
