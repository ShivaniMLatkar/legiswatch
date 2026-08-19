#!/usr/bin/env python3
"""Regenerate the recorded response fixtures used by the `replay` provider.

The fixtures hold one recorded response per model call the pipeline makes, keyed
by the same cache tags the live path uses. With `--provider replay` these are
served instead of calling an API, which makes runs deterministic and the test
suite hermetic.

Provenance
----------
These responses were produced by a language model reading the source documents
in `data/corpus/corpus.json`, then reviewed by hand. They are representative of
live output rather than a transcript of one specific run. Regenerate them
against a live provider with:

    legiswatch-run --provider anthropic --record

Fixture design
--------------
Two of the recorded extractions are deliberately incorrect, so that both
guardrail layers are exercised on every run rather than only when a live model
happens to err. Each reproduces a failure mode observed in practice:

  1. `IC-21-39.5-2-4-quarterly-complaint-dashboard` -- a fluent, plausible
     obligation whose quote does not occur in the source. Caught by the
     deterministic groundedness check before any verification call is made.

  2. `CHE-HEA1001-preliminary-list-response` -- an accurate quote whose summary
     silently introduces a deadline the source does not state. Exact matching
     accepts it; the adversarial verifier rejects it.

Retaining both keeps the guardrail behaviour continuously covered and gives the
test suite deterministic failure cases to assert against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPLAY_DIR = ROOT / "data" / "replay"


def w(tag: str, schema: str, response: Dict[str, Any]) -> None:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    (REPLAY_DIR / f"{tag}.json").write_text(
        json.dumps(
            {"key": tag, "schema": schema, "model": "claude-sonnet-4-5", "response": response},
            indent=2,
        )
    )


def ob(
    oid: str,
    summary: str,
    otype: str,
    recurrence: str,
    quote: str,
    citation: str,
    unit: str,
    unit_rationale: str,
    confidence: float,
    deadline_text: str | None = None,
    first_due_date: str | None = None,
) -> Dict[str, Any]:
    return {
        "obligation_id": oid,
        "summary": summary,
        "obligation_type": otype,
        "recurrence": recurrence,
        "verbatim_quote": quote,
        "citation": citation,
        "deadline_text": deadline_text,
        "first_due_date": first_due_date,
        "responsible_unit": unit,
        "unit_rationale": unit_rationale,
        "confidence": confidence,
    }


def verdict(
    oid: str,
    conf: float,
    *,
    grounded: bool = True,
    faithful: bool = True,
    deadline_ok: bool = True,
    unit_ok: bool = True,
    objection: str | None = None,
) -> Dict[str, Any]:
    return {
        "obligation_id": oid,
        "quote_is_grounded": grounded,
        "summary_is_faithful": faithful,
        "deadline_is_correct": deadline_ok,
        "unit_is_defensible": unit_ok,
        "verifier_confidence": conf,
        "objection": objection,
    }


# ==========================================================================
# TRIAGE
# ==========================================================================

TRIAGE = {
    "IC-21-18-9-10.7": (True, 0.97, "Imposes a direct duty on state educational institutions to request commission approval before continuing degree programs that fall below statutory graduate thresholds, and to eliminate programs where approval is refused.", ["degree programs", "academic program review", "commission approval"]),
    "IC-21-39.5-2-2": (True, 0.98, "Directs the board of trustees of an institution to conduct five-year post-tenure reviews against enumerated criteria, adopt a disciplinary policy, and submit its process to the commission for higher education.", ["faculty governance", "post-tenure review", "board of trustees"]),
    "IC-21-39.5-2-3": (True, 0.94, "Requires the institution to give substantial consideration to specified criteria before renewing faculty contracts, awarding bonuses, or completing performance assessments.", ["faculty governance", "employment review"]),
    "IC-21-39.5-2-4": (True, 0.99, "Creates a cluster of institutional duties: establish and publicise a complaint procedure, refer and surface complaints, and file an annual report with the commission for higher education.", ["complaint procedure", "annual reporting", "faculty governance"]),
    "IC-21-39.5-3-2": (True, 0.93, "Requires the institution to include specified free-expression and intellectual-diversity content in new student programming.", ["new student programming", "student orientation"]),
    "IC-21-42-3-5": (True, 0.96, "Governs how a receiving state educational institution must treat the Indiana college core and award transfer credit, constraining registrar practice directly.", ["transfer credit", "Indiana college core", "registrar"]),
    "IC-21-42-5-4": (True, 0.95, "Names Indiana University explicitly and requires it to identify transfer equivalents that apply consistently across all regional campuses in the system.", ["transfer credit", "core transfer library", "regional campuses"]),
    "CHE-HEA1001-GUIDANCE": (True, 0.97, "Commission guidance implementing IC 21-18-9-10.7; specifies the metrics, narrative submissions, and implementation timeline institutions must meet.", ["degree programs", "commission submission", "program thresholds"]),
    "NEG-CONTROL-01": (False, 0.96, "Concerns registration of agricultural implements with the bureau of motor vehicles and county highway mileage reporting. No duty attaches to a state educational institution.", []),
    "NEG-CONTROL-02": (False, 0.97, "Concerns rate schedule filings by municipally owned water utilities before the utility regulatory commission. No institutional duty.", []),
}

for doc_id, (rel, conf, rationale, topics) in TRIAGE.items():
    w(
        f"triage__{doc_id}",
        "TriageResult",
        {"is_relevant": rel, "confidence": conf, "rationale": rationale, "matched_topics": topics},
    )


# ==========================================================================
# EXTRACTION
# ==========================================================================

EXTRACTIONS: Dict[str, Dict[str, Any]] = {}

# ---- IC 21-18-9-10.7 -----------------------------------------------------
EXTRACTIONS["IC-21-18-9-10.7"] = {
    "obligations": [
        ob(
            "IC-21-18-9-10.7-request-approval-under-threshold",
            "Request Commission for Higher Education approval to continue any degree program whose three-year average graduate count falls below the statutory threshold for its award level.",
            "process", "continuous",
            "the state educational institution must request approval from the commission to continue the degree program",
            "IC 21-18-9-10.7(a)",
            "ACADEMIC_AFFAIRS",
            "The statute places the duty on the 'state educational institution' and concerns continuation of academic degree programs, which sits with academic affairs.",
            0.93,
        ),
        ob(
            "IC-21-18-9-10.7-eliminate-unapproved-programs",
            "Eliminate any under-threshold degree program, and the costs associated with it, where the Commission declines to approve continuation.",
            "process", "continuous",
            "If the commission does not grant approval under subsection (a), the state educational institution must eliminate",
            "IC 21-18-9-10.7(b)",
            "ACADEMIC_AFFAIRS",
            "Duty falls on the institution and concerns program discontinuation, an academic affairs function.",
            0.88,
        ),
        ob(
            "IC-21-18-9-10.7-monitor-graduate-averages",
            "Maintain three-year rolling average graduate counts for every degree program so threshold status can be determined.",
            "recordkeeping", "annual",
            "average number of students who graduate over the immediately preceding three (3) years is fewer than",
            "IC 21-18-9-10.7(a)(1)",
            "INSTITUTIONAL_ANALYTICS",
            "The threshold test depends on graduate counts, which are produced and certified by institutional analytics.",
            0.79,
        ),
    ],
    "extraction_notes": "The monitoring duty in the third obligation is implied by the threshold test rather than stated as an express command; flagged for reviewer judgement.",
}

# ---- IC 21-39.5-2-2 ------------------------------------------------------
EXTRACTIONS["IC-21-39.5-2-2"] = {
    "obligations": [
        ob(
            "IC-21-39.5-2-2-five-year-post-tenure-review",
            "Review each tenured faculty member against the five statutory criteria within five years of tenure and every five years thereafter.",
            "process", "multi_year",
            "Not later than five (5) years after the date that a faculty member is granted tenure by an institution and not later than every five (5) years thereafter, the board of trustees of an institution shall review and determine whether the faculty member has met the following criteria",
            "IC 21-39.5-2-2(a)",
            "BOARD_OF_TRUSTEES",
            "The statute names 'the board of trustees of an institution' as the actor.",
            0.95,
            deadline_text="Not later than five (5) years after tenure is granted, and every five (5) years thereafter",
        ),
        ob(
            "IC-21-39.5-2-2-certify-determination",
            "Certify the determination when the board finds a faculty member has met the criteria.",
            "recordkeeping", "continuous",
            "If the board determines a faculty member meets the criteria, the board shall certify that determination.",
            "IC 21-39.5-2-2(c)",
            "BOARD_OF_TRUSTEES",
            "Statute names the board as the certifying actor.",
            0.90,
        ),
        ob(
            "IC-21-39.5-2-2-adopt-disciplinary-policy",
            "Adopt a disciplinary policy specifying the actions available when a tenured faculty member fails to meet one or more criteria.",
            "policy", "one_time",
            "The institution shall adopt a disciplinary policy establishing actions including termination, demotion, salary reduction, or other disciplinary action",
            "IC 21-39.5-2-2(e)",
            "HUMAN_RESOURCES",
            "Duty is on 'the institution' and concerns employment discipline, which HR owns in partnership with academic affairs.",
            0.91,
        ),
        ob(
            "IC-21-39.5-2-2-review-renew-process-five-years",
            "Review and renew or amend the post-tenure review process and any board-established criteria at least every five years.",
            "process", "multi_year",
            "The board shall, at least every five (5) years, review and renew or amend the review process and any criteria established under subsection (a)(5).",
            "IC 21-39.5-2-2(f)",
            "BOARD_OF_TRUSTEES",
            "Statute names the board.",
            0.92,
            deadline_text="at least every five (5) years",
        ),
        ob(
            "IC-21-39.5-2-2-submit-process-to-che",
            "Submit the post-tenure review process and criteria to the Commission for Higher Education each time they are reviewed, renewed, or amended.",
            "report", "continuous",
            "The board shall submit to the commission for higher education the process and criteria each time they are reviewed, renewed, or amended.",
            "IC 21-39.5-2-2(g)",
            "BOARD_OF_TRUSTEES",
            "Statute names the board as the submitting party; the trigger is any revision of the process.",
            0.93,
        ),
    ],
    "extraction_notes": None,
}

# ---- IC 21-39.5-2-3 ------------------------------------------------------
EXTRACTIONS["IC-21-39.5-2-3"] = {
    "obligations": [
        ob(
            "IC-21-39.5-2-3-substantial-consideration-in-reviews",
            "Give substantial consideration to the section 2(a)(1)-(5) criteria before renewing a faculty contract, making a bonus decision, or completing a performance assessment.",
            "process", "continuous",
            "the institution must provide substantial consideration to performance regarding criteria in section 2(a)(1) through 2(a)(5) of this chapter",
            "IC 21-39.5-2-3(b)",
            "HUMAN_RESOURCES",
            "Duty attaches to the institution at the point of contract renewal, bonus, and performance assessment, all HR-administered processes.",
            0.89,
        ),
    ],
    "extraction_notes": None,
}

# ---- IC 21-39.5-2-4 ------------------------------------------------------
EXTRACTIONS["IC-21-39.5-2-4"] = {
    "obligations": [
        ob(
            "IC-21-39.5-2-4-establish-complaint-procedure",
            "Establish a procedure allowing students and employees to submit complaints that a faculty member is not meeting the statutory criteria.",
            "policy", "one_time",
            "Establish a procedure that allows both students and employees to submit complaints that a faculty member or person described in section 3(a) of this chapter is not meeting the criteria described in section 2(a)(1) through 2(a)(5) of this chapter.",
            "IC 21-39.5-2-4(a)(1)",
            "HUMAN_RESOURCES",
            "Complaint intake against employees is an HR-owned process at IU.",
            0.94,
        ),
        ob(
            "IC-21-39.5-2-4-publicize-procedure",
            "Publicise the complaint procedure at student orientations, on the institution's website, and during employee onboarding.",
            "disclosure", "continuous",
            "Provide information regarding the procedure established under subdivision (1): (A) at student orientations; (B) on the institution's website; and (C) during employee onboarding programs.",
            "IC 21-39.5-2-4(a)(2)",
            "STUDENT_SUCCESS",
            "Two of the three required channels are student orientation and the institutional website, both owned within student success.",
            0.87,
        ),
        ob(
            "IC-21-39.5-2-4-refer-complaints-to-hr",
            "Refer submitted complaints to appropriate HR professionals and supervisors for consideration in employee reviews and tenure and promotion decisions.",
            "process", "continuous",
            "Refer complaints submitted under subdivision (1) to appropriate human resource professionals and supervisors for consideration in employee reviews and tenure and promotion decisions.",
            "IC 21-39.5-2-4(a)(3)",
            "HUMAN_RESOURCES",
            "The statute names human resource professionals as the recipients.",
            0.92,
        ),
        ob(
            "IC-21-39.5-2-4-make-complaints-available-to-trustees",
            "Make submitted complaints and any relevant documents, summaries, or investigations available to the board of trustees.",
            "disclosure", "continuous",
            "Make complaints submitted under subdivision (1) and any relevant documents, summaries, or investigations available to the board of trustees of the institution.",
            "IC 21-39.5-2-4(a)(4)",
            "GENERAL_COUNSEL",
            "Production of investigation materials to the board is typically routed through counsel.",
            0.83,
        ),
        ob(
            "IC-21-39.5-2-4-annual-complaint-report-to-che",
            "File an annual report with the Commission for Higher Education summarising the complaint procedure, how it was publicised, and complaint counts disaggregated by type.",
            "report", "annual",
            "Submit a report to the commission for higher education not later than April 1, 2025, and not later than April 1 each year thereafter",
            "IC 21-39.5-2-4(a)(5)",
            "ACCREDITATION",
            "External compliance reporting to the Commission is coordinated through University Effectiveness and Accreditation.",
            0.96,
            deadline_text="not later than April 1, 2025, and not later than April 1 each year thereafter",
            first_due_date="2025-04-01",
        ),
        # --- Fixture: fabricated obligation, quote absent from source ---
        ob(
            "IC-21-39.5-2-4-quarterly-complaint-dashboard",
            "Publish a quarterly public dashboard of complaint volumes disaggregated by campus.",
            "disclosure", "annual",
            "the institution shall publish a quarterly dashboard summarizing complaint volumes disaggregated by campus",
            "IC 21-39.5-2-4(a)(6)",
            "INSTITUTIONAL_ANALYTICS",
            "Public dashboards of institutional metrics are produced by institutional analytics.",
            0.71,
        ),
        ob(
            "IC-21-39.5-2-4-anonymize-complaint-data",
            "Exclude identifying information about complainants and about faculty members who are the subject of complaints from any institutional or commission report.",
            "prohibition", "continuous",
            "An institution and the commission may not include identifying information regarding: (1) complainants; or (2) faculty members against whom complaints were submitted.",
            "IC 21-39.5-2-4(c)",
            "GENERAL_COUNSEL",
            "A statutory prohibition on disclosure of identifying information is a legal compliance constraint.",
            0.95,
        ),
    ],
    "extraction_notes": "Subsection (b) places reporting duties on the commission for higher education rather than on the institution; excluded as out of scope.",
}

# ---- IC 21-39.5-3-2 ------------------------------------------------------
EXTRACTIONS["IC-21-39.5-3-2"] = {
    "obligations": [
        ob(
            "IC-21-39.5-3-2-new-student-programming-content",
            "Include content on free inquiry, free expression, intellectual diversity, and appropriate responses to offensive speech in programming for new students.",
            "disclosure", "continuous",
            "An institution shall include the following information in the institution's programming for new students",
            "IC 21-39.5-3-2",
            "STUDENT_SUCCESS",
            "New student programming is delivered by student success and orientation units.",
            0.91,
        ),
    ],
    "extraction_notes": None,
}

# ---- IC 21-42-3-5 --------------------------------------------------------
EXTRACTIONS["IC-21-42-3-5"] = {
    "obligations": [
        ob(
            "IC-21-42-3-5-honor-completed-college-core",
            "Do not require a transfer student who has completed the Indiana college core to take additional college core courses, regardless of associate degree status or delivery method.",
            "prohibition", "continuous",
            "may not be required to complete additional courses in the Indiana college core at the state educational institution to which the individual transfers",
            "IC 21-42-3-5(a)",
            "REGISTRAR",
            "Application of transfer credit and core completion is administered by the registrar.",
            0.90,
        ),
        ob(
            "IC-21-42-3-5-award-credit-per-core-transfer-library",
            "Award credit for satisfactorily completed courses based on the course-to-course equivalencies in the core transfer library.",
            "process", "continuous",
            "The state educational institution to which the individual has transferred shall award credit to the individual for courses the individual has satisfactorily completed, based on the course to course equivalencies of the core transfer library established under IC 21-42-5.",
            "IC 21-42-3-5(b)",
            "REGISTRAR",
            "Statute places the duty on the receiving institution; credit award is a registrar function.",
            0.93,
        ),
        ob(
            "IC-21-42-3-5-thirty-credit-gen-ed-associate",
            "Treat an admitted holder of a Commission-approved associate of arts or science degree as having met at least 30 semester credit hours of general education requirements.",
            "process", "continuous",
            "is considered to have met at least thirty (30) semester credit hours of the state educational institution's general education requirement",
            "IC 21-42-3-5(c)",
            "REGISTRAR",
            "General education requirement satisfaction is recorded by the registrar.",
            0.88,
        ),
    ],
    "extraction_notes": None,
}

# ---- IC 21-42-5-4 --------------------------------------------------------
EXTRACTIONS["IC-21-42-5-4"] = {
    "obligations": [
        ob(
            "IC-21-42-5-4-identify-transfer-equivalents-across-regional-campuses",
            "Identify transfer equivalents so that a core transfer library course accepted by one IU regional campus is accepted by all other regional campuses offering the same equivalent course.",
            "process", "continuous",
            "Indiana University and Purdue University must identify transfer equivalents so that a course accepted by one (1) regional campus is accepted by all other regional campuses that offer the same transfer equivalent course",
            "IC 21-42-5-4(4)",
            "REGISTRAR",
            "The statute names Indiana University directly; equivalency identification is a registrar function.",
            0.94,
        ),
        ob(
            "IC-21-42-5-4-system-wide-course-equivalence",
            "Ensure equivalent courses, including those sharing a course number and title, count identically at every campus in the IU system where offered.",
            "process", "continuous",
            "Within the Indiana University system and Purdue University system, equivalent courses, including courses with the same course number and title, must count in the same way at all campuses within the system where the course is offered.",
            "IC 21-42-5-4(5)",
            "REGISTRAR",
            "Statute names the Indiana University system; course application rules are registrar-administered.",
            0.92,
        ),
    ],
    "extraction_notes": None,
}

# ---- CHE guidance --------------------------------------------------------
EXTRACTIONS["CHE-HEA1001-GUIDANCE"] = {
    "obligations": [
        ob(
            "CHE-HEA1001-submit-program-metrics",
            "Submit to the Commission the Replacement Ratio, Credit Hour Production Ratio, Viability Index, three-year graduate averages, a 250-word Program Overview, and a 500-word Rationale to Continue.",
            "report", "one_time",
            "Institutions must provide to the Commission the Replacement Ratio for all degree programs, the Credit Hour Production Ratio for undergraduate programs only, the Viability Index for undergraduate programs, three-year average graduate counts for under-threshold programs, a Program Overview limited to 250 words, and a Rationale to Continue limited to 500 words.",
            "CHE Guidance, HEA 1001-2025",
            "INSTITUTIONAL_ANALYTICS",
            "The required submissions are ratio and index metrics computed from institutional data.",
            0.93,
        ),
        ob(
            "CHE-HEA1001-implement-suspensions-before-ay2627",
            "Implement any approved program suspensions or mergers before academic year 2026-2027.",
            "process", "one_time",
            "Any approved suspensions or mergers must be implemented before academic year 2026-2027.",
            "CHE Guidance, HEA 1001-2025",
            "ACADEMIC_AFFAIRS",
            "Program suspension and merger execution is an academic affairs responsibility.",
            0.86,
            deadline_text="before academic year 2026-2027",
        ),
        # --- Fixture: accurate quote, summary introduces a false deadline ---
        ob(
            "CHE-HEA1001-preliminary-list-response",
            "Respond to the Commission's preliminary under-threshold program list within 30 days of receipt.",
            "report", "one_time",
            "In mid-to-late July the Commission will provide institutions a preliminary list of under-threshold programs.",
            "CHE Guidance, HEA 1001-2025",
            "ACADEMIC_AFFAIRS",
            "Institutional response to the preliminary list would be coordinated by academic affairs.",
            0.68,
            deadline_text="within 30 days of receipt",
        ),
    ],
    "extraction_notes": "The guidance describes a Commission action in mid-to-late July; whether it creates a responsive institutional deadline is not stated.",
}

for doc_id, payload in EXTRACTIONS.items():
    w(f"extract__{doc_id}", "ExtractionResult", payload)


# ==========================================================================
# VERIFICATION
# ==========================================================================

VERDICTS: List[Dict[str, Any]] = [
    verdict("IC-21-18-9-10.7-request-approval-under-threshold", 0.93),
    verdict("IC-21-18-9-10.7-eliminate-unapproved-programs", 0.90),
    verdict(
        "IC-21-18-9-10.7-monitor-graduate-averages", 0.62,
        objection="The quoted span states the threshold test, not an express duty to maintain rolling averages. "
                  "The monitoring duty is a reasonable operational inference but is not commanded by this text.",
    ),
    verdict("IC-21-39.5-2-2-five-year-post-tenure-review", 0.96),
    verdict("IC-21-39.5-2-2-certify-determination", 0.93),
    verdict("IC-21-39.5-2-2-adopt-disciplinary-policy", 0.92),
    verdict("IC-21-39.5-2-2-review-renew-process-five-years", 0.94),
    verdict("IC-21-39.5-2-2-submit-process-to-che", 0.94),
    verdict("IC-21-39.5-2-3-substantial-consideration-in-reviews", 0.88),
    verdict("IC-21-39.5-2-4-establish-complaint-procedure", 0.95),
    verdict("IC-21-39.5-2-4-publicize-procedure", 0.89),
    verdict("IC-21-39.5-2-4-refer-complaints-to-hr", 0.93),
    verdict(
        "IC-21-39.5-2-4-make-complaints-available-to-trustees", 0.71,
        unit_ok=False,
        objection="The statute directs materials to the board of trustees and names no intermediary. "
                  "Assigning ownership to General Counsel is plausible institutional practice but is not "
                  "supported by the text; BOARD_OF_TRUSTEES or UNASSIGNED is better grounded.",
    ),
    verdict("IC-21-39.5-2-4-annual-complaint-report-to-che", 0.97),
    verdict("IC-21-39.5-2-4-anonymize-complaint-data", 0.96),
    verdict("IC-21-39.5-3-2-new-student-programming-content", 0.93),
    verdict("IC-21-42-3-5-honor-completed-college-core", 0.91),
    verdict("IC-21-42-3-5-award-credit-per-core-transfer-library", 0.94),
    verdict("IC-21-42-3-5-thirty-credit-gen-ed-associate", 0.90),
    verdict("IC-21-42-5-4-identify-transfer-equivalents-across-regional-campuses", 0.95),
    verdict("IC-21-42-5-4-system-wide-course-equivalence", 0.93),
    verdict("CHE-HEA1001-submit-program-metrics", 0.94),
    verdict("CHE-HEA1001-implement-suspensions-before-ay2627", 0.88),
    verdict(
        "CHE-HEA1001-preliminary-list-response", 0.35,
        faithful=False, deadline_ok=False,
        objection="The source says only that the Commission will provide a preliminary list in mid-to-late July. "
                  "It states no institutional response window. The '30 days of receipt' deadline in the summary "
                  "and deadline_text appears nowhere in the source text.",
    ),
]

for v in VERDICTS:
    w(f"verify__{v['obligation_id']}", "VerificationVerdict", v)


print(f"Wrote {len(list(REPLAY_DIR.glob('*.json')))} cached responses to {REPLAY_DIR}")
