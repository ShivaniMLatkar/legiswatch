# LegisWatch

**Turning legal text into a compliance register an auditor can defend — with guardrails that assume the model is wrong.**

[![CI](https://github.com/ShivaniMLatkar/legiswatch/actions/workflows/ci.yml/badge.svg)](https://github.com/ShivaniMLatkar/legiswatch/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

LegisWatch reads legislative and regulatory documents and produces a register of
discrete institutional obligations. Each entry carries its deadline, its owning
office, and the exact statutory language it was drawn from.

The extraction is the easy part. This project is mostly about the other question:
**what happens when the model is wrong, and how would anyone know?**

---

## The problem class

LLM document extraction is straightforward to prototype and difficult to deploy
anywhere the output has consequences. Three failure modes recur:

| Failure | Why it is dangerous |
|---|---|
| **Fabrication** | A fluent, plausible obligation the source never states. Reads exactly like a correct one. |
| **Silent drift** | An accurate quote whose summary adds a deadline, broadens a duty, or changes who must act. |
| **Misattribution** | A real obligation assigned to an office that does not own it. Worse than a missing entry, because the register *looks* complete and nobody is actually accountable. |

A register containing any of these is worse than no register, because it
manufactures false confidence. LegisWatch is built around catching them.

## Architecture

```
                    legal document
                          │
                    ┌─────▼─────┐
                    │  triage   │  cheap relevance classifier
                    └─────┬─────┘
              not relevant │ relevant
              ◄────────────┤
                    ┌─────▼─────┐
                    │  extract  │  decompose into discrete duties,
                    └─────┬─────┘  each carrying a verbatim quote
                          │
        ╔═════════════════▼═════════════════╗
        ║  GUARDRAIL 1 · groundedness       ║  deterministic string match
        ║  no model call                    ║  quote absent → REJECT
        ╚═════════════════╤═════════════════╝
                          │ grounded
        ╔═════════════════▼═════════════════╗
        ║  GUARDRAIL 2 · adversarial verify ║  independent model instance,
        ║  prompted to refute               ║  prompted to find the error
        ╚═════════════════╤═════════════════╝
                          │
                    ┌─────▼─────┐
                    │   route   │  confidence gate + policy gate
                    └─────┬─────┘  no model call
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
    AUTO-FILE        HUMAN REVIEW        REJECTED
        └─────────────────┼─────────────────┘
                          ▼
             append-only decision journal
             (reversible, attributed, mandatory reason)
```

Implemented as a LangGraph state machine. Conditional edges short-circuit the
expensive stages: irrelevant documents never reach extraction, and documents
yielding nothing never reach verification.

## The guardrails

### 1. Groundedness — deterministic, no model involved

Every extracted obligation must carry a `verbatim_quote`. That quote is verified
by normalising both sides and testing containment, falling back to a fuzzy
partial ratio with a configurable floor (default 92.0).

```python
normalize(t) = collapse_whitespace(fold_unicode_punctuation(t)).strip().lower()

quote in source              → exact match, accepted
fuzz.partial_ratio >= 92.0   → accepted (absorbs typesetting drift)
otherwise                    → rejected as ungrounded
```

Normalisation folds curly quotes, dashes and non-breaking spaces onto ASCII,
because models routinely re-typeset punctuation. Wording is left strictly alone,
so a paraphrase fails.

**Why no model here.** "Did the source actually say this" is the highest-stakes
question in the system. Asking a language model to grade its own output on it
produces an answer that cannot be reproduced, cannot be explained to an auditor,
and fails in exactly the correlated way the original extraction failed. String
matching is deterministic, costs nothing, and returns the same verdict every
time anyone re-runs it.

### 2. Adversarial verification — semantic, model-based

Runs only on quotes that survive stage 1, so a fabricated quote never costs a
second inference call. A separate model instance is prompted to *refute* rather
than confirm, and returns four independent judgements:

| Check | Fails when |
|---|---|
| `quote_is_grounded` | the span is not in the source |
| `summary_is_faithful` | the summary broadens the duty, adds a deadline, or changes the actor |
| `deadline_is_correct` | a deadline is claimed that the source does not state |
| `unit_is_defensible` | the assigned office contradicts the actor the statute names |

### Why both layers are necessary

Neither catches what the other does. Two fixtures in the recorded response set
demonstrate this on every run:

- **Fabricated obligation** — *"Publish a quarterly public dashboard of complaint
  volumes disaggregated by campus."* Fluent and plausible; appears nowhere in the
  cited section. Groundedness scores it 52.4 against a 92.0 floor and rejects it
  before any verification call is made.
- **False deadline** — the model quotes a guidance document correctly, then
  summarises it as a duty to respond *"within 30 days of receipt."* The quote is
  genuine, so exact matching accepts it. Only the adversarial verifier catches it.

## Results

Evaluated against a hand-labelled gold set of 24 obligations across 10 source
documents (7 verbatim Indiana Code sections, 1 agency guidance document,
2 negative controls).

| Stage | Precision | Recall | F1 |
|---|---|---|---|
| Raw model output | 92.0% | 95.8% | 93.9% |
| **After guardrails** | **100.0%** | **95.8%** | **97.9%** |

- **+8.0 precision points** attributable to the guardrail layer
- **Zero true obligations lost** — the filter costs no recall
- **0 of 16** auto-filed entries incorrect
- **100%** of ungrounded quotes intercepted
- Triage: 100% precision and recall (2 negative controls correctly excluded)

```bash
legiswatch-eval    # reproduces every figure above
```

The evaluator **exits non-zero if any obligation was auto-filed that the gold set
marks incorrect**, which makes it usable as a CI gate rather than a report. A
prompt or threshold change that degrades safety fails the build.

> **On scale.** 10 documents and 24 gold obligations is a small corpus. The
> methodology — an independently-built gold set, precision measured before *and*
> after the guardrails, and an explicit auto-file safety number — is the
> transferable part. The percentages are honest for this corpus and should be
> re-measured on yours.

## Routing

Two decisions, deliberately separated:

**Ownership** (`responsible_unit`) is chosen by the model from a **closed enum**,
never free text. A model inventing a plausible-sounding office produces an
obligation nobody owns inside a register that looks complete. When the correct
owner is genuinely unclear the model returns `UNASSIGNED`, which routes to a
human by rule.

**Disposition** (`route`) is decided by **pure code** — an ordered chain with no
model call, readable end to end by a non-engineer:

```
REJECT   quote ungrounded │ verifier rejects provenance │ summary unfaithful
REVIEW   no semantic verdict │ owner unassigned │ type ∈ {report, prohibition}
         │ any verifier check failed │ verifier confidence < 0.75
         │ combined confidence < 0.85
AUTO     everything else
```

Combined confidence is a weighted blend:

```
0.25 · extractor  +  0.45 · verifier  +  0.30 · groundedness
```

The extractor's assessment of its own output carries the least weight, because a
model's confidence in its own work is the least reliable of the three signals.

**Reports and prohibitions never auto-file at any confidence.** That is a
consequence judgement rather than a confidence judgement: a missed external
report is a regulatory finding, and a missed prohibition is legal exposure. A
parametrised test asserts this holds even at confidence 1.0 with a clean verdict.

## Human review and the audit trail

A system that records what the model decided and nothing about what a person
decided afterwards has a one-sided audit trail. Every reviewer action is written
to an **append-only journal**.

**Append-only.** A reversal is a new record pointing at the one it supersedes,
never an edit. A mutable status field tells you where something ended up; it
cannot tell you that one reviewer rejected an item in March, another reinstated
it in April, and why either did.

**Reasons are mandatory.** Enforced by schema on every action, including plain
confirmations. An override without a stated reason is unusable as a signal and
indefensible under audit.

**Rejection is reversible.** Guardrails have false positives too — a quote can
fail groundedness because a source PDF was badly OCR'd, not because the model
invented it. But reinstating **re-runs the checks rather than bypassing them**: a
corrected quote that still is not in the source is rejected again. Reviewers can
be wrong as well.

### Override versus amend-and-reprocess

| | Override | Amend and reprocess |
|---|---|---|
| Actions | `reroute`, `reject`, `confirm` | `reassign`, `amend`, `reinstate` |
| What changes | the disposition, set by hand | the **input**; the system re-derives the disposition |
| Provenance | `analyst` | `analyst_corrected_agent_routed` |
| Guarantees | model opinion preserved in the journal | deterministic checks still hold |

Amend-and-reprocess is preferred because the guarantees survive it.

One subtlety worth naming: editing text a verdict judged (summary, quote,
deadline) invalidates that verdict, since it describes text that no longer
exists. The verdict is discarded — and because a discarded verdict would
otherwise *skip* every verdict-based gate, an explicit rule treats an absent
verdict as an unmet requirement rather than a waived one. Corrected obligations
route to human review, never straight to auto-file.

### Feedback loop

Every override is a labelled disagreement between the system and a domain expert:

```bash
legiswatch-review stats            # override rate, unit corrections, reinstatements
curl localhost:8000/audit/training-pairs   # disagreements as a regression set
```

- override rate on auto-filed items → whether the auto-file threshold is correct
- repeated corrections for one office → where the routing map is wrong
- reinstatement frequency → whether the guardrails are too aggressive

## Installation

```bash
git clone https://github.com/ShivaniMLatkar/legiswatch.git
cd legiswatch
pip install -e ".[dev,service,connectors]"
```

Requires Python 3.10+.

## Usage

```bash
# Recorded fixtures — deterministic, no API key, no network
legiswatch-run

# Live inference
legiswatch-run --provider anthropic
legiswatch-run --provider openai

# Fully local; no data leaves the environment
legiswatch-run --provider ollama --model llama3.1:8b

# Refresh the recorded fixtures from a live run
legiswatch-run --provider anthropic --record

# Evaluate, test, review
legiswatch-eval
pytest
legiswatch-review list --route human_review
legiswatch-review show <obligation-id>
legiswatch-review decide <obligation-id> --action reinstate \
    --reviewer j.doe --comment "Quote verified against source." \
    --set summary="Corrected statement of the duty."

# HTTP service
uvicorn legiswatch.api:app --reload
```

`make help` lists the equivalent Make targets.
`bash scripts/review_walkthrough.sh` exercises the review layer end to end.

Outputs are written to `out/`: `dashboard.html` (a self-contained reviewer queue
that opens in any browser, no build step or CDN), `run_summary.json`,
`eval_report.json`, and `decision_journal.jsonl`.

### The replay provider

`--provider replay` serves recorded responses keyed by the same cache tags the
live path uses. It exists so runs are deterministic, the test suite is hermetic,
and CI costs nothing — the same reason any system with a paid external dependency
records fixtures.

## HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /extract` | Run one ad-hoc document through the full pipeline |
| `POST /runs` | Execute the configured corpus |
| `GET /obligations` | Query results by disposition, office, confidence |
| `POST /obligations/{id}/decisions` | Record a reviewer decision |
| `GET /obligations/{id}/history` | Full lifecycle: model decision, then every human action |
| `GET /audit` | Complete decision journal, filterable |
| `GET /audit/stats` | Model/reviewer agreement metrics |
| `GET /audit/training-pairs` | Disagreements as labelled examples |

`GET /obligations` is the integration point that matters: once every duty carries
a named owner, the register stops being a spreadsheet somebody maintains and
becomes something a workflow engine, task system, or scheduled digest can read.

## Configuration

All behaviour is environment-configurable; see [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `LEGISWATCH_PROVIDER` | `replay` | `replay`, `anthropic`, `openai`, `ollama` |
| `LEGISWATCH_INSTITUTION` | `Indiana University` | Institution the prompts target |
| `LEGISWATCH_FUZZY_THRESHOLD` | `92.0` | Groundedness floor (0–100) |
| `LEGISWATCH_AUTO_FILE_THRESHOLD` | `0.85` | Minimum combined confidence to auto-file |
| `LEGISWATCH_MIN_VERIFIER_CONFIDENCE` | `0.75` | Verifier floor for auto-filing |
| `LEGISWATCH_WEIGHT_*` | `0.25/0.45/0.30` | Confidence blend; validated to sum to 1.0 |
| `LEGISWATCH_OUT_DIR` | `./out` | Output directory |

Thresholds are exposed rather than buried because they are expected to be tuned
against measured reviewer agreement after deployment.

## Repository layout

```
src/legiswatch/
  config.py           environment-driven settings, validated at import
  schemas.py          pydantic contracts for every model interaction
  llm.py              provider abstraction, retry-on-validation-error, fixtures
  graph.py            LangGraph state machine and runner
  nodes/triage.py     relevance classifier
  nodes/extract.py    obligation decomposition
  nodes/verify.py     groundedness + adversarial verification
  nodes/route.py      confidence and policy gates
  audit.py            append-only decision journal
  review.py           override vs amend-and-reprocess
  dashboard.py        self-contained HTML reviewer queue
  api.py              FastAPI service
  evaluate.py         gold-set scoring, usable as a CI gate
  connectors/iga.py   live legislative data client
data/corpus/          source documents with provenance URLs
data/gold/            hand-labelled gold set
data/replay/          recorded response fixtures
tests/                59 tests, hermetic
```

## Engineering notes

**Structured outputs, never parsed text.** Every model call returns a validated
pydantic model. On a validation failure the error is fed back into the prompt and
the call is retried, rather than pushing malformed data downstream.

**Error-driven refinement.** Up to `max_retries` attempts, each carrying the
previous validation error as context.

**Auditable logging.** Structured JSON, one object per line, with per-stage
latency and token accounting. Six months later, "why was this auto-filed" is
answerable from a log rather than from memory.

**Tests weighted by blast radius.** Of 59 tests, most cover the guardrail and
routing layers. A degraded extraction prompt lowers quality; a broken
groundedness check admits fabricated obligations into a compliance register.

## Limitations

- The corpus is small (10 documents). Percentages should be re-measured on a
  production corpus before being relied upon.
- The office routing map is inferred from public organisational structure. Any
  real deployment must replace it with the authoritative one.
- Recall is 95.8%, not 100%. One gold obligation is deliberately retained that
  the current prompt does not reliably surface — an evaluation set containing
  only what the system already finds cannot measure recall.
- Obligations are not yet versioned across runs, so amendments to a
  previously-tracked duty are not diffed.
- `api.py` caches the last run in process memory; a multi-worker deployment needs
  real persistence.

## Roadmap

- Versioned obligations with cross-run diffing — detecting when an amended bill
  changes a duty already being tracked
- Persistence behind the journal and run cache (Postgres, same append-only rule)
- Scheduled ingestion during legislative sessions, with per-office digests
- Push into downstream task systems rather than requiring a separate dashboard

## Data sources

Source documents are verbatim public records:

- [Indiana Code Title 21](https://law.justia.com/codes/indiana/title-21/) (Justia)
- [Indiana Commission for Higher Education](https://www.in.gov/che/) guidance
- [Indiana General Assembly MyIGA API](https://docs.api.iga.in.gov/) — live connector;
  tokens are issued free on request

Two documents in the corpus are flagged synthetic negative controls, used to
measure triage precision.

## License

MIT — see [LICENSE](LICENSE).
