"""Self-contained HTML review dashboard.

This is the part a non-engineer actually touches. The analyst's job is to work
the review queue: read the obligation, read the quote it came from, and either
confirm or correct. Everything on the page is in service of making that decision
fast and defensible -- every card carries its source quote and a link back to
the statute, so no one is ever asked to trust an assertion they cannot check.

No build step, no CDN, no network: one HTML file that opens anywhere. That
matters in a university setting where the reviewer may be a compliance officer
on a locked-down laptop, not a developer.
"""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

from .schemas import IUUnit, RouteDecision, RunSummary

# Palette slots -- validated defaults from the data-viz reference instance.
# Status colors are reserved for routing state and never reused as series hues;
# each ships with an icon and a text label so state never rests on color alone.
_CSS = """
:root {
  color-scheme: light;
  --surface-0: #f6f5f2;
  --surface-1: #fcfcfb;
  --surface-2: #f0efec;
  --border:    #e2e0da;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #83817a;
  --series-1: #2a78d6;
  --good:     #0ca30c;
  --warning:  #fab219;
  --critical: #d03b3b;
  --quote-bg: #f4f6f9;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211;
    --surface-1: #1a1a19;
    --surface-2: #232322;
    --border:    #383835;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #8e8d84;
    --series-1: #3987e5;
    --quote-bg: #1f242b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --surface-2: #232322;
  --border: #383835;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8e8d84;
  --series-1: #3987e5; --quote-bg: #1f242b;
}

* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 0 64px;
  background: var(--surface-0); color: var(--text-primary);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1120px; margin: 0 auto; padding: 0 24px; }

header.top {
  border-bottom: 1px solid var(--border); background: var(--surface-1);
  padding: 28px 0 22px; margin-bottom: 28px;
}
h1 { margin: 0 0 4px; font-size: 25px; letter-spacing: -0.02em; font-weight: 620; }
.sub { color: var(--text-secondary); font-size: 14px; margin: 0; }
.runmeta {
  margin-top: 14px; font-size: 12.5px; color: var(--text-muted);
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
.theme-toggle {
  float: right; background: var(--surface-2); border: 1px solid var(--border);
  color: var(--text-secondary); border-radius: 7px; padding: 6px 11px;
  font-size: 12.5px; cursor: pointer;
}

h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.07em;
     color: var(--text-muted); font-weight: 600; margin: 34px 0 14px; }

/* --- stat tiles: the headline numbers are numbers, not a chart --- */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); gap: 12px; }
.tile {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 11px; padding: 15px 17px;
}
.tile .val { font-size: 30px; font-weight: 640; letter-spacing: -0.025em; line-height: 1.1; }
.tile .lbl { font-size: 12.5px; color: var(--text-secondary); margin-top: 3px; }
.tile .note { font-size: 11.5px; color: var(--text-muted); margin-top: 5px; }
.tile.status { display: flex; gap: 11px; align-items: flex-start; }
.dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto; margin-top: 11px; }
.dot.good { background: var(--good); }
.dot.warning { background: var(--warning); }
.dot.critical { background: var(--critical); }

/* --- single-series horizontal bar --- */
.chart { background: var(--surface-1); border: 1px solid var(--border);
         border-radius: 11px; padding: 20px 22px; }
.brow { display: grid; grid-template-columns: 250px 1fr 34px; gap: 12px;
        align-items: center; margin-bottom: 9px; }
.brow .name { font-size: 12.5px; color: var(--text-secondary); text-align: right;
              overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.btrack { height: 15px; position: relative; }
.bfill { height: 15px; background: var(--series-1);
         border-radius: 0 4px 4px 0; transition: opacity .12s; }
.bfill:hover { opacity: .82; }
.brow .num { font-size: 12.5px; color: var(--text-secondary);
             font-variant-numeric: tabular-nums; }

/* --- filters --- */
.filters { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 16px; }
.chip {
  background: var(--surface-1); border: 1px solid var(--border); color: var(--text-secondary);
  border-radius: 999px; padding: 6px 14px; font-size: 13px; cursor: pointer;
}
.chip[aria-pressed="true"] {
  background: var(--series-1); border-color: var(--series-1); color: #fff; font-weight: 550;
}
select {
  background: var(--surface-1); border: 1px solid var(--border); color: var(--text-primary);
  border-radius: 7px; padding: 6px 10px; font-size: 13px; max-width: 340px;
}

/* --- obligation cards --- */
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-left-width: 3px; border-radius: 10px; padding: 15px 17px; margin-bottom: 11px;
}
.card.auto_file { border-left-color: var(--good); }
.card.human_review { border-left-color: var(--warning); }
.card.rejected { border-left-color: var(--critical); }

.card .hd { display: flex; justify-content: space-between; gap: 16px;
            align-items: flex-start; margin-bottom: 8px; }
.card .summ { font-size: 14.5px; font-weight: 560; line-height: 1.45; }
.badge {
  flex: 0 0 auto; font-size: 11px; font-weight: 620; letter-spacing: .03em;
  padding: 3px 9px; border-radius: 5px; white-space: nowrap;
  border: 1px solid var(--border); background: var(--surface-2); color: var(--text-secondary);
}
.meta { display: flex; flex-wrap: wrap; gap: 6px; margin: 9px 0; }
.pill {
  font-size: 11.5px; background: var(--surface-2); color: var(--text-secondary);
  border: 1px solid var(--border); border-radius: 5px; padding: 2px 8px;
  font-family: ui-monospace, Menlo, Consolas, monospace;
}
blockquote {
  margin: 10px 0 0; padding: 10px 13px; background: var(--quote-bg);
  border-left: 2px solid var(--series-1); border-radius: 0 6px 6px 0;
  font-size: 13px; color: var(--text-secondary); font-style: italic; line-height: 1.5;
}
.reason { margin-top: 9px; font-size: 12.5px; color: var(--text-muted); }
.reason b { color: var(--text-secondary); font-weight: 600; }
.actions { margin-top: 11px; display: flex; gap: 7px; align-items: center; }
.btn {
  font-size: 12.5px; padding: 5px 12px; border-radius: 6px; cursor: pointer;
  border: 1px solid var(--border); background: var(--surface-2); color: var(--text-secondary);
}
.btn.primary { background: var(--series-1); border-color: var(--series-1); color: #fff; }
.btn:disabled { opacity: .5; cursor: not-allowed; }
a { color: var(--series-1); }

/* --- analyst review panel --- */
.review {
  margin-top: 12px; padding: 13px 15px; border: 1px solid var(--border);
  border-radius: 9px; background: var(--surface-2);
}
.review.hidden { display: none; }
.review label {
  display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
  color: var(--text-muted); font-weight: 600; margin: 9px 0 4px;
}
.review label:first-child { margin-top: 0; }
.review select, .review textarea, .review input[type=text] {
  width: 100%; background: var(--surface-1); border: 1px solid var(--border);
  color: var(--text-primary); border-radius: 6px; padding: 7px 9px;
  font-size: 13px; font-family: inherit;
}
.review textarea { min-height: 62px; resize: vertical; }
.review .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.review .hint { font-size: 11px; color: var(--text-muted); margin-top: 5px; }
.review .err { font-size: 12px; color: var(--critical); margin-top: 7px; display: none; }
.review .fallback {
  display: none; margin-top: 10px; padding: 9px 11px; background: var(--quote-bg);
  border-radius: 6px; font-family: ui-monospace, Menlo, monospace; font-size: 11px;
  color: var(--text-secondary); white-space: pre-wrap; word-break: break-all;
}
.review .ok { display: none; margin-top: 9px; font-size: 12.5px; color: var(--good); }

/* --- decision history on a card --- */
.history { margin-top: 12px; border-top: 1px solid var(--border); padding-top: 11px; }
.history h5 {
  margin: 0 0 8px; font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
  color: var(--text-muted); font-weight: 600;
}
.hentry { font-size: 12.5px; color: var(--text-secondary); margin-bottom: 9px; }
.hentry .hhead { font-weight: 600; color: var(--text-primary); }
.hentry .hcomment { font-style: italic; margin-top: 2px; }
.hentry .htrace { font-size: 11px; color: var(--text-muted); margin-top: 3px; }
.prov {
  display: inline-block; font-size: 10.5px; font-family: ui-monospace, Menlo, monospace;
  padding: 2px 7px; border-radius: 4px; background: var(--quote-bg);
  color: var(--text-secondary); border: 1px solid var(--border);
}

table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border);
         vertical-align: top; }
th { color: var(--text-muted); font-size: 11.5px; text-transform: uppercase;
     letter-spacing: .05em; font-weight: 600; }
details.tableview { margin-top: 14px; }
details.tableview summary { cursor: pointer; color: var(--text-secondary); font-size: 13px; }

.callout {
  background: var(--surface-1); border: 1px solid var(--border);
  border-left: 3px solid var(--critical); border-radius: 10px;
  padding: 15px 17px; margin-bottom: 11px;
}
.callout h3 { margin: 0 0 6px; font-size: 14px; }
.hidden { display: none !important; }
footer { margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--border);
         color: var(--text-muted); font-size: 12.5px; }
"""

_BADGE = {
    RouteDecision.AUTO_FILE: ("● AUTO-FILED", "good"),
    RouteDecision.HUMAN_REVIEW: ("▲ NEEDS REVIEW", "warning"),
    RouteDecision.REJECTED: ("✕ REJECTED", "critical"),
}


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _provenance_for(journal, oid: str) -> str:
    """Who the current state belongs to: the agent, an analyst, or both."""
    if journal is None:
        return "decided by: agent"
    latest = journal.latest_for(oid)
    return f"decided by: {latest.decided_by.value}" if latest else "decided by: agent"


def _history_html(journal, oid: str) -> str:
    """Render every analyst decision recorded against this obligation."""
    if journal is None:
        return ""
    decisions = journal.history_for(oid)
    if not decisions:
        return ""

    rows = []
    for d in decisions:
        unit_move = ""
        if d.from_unit and d.to_unit and d.from_unit != d.to_unit:
            unit_move = f" · {_e(d.from_unit.name)} → {_e(d.to_unit.name)}"
        edits = ""
        if d.field_edits:
            listed = ", ".join(
                f"{_e(k)}" + ("" if v else " (cleared)") for k, v in d.field_edits.items()
            )
            edits = f'<div class="htrace">edited: {listed}</div>'
        trace = f'<div class="htrace">{_e(d.reprocess_trace)}</div>' if d.reprocess_trace else ""
        rows.append(
            f"""<div class="hentry">
              <div class="hhead">{_e(d.timestamp)} · {_e(d.reviewer)} ·
                {_e(d.action.value.upper())}</div>
              <div>{_e(d.from_route.value)} → {_e(d.to_route.value)}{unit_move}</div>
              <div class="hcomment">&ldquo;{_e(d.comment)}&rdquo;</div>
              {edits}{trace}
              <div class="htrace">{_e(d.decision_id)}"""
            + (f" · supersedes {_e(d.supersedes)}" if d.supersedes else "")
            + "</div></div>"
        )
    return f'<div class="history"><h5>Decision history ({len(decisions)})</h5>{"".join(rows)}</div>'


def _review_panel(oid: str, route: RouteDecision) -> str:
    """The form an analyst actually uses.

    Which actions are offered depends on where the obligation currently sits:
    a rejected item offers REINSTATE (which re-runs the checks) and never a bare
    CONFIRM, because confirming a rejection would bypass verification entirely.
    """
    if route == RouteDecision.REJECTED:
        options = [
            ("reinstate", "Reinstate — correct and re-run the checks"),
            ("reroute", "Force route — override the pipeline deliberately"),
            ("reject", "Uphold rejection"),
        ]
    else:
        options = [
            ("confirm", "Confirm — file as-is"),
            ("reassign", "Reassign to a different office"),
            ("amend", "Amend the text and re-run the checks"),
            ("reroute", "Force a different route"),
            ("reject", "Reject"),
        ]
    opts = "".join(f'<option value="{v}">{_e(t)}</option>' for v, t in options)
    units = "".join(f'<option value="{u.name}">{_e(u.value)}</option>' for u in IUUnit)

    return f"""
    <div class="review hidden" id="rv-{_e(oid)}">
      <label>Action</label>
      <select class="js-action">{opts}</select>

      <div class="row">
        <div>
          <label>Reassign to</label>
          <select class="js-unit"><option value="">— unchanged —</option>{units}</select>
        </div>
        <div>
          <label>Force route to</label>
          <select class="js-route">
            <option value="">— unchanged —</option>
            <option value="auto_file">auto_file</option>
            <option value="human_review">human_review</option>
            <option value="rejected">rejected</option>
          </select>
        </div>
      </div>

      <label>Corrected summary <span style="text-transform:none;font-weight:400">(optional)</span></label>
      <input type="text" class="js-summary" placeholder="Leave blank to keep the current summary">

      <label>Corrected quote <span style="text-transform:none;font-weight:400">(optional)</span></label>
      <input type="text" class="js-quote" placeholder="Must appear verbatim in the source — it is re-checked">

      <label>Reviewer</label>
      <input type="text" class="js-reviewer" placeholder="your username">

      <label>Why — required</label>
      <textarea class="js-comment" placeholder="State the reason. This is stored permanently and is what makes the override auditable."></textarea>
      <div class="hint">Minimum one real sentence. An override with no stated reason is rejected by the API.</div>

      <div class="err js-err"></div>
      <div class="ok js-ok"></div>
      <div class="fallback js-fallback"></div>

      <div class="actions">
        <button class="btn primary js-submit" data-oid="{_e(oid)}">Record decision</button>
        <button class="btn js-cancel" data-oid="{_e(oid)}">Cancel</button>
      </div>
    </div>"""


def _dur(ms: float) -> str:
    """Sub-second runs are the replay-mode case; showing '0.0s' hides the point."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 60_000:.1f}m"


def render_dashboard(
    summary: RunSummary,
    corpus_meta: dict,
    out_path: Path,
    journal=None,
    api_base: str = "http://127.0.0.1:8000",
) -> Path:
    """Render the review queue.

    `journal` is an optional AuditJournal. When supplied, each card shows its
    full decision history and current provenance. `api_base` is where the review
    panel POSTs decisions; if it is unreachable the panel falls back to printing
    the equivalent CLI command, so the page stays usable with no server running.
    """
    routed = [r for res in summary.results for r in res.routed_obligations]
    by_unit = Counter(r.obligation.responsible_unit.value for r in routed)
    units_sorted = by_unit.most_common()
    max_unit = max(by_unit.values()) if by_unit else 1

    filtered_docs = summary.documents_ingested - summary.documents_relevant
    pct_auto = (
        summary.obligations_auto_filed / summary.obligations_extracted * 100
        if summary.obligations_extracted
        else 0
    )

    # ---- stat tiles ------------------------------------------------------
    tiles = f"""
    <div class="tiles">
      <div class="tile">
        <div class="val">{summary.documents_ingested}</div>
        <div class="lbl">Documents ingested</div>
        <div class="note">{filtered_docs} filtered out before extraction</div>
      </div>
      <div class="tile">
        <div class="val">{summary.obligations_extracted}</div>
        <div class="lbl">Obligations extracted</div>
        <div class="note">from {summary.documents_relevant} relevant documents</div>
      </div>
      <div class="tile status">
        <span class="dot good"></span>
        <div><div class="val">{summary.obligations_auto_filed}</div>
        <div class="lbl">Auto-filed</div>
        <div class="note">{pct_auto:.0f}% of extractions</div></div>
      </div>
      <div class="tile status">
        <span class="dot warning"></span>
        <div><div class="val">{summary.obligations_needing_review}</div>
        <div class="lbl">Queued for review</div>
        <div class="note">analyst confirmation required</div></div>
      </div>
      <div class="tile status">
        <span class="dot critical"></span>
        <div><div class="val">{summary.obligations_rejected}</div>
        <div class="lbl">Rejected by guardrails</div>
        <div class="note">never reached a human queue</div></div>
      </div>
      <div class="tile">
        <div class="val">{_dur(summary.total_latency_ms)}</div>
        <div class="lbl">Wall clock</div>
        <div class="note">{summary.provider} · {_e(summary.model)}</div>
      </div>
    </div>"""

    # ---- single-series bar: workload by owning office --------------------
    bars = "".join(
        f"""<div class="brow">
              <div class="name" title="{_e(u)}">{_e(u)}</div>
              <div class="btrack"><div class="bfill" style="width:{c / max_unit * 100:.1f}%"
                   title="{_e(u)}: {c}"></div></div>
              <div class="num">{c}</div>
            </div>"""
        for u, c in units_sorted
    )

    # ---- rejected callouts ----------------------------------------------
    rejects = [r for r in routed if r.route == RouteDecision.REJECTED]
    reject_html = "".join(
        f"""<div class="callout">
              <h3>✕ {_e(r.obligation.obligation_id)}</h3>
              <div class="reason"><b>Claimed:</b> {_e(r.obligation.summary)}</div>
              <div class="reason"><b>Why it was stopped:</b> {_e(r.route_reason)}</div>
              <div class="meta">
                <span class="pill">exact_match: {str(r.groundedness.exact_match).lower()}</span>
                <span class="pill">fuzzy: {r.groundedness.fuzzy_score:.1f}</span>
                <span class="pill">self-reported confidence: {r.obligation.confidence:.2f}</span>
              </div>
              <blockquote>{_e(r.obligation.verbatim_quote)}</blockquote>
            </div>"""
        for r in rejects
    )

    # ---- obligation cards ------------------------------------------------
    def card(r) -> str:
        # Render CURRENT state, not the agent's original. An analyst who
        # reinstated an obligation an hour ago must not still see it as rejected,
        # and a corrected summary must be the one on the card. The agent's
        # original decision is never lost -- it is shown explicitly below and in
        # full in the decision history.
        lc = journal.lifecycle(r) if journal is not None else None
        cur_route = lc.current_route if lc else r.route
        o = lc.current_obligation if lc else r.obligation

        label = _BADGE[cur_route][0]
        deadline = (
            f'<span class="pill">due: {_e(o.deadline_text)}</span>' if o.deadline_text else ""
        )
        src = (
            f' · <a href="{_e(r.source_url)}" target="_blank" rel="noopener">source</a>'
            if r.source_url
            else ""
        )
        objection = (
            f'<div class="reason"><b>Verifier objection:</b> {_e(r.verdict.objection)}</div>'
            if r.verdict and r.verdict.objection
            else ""
        )
        oid = o.obligation_id
        history = _history_html(journal, oid)
        prov = _provenance_for(journal, oid)

        agent_original = ""
        if lc and (lc.current_route != r.route or lc.current_unit != r.obligation.responsible_unit):
            agent_original = (
                f'<div class="reason"><b>Agent originally:</b> '
                f"{_e(r.route.value)} &middot; {_e(r.obligation.responsible_unit.value)} "
                f"&mdash; superseded by analyst review below.</div>"
            )

        # Every obligation is actionable, including rejected ones. A rejection
        # the analyst cannot revisit is a silent data-loss path: guardrails have
        # false positives too, and a quote can fail groundedness because the
        # source was badly OCR'd rather than because the model invented it.
        return f"""
        <article class="card {cur_route.value}" data-route="{cur_route.value}"
                 data-unit="{_e(o.responsible_unit.value)}" data-oid="{_e(oid)}">
          <div class="hd">
            <div class="summ">{_e(o.summary)}</div>
            <span class="badge">{label}</span>
          </div>
          <div class="meta">
            <span class="pill">{_e(o.citation)}</span>
            <span class="pill">{_e(o.obligation_type.value)}</span>
            <span class="pill">{_e(o.recurrence.value)}</span>
            {deadline}
            <span class="pill">confidence {r.combined_confidence:.2f}</span>
            <span class="prov">{prov}</span>
          </div>
          <div class="reason"><b>Owner:</b> {_e(o.responsible_unit.value)} — {_e(o.unit_rationale)}</div>
          {agent_original}
          {objection}
          <div class="reason"><b>Agent routing:</b> {_e(r.route_reason)}</div>
          <blockquote>{_e(o.verbatim_quote)}</blockquote>
          <div class="reason">Grounded in {_e(r.source_doc_id)}
            (exact match: {str(r.groundedness.exact_match).lower()}){src}</div>
          {history}
          <div class="actions">
            <button class="btn primary js-open" data-oid="{_e(oid)}">Review this</button>
          </div>
          {_review_panel(oid, cur_route)}
        </article>"""

    # Sort and count by CURRENT state so the chips match what the cards show.
    def cur(r) -> RouteDecision:
        return journal.lifecycle(r).current_route if journal is not None else r.route

    order = {RouteDecision.HUMAN_REVIEW: 0, RouteDecision.REJECTED: 1, RouteDecision.AUTO_FILE: 2}
    cards = "".join(
        card(r) for r in sorted(routed, key=lambda r: (order[cur(r)], -r.combined_confidence))
    )
    cur_counts = Counter(cur(r) for r in routed)
    n_review = cur_counts.get(RouteDecision.HUMAN_REVIEW, 0)
    n_auto = cur_counts.get(RouteDecision.AUTO_FILE, 0)
    n_reject = cur_counts.get(RouteDecision.REJECTED, 0)
    n_reviewed = (
        len([r for r in routed if journal.history_for(r.obligation.obligation_id)])
        if journal is not None
        else 0
    )
    review_banner = (
        f'<p class="sub" style="margin:0 0 14px;font-size:13px">'
        f"Counts above are the pipeline's own output. "
        f"<b>{n_reviewed}</b> obligation(s) have since been reviewed by an analyst; "
        f"the queue below shows current state, with every agent decision preserved "
        f"in each card's history.</p>"
        if n_reviewed
        else ""
    )

    unit_options = "".join(
        f'<option value="{_e(u)}">{_e(u)} ({c})</option>' for u, c in units_sorted
    )

    # ---- accessibility: table view of the same data ----------------------
    rows = "".join(
        f"<tr><td>{_e(r.obligation.citation)}</td><td>{_e(r.obligation.summary)}</td>"
        f"<td>{_e(r.obligation.responsible_unit.value)}</td>"
        f"<td>{_e(r.obligation.deadline_text) or '—'}</td>"
        f"<td>{r.combined_confidence:.2f}</td><td>{_BADGE[r.route][0]}</td></tr>"
        for r in routed
    )

    # ---- skipped documents -----------------------------------------------
    skipped = [res for res in summary.results if res.skipped_reason]
    skipped_rows = "".join(
        f"<tr><td>{_e(s.citation)}</td><td>{_e(s.title)}</td><td>{_e(s.skipped_reason)}</td></tr>"
        for s in skipped
    )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LegisWatch — statutory obligation review queue</title>
<style>{_CSS}</style>
</head><body>

<header class="top"><div class="wrap">
  <button class="theme-toggle" id="tt">Toggle theme</button>
  <h1>LegisWatch</h1>
  <p class="sub">Statutory obligations extracted from Indiana legislative and regulatory text,
     verified against source, and routed to an owning office.</p>
  <div class="runmeta">{_e(summary.run_id)} · {_e(summary.started_at)} ·
     {_e(summary.provider)}/{_e(summary.model)}</div>
</div></header>

<div class="wrap">

  <h2>Run summary</h2>
  {tiles}

  <h2>Where the work lands</h2>
  <div class="chart">
    <p class="sub" style="margin:0 0 15px;font-size:13px">
      Obligations by owning office. This is the number that makes the case: the
      pipeline does not just find duties, it tells you whose desk each one belongs on.</p>
    {bars}
  </div>

  {"<h2>Guardrails that fired</h2>" + reject_html if rejects else ""}

  <h2>Review queue</h2>
  {review_banner}
  <div class="filters">
    <button class="chip" data-f="all" aria-pressed="true">All ({len(routed)})</button>
    <button class="chip" data-f="human_review" aria-pressed="false">▲ Needs review ({n_review})</button>
    <button class="chip" data-f="auto_file" aria-pressed="false">● Auto-filed ({n_auto})</button>
    <button class="chip" data-f="rejected" aria-pressed="false">✕ Rejected ({n_reject})</button>
    <select id="unitf"><option value="all">All offices</option>{unit_options}</select>
  </div>
  <div id="cards">{cards}</div>

  <details class="tableview"><summary>Table view (all {len(routed)} obligations)</summary>
    <table><thead><tr><th>Citation</th><th>Obligation</th><th>Owner</th>
      <th>Deadline</th><th>Conf.</th><th>Route</th></tr></thead>
      <tbody>{rows}</tbody></table>
  </details>

  <h2>Documents filtered out before extraction ({len(skipped)})</h2>
  <table><thead><tr><th>Citation</th><th>Title</th><th>Reason</th></tr></thead>
    <tbody>{skipped_rows}</tbody></table>

  <footer>
    <p><b>Corpus provenance.</b> {_e(corpus_meta.get("description", ""))}
       Retrieved {_e(corpus_meta.get("retrieved", ""))}.</p>
    <p>Every obligation on this page carries the verbatim span it was drawn from.
       Nothing is auto-filed without passing a deterministic groundedness check and an
       independent adversarial verification pass. External reporting duties and statutory
       prohibitions are never auto-filed — they always route to a human by policy, regardless
       of model confidence.</p>
  </footer>
</div>

<script>
(function () {{
  var tt = document.getElementById('tt');
  tt.addEventListener('click', function () {{
    var cur = document.documentElement.getAttribute('data-theme');
    var next = cur === 'dark' ? 'light' : (cur === 'light' ? 'dark' :
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'light' : 'dark'));
    document.documentElement.setAttribute('data-theme', next);
  }});

  var route = 'all', unit = 'all';
  var cards = Array.prototype.slice.call(document.querySelectorAll('#cards .card'));

  function apply() {{
    cards.forEach(function (c) {{
      var okR = route === 'all' || c.dataset.route === route;
      var okU = unit === 'all' || c.dataset.unit === unit;
      c.classList.toggle('hidden', !(okR && okU));
    }});
  }}

  document.querySelectorAll('.chip').forEach(function (b) {{
    b.addEventListener('click', function () {{
      document.querySelectorAll('.chip').forEach(function (x) {{
        x.setAttribute('aria-pressed', String(x === b));
      }});
      route = b.dataset.f; apply();
    }});
  }});
  document.getElementById('unitf').addEventListener('change', function (e) {{
    unit = e.target.value; apply();
  }});

  // ---- analyst review -------------------------------------------------
  var API = {api_base!r};

  function panel(oid) {{ return document.getElementById('rv-' + oid); }}

  document.querySelectorAll('.js-open').forEach(function (b) {{
    b.addEventListener('click', function () {{
      panel(b.dataset.oid).classList.toggle('hidden');
    }});
  }});
  document.querySelectorAll('.js-cancel').forEach(function (b) {{
    b.addEventListener('click', function () {{
      panel(b.dataset.oid).classList.add('hidden');
    }});
  }});

  function buildBody(p) {{
    var edits = {{}};
    var summary = p.querySelector('.js-summary').value.trim();
    var quote = p.querySelector('.js-quote').value.trim();
    if (summary) edits.summary = summary;
    if (quote) edits.verbatim_quote = quote;

    var body = {{
      action: p.querySelector('.js-action').value,
      reviewer: p.querySelector('.js-reviewer').value.trim(),
      comment: p.querySelector('.js-comment').value.trim(),
      field_edits: edits
    }};
    var u = p.querySelector('.js-unit').value;
    var rt = p.querySelector('.js-route').value;
    if (u) body.to_unit = u;
    if (rt) body.to_route = rt;
    return body;
  }}

  function cliEquivalent(oid, body) {{
    var parts = ['legiswatch-review decide ' + oid,
                 '  --action ' + body.action,
                 '  --reviewer ' + (body.reviewer || 'YOUR_NAME'),
                 '  --comment "' + body.comment.replace(/"/g, '\\\\"') + '"'];
    if (body.to_unit) parts.push('  --to-unit ' + body.to_unit);
    if (body.to_route) parts.push('  --to-route ' + body.to_route);
    Object.keys(body.field_edits).forEach(function (k) {{
      parts.push('  --set ' + k + '="' + body.field_edits[k].replace(/"/g, '\\\\"') + '"');
    }});
    return parts.join(' \\\\\\n');
  }}

  document.querySelectorAll('.js-submit').forEach(function (b) {{
    b.addEventListener('click', function () {{
      var oid = b.dataset.oid, p = panel(oid);
      var err = p.querySelector('.js-err'), ok = p.querySelector('.js-ok');
      var fb = p.querySelector('.js-fallback');
      err.style.display = ok.style.display = fb.style.display = 'none';

      var body = buildBody(p);

      // Client-side mirror of the server's rule. The API enforces it regardless;
      // this just avoids a round-trip to be told the comment is too thin.
      if (!body.reviewer) {{
        err.textContent = 'Reviewer is required — a decision has to belong to someone.';
        err.style.display = 'block'; return;
      }}
      if (body.comment.length < 10 || body.comment.split(/\\s+/).length < 3) {{
        err.textContent = 'A real reason is required. This is stored permanently and is what makes the override auditable.';
        err.style.display = 'block'; return;
      }}

      b.disabled = true; b.textContent = 'Recording…';
      fetch(API + '/obligations/' + encodeURIComponent(oid) + '/decisions', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(body)
      }})
      .then(function (res) {{ return res.json().then(function (j) {{ return {{ok: res.ok, j: j}}; }}); }})
      .then(function (r) {{
        if (!r.ok) {{
          err.textContent = 'Rejected: ' + (r.j.detail || JSON.stringify(r.j));
          err.style.display = 'block'; return;
        }}
        ok.innerHTML = '<b>' + r.j.decision.decision_id + '</b> recorded · ' +
          r.j.decision.from_route + ' &rarr; ' + r.j.decision.to_route +
          '<br>' + r.j.note;
        ok.style.display = 'block';
      }})
      .catch(function () {{
        // API unreachable. Degrade to showing the equivalent CLI invocation so
        // the reviewer can still record the decision.
        fb.textContent = 'API not reachable at ' + API +
                    '.\nStart it with:  uvicorn legiswatch.api:app\\n\\n' +
          'Or record this decision from the console:\\n\\n' + cliEquivalent(oid, body);
        fb.style.display = 'block';
      }})
      .finally(function () {{ b.disabled = false; b.textContent = 'Record decision'; }});
    }});
  }});
}})();
</script>
</body></html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path
