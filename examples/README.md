# Example artefacts

Committed output from a real run, so the project can be reviewed without
executing anything. Regenerate with `legiswatch-run` and
`bash scripts/review_walkthrough.sh`.

| File | Contents |
|---|---|
| `dashboard.html` | Reviewer queue. Open in any browser — self-contained, no build step. |
| `run_summary.json` | Full pipeline output including both guardrail rejections. |
| `eval_report.json` | Gold-set scores: precision, recall, groundedness, auto-file safety. |
| `decision_journal.jsonl` | Six reviewer decisions covering every path through the review layer. |
