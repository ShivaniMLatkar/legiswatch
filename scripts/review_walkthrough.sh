#!/usr/bin/env bash
# End-to-end exercise of the review layer.
#
# Executes six decisions covering every path through it:
#   1. a reassignment the verifier itself flagged
#   2. confirming that reassignment (a correction does not self-approve)
#   3. reinstating a rejected obligation whose quote was valid
#   4. reinstating one whose quote was fabricated, which correctly fails again
#   5. upholding that rejection explicitly, with an attributed reason
#   6. a deliberate override, accepted knowingly and attributed
#
# Usage:  bash scripts/demo_review_session.sh

set -euo pipefail
cd "$(dirname "$0")/.."

rm -f out/decision_journal.jsonl
legiswatch-run > /dev/null
echo "Pipeline run complete. Recording reviewer decisions..."
echo

R() { legiswatch-review decide "$@"; }

# 1 -- the verifier flagged General Counsel as an indefensible owner. Analyst agrees.
R IC-21-39.5-2-4-make-complaints-available-to-trustees \
  --action reassign --reviewer k.adams --to-unit BOARD_OF_TRUSTEES \
  --comment "Verifier was correct. The statute directs materials to the board with no intermediary; General Counsel was our assumption, not the text."

# 2 -- the reassignment came back to review rather than auto-filing, so confirm it.
R IC-21-39.5-2-4-make-complaints-available-to-trustees \
  --action confirm --reviewer k.adams \
  --comment "Reassignment checked against the statutory text. Board of Trustees confirmed as owner."

# 3 -- rejected for an invented deadline, but the quote was genuine. Correct and reinstate.
R CHE-HEA1001-preliminary-list-response \
  --action reinstate --reviewer k.adams \
  --comment "Verifier correctly caught the invented 30-day window, but the underlying duty is real and the quote is accurate. Removing the fabricated deadline rather than discarding the item." \
  --set summary="Receive and act on the Commission's preliminary list of under-threshold degree programs, expected mid-to-late July." \
  --set deadline_text=

# 4 -- reinstating a fabricated quote. Re-runs the checks and fails again, as it must.
R IC-21-39.5-2-4-quarterly-complaint-dashboard \
  --action reinstate --reviewer m.mendez \
  --comment "Checking whether this duty exists anywhere in the chapter before we discard it."

# 5 -- analyst upholds the rejection, on the record, with a reason.
R IC-21-39.5-2-4-quarterly-complaint-dashboard \
  --action reject --reviewer m.mendez \
  --comment "Confirmed by hand against the full text of IC 21-39.5-2-4. No quarterly dashboard duty exists anywhere in the chapter. Correctly caught by the pipeline."

# 6 -- a knowing override: the verifier's objection is fair but the duty is tracked anyway.
R IC-21-18-9-10.7-monitor-graduate-averages \
  --action reroute --reviewer m.mendez --to-route auto_file \
  --comment "Verifier flagged this as inferred rather than commanded, which is fair, but we do maintain these counts operationally and want it tracked. Accepting the risk deliberately."

# Rebuild the dashboard so decision history renders on each card.
legiswatch-run > /dev/null

echo
legiswatch-review stats
