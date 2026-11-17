# P4 — Revision and submission package

Role: Writer, with Daniel · Wave: 3 · Branch: `wave-3/P4-revision` · Depends on: P3 · Time box: 4 days of agent work,
plus Daniel's revisions and co-author review

## Goal

Answer the referee report, take in Daniel's and the co-authors' revisions, and prepare everything the journal asks
for at submission. Daniel submits.

## Specification

1. **Response** to every point of `paper/review/P3-referee-report.md`: change made (with location) or reasoned
   reply, in `paper/review/P3-response.md`. Computations needed for an answer go through P1's scripts.
2. **Daniel's revisions** and co-author comments merged; Daniel has the last word on wording and claims.
3. **Submission package** as `paper/JOURNAL.md` lists it: cover letter draft, highlights or significance statement
   if required, data and code availability (the Zenodo DOI of 1.0.0 and of the figure scripts), AI-assistance
   disclosure drafted to the journal's policy for Daniel to approve, conflict and funding statements (Daniel
   supplies), suggested and opposed referees (Daniel decides), and a preprint decision (arXiv or not; Daniel).
4. Final `paper/check.py` run and a PDF built from a clean checkout.

## Acceptance

Every referee point answered; Daniel approves the manuscript and the package.

## Deliverables

Revised manuscript, `paper/review/P3-response.md`, `paper/submission/`, `docs/reports/P4.md`.

## Out of scope

Submitting (Daniel does); the real journal review, which gets its own card when it arrives.
