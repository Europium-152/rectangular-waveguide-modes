# P4 — Revision and submission package

Role: Writer, with Daniel · Wave: 3 · Branch: `P4-revision` (manuscript repository) · Depends on: P3 · Time box: 4 days of agent work,
plus Daniel's revisions and co-author review

## Goal

Answer the referee report, take in Daniel's and the co-authors' revisions, and prepare everything the journal asks
for at submission. Daniel submits.

## Specification

1. **Response** to every point of `review/P3-referee-report.md`: change made (with location) or reasoned
   reply, in `review/P3-response.md`. Computations needed for an answer go through P1's scripts.
2. **Daniel's revisions** and co-author comments merged; Daniel has the last word on wording and claims.
3. **Submission package** as `JOURNAL.md` lists it: cover letter draft, highlights or significance statement
   if required, data and code availability (the Zenodo DOI of 1.0.0 and of the figure scripts), AI-assistance
   disclosure drafted to the journal's policy for Daniel to approve, conflict and funding statements (Daniel
   supplies), suggested and opposed referees (Daniel decides), and a preprint decision (arXiv or not; Daniel).
4. Final `check.py` run and a PDF built from a clean checkout.
5. **Timing**: the article is published only after the JINST smooth-bend paper (Daniel, 23 Nov). Before submission,
   update its citation to the published version, or check what TMTT allows for a citation that is not yet
   published.

## Acceptance

Every referee point answered; Daniel approves the manuscript and the package.

## Deliverables

Revised manuscript, `review/P3-response.md`, `submission/`, `reports/P4.md`.

## Out of scope

Submitting (Daniel does); the real journal review, which gets its own card when it arrives.
