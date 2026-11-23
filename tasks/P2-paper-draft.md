# P2 — Article draft

Role: Writer · Wave: 3 · Branch: `P2-draft-<a|b|c>` (manuscript repository) · Depends on: P0 (approved), P1 (merged) · Reviewed by:
Daniel at each checkpoint, then P3 · Time box: 8 days over three checkpoints (then status review)

## Goal

A complete manuscript in IEEEtran (journal mode, IEEE TMTT), written from the approved outline and the generated numbers, that
Daniel can revise as its author.

## Read first

`OUTLINE.md`, `CLAIMS.md`, `FIGURES.md`, `JOURNAL.md` (P0); the submodule's `paper/generated/numbers.tex` and the figures
(P1); `CONVENTIONS.md`; the theory notes `docs/theory/T0`–`T3b` (the source of every equation); the reports named in
the claims table; the JINST smooth-bend paper (submitted; cite it as submitted, do not copy it).

## Specification

1. **Checkpoints**, one PR each, merged by Daniel before the next starts:
   a. methods: framework and component models;
   b. validation and examples;
   c. introduction, conclusions, abstract, data and code availability, acknowledgements.
2. **Numbers** only through the macros of `numbers.tex`. A number that is not there is either added to P1's
   scripts (a fix PR) or not written. Rounding as in the JSON.
3. **Equations** follow CONVENTIONS (e^{+iωt}, power-normalized amplitudes, the GSM definition of §6); each one has a
   LaTeX comment naming the theory note and section it comes from.
4. **Claims** only as in `CLAIMS.md`. Statements about other authors' results use the wording Daniel approved in
   P0, give the evidence, and stay factual.
5. **Text reuse**: none from the JINST paper (paraphrase and cite); no long quotations from any source.
6. **SPARC**: at most one sentence of motivation and the acknowledgements; no design-study result.
7. **References**: IEEE style (`IEEEtran.bst`); only entries of `references.bib` (P0, lookup-checked); every cited statement checked against the
   source where the PDF is in Daniel's reference folder (read-only, in a session linked to his computer, or an
   excerpt from the coordinator).
8. `check.py`: builds the PDF; fails on undefined macros, references or citations; lists every digit in the
   prose outside macros, equations, citations and cross-references, for review.
9. Length within the journal's limit.

## Acceptance

* Daniel merges the three checkpoints.
* `check.py` passes; its list of literal numbers is reviewed and each one justified in the PR.
* No claim outside `CLAIMS.md`; no SPARC result.

## Deliverables

In the manuscript repository: `manuscript/` (LaTeX, bib; figures from the submodule), `check.py`, `reports/P2.md`.

## Out of scope

New computations (P1 fix PRs); submission material (P4).
