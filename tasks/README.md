# Task cards and swarm process

This folder drives the agent swarm that builds wrmodes 1.0 (plan: "wrmodes expansion plan: cascading, losses,
miters, tapers, twists", Daniel's project doc). Every unit of work is a **task card** in this folder. Agents start
from a card, work on their own branch, and finish with a pull request.

## Read before any task

1. `CONVENTIONS.md`: frozen conventions. A result that does not follow them is wrong, however good it looks.
2. This file.
3. Your card, and every file it lists under *Read first*.

## Roles

| Role | Does | Never does |
| --- | --- | --- |
| Coordinator | writes and updates cards, merges, runs the gates | writes component code |
| Theory | derivations in `docs/theory/`, with symbolic (sympy) and numerical (quadrature) checks | implements in `src/` |
| Implementer | code + unit tests for one work package | changes `CONVENTIONS.md` or another package's interface |
| Verifier | an independent reference built from the card and the literature, adversarial tests | reads the implementer's code before its own result is committed |
| Literature | digitizes and transcribes published data with uncertainties | sets tolerances |
| COMSOL scripter | COMSOL Java scripts + post-processing; Daniel runs them on the COMSOL machine | runs COMSOL |
| Reviewer | fresh-context review of physics and code of a PR | approves its own work |

**Independence rule.** A component is validated only when a verifier that did not see its code reproduces its
results by a different method within the tolerances of `CONVENTIONS.md`. Disagreements go into a *discrepancy
report* (`docs/discrepancies/<date>-<topic>.md`) and are settled by a theory agent and Daniel, not by the
implementer alone.

## Workflow

1. The coordinator writes the cards of a wave; Daniel approves them (a PR that only touches `tasks/`).
2. Each agent works on a branch `wave-<w>/<card-id>-<slug>` (e.g. `wave-1/WP1-cascade`), in its own clone or
   worktree, starting from the latest `main`.
3. Each PR carries code, tests, and a short report (`docs/reports/<card-id>.md`: what was done, results with
   numbers, open issues). CI must pass. The reviewer agent reviews, then Daniel merges.
4. When an agent's time box runs out, it stops and writes a status note in its report for review. The
   acceptance bar never drops to meet a date; the card is re-scoped or split instead.
5. At each gate, the coordinator runs the integration checks of the wave and Daniel runs the COMSOL campaign.

## References

The reference PDFs are in Daniel's `wrmodes_arbitrary_bends/references/` folder, not in this repository
(copyright). Agents that need them run in a session linked to Daniel's computer, or ask the coordinator for an
excerpt. Cards cite references by author, year and page (PDF page in parentheses when it differs).

## Card template

```markdown
# <ID> — <title>
Role: <role> · Wave: <w> · Branch: wave-<w>/<ID>-<slug> · Depends on: <cards>
Verified by: <card or role> · Time box: <days> (then status review)

## Goal
## Read first
## Specification            (interfaces, equations, conventions to use)
## Acceptance tests          (numbers and tolerances)
## Deliverables
## Out of scope
```

## Status board

| Card | Wave | Role | Status |
| --- | --- | --- | --- |
| T0 field convention | 0 | Theory | done |
| WP0 foundations | 0 | Coordinator | merged (#1) |
| Conventions update 1 | 1 | Coordinator | merged (#7) |
| T1 / T1b wall impedance | 1 | Theory | merged (#8, #2); agree to 8e-16 |
| T2 / T2b taper coupling | 1 | Theory | merged (#10, #13); agree to 7e-15 |
| T3 / T3b twist coupling | 1 | Theory | merged (#14, #9); agree to 8e-18 |
| L1 literature data | 1 | Literature | merged (#15; #5, #6 closed) |
| WP1 cascade | 1 | Implementer | merged (#3) |
| WP2 ohmic loss and filters | 1 | Implementer | merged (#16) |
| WP3 2D modal-port solver | 1 | Implementer | merged (#11) |
| WP4 steps and mode matching | 1 | Implementer | merged (#4) |
| WP7 verification, wave 1 | 1 | Verifier | merged (#12) |
| Conventions update 2, COMSOL campaign 1 | 1 | Coordinator | merged (#17) |
| Gate A: COMSOL campaign 1 | 1 | Coordinator + Daniel | passed (#18), 1.0.0a1 |
| Wave 2 cards, conventions update 3 | 2 | Coordinator | merged (#19) |
| L2 literature: Alimenti 1994, Kotik & Sivov 1965 | 2 | Literature | card written |
| WP3b miters vs literature, impedance walls | 2 | Implementer | card written |
| WP4b mode matching for 90° miters, step parity | 2 | Implementer | card written |
| WP5 tapers and strangled bend | 2 | Implementer | card written |
| WP6 twists | 2 | Implementer | card written |
| WP7b verification, wave 2 | 2 | Verifier | card written |
| CC2 COMSOL campaign 2 | 2 | Coordinator + Daniel | instructions in this PR |
| WP8a studies 1–2 (miters vs bends, chained miters) | — | Implementer | handed off with the SPARC studies (Daniel, 16 Nov) |
| WF2 wave 2 fixes (twist truncation, bend evanescent coupling, strangled-bend port bases) | 2 | Implementer | merged (#29) |
| Gate B: COMSOL campaign 2 | 2 | Coordinator + Daniel | passed with WF3 (#30, #31), 1.0.0b1 |
| WF3 cascades through buried reference planes; passivity check | 2 | Implementer | merged (#31); v1.0.0b1 tagged |
| Wave 3 cards | 3 | Coordinator | this PR |
| P0 article outline, claims, figure list | 3 | Writer | card written; waits for Daniel's decisions |
| WF4 open items before 1.0.0 | 3 | Implementer | card written |
| R1 release 1.0.0 (Gate C) | 3 | Implementer | card written |
| P1 figures, tables, numbers from the release | 3 | Implementer | card written |
| P2 article draft (three checkpoints) | 3 | Writer | card written |
| P3 independent referee review | 3 | Reviewer | card written |
| P4 revision and submission package | 3 | Writer + Daniel | card written |

## Wave 2 (Gate B: 1.0.0b1)

Gate A passed on 11 Nov 2026 (`benchmarks/validation_v1/REPORT.md`, "COMSOL campaign 1 — Gate A").

Wave 2 runs **in series** (Daniel, 11 Nov 2026), one agent at a time, to spread the work over time; nothing in it
needs parallel execution. Order:

1. CC2: `benchmarks/comsol/CAMPAIGN_2.md` (coordinator), first, so the COMSOL runs proceed in the background.
2. L2 (short; its data feed WP3b and WP4b).
3. WP4b, before WP3b, so that the mode-matching miters exist before the FEM work on miters resumes. WP4b reads no
   FEM code.
4. WP3b.
5. WP6.
6. WP5.
7. WP7b. It runs last, after the components are merged, so its independence rests on its rule: build and commit
   every reference before opening the wave 2 code. Its reviewer checks the commit order.

Each step: agent → PR → review → Daniel merges → next. A discrepancy found by WP7b goes back to the owning card as a
fix PR before Gate B. WP8a waits for wave 3. Gate B needs every wave 2 acceptance table met or a discrepancy report
settled, WP7b's comparisons, and campaign 2.

## Wave 3: the article about the tool (Gate C: 1.0.0)

Decided by Daniel on 16 Nov 2026. The work on the tool becomes its own article, with very little about SPARC. The
SPARC bend studies (plan studies 1–6, WP8a) leave this swarm: they become a separate article, written after the
design is chosen and the bends are built and tested. Wave 3 therefore releases 1.0.0 and writes the article. The
hand-off note for the studies is a project doc ("SPARC bend studies: hand-off"), not a file in this public
repository, so that the study plan stays out of it until it is published.

Order, in series as wave 2 (one agent at a time; each step ends with a PR that Daniel merges):

1. **P0** outline, claims table, figure list and the release's needs. Daniel approves before anything else is
   written; it sets the scope of WF4, R1 and P1.
2. **WF4** open items of the wave 2 reports and the library items of `paper/NEEDS.md`.
3. **R1** documentation, examples, packaging; 1.0.0rc1, then 1.0.0. Daniel enables Zenodo, tags `v1.0.0`, makes the
   GitHub release and uploads to PyPI. **Gate C** is this release: every wave 1–2 row met (Gates A and B), the
   guide reviewed from a clean install, clean-install CI on three OSes. No COMSOL campaign 3 is planned; P0 asks for
   one only if a claim cannot stand without it.
4. **P1** figures, tables and numbers, run against 1.0.0.
5. **P2** draft in three checkpoints (methods; validation and examples; introduction, conclusions, abstract).
6. **P3** referee review by an agent that saw none of the writing.
7. **P4** answers to the review, Daniel's and co-authors' revisions, submission package. Daniel submits.

A bug found while writing goes to a fix card and a 1.0.x release, and P1 reruns the figures against it.

**Rules for the article.** Every number comes from P1's generated macros; every claim is in `paper/CLAIMS.md`
with its evidence; statements about other authors' results use wording Daniel approved; nothing from the JINST
paper is reused (it is cited); no SPARC design-study result appears.

### Wave 3 decisions (Daniel)

| Question | Needed by | Decision |
| --- | --- | --- |
| Target journal (template, length, article type) | P0 | open |
| Status of the JINST smooth-bend paper (submitted, accepted, published; how to cite it) | P0 | open |
| Where the manuscript lives: `paper/` in this public repository, or a private repository with only the figure scripts here | P0 | open; default if unanswered: figure scripts here, manuscript in a private repository |
| Author list and internal reviewers before submission | P2 | open |
| Docs hosting (GitHub Pages or Read the Docs) | R1 | open; the build is checked in CI either way |
| Wording of the AI-assistance statements (README, article) | R1, P4 | open; drafts for approval |

### Notes carried from wave 2

The design-study notes (TM12 in strangled-bend tapers, the Butterweck fig. 15 reading, the fitted bottle profile,
twist convergence near cutoffs) moved to the SPARC hand-off note. Those that concern the tool's documentation go
to R1's "known limits" page:

* `profile='bottle'` is wrmodes' fit (λ = 3.5) to an unscaled sketch, so the −40 dB bottle check is not an
  independent validation (WP5).
* Near cutoffs, show convergence in `cutoff_factor` and `mode_factor` (CONVENTIONS §3, WP6, WF2).
* Open issues of WF2, WF3 and WP7b that WF4 does not close.
