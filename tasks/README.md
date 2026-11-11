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
| WP8a studies 1–2 (miters vs bends, chained miters) | 3 | Implementer | deferred to wave 3 (Daniel, 11 Nov) |

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
