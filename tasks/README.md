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

| Card | Wave | Role | Depends on | Status |
| --- | --- | --- | --- | --- |
| T0 field convention | 0 | Theory | — | done (`docs/theory/T0-field-convention.md`) |
| WP0 foundations | 0 | Coordinator | — | in review (this PR) |
| [T1 wall impedance](T1-wall-impedance.md) | 1 | Theory | T0 | ready |
| [T2 taper coupling](T2-taper-coupling.md) | 1 | Theory | T0 | ready |
| [T3 twist coupling](T3-twist-coupling.md) | 1 | Theory | T0 | ready |
| [L1 literature data](L1-literature-data.md) | 1 | Literature | — | ready |
| [WP1 cascade](WP1-cascade.md) | 1 | Implementer | WP0 | ready |
| [WP2 ohmic loss and filters](WP2-loss-and-filters.md) | 1 | Implementer | WP0, T1 | waits for T1 |
| [WP3 2D modal-port solver](WP3-2d-solver.md) | 1 | Implementer | WP0, WP1 (interface only) | ready |
| [WP4 steps and mode matching](WP4-steps.md) | 1 | Implementer | WP0 | ready |
| [WP7 verification, wave 1](WP7-verification-wave1.md) | 1 | Verifier + COMSOL scripter | WP0, L1 | ready |

Wave 2 cards (all miters, Alessandri segmentation, tapers, twists, COMSOL campaign 2) are written at Gate A.
