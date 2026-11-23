# WF4 — Open items before the 1.0.0 release

Role: Implementer · Wave: 3 · Branch: `wave-3/WF4-release-fixes` · Depends on: P0 (merged; its needs list,
copied into item 6 by the coordinator) ·
Checked against: the stored references of WP7b and WF2 (`benchmarks/validation_v2/`) and the frozen tests · Time
box: 5 days (then status review)

## Goal

Close the open items of the wave 2 reports that a 1.0.0 user or the article would hit, and add what P0's needs list
asks of the library. No change to any validated result beyond the bounds stated here.

## Read first

`CONVENTIONS.md`; `docs/reports/WF2.md` and `WF3.md` (Open issues); `docs/reports/WP7b.md` (Open issues);
`components/twist.py`, `filters.py`, `gsm.py`, `line.py`; `benchmarks/registry.json`.

## Items

1. **Run time of variable-rate twists** (WF2 open issue 1): 15–50 s per frequency at the default `mode_factor` 2
   for Doane's guide, against 1.5 s for a constant rate. Profile first and report where the time goes. Candidate
   fixes from WF2: a 6th-order Magnus integrator (fewer slices), grouping slices into transfer-matrix products
   within the growth limit, or reusing the exponentials of the slices (the rate profile changes the scale of
   θ′X + θ′²X2 only). Target: at least 3× faster at the defaults on Doane's cosine and triangular twists.
   Accuracy: every power per mode within 2e-6 of the present defaults (stored in the WF2 test references), and the
   comparison with WP7b's helical FEM no worse than now.
2. **`LossyWallFilter` at 320 modes per family** (WP7b open issue 5): RootFindingError. Fix the root search, or, if
   the limit is intrinsic, raise it as a documented limit with the largest working M in the docstring and the error
   message.
3. **`buried` through `GSM.reversed()`** (WF3 open issue 6): keep the buried-plane data when a GSM is reversed
   (depths swapped), with a test that a reversed component cascades as well as the original. Star products keep
   dropping it (documented).
4. **Registry** (`benchmarks/registry.json`): every entry carries its final status (several still read "waiting for
   data" or "predictions ready" although campaign 2 has run). Add `benchmarks/check_registry.py`: every
   `data_file` exists, every `test` id is collected by pytest, every status is one of a short documented
   vocabulary (pass, not met, qualitative, superseded, reference) with a free-text note. Run it in CI.
5. **Timing harness** (`benchmarks/timings.py`): wall time of one GSM per component at stated sizes and
   frequencies, plus a demonstration line, with the machine description (CPU, cores, BLAS, NumPy and SciPy
   versions) in the output. Daniel runs it on his laptop for the article's run-time table.
6. **Items from P0's needs list** (copied here by the coordinator before launch) that concern the library (e.g. one return format for the `convergence_study`
   functions of Taper, Twist, MiterMM, Step and fem2d). Items that concern documentation go to R1.

## Acceptance

* Full suite, the fem job and the adversarial tests pass; frozen tests unchanged.
* Item 1: the speed-up and the accuracy bound above, measured and reported (table in `docs/reports/WF4.md`).
* Items 2–6: tests for each; `check_registry.py` passes in CI.

## Deliverables

Code, tests, `docs/reports/WF4.md`, updated docstrings, registry.

## Out of scope

New components or physics; the EvanescentJunctionWarning's error estimate (WF2 open issue 4); tails for strong
twists beyond what the docstring documents (WF2 open issue 2).
