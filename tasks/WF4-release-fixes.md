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
6. **Items from P0's needs list** (copied by the coordinator on 24 Nov):
   * **One return format for every `convergence_study`** (Taper/VariableGuide, Twist, MiterMM, Step and fem2d return
     five layouts today): a fixed record with `component`, `f`, `input_mode`, `parameter` (`mode_factor`,
     `cutoff_factor`, `n_modes`, `mesh_density`, `slices`), `values`, `variants` (e.g. `tail` on/off), `n_modes` and
     `powers` per row (mode → |S|² of the input column, transmitted and reflected), `max_change` against the finest row,
     `richardson` where the series supports it (with the assumed order), `wall_time` per row, and `reference`. Keep the
     old keys as aliases for one release.
   * **A `SmoothBend` convergence study on an evanescent basis** (internal mode factor 1.5–8, same record), so that the
     fast convergence of coplanar bends can be shown on the same axes as tapers and twists.
   * **Timing harness sizes** (item 5): lossy straight (WR-90, 90 GHz, all modes); SmoothBend propagating and
     evanescent routes; Taper (raised cosine WR-90 → 1.6 mm, 90 GHz); StrangledBend (E3 geometry, 40 and 90 GHz); Twist
     constant and triangular (Doane, 18 GHz); MiterMM H and E (30, 60, 90 GHz); fem2d.Miter (H, 90 GHz, one family and
     full basis); Step (WR-90 → 1.6 mm, 90 GHz, parity on/off); LossyWallFilter (75 GHz, M = 80); a demonstration line
     per frequency (WR-62, copper, 15–30 GHz; geometry from the article's figure list, given to WF4 at launch). JSON
     output with CPU model, cores, RAM, OS, Python, NumPy, SciPy, BLAS vendor and threads, wrmodes version and commit;
     one BLAS thread by default, a flag for all.
   * **Registry** (item 4), in particular: Gate A results as entries (C1/C1b miters, C2 lossy bends, C3a/C3b steps;
     today `wp3-comsol-c1`, `wp2-comsol-c2`, `wp4-comsol-c3` say "superseded"); an entry for Marcuvitz §5.29b measured
     corners (`benchmarks/fem2d/data/marcuvitz_corners.npz`); final statuses for `wp4-marcuvitz-eplane-step`,
     `wp4-marcuvitz-hplane-step` (→ `wp7b-marcuvitz-524a`), `wave2-alessandri1994`, `wp5-comsol-e2`, `wp5-comsol-e3`,
     `wp6-e4-twist`, `wp7b-campaign2-postprocess`; the finite-difference H-plane step value of WP4's review (X/Z0 =
     11.07 ± 0.05) stored as a data file with `meta`.
   * **Internal amplitudes of a line**: `Line.internal_amplitudes(f, input_mode)` returning the forward and backward
     amplitudes at every junction, so that a trapped-mode resonance can be named (which mode, between which
     components), with a documented recipe.
   * **FEM determinism**: how far `fem2d` results depend on the gmsh version and platform (mesh node order); pin or
     document a range, and state the tolerance figure scripts can promise for FEM results.
   * **`SweepResult.absorbed_power`** (1 − Σ|S|² over the input column, i.e. wall loss), documented.
7. **Doane & Anderson Fig. 9 with Doane's five modes.** Doane (IEEE Trans. MTT 36, 1033, 1988; Table I) computed
   Fig. 9 with TE10, TE01, TE30, TE21 and TM21 in his labels, which are wrmodes' TE1,0, TE0,1, TE0,3, TE1,2 and TM1,2
   in tall WR187 (his first index runs along the broad side). Add to `benchmarks/twist/doane_fig9.py` (or a sibling
   script) the solution restricted to these five modes, and to these five plus wrmodes' TE2,1 (his TE12), on the
   119-point grid of `doane1989_fig9.csv`, with z-scores; store it; add a registry entry; append the result to the
   Resolution of `docs/discrepancies/2026-11-12-doane-fig9-twists.md`. Coordinator check of 24 Nov with
   `twist_gsm_blocks` (five modes): triangular 0.044 / 0.078 / 0.098 / 0.114 / 0.138 % at 15 / 16 / 16.5 / 17 / 18 GHz
   against Fig. 9's 0.049 / 0.080 / 0.099 / 0.116 / 0.149 %; with TE2,1 added, 0.55 % at 18 GHz (full solution 0.60 %).

## Acceptance

* Full suite, the fem job and the adversarial tests pass; frozen tests unchanged.
* Item 1: the speed-up and the accuracy bound above, measured and reported (table in `docs/reports/WF4.md`).
* Items 2–7: tests for each; `check_registry.py` passes in CI.

## Deliverables

Code, tests, `docs/reports/WF4.md`, updated docstrings, registry.

## Out of scope

New components or physics; the EvanescentJunctionWarning's error estimate (WF2 open issue 4); tails for strong
twists beyond what the docstring documents (WF2 open issue 2).
