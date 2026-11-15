# WF2 — Wave 2 fixes before Gate B

Role: Implementer · Wave: 2 · Branch: `wave-2/WF2-fixes` · Depends on: wave 2 (merged) · Checked against: WP7b's
committed references (`benchmarks/validation_v2/`) · Time box: 5 days (then status review)

## Goal

Fix the three findings of WP7b (`benchmarks/validation_v2/REPORT.md`, `docs/reports/WP7b.md`) that Daniel accepted on
15 Nov 2026, so that every wave 2 row is met before Gate B. The verifier's references already exist and are
independent; this card makes the components meet them.

## Read first

`CONVENTIONS.md`; `docs/reports/WP7b.md` and `benchmarks/validation_v2/REPORT.md`; the discrepancy files
`docs/discrepancies/2026-11-15-wp7b-twist-truncation.md` and `…-wp7b-strangled-bend-port-basis-order.md`; the
reports and code of WP1 (`components/bends.py`), WP5 (`components/taper.py`) and WP6 (`components/twist.py`).

## 1. Twist truncation (WP6)

At default settings the twist is 4e-4 of the input power off for constant-rate twists, 1.6e-4 for the cosine twist
at 18 GHz and up to 2.6e-2 for strong twists, and it converges toward WP7b's helical-FEM reference as the mode count
grows (about like 1/χ_max). The docstring's "(τ/k)²" claim does not hold for abrupt ends.

* Give `Twist` an **internal** mode list larger than its port bases (as WP5's `mode_factor`), with a default chosen
  from convergence, and/or Richardson extrapolation in the mode count; the port bases stay the line's.
* Correct the docstring's convergence statement; `convergence_study` reports the extrapolated value.
* Acceptance: at default settings, within 1e-4 of the input power per mode of WP7b's reference for Doane's constant,
  cosine and triangular twists at 15, 16.5, 18 GHz, the square guide at L = 5λ and 20λ, and τa = 0.1 at 15 and
  30 GHz. τa = 0.5 at 30 GHz: reach it with an option, or document the limit and the setting that gets closest.
  Report the run time at the new defaults (Doane's three twists, 3 frequencies).
* The strict xfail in `tests/adversarial/test_wp7b_references.py` that points to the twist discrepancy becomes a pass.

## 2. Bends next to discontinuities (WP1)

A `Line` [SmoothBend, Miter] at zero distance is off by up to 4.2e-2 of the input power (0.05 dB) at R = 2a, because
`SmoothBend` does not couple evanescent modes; raising `cutoff_factor` does not help. `Line` [FEM arc, Miter] with
`cutoff_factor = 3` is within 1.6e-4 (R = 2a) and 2.5e-5 (R = 10a).

* When the bases passed to `SmoothBend.gsm` contain evanescent modes, build the GSM with evanescent modes coupled:
  the direct slice-GSM method WP5 uses (for example by delegating to `VariableGuide` with constant a, b and κ(z), or
  shared code). The path with propagating modes only stays **bitwise unchanged**.
* `Line` warns once when a component that does not couple evanescent modes sits directly against a non-uniform
  component while the line's bases carry no evanescent modes there (define "directly" and document it).
* Acceptance: WP7b's bend-next-to-miter cases (`benchmarks/validation_v2/ref_evanescent_bends.py`) at
  `cutoff_factor = 3` within 2e-4 (R = 2a) and 3e-5 (R = 10a) of the single FEM domain; lossy bends unchanged on
  propagating modes to 1e-12; reciprocity, unitarity, passivity per CONVENTIONS §6; timing reported.

## 3. Strangled-bend port bases (WP5)

In 25 of 200 flat heights the port-2 basis order differs from port 1 (rounding in a(L) swaps the degenerate E-frame
modes TE4,0 / TE0,9 of WR-90). Build both port bases from exactly the same guide; the strict xfail in
`tests/adversarial/test_wp7b_adversarial.py` becomes a pass.

## Deliverables

Code and tests; `docs/reports/WF2.md`; updated rows in `benchmarks/validation_v2/REPORT.md` (a short "after WF2"
section, not a rewrite) and `benchmarks/registry.json`; the two discrepancy files get a "Resolution" section. Do not
modify CONVENTIONS.md (propose changes in the report).
