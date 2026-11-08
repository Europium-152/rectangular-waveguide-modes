# WP7 — Independent verification, wave 1

Role: Verifier (+ COMSOL scripter) · Wave: 1 · Branch: `wave-1/WP7-verification` · Depends on: WP0 (merged),
L1 for literature data · Verifies: WP1, WP2, WP3, WP4 · Time box: 6 days (then status review)

## Rule

Build every reference from this card, `CONVENTIONS.md` and the literature **only**. Do not open the
implementers' branches or code until your reference results are committed. Then compare, and file a discrepancy
report (`docs/discrepancies/`) for any disagreement beyond tolerance.

## Goal

Independent references and a benchmark registry for the wave 1 components. Also the COMSOL scripts that Daniel
runs on the COMSOL machine for Gate A.

## Specification

1. **Global-ODE reference for lines (verifies WP1).** For a line of straights and smooth bends in the same plane,
   integrate the coupled-mode system across the whole line at once: piecewise curvature, DOP853 at rtol 1e-11,
   transfer matrix of the full line, then $T$ and $R$ by the algebra of the JINST paper. For mixed planes, apply
   the frame change as a coordinate relabelling derived independently from the field definitions. Do not use
   `wrmodes.gsm`. Three reference lines: same plane, mixed planes, and one with 10 m straights.
2. **Surface-integral loss reference (verifies WP2).** For each mode, compute the attenuation as
   $P_{loss}/(2P)$, with $P_{loss} = \tfrac12 R_s\oint|H_{tan}|^2\,dl$ evaluated numerically from
   `transverse_field` and the longitudinal fields derived from it. Cover WR-90 and a square guide, aluminium and
   copper.
3. **COMSOL, Gate A campaign** (scripts in `benchmarks/comsol/`, in the style of the existing Java export; Daniel
   runs them):
   * C1: WR-90 H-plane plain miter, 3D, 18–90 GHz with 1 GHz steps below 30 GHz and 2 GHz above. Export the
     complex S of every propagating mode at both ports, with mode fields matching `CONVENTIONS.md` §4 (state how
     COMSOL's port modes are mapped and phase-referenced).
   * C2: 60 cm 60° sech H-plane bend in aluminium WR-90, with impedance boundary, at 10 frequencies.
   * C3: the strangling step WR-90 ↔ 22.86 × 1.6 mm, at 5 frequencies.

   Include post-processing scripts that convert COMSOL exports to the npz format of `CONVENTIONS.md` §9.
4. **Literature comparisons (verifies WP3, WP4).** Scripts that compare a GSM-producing function with the L1 data
   (D1, D3, D6), using the uncertainties in the CSV files.
5. **Benchmark registry.** Create `benchmarks/registry.json`. Each entry gives: id, component, reference type
   (independent method / COMSOL / literature), data file, tolerance, and the test that checks it.
6. **Adversarial tests.** For each wave 1 component, at least three tests aimed at its weak points: cutoff
   proximity, many modes (30+ per family), extreme aspect ratios, zero-length elements, very long straights (phase
   accuracy over 20 m at 90 GHz), and evanescent modes with large $|\beta|L$.

## Acceptance tests

* Each reference reproduces the existing 0.1 results where they overlap, to 1e-9 in power, with the 0.1 solver run at
  `atol=1e-10` for the comparison (its default `atol=1e-6` gives power errors of about 2e-6).
* The COMSOL scripts run on Daniel's COMSOL version without edits (check the version with him first).
* Every wave 1 acceptance number of WP1–WP4 that cites WP7 has a registry entry.

## Deliverables

* `benchmarks/validation_v1/`: references, comparisons, `REPORT.md`.
* `benchmarks/comsol/` scripts for C1–C3.
* `benchmarks/registry.json`.
* Adversarial tests in `tests/adversarial/`.
* Discrepancy reports, if any.

## Out of scope

Wave 2 components; running COMSOL.
