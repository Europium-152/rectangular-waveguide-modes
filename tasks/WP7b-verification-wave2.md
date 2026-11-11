# WP7b — Independent verification, wave 2

Role: Verifier · Wave: 2 · Branch: `wave-2/WP7b-verification` · Depends on: wave 1 (merged); compares against
WP3b, WP4b, WP5, WP6 when they land · Verifies: WP3b, WP4b, WP5, WP6 · Time box: 9 days (then status review)

## Rule

Build every reference from this card, the WP5/WP6/WP4b cards' specifications, `CONVENTIONS.md`, the theory notes
named below and the literature **only**. Do not open the implementers' branches or code until your reference
results are committed. Then compare, and file a discrepancy report (`docs/discrepancies/`) for any disagreement
beyond tolerance. You may use merged wave 1 code (it is validated) as a building block, but not the wave 2
component you are checking.

## Specification

1. **Tapers and the strangled bend (verifies WP5), by a different method than the cross-section method:**
   * H-plane tapers (a(z) only) and E-plane strangled bends (b(z) and κ(z) in the E plane): one family each, so
     the merged 2D FEM (`fem2d.Coplanar2D`, polygon outlines that you build) is an exact independent method.
     Cases: the T2b §9 linear taper (22.86 → 45 mm, 15 GHz, L = 0.1 and 0.2 m); a raised-cosine WR-90 →
     22.86 × 1.6 mm taper of 50 mm at 18, 30, 60, 90 GHz; a strangled bend of two such tapers and a 90° E-plane arc
     of R = 50 mm in the flat guide (the COMSOL campaign 2 geometry).
   * Tapers of both a and b (Butterweck's bottle and linear tapers, 140 mm, 75 GHz): a staircase of merged WP4
     `Step`s with Richardson extrapolation in the step count and the mode count, where feasible (2D steps converge
     slowly, WP4 open issue 5; use WP4b's parity reduction when it lands); otherwise report how far it got.
   * Tolerance: 0.01 dB on the main transmission, 1e-4 of the input power per converted mode (CONVENTIONS §8).
2. **Twists (verifies WP6), by a method that does not use the T3/T3b coupled-mode matrices.** For example: the
   eigenmodes of a uniformly twisted guide from a 2D finite-element or spectral eigenproblem of the cross-section in
   helical coordinates, matched to the straight guide at both ends (constant rate), and a stepped rotation (many
   short rotated straights with overlap matrices) extrapolated in the step size for variable rates. Cases:
   Doane & Anderson's three 20-inch 90° twists in WR187 at 15, 16.5, 18 GHz; a square guide; a strong short twist
   (τa = 0.1). Report the spurious power per mode (the modes of T3 §6, in the CONVENTIONS frame of the guide).
3. **Miters (verifies WP3b and WP4b):** FEM vs mode matching at 90°, H and E plane, d/a = 0.707, 0.8, 0.9, 1.0,
   WR-90 at 10–90 GHz (every 2 GHz, plus the near-cutoff points of Gate A), with the CONVENTIONS §8 tolerance.
4. **Mode filters (verifies WP2 with WP3b):** the lossy-wall filter by the FEM with impedance walls against WP2's
   exact LS method; the 5 % target of the plan.
5. **Wave 1 leftovers.**
   * The evanescent modes of bends are uncoupled in WP1 (WP1 open issue): quantify the error of a `Line` [short
     `SmoothBend`, `Miter`] against one `Coplanar2D` domain containing both, for bend lengths 0–2a.
   * Marcuvitz §5.24 (WP4 open issue 2): locate the suspected misprint in $N_{12}$ (and the $\lambda_g$ of (2a))
     independently, with the merged mode-matching `Step` as the reference.
   * Add the L1 literature files to `benchmarks/registry.json` (L1 open issue 7).
6. **COMSOL campaign 2 post-processing:** `benchmarks/comsol/campaign_2/compare_campaign_2.py` in the style of
   campaign 1, for the cases of `benchmarks/comsol/CAMPAIGN_2.md` (written by the coordinator).
7. **Adversarial tests** (`tests/adversarial/`), at least three per wave 2 component: a mode exactly at cutoff
   inside a taper, a taper whose end is within 1e-6 of a cutoff, strangling to $b_1$ just below and above a
   cutoff, very long tapers (1 m) at 90 GHz, a 360° twist, a twist of a nearly square guide (degenerate pairs),
   chained miters with zero spacer.

## Acceptance tests

* Each reference shows its own convergence (mesh, modes, steps) with numbers.
* Every wave 2 acceptance row that cites WP7b has a registry entry and a comparison result in `REPORT.md`.

## Deliverables

* `benchmarks/validation_v2/` (references, comparisons, `REPORT.md`), `benchmarks/comsol/campaign_2/` script,
  registry entries, adversarial tests, discrepancy reports if any, `docs/reports/WP7b.md`.

## Out of scope

Writing the wave 2 components; COMSOL instructions (coordinator); design studies.
