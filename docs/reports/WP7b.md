# WP7b — Independent verification, wave 2: report

Card `tasks/WP7b-verification-wave2.md`. Branch `wave-2/WP7b-verification`. Full tables:
`benchmarks/validation_v2/REPORT.md`.

## Independence

References built from the cards, CONVENTIONS.md, the theory notes, the literature and the PDFs, with merged wave 1
code as building blocks; wave 2 signatures from docstrings only. References committed in **8ec5755** ("WP7b
references: ..."); adversarial tests, the campaign 2 script and the first discrepancy file in **fc0245e** (seconds
later, same minute), also written before any forbidden file was opened. The session pushed each and `git ls-remote`
returned its hash before the comparisons started; after a server restart the coordinator found the branch absent on
GitHub and pushed the commits again, so GitHub cannot confirm the earlier push. Forbidden material (wave 2 code, tests, benchmarks, reports,
wave 2 discrepancy files) was first opened after fc0245e. Later additions to the reference data: the wave 1 leftover
`evanescent_bend_miter.npz` and the mode-count check `twist_doane_triangular_cf2.npz` (computed with the committed
scripts, unchanged methods).

## What was done

* **Tapers and the strangled bend (WP5)** by the merged 2D FEM on outlines built here (raised-cosine walls as
  chains of circular arcs), and tapers of a and b by a staircase of WP4 steps (parity class, n ≤ 800, cutoff factor
  ≤ 4, Richardson in 1/n).
* **Twists (WP6)** by a new full-wave method: 2D FEM eigenmodes of the twisted guide in helical coordinates
  (Nédélec/Lagrange order 3, quadratic eigenproblem in γ), mode matching to the straight guide at the end planes,
  cascades of constant-rate sections with Richardson in 1/n² for variable rates. No coupled-mode matrix. Doane's
  three 20-inch twists at 15, 16.5, 18 GHz, a square guide, strong twists (τa = 0.1, 0.5).
* **Miters (WP4b, WP3/WP3b)**: FEM on outlines built from CONVENTIONS §6, H and E plane, d/a = 0.707–1.0, WR-90
  10–90 GHz every 2 GHz plus the Gate A points (360 cases), mesh checks, refinement at the worst points.
* **Impedance walls (WP3b with WP2)**: Butterweck's lossy-wall filter, FEM against WP2's exact method.
* **Wave 1 leftovers**: the error of a short bend next to a miter (evanescent modes uncoupled in SmoothBend),
  Marcuvitz §5.24a's N12 misprint, L1 files in the registry.
* **COMSOL campaign 2** post-processing script with tests on synthetic exports.
* **Adversarial tests** (48 in four files) and registry entries.

## Results

| Component | Verdict | Numbers |
| --- | --- | --- |
| WP5 `Taper`, `StrangledBend` (one family) | pass | worst ΔP 2.3e-5, Δ main 2.2e-4 dB (90 GHz) against FEM errors ≤ 3e-5 |
| WP5 tapers of a and b | pass | ΔP 8e-6 (linear), 2.2e-5 (bottle); Δ main ≤ 4e-5 dB |
| WP5 adversarial | pass, one minor discrepancy | port-2 basis order of `StrangledBend` |
| WP6 triangular twists | pass | ΔP 7.7e-6 / 2.9e-5 / 7.4e-5 at 15 / 16.5 / 18 GHz |
| WP6 cosine twists | 15, 16.5 GHz pass; 18 GHz 1.6e-4 | 8.3e-5 at cutoff factor 2 |
| WP6 constant and strong twists | fail at the default | 4e-4 (Doane constant), 4.6e-3 (square, 5λ), 2.4e-4 / 2.6e-2 (τa = 0.1 / 0.5); converge towards the reference with cutoff_factor (1/χ_max) |
| WP4b `MiterMM` vs FEM | pass (card criterion) | Δ main ≤ 0.0018 dB, TE1,0 column ≤ 9.9e-5 over 360 cases; all-entries differences up to 7e-4 at chamfered E-plane points where the FEM reference itself is converged only to 1e-4–6e-4 |
| WP3b impedance walls (filter) | pass | −0.05 % on TE1,0–TE3,0, worst 1.1 % (TE1,5) of 94 modes; target 5 % |
| Doane Fig. 9 | — | the full-wave reference agrees with WP6/T3, not with Fig. 9's triangular and constant curves |
| Marcuvitz §5.24a | located | N12 brace: −R1 − α²R2 (printed + α²R2); with it, (1a) within 0.2 % (a/λ ≤ 1); (2a) within 1.1 % also needs λg′ in its denominator |
| Short bend + miter | reported | Line [SmoothBend, Miter] at zero distance 2–4.2 % of the power off for R = 2a, 0.4–0.6 % for R = 10a (≤ 0.05 dB on TE1,0); the uncoupled evanescent modes, not the bend model, cause it |

## Discrepancy reports

* `docs/discrepancies/2026-11-15-wp7b-twist-truncation.md` (WP6).
* `docs/discrepancies/2026-11-15-wp7b-strangled-bend-port-basis-order.md` (WP5, minor).

## Open issues

1. COMSOL campaign 2 not run yet; script tested on synthetic exports only.
2. Twist reference: unitarity defects up to 7e-4 in non-incident columns for strong twists (mode matching at
   abrupt rate changes); the TE1,0 column is within 1e-4 and converged in N.
3. Miter reference: all-entries accuracy 1e-4 needs density ≥ 24 for chamfered E-plane miters near corner
   resonances; only spot-refined.
4. Not done: square guide 2a/λ = 4.5 triangular twist; the bend + miter study at 60 GHz (memory).
5. `LossyWallFilter` raises RootFindingError at 320 modes per family (clear message; M ≤ 160 works).
6. A 2 % relative difference (8e-5 absolute) in TE1,2 of the triangular twist at 18 GHz between WP6 and the
   reference, inside the tolerance, unexplained; E4 can resolve it only with a ~5e-5 mesh check.

## Decisions for the coordinator or Daniel

1. WP6 constant-rate and strong twists (discrepancy report): raise the default truncation for constant rates /
   offer extrapolation in the mode count, or document the 4e-4 (Doane) to percent-level (τa ≥ 0.3) errors.
2. Accept that the full-wave reference supports WP6/T3 against Doane's Fig. 9 (triangular 2.3–4×, constant TE2,1
   lobe), pending COMSOL E4.
3. Marcuvitz §5.24a: adopt the corrected N12 (sign of α²R2) and λg′ in (2a) in `docs/literature/` as an annotated
   correction (L1 owns the transcription).
4. WP1: whether `SmoothBend` should couple its evanescent modes (0.4–4 % power error for a bend directly at a miter),
   or `Line` should warn when a bend and a discontinuity are closer than a few decay lengths.
5. WP5 minor fix of the port-2 basis order (then drop the strict xfail).
