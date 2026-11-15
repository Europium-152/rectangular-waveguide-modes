# Discrepancy: WP6 `Twist` at its default truncation vs WP7b's full-wave twist reference

*Opened by WP7b on 2026-11-15. Owner: WP6 (settled by a theory agent and Daniel). Data:
`benchmarks/validation_v2/data/twist_*.npz` (reference), `benchmarks/validation_v2/data/compare_twists.json`
(comparison), `benchmarks/validation_v2/REPORT.md` (tables).*

## Reference

WP7b solved the twists without the T3/T3b coupled-mode matrices: eigenmodes of the uniformly twisted guide from a
2D finite-element eigenproblem in helical coordinates (third-order Nédélec/Lagrange elements, straight guide filled
with ε0G, μ0G), mode matching to the straight guide at the end planes (and at the planes between constant-rate
sections for variable rates, Richardson in 1/n² with n = 32, 64, 128 sections). Its own convergence, in power per
mode for incident TE1,0: mesh h 2.5 → 2 mm ≤ 2e-7, straight-basis cutoff factor 2 → 2.6 ≤ 6e-7 (constant rates);
cutoff factor 1.6 → 2 ≤ 7e-8 and sections 128 → extrapolated ≤ 3e-6 (triangular, 18 GHz). Where WP6 is run with
growing `cutoff_factor`, it converges towards this reference (table below), which is the strongest evidence that
the reference is right.

## Observation

Card tolerance (CONVENTIONS §8): 1e-4 of the input power per converted mode, 0.01 dB on TE1,0. `Twist` with its
default `cutoff_factor=None` (port basis of propagating modes, completed by the missing propagating modes):

| case | f (GHz) | spurious, reference / WP6 (%) | worst ΔP (mode) | WP6 with cutoff_factor 2 | Δ TE1,0 (dB) |
| --- | --- | --- | --- | --- | --- |
| Doane constant, WR187 tall, 20 in | 15 | 1.4326 / 1.3798 | 4.1e-4 (TE2,1: 0.131 vs 0.090 %) | 2.2e-4 | +0.0023 |
| Doane constant | 18 | 2.1457 / 2.1373 | 4.0e-4 (TE2,1: 0.116 vs 0.076 %) | 1.8e-4 | +0.0004 |
| Doane cosine | 18 | 0.9599 / 0.9432 | 1.6e-4 (TE0,1: 0.922 vs 0.905 %) | 8.3e-5 | +0.0007 |
| 30 mm square, 2a/λ = 2.5, L = 5λ, constant | 12.49 | 89.254 / 89.597 | 4.6e-3 (TE1,0 stays 10.67 vs 10.33 %) | 3.4e-3 | −0.140 |
| same, L = 20λ | 12.49 | 91.2875 / 91.3040 | 1.7e-4 | 1.0e-4 | −0.008 |
| WR-90, τa = 0.1, constant | 30 | 0.4218 / 0.3979 | 2.4e-4 (TE0,1) | 3.8e-5 | +0.0010 |
| WR-90, τa = 0.5, constant | 30 | 9.075 / 6.036 | 2.6e-2 (TE0,1: 6.62 vs 4.05 %) | 4.3e-3 | +0.143 |

Within tolerance: Doane constant at 16.5 GHz, cosine at 15 and 16.5 GHz, triangular at 15, 16.5, 18 GHz (worst
7.4e-5), WR-90 τa = 0.1 at 15 GHz.

**Convergence of WP6 in `cutoff_factor`** (max |ΔP| against the reference over all propagating modes; the largest
mode in brackets, reference value last):

| case | None | 1.5 | 2 | 3 | 4 | 6 | reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Doane constant 15 GHz, TE2,1 (%) | 0.0900 | 0.1093 | 0.1093 | 0.1171 | 0.1207 | 0.1253 | 0.1313 |
| max \|ΔP\| | 5.1e-4 | 2.7e-4 | 2.8e-4 | 1.9e-4 | 1.4e-4 | 8.1e-5 | |
| Doane constant 18 GHz, max \|ΔP\| | 4.0e-4 | 2.5e-4 | 2.4e-4 | 1.9e-4 | 1.5e-4 | 9.8e-5 | |
| square 5λ, TE1,0 (%) | 10.328 | 10.802 | 10.841 | 10.792 | 10.773 | 10.746 | 10.667 |
| max \|ΔP\| | 4.6e-3 | 2.9e-3 | 3.4e-3 | 2.1e-3 | 1.7e-3 | 1.3e-3 | |
| WR-90 τa = 0.5, TE0,1 (%) | 4.046 | 7.198 | 7.047 | 6.930 | 6.863 | 6.787 | 6.618 |
| max \|ΔP\| | 3.0e-2 | 6.7e-3 | 4.9e-3 | 3.6e-3 | 2.8e-3 | 1.9e-3 | |
| WR-90 τa = 0.1 30 GHz, max \|ΔP\| | 2.4e-4 | 5.6e-5 | 3.8e-5 | 2.8e-5 | 2.2e-5 | 1.5e-5 | |

WP6 approaches the reference monotonically from cutoff factor 1.5 on, slowly (roughly like 1/χ_max, as T3 §7 and
T3b §5 predict for the coupled-mode series): with 6 × f it is still 8e-5–1e-4 off for Doane's constant twist and
1.3e-3–1.9e-3 off for the strong cases.

The triangular twist (smooth rate, zero at both ends) does not need the evanescent modes in the same way: WP6 agrees
within 7.4e-5 at its default; with cutoff factor 2 and 3 it moves by up to 1e-4 (TE1,2 0.384 → 0.386 → 0.384 %,
TE0,3 0.158 → 0.165 → 0.165 % at 18 GHz) while the reference stays at 0.376 % and 0.161 % (cutoff factor 1.6 and 2
agree to 7e-8). The remaining 2 % relative difference in TE1,2 at 18 GHz is inside the tolerance and not explained.

## Diagnosis (WP7b's view, not a ruling)

* Where the rate changes abruptly (constant twists, both ends) the field near the walls has the normal magnetic
  component w_n H_z that no straight-guide mode carries (T3 §7); the coupled-mode series converges like 1/χ_max, and
  the end junctions convert power into modes that are strongly coupled to TE1,0 at first order (TE2,1 in tall WR187,
  |K| = 1.68, WP6 report). The WP6 docstring states that truncation effects are of order (τ/k)² for τa ≪ 1; for
  Doane's constant twist (τ/k ≈ 0.01) the measured error is 4e-4 at the default and 1e-4 at cutoff factor 6, so the
  statement does not cover constant (abrupt) rates.
* The helical FEM represents the twisted modes exactly (to the mesh), so its only truncation is the mode matching
  at the end planes, which converges fast (cutoff factor 2 → 2.6 changes the powers by < 1e-6).
* Doane & Anderson's Fig. 9 and COMSOL E4 are not affected: the reference confirms WP6's and T3's triangular
  values (0.114 / 0.256 / 0.602 % against Fig. 9's 0.049 / 0.099 / 0.149 %) and the constant twist's TE2,1 lobe
  (0.609 % of TE2,1 at 16.5 GHz in both). E4 decides against Fig. 9 independently.

## Requested decision

1. WP6 acceptance for constant-rate and strong twists: either (a) a default that reaches 1e-4 (e.g. a larger
   internal truncation for constant rates, or Richardson extrapolation in 1/χ_max, which the table above supports),
   or (b) document the error (4e-4 for Doane's constant twist, percent level for τa ≥ 0.3) in the docstring and the
   report, and keep the default.
2. The 2 % relative difference of TE1,2 in the triangular twist at 18 GHz (8e-5 absolute, inside the tolerance) is
   recorded for E4: COMSOL's per-mode powers can resolve it only if their mesh check reaches ~5e-5.
