# WP7b — Independent verification, wave 2: REPORT

Card `tasks/WP7b-verification-wave2.md`. Branch `wave-2/WP7b-verification`. Verifies WP3b, WP4b, WP5, WP6.
All numbers below are reproduced by the scripts of this folder (`PYTHONPATH=src:benchmarks/validation_v2`);
data in `data/*.npz` (CONVENTIONS §9 meta), comparison numbers in `data/compare_*.json`.

## Independence

Every reference was built from the WP7b card, the WP3b/WP4b/WP5/WP6 cards' specifications, `CONVENTIONS.md`, the
theory notes (T1, T2/T2b, T3/T3b, WP3-normalization), the literature CSVs and the reference PDFs (Marcuvitz §5.24),
with merged wave 1 code as building blocks only: `fem2d.Coplanar2D` on outlines built here, `modematch.Step` (WP4,
with WP4b's parity reduction), `GSM`/`cascade`, `wrmodes.modes`, `Line`, `SmoothBend`, `fem2d.Miter` and
`filters.LossyWallFilter` (WP2). The wave 2 call signatures came from their docstrings (`help(...)`).

* References committed and pushed: **8ec5755** ("WP7b references: ..."): all reference scripts and npz data except
  the wave 1 leftover `evanescent_bend_miter.npz` (not a wave 2 reference; committed later) and the two mode-count
  checks `twist_doane_triangular_cf2.npz` (computed after, see below).
* Adversarial tests, campaign 2 script and the first discrepancy file, written from cards and docstrings only:
  **fc0245e**. This is the commit at which forbidden material (wave 2 code, tests, benchmarks, reports, wave 2
  discrepancy files) was first opened; no forbidden file was opened before it.
* Before fc0245e the wave 2 components were *executed* (adversarial tests, docstrings); one adversarial diagnosis
  (the strangled-bend port bases) compared the WP5 `StrangledBend` with the FEM at a geometry that is not one of the
  reference cases. One pytest traceback before fc0245e printed lines of `src/wrmodes/fem2d/components.py` (WP3/WP3b
  2D solver, not on the forbidden list).

## References and their convergence

| Reference | Script | Method |
| --- | --- | --- |
| One-family tapers and the strangled bend (WP5) | `ref_taper_fem.py` | 2D modal-port FEM (`fem2d.Coplanar2D`, order 3) on outlines built here: straight walls (H-plane linear taper), raised-cosine walls as chains of circular arcs through three profile samples each (geometric error O(s⁴); plain chords converged only as 1/n²), exact arcs for the bend |
| Tapers of a and b (WP5) | `ref_taper_staircase.py` | staircase of n sections at the midpoint cross-sections, n + 1 WP4 `Step.solve` junctions on exact bases of the parity class of TE1,0 (m odd, n even), straight phases, Redheffer cascade; Richardson in 1/n |
| Twists (WP6) | `helical_fem.py`, `ref_twist.py` | 2D FEM of the cross-section in helical coordinates (twisted guide = straight guide filled with ε0G, μ0G); third-order Nédélec/Lagrange pair; quadratic eigenproblem in γ, linearized, shift-invert Arnoldi with two shifts for the forward modes; H from curl E; projection on the CONVENTIONS straight-guide fields at the end planes; Galerkin mode matching gives the GSM of a constant-rate section; backward modes from the forward ones by the rotation by π about u (P = (−1)^(n+1)); variable rates as cascades of constant-rate sections at the midpoint rates, Richardson in 1/n². No T3/T3b matrix is used |
| 90° miters (WP4b; FEM side WP3/WP3b) | `ref_miter_fem.py` | 2D FEM on miter outlines built from CONVENTIONS §6 (reference planes through the inner corner, mirror at d·a from it along the bisector), symmetric mesh |
| Lossy-wall filter (WP3b impedance walls) | `ref_filter.py` | WP2 `LossyWallFilter` (exact LSᵘ modes), M = 80 / 120 / 160 |
| Short bend + miter (WP1 leftover) | `ref_evanescent_bends.py` | one FEM domain (arc + optical miter) against `Line` [SmoothBend, Miter], decomposed with the FEM arc alone and a `Line` [FEM arc, Miter] |
| Marcuvitz §5.24a (WP4 open issue 2) | `marcuvitz_n12.py` | WP4 `Step` → Marcuvitz's equivalent circuit (T at the junction, T′ at l on the large side) → the X0, X11, X22 the formulas must reproduce → the N12 the X11 and the X22 equations require |

Twist reference, in short. The helical problem is exact Maxwell for any constant rate; the only approximations are
the FEM mesh (h), the size N of the straight basis used to match at the planes (cutoff factor), and for variable
rates the piecewise-constant rate (n sections). The sections must be short enough that their "grating" wavenumber
2πn/L exceeds every beat Δβ; with n = 8 and 16 the piecewise rate converted resonantly into TE2,1 (1.9e-4 at
n = 16, < 1e-5 at n ≥ 64), so n = 32, 64, 128 is used. Doane's constant twist: h 2.5 → 2 mm and cutoff factor
2 → 2.6 change the total spurious power by ≤ 4e-7 (percent units: 1.43264 / 1.43262 / 1.43260 at 15 GHz); the
coarser settings of the variable rates (h 3 mm, cutoff factor 1.6) agree with them to 5e-7. Triangular at 18 GHz:
cutoff factor 1.6 → 2 changes every mode by ≤ 7e-8; sections 64 → 128 → extrapolated change TE1,2 by 9e-6 and
3e-6. Square guide, 20λ: local polarization angle −72.832° (T3b's coupled-mode table: −72.849°), other modes
4.75e-4 (4.7e-4), reflected 4.24e-5 (4.3e-5). Unitarity of the propagating block 1e-5–7e-4 (the TE1,0 column
4e-6–1e-4); the defects are in columns of near-cutoff modes and shrink with N.

### Convergence of the references
#### Tapers and strangled bend (2D FEM on WP7b outlines)

Reference = finest run (mesh density 16 or 18 elements per transverse wavelength, 64 arcs per curved wall). Changes are max over every entry of the propagating GSM.

| case | f (GHz) | ndof (finest) | max \|ΔS\| mesh (next coarser) | max \|Δ\|S\|²\| mesh | max \|ΔS\| walls (32 arcs) | max \|Δ\|S\|²\| walls | main transmission (dB) | time finest (s) |
|---|---|---|---|---|---|---|---|---|
| H_linear_L100 | 15 | 29169 | 2.9e-06 | 8.6e-09 | – (straight walls) | – | -0.01104 | 0.82 |
| H_linear_L200 | 15 | 53058 | 5.7e-06 | 7.3e-09 | – (straight walls) | – | -0.00286 | 1.52 |
| E_rc50 | 18 | 163524 | 0.0e+00 | 0.0e+00 | 9.4e-07 | 7.6e-09 | -0.00046 | 2.62 |
| E_rc50 | 30 | 327048 | 0.0e+00 | 0.0e+00 | 1.7e-06 | 4.9e-09 | -0.00006 | 4.97 |
| E_rc50 | 60 | 853467 | 2.6e-06 | 2.1e-08 | 4.2e-06 | 1.8e-08 | -0.01559 | 18.31 |
| E_rc50 | 90 | 1989920 | 6.1e-06 | 1.0e-07 | 3.1e-05 | 1.2e-06 | -0.06957 | 57.33 |
| strangled | 18 | 134124 | 1.4e-06 | 3.3e-09 | 9.7e-07 | 3.1e-08 | -0.00092 | 3.75 |
| strangled | 30 | 398166 | 2.6e-06 | 1.3e-08 | 1.7e-06 | 1.8e-08 | -0.00014 | 8.9 |
| strangled | 40 | 740292 | 3.4e-06 | 2.3e-08 | 1.9e-06 | 4.4e-08 | -0.00424 | 19.1 |
| strangled | 60 | 1773537 | 5.8e-06 | 4.5e-08 | 4.3e-06 | 2.4e-08 | -0.03118 | 45.29 |
| strangled | 90 | 4667447 | 9.3e-06 | 1.5e-07 | 3.0e-05 | 1.1e-06 | -0.13919 | 149.27 |

#### 2D tapers (staircase of WP4 steps, class m odd / n even)

| profile | cutoff factor | modes (large end) | TE1,0 (dB) n = 100 / 200 / 400 / 800 | P(TE3,0) n = 400 / 800 | Richardson (1/n) TE1,0 dB, P(TE3,0) | observed order (P(TE3,0)) | time n = 800 (s) |
|---|---|---|---|---|---|---|---|
| linear | 2 | 91 | -0.04453 / -0.04578 / -0.04600 / -0.04606 | 8.807e-03 / 8.816e-03 | -0.04612, 8.826e-03 | 1.55 | 2.1 |
| linear | 3 | 209 | -0.04480 / -0.04595 / -0.04604 / -0.04609 | 8.818e-03 / 8.825e-03 | -0.04615, 8.832e-03 | -1.65 | 10.6 |
| linear | 4 | 367 | -0.04516 / -0.04601 / -0.04611 / -0.04615 | 8.838e-03 / 8.840e-03 | -0.04619, 8.842e-03 | 1.09 | 59.7 |
| bottle_a | 2 | 91 | -0.00741 / -0.00594 / -0.00618 / -0.00628 | 6.444e-05 / 7.224e-05 | -0.00637, 8.004e-05 | 1.62 | 4.4 |
| bottle_a | 3 | 209 | -0.00773 / -0.00601 / -0.00618 / -0.00621 | 9.910e-05 / 9.323e-05 | -0.00621, 8.737e-05 | 1.62 | 22.0 |
| bottle_a | 4 | 367 | -0.00880 / -0.00591 / -0.00607 / -0.00621 | 8.296e-05 / 8.479e-05 | -0.00631, 8.662e-05 | 2.77 | 83.6 |

#### Twists (helical FEM + mode matching)

Spurious = power in every propagating mode other than TE1,0 at port 2 (% of the input). 'main' is the value used; checks: finer mesh (h), more modes (cutoff factor), for variable rates the section count (Richardson from n = 64, 128; 'sec' = max |ΔS21| column between n = 64 and 128, 'ext' = between n = 128 and the extrapolation). Unitarity: max |S^H S - I| of the propagating block; 'col' the same for the TE1,0 column.

| case | f (GHz) | setting | modes | spurious (%) | TE1,0 (dB) | unitarity / col | sec / ext | time (s) |
|---|---|---|---|---|---|---|---|---|
| doane_constant | 15 | main (h 2.5 mm, cf 2) | 69 | 1.43264 | -0.062882 | 1.5e-04 / 1.3e-05 | – | 6.8 |
| doane_constant | 15 | h_check (h 2 mm, cf 2) | 69 | 1.43262 | -0.062881 | 1.5e-04 / 1.3e-05 | – | 10.2 |
| doane_constant | 15 | cf_check (h 2.5 mm, cf 2.6) | 111 | 1.43260 | -0.062881 | 3.9e-05 / 1.2e-05 | – | 14.2 |
| doane_constant | 16.5 | main (h 2.5 mm, cf 2) | 78 | 0.69057 | -0.030282 | 4.9e-05 / 8.3e-06 | – | 8.9 |
| doane_constant | 16.5 | h_check (h 2 mm, cf 2) | 78 | 0.69056 | -0.030282 | 4.9e-05 / 8.3e-06 | – | 15.0 |
| doane_constant | 16.5 | cf_check (h 2.5 mm, cf 2.6) | 135 | 0.69055 | -0.030283 | 1.3e-05 / 3.8e-06 | – | 16.8 |
| doane_constant | 18 | main (h 2.5 mm, cf 2) | 96 | 2.14575 | -0.094296 | 3.3e-05 / 7.4e-06 | – | 10.1 |
| doane_constant | 18 | h_check (h 2 mm, cf 2) | 96 | 2.14575 | -0.094296 | 3.3e-05 / 7.4e-06 | – | 16.2 |
| doane_constant | 18 | cf_check (h 2.5 mm, cf 2.6) | 160 | 2.14575 | -0.094295 | 2.8e-05 / 3.1e-06 | – | 22.3 |
| doane_constant_3mm | 15 | main (h 3 mm, cf 1.6) | 44 | 1.43269 | -0.062879 | 1.5e-04 / 3.7e-05 | – | 4.1 |
| doane_constant_3mm | 16.5 | main (h 3 mm, cf 1.6) | 51 | 0.69052 | -0.030280 | 4.6e-05 / 2.4e-05 | – | 6.2 |
| doane_constant_3mm | 18 | main (h 3 mm, cf 1.6) | 63 | 2.14571 | -0.094294 | 9.9e-05 / 6.6e-06 | – | 6.6 |
| doane_cosine | 15 | main (h 3 mm, cf 1.6) | 44 | 0.34907 | -0.015186 | 6.5e-05 / 5.4e-06 | 3.6e-04 / 1.2e-04 | 805.1 |
| doane_cosine | 16.5 | main (h 3 mm, cf 1.6) | 51 | 0.87142 | -0.038010 | 8.3e-05 / 3.1e-06 | 2.0e-04 / 6.6e-05 | 669.4 |
| doane_cosine | 18 | main (h 3 mm, cf 1.6) | 63 | 0.95988 | -0.041887 | 6.4e-05 / 3.3e-06 | 3.4e-04 / 1.1e-04 | 565.4 |
| doane_triangular | 15 | main (h 3 mm, cf 1.6) | 44 | 0.11435 | -0.004963 | 1.1e-04 / 3.9e-05 | 6.4e-04 / 2.1e-04 | 794.7 |
| doane_triangular | 16.5 | main (h 3 mm, cf 1.6) | 51 | 0.25613 | -0.011129 | 1.0e-04 / 3.6e-05 | 2.8e-04 / 9.4e-05 | 671.8 |
| doane_triangular | 18 | main (h 3 mm, cf 1.6) | 63 | 0.60225 | -0.026229 | 1.0e-04 / 1.2e-05 | 5.9e-04 / 2.0e-04 | 570.1 |
| doane_triangular_cf2 | 18 | main (h 3 mm, cf 2) | 96 | 0.60225 | -0.026229 | 3.7e-05 / 1.3e-05 | 5.9e-04 / 2.0e-04 | 635.2 |
| square_const_5lam | 12.4914 | main (h 2.5 mm, cf 2) | 34 | 89.25377 | -9.719781 | 3.1e-04 / 2.3e-04 | – | 3.6 |
| square_const_5lam | 12.4914 | h_check (h 2 mm, cf 2) | 34 | 89.25376 | -9.719779 | 3.1e-04 / 2.3e-04 | – | 5.7 |
| square_const_5lam | 12.4914 | cf_check (h 2.5 mm, cf 2.6) | 68 | 89.25269 | -9.720219 | 1.6e-04 / 9.8e-05 | – | 6.2 |
| square_const_20lam | 12.4914 | main (h 2.5 mm, cf 2) | 34 | 91.28750 | -10.600670 | 1.6e-05 / 1.2e-05 | – | 3.5 |
| square_const_20lam | 12.4914 | h_check (h 2 mm, cf 2) | 34 | 91.28750 | -10.600670 | 1.5e-05 / 1.2e-05 | – | 5.3 |
| square_const_20lam | 12.4914 | cf_check (h 2.5 mm, cf 2.6) | 68 | 91.28749 | -10.600676 | 4.2e-06 / 4.0e-06 | – | 7.9 |
| strong_tau0p1 | 15 | main (h 2.5 mm, cf 2) | 12 | 0.00250 | -0.001397 | 1.3e-05 / 1.3e-05 | – | 0.8 |
| strong_tau0p1 | 15 | h_check (h 2 mm, cf 2) | 12 | 0.00250 | -0.001397 | 1.3e-05 / 1.3e-05 | – | 1.0 |
| strong_tau0p1 | 15 | cf_check (h 2.5 mm, cf 2.6) | 23 | 0.00251 | -0.001397 | 2.1e-06 / 2.1e-06 | – | 1.3 |
| strong_tau0p1 | 30 | main (h 2.5 mm, cf 2) | 57 | 0.42177 | -0.018418 | 1.8e-05 / 3.0e-06 | – | 2.0 |
| strong_tau0p1 | 30 | h_check (h 2 mm, cf 2) | 57 | 0.42177 | -0.018418 | 1.8e-05 / 3.0e-06 | – | 2.9 |
| strong_tau0p1 | 30 | cf_check (h 2.5 mm, cf 2.6) | 98 | 0.42177 | -0.018418 | 1.0e-05 / 8.6e-07 | – | 7.4 |
| strong_tau0p5 | 30 | main (h 2.5 mm, cf 2) | 57 | 9.07505 | -0.414308 | 7.1e-04 / 9.7e-05 | – | 2.5 |
| strong_tau0p5 | 30 | h_check (h 2 mm, cf 2) | 57 | 9.07525 | -0.414317 | 7.2e-04 / 9.8e-05 | – | 3.1 |
| strong_tau0p5 | 30 | cf_check (h 2.5 mm, cf 2.6) | 98 | 9.07437 | -0.414291 | 2.6e-04 / 4.5e-05 | – | 6.1 |

#### Square guide (30 mm, 2a/λ = 2.5): polarization of TE1,0 after a 90° twist

Local angle = arctan(Re(a01/a10)) of the field at the centre (T3b box 5: −cθ = −72.951° for the two-mode law), power in all other modes, reflected power.

| case | local angle (°) | ellipticity Im(a01/a10)/... | other modes | reflected |
|---|---|---|---|---|
| square_const_5lam | -70.801 | -5.5e-02 | 1.25e-02 | 8.34e-04 |
| square_const_20lam | -72.832 | -2.8e-03 | 4.75e-04 | 4.24e-05 |

#### 90° miters (2D FEM on WP7b outlines, mesh density 10)

Mesh check at 10, 30, 50, 70, 90 GHz: density 16 ('fine') and 20 evanescent port modes per polarization ('more_ports'); max over every entry of the propagating GSM.

| plane | d/a | max \|ΔS\| fine | max \|Δ\|S\|²\| fine | max \|ΔS\| more ports | max \|Δ main\| fine (dB) | reciprocity, S11 = S22 | time 90 GHz (s) |
|---|---|---|---|---|---|---|---|
| H | 0.7071 | 4.6e-05 | 1.5e-05 | 2.1e-11 | 6.5e-05 | 9.7e-14 | 17.28 |
| H | 0.8000 | 3.5e-05 | 1.4e-05 | 1.8e-11 | 1.3e-05 | 6.4e-14 | 30.07 |
| H | 0.9000 | 2.8e-05 | 2.4e-05 | 1.7e-11 | 1.7e-05 | 6.1e-14 | 24.21 |
| H | 1.0000 | 5.7e-05 | 1.6e-05 | 1.2e-11 | 3.0e-05 | 6.6e-14 | 42.25 |
| E | 0.7071 | 5.0e-05 | 2.3e-05 | 5.3e-10 | 2.7e-05 | 5.2e-14 | 5.76 |
| E | 0.8000 | 5.0e-05 | 4.7e-05 | 5.1e-10 | 4.4e-05 | 4.7e-14 | 7.08 |
| E | 0.9000 | 1.7e-04 | 3.9e-05 | 3.6e-10 | 4.0e-05 | 6.4e-14 | 8.37 |
| E | 1.0000 | 6.2e-04 | 2.2e-04 | 2.6e-10 | 1.8e-04 | 4.3e-14 | 9.58 |

#### Lossy-wall filter (WP2 exact LS method), attenuation in dB

| M | TE1,0 | TE2,0 | TE3,0 | TE0,1 | TE1,1 | TM1,1 | time (s) |
|---|---|---|---|---|---|---|---|
| 80 | 0.5691 | 2.3157 | 5.3386 | 2.8355 | 12.2400 | 2.6623 | 19.8 |
| 120 | 0.5689 | 2.3151 | 5.3374 | 2.8356 | 12.2410 | 2.6624 | 61.5 |
| 160 | 0.5688 | 2.3147 | 5.3366 | 2.8356 | 12.2414 | 2.6624 | 177.7 |


## Verification table (one row per verified card row)

Tolerance unless stated: 0.01 dB on the main transmission and 1e-4 of the input power per converted mode
(CONVENTIONS §8, two independent methods). "ΔP" = worst | |S|²(component) − |S|²(reference) | over the modes
converted from the incident TE1,0 (transmitted and reflected). Times are on the 2-core container.

| Card row | Component | Reference | Result | Verdict |
| --- | --- | --- | --- | --- |
| WP5: T2b §9 linear H-plane taper 22.86 → 45 mm, 15 GHz, L = 0.1, 0.2 m | `Taper` | FEM (error 6e-6 in S) | Δ main ≤ 1e-5 dB, ΔP ≤ 1.2e-6 (TE3,0); S21(TE3,0←TE1,0) FEM 0.050058 − 0.005290i (L = 0.1; T2b table 5.004e-2 − 5.35e-3i) | pass |
| WP5: raised-cosine WR-90 → 22.86 × 1.6 mm, 50 mm, 18 / 30 / 60 / 90 GHz | `Taper(plane='E')` | FEM (error ≤ 3e-5 in S, 1.2e-6 in power) | Δ main ≤ 1.1e-4 dB, ΔP ≤ 2.3e-5 (90 GHz, TM2,1 reflected) | pass |
| WP5: strangled bend (E3 geometry), 18 / 30 / 40 / 60 / 90 GHz | `StrangledBend` | FEM (error ≤ 3e-5 in S) | Δ main ≤ 2.2e-4 dB, ΔP ≤ 2.3e-5 (90 GHz) | pass |
| WP5: tapers of a and b (Butterweck, 140 mm, 75 GHz), linear | `Taper` | WP4 staircase (error 1.8e-5 in power) | Δ main 4e-5 dB, ΔP 8e-6 (TE3,0: 8.834e-3 vs 8.842e-3, −20.5 dB) | pass |
| WP5: same, bottle a(z) + linear b(z) | `VariableGuide` | WP4 staircase (error 4.6e-5) | Δ main < 1e-5 dB, ΔP 2.2e-5; TE3,0 8.64e-5 vs 8.67e-5 (−40.6 dB) | pass |
| WP5: strangling just below / above a cutoff, mode at cutoff inside, end within 1e-6 of a cutoff, 1 m taper at 90 GHz | `Taper`, `StrangledBend` | adversarial | finite, lossless, reciprocal, symmetric (by mode name), adiabatic 1 m taper within 0.02 rad of WKB | pass; **port-2 basis order differs from port 1 for 25 of 200 flat heights** (discrepancy, minor) |
| WP6: Doane's 20-inch twists, triangular, 15 / 16.5 / 18 GHz | `Twist` | helical FEM (error ≤ 2.4e-5) | ΔP 7.7e-6 / 2.9e-5 / 7.4e-5; spurious 0.1143 / 0.2561 / 0.6023 % (ref) vs 0.1142 / 0.2474 / 0.6030 % | pass |
| WP6: Doane cosine | `Twist` | helical FEM (error ≤ 2.2e-5) | ΔP 2.9e-5 / 4.4e-5 / **1.6e-4** (18 GHz, TE0,1 0.922 vs 0.905 %); cutoff factor 2: 8.3e-5 | 15, 16.5 pass; 18 GHz fails at the default |
| WP6: Doane constant | `Twist` | helical FEM (error ≤ 6e-7) | ΔP **4.1e-4** / 4.2e-5 / **4.0e-4** (TE2,1 0.131 vs 0.090 % at 15 GHz); converges towards the reference with cutoff_factor (8e-5 at 6) | 16.5 pass; 15, 18 fail (discrepancy) |
| WP6: square guide, 30 mm, 2a/λ = 2.5, 90° constant, L = 5λ / 20λ | `Twist` | helical FEM | ΔP **4.6e-3** (Δ TE1,0 −0.14 dB) / **1.7e-4** | fail (discrepancy) |
| WP6: strong twist, WR-90, τa = 0.1 (15, 30 GHz), τa = 0.5 (30 GHz) | `Twist` | helical FEM | ΔP 6.6e-5 / **2.4e-4** / **2.6e-2** (TE0,1 6.62 vs 4.05 %) | τa = 0.1 at 15 GHz pass; others fail (discrepancy) |
| WP6: 360° twist, nearly square guide, reversal, near cutoff | `Twist` | adversarial | 360° = two 180° to 1e-8; nearly square continuous to 1e-3; symmetric profile is its own reverse | pass |
| WP4b: FEM agreement, 90°, H and E plane, d/a = 0.707, 0.8, 0.9, 1.0, WR-90 10–90 GHz + Gate A points (360 cases) | `MiterMM` | FEM on WP7b outlines (mesh error ≤ 6e-5 in S, 1.7e-4 / 6.2e-4 for E d = 0.9 / 1.0) | Δ main ≤ 0.0018 dB; ΔP of the TE1,0 column ≤ 9.9e-5 (H, d = 1.0, 26 GHz, 0.9 % below the TE4,0 cutoff) | pass (card criterion); all-entries metric see miter section |
| WP3b/WP4b: chained miters with zero spacer, intersecting 45° mirrors, reactive impedance walls | `chained_miter_outline`, `Coplanar2D(wall_impedance)` | adversarial | U-turn refused (ValueError), periscope against the zero-length cascade |ΔS21| < 1e-2, passive and reciprocal for 250+250j, 5−400j, 0.05+0.05j Ω | pass |
| WP3b (with WP2): Butterweck lossy-wall filter, 75 GHz, 250 Ω, 15 cm, 5 % target | `Coplanar2D(wall_impedance=250)` | WP2 `LossyWallFilter`, M = 160 | TE1,0 / TE2,0 / TE3,0: −0.054 / −0.058 / −0.054 %; worst of 94 attenuated modes TE1,5 −1.07 % | pass |
| Wave 1: evanescent modes of bends (WP1 open issue) | `Line` [SmoothBend, Miter] | one FEM domain | see section; error 2–4 % of power at zero distance, dominated by the uncoupled evanescent modes | reported |
| Wave 1: Marcuvitz §5.24a N12 (WP4 open issue 2) | — | WP4 `Step` | misprint located: the brace of N12 should read −R1 − α²R2 (printed + α²R2) | reported |
| Wave 1: L1 literature files in the registry | — | — | `butterweck1968_numbers.csv`, `ma1997_table1.csv`, `docs/literature/doane1989.md` added | done |
| CC2: campaign 2 post-processing | `compare_campaign_2.py` | synthetic exports | 7 tests (parsing, S11–S81, de-embedding, criteria, missing files) | ready; waiting for data |

## Comparisons

### WP5 one-family tapers and the strangled bend vs the 2D FEM (WP7b outlines)

| case | f (GHz) | FEM main (dB) | FEM error est. (\|ΔS\|, mesh / walls) | WP5 Δ main (dB) | WP5 worst ΔP (mode) | pass | WP5 mode_factor 3: Δ main / worst ΔP | WP5 time (s) |
|---|---|---|---|---|---|---|---|---|
| H_linear_L100 | 15 | -0.01104 | 2.9e-06 | +0.00001 | 1.2e-06 (2:TE3,0) | yes | +0.00000 / 4.6e-07 | 0.1 |
| H_linear_L200 | 15 | -0.00286 | 5.7e-06 | +0.00000 | 1.1e-07 (2:TE3,0) | yes | +0.00000 / 5.9e-08 | 0.1 |
| E_rc50 | 18 | -0.00046 | 0.0e+00, 9.4e-07 | -0.00000 | 2.0e-08 (1:TE0,1) | yes | -0.00000 / 2.0e-08 | 0.0 |
| E_rc50 | 30 | -0.00006 | 0.0e+00, 1.7e-06 | -0.00000 | 1.4e-09 (1:TE0,1) | yes | -0.00000 / 6.9e-11 | 0.2 |
| E_rc50 | 60 | -0.01559 | 2.6e-06, 4.2e-06 | -0.00001 | 2.4e-06 (1:TM2,1) | yes | -0.00000 / 7.0e-07 | 1.4 |
| E_rc50 | 90 | -0.06957 | 6.1e-06, 3.1e-05 | -0.00011 | 2.3e-05 (1:TM2,1) | yes | -0.00003 / 6.7e-06 | 5.5 |
| strangled | 18 | -0.00092 | 1.4e-06, 9.7e-07 | -0.00000 | 3.0e-08 (1:TE0,1) | yes | -0.00000 / 4.6e-08 | 0.1 |
| strangled | 30 | -0.00014 | 2.6e-06, 1.7e-06 | -0.00000 | 5.5e-09 (1:TE0,1) | yes | -0.00000 / 2.5e-09 | 0.8 |
| strangled | 40 | -0.00424 | 3.4e-06, 1.9e-06 | -0.00000 | 4.0e-07 (2:TM2,1) | yes | -0.00000 / 7.2e-08 | 1.5 |
| strangled | 60 | -0.03118 | 5.8e-06, 4.3e-06 | -0.00002 | 2.4e-06 (2:TM2,1) | yes | -0.00001 / 7.0e-07 | 7.6 |
| strangled | 90 | -0.13919 | 9.3e-06, 3.0e-05 | -0.00022 | 2.3e-05 (1:TM2,1) | yes | -0.00006 / 6.7e-06 | 27.1 |

### WP6 twists vs the helical-coordinate FEM (incident TE1,0, powers in % of the input)

| case | f (GHz) | ref spurious | ref error est. | WP6 spurious | WP6 Δ main (dB) | WP6 worst ΔP (mode) | pass | WP6 cutoff_factor 2: worst ΔP | largest modes (ref → WP6) |
|---|---|---|---|---|---|---|---|---|---|
| doane_constant | 15 | 1.4326 | 6.4e-07 | 1.3798 | +0.00225 | 4.1e-04 (2:TE2,1) | NO | 2.2e-04 | TE0,1 1.103→1.086, TE2,1 0.131→0.090, TE0,3 0.113→0.117 |
| doane_constant | 16.5 | 0.6906 | 2.0e-07 | 0.6930 | -0.00011 | 4.2e-05 (2:TE0,1) | yes | 5.0e-05 | TE2,1 0.609→0.609, TE1,2 0.028→0.029, TE0,1 0.028→0.032 |
| doane_constant | 18 | 2.1457 | 1.2e-07 | 2.1373 | +0.00037 | 4.0e-04 (2:TE2,1) | NO | 1.8e-04 | TE0,1 1.787→1.820, TE0,3 0.184→0.183, TE2,1 0.116→0.076 |
| doane_cosine | 15 | 0.3491 | 1.4e-05 | 0.3519 | -0.00012 | 2.9e-05 (2:TE0,1) | yes | 2.4e-05 | TE0,1 0.342→0.345, TE0,3 0.004→0.004, TE2,1 0.001→0.001 |
| doane_cosine | 16.5 | 0.8714 | 1.2e-05 | 0.8672 | +0.00018 | 4.4e-05 (2:TE0,1) | yes | 2.6e-05 | TE0,1 0.853→0.848, TE0,3 0.013→0.013, TE1,2 0.005→0.004 |
| doane_cosine | 18 | 0.9599 | 2.2e-05 | 0.9432 | +0.00073 | 1.6e-04 (2:TE0,1) | NO | 8.3e-05 | TE0,1 0.922→0.905, TE1,2 0.023→0.023, TE2,1 0.005→0.005 |
| doane_triangular | 15 | 0.1143 | 1.2e-05 | 0.1142 | +0.00000 | 7.7e-06 (2:TM1,2) | yes | 3.5e-05 | TE1,2 0.080→0.080, TE0,3 0.019→0.019, TM1,2 0.011→0.011 |
| doane_triangular | 16.5 | 0.2561 | 7.8e-06 | 0.2474 | +0.00037 | 2.9e-05 (2:TE1,2) | yes | 6.3e-05 | TE1,2 0.172→0.169, TE0,3 0.045→0.043, TM1,2 0.025→0.023 |
| doane_triangular | 18 | 0.6023 | 2.4e-05 | 0.6030 | -0.00004 | 7.4e-05 (2:TE1,2) | yes | 9.8e-05 | TE1,2 0.376→0.384, TE0,3 0.161→0.158, TM1,2 0.049→0.046 |
| square_const_5lam | 12.4914 | 89.2538 | 1.1e-05 | 89.5972 | -0.14009 | 4.6e-03 (2:TE0,1) | NO | 3.4e-03 | TE1,0 10.666→10.328, TE1,2 0.608→0.540, TE2,1 0.570→0.523 |
| square_const_20lam | 12.4914 | 91.2875 | 1.5e-07 | 91.3040 | -0.00829 | 1.7e-04 (2:TE0,1) | NO | 1.0e-04 | TE1,0 8.708→8.692, TE1,2 0.022→0.022, TE2,1 0.022→0.022 |
| strong_tau0p1 | 15 | 0.0025 | 1.0e-07 | 0.0091 | -0.00036 | 6.6e-05 (2:TE0,1) | yes | 6.3e-05 | TE0,1 0.002→0.009, TE2,0 0.000→0.000 |
| strong_tau0p1 | 30 | 0.4218 | 1.6e-08 | 0.3979 | +0.00104 | 2.4e-04 (2:TE0,1) | NO | 3.8e-05 | TE0,1 0.421→0.397, TE2,1 0.000→0.000, TE3,0 0.000→0.000 |
| strong_tau0p5 | 30 | 9.0750 | 5.8e-06 | 6.0360 | +0.14286 | 2.6e-02 (2:TE0,1) | NO | 4.3e-03 | TE0,1 6.618→4.046, TM2,1 2.028→1.559, TE2,1 0.321→0.345 |

### WP5 tapers of a and b (140 mm, 75 GHz, TE1,0 from the small end) vs the WP4 staircase

Staircase reference: Richardson in 1/n from n = 400, 800 at the largest cutoff factor; error estimate = |extrapolated(cf 4) - extrapolated(cf 3)| + |S(800) - extrapolated|.

| profile | ref TE1,0 (dB) | ref P(TE3,0) | ref error est. (max ΔP) | WP5 Δ main (dB) | WP5 worst ΔP (mode) | pass | WP5 P(TE3,0) |
|---|---|---|---|---|---|---|---|
| linear | -0.04619 | 8.842e-03 | 1.8e-05 | +0.00004 | 8.0e-06 (2:TE3,0) | yes | 8.834e-03 |
| bottle_a | -0.00631 | 8.667e-05 | 4.6e-05 | +0.00000 | 2.2e-05 (1:TE1,0) | yes | 8.641e-05 |

### WP4b mode-matching miters (and the WP3/WP3b FEM miter) vs the FEM on WP7b outlines, 90°, WR-90

360 cases per component (45 frequencies × 4 chamfers × 2 planes). Columns: reference mesh error (max |ΔS| between
density 10 and 16 at 10, 30, 50, 70, 90 GHz); MM: worst Δ TE1,0 transmission, worst power difference of the modes
converted from TE1,0 (reflected and transmitted; the card's criterion) and the frequencies where it fails, the same
over every propagating entry of the GSM (all inputs; stricter than the card); `fem2d.Miter` at its default mesh;
MM time at 90 GHz.

| plane | d/a | ref mesh error | MM worst Δ main (dB) @ f | MM worst ΔP, TE1,0 column @ f | MM fails, column | MM worst ΔP, all entries @ f | MM fails, all entries (GHz) | FEM Miter worst Δ main | FEM Miter worst ΔP (column / all) | MM time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| H | 0.7071 | 4.6e-05 | -0.0002 @ 14 | 2.5e-05 @ 10 | none | 7.3e-05 @ 56 | none | +0.0007 @ 14 | 4.5e-05 / 1.6e-04 | 0.82 |
| H | 0.8000 | 3.5e-05 | -0.0003 @ 20 | 2.4e-05 @ 20 | none | 8.8e-05 @ 14 | none | +0.0003 @ 20 | 3.5e-05 / 2.6e-04 | 3.88 |
| H | 0.9000 | 2.8e-05 | -0.0002 @ 20 | 3.2e-05 @ 26.3 | none | 7.5e-05 @ 26.3 | none | +0.0007 @ 26 | 5.3e-05 / 1.7e-04 | 2.63 |
| H | 1.0000 | 5.7e-05 | -0.0018 @ 26 | 9.9e-05 @ 26 | none | 1.1e-04 @ 26 | 26 | +0.0038 @ 26 | 1.8e-04 / 2.2e-04 | 2.37 |
| E | 0.7071 | 5.0e-05 | -0.0003 @ 34 | 2.9e-05 @ 18 | none | 1.4e-04 @ 18 | 16, 18, 26 | -0.0003 @ 20 | 2.2e-05 / 1.1e-04 | 0.75 |
| E | 0.8000 | 5.0e-05 | -0.0005 @ 24 | 2.8e-05 @ 18 | none | 3.0e-04 @ 20 | 16, 18, 20, 36 | -0.0003 @ 32 | 1.7e-05 / 1.6e-04 | 4.98 |
| E | 0.9000 | 1.7e-04 | -0.0006 @ 24 | 3.7e-05 @ 24 | none | 7.0e-04 @ 66 | 36, 64, 66, 72, 78 | -0.0005 @ 36 | 2.2e-05 / 9.2e-04 | 2.29 |
| E | 1.0000 | 6.2e-04 | -0.0006 @ 20 | 4.3e-05 @ 24 | none | 7.1e-04 @ 32 | 20, 30, 32, 36, 44, 46, 58, 60, 68, 84, 90 | -0.0007 @ 30 | 4.0e-05 / 9.1e-04 | 2.81 |

Refinement of the reference at the worst all-entries points (E plane, d/a = 1.0; max | |S|² difference | over all
entries / over the TE1,0 column, against the finest FEM):

| f (GHz) | FEM density 10 | 16 | 24 | MM default | MM, M raised ×3 |
|---|---|---|---|---|---|
| 32 (finest: density 32) | 5.8e-4 / 1.8e-5 | 2.4e-4 / 4.6e-6 | 9.3e-5 / 1.3e-6 | 1.2e-4 / 9.0e-6 | 1.4e-4 / 1.6e-6 |
| 44 (finest: density 24) | 3.7e-4 / 6.9e-6 | 1.1e-4 / 2.0e-6 | – | 1.7e-4 / 8.0e-6 | 1.8e-4 / 1.6e-6 |

### WP3b impedance walls vs WP2's exact lossy-wall filter (Butterweck: 75 GHz, 250 Ω, 15 cm)

| mode | WP2 M = 160 (dB) | WP2 M = 80 | FEM density 6 (default) | FEM density 8 | Δ rel. (density 8) | pass (5 %) | Butterweck |
|---|---|---|---|---|---|---|---|
| TE1,0 | 0.5688 | 0.5691 | 0.5685 | 0.5685 | -5.38e-04 | yes | 0.6 |
| TE2,0 | 2.3147 | 2.3157 | 2.3134 | 2.3134 | -5.79e-04 | yes | 2.5 |
| TE0,1 | 2.8356 | 2.8355 | 2.8357 | 2.8357 | +3.82e-06 | yes |  |
| TE1,1 | 12.2414 | 12.2400 | 12.2424 | 12.2422 | +7.16e-05 | yes |  |
| TM1,1 | 2.6624 | 2.6623 | 2.6625 | 2.6625 | +3.71e-05 | yes |  |
| TE3,0 | 5.3366 | 5.3386 | 5.3338 | 5.3338 | -5.36e-04 | yes | 5.4 |
| TE4,0 | 9.8794 | 9.8835 | 9.8738 | 9.8738 | -5.72e-04 | yes |  |
| TE1,2 | 9.1405 | 9.1387 | 9.1418 | 9.1417 | +1.25e-04 | yes |  |
| TE6,0 | 25.3776 | 25.3836 | 25.3679 | 25.3678 | -3.86e-04 | yes |  |

Worst over all 94 attenuated modes: TE1,5 -1.07e-02 relative.

### Reference self-checks against the literature

* Kotik & Sivov (1965) Fig. 3, optical-mirror H-plane miter: the WP7b FEM sweep (2 GHz grid) lies within the digitizing uncertainty at 19 of 26 grid points with a CSV point within 0.05 in kd; outside: kd 12.60: 0.6656 vs 0.7584 ± 0.0070, kd 22.04: 0.9456 vs 0.9535 ± 0.0054, kd 25.15: 0.8970 vs 0.9208 ± 0.0056, kd 25.39: 0.9066 vs 0.9146 ± 0.0053, kd 28.75: 0.9547 vs 0.9627 ± 0.0052, kd 29.70: 0.9600 vs 0.9685 ± 0.0058, kd 30.66: 0.9593 vs 0.9703 ± 0.0066.
* Doane & Anderson (1989) Fig. 9, constant: 15 GHz: 1.433 % vs Fig. 9 1.303 ± 0.030 %; 16.5 GHz: 0.691 % vs Fig. 9 0.051 ± 0.011 %; 18 GHz: 2.146 % vs Fig. 9 1.685 ± 0.046 %.
* Doane & Anderson (1989) Fig. 9, cosine: 15 GHz: 0.349 % vs Fig. 9 0.351 ± 0.010 %; 16.5 GHz: 0.871 % vs Fig. 9 0.885 ± 0.010 %; 18 GHz: 0.960 % vs Fig. 9 1.048 ± 0.012 %.
* Doane & Anderson (1989) Fig. 9, triangular: 15 GHz: 0.114 % vs Fig. 9 0.049 ± 0.010 %; 16.5 GHz: 0.256 % vs Fig. 9 0.099 ± 0.011 %; 18 GHz: 0.602 % vs Fig. 9 0.149 ± 0.014 %.

## Findings per component

### WP5 (tapers, strangled bend)

All acceptance cases agree with the independent references by one to two orders of magnitude inside the tolerance:
the cross-section method with B5′ at its default `mode_factor` 2 is within 2.3e-5 of the FEM in power (worst at
90 GHz, a reflected TM2,1 near its cutoff in the taper) and within 2.2e-4 dB on TE1,0; `mode_factor` 3 brings it to
6.7e-6. The 2D tapers agree with the staircase within the staircase's own error (TE3,0 of the linear taper −20.54 dB
in both; bottle −40.6 dB in both, which only confirms the solver, not Butterweck's measurement, since the bottle
profile is wrmodes' fit). WP5's 2D-taper default at 75 GHz takes about 10 min (all families coupled, 75 GHz in
WR-90).

Adversarial: a `StrangledBend` returns a port-2 basis that differs from port 1 for some flat heights (exactly
degenerate modes of WR-90 in the E frame reordered by one rounding error in a(L)); within a `Line` this is harmless,
standalone cascades fail. `docs/discrepancies/2026-11-15-wp7b-strangled-bend-port-basis-order.md`, strict xfail.

### WP6 (twists)

The full-wave reference confirms the coupled-mode physics of T3/T3b and WP6 where the rate is smooth: the
triangular twist is within 7.4e-5 at 15, 16.5 and 18 GHz, including the second-order conversion to TE1,2, TE0,3,
TM1,2 that Doane & Anderson's Fig. 9 does not show (reference 0.114 / 0.256 / 0.602 % against Fig. 9's
0.049 / 0.099 / 0.149 %). It also confirms the constant twist's 0.61 % TE2,1 lobe at 16.5 GHz. These are
independent of the coupled-mode matrices, so Fig. 9 is not what Maxwell's equations give for these geometries
(COMSOL E4 will check this a third way).

Where the rate changes abruptly or is strong, WP6's default truncation (propagating modes, completed) is outside the
1e-4 tolerance: Doane's constant twist 4e-4 (TE2,1, TE0,1), cosine at 18 GHz 1.6e-4, the strong cases 2.4e-4 to
2.6e-2. WP6 converges towards the reference as `cutoff_factor` grows, slowly (about like 1/χ_max), e.g. 5.1e-4 →
2.8e-4 → 1.4e-4 → 8.1e-5 at cutoff factors 1, 2, 4, 6 for the constant twist at 15 GHz. The triangular twist moves
by up to 1e-4 with the cutoff factor (TE1,2 0.3836 / 0.3859 / 0.3844 % at 1 / 2 / 3), while the reference stays at
0.3761 % (cutoff factor 1.6 and 2 agree to 7e-8): a 2 % relative difference inside the tolerance that is not
explained. `docs/discrepancies/2026-11-15-wp7b-twist-truncation.md`.

### WP4b (mode-matching miters) and the FEM miters

The card's criterion (TE1,0 transmission within 0.01 dB, every mode converted from TE1,0 within 1e-4 of the input
power) is met at all 360 frequency points: worst 0.0018 dB and 9.9e-5 (H plane, d/a = 1.0, 26 GHz, 0.9 % below the
TE4,0 cutoff). Over every entry of the propagating GSM (all inputs, stricter than the card) MM and the reference
differ by more than 1e-4 at 24 of 360 points, all in E-plane miters with d/a ≥ 0.8 or near cutoffs, up to 7.1e-4.
At these points the reference itself is not converged to 1e-4 at density 10 (its own mesh check: 1.7e-4 for E,
d/a = 0.9 and 6.2e-4 for d/a = 1.0, in the families with the sharp corner resonances WP4b describes). Refinement at
two of them (E plane, d/a = 1.0): at 32 GHz the FEM moves by 5.8e-4 / 2.4e-4 / 9.3e-5 from density 10 / 16 / 24 to
32, and MM (default, or with M raised to 2151) is within 1.2e-4–1.4e-4 of density 32; at 44 GHz the FEM moves by
3.7e-4 / 1.1e-4 from density 10 / 16 to 24 and MM is within 1.7e-4 of density 24. The FEM converges slowly there
(order about 0.7–1 in h, irregular), so the all-entries comparison at chamfered E-plane points is **unresolved at
the 1e-4 level**: the MM–FEM differences (≤ 7e-4) are of the size of the FEM's own remaining change, and the TE1,0
column, which the card judges, agrees to ≤ 9e-6 at both points. Not a discrepancy beyond the card's tolerance;
recorded as an open issue.

`fem2d.Miter` at its default mesh (density 6) is within 0.0038 dB and 1.8e-4 (TE1,0 column; 9.2e-4 over all
entries) of the density-10 reference: adequate for transmission, not for 1e-4 per entry in chamfered E-plane miters.
MM is faster (0.7–6 s per frequency at 90 GHz, including its error control).

### WP3b (impedance walls)

The FEM with Leontovich walls reproduces WP2's exact filter within 0.06 % on TE1,0–TE3,0 (Butterweck's filter),
1.1 % at worst over the 94 attenuated modes (TE1,5), both inside the plan's 5 %. Z → 0 is linear to 1e-10 and equal
to the perfect miter within the FEM discretization; strongly reactive walls stay passive and reciprocal.

## Wave 1 leftovers

### Marcuvitz §5.24a: the N12 misprint (WP4 open issue 2, L1 open issue 1)

Reference: the WP4 mode-matching `Step` (default truncation, converged to 1e-6 per the WP4 report; X/Z0 = 11.02 at
α = 0.8, a/λ = 0.85, as WP4 and the review's FD 11.07 ± 0.05), converted to Marcuvitz's circuit (T at the junction,
T′ a distance l into the larger guide, Fig. 5.24-1; |Γ2| from the circuit equals |S22| to 1e-15 at every point,
which checks the conversion). Inverting (1a)–(3a) gives the X0, X11, X22 the formulas must reproduce; X0 agrees
with the static A′ − C²/A within the expected few percent, X11 and X22 are off by factors of 3–30. Solving the X11
equation and, separately, the X22 equation for N12 with the printed N11 and N22 gives two values that agree with
each other to 0–3 % (12 % at a/λ = 1.39, α = 0.4) on a grid a/λ = 0.6–1.39, α = 0.4–0.9, and are 5–25 times smaller
than the printed N12. So the misprint is in N12, not in N11, N22, (1a) or (2a).

Among the sign variants of the four terms of the brace of N12, one stands out (rms 26 % in N12 against 1360 % as
printed; all others ≥ 88 %): the term α²R2 must be **subtracted**, i.e.

    N12 = 2(a/λ)² {α′² − (4E(α)/π)[E(α′) − α²F(α′)] − R1 − α²R2} T + (the two Q terms as printed).

This is also the structure N11 and N22 suggest: their first lines are 1 − R1² − α²T² and α² − α²R2² − T², whose
cross term is −(R1 + α²R2)T. With this single sign change, (1a) agrees with mode matching to ≤ 0.2 % for
a/λ ≤ 1 and ≤ 1.8 % at a/λ = 1.39, and (2a) to 1–8 %; with, in addition, λg′ instead of λg in the denominator of
(2a) (the other suspected misprint), (2a) agrees to ≤ 1.1 % for a/λ ≤ 1, ≤ 4.2 % at 1.2 and ≤ 3.7 % at 1.39
(except α = 0.9 at a/λ = 1.39, 12.8 %, where X is 26 times Z0). Marcuvitz states < 1 % over most of the range and up
to 5 % at a/λ = 1.5: both corrections together reproduce that accuracy; the N12 correction alone does for (1a) but
not quite for (2a). (3a) with the static X0 is 6–90 % off in l/a, but l is below 7e-4 a (a phase of 1e-3 rad), so
this matters little. L1's evaluation at a/λ = 0.5 is outside the formulas' range (2a′/λ > 1 needs α > 1 there).
Table: `marcuvitz_524a.npz`, below.

### Evanescent modes of bends (WP1 open issue)

A short circular bend (centre-line radius R = 2a or 10a, length 0–2a) directly followed by an optical 90° miter in
WR-90, against one FEM domain containing both (FEM mesh change ≤ 3e-5). The decomposition separates the bend model
(SmoothBend against the FEM arc alone: ≤ 8e-4 in power at R = 2a, the coupled-mode bend with propagating modes is
accurate even for R = 2a) from the zero-distance cascade: a `Line` [FEM arc, Miter] with propagating modes only is
off by about 2 % of the power (1.7–2.1e-2), and with the evanescent modes at cutoff factor 3, which the FEM arc
couples, by ≤ 2e-4. The `Line` [SmoothBend, Miter] stays at 1.8–3.0e-2 even with cutoff factor 3, because
SmoothBend passes the evanescent modes as uncoupled straight modes. So the WP1 open issue costs about 2–3 % of the
power (and up to 0.05 dB on TE1,0) whenever a miter or step sits within a fraction of a wavelength of a bend; a
straight of a few evanescent decay lengths between them, or a bend that couples its evanescent modes, removes it.
Table below (20 and 40 GHz; 60 GHz was dropped, the fine-mesh runs exceeded the container's 7 GB).

### L1 literature files in the registry (L1 open issue 7)

Added where missing: `butterweck1968_numbers.csv` (the items not covered by WP5's entries: B1–B11, P1–P4, P7),
`ma1997_table1.csv` (its entry had no data file) and `docs/literature/doane1989.md`. The other L1 files were already
registered by WP3/WP3b/WP4b/WP5/WP6 and L2.

## COMSOL campaign 2 post-processing

`benchmarks/comsol/campaign_2/compare_campaign_2.py` (style of campaign 1): reads the campaign-1 global-evaluation
export and the table export, de-embeds the port distances of CAMPAIGN_2.md (15 mm for E1–E4, 37.86 mm for E5),
compares phases modulo 180°, and applies the Gate A criterion (mesh-check change + 0.02 dB on the main transmission;
for E3 the PEC mesh check also bounds the aluminium model). E4 is judged on the power of each of the seven output
modes (ports 2–8, names of the CAMPAIGN_2 table) with mesh-check change + 1e-4, and shows the WP7b helical-FEM
values next to WP6's. E1 shows both `fem2d.Miter` and `MiterMM`, E5 both the single chained domain and the
`Line` cascade. Missing files are skipped with a `[skip] <file>: not found` message. Tested on synthetic exports
generated by `tests/adversarial/test_wp7b_campaign2.py` (7 tests). Status: ready; COMSOL not yet run.

WP7b's predictions for E4 (helical FEM, % of the input, TE1,0 incident): 15 GHz: TE1,2 0.0799, TE0,3 0.0186,
TM1,2 0.0114, TE2,1 0.0034, TE0,1 0.0010; 18 GHz: TE1,2 0.376, TE0,3 0.161, TM1,2 0.049, TE2,1 0.0148,
TE0,1 0.0003; cosine at 18 GHz: TE0,1 0.922, TE1,2 0.023. Fig. 9 allows 0.049 % (15 GHz) and 0.149 % (18 GHz) in
total for the triangular twist.
#### Marcuvitz Sec. 5.24a (symmetric H-plane step) against the WP4 mode-matching Step

Normalized reactance Xλg/(2aZ0) and impedance ratio Z0'λg a/(Z0λg'a'): relative error of Marcuvitz's (2a)/(1a) as printed, with N12 corrected (sign of α²R2), and with the corrected N12 and λg' in the denominator of (2a). N12 'required' = the value the X11 and the X22 equations need (with the printed N11, N22) to reproduce mode matching.

| a/λ | α | MM X | printed | N12 corr. | N12 corr. + λg' | MM Z-ratio | printed | N12 corr. | N12 printed | N12 required (X11 / X22 eq.) | N12 corrected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.60 | 0.9 | 28.4368 | +2.000 | -0.0096 | -0.0027 | 1.1051 | +0.0073 | -0.0000 | 0.04458 | 0.00183 / 0.00184 | 0.00116 |
| 0.70 | 0.8 | 7.8980 | +0.580 | -0.0155 | -0.0008 | 1.2128 | +0.0157 | -0.0000 | 0.08267 | 0.01090 / 0.01099 | 0.00744 |
| 0.70 | 0.9 | 28.3726 | +10.268 | -0.0084 | -0.0015 | 1.1050 | +0.0503 | -0.0000 | 0.06069 | 0.00231 / 0.00231 | 0.00159 |
| 0.85 | 0.7 | 3.6372 | +0.393 | -0.0242 | -0.0020 | 1.3164 | +0.0352 | -0.0004 | 0.12833 | 0.03494 / 0.03565 | 0.02535 |
| 0.85 | 0.8 | 8.0169 | +1.272 | -0.0144 | +0.0005 | 1.2118 | +0.0354 | -0.0001 | 0.12220 | 0.01526 / 0.01535 | 0.01127 |
| 0.85 | 0.9 | 28.2304 | -3.920 | -0.0042 | +0.0027 | 1.1048 | -0.0108 | +0.0000 | 0.08955 | 0.00322 / 0.00322 | 0.00240 |
| 0.99 | 0.6 | 2.0286 | +0.278 | -0.0338 | -0.0040 | 1.4120 | +0.0582 | -0.0016 | 0.16471 | 0.07291 / 0.07594 | 0.05559 |
| 0.99 | 0.7 | 3.7936 | +0.696 | -0.0254 | -0.0023 | 1.3137 | +0.0623 | -0.0009 | 0.17539 | 0.04690 / 0.04761 | 0.03569 |
| 0.99 | 0.8 | 8.1664 | +3.733 | -0.0116 | +0.0037 | 1.2104 | +0.1302 | -0.0001 | 0.16628 | 0.02071 / 0.02081 | 0.01581 |
| 0.99 | 0.9 | 28.0194 | -2.223 | +0.0042 | +0.0112 | 1.1045 | -0.0060 | +0.0000 | 0.12157 | 0.00434 / 0.00435 | 0.00336 |
| 1.20 | 0.5 | 1.2998 | +0.247 | -0.0493 | -0.0092 | 1.4946 | +0.0986 | -0.0052 | 0.21472 | 0.13235 / 0.14138 | 0.10665 |
| 1.20 | 0.6 | 2.3326 | +0.632 | -0.0438 | -0.0098 | 1.4035 | +0.1298 | -0.0050 | 0.24951 | 0.11193 / 0.11467 | 0.08919 |
| 1.20 | 0.7 | 4.1931 | +2.283 | -0.0294 | -0.0038 | 1.3071 | +0.2385 | -0.0026 | 0.26224 | 0.07357 / 0.07421 | 0.05699 |
| 1.20 | 0.8 | 8.5132 | -6.006 | -0.0016 | +0.0147 | 1.2070 | -0.0717 | -0.0003 | 0.24612 | 0.03307 / 0.03321 | 0.02504 |
| 1.20 | 0.9 | 27.4213 | -1.609 | +0.0350 | +0.0423 | 1.1037 | -0.0035 | +0.0002 | 0.17896 | 0.00686 / 0.00687 | 0.00527 |
| 1.39 | 0.4 | 0.8737 | +0.184 | -0.0722 | -0.0185 | 1.5667 | +0.1301 | -0.0100 | 0.23725 | 0.18506 / 0.20841 | 0.15436 |
| 1.39 | 0.5 | 1.7509 | +0.609 | -0.0816 | -0.0295 | 1.4807 | +0.2356 | -0.0182 | 0.31083 | 0.20044 / 0.20822 | 0.16584 |
| 1.39 | 0.6 | 3.0888 | +2.840 | -0.0803 | -0.0372 | 1.3888 | +0.7012 | -0.0183 | 0.35376 | 0.17507 / 0.17674 | 0.13865 |
| 1.39 | 0.7 | 5.1212 | -27.026 | -0.0518 | -0.0213 | 1.2950 | -0.2206 | -0.0092 | 0.36334 | 0.11923 / 0.11948 | 0.08796 |
| 1.39 | 0.8 | 9.2211 | -2.296 | +0.0159 | +0.0344 | 1.2000 | -0.0266 | -0.0007 | 0.33477 | 0.05544 / 0.05563 | 0.03814 |
| 1.39 | 0.9 | 26.1002 | -1.410 | +0.1196 | +0.1278 | 1.1019 | -0.0011 | +0.0008 | 0.24095 | 0.01146 / 0.01151 | 0.00791 |

## Adversarial tests (`tests/adversarial/`)

* `test_wp7b_adversarial.py` (main CI job; 21 tests, one slow, one strict xfail): WP5: a mode exactly at cutoff at
  the middle of a taper (continuity in f to 1e-5), a taper end within 1e-6 and 2e-7 of a cutoff (clean result or the
  documented ValueError), strangling just below / above the cutoff of the curvature-coupled mode (lossless,
  reciprocal, S11 = S22 by mode name), port-basis equality of the strangled bend (**strict xfail**, discrepancy), a
  1 m raised-cosine taper at 90 GHz against the WKB phase (slow, 25–60 s), a taper turned around against the reverse
  taper. WP6: 360° = 2 × 180°, a slow 360° twist is adiabatic, a nearly square guide is continuous with the square
  one, the symmetric (triangular) twist is its own reverse, 1e-5 above the TE0,1 cutoff. WP4b: S11 = S22 without the
  symmetry shortcut, 1e-6 above a cutoff (H and E), the −90° miter is the mirror image, a zero-spacer U-turn of two
  MM miters (lossless; cutoff factor 3 vs 4), the parity reduction of an offset 2D step.
* `test_wp7b_fem_adversarial.py` (fem job, added to its command in `.github/workflows/tests.yml`; 7 tests): WP3b
  impedance walls with Z → 0 and strongly reactive Z, a refused zero-spacer U-turn, a zero-spacer periscope
  against the cascade, intersecting 45° mirrors.
* `test_wp7b_references.py` (13 tests, one strict xfail): the components against the stored WP7b references at
  cheap points (T2b tapers, raised-cosine taper, strangled bend, triangular twists, MM miters); Doane's constant
  twist at 15 GHz is a strict xfail (discrepancy).
* `test_wp7b_campaign2.py` (7 tests): the campaign 2 script on synthetic exports.

No timing assertions (CI runners can be 3× slower).

## Discrepancy reports

* `docs/discrepancies/2026-11-15-wp7b-twist-truncation.md` (WP6): default truncation 4e-4 (constant) to 2.6e-2
  (τa = 0.5) off the full-wave reference; converges slowly with `cutoff_factor`.
* `docs/discrepancies/2026-11-15-wp7b-strangled-bend-port-basis-order.md` (WP5, minor): port-2 basis order.

No tolerance was changed and no component was modified.

## Literature status (qualitative curves, reported only)

Per Daniel's decisions (registry): Alimenti Figs. 4–6, Ma's chamfered Fig. 11 and Table 1, Ma's c/a = 1 at 40–60
steps, Reiter Fig. 3(a) γ = 150, Figs. 3(b)–(d) and Fig. 4 are qualitative; WP7b did not use them as pass/fail and
did not add new comparisons with them. Self-checks of the references against quantitative literature:

* Kotik & Sivov (1965) Fig. 3 (optical H-plane miter): the FEM sweep is within the digitizing uncertainty at 19 of
  26 points; the seven others lie on the steep dips just above the TE4,0, TE8,0 cutoffs and near kd = 29–31 where the
  printed line is faint (0.008–0.024 off; at kd = 12.60, 0.27 % above the TE4,0 cutoff, 0.666 vs 0.758).
* Doane & Anderson (1989) Fig. 9: see WP6 above (cosine within 1σ at 15 GHz and 1.4σ at 16.5 GHz, 7σ low at 18 GHz;
  constant and triangular far off).

## Open issues

1. COMSOL campaign 2 not yet run; the script is tested on synthetic files only. E4 per-mode accuracy needed: ~5e-5
   to resolve the 2 % TE1,2 difference between WP6 and WP7b at 18 GHz, 2e-4 to decide against Fig. 9.
2. Twist reference: unitarity defects of non-incident columns up to 7e-4 (strong twist, square 5λ) from the
   mode matching at abrupt rate changes; the TE1,0 column is within 1e-4 everywhere, and the incident-column powers
   change by < 1e-6 with N. A symmetric (reciprocity-preserving) matching would remove the defect.
3. Miter reference: density 10 is not enough for 1e-4 over all entries of chamfered E-plane miters (d/a ≥ 0.9)
   near the corner resonances; refined points only (density 16–32). A complete density-24 sweep would cost ~1 day
   on 2 cores.
4. Square guide 2a/λ = 4.5 triangular twist (T3b table) not computed (127 modes, ~3 h).
5. The evanescent-bend study at 60 GHz was dropped (memory).
6. `wrmodes.filters.LossyWallFilter` raises RootFindingError at M = 320 (family n = 4, orthogonality 1.1e-6); M ≤ 160
   works. Clear error; noted for WP2.

## Timings (2-core container, other jobs running)

Taper FEM references 1–150 s per run (strangled bend at 90 GHz, 4.7 M dof: 150 s); staircase 2–84 s per run (n =
800, 367 modes); helical FEM: 1–25 s per constant twist, 10–13 min per variable-rate twist and frequency (112
eigen-solves); miter FEM sweep 0.3–37 s per point (≈ 3 h for 360 points with mesh checks); WP5 `VariableGuide` for
the 2D taper at 75 GHz ≈ 10 min; WP6 Twist 0–1 s (default), 35–320 s at cutoff factor 2–3 (triangular).
