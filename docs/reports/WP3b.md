# WP3b — Miters against the literature; impedance walls in the 2D solver: report

Branch `wave-2/WP3b-miters`, from `main` at ff9e797 (WP4b merged). Card `tasks/WP3b-miters-and-impedance-walls.md`.
All numbers below are for the committed code; every table is printed by a script in `benchmarks/fem2d/`, whose
`.npz` output (with `meta`, CONVENTIONS.md §9) is in `benchmarks/fem2d/data/`.

## What was done

| Item | Files |
| --- | --- |
| Impedance (Leontovich) walls on the in-plane walls: `wall_impedance` (number, callable Z_s(f) or `'guide'`) on `Coplanar2D` and `Miter`; exact Robin condition for family n = 0, coupled LSM-LSE condition for n ≥ 1; complex-symmetric system; perfect arms; mesh refinement at the ends of the impedance walls | `src/wrmodes/fem2d/solver.py` (`solve_impedance`, `Discretization.wall_matrices`), `components.py`, `geometry.py` (`extra_points`) |
| Derivation (signs included) | `docs/theory/WP3-normalization.md` §6 |
| `chained_miter_outline(angles_deg, spacers, chamfers=None, *, a, port_extension=0.0)` → (Outline, symmetry line, centre-line length); overlapping mirrors; refusal of self-overlapping geometries | `src/wrmodes/fem2d/components.py` |
| Literature scripts (FEM at the digitized abscissae, z = (ours − digitized)/uncertainty) | `benchmarks/fem2d/{ma1997, alessandri1994, alimenti1994, kotik1965, reiter1995}.py`, helpers `_lit.py` (incl. Ma's staircase miters) |
| Impedance-wall and chained-miter benchmarks | `benchmarks/fem2d/impedance_walls.py`, `chained_miters.py` |
| Tests (17 fast, 5 slow) | `tests/test_fem2d.py` (section "WP3b") |
| Registry | `benchmarks/registry.json`: 9 entries updated, 7 added (`wp3b-*`) |

Unchanged: the exact-cutoff rule (a family at k_t = 0 is uncoupled straight modes, also with impedance walls), the
'live' rule, the frame tags, the lossless code path (bitwise: the existing tests and `Z_s = 0` give identical
arrays), `src/wrmodes/modematch/`, CONVENTIONS.md.

## Impedance walls

### Derivation (WP3-normalization.md §6)

On an in-plane wall with outward normal n̂ and ŝ = v̂ × n̂, B1 of T1 reads E_s = Z_s H_v, E_v = −Z_s H_s. Writing the
family-n field as an LSM part (electric Hertz vector v̂ P cos k_v v′) plus an LSE part (magnetic Hertz vector
v̂ R sin k_v v′), with the solver's scalars ψ = k_t²P (E_v profile) and φ = k_t²R (H_v profile):

  ∂_n ψ = −(iη₀/k) (k_t² ψ / Z_s + k_v ∂_s φ),   ∂_n φ = −(i/(kη₀)) (Z_s k_t² φ + k_v ∂_s ψ).

* n = 0: a Robin condition on ψ alone, ∂_nψ = −iη₀kψ/Z_s, which is T1 B6 for LSE^u (TE_m0).
* n ≥ 1: the **two polarizations couple** through the tangential derivatives. A pure LSM field has E_s ≠ 0 on the
  wall, which the Leontovich condition forbids, so the card's "a Robin condition on ψ for LSM and on φ for LSE" is
  exact only for n = 0. The family index stays exact (the v = ±b/2 walls stay perfect conductors). This is why the
  exact modes of an impedance-walled guide are LS^u modes (T1 §4–5). The coupling is implemented, and it matters:
  with its sign reversed the TE1,1 / TM1,1 attenuation of a 5 cm lossy straight is off by 22 % / 120 %.
* Weak form: scaling the ψ equation by 1/η₀ and the φ equation by −η₀ makes the coupled matrix complex symmetric
  (the coupling matrix C_ij = ∮ w_i ∂_s w_j is antisymmetric on the free unknowns because ψ = 0 where the
  impedance walls meet the perfect arms; tested to 1e-11). The same scaling is the one for which the LS amplitudes
  are reciprocal (g²β/η₀ = −(−η₀h²β) = 4k_t²/(kb)), an independent check of the relative normalization of ψ and φ.
* Choices: the arms (behind the reference planes) stay perfect conductors, so the GSM is the lossy core between
  perfect guides, as `LossyWallFilter` and WP2's first-order straights; the walls at v = ±b/2 are not included
  (they couple families; docstring); the mesh is refined at the ends of the impedance walls (the wall condition
  jumps there: without it the TE0,2 reflection near cutoff converged 10× slower).

### Acceptance

**Per-mode attenuation** (`impedance_walls.py` §1): aluminium (σ = 3.5e7 S/m) on the narrow walls of WR-90, H
plane. α from the transfer-matrix eigenvalues of straight outlines of 0.2 m and 0.6 m (12 elements per
wavelength), extrapolated in 1/L (the discretization error of the two junctions enters as c/L; the raw values at
0.6 m are 1–14e-6 low).

| f | n | β (1/m) | α exact B6 (Np/m) | FEM, extrapolated | first order (`wrmodes.loss`) |
| --- | --- | --- | --- | --- | --- |
| 10 GHz | 0 | 158.24 | 4.441043e-3 | −1.5e-7 | +1.4e-4 |
| 30 GHz | 0 | 613.55 / 565.50 / 474.72 / 305.23 | 6.6133e-4 … 2.1270e-2 | +5.8e-7 / +3.1e-7 / +6.4e-8 / +5.3e-8 | +6.6e-5 … +1.3e-4 |
| 30 GHz | 1 (LSE^u) | 529.94 / 473.47 / 360.21 | 7.657e-4 … 1.014e-2 | +6.7e-7 / +2.9e-7 / +5.6e-8 | +6.6e-5 … +9.3e-5 |
| 30 GHz | 1 (LSM^u) | 547.47 / 529.95 / 473.49 / 360.22 | 7.763e-3 … 2.358e-2 | −3.4e-7 … −7.6e-8 | −7.3e-4 … +1.6e-5 |

(relative differences to the exact root). **The FEM equals the exact Leontovich roots of T1 to ≤ 6.7e-7 for all 12
modes.** Against the first-order `wrmodes.loss` the target of 1e-6 cannot be met by any exact solution: first
order differs from the exact Leontovich value by 7e-5–1.4e-4 for TE_m0 and up to 7.3e-4 for the split TE/TM 1,1
pair (T1 §5 states a 4e-3 bound for this O(ka Z/η₀) term); the FEM's difference to first order equals the exact
one's to within 7e-7. I read the card's "T1/`wrmodes.loss`" as T1's exact B6 for the 1e-6 target.

**Butterweck's lossy-wall filter** (75 GHz, 250 Ω on both narrow walls, 15 cm; `impedance_walls.py` §2), dB:

| mode | FEM 6/λ | FEM 10/λ | WP2 filter M = 80 | M = 160 | FEM vs M = 160 | Butterweck | FEM vs Butterweck |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TE1,0 | 0.5685 | 0.5685 | 0.5691 | 0.5688 | −0.05 % | 0.6 | −5.2 % |
| TE2,0 | 2.3134 | 2.3134 | 2.3157 | 2.3147 | −0.06 % | 2.5 | −7.5 % |
| TE3,0 | 5.3338 | 5.3338 | 5.3386 | 5.3366 | −0.05 % | 5.4 | −1.2 % |
| TE0,1 | 2.8357 | 2.8357 | 2.8355 | 2.8356 | +0.00 % | – | – |
| TE1,1 | 12.2424 | 12.2422 | 12.2400 | 12.2414 | +0.01 % | – | – |
| TM1,1 | 2.6625 | 2.6625 | 2.6623 | 2.6624 | +0.00 % | – | – |

TE1,0 → TE3,0: −33.573 dB (filter: −33.552 dB; Butterweck: −36 … −24.5 dB). Within 5 % of WP2: pass (0.06 %), and
the filter's own convergence (M = 80 → 160) moves towards the FEM. Within 5 % of Butterweck's 0.6 / 2.5 / 5.4 dB:
**not met for TE1,0 (−5.2 %) and TE2,0 (−7.5 %)**, for the same reason as in WP2 (−5.2 / −7.4 / −1.1 %, against a
10 % target there): 0.6 dB is a rounded 4 dB/m (first order gives 3.87 dB/m, 0.580 dB) and 2.5 dB is a measurement.
Every exact method gives these numbers; the 5 % target for Butterweck's figures is not achievable (decision below).

**Lossy straights against `LossyWallFilter`** (5 cm, every propagating family; §3 and the tests): max |ΔS| over the
complex GSM 1.2e-5 (18 GHz, aluminium), 1.2e-4 (54 GHz, aluminium, 45 modes), 5.7e-5 / 8.1e-5 (2.5 Ω); 25 Ω at
30 GHz: 2.2e-3, all in the reflection of TE0,2 1.7 % above cutoff, where both methods converge towards each other
(FEM 0.0167 / 0.0159 / 0.0155 with mesh refinement, filter 0.0146 / 0.0149 with 80 / 160 modes); 250 Ω: 9e-3–1.1e-2,
in the junction reflections of n ≥ 1 modes, again converging slowly in both methods (TE1,1 S11: FEM −0.0248 →
−0.0262, filter −0.0339 → −0.0295 with refinement). Transmissions agree to 1e-4.

**Properties**: reciprocity ≤ 3e-13 relative everywhere (complex-symmetric system), S11 = S22 to 4e-14 for lossy
miters (mirrored meshes), passivity (largest singular value 0.70–0.99999 < 1), lossy miters lose more power than
perfect ones in every column; Z_s → 0 is linear down to 1e-9 Ω (no penalty ill-conditioning); a lossy `Line`
(WP2 straights + `Miter(wall_impedance='guide')`) is reciprocal and passive.

## Chained miters

`chained_miter_outline` builds one outline from n miters (signed angles as `Miter`, chamfers as `Miter`) and n − 1
spacers (centre-line lengths between consecutive reference planes, i.e. the `Straight`s of the equivalent `Line`).
Overlapping mirrors (e.g. two optical 45° miters closer than s = 2a, whose mirrors reach a into the spacer) are
joined at their intersection; geometries where a mirror reaches the neighbour's inner corner (opposite turns), or
where the outline with arms of length a touches itself (a U-turn with s = 0, a closed loop), raise ValueError.
A palindromic chain gets its symmetry line (S11 = S22 to rounding). **COMSOL campaign 2, case E5**: with
`chained_miter_outline([-45, -45], [k a], a=W, port_extension=15e-3)` the vertices equal the two vertex lists of
`CAMPAIGN_2.md` (k = 2, 3) to the printed four decimals (shifted by a/2: here the first apex is the origin), and the
port planes are 37.86 mm from the reference planes (test).

Single domain vs cascade Miter–Straight(s)–Miter (`chained_miters.py`, WR-90, 30 GHz, family 0, max |ΔS| over the
propagating block; cf = the cascade's cutoff factor):

| miters | mesh/λ | s/a | cf 1 | cf 1.5 | cf 2 | cf 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 × 90° | 6 | 0.25 / 0.5 / 1 / 2 / 3 / 4 | 4.8e-2 / 6.5e-3 / 1.9e-4 / 5.9e-5 / 1.3e-4 / 3.1e-4 | 3.4e-4 / 3.4e-5 / 4.8e-5 / 6.6e-5 / 1.4e-4 / 3.3e-4 | 1.3e-4 … 4.4e-4 | 1.6e-4 … 4.6e-4 |
| 2 × 90° | 16 | 0.25 / 0.5 / 1 / 2 / 3 / 4 | 4.8e-2 / 6.5e-3 / 1.8e-4 / 4.5e-6 / 3.3e-6 / 7.5e-6 | 3.3e-4 / 4.1e-6 / 2.6e-6 / 4.6e-6 / 3.5e-6 / 7.7e-6 | 1.2e-5 … 2.2e-5 | 1.5e-5 … 2.9e-5 |
| 2 × 90° | 32 | 0.25 / 0.5 / 1 / 2 / 3 / 4 | 4.8e-2 / 6.5e-3 / 1.8e-4 / 1.2e-6 / 9.1e-7 / 1.4e-6 | 3.3e-4 / 3.0e-6 / 1.1e-6 / **8.5e-7** / **8.2e-7** / 1.3e-6 | 4.4e-6 … 9.0e-6 | 5.2e-6 … 9.3e-6 |
| 2 × 45° optical | 16 | 1 / 2 / 2.5 / 3 / 4 | 5.7e-2 / 1.1e-2 / 1.4e-3 / 3.3e-6 / 2.7e-7 | 3.0e-2 / 1.0e-4 / **2.6e-7** / **2.5e-7** / **3.1e-7** | singular | singular |

* The cascade converges to the single domain as the spacer grows and with evanescent modes; the s-dependent part
  (cf 1: 4.8e-2 at a/4, 6.5e-3 at a/2, 1.8e-4 at a, independent of the mesh) is the truncation the card asks to
  measure: one evanescent mode per family (cf 1.5) suffices from s = a/2 for 90° miters.
* The floor is the discretization of the two FEM models; it falls with the mesh (6 → 16 → 32 per λ: 6e-5 → 4.6e-6 →
  8.5e-7 at 2a) and grows with the domain length. **Target < 1e-6 at s ≥ 2a: met at 2a and 3a** (8.5e-7, 8.2e-7,
  32/λ, cf 1.5); at 4a the floor is 1.3e-6 (discretization, not truncation: 7.7e-6 at 16/λ).
* 45° optical mirrors: the uniform guide between the mirrors is s − 2a. At s = 2a (mirrors touching) the cascade
  cannot converge (1e-4 at cf 1.5); with more evanescent modes the analytically continued entries (factor
  e^{|β|a} per port, CONVENTIONS §6) make the star product singular ("singular": `LinAlgError`). From s = 2.5a it
  meets 1e-6 (2.6e-7). So for these miters the card's "s ≥ 2a" reads "s ≥ 2a + a/2".

## Literature

FEM with default settings (P3, 6 elements per λ) unless stated; z = (ours − digitized)/u, markers with the x
reading error through our slope. "Pass" = rms z ≲ 1 with outliers explained.

| Source / figure | curves | result | status |
| --- | --- | --- | --- |
| Kotik & Sivov 1965 Fig. 3 (H, optical mirror, kd = 10–31.4) | rigorous, 416 points | rms z 1.30, 96 % within 2u; outliers at the TE40 cutoff dip (kd 12.5–12.8, our dip deeper: 0.666 vs 0.758) and kd > 30.4 (−0.011) — as WP4b | **pass** (outliers explained) |
| Kotik asymptotic 1 − 7.4/(kd)^{3/2}, kd > 6π | | vs FEM within 3.5 % away from the cutoff dips (authors: 3 %); 6.1 % in the dip at 6π | reported |
| Alessandri 1994 Figs. 4–5 (E plane), **b/a = 0.5** | d/b = 1.41, 0.707, 0.87, 0.9 | rms z 0.30, 0.23, 0.93 (Fig. 4), 0.46, 0.91 (Fig. 5) | **pass** |
| | d/b = 0.85, 0.83 | rms z 1.63, 2.26; outliers only in the deep zeros (−57.8 vs −49.1 dB at f/fc 1.65; the flat bottom of 0.83 near −43 dB, ≤ 2e-3 in \|S11\|) | pass (outliers explained) |
| | ref. [7] crosses | 0.36, 0.67, 8.97 (d/b = 0.87 crosses lie below both lines) | reported |
| Alessandri, b/a = 0.444 (WR-90) | all | rms z 5.4–25 | rejected: b/a = 0.5 confirmed (as WP4b) |
| Alimenti 1994 Fig. 2 (unmitered) | MMFD line, MM dots | rms z 0.76, 1.12 | **pass** |
| Alimenti 1994 Fig. 3 (optical mirror) | MM line, MMFD diamonds | rms z 0.71, 0.95 | **pass** |
| Alimenti Figs. 4, 6 (chamfered) | t/a = 0.30, 0.34, 0.38; Fig. 6 MMFD, BCMM | rms z 5.1, 7.6, 12.4; 34, 13.5 | qualitative |
| Alimenti Fig. 5 (design chart) | zeros and 30 dB band vs t/a | zeros 1.302 … 1.757 (Alimenti 1.299 … 1.821); band 0.207 → 0.265 (t/a 0.30 → 0.386), **jump between t/a = 0.386 and 0.390** to 0.388, then falling (Alimenti: jump at 0.3867, 0.274 → 0.364) | qualitative |
| Ma 1997 Fig. 5 (sharp corner) | 12-term line, BEM | rms z 0.24, 0.62 | **pass** |
| Ma Fig. 8 (20-step staircases, division methods a, b, c) | (a), (b), (c) | rms z 0.64, 0.49, 0.62 | **pass** |
| Ma Fig. 6 (c/a = 1, \|S21\|) | 10 steps; BEM (exact mirror) | rms z 0.20; 0.57 | **pass** |
| | 20 steps; 40/60 steps | rms z 1.69; 2.37 (mean −0.33 dB) | not met (see below) |
| Ma Fig. 9 (60 steps) | | rms z 1.28 (91 % within 2u, mean +0.36 dB) | not met (see below) |
| Ma Fig. 11 | c/a = 0 | rms z 0.19 | **pass** |
| | c/a = 1.0 (60 steps (a)) | rms z 2.05 (mean +0.50 dB); exact mirror 1.66 | not met (see below) |
| | c/a = 0.60, 0.631, 0.70, 1.30 | rms z 31.5, 19.1, 16.8, 1.29 | qualitative |
| Ma Table 1 (16 GHz) | | exact mirror 0.2778; Ma's own staircase (a) 50 / 60 steps: 0.2817 / 0.2811 (Ma: 0.2641 / 0.2622); (b) 0.2885, (c) 0.2741 | qualitative |
| Reiter 1995 Fig. 3(a), sharp H bends | γ = 90, 105, 120, 135 | rms z 0.88, 1.07, 0.49, 0.35 | **pass** |
| | γ = 150 | rms z 7.3, we are 2.0 dB lower | not met (see below) |
| Reiter Fig. 3(b), mitered H | γ = 105, 120, 135 | rms z 1.36, 30, 5.4 | not met |
| Reiter Fig. 3(c), sharp E bends | γ = 90 … 150 | rms z 4.9 … 2.1, we are 0.6–1.4 dB higher | not met (see below) |
| Reiter Fig. 3(d), mitered E | γ = 105, 120, 135 | rms z 4.0, 7.9, 9.8 | not met |
| Reiter Fig. 4 (H miter + l + E miter, `Line`, 12/λ, cf 3) | l = 0; l = b | rms z 2.43 (mean +1.18 dB); 1.69 (+0.3 … +1.4 dB outside the zero) | not met (see below) |

Notes on the curves that are not met:

* **Ma, c/a = 1 with 40–60 steps.** Modelling Ma's staircases (`_lit.staircase_miter_outline`, Fig. 3/7 of the
  paper: method (a) steps through the slanted plane, (b) below, on the guide side, (c) above) reproduces Ma's
  20-step curves of Fig. 8 for all three methods and the 10-step curve of Fig. 6, and gives the published order of
  the methods ((b) above (a) above (c) in \|S11\|). With 50–60 steps it does not: our staircases converge to the
  exact mirror from above (\|S11\| 0.2876 / 0.2811 / 0.2778 for 20 / 60 steps / exact at 16 GHz; mesh check of the
  60-step outline 0.28108 / 0.28101 / 0.28096 at 6 / 10 / 14 per λ), Ma's values fall below it (0.2641 / 0.2622). The
  geometry interpretation is therefore right, and the Table 1 discrepancy (three methods: 0.2778) is not the
  staircase: it grows with Ma's number of steps, i.e. it is in Ma's computation at large N. Fig. 6 (40/60), Fig. 9
  and Fig. 11 c/a = 1.0 carry the same 0.3–0.5 dB. **Proposal**: treat them as qualitative, like Table 1.
* **Reiter, sharp corners.** For γ = 150 (30°, H plane) our \|S11\| agrees with Marcuvitz's measured
  arbitrary-angle H-plane corners (Sec. 5.29b, WR-90, λ = 3.0–3.4 cm, `marcuvitz_corners.py`): complex S11 at the
  inner-corner planes within 0.002 at 30° and 45°, 0.004–0.007 at 50°–75°, 0.008–0.015 at 90°, i.e. within 0.6 dB
  everywhere (30° at 3.4 cm: −27.83 dB FEM vs −27.58 dB measured), while Reiter's γ = 150 curve is 3 dB higher
  than the FEM. For the E plane, Reiter's 90° sharp corner lies 1.1–1.5 dB below Alessandri's d/b = 1.41 curve for
  the same guide (WR62 has b/a = 0.49997), which the FEM and MiterMM reproduce (rms z 0.30); Marcuvitz's static
  E-plane corner (Sec. 5.28b, a few % for 2b/λ_g < 0.51) gives −9.76 / −7.66 dB at 12.05 / 13.05 GHz, the FEM
  −9.85 / −7.52, Reiter −10.99 / −8.90. At 30°–75° the static formula lies 0.4–0.8 dB above the FEM and Reiter
  0.4–1.0 dB below it, so there Marcuvitz does not separate the two. Reiter's E-plane curves are matched by a
  narrower guide (in-plane side about 7.0–7.1 mm instead of 7.899 mm, b/a ≈ 0.44); I did not pursue this.
  The mitered curves (b), (d) disagree like Alimenti's chamfered ones (zeros shifted).
* **Reiter Fig. 4.** `MiterMM` in the same `Line` (l = b) gives the FEM's values to 0.01 dB (8.45–11.35 GHz). Our
  cascade is 0.3–1.4 dB more reflective than Reiter's theory and measurement band (and the deep 11.7 GHz zero is
  −42 dB, Reiter −54 to −60 dB), consistent with the ~1 dB E-plane offset of Fig. 3(c). l = 0 is a zero-length
  cascade: it converges in the cutoff factor only with 12 elements per λ (at the default 6, |S11| at 9 GHz jumps to
  −23.3 and −11.8 dB at cf 4 and 5; at 12: −18.072 / −18.071 / −18.084 / −18.088 for cf 2–5), and even then cf 5
  jumps at 10.5 and 11.7 GHz; cf 3 is used. Dead families matter at l = 0 (global TE2,0 reflected by the E-plane
  miter): `MiterMM`, which treats them as uncoupled straights (CONVENTIONS §3), gives −17.756 dB at every cf
  against the FEM's −18.07, so the two miters are **not** interchangeable in zero-length cascades.

## Acceptance table

| Check | Target | Result |
| --- | --- | --- |
| Ma (unmitered corner, figures other than the chamfered Fig. 11 curves) | within the digitizing uncertainty, outliers listed | Fig. 5, Fig. 8 (a, b, c), Fig. 6 10 steps and BEM, Fig. 11 c/a = 0: **pass** (rms z 0.19–0.64). c/a = 1 with 20–60 steps (Fig. 6, Fig. 9, Fig. 11 c/a = 1.0): **not met** (rms z 1.3–2.4, 0.3–0.5 dB; the Table 1 discrepancy, located in Ma's large-N computation) |
| Alessandri (b/a = 0.5) | uncertainty | **pass**: rms z 0.23–0.93; d/b = 0.85, 0.83 outliers in their deep zeros only; b/a = 0.444 rejected (rms z 5–25) |
| Alimenti Figs. 2–3 | uncertainty | **pass**: rms z 0.71–1.12 |
| Kotik Fig. 3 | uncertainty | **pass**: rms z 1.30, 96 % within 2u; outliers at the TE40 cutoff dip and kd > 30.4 (as WP4b) |
| Reiter Figs. 3–4 | uncertainty | Fig. 3(a) γ = 90–135 **pass**; γ = 150, Fig. 3(b)–(d) and Fig. 4 **not met**: the FEM agrees with Marcuvitz's measured H corners (γ = 150), with Alessandri and Marcuvitz's static formula at 90° (E plane) and with MiterMM (Fig. 4) where Reiter does not |
| Alimenti Figs. 4–6, Ma Fig. 11 chamfered, Ma Table 1 | qualitative | reported above (same as WP4b; 3 methods agree) |
| Chained miters | → cascade; < 1e-6 at s ≥ 2a | 90°: 8.5e-7 / 8.2e-7 at 2a / 3a (32/λ, cf 1.5), floor 1.3e-6 at 4a; 45° optical: 2.6e-7 from 2.5a (the mirrors reach a into the spacer). Truncation effect at short spacers reported |
| Impedance walls: straight channel, per-mode α | 1e-6 relative to T1/`wrmodes.loss` | **pass against T1's exact roots** (≤ 6.7e-7, 12 modes, families 0 and 1). First order differs from exact by 7e-5–7e-4 (physics, T1 §5) |
| Butterweck filter | 5 % of WP2 and of 0.6 / 2.5 / 5.4 dB | WP2: **pass** (0.06 %). Butterweck: −5.2 / −7.5 / −1.2 %, **not met** for TE1,0, TE2,0 (identical to WP2's exact result) |
| Reciprocity (lossless and lossy), unitarity (lossless), passivity (lossy) | CONVENTIONS §6 | **pass**: ≤ 3e-13; lossless paths unchanged (bitwise); largest singular value < 1 |
| E5 vertices | exact | **pass** (four decimals) |
| Fast suite / slow FEM tests | all pass | see "Tests" |

## Tests

New in `tests/test_fem2d.py`: wall-impedance arguments and lossless limit (bitwise); antisymmetry of the coupling
matrix; lossy straight vs `LossyWallFilter` (metal-like and 25 Ω, all families); α vs the exact B6 root (10 GHz,
1e-6); Butterweck vs WP2; lossy miters (H, E; plain, chamfered: reciprocity, S11 = S22, passivity); lossy line;
E5 vertices (2a, 3a); single-miter chains = `miter_outline`, overlap, refusals; single domain vs cascade (default
mesh); FEM `Miter` and `MiterMM` interchangeable in a `Line` (H, E, plain, chamfered: 2e-3). Slow: α of all 12 modes
at 30 GHz to 1e-6; chained miters to 1e-6 (four cases). No timing assertions.

## Open issues

1. **Reiter & Arndt 1995 Fig. 3(c)/(d) and Fig. 4** disagree with Alessandri, Marcuvitz, the FEM and MiterMM by
   about 1 dB (E plane); Fig. 3(a) γ = 150 by 3 dB (H plane). A discrepancy report is warranted (not written: the
   card asks for a report of outliers; I propose `docs/discrepancies/<date>-reiter-eplane.md` for the coordinator).
2. **Ma's c/a = 1 curves with 40–60 steps** (and Table 1): our staircase model explains Ma's 10- and 20-step curves
   but not the 50–60-step ones; the remaining difference grows with N. Propose qualitative status for Fig. 6
   (40/60), Fig. 9, Fig. 11 c/a = 1.0.
3. **Zero-length cascades** (Reiter Fig. 4, l = 0) need 12 elements per λ for stable evanescent entries and still
   show jumps at cutoff factor 5 (10.5, 11.7 GHz). The FEM's evanescent entries are accurate to ~1e-3 relative at
   the default mesh (WP3 report), which a zero-length cascade amplifies. A default that refines the mesh with the
   number of evanescent basis modes, or a warning, may be worth it (wave 3).
4. **Large wall impedance** (|Z_s| ~ η₀, 250 Ω): the junction reflections of n ≥ 1 modes converge slowly in both
   the FEM and `LossyWallFilter` (singular field at the junction between a perfect and an absorbing wall); they
   differ by up to 1e-2 and approach each other with refinement. Transmissions agree to 1e-4.
5. The walls at v = ±b/2 stay perfect in the FEM; in a lossy `Line`, `Miter(wall_impedance='guide')` leaves out
   the broad-wall loss of the miter itself (documented; it is a few 1e-4 dB for a WR-90 miter).
6. `MiterMM` treats dead families as uncoupled straights; in zero-length cascades this changes |S11| by 0.3 dB
   (Reiter Fig. 4 at 9 GHz). For WP4b/WP7b.

## Decisions for the coordinator / Daniel

1. **Card error, n ≥ 1 impedance walls**: the Leontovich condition couples LSM and LSE for n ≥ 1 (derivation §6.3);
   implemented as a coupled condition. Please confirm (a theory agent may want to re-derive §6 independently).
2. **α target**: accept "1e-6 relative to T1's exact B6 roots" (met) in place of "to T1/`wrmodes.loss`" (first order
   differs from exact by up to 7e-4).
3. **Butterweck 5 %**: the exact value is 5.2 % / 7.5 % below Butterweck's 0.6 / 2.5 dB in every method (WP2
   accepted 10 %); relax to 10 % or compare with WP2 only.
4. **Ma c/a = 1, 40–60 steps** (Fig. 6, Fig. 9, Fig. 11 c/a = 1.0) → qualitative, like Table 1.
5. **Reiter Fig. 3(b)–(d), Fig. 3(a) γ = 150, Fig. 4** → qualitative, with a discrepancy report (open issue 1).
6. **Chained-miter target for 45° optical mirrors**: "s ≥ 2a" should read "uniform guide between the mirrors ≥ a/2"
   (s ≥ 2.5a); at s = 2a no cascade converges.
7. CONVENTIONS proposals (not changed): §3 could say that impedance walls parallel to v keep n but couple the LS
   polarizations for n ≥ 1; §8 could note that zero-length cascades need the evanescent entries to the same
   accuracy as the propagating ones.

## Reproduction

```
PYTHONPATH=src python benchmarks/fem2d/impedance_walls.py --processes 1   # ~6 min
PYTHONPATH=src python benchmarks/fem2d/chained_miters.py --processes 2    # ~15 min
PYTHONPATH=src python benchmarks/fem2d/{kotik1965,alimenti1994,ma1997,reiter1995}.py --processes 2
PYTHONPATH=src python benchmarks/fem2d/alessandri1994.py --b-over-a 0.5 (and 0.4444444444444444)
```

The Marcuvitz cross-checks quoted above (Secs. 5.28b, 5.29b of `docs/literature/marcuvitz-steps-and-bends.md`)
are in `benchmarks/fem2d/marcuvitz_corners.py`.
