# Discrepancy: Reiter & Arndt 1995 Figs. 3–4 and Ma et al. 1997 c/a = 1 with 40–60 steps vs the FEM (WP3b)

*Opened by WP3b on 2026-11-12. To be settled by a theory agent and Daniel, not by WP3b. The registry statuses and
the cards are unchanged until Daniel decides (`wp3b-reiter1995-fig3`, `wp3b-reiter1995-fig4`,
`wp3-ma1997-figures`).*

## Observation

FEM = `wrmodes.fem2d.Miter` at default settings (P3, 6 elements per λ), except Reiter Fig. 4, where it is a
`wrmodes.Line` of two FEM miters, 12 elements per λ and cutoff factor 3. Scripts: `benchmarks/fem2d/reiter1995.py`
and `ma1997.py`; data: `benchmarks/fem2d/data/reiter1995.npz` and `ma1997.npz`. z = (FEM − digitized) / uncertainty;
the offset is the mean of FEM − digitized in dB (positive: the FEM reflects more, or for Ma Fig. 6, transmits more).

| Curve (geometry, CONVENTIONS.md §6) | rms z | within 2u | mean offset | remark |
| --- | --- | --- | --- | --- |
| Reiter Fig. 3(a), sharp H bend, γ = 150 (deflection 30°), WR62 | 7.27 | 14 % | −2.0 dB | the FEM reflects 2–3 dB less; γ = 90–135 agree (rms z 0.35–1.07) |
| Reiter Fig. 3(b), mitered H bend, γ = 105, a_opt = 0.949 a | 1.36 | 84 % | +0.35 dB | nearly passes; zero at 14.7 GHz in both |
| Reiter Fig. 3(b), γ = 120, a_opt = 0.955 a | 29.9 | 2 % | −1.55 dB | reflection zero: FEM 16.2 GHz, Reiter about 14.9 GHz |
| Reiter Fig. 3(b), γ = 135, a_opt = 0.956 a | 5.44 | 0 % | +0.29 dB | FEM zero at 14.9 GHz; Reiter's curve stays near −51 to −55 dB above 16 GHz |
| Reiter Fig. 3(c), sharp E bend, γ = 90, 105, 120, 135, 150 | 4.89, 4.17, 3.54, 2.77, 2.07 | 0–47 % | +1.41, +1.18, +0.99, +0.78, +0.58 dB | systematic: the FEM reflects more, by more at larger deflection |
| Reiter Fig. 3(d), mitered E bend, γ = 105, 120, 135 (b_2opt = 0.873, 0.898, 0.918 b) | 4.00, 7.88, 9.79 | 0–17 % | +0.88, +1.97, +2.80 dB | same sign as 3(c) |
| Reiter Fig. 4, WR90: H miter (a_opt = 0.976 a), straight l, E miter (b_opt = 0.874 b), l = 0 | 2.43 | 53 % | +1.18 dB | +0.5 to +1.9 dB over the band |
| Reiter Fig. 4, l = b | 1.69 | 82 % | +1.50 dB | +0.3 to +1.4 dB outside the deep zero. Zero: FEM −42 dB at 11.735 GHz; Reiter theory −54.3 dB at 11.705 GHz, measurement −60.2 dB at 11.726 GHz. The 9.4 GHz minimum is at the same frequency (9.398 vs 9.41 GHz). |
| Ma Fig. 6, c/a = 1, "40/60 steps" (\|S21\| dB; FEM with Ma's 60-step staircase, method (a)) | 2.37 | 61 % | −0.33 dB | |
| Ma Fig. 6, 20 steps (staircase (a)) | 1.69 | 68 % | −0.39 dB | |
| Ma Fig. 9, 60 steps (\|S11\| dB) | 1.28 | 91 % | +0.36 dB | exact mirror: rms z 1.03, +0.28 dB |
| Ma Fig. 11, c/a = 1.0 (\|S11\| dB vs f/fc; 60-step staircase) | 2.05 | 50 % | +0.50 dB | exact mirror: rms z 1.66, +0.41 dB |
| Ma Table 1, 16 GHz, \|S11\|, 50 / 60 steps | – | – | – | FEM staircase 0.2817 / 0.2811, exact mirror 0.2778; Ma 0.2641 / 0.2622 (already qualitative) |

For comparison, the same scripts pass the following:

* Reiter Fig. 3(a), γ = 90–135.
* Ma Fig. 5, the sharp corner (rms z 0.24; BEM circles 0.62).
* Ma Fig. 8, 20-step staircases with division methods (a), (b), (c): rms z 0.64, 0.49, 0.62.
* Ma Fig. 6, the 10-step staircase (0.20) and the BEM circles against the exact mirror (0.57).
* Ma Fig. 11, c/a = 0 (0.19).
* Alessandri 1994 at b/a = 0.5, Alimenti Figs. 2–3 and Kotik Fig. 3 (docs/reports/WP3b.md).

## Evidence

### 1. A second method: MiterMM (WP4b)

* For 90° miters `wrmodes.modematch.MiterMM` agrees with the FEM to 3e-4 dB in TE10 transmission and 7e-5 in
  \|S11\| (coordinator's comparison, 2026-11-11 discrepancy file). It also gives the same Alessandri and Ma Fig. 11
  results (docs/reports/WP4b.md).
* **Reiter Fig. 4, l = b:** the same `Line` built with `MiterMM` gives the FEM's \|S11\| to 0.01 dB at 8.45, 9.05,
  10.025, 10.95 and 11.35 GHz (−19.37, −29.02, −27.16, −26.14, −30.32 dB).
* **Reiter Fig. 4, l = 0:** this is a zero-length cascade, and there the two differ by 0.3 dB through the dead
  families (the reflection of global TE2,0 at the E-plane miter, which MiterMM treats as an uncoupled straight).
  Both are 0.5–1.9 dB above Reiter.
* MiterMM covers only 90°. The other angles of Fig. 3 rest on the FEM and on Marcuvitz (item 2).

### 2. Marcuvitz (1951), Secs. 5.29b and 5.28b (`benchmarks/fem2d/marcuvitz_corners.py`)

**H plane, measured.** Sec. 5.29b gives the measured T networks of sharp arbitrary-angle corners in WR-90 at
λ = 3.0, 3.2 and 3.4 cm, at reference planes through the inner corner (= ours). The FEM's complex S11 agrees with
them, so the 30° FEM value is supported by measurement and Reiter's γ = 150 curve is not:

| Angle | λ (cm) | Marcuvitz (measured) | FEM | \|ΔS11\| |
| --- | --- | --- | --- | --- |
| 30° | 3.0 | −29.21 dB | −28.78 dB | 0.0018 |
| 30° | 3.2 | −27.97 dB | −28.44 dB | 0.0021 |
| 30° | 3.4 | −27.58 dB | −27.83 dB | 0.0012 |
| 45° | 3.4 | −20.91 dB | −20.80 dB | 0.0012 |
| 50°–75° | 3.0–3.4 | | | 0.0018–0.0068 |
| 90° | 3.0–3.4 | | | 0.008–0.015 |

At 30°, a/λ = 0.67–0.76 in WR-90. Reiter's γ = 150 lies at a/λ = 0.63–0.95 in WR62. In that range the FEM gives
−27 to −29 dB and Reiter −24 to −27 dB.

**E plane, static formula.** Sec. 5.28b gives a static π network for sharp corners: "rigorous only in the static
limit", with measured B_b within a few per cent for 2b/λ_g < 0.51. Results in WR62:

| Angle | f (GHz) | 2b/λ_g | Marcuvitz | FEM | Reiter |
| --- | --- | --- | --- | --- | --- |
| 90° | 12.05 | 0.39 | −9.76 dB | −9.85 dB | −10.99 dB |
| 90° | 13.05 | 0.47 | −7.66 dB | −7.52 dB | −8.90 dB |

At 30°–75° the static values lie 0.4–0.8 dB above the FEM and Reiter lies 0.4–1.0 dB below it. There the static
formula does not separate the two; at 90° it sides with the FEM.

### 3. Alessandri et al. (1994) at the same aspect ratio

WR62 has b/a = 7.899/15.799 = 0.49997. Alessandri's figures are reproduced at b/a = 0.5 (WP4b; WP3b rms z 0.23–0.93),
so Alessandri's d/b = 1.41 curve (Fig. 4) is Reiter's γ = 90 E-plane bend of Fig. 3(c) in the same normalized
geometry:

* Read at the same f/fc (fc = 9.488 GHz), Reiter's curve lies 1.1 to 1.5 dB below Alessandri's. For example, at
  12.05 GHz Reiter gives −10.99 dB and Alessandri −9.92 dB; at 15.05 GHz, −5.13 and −3.80 dB.
* The FEM agrees with Alessandri's curve (rms z 0.30).
* Alessandri also plots crosses from their ref. [7], Reiter & Arndt's own BCMM (as identified by L1 and WP4b), for d/b = 1.41 and 0.707. These
  agree with Alessandri's lines and with the FEM (rms z 0.36, 0.67), so Reiter & Arndt's method gave the
  Alessandri value elsewhere.
* Reiter's sharp E-plane curves are matched by the FEM in a guide with an in-plane side of about 7.0–7.1 mm
  instead of 7.899 mm (b/a ≈ 0.44). This is an observation, not an explanation.

### 4. Ma's staircases converge from above (`_lit.staircase_miter_outline`)

The FEM was run in Ma's own geometry, not only the exact mirror. Ma's staircase replaces the 45° mirror of the
c/a = 1 miter by N equal steps, with three division methods (paper, Fig. 7 and p. 4):

* (a) the steps cross the slanted plane;
* (b) the steps lie below it, on the guide side;
* (c) the steps lie above it.

At 16 GHz:

| N | (a) | (b) | (c) | Ma, Table 1 "Ref." |
| --- | --- | --- | --- | --- |
| 10 | 0.2981 | 0.3598 | 0.2575 | |
| 20 | 0.2876 | 0.3130 | 0.2670 | |
| 50 | 0.2817 | 0.2908 | 0.2734 | 0.2641 |
| 60 | 0.2811 | 0.2885 | 0.2741 | 0.2622 |
| exact mirror | 0.2778 | | | |

* Mesh check of the 60-step outline: 0.28108, 0.28101 and 0.28096 at 6, 10 and 14 elements per λ.
* The independent finite-difference and mode-matching values of WP4b for the exact mirror are 0.27779 and 0.27779.

**Conclusion.** The staircase model with Ma's division methods reproduces Ma's 20-step curves of Fig. 8 for all
three methods. It also reproduces Ma's ordering (Ma p. 5: "(b) and (c) are, respectively, above and below the curve
with the division method (a)") and his 10-step curve of Fig. 6. So the geometry and the staircase reading are
right. As N grows, the staircases converge to the exact mirror from above, while Ma's values move away from it,
below (0.2641 at 50, 0.2622 at 60 steps). The disagreement therefore grows with Ma's N: it is not the staircase,
it is in Ma's computation at large N. A possible cause is Ma's rule of choosing the number of modal terms in the
step-junction regions from the dimension ratios (p. 4), which gives very thin regions few terms; this is not checked.

### 5. L1's digitization of Reiter is not the cause

Checked against the paper with `benchmarks/fem2d/reiter_overlay_check.py`. It renders PDF p. 4 at 400 dpi,
calibrates each panel of Fig. 3 from its frame alone (independent of both L1 calibrations) and overlays L1's
merged points and the FEM values. Images go to a scratch folder, not the repository.

* **Axis calibration.** 99.8 %, 99.0 %, 100 % and 100 % of L1's points of panels (a)–(d) lie on ink within 3 px
  (about 0.2 dB, 0.03 GHz). The printed tick labels fall on the frame-calibrated grid lines. The offsets above
  (0.6–3 dB, or 1 GHz in the zero of 3(b) γ = 120) are 3–15 times this.
* **Curve assignment.**
  * In 3(a) and 3(c) each line carries its γ label, and the order is monotonic in γ.
  * In 3(b) the FEM's γ = 105 curve reproduces the branch that L1 assigned to γ = 105, left and right of its zero,
    which confirms L1's choice between the two right-hand branches.
  * The ambiguous branch of 3(d) ("right branch, γ = 105 or 135") is reported only and is not in the table.
* **Fig. 4.** Both L1 digitizers read the band with a 0.09–0.14 dB mean difference. The FEM's 9.4 GHz minimum falls
  at the printed frequency, so the frequency axis and l are not the cause of the level offsets.

### 6. The γ convention is Reiter's

Re-read on the paper:

* Fig. 1 (p. 1) is the generic building block of the method: a cavity with k ports and a cylindrical coordinate
  system. Fig. 2 is the return loss of a T-junction. **Neither defines γ.**
* γ is defined only by the insets of Fig. 3 (p. 4, re-read at 400 dpi). In each inset the arc γ is drawn at the
  inner corner, from the inner wall of one arm to the inner wall of the other, on the metal side.
  * In inset (a) the arms run horizontally and down-right. γ is the obtuse angle between the inner walls, so
    γ = 180° is a straight guide and γ = 90° a right-angle corner.
  * The deflection angle of `Miter` is therefore 180° − γ, as used (and as in L1's header).
* In insets (b) and (d) the dimension a_opt (b_2opt) is drawn from the inner corner perpendicular to the miter wall:
  Alessandri's d, the chamfer of `Miter`.
* The in-plane side is labelled a in the H-plane insets and b in the E-plane insets, i.e. 15.799 mm and 7.899 mm of
  the WR62 housing, as used.
* An inverted convention (γ as the deflection) would give γ = 90 the same geometry and swap 105↔75 etc. The
  γ = 90–135 agreement of 3(a) (rms z 0.35–1.07), the reflection rising with deflection in 3(a) and 3(c), and the
  coincidence of the γ = 105 zero in 3(b) rule it out.

## Questions for the theory agent / Daniel

1. Should Reiter & Arndt 1995 Fig. 3(a) γ = 150, Figs. 3(b)–(d) and Fig. 4 become qualitative references for
   WP3b, WP7b and later work (registry `wp3b-reiter1995-fig3`, `wp3b-reiter1995-fig4`), with Fig. 3(a)
   γ = 90–135 kept as pass/fail?
2. Should Ma's c/a = 1 curves with 20–60 steps (Fig. 6 "20 steps" and "40/60 steps", Fig. 9, Fig. 11 c/a = 1.0)
   become qualitative like Table 1 (registry `wp3-ma1997-figures`), keeping Fig. 5, Fig. 8, Fig. 6 10 steps and
   BEM, and Fig. 11 c/a = 0 as pass/fail?
3. Is a different housing for Reiter's E-plane bends plausible (the curves fit an in-plane side of about 7.0–7.1 mm,
   b/a ≈ 0.44)? Both papers state WR62, and Alessandri's ref. [7] crosses (Reiter & Arndt's BCMM) agree with
   b/a = 0.5.
4. Reiter Fig. 4: the inset dimensions l between the two bends but not to which planes. The 9.4 GHz minimum,
   which fixes the electrical spacing, agrees with reference planes through the inner corners; should this
   reading be recorded in the registry entry?

## Resolution

*(left for Daniel's decision)*
