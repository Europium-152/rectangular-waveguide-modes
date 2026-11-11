# Literature benchmark data (card L1)

Published results that wrmodes 1.0 is validated against, as machine-readable data with uncertainties. Each figure was digitized twice, independently, by agents L1a and L1b (`raw/a/`, `raw/b/`, with their scripts and calibrations); the files in this folder are the merged result (card L1, 2026-11-08, `scripts/merge_ab.py`). Transcribed formulas are in `docs/literature/`.

## Files

Every curve file has the columns `curve, x, y, uncertainty, x_uncertainty, source` and a `#` header with the source, figure, axes and units, geometry and parameter definitions, both digitizers' calibrations, the merge method, the a-b agreement per curve and every decision taken in the merge. `uncertainty` is the uncertainty of `y` in the units of `y`; `x_uncertainty` is 0 for line curves (sampled on a regular grid; x reading errors are already converted into the y uncertainty) and the x uncertainty for markers (dots, crosses, circles: one row per printed symbol). `source` is `ab` (both digitizations), `a` or `b` (one only; uncertainty doubled) or `L1` (read by the merger on the page, Reiter Fig. 4 minima only). Per CONVENTIONS.md section 8, digitized data never set a tolerance tighter than their uncertainty.

| Source | Figure | File | Curves | y uncertainty: min-max (median) | a-b agreement: rms / max of a-b over all curves |
| --- | --- | --- | --- | --- | --- |
| Alessandri, Mongiardo & Sorrentino 1994 | Fig. 4: \|S11\| (dB) vs f/fc, 90° mitered E-plane bend | `alessandri1994_fig4.csv` | d/b = 1.41, 0.707 (lines), 0.87 (dots), crosses of their ref. [7] | 0.19-0.86 (0.20) dB | 0.053 / 0.20 dB |
| Alessandri et al. 1994 | Fig. 5: \|S11\| (dB) vs f/fc, 90° mitered E-plane bend | `alessandri1994_fig5.csv` | d/b = 0.9, 0.87, 0.85, 0.83 | 0.27-3.68 (0.29) dB | 0.094 / 0.98 dB |
| Doane & Anderson 1989 | Fig. 9: spurious mode power (%) vs GHz, 20" 90° twists, WR187 tall guide | `doane1989_fig9.csv` | constant, cosine, triangular rate of twist | 0.01-0.046 (0.011) % | 0.0033 / 0.0076 % |
| Ma, Yamane & Yamashita 1997 | Fig. 5: \|S21\|² vs GHz, right-angle H-plane corner | `ma1997_fig5.csv` | 3/6/12 terms (one line), BEM circles | 0.0059-0.086 (0.013)  | 0.0039 / 0.014  |
| Ma et al. 1997 | Fig. 6: \|S21\| (dB) vs GHz, mitered H-plane bend c/a = 1.0 | `ma1997_fig6.csv` | 40/60, 20, 10 steps, BEM circles | 0.14-9.68 (0.19) dB | 0.065 / 0.37 dB |
| Ma et al. 1997 | Fig. 8: \|S11\| (dB) vs GHz, 20 steps, division methods | `ma1997_fig8.csv` | (a), (b), (c) | 0.43-3.62 (0.52) dB | 0.16 / 0.50 dB |
| Ma et al. 1997 | Fig. 9: \|S11\| (dB) vs GHz, 60 steps | `ma1997_fig9.csv` | (a)=(b)=(c) | 0.37-3.62 (0.40) dB | 0.16 / 0.99 dB |
| Ma et al. 1997 | Fig. 11: \|S11\| (dB) vs f/fc, miter size | `ma1997_fig11.csv` | c/a = 0, 0.60, 0.631, 0.70, 1.0, 1.30 | 0.23-8.69 (0.28) dB | 0.12 / 2.44 dB |
| Ma et al. 1997 | Table 1: \|S11\|, \|S21\| at 16 GHz, 50/60 steps, step displacements (transcribed) | `ma1997_table1.csv` | Ref., (a)-(d) | half a unit of the last printed digit | identical (36 values) |
| Reiter & Arndt 1995 | Fig. 3(a): \|S11\| (dB) vs GHz, H-plane bends, WR62 | `reiter1995_fig3a.csv` | γ = 90, 105, 120, 135, 150° | 0.20-2.35 (0.30) dB | 0.039 / 0.17 dB |
| Reiter & Arndt 1995 | Fig. 3(b): mitered H-plane bends, WR62 | `reiter1995_fig3b.csv` | γ = 105, 120, 135° | 0.26-3.79 (0.32) dB | 0.13 / 1.40 dB |
| Reiter & Arndt 1995 | Fig. 3(c): E-plane bends, WR62 | `reiter1995_fig3c.csv` | γ = 90, 105, 120, 135, 150° | 0.28-0.95 (0.28) dB | 0.024 / 0.081 dB |
| Reiter & Arndt 1995 | Fig. 3(d): mitered E-plane bends, WR62 | `reiter1995_fig3d.csv` | γ = 105, 120, 135°, "right branch (gamma=105 or 135)" | 0.24-3.17 (0.24) dB | 0.092 / 1.01 dB |
| Reiter & Arndt 1995 | Fig. 4: cascaded mitered H- and E-plane bends, WR90, theory and measurement | `reiter1995_fig4.csv` | l = 0, l = b, three minimum rows | 0.30-12.88 (0.50) dB | 0.15 / 0.52 dB |
| Butterweck 1968; Butterweck & de Ronde 1968 | quoted numbers (D5), sentence and page of each | `butterweck1968_numbers.csv` | 20 items | - | identical where both transcribed (see header) |

Card item D8 (Kotik & Sivov 1965) was done by card L2, with Alimenti et al. 1994: see the L2 section at the end.

## Agreement of the two digitizations, per curve

Over the grid points (lines) or paired symbols (markers) covered by both sets, after the exclusions listed in the file headers. "Flags" counts points where |a - b| exceeds the combined reading uncertainty sqrt(u_a² + u_b²); every flagged point was checked on the page image and is explained in the file header.

| File | Curve | kind | n (both) | a only | b only | rms(a-b) | max abs(a-b) | mean(a-b) | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `alessandri1994_fig4.csv` | d/b=1.41 | line | 190 | 0 | 1 | 0.0465 | 0.134 | -0.02 | 0 |
| `alessandri1994_fig4.csv` | d/b=0.707 | line | 106 | 20 | 13 | 0.0545 | 0.204 | -0.038 | 0 |
| `alessandri1994_fig4.csv` | d/b=0.87 | markers | 36 | 2 | 0 | 0.0386 | 0.106 | -0.012 | 0 |
| `alessandri1994_fig4.csv` | ref7 d/b=1.41 | markers | 13 | 0 | 0 | 0.0839 | 0.191 | -0.034 | 0 |
| `alessandri1994_fig4.csv` | ref7 d/b=0.707 | markers | 13 | 0 | 0 | 0.0946 | 0.165 | -0.076 | 0 |
| `alessandri1994_fig4.csv` | ref7 d/b=0.87 | markers | 12 | 0 | 0 | 0.0559 | 0.113 | -0.035 | 0 |
| `alessandri1994_fig5.csv` | d/b=0.9 | line | 190 | 1 | 1 | 0.0625 | 0.203 | +0.037 | 0 |
| `alessandri1994_fig5.csv` | d/b=0.87 | line | 186 | 0 | 3 | 0.115 | 0.983 | +0.047 | 0 |
| `alessandri1994_fig5.csv` | d/b=0.85 | line | 181 | 0 | 3 | 0.109 | 0.398 | +0.064 | 0 |
| `alessandri1994_fig5.csv` | d/b=0.83 | line | 156 | 14 | 1 | 0.0761 | 0.184 | +0.052 | 0 |
| `doane1989_fig9.csv` | constant | line | 119 | 0 | 0 | 0.00453 | 0.00759 | -0.004 | 0 |
| `doane1989_fig9.csv` | cosine | line | 119 | 0 | 0 | 0.00178 | 0.0043 | -0.0014 | 0 |
| `doane1989_fig9.csv` | triangular | line | 113 | 6 | 0 | 0.00314 | 0.00694 | -0.0027 | 0 |
| `ma1997_fig5.csv` | 3/6/12 terms | line | 140 | 0 | 40 | 0.00402 | 0.0144 | -0.00087 | 0 |
| `ma1997_fig5.csv` | BEM | markers | 13 | 0 | 0 | 0.00216 | 0.00384 | -0.0006 | 0 |
| `ma1997_fig6.csv` | 40/60 steps | line | 125 | 10 | 33 | 0.0701 | 0.371 | +0.023 | 0 |
| `ma1997_fig6.csv` | 20 steps | line | 25 | 13 | 34 | 0.0733 | 0.129 | +0.041 | 0 |
| `ma1997_fig6.csv` | 10 steps | line | 54 | 8 | 23 | 0.0461 | 0.106 | +0.035 | 0 |
| `ma1997_fig6.csv` | BEM | markers | 9 | 1 | 0 | 0.0639 | 0.142 | +0.054 | 0 |
| `ma1997_fig8.csv` | (a) | line | 155 | 18 | 9 | 0.171 | 0.501 | +0.0066 | 0 |
| `ma1997_fig8.csv` | (b) | line | 59 | 0 | 18 | 0.104 | 0.187 | +0.077 | 0 |
| `ma1997_fig8.csv` | (c) | line | 57 | 6 | 73 | 0.182 | 0.429 | -0.018 | 0 |
| `ma1997_fig9.csv` | (a)=(b)=(c) | line | 156 | 0 | 25 | 0.159 | 0.99 | -0.11 | 1 |
| `ma1997_fig11.csv` | c/a=0 | line | 136 | 17 | 2 | 0.0257 | 0.12 | +0.017 | 0 |
| `ma1997_fig11.csv` | c/a=0.60 | line | 176 | 7 | 6 | 0.0408 | 0.187 | +0.007 | 0 |
| `ma1997_fig11.csv` | c/a=0.631 | line | 159 | 16 | 6 | 0.243 | 2.44 | -0.011 | 1 |
| `ma1997_fig11.csv` | c/a=0.70 | line | 166 | 22 | 0 | 0.0836 | 0.511 | +0.022 | 0 |
| `ma1997_fig11.csv` | c/a=1.0 | line | 106 | 69 | 6 | 0.0362 | 0.105 | +0.024 | 0 |
| `ma1997_fig11.csv` | c/a=1.30 | line | 55 | 23 | 5 | 0.0419 | 0.114 | +0.033 | 0 |
| `reiter1995_fig3a.csv` | gamma=90 | line | 84 | 5 | 0 | 0.0419 | 0.0914 | -0.035 | 0 |
| `reiter1995_fig3a.csv` | gamma=105 | line | 90 | 1 | 1 | 0.0516 | 0.161 | -0.031 | 0 |
| `reiter1995_fig3a.csv` | gamma=120 | line | 89 | 0 | 4 | 0.0427 | 0.168 | -0.014 | 0 |
| `reiter1995_fig3a.csv` | gamma=135 | line | 89 | 1 | 4 | 0.0291 | 0.084 | -0.007 | 0 |
| `reiter1995_fig3a.csv` | gamma=150 | line | 89 | 2 | 3 | 0.0244 | 0.067 | +0.011 | 0 |
| `reiter1995_fig3b.csv` | gamma=105 | line | 104 | 9 | 1 | 0.0871 | 0.384 | -0.017 | 0 |
| `reiter1995_fig3b.csv` | gamma=120 | line | 104 | 1 | 1 | 0.185 | 1.4 | +0.021 | 2 |
| `reiter1995_fig3b.csv` | gamma=135 | line | 66 | 1 | 4 | 0.0453 | 0.13 | -0.00095 | 0 |
| `reiter1995_fig3c.csv` | gamma=90 | line | 95 | 0 | 5 | 0.0263 | 0.0811 | -0.014 | 0 |
| `reiter1995_fig3c.csv` | gamma=105 | line | 97 | 1 | 2 | 0.0243 | 0.0666 | -0.014 | 0 |
| `reiter1995_fig3c.csv` | gamma=120 | line | 96 | 1 | 2 | 0.023 | 0.07 | -0.012 | 0 |
| `reiter1995_fig3c.csv` | gamma=135 | line | 96 | 0 | 2 | 0.0246 | 0.0788 | -0.0072 | 0 |
| `reiter1995_fig3c.csv` | gamma=150 | line | 97 | 0 | 1 | 0.0193 | 0.0604 | -0.004 | 0 |
| `reiter1995_fig3d.csv` | gamma=105 | line | 85 | 6 | 1 | 0.114 | 1.01 | +0.006 | 0 |
| `reiter1995_fig3d.csv` | gamma=120 | line | 110 | 7 | 2 | 0.0254 | 0.0529 | +0.015 | 0 |
| `reiter1995_fig3d.csv` | gamma=135 | line | 87 | 4 | 1 | 0.0298 | 0.142 | +0.011 | 0 |
| `reiter1995_fig3d.csv` | right branch (gamma=105 or 135) | line | 14 | 0 | 1 | 0.299 | 0.527 | +0.21 | 0 |
| `reiter1995_fig4.csv` | l=0 | line | 128 | 4 | 16 | 0.0981 | 0.22 | +0.09 | 0 |
| `reiter1995_fig4.csv` | l=b | line | 90 | 11 | 22 | 0.199 | 0.52 | +0.14 | 0 |

## Parameter definitions (they differ between sources)

* **Alessandri et al. 1994, d** (E-plane miter, b in the bend plane): distance from the inner corner to the miter wall, measured along the bend diagonal, i.e. perpendicular to the miter wall. d/b = sqrt(2) = 1.41 is the unmitered sharp corner; d/b = 1/sqrt(2) = 0.707 joins the ends of the two outer walls. x = f/fc, fc of the fundamental mode; guide dimensions not stated.
* **Ma et al. 1997, c** (H-plane miter, a = 15.799 mm in the bend plane, b = 7.899 mm): length cut from each outer wall, measured from the removed outer corner; c/a = 0 is the unmitered right-angle corner. fc = c0/2a = 9.488 GHz. Staircase model of the miter with N steps; division methods (a) steps through, (b) below, (c) above the slanted plane (text p. 4).
* **Reiter & Arndt 1995, a_opt, b_opt (printed b_2opt in Fig. 3(d)) and γ**: a_opt, b_opt = distance from the inner corner to the miter wall, measured perpendicular to it (same definition as Alessandri's d); γ = interior angle between the inner walls of the two arms, so γ = 90° is a right-angle bend and the deflection angle is 180° - γ. Fig. 3: WR62 (15.799 x 7.899 mm); Fig. 4: WR90 (22.86 x 10.16 mm), l = straight section between the H- and E-plane miters.
* **Marcuvitz 1951, θ** (Secs. 5.28b, 5.29b): the deflection (bend) angle between the arm axes, θ = 90° for a right-angle corner; his corners are sharp (not mitered) and the arbitrary-angle reference planes pass through the inner corner (`docs/literature/marcuvitz-steps-and-bends.md`).
* **Doane & Anderson 1989**: a = side perpendicular to the TE10 electric field, b parallel to it; their WR187 "tall" guide has a = 0.872 in, b = 1.872 in.

Published |S11| and |S21| magnitudes do not depend on the reference planes; any phase comparison must map the paper's planes to CONVENTIONS.md section 6.

## Decisions taken in the merge (details in the file headers)

* Ma Fig. 11: line styles matched to c/a by the labels printed next to the curves; L1 checked c/a = 0 against Fig. 5, c/a = 1.0 against Fig. 9 and c/a = 0.631 against the "39 %" bandwidth of the text (below). Both digitizers agreed. Where c/a = 0.60 and 0.631 nearly touch (f/fc 1.80-1.83) L1a points on the wrong line were dropped, and the hidden local maximum of c/a = 0.631 (1.83-1.86) is left as a gap.
* Reiter Fig. 3(b): right-hand branches assigned by the width of the dips below the frame (the more reflecting γ = 105° must have the narrower dip): rising to -25.5 dB is γ = 105°, to -34.5 dB is γ = 120°. Both agreed.
* Reiter Fig. 3(d): the branch rising from -65 dB at 17.2 GHz to -50 dB at 18 GHz is not identifiable from the figure (γ = 105° or 135°; fits of a smooth zero favour neither consistently). Kept as a separate curve "right branch (gamma=105 or 135)" (L1a had it under γ = 105°).
* Reiter Fig. 4: the 11.7 GHz minimum belongs to l = b (the measured l = 0 line keeps its ripple through both crossings, the l = b line is smooth from the 9.4 GHz minimum to the 11.7 GHz minimum). The printed minima are added as one-row curves read by L1. The figure has tick marks (L1a used them; L1b used the frame).
* Doane Fig. 9: the top frame, labelled 2 %, lies 9 % closer to the 1.5 % ticks than the tick spacing, on both sides of the figure and with the label itself; this is in the print. The frame is kept as 2 % and the difference to a tick-only scale (up to 0.02 %, constant curve above 17.87 GHz) is added to the uncertainty. The rise of the triangular curve from 0.05 % to 0.15 % across 15-18 GHz is real (L1 measured it against the bottom frame in the photograph).
* Single points dropped where the page shows the reading to be off the curve: Alessandri Fig. 5 (d/b = 0.9 at f/fc 1.024-1.025 on the tick mark, d/b = 0.83 at 1.25), Ma Fig. 6 (20 steps, 10.08-10.13 GHz), Ma Fig. 11 (c/a = 1.30 at 1.805-1.81 on a tick mark), Reiter Fig. 3(b) (γ = 105° at 13.85-13.90 GHz on a leader line).
* No interpolation across a gap whose both ends lie within 4 dB of the bottom frame (reflection zeros below the frame); the zeros themselves are not in the data.

## Cross-checks between files (computed from the merged files)

* Ma Fig. 11 c/a=0 against Fig. 5 (right-angle corner, |S11|^2 = 1 - |S21|^2, f/fc = f/9.488 GHz), 139 points with f/fc 1.05-1.80: rms difference 0.049 dB, max 0.139 dB; 0 points outside the combined uncertainty.
* Ma Fig. 11 c/a=1.0 against Fig. 9 (60 steps, c/a = 1.0), 181 points: rms difference 0.139 dB, max 0.325 dB; 0 points outside the combined uncertainty.
* Ma Fig. 11 c/a=0.631: |S11| < -30 dB from f/fc = 1.555 to the last point below -30 dB at 1.950 (the curve leaves the -30 dB line at about 1.96 just after its second zero), i.e. 39.5 % of the single-mode band f/fc = 1-2; text: "about 39%".
* Ma Fig. 9 at 16 GHz: -11.39 +- 0.38 dB; Table 1 (60 steps, Ref.) |S11| = 0.2622 = -11.63 dB.
* Ma Fig. 6 (40/60 steps) at 16 GHz: -0.37 +- 0.15 dB; Table 1 (60 steps, Ref.) |S21| = 0.9650 = -0.31 dB.
* Alessandri d/b=0.87, dots of Fig. 4 against the line of Fig. 5 (same curve printed twice), 38 dots: rms difference 0.132 dB, max 0.421 dB; 0 outside the combined uncertainty. Highest dot: -26.09 dB (text: "better than -26 dB in the whole band").
* Ma BEM circles: Fig. 5 at 10.42, 11.40, 12.33, 13.27, 14.24, 15.16, 15.67, 16.13, 16.60, 17.06, 17.57, 18.06, 18.50 GHz; Fig. 6 at 10.45, 11.43, 12.36, 13.29, 14.26, 15.19, 16.16, 17.11, 18.09, 19.01 GHz: every Fig. 6 circle up to 18.1 GHz lies within 0.05 GHz of a Fig. 5 circle (same BEM frequencies), which confirms the x calibrations of both figures.

## Transcriptions

* `butterweck1968_numbers.csv` (D5): numbers of Butterweck, IEEE MTT-16 (1968) and Butterweck & de Ronde, Philips Tech. Rev. 29 (1968), with sentence and page.
* `docs/literature/marcuvitz-steps-and-bends.md` (D6): Marcuvitz Secs. 5.24, 5.26, 5.28, 5.29, 5.34, 5.35, with ranges, accuracies, numerical checks against the book's curves and the checking record. Note: the H-plane step formulas (1a)-(3a) of Sec. 5.24a diverge from the book's own curves for a/λ ≳ 0.7.
* `docs/literature/doane1989.md` (D7): Doane & Anderson eqs. 14-18.

## Reproduction

`python3 benchmarks/literature/scripts/merge_ab.py` regenerates every merged CSV and this README from `raw/a` and `raw/b` (numpy only). `scripts/spot_check.py` re-reads three points per figure on the page images (needs the reference PDFs). `scripts/marcuvitz_checks.py` evaluates the Marcuvitz formulas (numpy, scipy). The raw digitizers' scripts in `raw/a/scripts` and `raw/b/scripts` need the PDFs and their scratch paths.

<!-- L2 begin (written by scripts/l2_merge.py) -->

## Card L2: Alimenti et al. 1994 and Kotik & Sivov 1965

Digitized twice by agent L2 with different rasters, calibrations and extraction methods (pass a: 400 dpi render, ridge follower, tick or gridline calibration with tilt correction; pass b: the embedded 300 ppi scan, column-run scan, major + minor ticks or straight gridline fits; raw sets in `raw/a/`, `raw/b/`, scripts in `scripts/l2/`), merged with the L1 rules by `scripts/l2_merge.py`; `scripts/l2_check.py` overlays the merged points on the rasterized figures and reports the pixel residuals. Same columns as above. Report: `docs/reports/L2.md`.

| Source | Figure | File | Curves | rows | y uncertainty: min-max (median) | a-b agreement: rms / max |
| --- | --- | --- | --- | --- | --- | --- |
| Alimenti, Mongiardo & Sorrentino 1994 | Fig. 2: \|S11\| (dB) vs f/fc, unmitered H-plane corner (d/a = 1.414) | `alimenti1994_fig2_mmfd.csv` | MMFD line (MM line coincides) | 151 | 0.0369-0.222 (0.0867) dB | 0.0326 / 0.105 dB |
| Alimenti et al. 1994 | Fig. 2 | `alimenti1994_fig2_mm.csv` | MM dots (10 markers) | 10 | 0.0353-0.0353 (0.0353) dB | 0.0288 / 0.0539 dB |
| Alimenti et al. 1994 | Fig. 3: \|S11\| (dB), fully mitered H-plane bend (t/a = 0, d/a = 0.707) | `alimenti1994_fig3_mm.csv` | MM line (MMFD coincides up to f/fc = 1.93) | 142 | 0.235-1.73 (0.376) dB | 0.13 / 0.476 dB |
| Alimenti et al. 1994 | Fig. 3 | `alimenti1994_fig3_mmfd.csv` | MMFD diamonds (10 markers) | 10 | 0.258-0.314 (0.258) dB | 0.208 / 0.628 dB |
| Alimenti et al. 1994 | Fig. 4: \|S11\| (dB) vs f/fc, mitered H-plane bends | `alimenti1994_fig4_t030.csv` | t/a = 0.30 (d/a = 0.919) | 150 | 0.169-2.08 (0.239) dB | 0.0895 / 0.4 dB |
| Alimenti et al. 1994 | Fig. 4 | `alimenti1994_fig4_t034.csv` | t/a = 0.34 (d/a = 0.948) | 147 | 0.184-2.61 (0.324) dB | 0.141 / 0.618 dB |
| Alimenti et al. 1994 | Fig. 4 | `alimenti1994_fig4_t038.csv` | t/a = 0.38 (d/a = 0.976) | 83 | 0.17-3.21 (0.481) dB | 0.569 / 1.91 dB |
| Alimenti et al. 1994 | Fig. 5: design chart vs t/a | `alimenti1994_fig5_matching_frequency.csv` | matching frequency f/fc (left axis) | 196 | 0.00166-0.00487 (0.0017) | 0.000258 / 0.00271 |
| Alimenti et al. 1994 | Fig. 5 | `alimenti1994_fig5_bandwidth.csv` | 30 dB bandwidth / fc (right axis), one-zero and two-zero branches, jump at t/a = 0.3867 | 196 | 0.00083-0.00233 (0.000987) | 0.000255 / 0.000935 |
| Alimenti et al. 1994 | Fig. 6: \|S11\| (dB), t/a = 0.386 (d/a = 0.980) | `alimenti1994_fig6_mmfd.csv` | MMFD | 118 | 0.0818-3.71 (0.124) dB | 0.0969 / 0.575 dB |
| Alimenti et al. 1994 | Fig. 6 | `alimenti1994_fig6_bcmm.csv` | BCMM line + 10 diamonds | 109 | 0.0816-0.591 (0.127) dB | 0.0916 / 0.588 dB |
| Kotik & Sivov 1965 | Fig. 3: \|B1\|² vs kd, 90° H-plane optical-mirror bend (d/a = 0.707) | `kotik1965_fig3_rigorous.csv` | rigorous solution (solid) | 416 | 0.00521-0.108 (0.00533) | 0.00098 / 0.00508 |
| Kotik & Sivov 1965 | Fig. 3 | `kotik1965_fig3_asymptotic.csv` | asymptotic formula 1 - 7.4/(kd)^(3/2) as drawn (dashed) | 442 | 0.00148-0.00338 (0.00149) | 0.00074 / 0.00188 |

**Geometry conversions (CONVENTIONS.md section 6, d = distance of the miter plane from the inner corner):**

* **Alimenti et al. 1994, t**: the dimension arrows of the inset of Fig. 4 run from the end of the 45° miter on the outer wall to the thin line that continues the other arm's inner wall, so t is the part of the outer wall left standing and the cut measured from the removed outer corner is c = a - t (Ma et al. 1997's c). Hence **d/a = (1 + t/a)/√2** (not (2 - t/a)/√2): t/a = 0 is the fully mitered bend of Fig. 3 (optical mirror, d/a = 0.707), t/a = 0.386 gives d/a = 0.980 (Ma's c/a = 0.614). Checked against Ma Fig. 11: the zero frequency rises with t/a here and falls with c/a there, and t/a = 0.30 (c/a = 0.70) puts the zero at f/fc ≈ 1.29 here, 1.38 in Ma.
* **Kotik & Sivov 1965**: d is the guide width (our a), kd = 2πa/λ; the mirror is the diagonal of the crossing square (Figs. 1-2), i.e. the optical mirror, d/a = 1/√2.
* Fig. 2 (unmitered corner) is d/a = √2.

**Agreement of the two passes, per curve** (same definitions as above):

| File | Curve | kind | n (both) | a only | b only | rms(a-b) | max abs(a-b) | mean(a-b) | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kotik1965_fig3_rigorous.csv` | rigorous | line | 365 | 30 | 21 | 0.00098 | 0.00508 | +0.00019 | 9 |
| `kotik1965_fig3_asymptotic.csv` | asymptotic | line | 440 | 1 | 1 | 0.00074 | 0.00188 | +1.1e-05 | 1 |
| `alimenti1994_fig2_mmfd.csv` | MMFD | line | 141 | 6 | 4 | 0.0326 | 0.105 | -0.017 | 0 |
| `alimenti1994_fig2_mm.csv` | MM markers | markers | 10 | 0 | 0 | 0.0288 | 0.0539 | -0.024 | 2 |
| `alimenti1994_fig3_mm.csv` | MM | line | 141 | 1 | 0 | 0.13 | 0.476 | +0.036 | 0 |
| `alimenti1994_fig3_mmfd.csv` | MMFD markers | markers | 10 | 0 | 0 | 0.208 | 0.628 | +0.095 | 1 |
| `alimenti1994_fig4_t030.csv` | t/a=0.30 | line | 145 | 0 | 5 | 0.0895 | 0.4 | -0.038 | 0 |
| `alimenti1994_fig4_t034.csv` | t/a=0.34 | line | 143 | 0 | 4 | 0.141 | 0.618 | -0.0022 | 0 |
| `alimenti1994_fig4_t038.csv` | t/a=0.38 | line | 48 | 33 | 2 | 0.569 | 1.91 | -0.11 | 1 |
| `alimenti1994_fig5_matching_frequency.csv` | matching frequency | line | 195 | 0 | 1 | 0.000258 | 0.00271 | +4.1e-05 | 0 |
| `alimenti1994_fig5_bandwidth.csv` | bandwidth one zero | line | 171 | 1 | 0 | 0.000108 | 0.000247 | -7.6e-05 | 0 |
| `alimenti1994_fig5_bandwidth.csv` | bandwidth two zeros | line | 17 | 7 | 0 | 0.000774 | 0.000935 | -0.00076 | 0 |
| `alimenti1994_fig6_mmfd.csv` | MMFD | line | 112 | 2 | 4 | 0.0969 | 0.575 | -0.02 | 0 |
| `alimenti1994_fig6_bcmm.csv` | BCMM | line | 91 | 2 | 6 | 0.0956 | 0.588 | -0.014 | 1 |
| `alimenti1994_fig6_bcmm.csv` | BCMM markers | markers | 10 | 0 | 0 | 0.0402 | 0.0955 | -0.00034 | 1 |

<!-- L2 end -->
