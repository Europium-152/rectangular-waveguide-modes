# L1a report: literature benchmark data (card L1, set a)

Agent L1a, branch `wave-1/L1a-literature-data`, 2026-11-08. L1b digitized the same figures independently; I did
not look at its work. The coordinator merges sets a and b.

## What was done

All figures were digitized from the native page images in the PDFs (`pdfimages`: 339 dpi 1-bit scans for
Alessandri and Reiter, 300 dpi 1-bit scans for Ma, a JPEG phone photograph for Doane), with pixel analysis in
numpy and scipy (`benchmarks/literature/raw/a/scripts/`):

* **Calibration** comes from the printed tick marks or dotted grid lines, located in pixels and fitted by least
  squares. Skewed scans are deskewed (Alessandri, 0.63° and 0.54°) or calibrated with an affine map (Ma Figs. 8–9,
  Reiter Fig. 3). The Doane photograph has perspective and page-curl distortion; it is calibrated with a Coons
  patch through the four frame curves, parameterized by the ticks. The calibration points are in every CSV header.
* **Curves** are tracked by a curvature-aware follower started from seeds, so that the follower bridges dash
  gaps and crossings. Each curve is then measured by ink centroids in 3 px bins along the track, with outlier
  rejection. Grid lines, labels, insets and other curves within a few px of a crossing are masked. Every
  extraction was checked visually on zoomed overlays.
* **Markers**: the crosses (Alessandri Fig. 4) are found as plus-shaped ink, and their centres are taken from the
  arm centroids. The circles (Ma Figs. 5–6) are found by ring template matching followed by a least-squares
  circle fit.
* **Uncertainty** (per row, y at the given x): half the printed line width (perpendicular, converted to vertical
  with the local slope), combined with a 1–1.5 px calibration error in both axes, the interpolation error across
  dash gaps, and model terms where needed (Doane: half the Coons-vs-homography difference and the y-scale
  anomaly below). The card's "axis resolution" (half tick spacing / px between ticks) is given in each header. It
  is always smaller (0.0006–0.05 in axis units) than the line-width term, which dominates.

Outputs:

* `benchmarks/literature/raw/a/*.csv` (14 files), index and reproduction notes in
  `benchmarks/literature/raw/a/README.md`, scripts in `scripts/`. `prepare.py` regenerates the derived page
  images bit for bit.
* `docs/literature/raw/a/butterweck1968_numbers.csv` (D5), `marcuvitz-steps-and-bends.md` (D6),
  `doane1989.md` (D7).

## Results

| Item | Files | Rows | Median y uncertainty |
| --- | --- | --- | --- |
| D1 Alessandri 1994 Figs. 4, 5 | `alessandri1994_fig4.csv`, `_fig5.csv` | 174 (incl. 38 crosses of ref. [7]), 718 | 0.17 dB, 0.20 dB |
| D2 Doane 1989 Fig. 9 | `doane1989_fig9.csv` | 337 | 0.008 % |
| D3 Ma 1997 Figs. 5, 6, 8, 9, 11 and Table 1 | `ma1997_fig5/6/8/9/11.csv`, `ma1997_table1.csv` | 144, 232, 295, 152, 841, 10 | 0.008 (lin.), 0.13, 0.23, 0.32, 0.19 dB |
| D4 Reiter 1995 Figs. 3(a)–(d), 4 | `reiter1995_fig3a/3b/3c/3d/4.csv` | 450, 285, 483, 313, 234 | 0.16–0.33 dB |
| D5 Butterweck / Butterweck & de Ronde | `butterweck1968_numbers.csv` | 20 quotes | — |
| D6 Marcuvitz pp. 296–335 | `marcuvitz-steps-and-bends.md` | Secs. 5.24, 5.26, 5.28, 5.29, 5.34, 5.35 | — |
| D7 Doane eqs. 16–18 | `doane1989.md` | eqs. 14–18 and the unnumbered ΔβL relations | — |
| D8 Kotik & Sivov 1965 | — | not done (scan not available) | — |

Internal consistency checks (all from code):

* Alessandri prints the d/b = 0.87 curve twice, as dots in Fig. 4 and as a line in Fig. 5. At the 38 dots of
  Fig. 4, the two independent extractions differ by 0.12 dB rms and 0.33 dB at most. All 38 agree within the
  combined uncertainty.
* Alessandri Fig. 4: the dotted grid lines agree with the tick calibration within 1.5 px; the tick fit residuals
  are ≤ 0.036 dB and ≤ 0.0014 in f/fc.
* Ma Fig. 9 (60 steps, method a) at 16 GHz gives −11.51 dB (interpolated), against Table 1 (Ref., 60 steps):
  |S11| = 0.2622, which is −11.63 dB. The difference is 0.12 dB; the row uncertainty is 0.30 dB.
* Ma Fig. 5 / Fig. 6: the BEM circles fall at the same frequencies in both figures (10.42–10.45, 11.40–11.43,
  12.33–12.36, 13.27–13.29, 14.24–14.26, 15.17–15.19, 16.13–16.16 GHz). This confirms the x calibration of both.
* Marcuvitz Sec. 5.29: the printed constants A₁ = 0.3366 and A₁' = 0.3071 were recomputed from the transcribed
  expressions: 0.336574 and 0.307152.

## Acceptance tests

| Check | Target | Status |
| --- | --- | --- |
| Every curve has two independent digitizations | yes, with their difference recorded | **set a done** for D1–D4. The difference needs L1b's set (coordinator). |
| Spot check by the coordinator: 3 points per figure re-read by eye | within the stated uncertainty | not done (coordinator). The overlays I checked are reproducible with the scripts. |
| Transcribed formulas (D6, D7) checked symbol by symbol by the other L1 agent | yes | not done (L1b). Items needing a second look are listed at the end of the D6 file. |

## Open issues

1. **Doane Fig. 9 y-axis.** The side ticks 0, 0.5, 1, 1.5 % are evenly spaced (about 190 px per 0.5 %). The top
   frame, labelled 2 %, is only about 174 px above the 1.5 % ticks on both sides. With a tick-only scale the top
   frame would be at 1.94–1.96 %. The data use the frame as 2 %. The difference to a tick-only calibration
   (≤ 0.009 %, larger than 0.005 % only above 1.5 %) is included in the uncertainty. Also, the "triangular" curve
   rises from 0.05 % at 15 GHz to 0.14 % at 17.75 GHz relative to the axes. It looks flat in the photograph only
   because of the perspective.
2. **Parts not digitized** (stated in each header): reflection zeros below the plot frames; the curve segments
   hidden by labels in Reiter Fig. 3(a)/(c) (about 16.1–17.2 GHz); Ma Fig. 6 the 40-step line (it coincides with
   the 60-step line); Ma Fig. 8 (b) above 14.9 GHz and (c) above 16.3 GHz (within about 3 px of (a)); Reiter
   Fig. 4, the theory/measurement split near the two l = b minima and near the crossings (only the measured
   minimum at 11.73 GHz, −59.8 dB, is given). Wider uncertainties where curves touch: Ma Fig. 11 c/a = 0.60 at
   1.800–1.825 (≥ 0.7 dB), and curves merging with the 0 dB frame (≥ 0.3 dB).
3. **Ma et al. reference labels**: the legends say "BEM [9]", but the reference list ends at [8]. The text cites
   the BEM results as [6] (Koshiba and Suzuki 1986). Table 1 does not state c/a; c/a = 1.0 is inferred from the
   surrounding figures.
4. **Corrections to the card's descriptions**:
   * Reiter & Arndt 1995 **does** contain bend results. Fig. 3 has (a) H-plane bends, (b) mitered H-plane bends,
     (c) E-plane bends and (d) mitered E-plane bends, all in WR62 for γ = 90–150°. Fig. 4 is a cascaded pair of
     mitered H- and E-plane bends in WR90 (a_opt = 0.976a, b_opt = 0.874b), theory and measurement. These were
     digitized. Figs. 5–8 (filters, rat race) are not bends and were not digitized.
   * D5: the print says λ **≧** 4 mm for the strangled E-plane bend, not λ ≈ 4 mm (Philips p. 98, Fig. 15). The
     0.6/2.5/5.4 dB measured attenuations are in the Philips paper (p. 95). The IEEE paper gives the computed
     0.6 and 5.4 dB (p. 279). The "< 0.4 dB above 8 f_c" claim (Philips p. 97) refers to the H-plane bend of the
     "optical approach", a plane-mirror corner (Fig. 14). It is credited to C. J. Bouwkamp (unpublished), and ref.
     [18] also cites Kotik and Sivov 1965 (D8).
   * D6: pp. 296–335 contain no mitered-bend section. They give sharp corners (Secs. 5.28, 5.29, including
     arbitrary angles; the arbitrary-angle H-plane corner as a measured table only) and circular bends
     (Secs. 5.34, 5.35). I transcribed all of these, plus the H-plane steps (5.24) and E-plane steps (5.26).
5. Alessandri 1994 does not state the guide dimensions; the results are given against f/fc only.
6. The scripts contain absolute scratch paths and need the reference PDFs. They document the method, but CI
   cannot run them.

## Conventions issues

* None blocking. CONVENTIONS.md §9 asks for "a header comment giving the source (reference, figure, axes, units)
  and an `uncertainty` column". The card asks for the calibration points and digitizers in the header as well;
  both are present. I added an `x_uncertainty` column for markers and dots, which carry their own x. The
  coordinator may drop it or keep it in the merged files.
* Geometry definitions differ between sources and from CONVENTIONS.md §6 (miter reference planes through the
  inner corner). Alessandri's d and Reiter's a_opt/b_opt are distances from the inner corner to the miter wall,
  measured perpendicular to it. Ma's c is the length cut from each outer wall, measured from the removed outer
  corner. Reiter's γ is the interior angle (deflection = 180° − γ). Marcuvitz's θ is the deflection angle. The
  published |S11| and |S21| magnitudes do not depend on the reference planes, but any phase comparison does. Each
  CSV header states the definition used.
