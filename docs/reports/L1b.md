# L1b report: literature benchmark data (card L1, second digitizer)

Agent L1b, branch `wave-1/L1b-literature-data`. Worked from the PDFs in Daniel's `references/` folder (staged copies),
independently of L1a (L1a's worktree and branch were not opened).

## What was done

**Digitized curves** (`benchmarks/literature/raw/b/*.csv`, index in `raw/b/README.md`). Method, the same for all
figures: page rendered with `pdftoppm -r 300`; axis calibration from tick marks or gridlines detected in the pixels
(positions recorded in each CSV header); curves traced in Python (numpy, scipy, Pillow): labels, insets and gridlines
masked, dark pixels assigned to hand-placed guide polylines, column-centre line where the curve is shallow, row-centre
line where it is steep, points close to another curve dropped. Every result was overlaid on the page image and
checked by eye. The `uncertainty` column is L1b's reading uncertainty of y: 2 px (half the line width) in both
directions, converted through the local slope, plus the calibration residual; capped (10 dB, or 0.1 in Ma Fig. 5)
where the curve is near vertical, where `x_uncertainty` is the useful number.

| Item | Figure | Status | Points | y uncertainty, median (range) |
| --- | --- | --- | --- | --- |
| D1 | Alessandri 1994 Fig. 4 (d/b = 1.41, 0.707, 0.87 + 38 crosses of ref. [7]) | done | 230 | 0.21 dB (0.19-0.87) |
| D1 | Alessandri 1994 Fig. 5 (d/b = 0.9, 0.87, 0.85, 0.83) | done | 564 | 0.28 dB (0.27-cap) |
| D2 | Doane & Anderson 1989 Fig. 9 (constant, cosine, triangular) | done | 786 | 0.010 % (0.010-0.042) |
| D3 | Ma 1997 Fig. 5 (|S21|², corner bend, 3/6/12 terms, BEM) | done | 358 | 0.008 (0.006-cap) |
| D3 | Ma 1997 Fig. 6 (|S21| dB, c/a = 1, 10/20/40/60 steps, BEM) | done | 503 | 0.16 dB |
| D3 | Ma 1997 Fig. 8 (|S11|, division methods a/b/c, 20 steps) | done | 760 | 0.51 dB |
| D3 | Ma 1997 Fig. 9 (|S11|, 60 steps, a=b=c) | done | 449 | 0.38 dB |
| D3 | Ma 1997 Fig. 11 (|S11| vs f/f_c, c/a = 0, 0.60, 0.631, 0.70, 1.0, 1.30) | done | 559 | 0.28 dB |
| D3 | Ma 1997 Table 1 (|S11|, |S21| at 16 GHz, 50/60 steps) | transcribed | 36 values | last digit |
| D4 | Reiter & Arndt 1995 Fig. 3(a)-(d) (H, mitered H, E, mitered E bends, γ = 90-150°) | done | 1663 | 0.25-0.34 dB |
| D4 | Reiter & Arndt 1995 Fig. 4 (cascaded mitered H + E bends, l = 0, b) | done | 334 | 0.50 dB |
| D8 | Kotik & Sivov 1965 | skipped (scan not available) | | |

**Transcriptions** (`docs/literature/raw/b/`):

* D5 `butterweck1968_numbers.csv`: 14 rows (7 from IEEE MTT-16, 7 from Philips Tech. Rev. 29), each with the exact
  sentence, journal page and PDF page, read on page images (both OCR layers contain errors).
* D6 `marcuvitz-steps-and-bends.md`: Secs. 5.24a/b (H-plane steps), 5.26a/b (E-plane steps), 5.28a/b and 5.29a/b
  (E- and H-plane corners, right-angle and arbitrary angle, including the measured table of 5.29b), 5.34 and 5.35
  (circular bends): every formula, the definitions, reference planes, stated ranges and accuracies, and a summary
  table. Checked symbol by symbol against 400-600 dpi renders.
* D7 `doane1989.md`: eqs. 14-18 and the unnumbered Δβ L_B formulas, with the worked WR187 example.

## Results and checks (numbers from code run by L1b)

* Calibration residuals: Alessandri x ≤ 0.0015, y ≤ 0.06 dB; Ma ≤ 1 px plus half the difference between opposite
  axes (up to 3.75 px in Ma Fig. 8, included in the uncertainty); Reiter (affine fit to gridline intersections)
  ≤ 1.2 px. Doane Fig. 9 is a photograph of a curved page: Coons-patch calibration through the four traced frame
  lines and 16 ticks; a single homography through the same points differs by ≤ 0.018 GHz and ≤ 0.028 %, and that
  difference is part of the uncertainty.
* Consistency with the papers' own numbers: Alessandri Fig. 4, the highest d/b = 0.87 point is -26.07 dB (at
  f/f_c = 1.41; the curve leaves the axes below -40 dB near 1.8-1.87) ("better than -26 dB in the whole band", p. 410). Ma Fig. 9 at 16 GHz: -11.27 dB vs Table 1 (60 steps,
  Ref.) 20 log₁₀ 0.2622 = -11.63 dB (difference 0.36 dB, within the 0.38 dB uncertainty); Ma Fig. 11, c/a = 1.0 at
  f/f_c = 16/9.488: -11.62 dB; Ma Fig. 6 at 16 GHz: -0.42 dB vs 20 log₁₀ 0.9650 = -0.31 dB (difference 0.11 dB,
  within 0.16 dB). Ma Table 1 Ref. rows: |S11|² + |S21|² = 1.00001 and 0.99997.
* Marcuvitz transcription checked numerically against the book's own curves: E-plane right-angle corner d/b = 0.392,
  Bλ_g/Y₀2b = 2.066 at 2b/λ_g = 0.5 (Fig. 5.28-3: ≈ 0.39, ≈ 2.07); H-plane corner d/a = 0.653 and B2a/Y₀λ_g = 0.538
  at x → 0 (Fig. 5.29-3: ≈ 0.654, ≈ 0.55); the printed constants A₁ = 0.3366, A₁' = 0.3071 are reproduced (0.336574,
  0.307152). Arbitrary-angle E-plane corner: Marcuvitz's Ψ(x) "derivative of the logarithm of x!" is ψ(x+1)
  (digamma of x+1); with it Eq. (1) gives 0.645/0.878/1.073 at 60/90/120°, as in Fig. 5.28-6; the plain digamma does
  not (2.88 at 90°).
* Doane eq. 16 example: 5a²/λ_min = 5.70 in and 16b²/λ_min = 84.1 in at 17.7 GHz vs the paper's "5"" and "70""
  (paper does not state its λ). Eq. 17 integrates to θ_tot (sympy).

## Acceptance tests

| Check | Status | Value |
| --- | --- | --- |
| Every curve has two independent digitizations | L1b half done | all D1-D4 curves above digitized by L1b; difference to L1a is for the coordinator's merge |
| Spot check by the coordinator: 3 points per figure re-read by eye | not done (coordinator) | overlays checked by L1b for every figure |
| Transcribed formulas (D6, D7) checked symbol by symbol by the other L1 agent | not done (needs L1a) | L1b checked its own transcription against 400-600 dpi renders and numerically (above) |
| Full test suite | pass | 93 passed |

## Open issues

1. **Curve identification judgments** (stated in the CSV headers): Ma Fig. 11 line styles mapped to c/a labels by
   L1b (the c/a = 0.631 double-null curve follows the paper's 39 % bandwidth statement); Reiter Fig. 3(b) right-hand
   branches assigned to γ = 105/120° by ordering; Reiter Fig. 3(d) rising branch at 17.2-18 GHz left as "right
   branch" (not identifiable); Reiter Fig. 4 null at 11.7 GHz assigned to l = b by continuity. L1a's reading of these
   should be compared before merging.
2. **Gaps**: Alessandri Fig. 4 dashed d/b = 0.707 curve is hidden by its crosses for f > 1.45 (sparse there);
   Reiter Fig. 3(a)/(c) curves are interrupted by labels between 16.0 and about 16.95 GHz; Reiter Fig. 4 theory and
   measurement are only traced where they coincide (separate near the nulls, minima noted in the header); Ma Fig. 6
   20- and 10-step curves are sparse (curves 3-8 px apart); Ma Fig. 11 c/a = 1.30 below 1.55 and c/a = 0 above 1.9
   run along the top frame.
3. **Doane Fig. 9 triangular curve** rises from 0.047 % at 15 GHz to 0.150 % at 18 GHz in L1b's calibration; it is a
   straight line in the photograph while the page (and frame) is curved. Both calibrations agree within 0.012 %; if
   L1a finds a flat curve, the page curvature model is the cause.
4. **Card item D5 "0.6/2.5/5.4 dB over 15 cm"**: the three values appear together only in Philips Tech. Rev. p. 95
   (measured); IEEE MTT p. 279 gives 0.6 and 5.4 dB as theory. **Strangled bend**: the caption reads "λ ≧ 4 mm", not
   "λ ≈ 4 mm". **"H-plane miter claim"**: Philips p. 97 attributes "< 0.4 dB if the frequency exceeds eight times the
   cut-off frequency" to Bouwkamp's theory of the optical-approach H-plane bend (Fig. 14, a reflecting corner), not
   explicitly to a "miter".
5. **Marcuvitz has no mitered (chamfered) bend in pp. 296-335**; only sharp corners (right-angle and arbitrary-angle,
   reference planes through the inner corner for the latter). Typographic ambiguity in 5.26a eq. (2a)/(2b): the scope
   of `ln` (read as covering the power factor) is noted in the file.
6. Card branch name says `wave-1/L1-literature-data`; I used `wave-1/L1b-literature-data` as instructed by the
   coordinator. Deliverable paths follow the coordinator's instruction (`raw/b/`), not the final merged names; the
   merged `benchmarks/literature/README.md` index is left to the coordinator.

## Conventions issues

* CONVENTIONS §9 asks for an `uncertainty` column; I added `x_uncertainty` as a fifth column because several curves
  are near vertical (nulls), where a y uncertainty is meaningless. Merging code should keep or drop it consistently.
* Reiter & Arndt's γ in Fig. 3 is the interior angle at the inner corner (γ = 90° is a right-angle bend, bend angle
  180° − γ), whereas CONVENTIONS §6 speaks of the bend angle θ; whoever uses these data must convert.
* Ma et al.'s c and Alessandri's d describe the miter differently (Ma, Fig. 8 inset: c = length of the legs of the
  triangle cut from the outer corner, c/a = 0 is unmitered; Alessandri: d = distance from the inner corner to the
  miter wall, d/b = 1.41 is unmitered; Reiter: a_opt, b_2opt measured the same way as Alessandri's d). Not a conventions error, but the miter geometry parameter of WP5-type components must be mapped per
  paper.
