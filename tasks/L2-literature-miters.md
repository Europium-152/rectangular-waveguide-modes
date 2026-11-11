# L2 — Literature data: Alimenti et al. (1994) and Kotik & Sivov (1965)

Role: Literature · Wave: 2 · Branch: `wave-2/L2-literature` · Depends on: L1 (merged; same formats) · Used by:
WP3b, WP4b, WP7b, WP8a · Time box: 2 days

## Sources (Daniel's reference folder; read-only, never committed)

* Alimenti, Mongiardo & Sorrentino, "Design of mitered H-plane bends in rectangular waveguides by combined mode
  matching and finite differences", 24th EuMC (1994) 289–293.
* Kotik & Sivov, "Diffraction of electromagnetic waves on a mirror in a break of a plane waveguide", Radio
  Engineering and Electronic Physics 10 (1965) 140–142 (a scan; the paper is pp. 2–4 of the PDF).

## What to extract

1. **Alimenti et al.** (H-plane 90° bends, single-mode band, abscissa $f/f_c$ of TE₁₀):
   * Fig. 2: $\lvert S_{11}\rvert$ (dB) of the unmitered corner, MM and MMFD;
   * Fig. 3: the fully mitered bend (miter through the inner corner);
   * Fig. 4: $\lvert S_{11}\rvert$ (dB) for $t/a$ = 0.30, 0.34, 0.38 (zeros and their frequencies);
   * Fig. 5: the design chart, normalized matching frequency (left axis) and normalized 30 dB bandwidth (right
     axis) vs $t/a$ = 0.30–0.40, including the jump at $t/a$ = 0.386;
   * Fig. 6: $\lvert S_{11}\rvert$ (dB) at $t/a$ = 0.386, MMFD and BCMM.
   * The geometry: $t$ is the cut measured along each outer wall from the outer corner (insets of Figs. 4–5).
     Give the conversion to the CONVENTIONS §6 chamfer $d$ (distance of the miter plane from the inner corner,
     normalized to $a$), which should be $d/a = (2 - t/a)/\sqrt2$; check it against the insets and state any doubt.
2. **Kotik & Sivov** (90° bend of a parallel-plate guide of width $d$ with the $E$ field parallel to the mirror
   edge, i.e. the H-plane TE$_{n0}$ problem; the mirror on the diagonal of the crossing square, i.e. the
   **optical mirror**, CONVENTIONS $d/a = 1/\sqrt2$; check this reading of Figs. 1–2):
   * Fig. 3: $\lvert B_1\rvert^2$ (TE₁₀ power transmission) vs $kd$ from 10 to about $10\pi$, solid (their
     rigorous solution, up to $N = 18$ modes) and dashed (the asymptotic formula);
   * the asymptotic formula $\lvert\tilde B_1\rvert^2 = 1 - 7.4/(kd)^{3/2}$ (after Katsenelenbaum 1963), its
     stated validity ($kd > 6\pi$, relative deviation ≤ 3 %), and the remarks on $kd$ = 30 (3 % loss) and on the
     critical frequencies $kd = n\pi$;
   * a short summary of their method (eqs. 2–10: image system of two crossed guides, Green's theorem, truncation)
     in your own words, for WP4b/WP7b, who may use it as a further independent method.

## Format

As L1: one CSV per figure curve in `benchmarks/literature/` (`alimenti1994_fig2.csv`, …, `kotik1965_fig3.csv`)
with a header (source, figure, axes and units, geometry in CONVENTIONS terms, digitizing method, uncertainty in
both axes), entries in `benchmarks/literature/README.md` and `benchmarks/registry.json`. No reproduction of text
beyond short quotations; figures are digitized, not copied.

## Acceptance

* Every curve listed above, with uncertainties from a repeat digitization.
* A check script that overlays the digitized points on the rasterized figure (kept out of the repository) and
  reports the residual in pixels, as in L1.
* `docs/reports/L2.md`: what was extracted, the geometry conversions and their doubts.
