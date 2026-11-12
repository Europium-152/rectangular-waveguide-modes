# WP3b — All miters against the literature; impedance walls in the 2D solver

Role: Implementer · Wave: 2 · Branch: `wave-2/WP3b-miters` · Depends on: WP3 (merged), L1 data (merged) ·
Uses: L2 data · Verified by: WP4b (mode matching), WP7b, COMSOL campaign 2 · Time box: 7 days (then status review)

## Goal

The 2D modal-port solver already handles H- and E-plane, plain and chamfered miters and general outlines. Wave 2
validates all of them against the published data, adds chained-miter outlines for the design studies, and adds
impedance (Leontovich) walls on the walls the 2D reduction keeps exact, which gives an independent check of the
WP2 lossy-wall filter and of the in-plane part of bend loss.

## Read first

* `CONVENTIONS.md` §6 (miter reference planes; plain miter = optical mirror, decided 9 Nov 2026), §7, §8.
* `docs/reports/WP3.md` (open issues 1–4, the Ma geometry section) and `src/wrmodes/fem2d/`.
* `benchmarks/literature/README.md` and the CSV files: `alessandri1994_fig4/5`, `ma1997_fig5/6/8/9/11`,
  `ma1997_table1`, `reiter1995_fig3a–d`, `reiter1995_fig4` (uncertainties in the files; L1 open issues 2–3).
* L2's files `alimenti1994_fig2–6` and `kotik1965_fig3` (when L2 lands).
* Alessandri et al. (1994) Figs. 1, 4, 5; Alimenti et al. (1994); Kotik & Sivov (1965); Ma et al. (1997); Reiter & Arndt (1995) Figs. 3–4 (geometry
  definitions: bend angle γ, miter size, WR62 and WR90 dimensions, the spacing l of Fig. 4).
* `benchmarks/comsol/campaign_1/` and the Gate A section of `benchmarks/validation_v1/REPORT.md`.
* `docs/theory/T1-wall-impedance.md` (Leontovich condition, sign conventions).

## Specification

1. **Literature acceptance.** A script per source in `benchmarks/fem2d/` that computes the published quantity in
   the published geometry and compares it within the CSV uncertainty:
   * Ma et al. (1997): Figs. 5, 6, 8, 9, 11 and Table 1 (H-plane, right-angle corner and mitered bends);
   * Alessandri et al. (1994): Figs. 4 and 5 (90° E-plane, chamfers d/b = 0.707–1.41). The paper does not give
     the broad side; compute both b/a = 0.5 and the WR-90 ratio 0.444 and report which matches (WP7 open issue);
   * Alimenti et al. (1994): Figs. 2–6 (H-plane, unmitered, fully mitered, $t/a$ = 0.30–0.40), including the
     design chart of Fig. 5 (matching frequency and 30 dB bandwidth vs $t/a$, the jump at 0.386);
   * Kotik & Sivov (1965): Fig. 3, TE₁₀ transmission of the optical-mirror H-plane miter for $ka$ = 10–32
     (WR-90: 21–67 GHz), and their asymptotic $1 - 7.4/(ka)^{3/2}$ above $ka = 6\pi$;
   * Reiter & Arndt (1995): Fig. 3 (H- and E-plane bends and mitered bends, γ = 90–150°, WR62) and Fig. 4
     (cascaded mitered H- and E-plane bends in WR90, l = 0 and l = b) as a `Line` with the frame change between
     the two miters. l = 0 is a zero-length cascade: evanescent modes must be in the bases (CONVENTIONS §3), and
     WP3 open issue 1 applies.
2. **Chained miters.** `chained_miter_outline(angles_deg, spacers, chamfers=None, a=...)`: one outline with n
   miters joined by straight spacers (centre-line lengths), with the reference planes of the first and last miter.
   Test: as the spacers grow, the single-domain result converges to the `Line` cascade of `Miter`s and
   `Straight`s (evanescent modes included), and the difference at short spacers is reported (it measures how many
   evanescent modes a cascade needs).
3. **Impedance walls.** Optional `wall_impedance` on `Coplanar2D` and `Miter`: a surface impedance $Z_s(f)$
   (callable or a `Guide`'s σ) on the walls **parallel to v** (the in-plane walls of the outline), where the
   Leontovich condition keeps the per-family reduction exact: a Robin condition on ψ for LSM and on φ for LSE.
   Derive both conditions (signs included) in `docs/theory/WP3-normalization.md` (new section). The walls at
   $v = \pm b/2$ are not included (they couple families); say so in the docstring. The discrete system stays
   complex-symmetric.
4. Keep the exact-cutoff rule, the 'live' rule and the frame tags of wave 1 unchanged.

## Acceptance tests

| Check | Target |
| --- | --- |
| Ma (unmitered corner and the figures other than the chamfered curves of Fig. 11), Alessandri (b/a = 0.5, WP4b), Alimenti Figs. 2–3, Kotik, Reiter Figs. 3–4 | within the digitizing uncertainty of each CSV point; outliers listed with a reason (near-cutoff points, figure anomalies of L1) |
| Reiter Fig. 3(a) γ = 150, Figs. 3(b)–(d), Fig. 4; Ma c/a = 1 at 40–60 steps | qualitative (Daniel, 12 Nov 2026), `docs/discrepancies/2026-11-12-reiter-and-ma-many-step-miters.md` |
| Alimenti Figs. 4–6, Ma Fig. 11 chamfered curves | qualitative (Daniel, 11 Nov 2026): report the comparison; see `docs/discrepancies/2026-11-11-alimenti-chamfered-miters.md` |
| Ma Table 1 | qualitative: FEM, mode matching and finite differences give 0.2778 against 0.2622 (same discrepancy file) |
| Chained miters | single domain → cascade as spacers grow; difference < 1e-6 at spacers ≥ 2a (90° pairs), ≥ 2.5a (45° optical pairs) |
| Impedance walls: straight channel | per-mode α equals T1's exact roots to 1e-6 relative (first-order `wrmodes.loss` differs from exact by up to 7e-4); for n ≥ 1 the Leontovich condition couples LSM and LSE (CONVENTIONS §3) |
| Butterweck lossy-wall filter (75 GHz, 250 Ω, 15 cm; WP2 report) | the FEM, with a straight outline of 15 cm, within 5 % of WP2; Butterweck's 0.6 / 2.5 / 5.4 dB reported |
| Reciprocity (lossless and lossy), unitarity (lossless), passivity (lossy) | CONVENTIONS §6 |

## Deliverables

* Changes in `src/wrmodes/fem2d/`; `tests/test_fem2d.py` additions.
* `benchmarks/fem2d/` comparison scripts and npz results (CONVENTIONS §9); registry entries.
* `docs/reports/WP3b.md`.

## Out of scope

Walls at $v = \pm b/2$ with loss; sheets inside the guide; non-coplanar components.
