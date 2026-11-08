# T1 — Ohmic loss and impedance walls in the coupled-mode equations

Role: Theory · Wave: 1 · Branch: `wave-1/T1-wall-impedance` · Depends on: T0
Verified by: a second, independent theory agent (T1b, same card, separate context), then WP2 tests and the
2D solver with impedance walls (wave 2) · Time box: 4 days (then status review)

## Goal

Derive, in the conventions of `CONVENTIONS.md`, the coupled-mode equations of a rectangular guide whose walls
have a surface impedance $Z_s$. These equations must cover:

* finite conductivity on all four walls, as in an aluminium transmission line;
* a large impedance on some walls, as in Butterweck's lossy-wall mode filter.

Extend the result to bends, to first order in $Z_s$, and to a thin resistive sheet parallel to the broad walls,
as in Butterweck's sheet filter. WP2 implements what this note derives.

## Read first

* `CONVENTIONS.md`, `docs/theory/T0-field-convention.md`. The projection used in T0 is the starting point; here
  the boundary terms that vanished for perfect walls are kept.
* Butterweck, IEEE MTT-16, 274–281 (1968): eqs. 1–19 (resistive sheet, LS modes) and eqs. 20–31 (lossy side
  walls, coupled-mode treatment).
* Marcuvitz, *Waveguide Handbook* (1951), §1.6 / attenuation formulas, for the per-mode attenuation constants.

## Specification

1. **Leontovich boundary condition** $E_t = Z_s\, H_t\times\hat n$ on a wall with outward normal $\hat n$, sign
   checked for the $e^{+i\omega t}$ convention, with $Z_s = (1+i)R_s$ for a good conductor.
2. **Straight guide, all walls lossy.** Write $dV_p/dz$ and $dI_p/dz$ (T0 notation) with the boundary terms
   $\oint$ over the wall contour. Then convert to power waves:
   $da^\pm/dz = \mp i\beta a^\pm - \sum_q (\Lambda^{\pm\pm}_{pq} a^\pm_q + \Lambda^{\pm\mp}_{pq} a^\mp_q)$.
   Give closed forms of $\Lambda$ for TE–TE, TM–TM and TE–TM pairs of the same and of different $(m, n)$, and
   per wall, so that walls can have different $Z_s$.
3. **Checks of item 2:**
   * The diagonal terms must reproduce the textbook attenuation constants of TE$_{mn}$ and TM$_{mn}$ (e.g. TE₁₀:
     $\alpha = R_s/(\eta_0 b\sqrt{1-(f_c/f)^2})\,[1 + (2b/a)(f_c/f)^2]$).
   * State which off-diagonal terms matter. Phase-matched pairs (the degenerate TE$_{mn}$/TM$_{mn}$) couple
     resonantly, so loss selects LS-like combinations; non-matched pairs average out.
   * Quantify both effects for WR-90 at 18–90 GHz with aluminium.
4. **Bends.** Combine with the curvature terms of T0 to first order in $Z_s$, including the factor $h = 1+\kappa u$
   on the wall integrals. State the neglected cross terms ($\kappa Z_s$) and estimate them for the SPARC bends.
5. **Large wall impedance** (lossy-wall filter, $|Z_w|$ up to about 250 Ω). Say where the first-order
   (Leontovich-perturbation) treatment fails. Give the exact alternative for a guide with two impedance walls
   ($u = \pm a/2$): the modes of the impedance-walled guide from a transcendental equation, then mode matching to
   the ideal guide. Reproduce Butterweck's eq. 21 in the low-impedance limit.
6. **Resistive sheet** at $v = v_0$ parallel to the broad walls, sheet resistance $R_\square$. The LS eigenvalue
   equations (Butterweck eqs. 8–9, generalized to an off-centre sheet). The junction from the empty guide to the
   sheet-loaded guide, as overlap integrals of the empty-guide modes with the sheet-guide modes.

## Acceptance tests

| Check | Target |
| --- | --- |
| Diagonal attenuation vs textbook TE/TM formulas, WR-90 and a square guide | 1e-6 relative |
| Item 2 coefficients vs direct numerical surface integrals of $\oint \lvert H_t\rvert^2$ and the cross terms (script in `docs/theory/T1/`) | 1e-8 relative |
| Butterweck lossy-wall example (X-band, 75 GHz, $Z_w$ = 250 Ω real) | attenuation 4 dB/m (H₁₀), 36 dB/m (H₃₀); over 15 cm, 0.6 dB (H₁₀) and 5.4 dB (H₃₀) |
| Butterweck resistive-sheet example ($b$ = 10 mm, λ = 4 mm) | optimum $R_\square$ ≈ 56 Ω, maximum EH₁₁⁽²⁾ attenuation ≈ 1.5 dB/cm |
| T1 and T1b, derived independently | identical coefficient formulas (or a discrepancy report) |

## Deliverables

* `docs/theory/T1-wall-impedance.md`: derivation, final formulas boxed, validity limits, and the WR-90 aluminium
  numbers of item 3.
* `docs/theory/T1/`: sympy and quadrature scripts that check every final formula.

## Out of scope

Surface roughness, anisotropic or frequency-dispersive wall materials, dielectric sheets of finite thickness
(note where thickness matters), implementation in `src/` (WP2).
