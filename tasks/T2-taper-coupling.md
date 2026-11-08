# T2 — Coupled-mode equations of tapers (cross-section method), with curvature

Role: Theory · Wave: 1 · Branch: `wave-1/T2-taper-coupling` · Depends on: T0
Verified by: T2b (independent derivation), then the staircase-of-steps reference of WP4 in wave 2 ·
Time box: 5 days (then status review)

## Goal

Derive the coupled local-mode equations of a rectangular guide whose cross-section varies along $z$: $a(z)$ and
$b(z)$, with an optional centre offset $u_0(z)$, $v_0(z)$, and an optional curvature $\kappa(z)$ in the $u$–$z$
plane. Do it in a form that stays regular when a mode crosses cutoff inside the taper. These equations are the
basis of WP5 (tapers and the strangled E-plane bend, wave 2).

## Read first

* `CONVENTIONS.md`, `docs/theory/T0-field-convention.md`.
* Katsenelenbaum, *Theory of Nonuniform Waveguides: The Cross-Section Method* (1998): chapters on the local-mode
  expansion and on tapers (coupling coefficients proportional to the wall slope).
* Butterweck & de Ronde, Philips Tech. Rev. 29 (1968), pp. 95–97: the "bottle" taper, reflection of a mode at its
  cutoff plane, and which modes a symmetric width or height taper excites.
* Huang, *Microwave Principles* II (1965): check whether tapers are treated, and compare if so.

## Specification

1. **Local modes.** At each $z$, expand in the modes of the local rectangle, with the normalization of
   `CONVENTIONS.md` evaluated with the local $a, b$. Derive the coupling coefficients in the T0
   voltage–current form ($dV/dz$, $dI/dz$), proportional to $a'(z)$, $b'(z)$, $u_0'$, $v_0'$. Then derive the
   power-wave form. Give the closed forms for all TE/TM pairs.
2. **Selection rules.** Spell them out:
   * A symmetric width taper couples $\Delta m$ even within a family.
   * A symmetric height taper couples $\Delta n$ even, across families.
   * An offset couples odd differences.

   Say which conserve the family structure of `CONVENTIONS.md` §3.
3. **Cutoff crossings.** The power-wave coefficients contain $1/\sqrt{\beta_p\beta_q}$ and diverge at cutoff. Show
   that the V–I (or impedance/admittance Riccati) form is regular there. Specify a solution method that keeps
   evanescent modes and remains stable: Riccati for the input admittance matrix, or GSM slices with a local
   basis per slice. Specify how evanescent modes are counted and how the result converges in that number.
4. **Curvature and taper together** (strangled bend). Combine with the T0 curvature terms when the local
   cross-section varies. State whether the curvature coefficients simply take the local $a(z)$, $b(z)$, and which
   cross terms appear.
5. **Small-slope limit.** Give the first-order (single-scattering) conversion formula, an integral of the coupling
   coefficient times $e^{-i\int\Delta\beta}$, used to check the full solution and to explain the bottle shape.

## Acceptance tests

| Check | Target |
| --- | --- |
| Closed forms vs numerical derivative of the overlap $\iint e_p(a+\delta a)\cdot e_q(a)$ | 1e-8 relative |
| Uniform-guide limit ($a' = b' = 0$) | equations reduce to T0 exactly |
| Lossless: power-wave coupling matrix structure | net power flux conserved, shown analytically |
| T2 and T2b, derived independently | identical formulas (or a discrepancy report) |

## Deliverables

* `docs/theory/T2-taper-coupling.md` with boxed final formulas, the cutoff-crossing method, and validity limits.
* `docs/theory/T2/` check scripts.

## Out of scope

Implementation (WP5, wave 2), non-rectangular local cross-sections, twists (T3).
