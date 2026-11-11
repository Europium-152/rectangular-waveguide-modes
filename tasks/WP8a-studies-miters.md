# WP8a — Design studies 1–2: H-plane miters vs smooth bends; chained miters (optional in wave 2)

Role: Implementer (study) · Wave: 2 (starts only if Daniel agrees; otherwise wave 3) · Branch:
`wave-2/WP8a-studies-miters` · Depends on: wave 1 (merged; Gate A passed); WP3b for miter wall loss ·
Reviewed by: Reviewer + Daniel · Time box: 6 days (then status review)

## Goal

Answer the first two design questions of the plan with the components validated at Gate A, in WR-90 over
18–90 GHz with aluminium walls (σ = 3.5e7 S/m) and the SPARC 1 dB requirement.

## Questions

1. **H-plane miter vs smooth H-plane bend.** Does the plain (optical) or a chamfered H-plane miter stay below
   0.4 dB above 52 GHz, as Bouwkamp claims (Butterweck & de Ronde 1968, p. 97; WP3 report)? How bad is it at
   18–52 GHz? Compare with the 60–70 cm sech H-plane bend, lossy. Output: loss vs frequency for both, the
   crossover frequency, and the floor-space saving.
2. **Chained miters.** A 90° turn made of n miters of 90°/n (n = 2, 3) with straight spacers. Can the spacer
   make the converted modes cancel, as two opposite bends do in §6 of the JINST paper? Robustness to ±1 mm of
   spacer error; trapped-mode resonances between the miters. Output: loss envelope vs spacer and n, and the
   shortest design that meets 1 dB.

## References

Kotik & Sivov (1965) Fig. 3 and the asymptotic $1 - 7.4/(ka)^{3/2}$ (L2): the rigorous overmoded result for the
optical-mirror H-plane miter up to $ka \approx 10\pi$ (67 GHz in WR-90), with the dips at $ka = n\pi$; Alimenti et al.
(1994) and Ma et al. (1997) for chamfer optimization in the single-mode band. Compare before concluding.

## Method

* Lines built from merged components only (`Miter`, `Straight`, `SmoothBend` with loss, `Line.sweep`), with
  evanescent modes in the bases between close miters (convergence in `cutoff_factor` shown).
* Miter wall loss: until WP3b lands, the straight-guide loss of the miter's centre-line length, stated as an
  estimate; then the impedance-wall FEM for the in-plane walls plus the broad-wall estimate, with the difference
  reported.
* Every number in the output comes from a script in `studies/` with its npz result (CONVENTIONS §9).

## Deliverables

* `studies/01_miter_vs_bend/` and `studies/02_chained_miters/`: scripts, npz, figures, a short `README.md` each with
  the answer to the question, the assumptions, and the open points.
* `docs/reports/WP8a.md`.

## Out of scope

E-plane miters (wave 3, after WP4b); filters; the full SPARC line.
