# WP4 — Step junctions by mode matching

Role: Implementer · Wave: 1 · Branch: `wave-1/WP4-steps` · Depends on: WP0 (merged); WP1 `Component` interface
Verified by: WP7 (Marcuvitz formulas, D6 of L1); WP3 2D solver in wave 2 · Time box: 5 days (then status review)

## Goal

The GSM of a junction between two rectangular guides, one contained in the other, by mode matching with
closed-form overlap integrals. Steps are needed by the staircase reference for tapers (wave 2), by the
strangled bend with abrupt transitions, and by the sheet and lossy-wall filters (WP2). In wave 2 the same
library will host Alessandri's segmentation for 90° miters.

## Read first

* `CONVENTIONS.md` (§3–§6).
* `src/wrmodes/modes.py`, `src/wrmodes/gsm.py`.
* Marcuvitz (1951), pp. 296–335: steps and their equivalent circuits.
* Alessandri, Mongiardo & Sorrentino, IEEE MTT 40 (1992): generalized admittance/scattering formulation.

## Specification

1. **`Step(a1, b1, a2, b2, du=0.0, dv=0.0)`**: guide 1 ($a_1\times b_1$) to guide 2 ($a_2\times b_2$), with the centre
   of the smaller offset by $(d_u, d_v)$ in the larger one's frame. One cross-section must contain the other. Both
   reductions (large to small and small to large) are supported, and either may be offset.
2. **Overlaps.** $X_{pq} = \iint_{\text{small}} e^{\text{small}}_p\cdot e^{\text{large}}_q\,ds$ in closed form for all
   TE/TM pairs, using `transverse_field` conventions. Check them against quadrature.
3. **Matching.**
   * Tangential $E$ is continuous on the aperture and zero on the metal; tangential $H$ is continuous on the
     aperture.
   * Build the GSM in the normalized amplitudes of `CONVENTIONS.md` §5, with the standard formulation that is
     symmetric by construction (e.g. $S$ from $X$ and the admittance matrices).
   * Each side keeps its own basis.
4. **Truncation.**
   * Use the same cutoff ratio on both sides (relative convergence: numbers of modes roughly in proportion to the
     dimensions).
   * Provide `convergence_study()`. Report how |S| of the propagating modes converges with the number of
     evanescent modes.
5. **The exp(θK) study** (for the article). For a WR-90 90° H-plane and E-plane "zero-length bend", compute
   $T = e^{\theta K}$ in the wrmodes family basis, with $K$ the curvature matrix. Report its unitarity and its
   conversion vs frequency. Store it as data for comparison with the miter of WP3 at Gate A. This documents why
   coupled-mode theory cannot model a miter.

## Acceptance tests

| Check | Target |
| --- | --- |
| Overlaps vs quadrature | 1e-12 relative |
| Identity step ($a_1 = a_2$, $b_1 = b_2$) | S = zero-length straight to 1e-12 |
| Reciprocity | 1e-10 |
| Unitarity (propagating modes) | 1e-10 at every truncation |
| Step followed by its reverse at zero distance, propagating modes | identity to 1e-10 when the bases match |
| Single-mode E-plane and H-plane steps vs Marcuvitz equivalent circuits (D6 of L1), in their stated ranges | within Marcuvitz's stated accuracy |
| Convergence of |S11| for TE₁₀ in the WR-90 → 22.86 × 2 mm step | change < 1e-4 between the two finest truncations |

## Deliverables

* `src/wrmodes/modematch/`, `tests/test_steps.py`.
* `examples/strangling_step.py`: WR-90 to 22.86 × 1.6 mm and back.
* `docs/reports/WP4.md`, including the exp(θK) study data in `benchmarks/data/`.

## Out of scope (wave 2)

Alessandri segmentation of miters, tapers as staircases, junctions with impedance-walled or sheet-loaded guides
(WP2 builds those on this library).
