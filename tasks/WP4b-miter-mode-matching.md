# WP4b — Mode matching for 90° miters (Alessandri segmentation); symmetric 2D steps

Role: Implementer · Wave: 2 · Branch: `wave-2/WP4b-miter-mm` · Depends on: WP4 (merged) · Verified by: WP3/WP3b
(FEM) through WP7b · Time box: 7 days (then status review)

## Goal

A mode-matching backend for 90° plain and chamfered miters, H- and E-plane, that needs no FEM install and is fast
at 90 GHz. It is the independent method that validates the FEM miters (and the reverse), and the fallback of the
plan for users without `wrmodes[fem]`. Also: the parity reduction of centred 2D steps, which WP5's staircase
comparisons need.

## Read first

* `CONVENTIONS.md` §6 (miter reference planes through the inner corner; plain miter = optical mirror, at 90°
  $d/a = 1/\sqrt2$), §7, §8.
* Alessandri, Mongiardo & Sorrentino (1994): the segmentation of the mitered corner and the mode-matching
  formulation; Figs. 4–5 (`benchmarks/literature/alessandri1994_fig4/5.csv`).
* `docs/reports/WP4.md` (open issues 3–6) and `src/wrmodes/modematch/`.
* Alimenti, Mongiardo & Sorrentino (1994): the same group's segmentation of the H-plane mitered bend (Fig. 1)
  and L2's digitized Figs. 2–6; Kotik & Sivov (1965): a rigorous image-system formulation for the optical
  mirror (L2's summary), usable as a third method for that case.
* Do **not** read `src/wrmodes/fem2d/` or the WP3 reports' numbers before your own results are committed.

## Specification

1. `wrmodes.modematch.MiterMM(chamfer=None, plane='H', guide=None, n_modes=None)`: 90° only; `chamfer` as in
   `fem2d.Miter` (Alessandri's $d$ from the inner corner, normalized to the in-plane side; `None` the optical
   mirror). Same port bases, frame tags, `arm_length` and reference planes as `fem2d.Miter`, so the two are
   interchangeable in a `Line`. Every family of the basis (LSM and LSE per family, rotated to TE/TM as in
   CONVENTIONS §7).
2. Method: the segmentation of Alessandri et al. (or an equivalent rigorous mode-matching decomposition of the
   corner; document the choice and why). Show convergence in the number of modes of every region and choose
   defaults from it.
3. `Step` parity reduction: for centred steps ($d_u = d_v = 0$) solve the parity classes of $m$ and $n$
   separately, giving the same GSM as now to 1e-12 and reporting the speed-up for a WR-90 ↔ 22.86 × 1.6 mm step
   at 90 GHz and for a 2D step (both sides change).
4. Performance: WR-90 H-plane plain miter at 90 GHz, full default basis, under 1 s per frequency.

## Acceptance tests

| Check | Target |
| --- | --- |
| Alessandri Figs. 4–5 (E plane) | within the digitizing uncertainty (the plan's 0.5 dB); state the b/a used |
| Alimenti Figs. 2–6 (H plane), Kotik Fig. 3 (H plane, optical mirror, overmoded) | within the digitizing uncertainty |
| Reciprocity; unitarity; $S_{11} = S_{22}$ (symmetric miter) | CONVENTIONS §6 |
| Convergence in the mode counts | reported; defaults reach 1e-4 in every propagating entry |
| Agreement with the FEM (run by WP7b after both are committed) | 0.01 dB on TE₁₀ transmission and 1e-4 of the input power per converted mode, H and E plane, d/a = 0.707–1.0, WR-90 at 10–90 GHz |
| Parity-reduced `Step` | equal to the full solve to 1e-12 |

## Deliverables

* `src/wrmodes/modematch/miter.py` (exported), step parity changes, `tests/test_modematch_miter.py`.
* `docs/reports/WP4b.md` with the method, convergence and timing.

## Out of scope

Angles other than 90° (FEM covers them); impedance walls; double miters in one region.
