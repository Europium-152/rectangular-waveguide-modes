# WP2 — Ohmic loss and mode filters

Role: Implementer · Wave: 1 · Branch: `wave-1/WP2-loss-and-filters` · Depends on: WP0, T1 (merged); WP1
interface (`Component`) · Verified by: WP7 (surface-integral loss reference) and T1's check scripts ·
Time box: 6 days after T1 is merged (then status review)

## Goal

Implement the results of T1:

* finite wall conductivity in straight sections and smooth bends;
* Butterweck's two mode filters, as cascadable components.

## Read first

`CONVENTIONS.md`, `docs/theory/T1-wall-impedance.md` (the specification of every formula used here),
`tasks/WP1-cascade.md` (component interface), Butterweck, IEEE MTT-16 (1968).

## Specification

1. **`wrmodes/loss.py`**:
   * `surface_resistance(f, sigma)`;
   * per-mode attenuation constants for TE and TM;
   * the loss coupling matrices of T1 (per wall, so that walls can differ).
2. **Straight sections with loss.**
   * A `Straight` in a `Guide` with `sigma` becomes lossy: the matrix exponential of $(D - \Lambda_{\text{full}})L$
     (sign and block structure exactly as written in T1) for the propagating modes, including the degenerate TE/TM
     coupling.
   * Evanescent modes get their attenuation added to their decay.
   * Lossless guides (`sigma=None`) must give exactly the 0.1 results.
3. **Smooth bends with loss.**
   * Use $D + \kappa K - \Lambda_{\text{full}}$ in the transfer-matrix solver.
   * Do it through a new optional argument of `solve_transfer_matrix`, e.g. `loss_matrix`, that defaults to the
     0.1 behaviour.
   * The error control and the adaptive slicing must remain valid. Test it.
4. **`LossyWallFilter(length, Zw, walls=('u-','u+'))`** with the exact method of T1 item 5 (impedance-walled guide
   modes plus junctions). The first-order formula is used only as a check in its regime of validity.
5. **`ResistiveSheetFilter(length, R_sheet, v0=0.0, n_sheets=1)`**: LS modes of the sheet-loaded guide (T1 item 6)
   and junctions to the empty guide. Symmetric placement, plus several equidistant sheets as in Butterweck's
   experiment (three sheets, 70 Ω).
6. **Wall materials.** Aluminium σ = 3.5×10⁷ S/m is the SPARC baseline; copper 5.8×10⁷ S/m. Keep σ an explicit
   argument, with no default material.

## Acceptance tests

| Check | Target |
| --- | --- |
| TE₁₀ attenuation in WR-90, aluminium, 18–90 GHz | textbook formula to 1e-6; 0.12 dB/m at 18 GHz, 0.23 dB/m at 90 GHz |
| Lossless limit | identical to the 0.1 results and to WP1 |
| Passivity of every lossy component | largest singular value ≤ 1 |
| Reciprocity | 1e-10 |
| Lossy straight vs WP7 surface-integral reference | 1e-6 relative on α for every mode |
| Lossy-wall filter, Butterweck example (X-band guide 22.86 × 10.16 mm at 75 GHz, $Z_w$ = 250 Ω, 15 cm) | 0.6 dB (H₁₀), 2.5 dB (H₂₀), 5.4 dB (H₃₀) within 10 % (measurements; 0.6 and 5.4 dB also equal his theory, 4 and 36 dB/m) |
| Lossy-wall filter H₁₀→H₃₀ conversion over 15 cm | between −36 and −24.5 dB, as in Butterweck |
| Resistive sheet, $b$ = 10 mm, 75 GHz | optimum ≈ 56 Ω, EH₁₁⁽²⁾ ≈ 1.5 dB/cm (theory values, 5 %) |
| Bend with loss: power balance (transmitted + reflected + dissipated = 1, dissipated from the wall integral) | 1e-6 |

## Deliverables

* `src/wrmodes/loss.py` and the filter components.
* Tests in `tests/test_loss.py`.
* `examples/lossy_line.py`: a 20 m aluminium WR-90 line with three bends.
* `docs/reports/WP2.md`.

## Out of scope

Roughness, temperature dependence, filters of other types (mode-selective couplers).
