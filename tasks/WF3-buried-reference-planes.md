# WF3 — Cascades through reference planes inside a discontinuity

Role: Implementer · Wave: 2 (last item before Gate B) · Branch: `wave-2/WF3-buried-planes` · Depends on: wave 2 and
WF2 (merged) · Checked against: the single-domain FEM (`fem2d.chained_miter_outline`) and COMSOL campaign 2 case E5 ·
Time box: 3 days (then status review)

## Problem

`docs/discrepancies/2026-11-15-gateb-45deg-miter-cascade-evanescent.md`. A `Line` [Miter(45), Straight(s),
Miter(45)] in WR-90 returns non-physical results when the bases carry evanescent modes: +26 dB of gain at
`cutoff_factor = 3` (s = 2a, 30 GHz), errors of tens of dB or a `LinAlgError` at 1.2–3, and no warning. With
propagating modes only it agrees with COMSOL and the single domain to 0.012 dB.

The cause: for a 45° optical mirror the CONVENTIONS §6 reference plane (through the inner corner, 4.73 mm from the
apex) lies δ = 22.86 mm inside the mirror, which reaches 27.59 mm back along the arm. The GSM at that plane contains
evanescent entries scaled by e^{|β|δ} (CONVENTIONS §6, "Evanescent entries at reference planes inside a
discontinuity"; `fem2d.PortPlane.deembed`; `Miter` docstring), about 1e13 at cf 3, and the star product with the
next component loses everything to rounding.

## Specification

1. **Keep the GSM definition.** A component's GSM at its CONVENTIONS reference planes is unchanged: it is the
   documented quantity, and standalone users keep it.
2. **Cascade through the buried plane well-conditioned.** In a `Line`, when a component's port plane lies a depth
   δ > 0 inside its discontinuity and the neighbour on that side is a `Straight` of length s ≥ δ (or another
   uniform section), the line combines them so that the plane shift is applied as e^{−|β|(s − δ)} instead of
   e^{+|β|δ} followed by e^{−|β|s}. For example: the component exposes its GSM at the uniform-arm planes plus the
   depth δ per port (in `GSM.info` or a method), and `Line` shortens the straight. When s < δ, or the neighbour is not
   uniform, keep the present behaviour and warn. Design the protocol so that `MiterMM`, `Coplanar2D`, `Step` and
   later components can take part. Document it in the `Line` and `Miter` docstrings and propose the CONVENTIONS text.
3. **Passivity check.** `Line.gsm` checks lossless results: largest singular value ≤ 1 + 1e-8, otherwise a
   `NonPassiveCascadeWarning` (or an error, your choice; justify it) that names the junction. Lossy lines: largest
   singular value ≤ 1 + 1e-8 as well. The cost must be small next to the cascade; report it.
4. Do not change any GSM of a line without buried planes (bitwise, on the WF2 frozen cases
   `tests/data/smoothbend_propagating.npz` and a few miter/step lines of your choice frozen the same way).

## Acceptance tests

| Check | Target |
| --- | --- |
| The discrepancy table (s = 2a, 3a; 30, 56 GHz; cf 1.0, 1.2, 1.5, 2, 3) | every entry within 1e-4 of the input power (TE₁₀ and converted modes) of the single-domain FEM; no `LinAlgError` |
| COMSOL E5 (`benchmarks/comsol/campaign_2/compare_campaign_2.py E5`, cascade at cf 1.5 and 3) | within the campaign criterion |
| 90° plain miters with a short straight (Gate A geometry, cf 1–3) | unchanged where they were already well conditioned; reported |
| Passivity check | fires on the old behaviour (reproduce it with the shortening disabled), silent on every test of the suite |
| Lines without buried planes | bitwise unchanged |

## Deliverables

Code and tests; `docs/reports/WF3.md`; Resolution section of the discrepancy file; update the E5 cascade in
`compare_campaign_2.py` to cf 3 and regenerate `benchmarks/comsol/campaign_2/RESULTS.md`; the Gate B section of
`benchmarks/validation_v2/REPORT.md` updated (one paragraph). Do not modify CONVENTIONS.md (propose the text).
