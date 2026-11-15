# Cascade of 45° optical miters with evanescent modes is silently ill-conditioned

Found by the coordinator while post-processing COMSOL campaign 2, case E5 (15 Nov 2026). Owner: WP3/WP3b (`fem2d.Miter`)
and `Line`.

## Observation

`Line(Guide(22.86e-3, 10.16e-3), [Miter(45, plane='H'), Straight(s), Miter(45, plane='H')])`, TE1,0 transmission (dB):

| s | f (GHz) | cf 1.0 | cf 1.2 | cf 1.5 | cf 2.0 | cf 3.0 | COMSOL (fine) | single FEM domain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2a | 30 | −3.5465 | −3.5605 | −3.5592 | error | **+26.36** | −3.5588 | −3.5588 |
| 2a | 56 | −1.4750 | error | −30.53 | −10.03 | −27.90 | −1.4782 | −1.4784 |
| 3a | 30 | −2.6942 | −2.6942 | −2.6941 | error | error | −2.6934 | −2.6942 |
| 3a | 56 | −0.2361 | error | −35.20 | −27.61 | −52.20 | −0.2362 | −0.2361 |

"error" is `LinAlgError` from `GSM.star` ("exact resonance of a trapped mode"). With propagating modes only (cf 1) the
cascade agrees with COMSOL and with the single-domain FEM (`chained_miter_outline`) to 0.012 dB (s = 2a) and 0.001 dB
(s = 3a). With evanescent modes it returns non-physical results (a lossless line with +26 dB gain) **without a
warning**.

## Cause (coordinator's reading)

For a 45° deflection the optical mirror makes 22.5° with the axis and reaches (a/2)/tan(22.5°) = 27.59 mm back along
each arm from the apex, while the CONVENTIONS §6 reference plane (through the inner corner) is 4.73 mm from the apex:
the plane lies δ = 22.86 mm inside the discontinuity. The evanescent entries then carry e^{|β|δ} per port (CONVENTIONS
§6, "Evanescent entries at reference planes inside a discontinuity"); at cf 3 and 30 GHz the highest TE_m0 modes have
|β|δ ≈ 30, so entries of order 1e13 enter the star product and the 22.86–45.72 mm straight between the miters cannot
compensate in floating point.

## Proposed fix (for a fix card)

1. `Line` merges a component whose reference planes lie inside its discontinuity (depth δ reported in the GSM `info`)
   with an adjacent `Straight` of length ≥ δ, so that the plane shift is applied as a decaying factor e^{−|β|(s−δ)}
   instead of e^{+|β|δ}; and/or the miter returns its GSM at the uniform-arm planes with the shift deferred.
2. `Line` checks passivity of lossless results (largest singular value ≤ 1 + 1e-8) and raises or warns otherwise.
3. Test: the table above at cf 1.5–3 must agree with the single-domain FEM to 1e-4.

## Status

Open. Campaign 2 compares the cascade at cf 1 (propagating modes only), which is valid here.
