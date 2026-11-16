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

Resolved by WF3 (`tasks/WF3-buried-reference-planes.md`, branch `wave-2/WF3-buried-planes`; report
`docs/reports/WF3.md`), pending review. Campaign 2 now compares the cascade at cf 1.5 and 3.

## Resolution (WF3, 16 Nov 2026)

**Cause confirmed.** The reading above is right: δ = a = 22.86 mm for the optical 45° miter ((cos 22.5° −
d)/sin 22.5° with d = 1/(2 cos 22.5°)), and the pre-WF3 cascade (`Line.gsm(..., buried_planes='keep')`) reproduces
the table above digit for digit, LinAlgErrors included.

**Fix.** The GSM definition is unchanged (CONVENTIONS §6): `fem2d.Miter` / `Coplanar2D` still return the GSM at
the reference planes, and now also attach `g.buried = BuriedPlanes(depth=(δ1, δ2), gsm=g_u)`, the GSM at the
last uniform cross-sections (the outline's port planes), computed in the same solve. `Line.gsm` takes each depth
from the run of straights and frame changes next to the port: the component enters the cascade as g_u with the
propagating modes shifted by δ at the component (a phase), and the evanescent modes moved into the straight, which
then decays as e^{−|β|(s − δ)}. Exact, because evanescent modes are uncoupled in every uniform element (lossless or
lossy straight, frame change). Where the straight is shorter than the depths (the mirrors overlap) the port keeps
its reference-plane GSM and `Line` warns (`BuriedPlaneWarning`); at the line's own ports the shift is applied last
and the result carries `buried` too. `Line.gsm` also checks passivity (largest singular value of the propagating
block ≤ 1 + 1e-8, `NonPassiveCascadeWarning`, an error in the test suite).

**Table after the fix** (TE1,0 dB, default mesh 6/λ; `benchmarks/fem2d/buried_planes.py`):

| s | f (GHz) | cf 1.0 | cf 1.2 | cf 1.5 | cf 2.0 | cf 3.0 | COMSOL (fine) | single FEM domain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2a | 30 | −3.5465 | −3.5605 | −3.5592 | −3.5587 | −3.5587 | −3.5588 | −3.5588 |
| 2a | 56 | −1.4750 | −1.4809 | −1.4794 | −1.4785 | −1.4784 | −1.4782 | −1.4784 |
| 3a | 30 | −2.6942 | −2.6942 | −2.6941 | −2.6934 | −2.6934 | −2.6934 | −2.6942 |
| 3a | 56 | −0.2361 | −0.2361 | −0.2361 | −0.2362 | −0.2362 | −0.2362 | −0.2361 |

No `LinAlgError`, every result passive (largest singular value 1 + at most 6e-14). Before the fix the passivity check fires
on all seven non-physical results of the original table (largest singular values 5.0–105) and the other five
evanescent cases raise `LinAlgError`.

**Against the single domain** (max |ΔP| over the TE1,0 column: every propagating output mode, transmitted and
reflected, as a fraction of the input power):

| s | f | mesh/λ | cf 1.0 | cf 1.2 | cf 1.5 | cf 2.0 | cf 3.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2a | 30 | 6 / 12 / 16 | 1.2e-3 (all) | 1.8e-4 / 1.7e-4 / 1.7e-4 | 5.0e-5 / 4.9e-5 / 4.9e-5 | 9.6e-6 / 7.8e-6 / 7.7e-6 | 8.4e-6 / 5.8e-6 / 5.7e-6 |
| 2a | 56 | 6 / 12 | 6.4e-3 / 6.4e-3 | 4.5e-4 / 4.5e-4 | 1.5e-4 / 1.5e-4 | 3.7e-5 / 3.2e-5 | 2.9e-5 / – |
| 3a | 30 | 6 / 12 / 16 | 1.8e-5 / 7.5e-7 / 2.8e-7 | 1.4e-5 / 2.0e-6 / 2.3e-7 | 4.3e-5 / 8.7e-7 / 2.3e-7 | 2.3e-4 / 1.1e-5 / 3.5e-6 | 2.4e-4 / 1.2e-5 / 4.6e-6 |
| 3a | 56 | 6 / 12 | 1.1e-5 / 2.1e-6 | 1.4e-5 / 2.4e-7 | 1.6e-5 / 2.1e-7 | 3.4e-5 / 9.4e-7 | 3.5e-5 / – |

* s = 3a: within 1e-4 everywhere once both FEM models are converged; the 2.4e-4 at the default mesh (cf 2–3, 30
  GHz) is the single domain's own mesh error (its TE1,0 −2.6942 dB at 6/λ becomes −2.6934 dB at 12/λ, COMSOL
  fine −2.6934), not the cascade's.
* s = 2a: the two uniform planes coincide (the mirrors touch), so the cascade is a zero-length junction and needs
  evanescent modes. Within 1e-4 from cf 1.5 at 30 GHz and from cf 2 at 56 GHz; **not met at cf 1.0 and 1.2** (and
  cf 1.5 at 56 GHz), at a level that does not change with the mesh: truncation of the junction, which converges
  with cf (30 GHz: 1.2e-3, 1.7e-4, 4.9e-5, 7.7e-6, 5.7e-6; 56 GHz: 6.4e-3, 4.5e-4, 1.5e-4, 3.2e-5). The pre-WF3 cascade has the same values at cf 1–1.5 where it
  ran. This is physics of the truncated cascade, not conditioning; the tolerance is left as it is.
* COMSOL E5 (`compare_campaign_2.py E5`): the cascade at cf 1.5 within 0.0012 dB and at cf 3 within 0.0002 dB of
  COMSOL's fine mesh (allowed 0.020–0.022 dB): pass.
