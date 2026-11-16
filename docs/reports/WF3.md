# WF3 — Cascades through reference planes inside a discontinuity; passivity check: report

Branch `wave-2/WF3-buried-planes`, from the card commit d795a12 (main cce2b6f + card). Card
`tasks/WF3-buried-reference-planes.md`; discrepancy `docs/discrepancies/2026-11-15-gateb-45deg-miter-cascade-evanescent.md`
(Resolution section added). Times: 2-core container, one BLAS thread, other jobs running.

## What was done

| Item | Files |
| --- | --- |
| `BuriedPlanes` (depth per port + GSM at the uniform-arm planes), `GSM.buried` (class default None), `shift_planes` | `src/wrmodes/gsm.py`, exported by `wrmodes` |
| `fem2d.Coplanar2D` / `Miter`: the GSM at the outline's port planes from the same solve, attached when `PortPlane.deembed` > 0 (mirror image for negative angles); docstrings | `src/wrmodes/fem2d/components.py` |
| `Line.gsm`: absorption of the depths by the neighbouring uniform run, `BuriedPlaneWarning`, deferral to the line's ports (`buried` on the line's GSM), near-field warning on the uniform guide left, passivity check (`NonPassiveCascadeWarning`, `PASSIVITY_TOL`), `buried_planes='keep'` | `src/wrmodes/line.py` (module docstring: the protocol) |
| The suite treats `NonPassiveCascadeWarning` as an error | `pyproject.toml` (`filterwarnings`) |
| Tests | `tests/test_line.py` (WF3 section, 11 tests, no FEM), `tests/test_fem2d.py` (WF3 section, 6 tests + 1 slow) |
| Frozen bitwise data (generated at d795a12 from a separate checkout) | `tests/data/make_lines_unburied.py`, `lines_unburied.npz` (8 lines), `lines_unburied_fem.npz` (4 FEM lines) |
| Benchmark | `benchmarks/fem2d/buried_planes.py` → `benchmarks/fem2d/data/buried_planes_d{6,12,16}_*.npz` |
| E5 at cf 1.5 and 3; RESULTS regenerated | `benchmarks/comsol/campaign_2/compare_campaign_2.py`, `RESULTS.md` |
| Records | discrepancy Resolution, `benchmarks/validation_v2/REPORT.md` (Gate B), `benchmarks/registry.json` |

## Design: the protocol between components and `Line`

1. **The GSM definition is unchanged.** A component returns its GSM at its CONVENTIONS §6 reference planes. When a
   plane lies a depth δ_p > 0 inside the discontinuity, the component attaches
   `g.buried = BuriedPlanes(depth=(δ1, δ2), gsm=g_u)`: g_u is the GSM in the same port bases at the planes δ_p
   further out along each arm, where the arm is still uniform, so that g = `shift_planes(g_u, δ1, δ2)` exactly
   (S_ij → e^{iβ_iδ_i} S_ij e^{iβ_jδ_j}; a negative distance is a lossless straight). Standalone users see nothing
   new. Operations on GSMs (`@`, `reversed`) do not carry `buried`.
2. **Who takes part.** `Coplanar2D` (any outline, e.g. `chained_miter_outline`) and `Miter` attach it whenever a
   port plane of the outline is de-embedded into the core (`PortPlane.deembed` > 0): the optical mirror below 90°
   (δ = a at 45°, 0.577a at 60°) and deep chamfers. The FEM already solves at the outline's port planes and shifts
   from there, so g_u costs one more matrix product. `MiterMM`, `Step`, bends, tapers and twists have their planes
   on uniform cross-sections and leave `buried` None; a later component (e.g. a mode-matching miter below 90°)
   takes part by attaching `BuriedPlanes` to its GSM, nothing else. `Line` reads only `g.buried`, so it works
   with the memo and the `cache` of `Line.gsm`, and a `Line`'s own GSM carries it too (below).
3. **What `Line.gsm` does** (`_absorb`). For every buried port it walks the run of uniform elements (`Straight`,
   `FrameChange`) on that side. If the run's remaining length is ≥ δ (relative tolerance 1e-9, for the exact
   s = 2δ of two touching mirrors), the component enters the cascade as g_u with:
   * the propagating modes shifted by δ at the component itself (a phase: exact and harmless), and
   * the evanescent modes moved into the straights of the run, nearest first (their evanescent rows and columns are
     multiplied by e^{|β|c}, c the length consumed, so a straight s decays as e^{−|β|(s−δ)} instead of e^{+|β|δ}
     followed by e^{−|β|s}).

   This is exact, not an approximation: in every uniform element an evanescent mode is only multiplied by its own
   factor (lossless straight; lossy straight, where `coupling_mask` keeps evanescent modes uncoupled; a frame change
   maps each mode to one with the same β), so the diagonal evanescent shift commutes with the whole run. Moving the
   propagating shift as well would not commute with the coupled propagating block of a lossy straight; keeping it
   at the component keeps the loss of the whole length s, i.e. the result is the defined cascade at the reference
   planes. Two buried ports share the straight between them (s ≥ δ2 + δ1').
4. **Overlap.** If the run is shorter than δ and ends at another non-uniform component, the discontinuities
   overlap: the cascade does not describe the physical geometry (e.g. two 45° optical miters with s < 2a, whose
   mirrors cross). That port keeps its reference-plane GSM (the pre-WF3 behaviour), and `Line` issues one
   `BuriedPlaneWarning` per line if the basis there has evanescent modes (with propagating modes only the shift is
   a phase and nothing is ill-conditioned, so no warning). The message names both components, the depth, the
   uniform length available and the largest factor, and points to `chained_miter_outline`.
5. **Line ports.** If the run reaches the line's own port (a buried component first or last), the line's port lies
   (δ − available) inside the discontinuity: the cascade is done with the evanescent shift deferred and applied to
   the result last, which is then the GSM at the line's reference planes, as for the component alone, and carries
   `buried` (the line's GSM at its uniform planes) like a component. A line [Miter(45), Straight(2a), Miter(45)]
   therefore has `buried.depth = (a, a)`.
6. **Near-field warning (WF2)** counts the uniform guide actually left, s − δ2 − δ1'.
7. `Line.gsm(..., buried_planes='keep')` is the pre-WF3 cascade at the reference planes (used to reproduce the
   discrepancy and to show that the passivity check fires).

## Passivity check

`Line.gsm` computes the 2-norm (largest singular value) of the propagating block of its result and issues a
`NonPassiveCascadeWarning` above 1 + `PASSIVITY_TOL` = 1 + 1e-8, for lossless and lossy lines. The message gives
the singular value and **where**: the first element whose own GSM is not passive, or the junction (with the last
non-uniform element before it) at which the partial cascade first exceeds the bound, found by replaying the cascade
only when the check fails.

* **Warning, not error** (the card leaves the choice): the GSM stays available for inspection, a `sweep` is not
  interrupted at one bad frequency, and the check is a heuristic that a legitimate but slightly non-unitary
  component result could trip (none does in the suite). `warnings.simplefilter('error', NonPassiveCascadeWarning)`
  makes it an error; **the test suite does so** (`pyproject.toml`), so it cannot fire silently in any test, and the
  tests that expect it use `pytest.warns`.
* **Behaviour**: on the pre-WF3 cascade (`'keep'`) it fires on every non-physical entry of the discrepancy table
  (7 of 7 results that did not raise `LinAlgError`; largest singular values 5.0, 21, 27, 34, 44, 85, 105), naming
  "the junction of element 0 (H-plane miter 45 deg) and element 2 (H-plane miter 45 deg)" (test). With the fix it
  is silent on all 20 entries (largest singular values 1 + at most 6e-14). It is silent in the whole suite (an error there).
* **Cost**: one SVD of a 2Np × 2Np matrix plus two mode masks. Measured (`SmoothBend`–`Straight`–`MiterMM`–
  `Straight`–`SmoothBend`, WR-90): 0.21 / 0.37 / 1.3 / 2.9 ms for N = 12 / 97 / 111 / 308 modes (Np = 12 / 12 /
  51 / 51), against 0.9 / 31 / 41 / 518 ms for the four star products alone (23 % / 1.2 % / 3.1 % / 0.6 %) and
  110 ms / 260 ms for the whole `Line.gsm` at 30 GHz, cf 1 / 3 (0.2 % / 0.1 %). The relative cost is only
  noticeable for tiny propagating-only cascades, where everything takes well under a millisecond.
  `tests/test_line.py::test_passivity_check_cost_is_small` asserts < 20 % of a small line's `gsm` (ratio, so
  CI-speed independent).

## Acceptance

### The discrepancy table

TE1,0 transmission (dB) with the fix, default mesh (6/λ), and the pre-WF3 cascade (`'keep'`) for comparison:

| s | f (GHz) | cf 1.0 | cf 1.2 | cf 1.5 | cf 2.0 | cf 3.0 | before WF3 (cf 1.2 / 1.5 / 2 / 3) | COMSOL fine | single domain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2a | 30 | −3.5465 | −3.5605 | −3.5592 | −3.5587 | −3.5587 | −3.5605 / −3.5592 / error / **+26.36** | −3.5588 | −3.5588 |
| 2a | 56 | −1.4750 | −1.4809 | −1.4794 | −1.4785 | −1.4784 | error / −30.53 / −10.03 / −27.90 | −1.4782 | −1.4784 |
| 3a | 30 | −2.6942 | −2.6942 | −2.6941 | −2.6934 | −2.6934 | −2.6942 / −2.6941 / error / error | −2.6934 | −2.6942 |
| 3a | 56 | −0.2361 | −0.2361 | −0.2361 | −0.2362 | −0.2362 | error / −35.20 / −27.61 / −52.20 | −0.2362 | −0.2361 |

No `LinAlgError`, every result passive. Max |ΔP| over the TE1,0 column (every propagating output mode, transmitted
and reflected) against the single FEM domain (`chained_miter_outline`), at mesh densities 6 / 12 / 16 per λ:

| s | f | cf 1.0 | cf 1.2 | cf 1.5 | cf 2.0 | cf 3.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 2a | 30 | 1.2e-3 / 1.2e-3 / 1.2e-3 | 1.8e-4 / 1.7e-4 / 1.7e-4 | **5.0e-5 / 4.9e-5 / 4.9e-5** | **9.6e-6 / 7.8e-6 / 7.7e-6** | **8.4e-6 / 5.8e-6 / 5.7e-6** |
| 2a | 56 | 6.4e-3 / 6.4e-3 | 4.5e-4 / 4.5e-4 | 1.5e-4 / 1.5e-4 | **3.7e-5 / 3.2e-5** | **2.9e-5** / – |
| 3a | 30 | **1.8e-5 / 7.5e-7 / 2.8e-7** | **1.4e-5 / 2.0e-6 / 2.3e-7** | **4.3e-5 / 8.7e-7 / 2.3e-7** | 2.3e-4 / **1.1e-5 / 3.5e-6** | 2.4e-4 / **1.2e-5 / 4.6e-6** |
| 3a | 56 | **1.1e-5 / 2.1e-6** | **1.4e-5 / 2.4e-7** | **1.6e-5 / 2.1e-7** | **3.4e-5 / 9.4e-7** | **3.5e-5** / – |

Bold: within 1e-4. (cf 3 at 56 GHz was not rerun at 12/λ: ~2000 s and ~3 GB per point.)

* **s = 3a**: met everywhere with converged meshes. The 2.3–2.4e-4 at the default mesh (cf 2–3, 30 GHz) is the
  single domain's mesh error, not the cascade's: the domain's TE1,0 moves from −2.6942 dB (6/λ) to −2.6934 dB
  (12/λ), which is COMSOL's fine value, while the cascade at cf ≥ 2 gives −2.6934 already at 6/λ.
* **s = 2a: not met at cf 1.0 and 1.2, and at cf 1.5 at 56 GHz (1.5e-4)**. Here the two uniform-arm
  planes coincide (the mirrors touch, s − 2δ = 0), so the cascade is a zero-length junction of two miters and
  needs evanescent modes; the remainder does not move with the mesh (1.2e-3, 1.7e-4, 4.9e-5 at 6, 12 and 16/λ) and
  falls with cf (30 GHz: 1.2e-3 → 1.7e-4 → 4.9e-5 → 7.7e-6 → 5.7e-6; 56 GHz: 6.4e-3 → 4.5e-4 → 1.5e-4 → 3.2e-5, the same at 6 and 12/λ). It is the truncation of the cascade, identical in the
  pre-WF3 code where that ran (cf 1–1.5), not conditioning. I have not changed the tolerance; the row is met for
  every cf at which a zero-length junction can be expected to converge (≥ 1.5 at 30 GHz, ≥ 2 at 56 GHz).
* The criterion over the whole propagating block (all input modes) is looser near cutoffs (1.3e-4 at s = 2a,
  30 GHz, cf 3, 16/λ; 5.8e-4 at 3a, 56 GHz, cf 3, 6/λ); the card asks for the TE1,0 column.

### COMSOL E5 (`compare_campaign_2.py E5`)

| s | f (GHz) | COMSOL fine | single domain | cascade cf 1.5 | Δ | cascade cf 3 | Δ | allowed | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2a | 30 | −3.5588 | −3.5588 | −3.5592 | −0.0004 | −3.5587 | +0.0001 | 0.0202 | yes |
| 2a | 56 | −1.4782 | −1.4784 | −1.4794 | −0.0012 | −1.4784 | −0.0002 | 0.0224 | yes |
| 3a | 30 | −2.6934 | −2.6942 | −2.6941 | −0.0007 | −2.6934 | −0.0000 | 0.0214 | yes |
| 3a | 56 | −0.2362 | −0.2361 | −0.2361 | +0.0000 | −0.2362 | +0.0000 | 0.0202 | yes |

|S11| of the cascade at cf 3 equals COMSOL's and the domain's to the 4th decimal (0.0586, 0.0332, 0.2062, 0.0228–9);
at cf 1 it was 0.0564 and 0.0280 at s = 2a. `RESULTS.md` regenerated: 126 comparisons (4 more: the second cascade column), 0 outside the criterion
(E1–E4 unchanged).

### 90° plain miters with a short straight (Gate A geometry), and 60° optical miters

* `[Miter(90), Straight(s), Miter(90)]`, WR-90, 30 GHz, s = a/4, a/2, a, cf 1, 1.5, 2, 3 (12 GSMs): **bitwise
  identical** to the pre-WF3 code (separate checkout of d795a12). The 90° optical mirror has its planes on uniform
  cross-sections (δ = 0), so nothing changes; where these were well conditioned they stay so, and they were (WP3b:
  1e-6 agreement with the single domain at 32/λ). Two of these lines are frozen in `lines_unburied_fem.npz`.
* Bonus, the C1b geometry of Gate A, two 60° optical miters (δ = 0.577a), 30 GHz, max |ΔP| of the TE1,0 column vs
  the single domain: s = 1.25a: 9.3e-3 / 1.9e-4 / 6.6e-6 / 1.5e-5 at cf 1 / 1.5 / 2 / 3; s = 1.5a: 7.2e-4 /
  3.2e-7 / 3.0e-6 / 2.6e-6; s = 2a: 1.2e-5 / 4.4e-6 / 1.5e-5 / 1.7e-5. Pre-WF3: identical at cf 1–2, `LinAlgError`
  at cf 3 for every s.

### Lines without buried planes: bitwise unchanged

* `lines_unburied.npz` (8 GSMs: `MiterMM` U-turn at cf 1, 2; E- then H-plane `MiterMM` with frame changes at
  cf 1, 1.5; two `Step` pairs, one E-plane with an offset; a lossy line with a bend) and `lines_unburied_fem.npz`
  (4: FEM 90° U-turn at cf 1, 1.5; 45° miters with chamfer 0.95, whose mirror stays outside the arms; FEM E-plane
  miter with `MiterMM`), generated at d795a12 from a separate checkout: **`np.array_equal` for all 12** with the
  WF3 code; the tests accept 1e-11 absolute across platforms (as `test_components`), and the FEM test skips with a
  different gmsh version (meshes).
* `tests/data/smoothbend_propagating.npz` (WF2's 20 cases): still bitwise (the existing test).
* The restructured `Line.gsm` (component GSMs first, then absorption, warnings, cascade) does the same arithmetic in
  the same order when no GSM is buried; the FEM's reference-plane GSM is computed by the same expression as before.

### Passivity check

| Check | Result |
| --- | --- |
| Fires on the old behaviour | 7 of 7 non-physical pre-WF3 results of the table (the rest raise `LinAlgError`); test `test_passivity_check_fires_on_the_cascade_before_wf3` (s = 2a, 30 GHz, cf 3: +26 dB, junction named) |
| Names the junction / the element | `test_passivity_check_names_the_junction_or_the_element` (no FEM) |
| Silent on the suite | an error filter in `pyproject.toml`; full fast suite and FEM tests pass (below) |

## Tests

New, `tests/test_line.py` (no FEM; a test component `_Buried` moves the planes of `MiterMM` or `SmoothBend` a depth
into it): `shift_planes` = straight of negative length; [X, Straight(s), X] = [X_u, Straight(s − 2d), X_u] to
1e-13 at cf 1, 2, 3 and `LinAlgError` with `'keep'`; absorption across frame changes and several straights; lossy
straight (loss of the whole length kept, equal to `'keep'` to 1e-10 relative, passive, reciprocal); overlap warning
(once, not at cf 1); partial depth deferred to the line port; near-field warning on the uniform guide left;
passivity warning naming the junction and the element; frozen bitwise lines; cost of the check.
`tests/test_fem2d.py`: `buried` of the 45° miter (depth (a, a), g = shift_planes(g_u, a, a), mirror image for
−45°, none for 90° and chamfer 0.95); cascade vs single domain (s = 2a cf 2, s = 3a cf 1.5; slow: cf 3); the
passivity check on the pre-WF3 cascade; H/E line with a frame change in the run and lossy miters in a lossy line;
FEM frozen lines.

Runs: SUITE_RESULTS

## Open issues

1. **s = 2a at cf ≤ 1.2** (touching mirrors) stays a truncated zero-length junction (1.2e-3 / 1.7e-4 of the power);
   users who need it at cf 1 should model the pair as one domain (`chained_miter_outline`). The
   `EvanescentJunctionWarning` fires there at cf 1 (gap 0), which is the right advice.
2. **Overlapping discontinuities** (s < δ2 + δ1') are refused only by a warning, because the GSMs exist and the
   result is the pre-WF3 one; the cascade is not the physical geometry (a ValueError would be the stricter choice;
   kept as a warning to follow the card).
3. **Partial depth at an inner junction** is not taken (all-or-nothing per port, as the card's "keep the present
   behaviour"); taking the available part would only reduce the growth of a cascade that is geometrically wrong.
4. The FEM's own evanescent entries at the uniform planes are as accurate as before (the evanescent gain of the
   arm, `evanescent_gain`); zero-length junctions of FEM components (s = 2δ) still need the WP3b open issue 3 mesh
   care (here 6/λ suffices for 1e-4).
5. `GSM.reversed()` and star products drop `buried`; a user cascading line GSMs by hand gets the reference-plane
   GSMs and should cascade `g.buried.gsm` with shortened straights (documented in `BuriedPlanes`).

## Decisions for the coordinator / Daniel

1. Warning (not error) for non-passive cascades, an error in the suite (justification above).
2. Accept the s = 2a, cf 1.0/1.2 entries as truncation (they cannot be met by any cascade with so few modes at a
   zero-length junction), or restate the row as "cf ≥ 1.5".
3. The CONVENTIONS text below.

## Proposed CONVENTIONS.md text (not changed here)

§6, replace the last sentence of the bullet "Evanescent entries at reference planes inside a discontinuity"
("Cascades with such components keep enough straight length, or enough evanescent modes, ...") by:

> Such a component also returns its GSM at the uniform-arm planes and the depths (`GSM.buried`, a
> `wrmodes.gsm.BuriedPlanes`; g = `shift_planes(g_u, δ1, δ2)`). `Line` takes each depth from the uniform guide next
> to the port (straights and frame changes, in which evanescent modes are uncoupled), so that no factor
> $e^{|\beta|\delta}$ enters a star product: the straight must be at least as long as the depths it serves
> ($s \ge \delta_2 + \delta_1'$ between two such components, i.e. the discontinuities must not overlap). Where they
> overlap, `Line` keeps the GSM at the reference plane and warns (`BuriedPlaneWarning`); a line whose own port lies
> inside a discontinuity returns its GSM at that plane, like the component (WF3).

§6, "Properties every component must have", add after the bullets:

> `Line.gsm` checks every cascade: largest singular value of the propagating block ≤ 1 + 1e-8, otherwise a
> `NonPassiveCascadeWarning` naming the element or junction (the test suite treats it as an error).

§8, zero-length cascades, add:

> Components whose reference planes lie inside their discontinuities are joined directly when the uniform guide
> left between them, $s - \delta_2 - \delta_1'$, is shorter than three decay lengths (`Line`'s near-field warning
> counts this length).

§10, add:

> A component whose reference plane lies inside its discontinuity attaches `BuriedPlanes` to its GSM (WF3).
