# WP5 — Tapers and the strangled E-plane bend: report

Branch `wave-2/WP5-tapers`, from `main` at 84d99a0. Card `tasks/WP5-tapers-strangled-bend.md`.

## What was done

| Item | Files |
| --- | --- |
| Profiles `'linear'`, `'raised_cosine'`, `'bottle'` (fitted, see below), any callable (finite-difference derivative with a warning) | `src/wrmodes/profiles.py` |
| `VariableGuide(length, a, b, kappa, u0, v0, plane, guide, ...)`, `Taper(length, a2, b2, profile, guide, du, dv)`, `StrangledBend(angle_deg, radius, b_flat, taper_length, profile, flat_length, guide)`, `GeometryFunction`, `convergence_study`, vectorized coefficients (`vi_system`, `taper_coupling_matrix`) | `src/wrmodes/components/taper.py`; exported in `wrmodes.components` and as `wrmodes.VariableGuide`, `Taper`, `StrangledBend` |
| `Line`: a component whose `guides` is None until attached (`Taper`/`StrangledBend` without a guide) takes the output guide of the attached copy (6 lines; no interface change) | `src/wrmodes/line.py` |
| Tests: 35 fast, 5 slow | `tests/test_taper.py` |
| Example: strangled bend in a line, taper in a line, convergence report | `examples/strangled_bend.py` |
| Benchmarks (npz with `meta`, CONVENTIONS §9) | `benchmarks/taper/`: `checks.py`, `t2b_table.py` (also E2), `e3_predictions.py`, `fit_bottle.py`, `butterweck_bottle.py`, `butterweck_fig15.py`, `staircase.py`, `timing.py`; data in `benchmarks/taper/data/` |
| Registry | `benchmarks/registry.json`: ten `wp5-*` entries |

Read: CONVENTIONS.md, T2, T2b and their reports and prototypes, T1, `loss.py`, the component code, WP4/WP4b/WP6
reports, CAMPAIGN_2 (E2, E3), Butterweck & de Ronde 1968 pp. 95–98 (page images of p. 96 fig. 13 and p. 98 fig. 15).
Huang §11.14/11.17 was not re-read (T2 §11 summarizes it; nothing in WP5 depends on it).

Data provenance: `src/` and `tests/` are unchanged since commit 380d2c8. The npz files record the commit at which
they were written: c78eb0c (checks, t2b_table, e2/e3 predictions, fit_bottle, butterweck_bottle, staircase), and the
later benchmark-script commits for butterweck_fig15, timing and the long staircase runs (scripts only changed).

## Method and decisions

* **Equations.** T2b B1–B3 in V–I form, dV/dz = T V − iZ I, dI/dz = −iY V − Tᵀ I, with T = Nᵀ of T2 (closed forms,
  vectorized over z and the mode pairs) and Z, Y the T0 curvature matrices of the local a(z), b(z) with
  h0 = 1 + κu0. Checked against T2's `vi_matrix` and T2b's `T_matrix` to 5.5e-16 (all four slopes, κ, u0).
* **Slices (T2 §8.2).** 4th-order Magnus with two Gauss points per slice; the slice GSM is built directly from the
  propagator with the local admittances at the slice ends; slices are joined by batched star products (tree
  reduction). No global transfer matrix. Interior slice edges avoid cutoff planes by δc = 1e-6 k². Slice density
  ∝ max(max_prop |β|/2, max_evan |β|/6, k/2); **adaptive refinement** to `atol` (default 1e-6, as `SmoothBend`) on
  the entries of the port modes, from the Richardson estimate of the 4th-order error, jumping to the predicted
  density (factor 2–4). Uniform intervals (the flat straight and arc of a strangled bend) are one slice doubled by
  star products (exact). The accuracy is governed by the evanescent modes of the narrow sections (error ∝
  (|β_evan| h)⁴): a WR-90 → 1.6 mm taper at 90 GHz needs about 600 slices per class.
* **Mode list (T2 §8.2 step 1).** Classes from the selection rules (a′: m parity; u0′ or κ: all m; b′: n parity; v0′:
  all n; equal lossy walls keep both parities); only the classes of the port modes are solved. Internal list of a
  class: every mode with min_z χ < `mode_factor`·k (the widest cross-section) plus the port modes of the class. Port
  modes outside it are dropped at that port. **Default mode_factor = 2** with B5′ (the T2b/E3 studies below show
  1e-4–1e-3 changes from 2 to 4; mode_factor 4 is used for the predictions).
* **B5′ (T2b §7), on by default.** The modes of the class outside the list are eliminated quasi-statically,
  Y → Y − T_outᵀ Z_out⁻¹ T_out. The sum over outside indices converges only as 1/J (T2b's m_tail = 500 is 7e-4
  from its limit in the E2 taper); the implementation sums 16 indices beyond the list explicitly and adds the rest
  in closed form, fitting Σ_r C_r/j^{2r} (r ≤ 4) to the last explicit terms of each parity and summing with
  polygamma functions: 4e-8 from an 800-index explicit sum (test).
* **Wall loss.** First order, local cross-section: Z → Z − iN_w, Y → Y − iG_w with the T1 B2 wall integrals of the
  local a, b weighted by h = 1 + κ(u0 + ũ) (vectorized copy of `loss.wall_integrals`, equal to it to 1e-12). This
  V–I form is **regular through cutoff** (the power-wave Λ diverges as 1/β there). T1 §5 rule: off-diagonal loss
  entries only between modes that propagate somewhere in the component (in a uniform guide: the propagating modes,
  exactly as `Straight`/`SmoothBend`). Difference to `Straight` with evanescent port modes: the V–I diagonal keeps
  the first-order change of the evanescent wave impedance (Λ⁻_pp), which `Straight` drops; propagating block
  identical, evanescent S11 of a mode just below cutoff differs by up to 8e-3 (TE4,1 at |β| = 50 m⁻¹), transmission
  by 2e-5. `families='separate'` (default, as `SmoothBend`) or `'coupled'` (equals a lossy `Straight`). Tilted-wall
  and impedance-wall boundary terms on moving walls (T2 §13) are neglected (second order).
* **Frames and ports.** `plane` is the frame of the local a, b and of the port bases; `guides` gives the global end
  guides. `StrangledBend` is one E-plane `VariableGuide` (local a = h, b = w), κ > 0 turning towards −u; port planes
  at the full-height ends; `length` = 2·taper + flat + R|θ| (centre line). With offsets (u0, v0) the length is along
  the reference line (T2b §10).
* **Profiles.** Analytic p′ for the named ones. **'bottle'**: see the next section.

## The bottle profile: source and fitted form

Butterweck & de Ronde (1968, p. 96, fig. 13) describe a 140 mm taper from 3.10 × 1.55 mm to 22.86 × 10.16 mm, b(z)
linear, a(z) "bottle-like", "the theoretical design providing a minimum H10 → H30 mode conversion", measured below
−40 dB (P5), against about −18 dB for a linear a(z) (P6). **The paper gives no formula and no dimensions**: fig. 13
is an unscaled perspective sketch. Its text gives only the rule "very small slope at the wide end, greater
inclination at the narrower part" (p. 96) and "the rest [where H30 is cut off] can be designed more or less
arbitrarily: we chose a linear function" (p. 97).

Fitted form (ours): p(t) = (1 − e^{−λt})/(1 − e^{−λ}), slope ratio e^λ between the ends, **λ = 3.5**, fitted to a
step of 0.5 by the lowest worst-case TE10 → TE30 conversion over 65–85 GHz in our model (family n = 0,
`fit_bottle.py`):

| λ | 65 GHz | 70 | 75 | 80 | 85 | worst (dB) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (linear) | −21.8 | −21.2 | −20.5 | −20.0 | −19.4 | −19.4 |
| 2 | −32.0 | −31.3 | −30.7 | −30.2 | −29.7 | −29.7 |
| 3 | −38.3 | −38.2 | −38.4 | −39.1 | −40.4 | −38.2 |
| **3.5** | −40.5 | −40.3 | −40.7 | −41.7 | −43.1 | **−40.3** |
| 4 | −40.5 | −39.2 | −38.1 | −36.9 | −35.3 | −35.3 |
| 5 | −35.5 | −32.8 | −30.4 | −28.3 | −26.3 | −26.3 |

A first attempt that implements the paper's argument literally (constant conversion per beat length beyond the
TE30 cutoff plane, a′ = C a (β1 − β3)) gives only −23 dB. The bottle is therefore a reconstruction, not
Butterweck's shape, and P5 is not an independent check of it.

## Acceptance tests

| Check | Target | Result |
| --- | --- | --- |
| Uniform limit (a, b constant; κ = 0 or 1/R) vs `Straight` / `SmoothBend`, lossless and aluminium, H and E | 1e-10 | **pass**: 5.2e-14 / 5.8e-14 (straight), 2.6e-13–5.3e-13 (bends; `SmoothBend` at atol 1e-12), propagating modes. With evanescent port modes: propagating block 1e-10, see "Wall loss" |
| Airy (single mode through cutoff), Bessel (linear taper) | 1e-8 | **pass**: 5.0e-12 and 9.9e-12 at atol 1e-11 (4.9e-9, 5.6e-9 at 1e-8; 5.8e-7, 6.3e-7 at the default 1e-6) |
| Total reflection of TE3,0 at its cutoff plane (a′ = 0.028) | 1 − \|S22\| ≤ 1e-4; phase 0.02 rad | **pass**: 2.1e-5; 0.0172 rad (mf 2), 0.0171 (mf 4); without B5′ 0.0127 / 0.0147 |
| T2b §9 table (L = 0.1–0.8 m) | 1e-4 abs | **pass**: S21(TE30←TE10) within 4.8e-6, 1.7e-6, 4.6e-7, 1.4e-7; \|S11\| within 8.5e-6 … 8.2e-7 (default settings) |
| Staircase of WP4 `Step`s (H-plane and b tapers) | extrapolated ≤ 1e-4 | **pass**: b-taper 8.6e-5 raw at n = 1280, 5.6e-5 extrapolated; H-plane 4.7e-4 raw at n = 1280, **2.4e-5** extrapolated (observed order 1.01). Table in "Staircase" below |
| Butterweck bottle vs linear (P5, P6), 75 GHz | bottle < −40 dB; linear reported | bottle **−40.6 dB** (fitted profile, full model with b linear, mf 1.5/2, B5′ on/off: −40.63 … −40.65); linear **−20.5 dB** vs about −18 dB measured |
| Butterweck fig. 15 (P8, P9) | reported | see below: "a few tenths of a dB" reproduced for the symmetric readings; "mainly wall loss" only with a kink-free taper |
| Reciprocity; unitarity (lossless); passivity (lossy) | CONVENTIONS §6 | **pass**: general geometry (a′, b′, u0′, v0′, κ), cf 1.5: 6.4e-15 / 1.3e-14, lossy margin 1.5e-3; strangled bend 40 GHz cf 1.5: 3.6e-13 / 3.8e-13, lossy margin 2.8e-3; every benchmark GSM ≤ 2e-13 |
| Convergence in M with and without B5′ | reported, every case | tables below (T2b/E2, E3, bottle, total reflection, fig. 15 one case) |

Cross-check against the T2 prototype `vi_solver.py` (a′, b′, u0′, κ, families 0 and 2, 13 modes): 7.9e-8 (its 160×2
slices).

### Convergence in M (T2b table / E2 taper), max |S − S(mf 8, B5′)| over S11, S21, S31

| L (m) | B5′ mf 2 / 4 | no B5′ mf 2 / 4 / 8 | Richardson (no B5′, 4–8) |
| --- | --- | --- | --- |
| 0.1 | 2.0e-4 / 3.2e-5 | 1.4e-3 / 8.6e-4 / 4.5e-4 | 3.1e-5 |
| 0.2 | 9.8e-5 / 1.5e-5 | 7.1e-4 / 4.3e-4 / 2.2e-4 | 1.5e-5 |
| 0.4 | 4.9e-5 / 7.5e-6 | 3.6e-4 / 2.2e-4 / 1.1e-4 | 7.4e-6 |
| 0.8 | 2.4e-5 / 3.7e-6 | 1.8e-4 / 1.1e-4 / 5.6e-5 | 3.6e-6 |

Plain truncation halves the error per doubling (O(1/M), T2 §8.4); one Richardson step or B5′ gains a factor 15–30.

## Staircase of WP4 Steps

`benchmarks/taper/staircase.py`. The taper (L = 0.1 m) is replaced by n uniform sections with the sides of the
taper at the section midpoints, joined by WP4 mode-matching `Step`s (default truncation, parity-reduced), plus a
step from each end cross-section. The sections carry the modes of the coupled class with cutoff below cf·f,
cf = max(6, 4n/(kL)) (every mode with |β|·L/n ≲ 4, through which neighbouring steps interact). Reference:
`Taper(..., mode_factor=8, atol=1e-9)`. Cases: 'H-plane' 22.86 → 45 mm, b = 10.16 mm, 15 GHz (TE3,0 crosses
cutoff), family 0; 'b-taper' 22.86 × 10.16 → 22.86 × 5.08 mm, 35 GHz (TE1,2/TM1,2 cross cutoff), m = 1.
Max |S_stair − S_taper| over every propagating entry:

| n | H-plane | b-taper |
| --- | --- | --- |
| 20 | 2.1e-1 | 2.7e-1 |
| 40 | 1.1e-1 | 6.6e-2 |
| 80 | 3.1e-2 | 2.3e-2 |
| 160 | 5.9e-3 | 7.4e-3 |
| 320 | 2.0e-3 | 2.1e-3 |
| 640 | 9.7e-4 | 3.0e-4 |
| 1280 | 4.7e-4 | 8.6e-5 |
| Richardson, n = 640/1280 (observed order) | **2.4e-5** (1.01) | **5.6e-5** (3.0) |
| section truncation ×2 | 4.7e-3 at n = 320 | 1.1e-5 at n = 320, 5.8e-4 at n = 640 |

(The truncation of each n = 1280 run equals the doubled truncation of n = 640.) Before n = 640 the convergence is
erratic (observed orders 1.5–2.7) and the staircase's own section truncation dominates. Times per staircase:
H-plane 30 s / 286 s / 2927 s and b-taper 20 s / 100 s / 760 s at n = 320 / 640 / 1280. Data: `staircase.npz`
(n = 20–320), `staircase_b_long.npz`, `staircase_b_1280.npz`, `staircase_h_640.npz`, `staircase_h_1280.npz`.

## COMSOL campaign 2 predictions

### E2: linear H-plane taper 22.86 → 45 mm, b = 10.16 mm, 15 GHz (`e2_predictions.npz`)

Reference planes at the taper ends; mode_factor 8 with B5′; ± = largest change over the convergence study.

| L | S11 (TE10) | S21 (TE10 ← TE10) | S31 (TE30 ← TE10) |
| --- | --- | --- | --- |
| 0.1 m | 0.00028 + 0.00199i (−53.92 dB, 81.9°) ± 3e-6 | 0.03564 + 0.99809i (−0.011 dB, 87.95°) ± 2e-4 | 0.05006 − 0.00529i (−25.96 dB, −6.0°) ± 7e-5 |
| 0.2 m | 0.00000 + 0.00085i (−61.40 dB, 89.9°) ± 2e-6 | −0.99847 + 0.04894i (−0.003 dB, 177.19°) ± 1e-4 | 0.00271 + 0.02552i (−31.81 dB, 83.9°) ± 1e-5 |

### E3: strangled E-plane bend, WR-90, 50 mm raised-cosine tapers to 1.6 mm, R = 50 mm, 90° (`e3_predictions.npz`)

Global mode names (TE1,0: E across 10.16 mm); reference planes at the full-height ends; mode_factor 4 with B5′.

| Case | S11 (TE10) | S21 (TE10) | reflected power | converted power (port 2) |
| --- | --- | --- | --- | --- |
| 18 GHz, PEC | −0.00172 + 0.01443i (−36.76 dB) | 0.992838 + 0.118584i (−0.0009 dB) | TE1,0 2.11e-4 | none above 1e-30 (TE1,1/TM1,1 propagate but are cut off in the flat guide) |
| 18 GHz, Al 3.774e7 | −0.00165 + 0.01420i (−36.89 dB) | 0.984597 + 0.108204i (**−0.0827 dB**) | TE1,0 2.05e-4 | none |
| 40 GHz, PEC | −0.00006 − 0.00288i (−50.82 dB) | −0.994994 + 0.094927i (−0.0042 dB) | TM1,2 4.75e-4, TE1,2 1.0e-5, TE1,0 8.3e-6 | TM1,2 4.73e-4, TE1,2 1.0e-5 |
| 40 GHz, Al 3.774e7 | −0.00006 − 0.00281i (−51.04 dB) | −0.980875 + 0.106426i (**−0.1169 dB**) | TM1,2 4.77e-4, TE1,2 1.0e-5, TE1,0 7.9e-6 | TM1,2 4.64e-4, TE1,2 9.9e-6 |

Convergence (TE10 S21): mf 2 / 3 → 4 change 2.8e-4 / 2.6e-4 (18 GHz, a phase change; |S21| in dB to 1e-5),
1.0e-3 / 8.4e-5 (40 GHz); without B5′ 2e-2–3e-2 (mf 2) and 1.3e-2 (mf 4): B5′ matters here (E-plane taper, wall
term W ≠ 0). Converted TM1,2 at 40 GHz: 4.828e-4 / 4.825e-4 / 4.825e-4 (mf 2/3/4), 5.00e-4 without B5′. Slice
refinement ×2: 6e-9 (18 GHz), 9e-10 (40 GHz). The dB values are converged to 1e-4 dB.

**Physics note for SPARC.** The TM1,2 power is generated near its cutoff plane in each taper and totally reflected
there (T2 §8.5): it appears as reflection at port 1 and as conversion at port 2. At 90 GHz the same bend reflects
1.6 % of the input into TM1,2 and transmits 1.4 % in it (TE1,0 loss 0.31 dB with aluminium, `examples/strangled_bend.py`);
the plain full-height E-plane bend of the same radius loses 3.1 dB.

## Butterweck & de Ronde fig. 15 (P8, P9)

The paper gives 22.86 × 10.13 mm, b1 = 2 mm, r = 420 mm, use at λ ≥ 4 mm, "a few tenths of a decibel, mainly due to
the wall losses in the flat guide". Taper length, taper shape and the bend radius of the flat guide are not given.
Our reading (drawing and photograph with its 5 cm scale): r is the radius of the curved taper walls (a 420 mm bend
radius cannot fit the 10 cm block), the walls are arcs tangent to the flat guide, and the flat bend is compact
(R_b = 10–40 mm computed). Three readings (`butterweck_fig15.py`): asymmetric (inner wall straight, taper 82.2 mm,
axis offset 4.07 mm), symmetric (58.3 mm), and symmetric with a raised-cosine height profile of the same length (no
wall kink at the full-height end; the arcs meet it at 8.0° (symmetric) or 11.3° (asymmetric)). TE1,0 loss in dB:

R_b = 20 mm (PEC / aluminium 3.5e7 / copper 5.8e7):

| Reading | 40 GHz | 60 GHz | 75 GHz |
| --- | --- | --- | --- |
| asymmetric arcs (82.2 mm) | 0.432 / 0.528 / 0.506 | 0.919 / 1.037 / 1.010 | 1.350 / 1.472 / 1.445 |
| symmetric arcs (58.3 mm) | 0.090 / 0.166 / 0.149 | 0.168 / 0.260 / 0.239 | 0.207 / 0.310 / 0.287 |
| symmetric raised cosine (58.3 mm) | 0.002 / 0.068 / 0.053 | 0.015 / 0.095 / 0.077 | 0.051 / 0.142 / 0.122 |

R_b from 10 to 40 mm moves the aluminium values by −0.012/+0.024 dB (symmetric arcs, 40 GHz) up to −0.014/+0.023 dB
(raised cosine, 75 GHz): the ohmic part grows with the flat-guide length (0.054–0.124 dB in aluminium over the
whole table for the raised cosine), the conversion part does not depend on R_b (the flat-guide modes other than
TE1,0 are cut off).

* The symmetric readings give "a tenth to a few tenths of a dB": 0.15–0.33 dB in aluminium with the arc walls,
  0.06–0.17 dB with the raised cosine.
* With the arc tapers, half or more of it is conversion (TM1,2 generated near its cutoff plane and totally
  reflected there, 1.0–2.4 % reflected power, see E3), caused by the wall kink at the full-height end: with the
  kink-free raised cosine the conversion drops to 0.002–0.05 dB and the loss is mainly ohmic (40 GHz: 0.002 dB of
  0.068 dB), as the paper says. The asymmetric reading (offset axis: u0′ couples TE1,0 to TE1,1/TM1,1, 3.6 % reflected into TM1,1 at
  40 GHz) contradicts the paper and is probably not their geometry. Converged in M (mf 2/3, B5′ on/off: changes
  below 3e-3 of the power). This is a report, not a pass/fail item.

## Timings

`benchmarks/taper/timing.py`, this container (2 cores, Intel Xeon 2.1 GHz, one BLAS thread inside the component),
best of 2 runs, default settings, the line's propagating basis at the ports:

| Case | Time | Target (card) | Classes / modes per class / slices |
| --- | --- | --- | --- |
| WR-90 → 22.86 × 1.6 mm raised-cosine taper, 50 mm, 90 GHz | 4.6 s | < 5 s | 28 / ≤ 13 / 15 561 |
| SPARC strangled bend, two such tapers, R = 50 mm, 90°, 90 GHz | 23.2 s | < 30 s | 14 / ≤ 25 / 30 556 |
| same, R = 200 mm | 22.9 s | < 30 s | 14 / ≤ 25 / 73 564 (arc doubled, not sliced) |
| E3 strangled bend, 40 GHz | 1.6 s | – | 7 / ≤ 11 / 8 310 |

The targets are met on this machine, with little margin for the taper. The tests assert 3× the targets (CI) and the
targets themselves only with WRMODES_PERF=1. Other timings: E3 at 40 GHz with mode_factor 4, 10 s; the Butterweck
bottle (one class of 49 / 91 modes at 75 GHz), 22 s / 117 s; the full fast test suite, 14 min with two benchmark jobs
running beside it.

## Proposals for CONVENTIONS.md (not changed)

1. §3: "solvers that need evanescent modes state their own default": WP5's internal count is `mode_factor` = 2,
   applied at the widest cross-section, separate from the port bases (the line's rule). Worth a sentence.
2. §6 "lengths along the centre line": a taper with offsets u0, v0 measures along its reference line (T2b §10); the
   axis length differs at second order in the offset slope (`VariableGuide.centre_line_length`). State it.
3. §3 / T1 §5 for components with cutoff crossings: the loss rule "propagating modes" becomes "modes that propagate
   somewhere in the component", and the evanescent diagonal keeps Λ⁻ (V–I form, regular at cutoff); see "Wall loss".

## Open issues

1. **Staircase cost.** The WP4 staircase converges slowly (first order in the step count once asymptotic) and
   needs section mode counts that grow with n; the H-plane point at n = 1280 took 49 min. The staircase is a
   benchmark, not a test (the slow test checks the trend for the b-taper only).
2. The bottle profile is a reconstruction (λ fitted); P5 is a consistency check, not a validation.
3. Fig. 15 geometry is a reading of a sketch; the conclusion depends on the taper shape.
4. Performance: dominated by the evanescent modes of the narrow sections; the adaptive slicing makes the default
   path 1.5–3× the cost of a single well-chosen slicing. The default `atol` 1e-6 can be relaxed for design sweeps.
5. The impedance-wall boundary terms on tilted walls (T2 §13) are not derived; loss uses the local cross-section.
6. κ with b′ ≠ 0 (curvature together with moving v-walls) has no exact-field check (T2 §13). The T2-prototype
   comparison (κ, a′, b′, u0′ together, 7.9e-8) checks the implementation of the same equations, not the physics.

## Decisions for the coordinator / Daniel

* Accept the fitted bottle (λ = 3.5) as the meaning of `'bottle'`, or ask for another form.
* Choose the fig. 15 reading for the record (we suggest the symmetric one; R_b ≈ 20 mm).
* The SPARC strangled bend: the TM1,2 near-cutoff reflection (1.6 % at 90 GHz for 50 mm raised-cosine tapers to
  1.6 mm) is a design issue for WP8 (longer or shaped tapers).
