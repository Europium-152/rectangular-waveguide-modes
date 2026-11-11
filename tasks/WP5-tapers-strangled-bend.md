# WP5 — Tapers and the strangled E-plane bend

Role: Implementer · Wave: 2 · Branch: `wave-2/WP5-tapers` · Depends on: T2, T2b (merged), WP1, WP2, WP4 (merged) ·
Verified by: WP7b (WP3 FEM for one-family cases, COMSOL campaign 2) · Time box: 10 days (then status review)

## Goal

Components whose cross-section varies along the axis, optionally while bending: H- and E-plane tapers, general
tapers of a and b with centre offsets, and the **strangled E-plane bend** of Butterweck & de Ronde (1968): a
full-height guide tapered down to a flat guide of height $b_1$, bent in the E-plane, and tapered back up. Built
with the cross-section (local-mode) method of T2, through cutoff crossings, with first-order wall loss.

## Read first

* `CONVENTIONS.md` (§2 frames, §3 truncation and the exact-cutoff rule, §6 reference planes and lengths along the
  centre line, §8 tolerances).
* `docs/theory/T2-taper-coupling.md`: §2.4 (result), §3 (closed forms), §6 (curvature with taper), §8 (the
  GSM-slice method, the mode count protocol and the total reflection at a cutoff plane), §13.
* `docs/theory/T2b-taper-coupling.md`: §5, §7 (B5' `tail_correction`, which restores fast convergence in $M$),
  §9 (small-slope table), §10. The two notes agree to 7e-15 (`docs/reports/T2b.md`).
* Prototypes: `docs/theory/T2/vi_solver.py`, `docs/theory/T2b/t2b_solver.py` (check tools, slow; do not import
  them from `src/`).
* `docs/theory/T1-wall-impedance.md` and `src/wrmodes/loss.py` (local loss matrices).
* Butterweck & de Ronde (1968) pp. 95–98: the bottle taper (p. 96: 140 mm, 1.55 × 3.10 mm to 10.16 × 22.86 mm,
  H10→H30 below −40 dB against about −18 dB for a linear a(z)), and the E-plane bend of fig. 15 (p. 98:
  a × b = 22.86 × 10.13 mm, r = 420 mm, $b_1$ = 2 mm; "a few tenths of a dB", mainly wall loss in the flat guide).
  `benchmarks/literature/butterweck1968_numbers.csv` items P5, P6, P8, P9.
* Huang vol. II §11.14 and §11.17 (pp. 474–479, 489–494), as summarized in T2 §11.

## Specification

1. **Profiles** (`src/wrmodes/profiles.py`): a profile is $p(t)$ on $t \in [0, 1]$ with $p(0) = 0$, $p(1) = 1$:
   `'linear'`, `'raised_cosine'` ($\tfrac12(1-\cos\pi t)$), `'bottle'` (Butterweck's form; take it from p. 96
   and fig. 13 and document the exact formula and its source in the docstring; if the paper gives only a shape,
   say so and state the fitted form) and any callable. Provide $p'(t)$ (analytic for the named profiles,
   finite differences with a warning for callables).
2. **Components.**
   * `VariableGuide(length, a, b, kappa=None, u0=None, v0=None, plane='H', guide=None)`: a and b (m) and
     kappa (1/m, signed as in `SmoothBend`) are callables of the centre-line coordinate $z \in [0, L]$; u0, v0 are
     centre offsets (default 0). This is the general T2 component. `plane` is the frame of its port bases;
     `guides` gives the end cross-sections in the global frame (CONVENTIONS §10). With `guide.sigma` set, wall
     loss enters as the T1 loss matrices of the **local** cross-section, to first order, as in bends.
   * `Taper(length, a2, b2, profile='linear', guide=None, du=0, dv=0)`: from the line's current guide to
     $(a_2, b_2)$, both dimensions with the same profile, with optional end offsets of the centre.
   * `StrangledBend(angle_deg, radius, b_flat, taper_length, profile='raised_cosine', flat_length=0.0,
     guide=None)`: taper of $b$ (the E-plane side) to `b_flat`, an optional straight flat section of
     `flat_length` split equally before and after the bend, an E-plane arc of `radius` (centre line) and angle in
     the flat guide, the mirror taper back. Port planes at the two full-height ends. It may be built internally
     as a cascade; its `gsm` returns the whole component. `arm_length`-style metadata: total centre-line length.
3. **Method.** The GSM-slice algorithm of T2 §8.2 (V–I Magnus propagators, slice GSMs built directly, star
   cascade), with B5' (T2b §7) on by default and switchable. Mode list from the widest cross-section and the
   reachable families (T2 §8.2 step 1, selection rules of §5); keep every mode that propagates anywhere inside.
   Slices avoid cutoff planes by $\delta_c$ (T2 §8.2 step 2). Ports: the line's bases at the end guides
   (CONVENTIONS §3 'live' rule); modes of the internal list outside a port basis are dropped at that port.
   Closed-form coefficients, vectorized; no quadrature in the default path.
4. **Convergence report.** `convergence_study(f)` reporting $S$ for two mode counts ($f_c$ and $\approx 2f_c$),
   the Richardson extrapolation (T2 §8.4) and the slice refinement, as `Miter.convergence_study` does in WP3.
5. **Performance.** WR-90 → 22.86 × 1.6 mm raised-cosine taper of 50 mm at 90 GHz in under 5 s per frequency at
   default settings on a laptop; the strangled bend of the SPARC study (two such tapers, R = 50–200 mm, 90°) in
   under 30 s.

## Acceptance tests

| Check | Target |
| --- | --- |
| Uniform limit ($a$, $b$ constant, κ = 0 or constant) | `Straight` / `SmoothBend` (lossless and lossy) to 1e-10 |
| Single mode through cutoff (Airy) and linear taper (Bessel), as in `T2/check_solver.py` | exact solutions to 1e-8 |
| Total reflection at a cutoff plane (TE30, linear H-plane taper, $a'$ = 0.028, T2 §8.5) | $1-\lvert S_{22}\rvert \le 10^{-4}$, phase within 0.02 rad of $i e^{-2i\int\beta}$ |
| T2b §9 table (22.86 → 45 mm, b = 10.16 mm, 15 GHz, L = 0.1–0.8 m) | $S_{21}$(TE30←TE10) and $S_{11}$(TE10) to 1e-4 absolute |
| Staircase of WP4 `Step`s (H-plane taper and a taper of b, both) | converges to the WP5 result as the step count grows; extrapolated difference ≤ 1e-4 in every propagating entry (report the convergence) |
| Butterweck bottle vs linear taper (P5, P6), 75 GHz | bottle H10→H30 below −40 dB; linear about −18 dB (report the value; the measurement has no stated uncertainty) |
| Butterweck fig. 15 strangled bend (P8, P9), aluminium and copper | total loss reported against "a few tenths of a dB, mainly wall loss" (not pass/fail) |
| Reciprocity; unitarity lossless; passivity lossy | CONVENTIONS §6 |
| Convergence in $M$ with and without B5' | reported for every acceptance case |

## Deliverables

* `src/wrmodes/profiles.py`, `src/wrmodes/components/taper.py` (or a `taper/` package), exports in
  `wrmodes.components`.
* `tests/test_taper.py`; slow cases marked `slow`.
* `docs/reports/WP5.md` and `examples/strangled_bend.py`.

## Out of scope

Twists inside tapers; non-rectangular cross-sections; loss beyond first order; the design study itself (WP8).
