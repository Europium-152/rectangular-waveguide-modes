# WF2 — Wave 2 fixes before Gate B: report

Branch `wave-2/WF2-fixes`, from `main` at 7cb7cf1 (card commit 1fca34d). Card `tasks/WF2-wave2-fixes.md`. The three
findings of WP7b that Daniel accepted on 15 Nov 2026 are fixed; every wave 2 row is now met against WP7b's committed
references (τa = 0.5 with an option, as the card allows). WP7b's reference scripts and data are unchanged; the
comparison is `benchmarks/validation_v2/compare_wf2.py` (REPORT.md, "After WF2"; data `data/compare_wf2_*.json`).
Times: 2-core container, one BLAS thread per GSM (as `Twist.gsm` and `Line.sweep` set it).

## What was done

| Item | Files |
| --- | --- |
| 1. Twist truncation: internal mode list (`mode_factor`), quasi-static tail (`twist_tail_matrix`, `twist_tail_system_matrix`), θ′² term in the expm and Magnus slices, corrected docstring, new `convergence_study` | `src/wrmodes/components/twist.py`; tests `tests/test_twist.py`; WP6 benchmark scripts pinned to WP6's model (`benchmarks/twist/*.py`); `examples/twist.py` |
| 2. Bends next to discontinuities: `SmoothBend` with evanescent modes solved as a `VariableGuide`; `Line` near-field warning | `src/wrmodes/components/bends.py`, `src/wrmodes/line.py`; tests `tests/test_components.py`, `tests/test_line.py`, `tests/test_bend_evanescent_fem.py` (fem job), bitwise data `tests/data/smoothbend_propagating.npz` (+ generator) |
| 3. Strangled-bend port bases built from the given guides | `src/wrmodes/components/taper.py`; tests `tests/test_taper.py` |
| Comparison, records | `benchmarks/validation_v2/compare_wf2.py`, REPORT.md section, `benchmarks/registry.json` (2 entries updated, 2 added), Resolution sections in both discrepancy files |

Commits: e60074c (item 3), f4a9f6c (item 2), 49ebb5e (item 1), then the records; each pushed and checked with
`git ls-remote`.

## 1. Twist truncation (WP6)

**Diagnosis.** In V-I form the twist system (T3b box 2) is dV/dz = τK V − iZ I, dI/dz = −iY V − τKᵀ I with
Z = diag(β/Y_p), Y = diag(βY_p): exactly the structure of the taper system of T2b B1, with T = τK. The slow
convergence has the same origin as for tapers (T2b §7): on the helicoidal walls H_n = w_n H_z ≠ 0 (T3 §7, T3b §5)
while every basis field has H_n = 0, and the TE modes outside the list answer the twist quasi-statically,
I_q = τ Σ K_qp V_p /(iZ_q), with Z_q = kη0 not growing with q. This is first order in the series for any rate, so the
docstring's "(τ/k)²" statement was wrong for practical twists too; it is replaced by measured numbers.

**Method.**

* `Twist(mode_factor=None, tail=True)`. The internal list holds the port modes, every mode (all families) below
  mode_factor·f and cutoff_factor·f, completed up to its highest cutoff. Defaults (`DEFAULT_MODE_FACTOR`): 3 for a
  constant rate (abrupt ends; one expm, cheap) and 2 for variable rates (cost ∝ N³ × slices). The port bases stay the
  line's; internal modes outside them are dropped at the ports (as WP5's `mode_factor`).
* `twist_tail_matrix`: the modes outside the list are eliminated, Y → Y − θ′² Q,
  Q_pp′ = Σ_q K_qp K_qp′ / (Z_q √(r_qp r_qp′)), r_qp = (χ_q² − χ_p²)/(χ_q² − k²). r is the response factor of an
  outside mode to a drive ∝ exp(∓iβ_p z) (exact for one harmonic drive in a uniform twist; the plain B5′ form has
  r = 1). The geometric mean keeps Q real symmetric. The lattice sum runs to R·χ_max for R = 8 and 16 and is
  extrapolated in 1/R (the terms fall like 1/j² along the index axes); R = 12/24 changes Q by < 1e-3 relative; in
  the prototype study a three-point extrapolation changed the worst ΔP by ≤ 6e-7 (≤ 3e-6 for τa = 0.5). Q is block diagonal in the parity classes.
* `twist_tail_system_matrix`: in power waves the term is θ′² X2, X2 = (i/2)[[W, W], [−W, −W]],
  W = Y^(-1/2) Q Y^(-1/2). It meets the power condition X2ᴴJ + JX2 = 0 and the reciprocity condition
  X2ᵀK + KX2 = 0 that D and X meet (tested), so GSMs stay reciprocal and lossless to round-off (≤ 2.3e-13 in all the
  comparison cases). Constant rates: one expm of D + θ′X + θ′²X2. Variable rates: 4th-order Magnus with the general
  commutator [M2, M1] = (t1 − t2)[D, X] + (t1² − t2²)[D, X2] + t1t2(t1 − t2)[X, X2].
* Tried and dropped: (a) the tail alone on the propagating list (mode_factor 1): good for smooth rates far from
  cutoff, but 0.31 off for the Doane constant twist at 16.5 GHz with the plain B5′ form (a TM mode 0.07 % below
  cutoff has Z_q → 0) and 1.5e-4 for the triangular twist at 18 GHz with the dynamic form; (b) the dual Z correction
  (outside TM modes in the V equation): 30 % better at mode_factor 1.5, but its lattice sum grows with the list and
  it brings nothing at the defaults; (c) Richardson on S in the mode count: breaks unitarity at the 1e-5 level.

**Acceptance** (worst ΔP over the modes converted from TE1,0, transmitted and reflected, against WP7b's helical FEM;
tolerance 1e-4 and 0.01 dB):

| case | f (GHz) | WP6 default | WF2 default | Δ TE1,0 (dB) | modes | time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Doane constant | 15 / 16.5 / 18 | 4.1e-4 / 4.2e-5 / 4.0e-4 | 5.1e-6 / 1.1e-6 / 9.0e-6 | ≤ 5e-5 | 150 / 182 / 210 | 0.3 / 0.5 / 0.7 |
| Doane cosine | 15 / 16.5 / 18 | 2.9e-5 / 4.4e-5 / 1.6e-4 | 2.4e-6 / 1.3e-6 / 1.2e-5 | ≤ 4e-5 | 69 / 78 / 96 | 16.5 / 22.9 / 44.2 |
| Doane triangular | 15 / 16.5 / 18 | 7.7e-6 / 2.9e-5 / 7.4e-5 | 5.7e-6 / 1.4e-5 / 2.3e-5 | ≤ 1.5e-4 | 69 / 78 / 96 | 16.5 / 24.4 / 42.7 |
| square 30 mm, 2a/λ = 2.5, L = 5λ / 20λ | 12.49 | 4.6e-3 / 1.7e-4 | 6.1e-5 / 2.6e-6 | −0.0024 / −0.0001 | 88 | 0.1 |
| WR-90, τa = 0.1 | 15 / 30 | 6.6e-5 / 2.4e-4 | 3.1e-7 / 9.6e-7 | < 1e-5 | 33 / 129 | < 0.3 |
| WR-90, τa = 0.5 | 30 | 2.6e-2 | 1.9e-4 (default), 8.0e-5 (mode_factor 4), 2.0e-5 (6) | +0.0010 / +0.0004 / +0.0001 | 129 / 236 / 525 | 0.2 / 0.8 / 6.3 |

* **Run time at the new defaults, Doane's three twists at 15, 16.5, 18 GHz: 169 s** (constant 1.5 s, cosine 84 s,
  triangular 84 s), against 0.6–1 s per variable-rate GSM with WP6's default. The variable rates spend ~60 % in the
  slice exponentials (2300–2600 Magnus slices over the adaptive levels of 96 × 2 = 192-row systems at 18 GHz).
* τa = 0.5: reached with `mode_factor=4` (8.0e-5) or 6 (2.0e-5); the default is kept at 3 (1.9e-4) because 4 doubles
  the mode count of every constant twist; documented in the docstring.
* The triangular twist's unexplained 2 % in TE1,2 at 18 GHz (WP7b open issue 6) was the truncation: 0.3739 % against
  the reference's 0.3761 % (WP6 0.3836 %).
* Convergence (Doane constant, 15 GHz): with the tail, mode_factor 1.5 / 2 / 3 / 4 gives 5.2e-5 / 1.3e-5 / 5.1e-6 /
  2.5e-6; without it 2.2e-4 / 2.2e-4 / 1.4e-4 / 1.4e-4. The remainder falls like about mode_factor^-2 to ^-3.
  `convergence_study` now returns rows in mode_factor with and without the tail, the Richardson extrapolation of the
  plain series in 1/χ_max (T3b §5), and the change between the best row and that extrapolation (1.5e-5 here, an
  independent estimate that agrees with the reference).
* The strict xfail `test_doane_constant_twist_against_helical_fem_15GHz` passes; marker removed.

## 2. Bends next to discontinuities (WP1)

**Method.** `SmoothBend.gsm` with a basis that contains evanescent modes now builds the bend as
`VariableGuide(L, a, b, kappa=κ(z), plane, guide, uniform=[(0, L)] for constant κ, breakpoints, atol, families)` and
returns its GSM on the given bases (`SmoothBend.variable_guide()` exposes it). That is WP5's direct slice-GSM method
(4th-order Magnus slices, slice GSMs from the local admittances, Redheffer products; exact doubling for constant κ).
The internal list: twice the cutoff factor the port basis reaches, at least 2 (`SmoothBend(mode_factor=None)`).
Measured at cutoff_factor 3 for the WP7b cases: internal factor 2 / 4 / 6 gives 2.8e-4 / 1.9e-4 / 1.6e-4 at the worst
point (R = 2a, 20 GHz, L = a), so the default doubles the port factor. Both solvers integrate the same coupled-mode
equations (JINST / T0 curvature matrices, exact in κ for a coplanar bend). The B5′ tail of VariableGuide is inactive
here (it corrects the taper term T, which is zero).

**Bitwise check.** With propagating-mode bases the old transfer-matrix route is untouched: 20 GSMs (perfect and
aluminium walls, H and E plane, sech and constant curvature, families 'separate' and 'coupled', alone and in a Line;
18–90 GHz) are bitwise equal to `origin/main` (7cb7cf1), compared from a separate checkout. 14 of them are frozen in
`tests/data/smoothbend_propagating.npz` (generated on that commit) and the test requires `np.array_equal`. This is
also the card's "lossy bends unchanged on propagating modes" (to 0, not only 1e-12).

**Line warning.** `Line.gsm` issues one `EvanescentJunctionWarning` per line when two non-uniform components
(anything but `Straight` and `FrameChange`) are joined **directly** while the basis at that junction has no
evanescent modes. "Directly" (module docstring of `wrmodes.line`): the straights between them, frame changes counted
as zero length, are shorter than `NEAR_FIELD_DECAY_LENGTHS` = 3 decay lengths 1/|β| of the least-attenuated
evanescent mode of the guide there (that mode keeps > e⁻³ = 5 % of its amplitude across the gap). Example: WR-90 at
20 GHz, TE3,1, 10 mm. After WF2 every component of the library couples the evanescent modes it is given, so the
condition reduces to the basis.

**Acceptance** (max | |S|² difference | over the propagating GSM against WP7b's single FEM domain; full table in
REPORT.md):

| R | f (GHz) | L/a | before WF2, cf 3 | WF2, cf 3 | target | FEM-arc cascade, cf 3 (WP7b) |
| --- | --- | --- | --- | --- | --- | --- |
| 2a | 20 | 0.25 / 0.5 / 1 / 2 | 2.5e-2 / 3.0e-2 / 4.2e-2 / 2.4e-2 | 1.7e-4 / 1.2e-4 / 1.6e-4 / 1.1e-4 | 2e-4 | 1.6e-4 / 1.1e-4 / 1.4e-4 / 1.1e-4 |
| 2a | 40 | 0.25 / 0.5 / 1 | 2.0e-2 / 2.0e-2 / 1.8e-2 | 5.7e-5 / 4.8e-5 / 6.3e-5 | 2e-4 | 5.6e-5 / 4.6e-5 / 6.2e-5 |
| 10a | 20 | 0.25 / 0.5 / 1 / 2 | 4.5e-3 / 5.2e-3 / 5.8e-3 / 5.0e-3 | 2.4e-5 / 2.4e-5 / 2.5e-5 / 2.3e-5 | 3e-5 | 2.4e-5 / 2.4e-5 / 2.5e-5 / 2.3e-5 |
| 10a | 40 | 0.25 / 0.5 / 1 | 4.0e-3 / 4.3e-3 / 4.2e-3 | 8.8e-6 / 9.7e-6 / 1.0e-5 | 3e-5 | 8.8e-6 / 9.7e-6 / 1.0e-5 |

* TE1,0 transmission within 0.001 dB in every case. Unitarity of the line ≤ 6e-14; the warning fires at
  cutoff_factor 1 and not at 3 in every case. The remainder is the truncation of the junction itself (line at
  cutoff_factor 4: 7.5e-5 at the worst R = 2a point), as for WP7b's FEM-arc cascade.
* Reciprocity, unitarity, passivity (CONVENTIONS §6): tests for H and E plane, constant and sech curvature, lossy
  E-plane bend (passive, reciprocal), symmetric sech bend S11 = S22 to 1e-9, straight limit to 1e-12.
* Timing: the bend itself 0.02–0.2 s for these short bends (20–40 GHz, internal factor 6); the 1.3–13.8 s per line are
  the FEM miter. A long bend costs more: the 40 cm 90° sech bend in WR-90 at 40 GHz with cutoff_factor 3 takes 110 s
  (9 families of ~65 modes, ~650 slices), against milliseconds with propagating modes only.
* Side finding: coupling the evanescent modes changes the propagating block of ordinary bends, by little on the
  TE1,0 column but noticeably on entries between high-order modes near their cutoffs: 40 cm 90° sech bend at 40 GHz,
  TE1,0 column ≤ 1.4e-7, TE2,2 / TM2,2 entries 2.8e-4 (H plane), TM1,5 4.8e-4 (E plane), converged in the internal
  list; WP1's mixed H/E line at 33 GHz with cutoff_factor 1.5: TE1,0 column 2e-6, block 1.3e-3. These are truncation
  errors of the propagating-only coupled-mode model for near-cutoff modes; they do not affect TE1,0 studies.

## 3. Strangled-bend port bases (WP5)

`VariableGuide` builds its port guides from `_end_sides()`; `StrangledBend` returns the given guide at both ports
(local a = h, b = w, exactly) and `Taper` returns (a1, b1) and the given (a2, b2). The profile still gives a(L) one
rounding error from h inside the solver, which is harmless. `StrangledBend.guides == (guide, guide)`, so a line
continues in exactly the input guide. WR-90 at 60 GHz: 0 of 200 flat heights give basis2 ≠ basis1 (25 before);
`g @ g` and `g.reversed()` work. The strict xfail `test_strangled_bend_port_bases_are_equal` passes; marker removed.

## Tests

* New: tail matrix properties (symmetry, parity blocks, order independence, radius convergence, refusal of incomplete
  lists), X2 power and reciprocity conditions, expm vs sliced route with the tail (1e-8), defaults, constant cases
  against the stored helical-FEM references (fast), cosine/triangular at 18 GHz and τa = 0.5 with mode_factor 6
  (slow); SmoothBend bitwise regression, evanescent route = VariableGuide (bitwise), coupling present, lossy
  passivity, E-plane dead families; Line warning (once, not with evanescent modes, not beyond 3 decay lengths, frame
  changes count as zero length); bend + miter against the FEM domain (fem job, added to `.github/workflows/tests.yml`);
  port bases of StrangledBend/Taper, cascade with itself and reversal (slow).
* Changed because they pinned the old behaviour:
  * `tests/test_twist.py`: the T3/T3b tables, the exact two-mode law, Doane Fig. 9, the near-cutoff study and the
    +90/−90 pair compare with T3/T3b/WP6 numbers of the plain truncated model, so they now say
    `mode_factor=1, tail=False` explicitly; property tests of variable rates use mode_factor 1.5 (cost); the timing
    test has a WP6-model guard (2 s / 20 s) and a WF2-default guard (60 s with WRMODES_PERF=1, 240 s otherwise).
  * `tests/adversarial/test_wp7b_adversarial.py::test_360_degree_twist_is_two_180_degree_twists` (WP7b): the halves
    are joined in a basis that carries the internal modes (the twist drops them at its ports, so a propagating-only
    junction is not exact any more); the property itself is unchanged.
  * `tests/adversarial/test_wp1_adversarial.py::test_evanescent_modes_through_a_long_straight_do_not_overflow` (WP7):
    its clause "the propagating block must not change" (WP1 card) is superseded by WF2; it now bounds the change
    (TE1,0 column 1e-6, block 1e-3; measured 3e-9 / 1.0e-4).
  * `tests/test_components.py` and `tests/test_line.py`: the WP1 tests that required evanescent modes to be
    uncoupled now check that they are coupled and bound the change of the propagating block.
* Full fast suite: see the PR (run at the end of this card). Slow tests of the touched files: `test_twist.py` 6,
  `test_taper.py` and `test_wp7b_adversarial.py` slow tests pass.

## Open issues

1. **Cost of variable-rate twists** at the new default (15–45 s per frequency for Doane's guide; minutes for larger
   guides). A cheaper integrator (6th-order Magnus, or grouping slices into transfer-matrix products within the
   growth limit, ≈ 25 %) or an elimination accurate enough at mode_factor 1.5 would cut it 3–10×; not done.
2. **τa ≥ 0.5** needs mode_factor 4–6 for 1e-4; the remainder of the tail elimination for strong twists is the
   coupling of the outside modes among themselves (order τ/|β_q|), not included.
3. **Cost of SmoothBend with evanescent modes** for long bends at high frequency (110 s for a 40 cm bend at 40 GHz,
   cutoff_factor 3). Only lines that ask for evanescent modes pay it.
4. The Line warning fires for every direct junction at cutoff_factor 1, also where the error is small (e.g. a sech
   bend, whose curvature at its ends is 1/cosh(T) of its peak, next to a twist). It does not estimate the error.
5. WP6's stored E4 predictions (`benchmarks/twist/data/e4_predictions.npz`) are WP6-model values, up to 1.6e-4 off
   WP7b's helical-FEM predictions (cosine, 18 GHz, TE0,1); the new defaults agree with the latter to ≤ 2.3e-5 per
   mode at the E4 points. `compare_campaign_2.py` computes with the current defaults; the stored WP6 file is left as
   WP6's record.

## Proposals for CONVENTIONS.md (not changed here)

* §3, last bullet: replace the cutoff_factor advice for twists by: "components that couple families or whose
  coupled-mode series converges like 1/χ_max solve on an internal list larger than the port basis (`mode_factor`)
  and eliminate the modes outside it quasi-statically where the theory supports it (T2b B5′ for tapers, WF2 for
  twists); the line's cutoff_factor only sets the port bases."
* §8, zero-length cascades: add "two non-uniform components closer than 3 decay lengths of the least-attenuated
  evanescent mode are joined directly; `Line` warns when the basis there has no evanescent modes."
* §10: add "a component given port bases with evanescent modes couples them (no component passes evanescent modes
  as uncoupled straight modes any more, WF2)".

## Decisions for the coordinator or Daniel

1. Accept the new Twist defaults (mode_factor 3 / 2 with the tail) and their cost (169 s for Doane's 9 GSMs), or
   prefer a cheaper default for variable rates with the documented error (mode_factor 1.5: ≤ 1.1e-4 over Doane's
   cosine and triangular points, about 4× faster; 1 with the tail: 1.5e-4 for the triangular twist at 18 GHz, other
   points not measured) until a faster integrator exists.
2. τa = 0.5: accept "reached with mode_factor 4 (8e-5) or 6 (2e-5), default 1.9e-4, documented", as the card allows.
3. The changes to two verifier-owned tests (WP7b's 360° twist, WP7's WP1 evanescent clause) listed above.
4. Whether the `EvanescentJunctionWarning` should stay a `UserWarning` (shown once per line) or become opt-in.
5. The CONVENTIONS proposals above.
