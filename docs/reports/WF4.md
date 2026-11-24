# WF4 — Open items before the 1.0.0 release: report

Branch `wave-3/WF4-release-fixes`, from `main` at a1943b4. Card `tasks/WF4-release-fixes.md` (items 1–7, with P0's
needs copied into item 6 and the demonstration line given at launch). Times: 2-core container (Intel Xeon @ 2.30 GHz,
7.8 GB, OpenBLAS 0.3.34), one BLAS thread unless stated, no other job running for the timing tables.

## What was done

| Item | Files |
| --- | --- |
| 1. Variable-rate twists 7× faster | `src/wrmodes/components/twist.py` (`coupled_groups`, `_Blocks`, `_group_products`, `ACCEPT_MARGIN`); `benchmarks/twist/wf4_speed.py` → `data/wf4_twist_baseline.npz`, `data/wf4_twist_compare.npz`; tests in `tests/test_twist.py` |
| 2. `LossyWallFilter` at 320 modes per family | `src/wrmodes/filters.py` (`lsu_quadrature_points`); tests in `tests/test_wf4.py` |
| 3. `buried` through `GSM.reversed()` | `src/wrmodes/gsm.py`; test in `tests/test_line.py` |
| 4. Registry: final statuses, `check_registry.py`, CI | `benchmarks/registry.json` (73 → 83 entries), `benchmarks/check_registry.py`, `.github/workflows/tests.yml` (fem job); tests in `tests/test_wf4_tools.py` |
| 5, 6. Timing harness, COMSOL cases, log parser | `benchmarks/timings.py`; outputs `benchmarks/data/timings_wf4_container_{1thread,allthreads}.{json,md}`, `timings_comsol_wf4_container.{json,md}` |
| 6. One convergence record | `src/wrmodes/convergence.py` (`ConvergenceRecord`, `build_record`); `Twist`, `Taper`/`VariableGuide`/`StrangledBend`, `MiterMM`, `Step` and `modematch.convergence_study`, `fem2d.convergence_study` |
| 6. `SmoothBend.convergence_study` (evanescent basis) | `src/wrmodes/components/bends.py` |
| 6. `Line.internal_amplitudes`, `InternalAmplitudes` | `src/wrmodes/line.py` (`_prepare` factored out of `gsm`, no change of results) |
| 6. FEM determinism | `benchmarks/fem2d/determinism.py` → `data/determinism_<gmsh>.npz` (4 versions); `src/wrmodes/fem2d/__init__.py` docstring; `pyproject.toml` (`gmsh>=4.10`); test in `tests/test_fem2d.py` |
| 6. `SweepResult.absorbed_power` | `src/wrmodes/line.py` |
| 6. FD H-plane step value as a data file | `benchmarks/modematch/fd_hplane_step_review.py` → `data/fd_hplane_step_review.npz`; `tests/test_steps.py` reads it |
| 7. Doane Fig. 9 with Doane's five modes | `benchmarks/twist/doane_fig9_five_modes.py` → `data/doane_fig9_five_modes.npz`; addendum to the Resolution of `docs/discrepancies/2026-11-12-doane-fig9-twists.md`; registry `wf4-doane-fig9-five-modes`; test in `tests/test_twist.py` |

## 1. Run time of variable-rate twists

**Profile** (before WF4, Doane cosine twist at 18 GHz, defaults, 31.8 s). The internal list (mode_factor 2) has
evanescent modes, so the twist takes the GSM route: 96 modes, 192 × 192 slice matrices, adaptive levels of 162, 324,
648 and 1296 slices, i.e. 2430 slice exponentials. Time: `expm` 19.6 s (62 %), star products 7.7 s (24 %, of which
LAPACK solves 4.9 s), conversion of every slice to a GSM 2.4 s (7 %), forming the Magnus exponents 1.9 s (6 %), tail
matrix 0.05 s. The last level was needed only to confirm the asymptotic ratio: its estimate (1.6e-8 at 648 slices)
was already 60 times below atol = 1e-6.

**Changes** (each exact or below the integration error):

1. **Parity groups** (`coupled_groups`). The selection rule of K (m + m′ and n + n′ odd) couples only
   {(even, even), (odd, odd)} with each other and {(even, odd), (odd, even)} with each other; the tail Q (block
   diagonal in the four parity classes) and the wall loss (which keeps both parities) do not join them. Each group
   (47 and 49 modes at 18 GHz) is solved with the same slicing; the blocks between groups are zero in exact
   arithmetic. Cost ∝ N³: 31.8 → 9.1 s.
2. **Grouped slice products** (`_group_products`). On the GSM route, runs of consecutive slices of total length
   ≤ hmax (|Im β| hmax ≤ 4, ‖Λ‖ hmax ≤ 0.5, the cap a single slice already obeys) are multiplied as transfer matrices
   and converted once; the star products fall from 24 % to 5 % of the time.
3. **Acceptance** (`ACCEPT_MARGIN` = 0.1). A level is accepted after one halving ratio ≥ 8 when its Richardson
   estimate is ≤ 0.1 atol (before: always two ratios). This saves the last doubling (1296 → 648 slices); the
   results move by the integration error of the 648-slice level, ≈ 1e-8 in amplitude. 9.1 → 4.5 s.

Tried and dropped: a 6th-order Magnus integrator (three Gauss points, Blanes–Casas–Ros form). Its slices cost 1.6
times more, and at the coarse levels it is only 3 times more accurate (1.2e-6 against 3.2e-6 at 162 slices; 2.3e-8
against 2.5e-7 at 324): the coarse levels are limited by the phase slip of the fast modes over a slice, not by the
order. Reusing slice exponentials does not apply: the Magnus exponent of each slice is a different combination of
D, X, X2 and their commutators.

**Result** (`benchmarks/twist/wf4_speed.py compare`; both runs on the idle container, the baseline with the code of
a1943b4; ΔP = largest change of any |S_pq|² of the propagating GSM, every input mode, transmitted and reflected;
FEM = worst power difference per mode against WP7b's helical FEM, incident TE1,0):

| case | f (GHz) | before (s) | WF4 (s) | speed-up | slices | max ΔP | max ΔP, TE1,0 column | FEM before → after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cosine | 15 | 12.37 | 2.03 | 6.1 | 1120 → 560 | 4.9e-10 | 1.7e-11 | 2.386e-6 → 2.386e-6 |
| cosine | 16.5 | 18.32 | 2.58 | 7.1 | 1200 → 600 | 3.7e-10 | 2.3e-12 | 1.331e-6 → 1.331e-6 |
| cosine | 18 | 32.28 | 4.49 | 7.2 | 1296 → 648 | 2.9e-10 | 1.0e-11 | 1.177e-5 → 1.177e-5 |
| triangular | 15 | 12.60 | 1.73 | 7.3 | 1120 → 560 | 8.2e-10 | 4.3e-11 | 5.742e-6 → 5.742e-6 |
| triangular | 16.5 | 17.58 | 2.78 | 6.3 | 1216 → 608 | 4.8e-10 | 9.1e-12 | 1.357e-5 → 1.357e-5 |
| triangular | 18 | 32.34 | 4.45 | 7.3 | 1312 → 656 | 4.2e-10 | 2.7e-11 | 2.275e-5 → 2.275e-5 |
| constant | 15 / 16.5 / 18 | 0.25 / 0.37 / 0.55 | 0.18 / 0.24 / 0.40 | 1.4–1.6 | 128 / 128 / 256 | ≤ 1.8e-13 | ≤ 3.2e-14 | unchanged |

* **Speed-up at the defaults: 6.9× (cosine, 63.0 → 9.1 s over the three frequencies) and 7.0× (triangular,
  62.5 → 9.0 s)**, against the card's 3×. **Worst power change 8.2e-10** (bound 2e-6). The comparison with the
  helical FEM is unchanged to four digits.
* Constant rates: one matrix exponential per group; the tail matrix and the exponentials of the two half-size
  systems now cost about the same.
* Docstrings: `Twist` (Cost), `twist_gsm_blocks` (Method), `ACCEPT_MARGIN`; `info` carries `n_groups`.
* Tests: the groups are the parity classes and the group-wise solution equals the whole-system solution, with and
  without grouping of slices (lossy, tail, 1e-12); the defaults against the stored baseline within 2e-6 (15 GHz fast,
  16.5 and 18 GHz slow); the timing guard tightened (constant + triangular at 18 GHz: 15 s with WRMODES_PERF=1, 60 s
  otherwise; was 60 / 240 s).

## 2. `LossyWallFilter` at 320 modes per family

**Cause: the quadrature across the guide, not the root search.** The reaction and overlap integrands are products
of two profiles with up to M half-periods each (wavenumber κ = πM on the reference interval); an n-point Gauss rule
needs n ≳ κ/2 + O(κ^{1/3}). The old rule 1.5M + 48 gives 528 points at M = 320, where πM/2 = 503 plus the margin
is ≈ 540: the loaded modes of family n = 4 failed the orthogonality check (1.1e-6 > ORTHO_TOL = 1e-6) and the filter
raised RootFindingError. Families 0–3 had passed, and the roots themselves were fine.

**Fix**: `lsu_quadrature_points(M)` = ⌈πM/2 + 4(πM)^{1/3}⌉ + 32 (183 / 316 / 575 points at M = 80 / 160 / 320).
M = 320 now works: Butterweck's filter (WR-90, 75 GHz, 250 Ω, 15 cm, propagating basis) in 468 s and 2.9 GB peak
memory on one core; the attenuation of every port mode changes by at most 0.24 dB (0.33 %) from M = 160 to 320. Up
to M = 160 the quadrature change moves the filter by 2e-11 dB (1.5M + 48 against 2M + 48 points), so WP7b's stored
M = 80/120/160 references are unchanged. The docstring states the tested range and the cost (cost ∝ M³, memory ∝ M²
per family). Tests: the rule integrates products of profiles with M − 3 … M half-periods to 1e-12 for M = 80–640,
while the old rule fails at 320 (fast); the filter at M = 160 and 320 (slow, ≈ 9 min).

## 3. `buried` through `GSM.reversed()`

`GSM.reversed()` now returns `buried = BuriedPlanes((depth2, depth1), buried.gsm.reversed())`. Star products keep
dropping it (documented in `GSM`, `GSM.reversed` and the test). Test: a component with buried planes (MiterMM with
depths (a, 0.4a)) turned around: depths swapped, the identity g = shift_planes(g_u, δ2, δ1), double reversal, and
`Line([R, Straight(s), R])` at cutoff factor 3 equal to the cascade at the uniform planes (1e-13), as the original;
it fails on the old `gsm.py`.

## 4. Registry

Every entry now has `data_file` (list of files), `test` (pytest node ids), `script` (list), `status` in a
documented vocabulary and `note` (free text, with the earlier status text kept after "Before WF4:"); the earlier
free text of the file and test fields is kept in `files_note` / `test_note` where it said more than the lists.
Vocabulary (`status_vocabulary` in the file): pass, not met, qualitative, superseded, reference.

* Statuses: 60 pass, 11 qualitative, 7 superseded, 4 reference, 1 not met (`wp3-ma1997-figures`: Ma's Fig. 6 at 20
  steps, rms z 1.69, which the resolution of the Reiter/Ma discrepancy keeps pass/fail; every other part of that
  entry is pass or qualitative). No entry says "waiting for data" or "predictions ready".
* New entries: the Gate A results `gatea-comsol-c1`, `-c1b`, `-c2`, `-c3a` (pass) and `-c3b` (qualitative: the
  resonator explained by a 1° round-trip phase); `fem2d-marcuvitz-529b-corners` (qualitative, Marcuvitz's measured
  corners); `wp4-review-fd-hplane-step` (pass; the data file of item 6); `wf4-doane-fig9-five-modes` (pass);
  `wf4-twist-speed`, `wf4-fem-determinism` (pass).
* Final statuses asked for by the card: `wp4-marcuvitz-eplane-step` pass; `wp4-marcuvitz-hplane-step` superseded by
  `wp7b-marcuvitz-524a` and `wp4-review-fd-hplane-step`; `wave2-alessandri1994` pass with b/a = 0.5;
  `wp5-comsol-e2`, `wp5-comsol-e3`, `wp6-e4-twist` superseded by `gateb-comsol-e2/e3/e4`;
  `wp7b-campaign2-postprocess` pass; `wp3-comsol-c1`, `wp2-comsol-c2`, `wp4-comsol-c3` superseded by the Gate A
  entries (their WP7 data files were never produced and are no longer listed). `wp6-doane-fig9` is qualitative (the
  Gate B resolution), with the five-mode entry as its pass.
* `benchmarks/check_registry.py`: fields, unique ids, reference type, status vocabulary, non-empty note, a
  superseded entry names an existing one, every data file and script exists, every test id is collected by pytest.
  It runs in the fem job of CI (all optional dependencies, so every module is collected): 83 entries, 0 problems.

## 5, 6. Timing harness (`benchmarks/timings.py`)

JSON (and a markdown copy) with the machine (CPU model, logical and physical cores, RAM, OS, Python, NumPy, SciPy,
BLAS vendor, version and threads per library, wrmodes version and commit; no host name) and one record per item.
One BLAS thread by default, `--threads all` (or N); `--repeat`, `--quick`, `--only`. `Twist.gsm` and the
propagating route of `SmoothBend` limit BLAS to one thread themselves (`self_limited` in the records). No POSIX-only
calls (CPU model and RAM by /proc, sysctl or the Windows registry and `GlobalMemoryStatusEx`; psutil if present), no
multiprocessing, so it runs the same on Windows.

**This machine** (one BLAS thread; `--threads all` on two cores gives the same times within ±10 %, these matrices
are too small for threaded BLAS):

| item | modes (port 1, 2) | s |
| --- | --- | --- |
| lossy straight 1 m, WR-90 Al, all 129 propagating modes, 90 GHz | 129, 129 | 0.077 |
| SmoothBend H sech 90°, 20 cm, WR-90 Al, 40 GHz: propagating route / evanescent route (basis 1.5 f) | 26 / 57 | 0.18 / 2.2 |
| Taper raised cosine WR-90 → 22.86 × 1.6 mm, 5 cm, 90 GHz | 129, 13 | 3.6 |
| StrangledBend E3 geometry, 40 / 90 GHz | 26 / 129 | 1.1 / 17.8 |
| Twist Doane 18 GHz, constant / triangular (defaults) | 23 | 0.30 / 4.4 |
| MiterMM H, 30 / 60 / 90 GHz | 12 / 57 / 129 | 0.048 / 0.20 / 0.54 |
| MiterMM E, 30 / 60 / 90 GHz | 12 / 57 / 129 | 0.072 / 0.22 / 0.55 |
| fem2d.Miter H 90°, 90 GHz, one family / full basis | 13 / 129 | 0.55 / 3.8 |
| Step WR-90 → 22.86 × 1.6 mm, 90 GHz, parity on / off | 3415, 526 | 0.29 / 2.4 |
| LossyWallFilter 15 cm, 250 Ω, M = 80, 75 GHz | 94 | 6.4 |
| demonstration line, perfect walls / copper, per frequency (16 frequencies 15–30 GHz) | | 0.58 / 0.84 (0.20–1.27 / 0.40–1.75) |

**Demonstration line** (the coordinator's, generic; WR-62 15.799 × 7.899 mm): straight 1 m → H-plane sech bend 90°,
30 cm → straight 0.5 m → constant-rate 90° twist, 15 cm → straight 0.3 m → E-plane 90° chamfered miter → straight
0.2 m → linear H-plane taper to 22.86 × 10.16 mm over 10 cm → straight 2 m → two 45° optical H-plane miters with
spacer 3a (a = 22.86 mm) → straight 0.5 m. Choices: the chamfer is d/h = 0.87, COMSOL E1's validated case; with
perfect walls the E-plane miter is `MiterMM` and the 45° miters are `fem2d.Miter` (MiterMM is 90° only); with
copper every miter is `fem2d.Miter` with the guide's wall impedance (a lossy `MiterMM` does not exist and `Line`
refuses MiterMM in a lossy guide). Default cutoff factor 1: `Line` issues its `EvanescentJunctionWarning` (recorded
in the JSON); a study would use cutoff_factor ≥ 1.5 at the 45° pair.

**COMSOL cases** (`--comsol-cases`): C1, C1b, C2, C3a, C3b (campaign 1) and E1–E5 (campaign 2), at the frequencies and
with the components and settings of `compare_campaign_1.py` / `compare_campaign_2.py` (campaign 2 through its
`model` hook, so the settings cannot drift), one BLAS thread and all threads. The accuracy columns are the largest
|S21| difference (dB; for steps also |S11|; E4: power per mode) against the finest COMSOL run, and COMSOL's
mesh-check change or the allowance. The log parser (`parse_comsol_log`) reads blocks `<run>:` with COMSOL's
"Number of degrees of freedom solved for" (several = a parameter sweep), "Solution time (Study n)" and "Complete
mesh consists of" lines, and matches each run to its export in `raw/` (by name; `E4_twist_triangle` →
`E4_twist_triangular.txt`) for the frequencies and parameter combinations, i.e. the number of solves; a case
without a log entry prints "–". Further logs in the same format: `--comsol-log FILE`. Container results
(`benchmarks/data/timings_comsol_wf4_container.md`; COMSOL times are Daniel's machine, wrmodes times this one, so the
ratio is only indicative until Daniel runs it on the COMSOL machine):

| case | COMSOL: DOF, s per solve (coarse / fine) | wrmodes s per frequency (1 thread) | max \|Δ\| vs COMSOL finest |
| --- | --- | --- | --- |
| C1 90° H miter (fem2d) | – (log to come) | 1.1 | 0.046 dB (the 4 near-cutoff points of Gate A) |
| C1b 60° / 120° | – | 3.1 / 3.8 | 0.002 / 0.005 dB |
| C2 sech / constant E bends, Al | – | 0.12 / 0.08 | 0.007 / 0.005 dB |
| C3a steps h = 2.0 / 1.6 mm | – | 0.003 / 0.004 | \|S21\| 0.005, \|S11\| ≤ 0.023 dB |
| C3b strangled section | – | 0.05 | \|S11\| 0.11 dB (\|S21\| 2.1 dB at the resonance; qualitative) |
| E1 E-plane miters plain / d = 0.87 (FEM; MM) | 0.40 / 3.3 M DOF, 5.8–6.2 / 41–43 s | 0.06–0.07; MM 0.03–0.10 | 0.008 / 0.019 dB |
| E2 linear taper | 27–164 k DOF, 2.3 s | 0.03–0.04 | 0.0000 dB |
| E3 strangled bend PEC / Al | 2.1 / 3.6 M DOF, 30–35 / 53 s | 0.71 / 0.78 | 0.0007 / 0.0033 dB |
| E4 twists triangular / cosine | 1.75 / 14.4 M DOF, 29–33 / 220 s | 3.1 / 3.1 | 3e-4 / 1.8e-3 of the input power (worst output mode, main mode included; within the allowance of COMSOL's power balance) |
| E5 two 45° miters, domain / cascade cf 1.5 / cf 3 | 0.21–0.97 M DOF, 4.5–5.3 s | 1.6–1.9 / 4.8–5.5 / 105–111 | ≤ 0.0008 / 0.0012 / 0.0002 dB |

## 6. P0's other needs

* **One convergence record** (`wrmodes.convergence`). Every `convergence_study` returns a `ConvergenceRecord` with
  `component`, `f`, `input_mode`, and per row `parameter` (mode_factor, cutoff_factor, n_modes, mesh_density,
  n_evanescent, slices), `values`, `variants` (e.g. {'tail': True}), `n_modes`, `powers` ({'transmitted',
  'reflected'} per mode, for the input column), `max_change` (against the finest row of the same parameter and
  variant), `wall_time`; `richardson` where the series supports it (Twist: plain series in 1/χ_max, order 1;
  Taper: in 1/M, order 1; None for the others, whose series converge faster than a fixed power or have no
  established order), and `reference`. The earlier keys stay in the same dict as aliases for one release (listed in
  each docstring); two needed care: Taper's old `richardson` {tail: S} entries are kept inside the new
  `richardson` dict, and MiterMM's / Step's old `n_modes` arrays are the record's `n_modes` (one value or pair per
  row). One existing test changed: `test_taper.py::test_convergence_study_report` now checks
  `{True, False} <= set(r["richardson"])` instead of equality, plus the new keys.
* **`SmoothBend.convergence_study`**: port basis to 1.2 f (or to the first evanescent mode if that adds none, so
  that the evanescent route is taken), internal mode factor 1.5–8 (default 1.5, 2, 3, 4, 6, 8), plus a row of the
  propagating route. Example: 10 cm sech 90° H bend, WR-90, 30 GHz: max change against mode factor 8 is 1.0e-5 /
  4.7e-6 / 2.1e-6 / 9.5e-7 / 2.1e-7 at 1.5 / 2 / 3 / 4 / 6, about 1/mode_factor² (0.1–7 s per row).
* **`Line.internal_amplitudes(f, input_mode, ...)`**: forward and backward amplitudes at every junction for unit
  incidence, a+ = (I − L22 R11)⁻¹ L21 a_in, a− = R11 a+, from the partial cascades from both ends (≈ 2× `gsm`), with
  the GSMs of `gsm` (buried planes absorbed); `InternalAmplitudes.largest(n)` lists the largest amplitudes with
  junction, neighbouring elements, mode and direction. The docstring gives the recipe for naming a trapped-mode
  resonance. Tests: port amplitudes equal the GSM column (1e-13), net forward power at every junction equals the
  transmitted power (1e-12), and on C3b's resonator the largest inner amplitude at a transmission resonance is
  TE1,0 inside the strangled section, > 1.
* **FEM determinism**: gmsh 4.10.5, 4.12.2, 4.13.1 and 4.15.2 give bitwise identical GSMs for the four test cases
  (90° H miter, chamfered E miter, 45° optical miter, two chained 45° miters; Linux x86-64, scikit-fem 12.0.2): the
  mesh is generated with a fixed algorithm and seed. A random renumbering of the mesh nodes changes S by ≤ 4e-14, a
  repeated solve is bitwise equal; a 1.5× finer mesh changes the powers by up to 2.5e-4 (chained miters) or 5e-5
  (single miters), the scale of the discretization error. **Figure scripts may promise FEM results to 1e-10 in
  every entry of S** across gmsh ≥ 4.10 and platforms; `pyproject.toml` now requires `gmsh>=4.10`; a CI test (fem
  job) checks a stored GSM to 1e-10. Not measured: macOS and Windows (no such runner for the fem job).
* **`SweepResult.absorbed_power`** = 1 − Σ|S|² over the input column (wall loss); NaN where the input mode does not
  propagate. Test: lossless line < 1e-12; 1 m aluminium WR-90 straight = 1 − e^(−2αL) to 7e-6 (the rest is the
  loss coupling to other modes).
* **FD H-plane step value** (X/Z0 = 11.07 ± 0.05 at a/λ = 0.85, 2aZ0/(λg X) = 0.1077 ± 0.0003 at a/λ = 1.4,
  a′/a = 0.8): `benchmarks/modematch/data/fd_hplane_step_review.npz` with meta; the test reads it.

## 7. Doane & Anderson Fig. 9 with Doane's five modes

`benchmarks/twist/doane_fig9_five_modes.py`: the three 20-inch twists on the 119-point grid, plain truncated system
(`twist_gsm_blocks`) on TE1,0, TE0,1, TE0,3, TE1,2, TM1,2 (Doane's TE10, TE01, TE30, TE21, TM21), and with TE2,1
(his TE12) added; z = (ours − Fig. 9)/uncertainty:

| profile | five modes: within 1σ / rms z / max \|z\| | five + TE2,1 |
| --- | --- | --- |
| constant | 58.8 % / 1.17 / 2.2 | 7.6 % / 37.4 / 81.3 |
| cosine | 75.6 % / 1.04 / 2.3 | 46.2 % / 5.0 / 9.5 |
| triangular | 100 % / 0.52 / 1.0 | 0 % / 17.1 / 29.4 |

Triangular at 15.03 / 16 / 16.5 / 17 / 17.98 GHz: five modes 0.044 / 0.078 / 0.098 / 0.113 / 0.137 %, Fig. 9
0.049 / 0.080 / 0.099 / 0.116 / 0.149 %, with TE2,1 0.114 / 0.188 / 0.240 / 0.310 / 0.543 %, full solution at the
defaults 0.115 / 0.196 / 0.254 / 0.333 / 0.590 %. The coordinator's check is reproduced. All three curves of Fig. 9
are a five-mode calculation; the difference from the full-wave result is the power converted to TE2,1.

## Tests

* Full suite, slow tests included (as CI runs it), scikit-fem and gmsh installed: **787 passed, 4 skipped, 1 xfailed
  (WP3's strict xfail, unrelated) in 19 min** on the 2-core container. The fem job's files
  (`tests/test_fem2d.py tests/test_bend_evanescent_fem.py tests/adversarial/test_wp3_adversarial.py
  tests/adversarial/test_wp7b_fem_adversarial.py`) with the frozen tests: 121 passed, 1 skipped, 1 xfailed. The
  adversarial tests are part of the full run.
* Frozen tests unchanged and passing: `tests/test_components.py::test_propagating_path_is_bitwise_unchanged`,
  `tests/test_regression_v0_1.py`, the WF3 frozen lines (`tests/data/` untouched).
* `python benchmarks/check_registry.py`: 83 entries, 0 problems (also in CI, fem job).
* New tests: `tests/test_wf4.py` (records of all six studies, SmoothBend study, internal amplitudes, absorbed power,
  filter quadrature; the M = 320 filter run is slow and needs `WRMODES_LONG=1`, about 9 min and 3 GB: passed here),
  `tests/test_wf4_tools.py` (registry check, log parser, harness), and in `test_twist.py`, `test_line.py`,
  `test_fem2d.py`, `test_steps.py` as listed per item.
* Changed: `tests/test_taper.py::test_convergence_study_report` (the old `richardson` keys are a subset now),
  `tests/test_twist.py::test_timing_doane_twist` (tighter guard), `tests/test_steps.py::
  test_hplane_symmetric_step_against_fd_reference` (reads the new data file; same values).

## Open issues

1. The COMSOL-cases timings must be run on the COMSOL machine (Daniel); campaign 1 has no log yet, its rows print
   "–" until a log is given with `--comsol-log`.
2. FEM determinism is measured on Linux x86-64 only; macOS/Windows differ only by BLAS/sparse round-off in
   principle (not measured; the fem job runs on Linux).
3. `LossyWallFilter` beyond M = 320 is untested (memory ≈ M² per family: about 12 GB at 640); the docstring states
   the tested range.
4. `SmoothBend.convergence_study` and the evanescent route cost grows fast with the mode factor (7 s at 8 for a
   10 cm bend at 30 GHz; WF2 open issue 3 unchanged).
5. `wp3-ma1997-figures` stays "not met" for Ma's Fig. 6 at 20 steps (rms z 1.69), per the resolution that keeps the
   20-step curve pass/fail; Daniel may want to make it qualitative like the 40/60-step curves.
6. `Twist` and the propagating route of `SmoothBend` limit BLAS to one thread inside `gsm`, so `--threads all`
   does not speed them up (by design: results independent of the thread count).
7. Not in scope and unchanged: WF2 open issues 2–5 (strong twists, SmoothBend evanescent cost, the junction
   warning's error estimate, WP6's stored E4 predictions), WF3 open issues 1–5.
