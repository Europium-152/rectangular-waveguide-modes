# WP3 — 2D modal-port FEM solver (wave 1: H-plane miter): report

Branch `wave-1/WP3-2d-solver`, from `main` at 223aeeb. Time box: within budget. Status: done, except the Ma et al.
comparison (needs L1's digitized data) and the performance target (missed at default settings, see below).

## What was done

| Item | Files |
| --- | --- |
| Component stub (byte-identical to the briefing) | `src/wrmodes/components/__init__.py`, `base.py` |
| Geometry: outlines of straight segments and arcs with two port planes; gmsh meshing of core + arms; mirror-symmetric meshes; re-entrant corner refinement | `src/wrmodes/fem2d/geometry.py` |
| Scalar solver: P2 (or P3) Lagrange elements with scikit-fem, isoparametric P2 geometry on arcs, modal (DtN) ports at the end of straight arms, complex-symmetric system, SuperLU | `src/wrmodes/fem2d/solver.py` |
| Normalization 2D amplitude → LS → TE/TM, computed from `transverse_field`, `admittance` and `ls_rotation` | `src/wrmodes/fem2d/normalization.py`, `docs/theory/WP3-normalization.md` |
| Components `Coplanar2D(outline, b, frame, mirror, **options)` and `Miter(angle, chamfer=None, plane='H', *, guide, **options)`, `FEMOptions`, outline helpers (`miter_outline`, `arc_bend_outline`, `straight_outline`) | `src/wrmodes/fem2d/components.py` |
| `convergence_study(component, f, mesh_densities, n_evanescent)` | `src/wrmodes/fem2d/convergence.py` |
| Tests (29, skipped without the extra) | `tests/test_fem2d.py` |
| Example: WR-90 H-plane miter loss curve | `examples/h_plane_miter.py` → `benchmarks/fem2d/wr90_h_miter_loss.npz` |
| Ma et al. 1997 geometries computed for the L1/WP7 comparison | `benchmarks/fem2d_ma1997.py` → `benchmarks/fem2d/ma1997_wp3.npz` |

`import wrmodes` does not import scikit-fem or gmsh (tested); `import wrmodes.fem2d` without them raises an
ImportError naming the extra and the system libraries. `pyproject.toml` is unchanged (the existing `fem` extra
`["scikit-fem", "gmsh"]` is what the code uses). `src/wrmodes/__init__.py` is unchanged: the public names live in
`wrmodes.fem2d`, so that the core imports without the extra.

### Method in brief

* Per family $n$ and polarization (LSM: $E_v$, Dirichlet; LSE: $H_v$, Neumann), $\nabla^2\psi + k_t^2\psi = 0$ in the
  outline plus two straight arms of length `port_distance` (default: one guide width, as on the card). At the arm
  ends, a modal condition with all propagating port modes plus $M$ evanescent ones (`n_evanescent`, default 10).
  Incident modes enter through these ports; $S^{2D}_{ij} = 2i\beta_j b_i^TA^{-1}b_j - \delta_{ij}$ with the symmetric
  matrix $A$, so reciprocity holds to rounding and, the discrete system being lossless, so does unitarity.
* Shift to the reference planes with $e^{i\beta L}$ per index, then 2D → LS amplitudes with the factors $g$ (LSM)
  and $h$ (LSE, with a sign change of the reflections), then $S = Q^TS^{LS}Q$ per port. Derivation:
  `docs/theory/WP3-normalization.md`.
* One mesh per family, sized $h = \lambda_t/10$ for that family (at most $a/4$), refined to $0.05h$ at re-entrant
  corners. Miters are meshed on one half and mirrored, so that $S_{11} = S_{22}$ holds to rounding.
* Miter geometry (CONVENTIONS.md §2, §6): turns towards $-u$, arms' axes meet at the origin, inner corner
  $O = (-a/2, -(a/2)\tan(\theta/2))$, reference planes through $O$ at `arm_length` $= (a/2)\tan(\theta/2)$, port 2's
  $u$ on the outer side. `chamfer` is Alessandri's $d/a$ (distance from $O$ to the mirror along the bisector);
  `None` = mirror through the outer-wall points of the two reference planes, $d = \cos(\theta/2)$; $d \ge
  1/\cos(\theta/2)$ = sharp corner. For $d < \cos(\theta/2)$ the mirror cuts the outer walls before the reference
  planes; the field is then solved from the last uniform cross-section and shifted analytically to the reference
  planes (`PortPlane.deembed`).

### Design decisions worth knowing

* **Excitation.** I first implemented a total-field/scattered-field injection at the reference planes with a
  reaction (reciprocity) read-out, to avoid de-embedding evanescent modes over the arms. It was reciprocal to
  rounding but its errors were large (straight channel: 1.3e-4 with evanescent modes vs 4e-6 with modal ports;
  unitarity 1e-4), so it was removed. Evanescent modes requested in a basis are de-embedded over the arms; their
  entries carry the discretization error times $e^{2|\beta|L_p}$ (documented). The default basis has propagating
  modes only (`cutoff_factor = 1`); the solver itself does not need evanescent modes in the returned basis.
* **Mesher.** gmsh, as the card asks. The pip wheel needs `libGLU.so.1` and `libXft.so.2`; in this container
  `apt-get install libglu1-mesa libxft2` made it work. No fallback mesher was needed. The error message of
  `wrmodes.fem2d` names these libraries.

## Results

### WR-90 plain 90° H-plane miter, 18–90 GHz

Loss $= -20\log_{10}|S_{21}|$, TE10 → TE10 (conversion to TE$_{m0}$ plus reflection, perfect walls), default
settings (P2, 10 elements per wavelength, arms of one guide width, $M = 10$), family $n = 0$ (TE10 couples only to
it). Convergence columns: same points with P2 at 15 elements per wavelength, and P3 at 10.

| f (GHz) | f/fc | loss (dB) | \|S11\|² | P2, d = 15 | P3, d = 10 |
| --- | --- | --- | --- | --- | --- |
| 18 | 2.75 | 0.8317 | 2.45e-02 | 0.83100 | 0.83078 |
| 20 | 3.05 | 0.4187 | 6.06e-03 | | |
| 22 | 3.36 | 0.7332 | 2.34e-03 | | |
| 24 | 3.66 | 0.7146 | 1.51e-03 | 0.71472 | 0.71477 |
| 26 | 3.97 | 1.1055 | 7.48e-03 | | |
| 28 | 4.27 | 0.9056 | 3.28e-03 | | |
| 30 | 4.58 | 0.4501 | 1.41e-03 | 0.44991 | 0.44986 |
| 32 | 4.88 | 0.2752 | 9.98e-04 | | |
| 34 | 5.19 | 0.4267 | 7.65e-04 | | |
| 36 | 5.49 | 0.3993 | 4.72e-04 | 0.39936 | 0.39939 |
| 38 | 5.80 | 0.4173 | 4.40e-04 | | |
| 40 | 6.10 | 0.5699 | 7.13e-04 | | |
| 42 | 6.41 | 0.3833 | 4.61e-04 | 0.38317 | 0.38315 |
| 44 | 6.71 | 0.2474 | 4.20e-04 | | |
| 46 | 7.02 | 0.2421 | 5.80e-04 | | |
| 48 | 7.32 | 0.2639 | 2.20e-04 | 0.26395 | 0.26397 |
| 50 | 7.63 | 0.2409 | 1.89e-04 | | |
| 52 | 7.93 | 0.2909 | 5.18e-04 | | |
| 54 | 8.24 | 0.3375 | 3.49e-04 | 0.33733 | 0.33729 |
| 56 | 8.54 | 0.1962 | 1.90e-04 | | |
| 58 | 8.85 | 0.1324 | 1.56e-04 | | |
| 60 | 9.15 | 0.2013 | 1.56e-04 | 0.20126 | 0.20123 |
| 62 | 9.46 | 0.1771 | 1.07e-04 | | |
| 64 | 9.76 | 0.1805 | 1.35e-04 | | |
| 66 | 10.07 | 0.2798 | 2.09e-04 | 0.27988 | 0.27988 |
| 68 | 10.37 | 0.1803 | 8.89e-05 | | |
| 70 | 10.68 | 0.1261 | 7.89e-05 | | |
| 72 | 10.98 | 0.1143 | 1.51e-04 | 0.11447 | 0.11451 |
| 74 | 11.29 | 0.1435 | 6.93e-05 | | |
| 76 | 11.59 | 0.1302 | 5.64e-05 | | |
| 78 | 11.90 | 0.1457 | 1.10e-04 | 0.14576 | 0.14579 |
| 80 | 12.20 | 0.1829 | 9.13e-05 | | |
| 82 | 12.51 | 0.1210 | 5.93e-05 | | |
| 84 | 12.81 | 0.0822 | 5.35e-05 | 0.08218 | 0.08217 |
| 86 | 13.12 | 0.1236 | 6.05e-05 | | |
| 88 | 13.42 | 0.1070 | 3.83e-05 | | |
| 90 | 13.73 | 0.1010 | 4.91e-05 | 0.10099 | 0.10099 |

* The loss peaks just above each TE$_{m0}$ cut-off ($m f_c$, $f_c$ = 6.557 GHz) and falls between them. The data file
  also holds points at $m f_c(1+\delta)$, $\delta$ = −0.01 … +0.04. Highest values: 1.81 dB at 26.24 GHz (just above
  TE40); above 8 $f_c$: **0.471 dB at 52.48 GHz** (0.04 % above the TE80 cut-off), 0.32 dB above TE10,0 (65.60 GHz),
  0.23 dB above TE12,0 (78.73 GHz).
* **Bouwkamp (< 0.4 dB above 8 f_c = 52.5 GHz, Butterweck & de Ronde 1968, p. 97).** The computed loss stays below
  0.4 dB above 8 f_c except in a narrow band 52.46–53.2 GHz right above the TE80 cut-off, where it reaches 0.47 dB
  (0.4706 dB at default settings, 0.4736 with P3, 0.4729 with P2 d = 15: converged to 3e-3 dB). Above 8.1 f_c the
  maximum found is 0.38 dB (53.5 GHz). Our number includes reflection ($|S_{11}|^2 \le 1.2\times10^{-3}$, i.e. at most
  0.005 dB); Bouwkamp's statement is about conversion into H$_{m0}$ modes. Reported, not pass/fail, as the card asks.
* Convergence over the whole curve: the three settings agree to 9e-4 dB at 18 GHz and to 2.1e-4 dB elsewhere.

### Convergence study (`convergence_study`, 90° WR-90 miter, family 0, other settings default)

Changes are relative to the finest run of each series.

| f | elements per λ | ndof | max \|Δ\|S\|\| | max \|ΔS\| (complex) | loss (dB) |
| --- | --- | --- | --- | --- | --- |
| 30 GHz | 8 | 4 335 | 2.2e-4 | 5.5e-3 | 0.45038 |
| | 10 (default) | 6 495 | 1.3e-4 | 2.4e-3 | 0.45007 |
| | 15 | 14 151 | 3.3e-5 | 4.4e-4 | 0.44991 |
| | 20 | 24 677 | 1.4e-5 | 1.2e-4 | 0.44988 |
| | 30 (reference) | 55 171 | 0 | 0 | 0.44986 |
| 90 GHz | 8 | 35 499 | 8.1e-4 | 1.6e-2 | 0.10100 |
| | 10 (default) | 55 171 | 3.1e-4 | 5.7e-3 | 0.10099 |
| | 15 (reference) | 123 311 | 0 | 0 | 0.10099 |

| f | M = 2 | 4 | 6 | 10 (default) | 16 (reference) |
| --- | --- | --- | --- | --- | --- |
| 30 GHz, max \|ΔS\| | 1.1e-8 | 1.1e-8 | 1.0e-8 | 8.7e-9 | 0 |
| 90 GHz, max \|ΔS\| | 8.2e-10 | 8.1e-10 | 7.9e-10 | 7.6e-10 | 0 |

Arm length: $L_p = a/4$ and $a/2$ against $a$ change $|S|$ by at most 6.7e-5 (within the mesh error) and the complex
$S$ by 1.3e-3 (30 GHz) and 3.4e-3 (90 GHz): the phase error of P2 over the arms, which is also why the complex
change under mesh refinement is 20 times the change of $|S|$. With $L_p = a/4$ (30 GHz, P2 at 20 per λ), going from
$M$ = 6 to 12 changes $S_{21}$ by 5e-11.
For phase-accurate work, P3 is much better per unknown (at 90 GHz, P3 with 6 elements per λ: 19 067 unknowns,
complex $S_{21}$ within 1.2e-4 of P3 at 8 per λ; P2 at 10 per λ: 22 551 unknowns, 3.8e-3; both with $L_p = a/4$).

### Ma et al. (1997): geometry and computed values

Geometry, from p. 4 and the figure insets: $a$ = 15.799 mm, $b$ = 7.899 mm, 90° H-plane bend, single-mode band
9.49–19.0 GHz. The outer corner is cut by a mirror whose ends are on the two outer walls at distance $c$ from the outer
corner (Figs. 6, 8, 9 insets); $c/a = 0$ is the sharp corner (Fig. 5), $c/a = 1$ the plain miter (the mirror starts on
the outer wall right above the inner corner). In the card's parameter: $d/a = (2 - c/a)/\sqrt2$. Cases: Fig. 5
($c/a = 0$, $|S_{21}|^2$), Figs. 6, 8, 9 and Table 1 ($c/a = 1$), Fig. 11 ($c/a$ = 0, 0.6, 0.631, 0.7, 1.0, 1.3,
$|S_{11}|$ in dB vs $f/f_c$ = 1–2). All are computed in `benchmarks/fem2d/ma1997_wp3.npz` (complex $S_{11}$, $S_{21}$; 0.1 GHz
steps and $f/f_c$ steps of 0.02) by `benchmarks/fem2d_ma1997.py`. Selected values (default settings; P2 at 20 per λ
changes $|S_{11}|$ at 16 GHz by 3e-5):

* $c/a = 1$, 16 GHz: $|S_{11}|$ = 0.2778, $|S_{21}|$ = 0.9606. Ma's Table 1 (60 steps): 0.2622 and 0.9650; their value
  moved by 0.002 between 50 and 60 steps. A first look only; the comparison is WP7's, with L1's data.
* $c/a = 0$: $|S_{21}|^2$ = 0.634 (10.1 GHz), 0.860 (12.6), 0.834 (14.1), 0.548 (16.1), 0.016 (17.6), 0.205 (18.6).
* Fig. 11, $|S_{11}|$ (dB) at $f/f_c$ = 1.12 / 1.42 / 1.72: $c/a$ = 0: −6.1 / −8.4 / −2.8; 0.6: −11.8 / −21.5 / −43.2;
  0.631: −13.6 / −26.3 / −29.0; 0.7: −22.1 / −29.8 / −19.2; 1.0: −1.6 / −6.0 / −12.0; 1.3: −0.01 / −0.14 / −1.33.

## Acceptance tests

| Check | Target | Result |
| --- | --- | --- |
| Straight channel (no discontinuity), 20 GHz, 8 propagating modes in families 0 and 1 | S = `gsm.straight` to 1e-6, signs included | **pass**: 5.3e-7 (P3, 15 per λ, $L_p = a/4$). With evanescent modes (10 modes, P3 at 20 per λ): 4.4e-6 (extra test, bound 1e-5). At default settings the error is 6.7e-3 (P2 phase error over 2.5 cm of arms and channel) |
| Smooth H- and E-plane bends, $\kappa/\beta \le 0.03$ for every propagating mode, 3 frequencies (10, 17.5, 22 GHz; R = 0.5 m, 10°; 1, 5 and 8 modes) | complex S21 of every propagating mode (all pairs) = wrmodes to 1e-3 | **pass**: max $|\Delta S_{21}|$ H: 3.1e-4, 2.9e-4, 6.8e-4; E: 2.5e-6, 5.0e-4, 2.2e-4 (P3, 10 per λ, $L_p = a/4$) |
| Reciprocity | 1e-10 | **pass**: ≤ 1.7e-14 (relative) in every test; ≤ 9.6e-14 along the 18–90 GHz curve |
| Unitarity at default settings, decreasing under refinement | ≤ 1e-5 | **pass**: ≤ 3e-14 in the tests, ≤ 1e-13 along the 18–90 GHz curve. It does not decrease under refinement because the discrete modal-port system conserves power exactly (theory note §4); it is therefore not an accuracy indicator here, the convergence study is |
| Symmetric miter $S_{11} = S_{22}$ | 1e-10 | **pass**: ≤ 2.5e-13 (90°, 60°, chamfered, sharp corner, E-plane) |
| H-plane miter vs Ma et al. 1997 (D3 of L1) | within the digitizing uncertainty | **not done**: L1's data not available yet. Geometry extracted and results computed (above, and the .npz) |
| WR-90 H-plane plain miter, 18–90 GHz | loss curve with convergence evidence; Bouwkamp comparison reported | **done**: see above. Below 0.4 dB above 8 f_c except 52.46–53.2 GHz (max 0.47 dB) |
| Performance: WR-90 H-plane miter at 90 GHz, default settings | < 2 s per frequency | **fail** in this container (2-core Xeon 2.1 GHz): 10.9–12 s for family 0 (55 171 unknowns; SuperLU factorization 8–9 s), 87 s for the full default basis (129 modes, 6 families, 11 factorizations). With $L_p = a/4$: 4.2 s; with $L_p = a/4$ and 8 per λ: 2.1 s (loss 0.10097 vs 0.10099 dB) |

Other tests in `tests/test_fem2d.py`: $g$ and $h$ from the library fields equal the closed forms to 2.5e-15 (H and E
frames, propagating and evanescent), $g^2\beta$ and $h^2\beta$ constant per family; single-mode bends against the exact
bend mode of the annulus (Bessel functions): $|S_{21} - e^{-i\nu\varphi}|$ = 2.1e-5 (H, LSM) and 2.2e-5 (E, LSE), for
a 20° bend of R = 0.25 m at 10 GHz; continuity of $S$ through the changes of outline at $d = \cos(\theta/2)$ (linear in
$d$, same slope on both sides) and $d = 1/\cos(\theta/2)$; E-plane miter smoke test; families with $k_t^2 \le 0$ and
bases of another frame are refused; `convergence_study` runs; the core imports without the extra. Full suite: 122
passed (the 29 new tests take about 20 s).

## Open issues

1. **Performance.** The default settings (card: P2, λ/10, arms of one guide width) miss the 2 s target by a factor
   of 5–6 here, mostly SuperLU (no faster sparse direct solver is pip-installable for complex matrices on all
   platforms; UMFPACK and MUMPS need system libraries, PARDISO through pypardiso is real-only and Intel-only).
   Recommendations: default `port_distance = a/4` (same $|S|$ within 7e-5, smaller phase error, 2.6× faster);
   for phase-sensitive work P3 at 6–8 per λ. The full default basis at 90 GHz costs one factorization per family and
   polarization (11 at 90 GHz); lines that only need the TE10 family can pass a family-0 basis.
2. **The card's smooth-bend criterion and wrmodes' own accuracy.** wrmodes keeps only propagating modes, which
   leaves out a second-order phase. For a 20° bend of R = 0.25 m (κ/β = 0.025) at 10 GHz, wrmodes' TE10 $S_{21}$ is
   1.24e-3 from the exact bend mode, while the FEM is 2.1e-5 from it. At 19 GHz in the E-plane frame (κ/β ≤ 0.0095),
   wrmodes' TE02 is 1.05e-3 off because the TE12/TM12 pair is evanescent just above its cut-off (19.75 GHz); the FEM
   agrees with the exact bend mode there. The acceptance test therefore uses 10°, R = 0.5 m and frequencies away from
   evanescent cut-offs; all the remaining differences (≤ 6.8e-4) are of this kind (refining the FEM from 10 to 15
   per λ changes $S_{21}$ by 9.4e-6 at 22 GHz, H, and 7.4e-6 at 17.5 GHz, E). WP7 and WP2 should expect this when comparing against wrmodes.
3. **Ma Table 1:** $|S_{11}|$ = 0.2778 here vs 0.2622 in Ma (c/a = 1, 16 GHz); converged here to 3e-5. To be settled
   by WP7 with L1's data (and possibly COMSOL); Ma's staircase approximation of the mirror is a candidate cause.
4. **Gmsh in CI.** `.github/workflows/tests.yml` installs only `.[test]`, so `tests/test_fem2d.py` is skipped in CI.
   To run it there: `pip install -e ".[test,fem]"`, plus `apt-get install libglu1-mesa libxft2` on Ubuntu. I did not
   edit the workflow (not on my card).
5. gmsh keeps global state: `Coplanar2D.gsm` is safe in parallel processes (as WP1's `sweep` does), not in threads.

## Conventions issues

1. **Plain miter for θ ≠ 90°** (card spec, not CONVENTIONS). "The plain miter whose mirror joins the two outer
   corners" is read here as the mirror through the points where the outer walls cross the reference planes,
   $d = \cos(\theta/2)$. For 90° this is the classical plain miter (Ma $c/a = 1$, Alessandri $d/b = 0.707$, "only the
   triangular region"). For other angles it is not the optical mirror (through the intersection of the arm axes,
   $d = 1/(2\cos(\theta/2))$): at 30 GHz in WR-90, 60° loses 7.08 dB with $d = \cos 30°$ but 0.46 dB with the optical
   $d = 0.577$; 120° loses 5.67 dB with $d = 0.5$ but 0.89 dB with $d = 1$. Daniel should decide which one `chamfer=None`
   means for θ ≠ 90°; it is a one-line change in `Miter`.
2. **§6, reference planes inside the discontinuity.** For chamfers deeper than the plain miter ($d < \cos(\theta/2)$,
   e.g. Ma's $c/a = 1.3$), the mirror cuts the outer wall before the reference plane through the inner corner. The GSM
   is then defined by continuing the uniform arm analytically to that plane ($S \to e^{i\beta\delta}Se^{i\beta\delta}$
   from the last uniform cross-section). This is the natural reading of §6 but is not stated there; the mode-matching
   code (WP4/wave 2) and COMSOL post-processing should use the same definition.
3. **§8 / card, unitarity as an accuracy check.** With exact modal ports the discrete GSM is unitary to rounding, so
   "unitarity decreasing under refinement" cannot be observed; this is not a defect.

## For other cards

* **WP1.** `Miter(angle, chamfer=None, plane='H', *, guide)`: angle in **radians**; `guide` is `(w, h)` or any object
  with attributes `w` and `h` (WP1's `Guide` should work as is). `arm_length` is a property. Default bases carry the
  frame tag `'H'` or `'E'` (local frame, E-plane: $a = h$, $b = w$); the component accepts bases tagged with its own
  frame or `None`, and raises otherwise. `gsm(f, basis1, basis2)` handles any families of the bases.
  `default_basis(f, cutoff_factor)` is `ModeBasis.propagating` per port. The returned GSM has an extra attribute
  `fem_info` (element sizes, unknowns per problem).
* **WP7.** Data for the comparisons: `benchmarks/fem2d/wr90_h_miter_loss.npz` (rows: f, loss dB, |S11|², unitarity,
  reciprocity, |S11 − S22|, time, unknowns) and `benchmarks/fem2d/ma1997_wp3.npz` (complex $S_{11}$, $S_{21}$ for Ma's
  cases), both with `meta`. The 2D solver's reference planes are those of CONVENTIONS §6, phases included.
* **WP4 (wave 2 mode matching).** `Coplanar2D` takes any polygon/arc outline with two port planes of possibly
  different widths, so steps in $a$ can be cross-checked against it (not tested yet beyond straight channels).
