# WP3 — 2D modal-port FEM solver (wave 1: H-plane miter): report

Branch `wave-1/WP3-2d-solver`, from `main` at 223aeeb, with `origin/wave-1/WP4-steps` merged in (WP1's `Line`,
component protocol and conventions update 1). Status: done, except the Ma et al. comparison, which needs L1's
digitized data. The first version was reviewed (PR #11); the section "Review fixes" lists the changes. All numbers
below are for the current code and defaults.

## What was done

| Item | Files |
| --- | --- |
| Geometry: outlines of straight segments and arcs with two port planes; gmsh meshing of core + arms; mirror-symmetric meshes; re-entrant corner refinement | `src/wrmodes/fem2d/geometry.py` |
| Scalar solver: P3 (or P2) Lagrange elements with scikit-fem, isoparametric P2 geometry on arcs, modal (DtN) ports at the end of straight arms, complex-symmetric system, SuperLU | `src/wrmodes/fem2d/solver.py` |
| Normalization 2D amplitude → LS → TE/TM, computed from `transverse_field`, `admittance` and `ls_rotation` | `src/wrmodes/fem2d/normalization.py`, `docs/theory/WP3-normalization.md` |
| Components `Coplanar2D(outline, b=None, plane='H', guide=None, mirror=None, length=None, **options)` and `Miter(angle_deg, chamfer=None, plane='H', guide=None, **options)`, `FEMOptions`, outline helpers (`miter_outline`, `arc_bend_outline`, `straight_outline`, `mirror_outline`), `mode_parity` | `src/wrmodes/fem2d/components.py` |
| `convergence_study(component, f, mesh_densities, n_evanescent)` | `src/wrmodes/fem2d/convergence.py` |
| Tests (40, skipped without the extra; about 10 s) | `tests/test_fem2d.py` |
| CI job installing `.[test,fem]` with `libglu1-mesa libxft2` (Ubuntu) and running the FEM tests | `.github/workflows/tests.yml` |
| Example: WR-90 H-plane miter loss curve | `examples/h_plane_miter.py` → `benchmarks/fem2d/wr90_h_miter_loss.npz` |
| Ma et al. 1997 geometries computed for the L1/WP7 comparison | `benchmarks/fem2d_ma1997.py` → `benchmarks/fem2d/ma1997_wp3.npz` |

`import wrmodes` does not import scikit-fem or gmsh (tested); `import wrmodes.fem2d` without them raises an
ImportError naming the extra and the system libraries. `pyproject.toml` and `src/wrmodes/__init__.py` are unchanged:
the public names live in `wrmodes.fem2d`, so that the core imports without the extra. The components follow WP1's
protocol (`plane`, `guide`/`with_guide`, `Guide.require_lossless`, `default_basis(f, cutoff_factor=None)`) and work
in a `wrmodes.Line`, H- and E-plane, with the frame changes the line inserts.

### Method in brief

* Per family $n$ and polarization (LSM: $E_v$, Dirichlet; LSE: $H_v$, Neumann), $\nabla^2\psi + k_t^2\psi = 0$ in the
  outline plus two straight arms of length `port_distance` (default $a/4$). At the arm ends, a modal condition with
  the propagating port modes plus evanescent ones, up to the first mode that decays by 16 e-folds over the arm.
  Incident modes enter through these ports; $S^{2D}_{ij} = 2i\beta_j b_i^TA^{-1}b_j - \delta_{ij}$ with the symmetric
  matrix $A$, so reciprocity holds to rounding and, the discrete system being lossless, so does unitarity.
* Shift to the reference planes with $e^{i\beta L}$ per index, then 2D → LS amplitudes with the factors $g$ (LSM)
  and $h$ (LSE, with a sign change of the reflections), then $S = Q^TS^{LS}Q$ per port. Derivation:
  `docs/theory/WP3-normalization.md`.
* One mesh per family, $h = 2\pi/k_{\text{eff}}/6$ with $k_{\text{eff}} = \max(k_t, |\beta|$ of the evanescent basis
  modes$)$, at most $a/4$, refined to $0.05h$ at re-entrant corners. Miters are meshed on one half and mirrored, so
  that $S_{11} = S_{22}$ holds to rounding.
* Every family of the bases is solved, also those without a propagating mode ($k_t^2 < 0$, the line's 'live' rule).
  A family exactly at $k_t = 0$, where the LS rotation does not exist, is returned as uncoupled straight modes over the
  centre-line length.
* Evanescent basis modes: the arms are shortened so that the shift $e^{|\beta|L}$ from the modal port amplifies the
  discretization error by at most 10 (`evanescent_gain`).
* Miter geometry (CONVENTIONS.md §2, §6): angle in degrees, positive turning towards $-u$, negative towards $+u$ (GSM
  $DSD$, $D = (-1)^{m+1}$, the mode parity under $u\to-u$). Inner corner $O$, reference planes through $O$ at
  `arm_length` $= (a/2)\tan(|\theta|/2)$, port 2's $u$ on the outer side. `chamfer` is Alessandri's $d/a$; `None` is the
  optical mirror through the intersection of the arm axes, $d = 1/(2\cos(\theta/2))$ (the plain miter at 90°);
  $d = \cos(\theta/2)$ is the mirror through the outer-wall points of the reference planes; $d \ge 1/\cos(\theta/2)$ is
  the sharp corner. When the mirror cuts the outer walls before the reference planes ($d < \cos(\theta/2)$, e.g. the
  optical mirror below 90°), the field is solved from the last uniform cross-section and shifted analytically to the
  reference planes (`PortPlane.deembed`).
* Mesher: gmsh, as the card asks. The pip wheel needs `libGLU.so.1` and `libXft.so.2`
  (`apt-get install libglu1-mesa libxft2`); no fallback mesher was needed.

## Results

### WR-90 plain 90° H-plane miter, 18–90 GHz

Loss $= -20\log_{10}|S_{21}|$, TE10 → TE10 (conversion to TE$_{m0}$ plus reflection, perfect walls), default settings
(P3, 6 elements per wavelength, arms of $a/4$), family $n = 0$ (TE10 couples only to it). The data file also holds
points at $mf_c(1+\delta)$ around every TE$_{m0}$ cut-off, $\delta$ = −0.01 … +0.04.

| f (GHz) | loss (dB) | f (GHz) | loss (dB) | f (GHz) | loss (dB) | f (GHz) | loss (dB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | 0.8309 | 38 | 0.4174 | 58 | 0.1324 | 78 | 0.1458 |
| 20 | 0.4197 | 40 | 0.5699 | 60 | 0.2012 | 80 | 0.1828 |
| 22 | 0.7335 | 42 | 0.3832 | 62 | 0.1771 | 82 | 0.1208 |
| 24 | 0.7147 | 44 | 0.2473 | 64 | 0.1806 | 84 | 0.0822 |
| 26 | 1.1074 | 46 | 0.2430 | 66 | 0.2799 | 86 | 0.1234 |
| 28 | 0.9051 | 48 | 0.2640 | 68 | 0.1802 | 88 | 0.1070 |
| 30 | 0.4499 | 50 | 0.2409 | 70 | 0.1260 | 90 | 0.1010 |
| 32 | 0.2751 | 52 | 0.2912 | 72 | 0.1145 | | |
| 34 | 0.4269 | 54 | 0.3373 | 74 | 0.1435 | | |
| 36 | 0.3994 | 56 | 0.1960 | 76 | 0.1302 | | |

* **Cut-off cusps.** The loss is continuous across each TE$_{m0}$ cut-off ($mf_c$, $f_c$ = 6.557 GHz), with a
  square-root threshold cusp: it rises steeply just below the cut-off and peaks at or just above it. Highest values:
  **1.841 dB at the TE40 cut-off** (26.2286 GHz, 1e-6 above it), 0.672 dB at TE60, **0.4744 dB just above TE80**
  (52.4624 GHz, 1e-4 above it), 0.332 dB at TE10,0, 0.233 dB at TE12,0.
* **Bouwkamp (< 0.4 dB above 8 $f_c$ = 52.46 GHz; Butterweck & de Ronde 1968, p. 97).** Above 8 $f_c$ the loss
  exceeds 0.4 dB only in a narrow window just above the TE80 cut-off (about 52.42–53.25 GHz, starting just below the
  cut-off), up to 0.474 dB; elsewhere above 8 $f_c$ it stays below 0.38 dB (0.3799 dB at 53.5 GHz). Our loss includes
  reflection ($|S_{11}|^2 \le 1.2\times10^{-3}$, at most 0.005 dB); Bouwkamp's statement is about conversion into H$_{m0}$
  modes. Reported, not pass/fail, as the card asks.
* **Convergence of the curve** (13 points, 18–90 GHz every 6 GHz): P3 at 10 per λ and P3 at 6 per λ with arms of one
  guide width $a$ agree with the default to 1.6e-4 dB (at 18 GHz) and to 1e-4 dB elsewhere.
* **Independent check by the reviewer** (mode matching of the triangle with an odd image, extrapolated in the number of
  modes), against this solver; "FEM, first default" is the first version's default (P2, 10 per λ, arms of $a$),
  "FEM, P3" its P3 run, essentially the current default:

  | f | MM extrapolated | FEM, first default | FEM, P3 |
  | --- | --- | --- | --- |
  | 18 GHz | 0.8306 dB | 0.8317 | 0.83078 |
  | 26.2417 GHz | 1.8228 | 1.8133 | 1.8225 |
  | 30 GHz | 0.44985 | 0.4501 | 0.44986 |
  | 52.4834 GHz | 0.4736 | 0.4706 | 0.4736 |
  | 90 GHz | 0.1010 | 0.1010 | 0.10099 |

### Convergence study (`convergence_study`, 90° WR-90 miter, family 0, other settings default)

Changes relative to the finest run of each series.

| f | elements per λ | unknowns | max \|Δ\|S\|\| | max \|ΔS\| (complex) | loss (dB) |
| --- | --- | --- | --- | --- | --- |
| 30 GHz | 4 | 1 808 | 1.6e-4 | 4.2e-4 | 0.44996 |
| | 6 (default) | 2 867 | 3.4e-5 | 7.2e-5 | 0.44988 |
| | 8 | 4 532 | 1.8e-5 | 2.7e-5 | 0.44986 |
| | 12 | 9 017 | 5.2e-6 | 7.0e-6 | 0.44986 |
| | 16 (reference) | 15 281 | 0 | 0 | 0.44985 |
| 90 GHz | 4 | 9 017 | 1.5e-4 | 1.6e-3 | 0.10102 |
| | 6 (default) | 19 067 | 3.7e-5 | 1.5e-4 | 0.10099 |
| | 8 | 32 804 | 1.2e-5 | 2.5e-5 | 0.10099 |
| | 12 (reference) | 72 548 | 0 | 0 | 0.10099 |

* Port modes: with the decay rule (the first omitted port mode decays by 16 e-folds over the arm, i.e. at least 20
  evanescent port modes for arms of $a/4$), the minimum `n_evanescent` = 2 … 16 changes $S$ by < 1e-11.
* Arm length: $L_p = a/8$, $a/2$, $a$ change $S$ (complex) by at most 1.2e-4 (30 GHz) and 2.8e-4 (90 GHz) against the
  finest mesh, i.e. within the mesh error.
* Unitarity and reciprocity: ≤ 4e-14 along the curve (exact by construction, see the theory note §4).

### Evanescent modes in the basis

WR-90 90° miter, 30 GHz, family 0, cutoff_factor 1.6 (TE5,0–TE7,0 evanescent, true entries up to 0.29): two fine
references (P3, 16 per λ) with arms of $a/40$ and $a/10$, whose shifts $e^{|\beta|L}$ differ by a factor up to 28,
agree to 4.7e-6. The default settings agree with them to 6.2e-4 (arms shortened to 1.58 mm automatically); P3 at
10 per λ to 4.8e-5. The first version gave errors of 3.2e11 here (reviewer). For a straight channel with a basis of
24 modes including families without propagating modes (20 GHz, cutoff_factor 2): propagating block exact to 1e-5,
evanescent entries to 2.2e-4.

### Optical mirror versus outer-corner mirror (`chamfer=None` changed at review)

TE10 loss (dB), WR-90, family 0, default settings:

| θ (°) | optical d/a | 30 GHz | 60 GHz | outer-corner d/a = cos(θ/2) | 30 GHz | 60 GHz |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.518 | 0.080 | 0.550 | 0.966 | 7.99 | 27.97 |
| 45 | 0.541 | 1.094 | 0.347 | 0.924 | 16.53 | 18.73 |
| 60 | 0.577 | 0.458 | 0.199 | 0.866 | 7.08 | 9.21 |
| 90 | 0.707 | 0.450 | 0.201 | 0.707 | 0.450 | 0.201 |
| 120 | 1.000 | 0.892 | 0.273 | 0.500 | 5.67 | 8.02 |
| 150 | 1.932 | 0.385 | 1.045 | 0.259 | 26.10 | 36.05 |

### Ma et al. (1997): geometry and computed values

Geometry, from p. 4 and the figure insets: $a$ = 15.799 mm, $b$ = 7.899 mm, 90° H-plane bend, single-mode band
9.49–19.0 GHz. The outer corner is cut by a mirror whose ends are on the two outer walls at distance $c$ from the outer
corner (Figs. 6, 8, 9 insets); $c/a = 0$ is the sharp corner (Fig. 5), $c/a = 1$ the plain miter. In the card's
parameter: $d/a = (2 - c/a)/\sqrt2$. Cases: Fig. 5 ($c/a = 0$, $|S_{21}|^2$), Figs. 6, 8, 9 and Table 1 ($c/a = 1$),
Fig. 11 ($c/a$ = 0, 0.6, 0.631, 0.7, 1.0, 1.3, $|S_{11}|$ in dB vs $f/f_c$ = 1–2). All are in
`benchmarks/fem2d/ma1997_wp3.npz` (complex $S_{11}$, $S_{21}$; 0.1 GHz steps and $f/f_c$ steps of 0.02).

* $c/a = 1$, 16 GHz: $|S_{11}|$ = 0.2778, $|S_{21}|$ = 0.9606. Ma's Table 1 (60 steps): 0.2622 and 0.9650. The reviewer's
  mode matching gives 0.27777 and attributes Ma's value to his staircase approximation of the mirror.
* $c/a = 0$: $|S_{21}|^2$ = 0.634 (10.1 GHz), 0.860 (12.6), 0.834 (14.1), 0.548 (16.1), 0.016 (17.6), 0.205 (18.6).
* Fig. 11, $|S_{11}|$ (dB) at $f/f_c$ = 1.12 / 1.42 / 1.72: $c/a$ = 0: −6.1 / −8.4 / −2.8; 0.6: −11.8 / −21.5 / −43.3;
  0.631: −13.6 / −26.3 / −28.9; 0.7: −22.1 / −29.8 / −19.2; 1.0: −1.6 / −6.0 / −12.0; 1.3: −0.01 / −0.14 / −1.33.

## Acceptance tests

| Check | Target | Result |
| --- | --- | --- |
| Straight channel, 20 GHz, 8 propagating modes in families 0 and 1 | S = `gsm.straight` to 1e-6, signs included | **pass**: 5.3e-7 (P3, 15 per λ). With evanescent modes and dead families (24 modes, default settings): propagating block < 1e-5, all entries 2.2e-4 |
| Smooth H- and E-plane bends, $\kappa/\beta \le 0.03$ for every propagating mode, 10, 17.5, 22 GHz (R = 0.5 m, 10°; 1, 5 and 8 modes) | complex S21 of every propagating mode (all pairs) = wrmodes to 1e-3 | **pass**: max $|\Delta S_{21}|$ H: 3.1e-4, 2.9e-4, 6.8e-4; E: 2.5e-6, 5.0e-4, 2.2e-4 (P3, 10 per λ) |
| Reciprocity | 1e-10 | **pass**: ≤ 2e-14 (relative) in every test; ≤ 4e-14 along the curve |
| Unitarity at default settings | ≤ 1e-5 | **pass**: ≤ 4e-14 (exact by construction; not an accuracy indicator here) |
| Symmetric miter $S_{11} = S_{22}$ | 1e-10 | **pass**: ≤ 5e-14 (90°, 60°, 120°, chamfered, sharp corner, E-plane) |
| H-plane miter vs Ma et al. 1997 (D3 of L1) | within the digitizing uncertainty | **not done**: L1's data not available yet. Geometry extracted and results computed |
| WR-90 H-plane plain miter, 18–90 GHz | loss curve with convergence evidence; Bouwkamp comparison reported | **done** (above) |
| Performance: WR-90 H-plane miter at 90 GHz, default settings | < 2 s per frequency | **pass** for the TE10 family: 0.65 s here (2-core Xeon 2.1 GHz; 19 067 unknowns). The full default basis (129 modes, 6 families, 11 factorizations) takes 4.4 s |

Other tests in `tests/test_fem2d.py`:

* the closed forms of $g$ and $h$ (2.5e-15);
* the LSE reflection sign on a symmetric E-plane step at 7 GHz: $a\to a/2$ gives $S_{11}$ = −0.3336−0.0145j and
  $a \to a/4$ gives −0.6005−0.0136j, against the static values −1/3 and −0.6 and the reviewer's independent values;
* single-mode bends against the exact bend mode of the annulus (2.1e-5, H and E);
* a family at $k_t = 0$;
* evanescent entries against two references;
* the negative angle against the FEM of the mirrored outline (independent mesh, ≤ 4.7e-5);
* `chamfer=None` is the optical mirror;
* continuity of $S$ through the changes of outline;
* the component protocol (no guide, `with_guide`, lossy guide refused, frames);
* a `Line` [Straight, Miter(90), Straight, Miter(−90), Straight] at 10, 15 and 20 GHz, and the same with E-plane
  miters and the inserted frame changes, equal to the manual cascades to < 1e-12;
* a `Line` with the 'live' rule at 30 GHz, cutoff_factor 1.5 (a family with $k_t^2 < 0$);
* `convergence_study`;
* the core imports without the extra.

Full suite: 272 passed.

## Review fixes (PR #11)

| Review item | Change |
| --- | --- |
| Merge `origin/wave-1/WP4-steps` | merged; WP1's `components/base.py` and `__init__.py` taken |
| Blocker 1: evanescent modes in requested bases | arms shortened automatically (`evanescent_gain` = 10), with port modes up to 16 e-folds of decay; verified against two references of different arm lengths (above); entries stay exactly reciprocal. Reading the amplitudes on the reference planes was considered and not taken: with modal-port excitation the incident evanescent part there is not known exactly. Ports placed on the reference planes converge too slowly in $M$, because the inner corner lies on the port lines: TE10 loss 0.4428 / 0.4469 / 0.4486 dB for $M$ = 40 / 80 / 160, against 0.44985 converged |
| Blocker 2: families with $k_t^2 \le 0$ | solved like the others (CONVENTIONS §3 update); exactly at $k_t = 0$, uncoupled straight modes over the centre-line length (`length`; `Miter` sets $2\cdot$`arm_length`) |
| Defaults P3, 6 per λ, arms $a/4$ | done; card text updated (`tasks/WP3-2d-solver.md`) |
| `chamfer=None` | optical mirror $d = 1/(2\cos(\theta/2))$; the outer-corner mirror is `chamfer=cos(θ/2)`; table above |
| Line integration | `plane`, `guide=None`, `with_guide`, `require_lossless`, `default_basis(f, None)`, `angle_deg` in degrees, negative angles (mode parity $DSD$, checked against a mirrored mesh); Line tests in H and E |
| Report: cusp and Bouwkamp | corrected (above) |
| LSE sign test | E-plane step test added |
| CI | `fem` job added |
| Even/odd speed-up | not done: with the new defaults the 90 GHz TE10 family takes 0.65 s, so it was not needed |

## Open issues

1. **Reference planes inside the discontinuity.** For the optical mirror below 90° (and any $d < \cos(\theta/2)$), the
   reference planes lie where the mirror has already cut the outer wall. The GSM there is defined by continuing the
   uniform arm analytically.
   * For evanescent modes this multiplies the entries by $e^{|\beta|\delta}$ per port, with
     $\delta = (\cos(\theta/2) - d)\,a/\sin(\theta/2)$ (0.58 $a$ at 60°). No choice of arm length avoids it.
   * Those entries are therefore large, and their relative accuracy is the solver's multiplied by this factor.
   * Propagating entries are unaffected.

   This is worth a CONVENTIONS note (see below).
2. **wrmodes' own accuracy** (unchanged from the first report). Its propagating-mode truncation errs by more than
   1e-3 for some bends allowed by κ/β ≤ 0.03:
   * TE10, R = 0.25 m, 20°, 10 GHz: wrmodes is 1.24e-3 from the exact bend mode, the FEM 2.1e-5;
   * near evanescent cut-offs: 1.05e-3 even at κ/β < 0.01.

   The test geometry avoids those cases; the reviewer confirmed the claim.
3. **Ma Table 1:** our 0.2778 against Ma's 0.2622; the reviewer's mode matching gives 0.27777. Recorded for WP7.
4. **gmsh keeps global state:** run frequencies in separate processes (as `Line.sweep` does), not threads.

## Conventions issues

1. **§6, reference planes inside the discontinuity** (see open issue 1): the GSM at a reference plane that is not a
   uniform cross-section is defined here by the analytic continuation of the uniform arm. The mode-matching code and
   COMSOL post-processing should use the same definition; CONVENTIONS could state it.
2. **Unitarity as an accuracy check** (§8 / card): exact for this formulation, so it says nothing about accuracy here.

## For other cards

* **WP1 / users:**
  * `Miter(angle_deg, chamfer=None, plane='H', guide=None)`: angle in degrees; a negative angle turns towards $+u$;
    `chamfer=None` is the optical mirror.
  * `Coplanar2D(outline, b=None, plane='H', guide=None, ...)` takes any outline of constant width.
  * The GSM has an extra attribute `fem_info` (element sizes, arm length, unknowns, uncoupled families).
* **WP7:** data in `benchmarks/fem2d/wr90_h_miter_loss.npz` (rows: f, loss dB, |S11|², unitarity, reciprocity,
  |S11 − S22|, time, unknowns) and `benchmarks/fem2d/ma1997_wp3.npz`, both with `meta` (git commit 8b90380).
* **WP4 (wave 2 mode matching):** `Coplanar2D` accepts port planes of different widths (an E-plane step is in the
  tests), so steps in $a$ can be cross-checked against it.
