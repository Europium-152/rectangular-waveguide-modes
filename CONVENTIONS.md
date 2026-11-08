# wrmodes conventions

*Frozen at the end of wave 0. Every component, solver, test and benchmark of wrmodes 1.0 follows this file.
Changing it requires a pull request that updates this file, the affected code and the tests together, and
Daniel's approval.*

The implementation of sections 1–6 is `src/wrmodes/modes.py` and `src/wrmodes/gsm.py`; the tests that pin them
down are `tests/test_conventions.py` and `tests/test_gsm.py`. The derivation behind section 4 is
`docs/theory/T0-field-convention.md`.

## 1. Units and phasors

* SI units throughout: metres, hertz, seconds, siemens per metre.
* Time dependence $e^{+i\omega t}$, written with `1j` in code. A forward mode varies as $e^{-i\beta z}$.
* $k = \omega/c$ with `scipy.constants.c`; free-space impedance $\eta_0 = \mu_0 c$ (`wrmodes.modes.ETA0`), so that
  $\omega\mu_0 = k\eta_0$ and $\omega\varepsilon_0 = k/\eta_0$ hold exactly. Do not use `scipy.constants.epsilon_0`
  together with `mu_0` and `c`: the CODATA values satisfy $\mu_0\varepsilon_0c^2 = 1$ only to about 1e-10.
* Surface impedance of a wall of conductivity σ: $Z_s = (1 + i)R_s$ with $R_s = \sqrt{\omega\mu_0/(2\sigma)}$
  (for the $e^{+i\omega t}$ convention written with $i$; textbooks with $e^{j\omega t}$ write the same $(1+j)R_s$).
  Baseline SPARC wall: aluminium, σ = 3.5×10⁷ S/m until the alloy is fixed. Copper: σ = 5.8×10⁷ S/m.

## 2. Cross-section frame

* Each port of each component has a **local frame** $(u, v, z)$, right-handed ($\hat u\times\hat v = \hat z$),
  centred on the guide axis: $u\in[-a/2, a/2]$, $v\in[-b/2, b/2]$, $z$ along the axis in the direction of
  propagation from port 1 to port 2.
* Side **a** lies along $u$, side **b** along $v$. In a coplanar bend, the bend plane is the $u$–$z$ plane: this is
  the convention of the JINST paper ("a in the bend plane").
* Curvature: the $z$ scale factor is $h = 1 + \kappa u$. A positive κ puts the centre of curvature at
  $u = -1/\kappa$, so the guide turns towards $-u$ and the outer wall is at $u = +a/2$.

### Global guide frame and plane changes

* A transmission line has a **global guide frame** $(x, y, z)$ with the guide's own dimensions $w$ along $x$ and
  $h$ along $y$. For SPARC, WR-90 with $w = 22.86$ mm and $h = 10.16$ mm; TE₁₀ has its electric field along $y$.
* A component in the **H plane** of the global TE₁₀ mode bends in the $x$–$z$ plane: local $u = x$, $v = y$,
  $a = w$, $b = h$, and the local mode numbers equal the global ones.
* A component in the **E plane** bends in the $y$–$z$ plane: local $u = y$, $v = -x$, $a = h$, $b = w$. Local mode
  $(m, n)$ is global mode $(n, m)$, with a sign set by the field definitions.
* Frame changes are never implicit. They are explicit, lossless components (`FrameChange`, WP1) whose matrix is
  the overlap $\iint e^{\text{to}}_p\cdot e^{\text{from}}_q\,ds$ of the normalized fields, a signed permutation for
  rotations by multiples of 90°. A twist (WP6) ends in a rotated frame and is followed by the corresponding
  `FrameChange` if the next component expects the global frame.
* A bend turning the other way within the same plane uses a negative κ, not a different frame.

## 3. Modes and bases

* A mode is `Mode(kind, m, n)`: `kind` is `'TE'` or `'TM'`; `m` half-periods along $u$ (side a), `n` along $v$
  (side b). TE needs $m + n > 0$; TM needs $m, n \ge 1$. Names follow wrmodes 0.1: `'TE1,0'`, `'TM2,1'`.
* Cutoff wavenumber $\chi = \pi\sqrt{(m/a)^2 + (n/b)^2}$; propagation constant
  $\beta = \sqrt{k^2 - \chi^2}$ above cutoff and $\beta = -i\sqrt{\chi^2 - k^2}$ below, so that every forward mode,
  propagating or evanescent, is $e^{-i\beta z}$ with $\operatorname{Im}\beta\le 0$.
* A `ModeBasis(a, b, modes)` is an ordered list of modes of one cross-section. **Order**: cutoff frequency, then
  TE before TM, then $m$, then $n$. Two bases are equal when $a$, $b$ (to 1e-12 relative) and the ordered modes are
  equal; the cascade refuses to join unequal bases.
* A **family** is the set of modes with the same $n$. Components that do not vary along $v$ (straights, coplanar
  bends, miters, steps in $a$ only, tapers of $a$ only) couple only modes of the same family, and may return one
  GSM per family, combined with `wrmodes.gsm.block_diag`.
* Truncation: a basis contains the propagating modes plus the evanescent modes with cutoff below
  `cutoff_factor × f` (`ModeBasis.propagating`). Coupled-mode (transfer-matrix) solvers use `cutoff_factor = 1`;
  solvers that need evanescent modes state their own default and show convergence in it.

## 4. Mode fields

With $u' = u + a/2$ and $v' = v + b/2$, the normalized potentials of the JINST paper (eq. 5.4) are

```latex
\Pi_{mn} = \frac{2}{\pi\sqrt{m^2 b/a + n^2 a/b}}\,\sin\frac{m\pi u'}{a}\,\sin\frac{n\pi v'}{b},\qquad
\Gamma_{mn} = \frac{\sqrt{\epsilon_m\epsilon_n}}{\pi\sqrt{m^2 b/a + n^2 a/b}}\,\cos\frac{m\pi u'}{a}\,\cos\frac{n\pi v'}{b},
```

with $\epsilon_0 = 1$, $\epsilon_{l>0} = 2$. The transverse electric fields are real and orthonormal
($\iint e_p\cdot e_q\,ds = \delta_{pq}$):

```latex
e_{TM} = \nabla_t\Pi ,\qquad e_{TE} = \nabla_t\Gamma\times\hat z = \big(\partial_v\Gamma,\; -\partial_u\Gamma\big).
```

The transverse magnetic field shape is $\hat z\times e$. These definitions, including the relative sign of TE and
TM, are the ones for which the closed-form curvature coupling coefficients of wrmodes 0.1 hold
(`docs/theory/T0-field-convention.md`). Code that needs field values uses `wrmodes.modes.transverse_field`; it
must not re-implement them.

## 5. Amplitudes and normalization

The transverse fields at any plane are

```latex
E_t = \sum_p \big(a^+_p + a^-_p\big)\sqrt{2/Y_p}\; e_p ,\qquad
H_t = \sum_p \big(a^+_p - a^-_p\big)\sqrt{2Y_p}\; \hat z\times e_p ,
```

with wave admittances $Y_{TE} = \beta/(k\eta_0)$, $Y_{TM} = k/(\eta_0\beta)$ and the **principal** square root.
Equivalently, the amplitudes are normalized with the unconjugated reaction,
$\tfrac12\iint(E^+_p\times H^+_q)\cdot\hat z\,ds = a_p a_q\,\delta_{pq}$.

* For propagating modes, $|a^\pm_p|^2$ is the power carried in the $\pm z$ direction (the paper's normalization).
* Backward modes share the transverse electric field of the forward mode and reverse the magnetic field.
* Below cutoff the same formulas apply with complex $\beta$ and $Y$. Evanescent amplitudes carry no power on their
  own, but with this normalization **every reciprocal component has a symmetric GSM**, evanescent modes
  included.

## 6. Generalized scattering matrices

* A two-port component is a `GSM` with port bases `basis1` (input plane, local $z = 0$) and `basis2` (output
  plane, local $z = L$). Incident: $a_1 = A^+$ at port 1, $a_2 = A^-$ at port 2. Scattered: $b_1 = A^-$ at port 1,
  $b_2 = A^+$ at port 2. $b = S a$ with $S = [[S_{11}, S_{12}], [S_{21}, S_{22}]]$.
* **Reference planes** are the physical end planes of the component; lengths are measured along the centre line.
  A component never includes connecting straight sections unless its name says so.
* **Cascade**: Redheffer star product, `A @ B` or `cascade(A, B, ...)`, in the order of propagation. It requires
  `A.basis2 == B.basis1` and equal frequencies.
* **Properties every component must have**, checked in its tests at each frequency tested:
  * reciprocity: $\max|S - S^T| \le 10^{-10}$ (relative to $\max|S|$);
  * lossless components: unitarity over the propagating modes, $\max|S_p^HS_p - I| \le 10^{-10}$;
  * lossy components: passivity, largest singular value of $S_p \le 1$;
  * the geometric symmetries of the component (e.g. $S_{11} = S_{22}$ for a symmetric miter).
* Coupled-mode solvers produce a transfer matrix $y(L) = \Phi\,y(0)$, $y = [A^+; A^-]$, and convert it with
  `from_transfer_matrix`. Transfer matrices are only used with propagating modes; with evanescent modes, build
  the GSM directly.

## 7. Longitudinal-section (LS) modes

Components that do not vary along $v$ split each family $n$ into two independent scalar 2D problems in the
$u$–$z$ plane, with transverse wavenumber $k_t^2 = k^2 - (n\pi/b)^2$:

| LS mode | defined by | built from | condition on walls parallel to $v$ | contains |
| --- | --- | --- | --- | --- |
| LSM (TM to $v$) | $H_v = 0$ | $E_v \propto \cos(n\pi v'/b)\,\psi(u, z)$ | Dirichlet, $\psi = 0$ | TE$_{m0}$; LSM$_{mn}$ |
| LSE (TE to $v$) | $E_v = 0$ | $H_v \propto \sin(n\pi v'/b)\,\phi(u, z)$ | Neumann, $\partial_n\phi = 0$ | TE$_{0n}$; LSE$_{mn}$ |

The H-plane bend of TE₁₀ (family $n = 0$) is therefore a pure LSM (Dirichlet) problem. For $m, n \ge 1$,
`wrmodes.modes.ls_rotation` maps TE/TM amplitudes to LS amplitudes, $a_{LS} = Q\,a_{TE/TM}$:

```latex
\mathrm{LSE}_{mn} = \cos\varphi\,\mathrm{TE}_{mn} - \sin\varphi\,\mathrm{TM}_{mn},\qquad
\mathrm{LSM}_{mn} = \sin\varphi\,\mathrm{TE}_{mn} + \cos\varphi\,\mathrm{TM}_{mn},\qquad
\tan\varphi = \frac{m\,b\,k}{n\,a\,\beta}.
```

$Q$ is orthogonal above cutoff and complex-orthogonal ($QQ^T = I$) below it, so a GSM computed in the LS basis
converts to the TE/TM basis as $S_{TE/TM} = Q^T S_{LS} Q$ (applied per port) and stays symmetric. Solvers in
the LS basis return their result converted to TE/TM; the TE/TM basis is the only one exchanged between
components.

## 8. Tolerances and validation

| Quantity | Target |
| --- | --- |
| Reciprocity, unitarity (lossless) | 1e-10 |
| Agreement of two independent methods for the same component | 0.01 dB on the main transmission; 1e-4 of input power for every converted mode |
| Agreement with COMSOL | 0.02 dB |
| Agreement with digitized literature curves | the digitizing uncertainty recorded with the data |
| Regression against wrmodes 0.1 (`tests/test_regression_v0_1.py`) | 1e-10 on T and R when the adaptive slicing is unchanged |

A component counts as validated when an agent that did not write it reproduces its results with a different
method, within these targets (see `tasks/README.md`).

## 9. Data files

* Numerical results that tests or reports depend on are stored as `.npz` with a `meta` entry: a JSON string with
  at least `wrmodes_version`, `git_commit`, `created` (ISO date), `generator` (script path) and the parameters.
* Digitized literature data are CSV files under `benchmarks/literature/` with a header comment giving the source
  (reference, figure, axes, units) and an `uncertainty` column.
* Every benchmark is listed in `benchmarks/registry.json` (created in wave 1 by WP7) with its data file,
  reference, tolerance and the test that uses it.

## 10. Code

* Components implement `gsm(f, basis1=None, basis2=None) -> GSM`, hold no global state and do not modify their
  inputs. A component that only couples within families documents it.
* New dependencies beyond numpy and scipy are optional extras (`wrmodes[fem]`). The core must import without them.
* Public functions have docstrings that state units, conventions used and the reference (paper, equation) they
  implement.
* The wrmodes 0.1 API (`solve_mode_coupling`, `solve_transfer_matrix`, `transmission_matrix`,
  `straight_section`, `select_coupled_modes`, `coupled_mode_matrices`) keeps its signatures and results.
