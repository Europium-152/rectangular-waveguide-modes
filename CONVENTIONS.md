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
  together with `mu_0` and `c`: the CODATA values satisfy $\mu_0\varepsilon_0c^2 = 1$ only to about 1e-12, which shows up when large terms cancel.
* Surface impedance of a wall of conductivity σ: $Z_s = (1 + i)R_s$ with $R_s = \sqrt{\omega\mu_0/(2\sigma)}$
  (for the $e^{+i\omega t}$ convention written with $i$; textbooks with $e^{j\omega t}$ write the same $(1+j)R_s$).
  Baseline SPARC wall: aluminium, σ = 3.5×10⁷ S/m until the alloy is fixed. Copper: σ = 5.8×10⁷ S/m.

## 2. Cross-section frame

* Each port of each component has a **local frame** $(u, v, z)$, right-handed ($\hat u\times\hat v = \hat z$),
  centred on the guide axis: $u\in[-a/2, a/2]$, $v\in[-b/2, b/2]$, $z$ along the axis in the direction of
  propagation from port 1 to port 2.
* Side **a** lies along $u$, side **b** along $v$. In a coplanar bend, the bend plane is the $u$–$z$ plane: this is
  the convention of the JINST paper ("a in the bend plane").
* Curvature: the $z$ scale factor is $h = 1 + \kappa(z) u$. A positive κ puts the centre of curvature at
  $u = -1/\kappa$, so the guide turns towards $-u$ and the outer wall is at $u = +a/2$.
* **Port 2 frame.** The frame of port 2 is the frame of port 1 transported along the centre line: for a bend or a
  miter, port 1's frame rotated about $\hat v$ by the bend angle. For κ > 0, $u$ is on the outer side at both
  ports. With this definition a symmetric miter has $S_{11} = S_{22}$ with no sign matrix.
* **Turning a component around** (its port 2 becomes port 1) is the rotation by π about $\hat u$:
  $(u, v, z)\to(u, -v, -z)$. It keeps the sign of κ. Mode $p$ maps to $P_p$ times itself with
  $P = (-1)^{n+1}$ for TE and TM alike, so $S' = P\,S_{\text{swapped}}\,P$ (`GSM.reversed`, `modes.reversal_signs`).
  The signs cancel within a family and matter for components that couple families. A component with an offset
  $d_v$ turned around has its offset at $-d_v$.

### Global guide frame and plane changes

* A transmission line has a **global guide frame** $(x, y, z)$ with the guide's own dimensions $w$ along $x$ and
  $h$ along $y$. For SPARC, WR-90 with $w = 22.86$ mm and $h = 10.16$ mm; TE₁₀ has its electric field along $y$.
* A component in the **H plane** of the global TE₁₀ mode bends in the $x$–$z$ plane: local $u = x$, $v = y$,
  $a = w$, $b = h$, and the local mode numbers equal the global ones.
* A component in the **E plane** bends in the $y$–$z$ plane: local $u = y$, $v = -x$, $a = h$, $b = w$. Global
  mode $(M, N)$ is local mode $(N, M)$ times $(-1)^M$ for TE and $(-1)^{M+1}$ for TM: TE₁₀ → −TE₀₁, TE₂₀ → +TE₀₂,
  TM₁₁ → +TM₁₁, TM₂₁ → −TM₁₂ (`modes.eplane_frame_signs`, checked against the fields in
  `tests/test_conventions.py`). The relative sign of TE and TM changes under this frame change.
* Every `ModeBasis` may carry a **frame tag** (`'H'` or `'E'`). Bases with different tags are
  unequal, so the cascade refuses to join them even when the dimensions agree (square guides).
* Frame changes are never implicit. They are explicit, lossless components (`FrameChange`, WP1) whose matrix is
  the overlap $\iint e^{\text{to}}_p\cdot e^{\text{from}}_q\,ds$ of the normalized fields, a signed permutation for
  rotations by multiples of 90°.
* The global guide frame is attached to the guide: $x$ is always along side $w$. Bends transport it along the centre
  line (rotation about the local $v$). A twist (WP6) rotates it together with the cross-section, so a twist's GSM
  has the global frame at both ports and is **not** followed by a `FrameChange`. The twist angle changes only the
  orientation in space of the components after it.
* A bend turning the other way within the same plane uses a negative κ, not a different frame.

## 3. Modes and bases

* A mode is `Mode(kind, m, n)`: `kind` is `'TE'` or `'TM'`; `m` half-periods along $u$ (side a), `n` along $v$
  (side b). TE needs $m + n > 0$; TM needs $m, n \ge 1$. Names follow wrmodes 0.1: `'TE1,0'`, `'TM2,1'`.
* Cutoff wavenumber $\chi = \pi\sqrt{(m/a)^2 + (n/b)^2}$; propagation constant
  $\beta = \sqrt{k^2 - \chi^2}$ above cutoff and $\beta = -i\sqrt{\chi^2 - k^2}$ below, so that every forward mode,
  propagating or evanescent, is $e^{-i\beta z}$ with $\operatorname{Im}\beta\le 0$. A mode propagates when
  $\chi < k$; a mode exactly at cutoff does not, and its admittance is undefined (`wave_admittance` raises).
* The wrmodes 0.1 functions (`coupled_mode_matrices`, `straight_section`) use $+i|\beta|$ below cutoff. They are
  only for propagating modes: `coupled_mode_matrices` warns and `ModeBasis.legacy_modes(f)` raises if a mode is
  evanescent.
* A `ModeBasis(a, b, modes)` is an ordered list of modes of one cross-section. **Order**: cutoff frequency, then
  TE before TM, then $m$, then $n$. Two bases are equal when $a$, $b$ (to 1e-12 relative) and the ordered modes are
  equal; the cascade refuses to join unequal bases.
* A **family** is the set of modes with the same $n$. Components that do not vary along $v$ (straights, coplanar
  bends, miters, steps in $a$ only, tapers of $a$ only) couple only modes of the same family, and may return one
  GSM per family, combined with `wrmodes.gsm.block_diag`. A step that keeps side $a$ with $d_u = 0$ also conserves
  $m$.
* Wall loss on the walls perpendicular to $v$ (the walls at $v = \pm b/2$) couples $n$ with $n\pm2$ (equal walls)
  or with any $n'$ (unequal walls), so a lossy component is not strictly family-conserving. For metal walls the
  effect is at most about 6e-6 of the power (T1b). A component that treats families separately in the presence
  of loss says so in its docstring and bounds the neglected coupling.
* Truncation: a basis contains the propagating modes plus the evanescent modes with cutoff below
  `cutoff_factor × f`. A family is a fixed $n$ in a *local* frame. A line therefore states the rule in the global
  frame for every frame it uses: an evanescent global mode $(M, N)$ is kept when some propagating mode has the same
  $N$ (a live H-frame family) or the same $M$ (a live E-frame family). Every component of a line receives this one
  basis, mapped to its local frame. Components that conserve $n$ must accept local families without a propagating
  mode; they may treat them as uncoupled (straight modes over their length, no scattering for zero-length
  components) and say so. `ModeBasis.propagating(..., frame=X)` used for a single component in its own frame keeps
  only the live families of that frame. Coupled-mode (transfer-matrix) solvers use `cutoff_factor = 1`;
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
* **Miters** (plain or chamfered, any bend angle θ): the reference planes are perpendicular to each arm's axis and
  pass through the **inner corner** (where the inner wall turns). Each plane lies $(a/2)\tan(\theta/2)$ from the
  intersection of the two arm axes, and the component reports this distance as `arm_length`, so that a line keeps
  its centre-line geometry. The 2D solver, the mode-matching code and COMSOL models use these planes, phases
  included.
* **Plain miter** (no chamfer given) means the optical mirror: the plane through the intersection of the two arm
  axes, perpendicular to the bisector of the bend, so that it reflects one arm's axis into the other's. Its distance
  from the inner corner is $d/a = 1/(2\cos(\theta/2))$; at 90° this is the classical $d/a = 1/\sqrt2$ (Ma's
  $c/a = 1$, Alessandri's $d/b = 0.707$). Chamfered miters give $d$ explicitly, measured from the inner corner as in
  Alessandri (1994). Decided by Daniel, 9 Nov 2026.
* **Evanescent entries at reference planes inside a discontinuity.** When a component's reference plane lies
  where the discontinuity has already changed the cross-section (the optical mirror below 90°, any
  $d < \cos(\theta/2)$), the GSM is defined by continuing the uniform arm analytically to that plane. Propagating
  entries are unaffected; evanescent entries carry a factor $e^{|\beta|\delta}$ per port ($\delta$ the depth of
  the plane inside the discontinuity) and are only as accurate, relatively, as the solver times that factor.
  Cascades with such components keep enough straight length, or enough evanescent modes, for the product to be
  well conditioned (WP3 report, open issue 1).
* **Components with a varying cross-section** (tapers, strangled bends) measure their length along the centre
  line, and their curvature and offsets refer to it (T2 §10).
* **Cascade**: Redheffer star product, `A @ B` or `cascade(A, B, ...)`, in the order of propagation. When
  $I - A_{22}B_{11}$ is singular because a mode is trapped between the two components without coupling to the
  outer ports (e.g. two junctions on the same plane), the product uses the pseudo-inverse, which is exact for such
  hidden modes. If the trapped mode couples to the outer ports, the cascade is at an exact resonance and raises
  `LinAlgError`. It requires
  `A.basis2 == B.basis1` (frame tags included) and equal frequencies; a GSM with a frequency cannot be joined to
  one without.
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

$Q$ is normalized with $r^2 = (na\beta)^2 + (mbk)^2 = (n^2a^2 + m^2b^2)k_t^2$. It is **singular at $k_t = 0$**
($f = nc/2b$), where the LS amplitudes of evanescent pairs with $m\ge1$ do not exist, and ill-conditioned near it
(entries of order $k/|k_t|$); `ls_rotation` raises within $|k_t^2| < 10^{-10}k^2$. A family with $k_t^2\le 0$ has
no propagating mode, so LS solvers leave it out (the default basis already does). $Q$ is orthogonal above cutoff
and complex-orthogonal ($QQ^T = I$) below it, so a GSM computed in the LS basis
converts to the TE/TM basis as $S_{TE/TM} = Q^T S_{LS} Q$ (applied per port) and stays symmetric. Solvers in
the LS basis return their result converted to TE/TM; the TE/TM basis is the only one exchanged between
components.

## 8. Tolerances and validation

| Quantity | Target |
| --- | --- |
| Reciprocity, unitarity (lossless) | 1e-10 |
| Agreement of two independent methods for the same component | 0.01 dB on the main transmission; 1e-4 of input power for every converted mode |
| Agreement with COMSOL | 0.02 dB, beyond the COMSOL error shown by a mesh check |
| Agreement with digitized literature curves | the digitizing uncertainty recorded with the data |
| Regression against wrmodes 0.1 (`tests/test_regression_v0_1.py`) | 1e-10 on T and R when the adaptive slicing is unchanged |

COMSOL references (Gate A lessons, campaign 1): user-defined meshes with an explicit maximum element size (the
physics-controlled mesh shifted a 15 cm bend by 0.02–0.03 dB); a mesh check at two frequencies per case; and,
within about 2 % above the cutoff of an excited mode, ports for every propagating mode instead of a PML, which
reflects near-cutoff modes.

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

* Components derive from `wrmodes.components.Component` and implement `gsm(f, basis1=None, basis2=None) -> GSM`
  and `default_basis(f, cutoff_factor=None)`, where `None` means the component's own default truncation. They
  hold no global state and do not modify their inputs. A component that only couples within families documents
  it.
* Every component declares a class or instance attribute `plane`: `'H'` or `'E'` for the frame in which its local
  $a, b$ are given, or `None` only for components whose GSM is unchanged by the H↔E frame change (lossless or
  uniform-wall straights, frame changes). `Line` raises if it is missing.
* A component given port bases smaller than its own default truncation (e.g. propagating modes only) still solves
  on its own default basis, or on the union of both, and returns the requested sub-block. Dropping evanescent port
  modes assumes the neighbouring components are far enough away for those modes to decay; a component documents
  the distance it needs.
* New dependencies beyond numpy and scipy are optional extras (`wrmodes[fem]`). The core must import without them.
* Public functions have docstrings that state units, conventions used and the reference (paper, equation) they
  implement.
* The wrmodes 0.1 API (`solve_mode_coupling`, `solve_transfer_matrix`, `transmission_matrix`,
  `straight_section`, `select_coupled_modes`, `coupled_mode_matrices`) keeps its signatures and results.
