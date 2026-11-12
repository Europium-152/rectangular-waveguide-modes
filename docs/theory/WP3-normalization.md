# WP3 — From the 2D scalar solution to CONVENTIONS amplitudes

*Theory note for the 2D modal-port solver (`src/wrmodes/fem2d`). Status: derived here; checked numerically by
`tests/test_fem2d.py` (closed forms against the library fields, straight channel, smooth bends against wrmodes and
against exact bend modes).*

## 1. The scalar problems

A component that does not vary along $v$ couples only modes of one family $n$ (CONVENTIONS.md §3, §7). In the
family, the field splits into two independent longitudinal-section problems in the $u$–$z$ plane, with
$k_t^2 = k^2 - (n\pi/b)^2$ and $v' = v + b/2$:

| | scalar $\psi$ | walls | port modes $\phi_m(u)$, orthonormal on $u'\in[0,a]$ |
| --- | --- | --- | --- |
| LSM ($H_v = 0$) | $E_v = \cos(n\pi v'/b)\,\psi(u,z)$ | $\psi = 0$ | $\sqrt{2/a}\,\sin(m\pi u'/a)$, $m\ge1$ |
| LSE ($E_v = 0$) | $H_v = \sin(n\pi v'/b)\,\psi(u,z)$ | $\partial_n\psi = 0$ | $\sqrt{\epsilon_m/a}\,\cos(m\pi u'/a)$, $m\ge0$ |

In both cases $\nabla^2\psi + k_t^2\psi = 0$, and in a straight arm
$\psi = \sum_m(\alpha^+_m e^{-i\beta_m z} + \alpha^-_m e^{+i\beta_m z})\phi_m(u)$ with
$\beta_m^2 = k_t^2 - (m\pi/a)^2 = k^2 - \chi_{mn}^2$: the propagation constant of the 3D mode $(m, n)$, with the branch
of CONVENTIONS.md §3. The $\alpha$ are the **2D amplitudes**.

## 2. The field of one LS mode

CONVENTIONS.md §5 writes $E_t = \sum_p(a^+_p + a^-_p)\sqrt{2/Y_p}\,e_p$ and
$H_t = \sum_p(a^+_p - a^-_p)\sqrt{2Y_p}\,\hat z\times e_p$, and §7 defines the LS amplitudes $a_{LS} = Q\,a$ with
$Q$ complex-orthogonal, so $a = Q^Ta_{LS}$ and

```latex
E_t = \sum_l (a^+_l + a^-_l)\,\mathcal E^{(l)},\qquad \mathcal E^{(l)} = \sum_p Q_{lp}\sqrt{2/Y_p}\;e_p ,\qquad
H_t = \sum_l (a^+_l - a^-_l)\,\mathcal H^{(l)},\qquad \mathcal H^{(l)} = \sum_p Q_{lp}\sqrt{2Y_p}\;\hat z\times e_p .
```

Only the TE/TM pair $(m,n)$ enters row $l$. With $k_x = m\pi/a$, $k_y = n\pi/b$, $\chi^2 = k_x^2+k_y^2$, $S = \sin k_xu'$,
$C = \cos k_xu'$ and the fields of CONVENTIONS.md §4 ($m, n\ge1$, both normalization constants $c = 2/(\sqrt{ab}\,\chi)$):

```latex
e^{TE}_v = c\,k_x\,S\cos k_yv',\quad e^{TM}_v = c\,k_y\,S\cos k_yv',\qquad
e^{TE}_u = -c\,k_y\,C\sin k_yv',\quad e^{TM}_u = c\,k_x\,C\sin k_yv'.
```

With $\tan\varphi = k_xk/(k_y\beta)$, i.e. $\sin\varphi = k_xk/\rho$, $\cos\varphi = k_y\beta/\rho$ and
$\rho^2 = k_x^2k^2 + k_y^2\beta^2 = \chi^2k_t^2$ (the normalization $r$ of `ls_rotation`, up to the factor $ab/\pi$),
and $Y_{TE} = \beta/(k\eta_0)$, $Y_{TM} = k/(\eta_0\beta)$:

* **LSM** (row $\sin\varphi\,\mathrm{TE} + \cos\varphi\,\mathrm{TM}$):
  $\mathcal E^{(l)}_v = \frac{c\sqrt{2\eta_0}}{\rho}\big(k_x^2k\sqrt{k/\beta} + k_y^2\beta\sqrt{\beta/k}\big)S\cos k_yv'
  = c\sqrt{2\eta_0}\,\frac{\rho}{\sqrt{k\beta}}\,S\cos k_yv'$.
* **LSE** (row $\cos\varphi\,\mathrm{TE} - \sin\varphi\,\mathrm{TM}$): $\mathcal E^{(l)}_v \propto
  \beta\sqrt{k/\beta} - k\sqrt{\beta/k} = 0$, as it must, and, since $(\hat z\times e)_v = e_u$,
  $\mathcal H^{(l)}_v = -c\sqrt{2/\eta_0}\,\frac{\rho}{\sqrt{k\beta}}\,C\sin k_yv'$.

Below cutoff, $\beta = -i|\beta|$ and the principal roots give $\beta\sqrt{k/\beta} = k\sqrt{\beta/k} = e^{-i\pi/4}\sqrt{k|\beta|}$,
so the cancellation and the two expressions hold with $\sqrt{k\beta}$ read as $\beta\sqrt{k/\beta}$. Writing
$\mathcal E^{(l)}_v = g\,\cos(k_yv')\,\phi_m(u)$ for LSM and $\mathcal H^{(l)}_v = h\,\sin(k_yv')\,\phi_m(u)$ for LSE,
and repeating the computation for $\mathrm{TE}_{m0}$ (LSM, $Q = 1$, $c = \sqrt2/(\sqrt{ab}\chi)$) and $\mathrm{TE}_{0n}$
(LSE, $Q = 1$):

```latex
g = \sqrt{\epsilon_n}\;k_t\,\sqrt{\frac{2\eta_0}{k\beta b}},\qquad
h = -\,\frac{2k_t}{\sqrt{\eta_0 k\beta b}}\quad(n\ge1).
```

`normalization.ls_amplitude_factors` does **not** use these closed forms: it evaluates
$\sum_pQ_{lp}\sqrt{2/Y_p}\,e_{p,v}$ (or $\sqrt{2Y_p}\,e_{p,u}$) with `wrmodes.modes.transverse_field`,
`ModeBasis.admittance` and `ls_rotation` at four generic points, checks that the ratio to
$\cos(k_yv')\phi_m$ (or $\sin(k_yv')\phi_m$) is the same at all of them, and returns it. So the sign and branch of
$g$ and $h$ are whatever the library conventions imply; `tests/test_fem2d.py` checks them against the closed forms,
propagating and evanescent.

## 3. 2D amplitudes and LS amplitudes, sign included

At a port plane, take the local frame $(u, v, z)$ of the port (CONVENTIONS.md §2) and $\psi$ expanded with $\phi_m$
of that port's $u$. Both ports of a coplanar component share $\hat v$ (port 2's frame is port 1's rotated about
$\hat v$), so $E_v$ and $H_v$ are the same physical scalars at both ports and $\psi$ is one function on the whole
domain. The wave travelling **into** the domain is $a^+$ at port 1 and $a^-$ at port 2; the outgoing one is $a^-$ at
port 1 and $a^+$ at port 2. From §2:

* LSM: $E_v$ carries $a^+ + a^-$, so $\alpha = g\,a$ for every wave, in or out, at either port.
* LSE: $H_v$ carries $a^+ - a^-$, so $\alpha = h\,a$ for $a^+$ and $\alpha = -h\,a$ for $a^-$. With $s_1 = +1$,
  $s_2 = -1$: incoming $\alpha^{in}_q = s_q\,h\,a^{in}_q$, outgoing $\alpha^{out}_p = -s_p\,h\,a^{out}_p$.

Hence, for the 2D scattering matrix $\alpha^{out} = S^{2D}\alpha^{in}$ (entry $i$ at port $p$, entry $j$ at port $q$):

```latex
\text{LSM:}\quad S^{LS}_{ij} = S^{2D}_{ij}\,\frac{g_j}{g_i},\qquad
\text{LSE:}\quad S^{LS}_{ij} = -\,s_ps_q\,S^{2D}_{ij}\,\frac{h_j}{h_i}.
```

So in the LSE problem the reflections change sign and the transmissions do not. Check with a short circuit (a PEC
wall across the guide): CONVENTIONS amplitudes give $S_{11} = -1$ ($E_t = 0$). For LSM, $\psi = E_v = 0$ on the wall
gives $S^{2D} = -1$; for LSE, $E_u\propto\partial_zH_v = 0$ gives a Neumann wall and $S^{2D} = +1$, i.e.
$S^{LS} = -1$ again.

**Reciprocity.** The discrete 2D problem gives $\beta_iS^{2D}_{ij} = \beta_jS^{2D}_{ji}$ (§4). $S^{LS}$ is symmetric iff
$g_i^2\beta_i$ (or $h_i^2\beta_i$) is the same for all modes of the problem, which the closed forms confirm:
$g^2\beta = 2\epsilon_n\eta_0k_t^2/(kb)$ and $h^2\beta = 4k_t^2/(\eta_0kb)$ do not depend on $m$ or on $a$. Only $g^2$ and
$h^2$ enter this condition; the **sign** of $g$ and $h$ is what the smooth-bend phase comparison checks.

**TE/TM basis.** Per port, $a_{LS} = Qa$, so $S^{TE/TM}_{pq} = Q_p^TS^{LS}_{pq}Q_q$ (CONVENTIONS.md §7).

## 4. The discrete problem and its scattering matrix

Weak form (P3 by default, or P2, Lagrange elements, scikit-fem), with $b_m = \int_{\text{port}}w\,\phi_m\,ds$:

```latex
\int\nabla w\cdot\nabla\psi - k_t^2w\psi\;+\;\sum_{\text{ports}}\sum_{m\le M}i\beta_m\,b_m(w)\,b_m(\psi)
\;=\;\sum_{\text{ports}}\sum_m 2i\beta_m\,\alpha^{in}_m\,b_m(w),
```

from the port expansion $\partial_n\psi = \sum_mi\beta_m(2\alpha^{in}_m - c_m)\phi_m$, $c_m = b_m(\psi)$, with $s$ the
distance into the domain and $\alpha^{in}$ the amplitude of $e^{-i\beta s}$. The matrix $A$ is complex symmetric, so
$S^{2D}_{ij} = 2i\beta_j\,b_i^TA^{-1}b_j - \delta_{ij}$ satisfies $\beta_iS^{2D}_{ij} = \beta_jS^{2D}_{ji}$ to rounding.
$K$ and $M$ are real, $i\beta_m$ is imaginary for propagating and real positive for evanescent port modes, so
$\operatorname{Im}(\bar\psi^TA\psi) = \sum_{\text{prop}}\beta_m|c_m|^2$ and the discrete $S$ conserves power over the
propagating modes to rounding: unitarity is not a discretization indicator for this formulation. Modes beyond $M$
see a natural (Neumann) condition on the port, which is why the ports sit at the end of straight arms where those
modes have decayed; convergence in $M$ and in the arm length is shown in `docs/reports/WP3.md`.

The modal ports are at distance $L_p$ (`port_distance`) beyond the port planes of the core, and the GSM is shifted to
the reference planes with $S_{ij}\to e^{i\beta_iD_p}S_{ij}e^{i\beta_jD_q}$, $D = L_p + \delta$, where $\delta$ is the
extra de-embedding of the outline (non-zero for a miter whose mirror cuts the outer wall before the reference plane;
CONVENTIONS.md §6). The diagonal phases preserve symmetry and, for propagating modes, unitarity. The finite-element
phase error over the arms ($\sim\beta L_p(kh)^{2p}$) enters here; it is part of the discretization error shown by the
convergence study.

For an evanescent mode the shift is $e^{|\beta|L_p}$ per index, and it multiplies the discretization error of the
corresponding entries (the discrete mode decays at a slightly different rate than the exact one). By default the arms
are therefore shortened, when the bases contain evanescent modes, to
$L_p = \ln(G)/(2|\beta|_{\max})$ with $G$ = `evanescent_gain` = 10, and the modal condition then keeps every port
mode up to the first one that decays by `port_decay` = 16 e-folds over the arm. Putting the modal ports exactly on
the reference planes ($L_p = 0$) is not a usable alternative: the inner corner lies on both port lines, and the
truncated modal expansion of the singular field converges slowly in $M$ (WR-90 miter, 30 GHz: the TE10 loss is
0.4428, 0.4469, 0.4486 dB for $M$ = 40, 80, 160, against 0.44985 dB converged).

## 5. Checks

| Check | What it pins down | Result (tests/test_fem2d.py) |
| --- | --- | --- |
| $g$, $h$ from the library fields vs the closed forms of §2, propagating and evanescent | sign and branch of $g$, $h$ | 2.5e-15 relative (test bound 1e-12) |
| Straight channel vs `gsm.straight` | de-embedding phases, LSE transmission sign, $Q$ | 1e-6, signs included |
| Smooth H- and E-plane bends vs wrmodes, 3 frequencies, every propagating mode | sign and gauge of every cross-mode coupling, LSM and LSE, TE/TM rotation | 1e-3 absolute |
| Smooth bends vs exact bend modes (Bessel functions of order $\nu = \beta_{\text{bend}}R$) | the phase of $S_{21}$ beyond the propagating-mode coupled-mode theory | see the report |
| Reciprocity, symmetric miter $S_{11} = S_{22}$ | §3 reciprocity condition, port frames | 1e-10 |

## 6. Impedance (Leontovich) walls (WP3b)

*Added by WP3b. Checked numerically in `tests/test_fem2d.py` (lossy straight channel against the exact
`wrmodes.filters.LossyWallFilter` for families $n = 0$ and $n\ge1$; per-mode attenuation against the exact roots of
T1 box B6; a reversed coupling sign fails these checks).*

### 6.1 The condition on a wall parallel to $v$

T1 box B1: $E_{\tan} = Z_s\,H_{\tan}\times\hat n$, $\hat n$ from the guide into the wall, $Z_s = (1+i)R_s$ for
$e^{+i\omega t}$. On an in-plane wall of a coplanar component (an edge of the outline, extended along $v$) take the
right-handed wall frame $(\hat n, \hat s, \hat v)$, i.e. $\hat s = \hat v\times\hat n$ (in the $(x, y)$ plane of the
outline, with $\hat v = \sigma_v\hat e_3$, $\hat s = \sigma_v(-n_y, n_x)$). Then
$H_{\tan}\times\hat n = H_v\,\hat s - H_s\,\hat v$, so

```latex
E_s = Z_s\,H_v ,\qquad E_v = -Z_s\,H_s .\tag{6.1}
```

Check against T1's form $E_t = ZH_z$, $E_z = -ZH_t$ in the frame $(\hat n, \hat t = \hat z\times\hat n, \hat z)$: on the wall `u+`, $\hat n = \hat u$, $\hat t = \hat v$ and $\hat s = \hat v\times\hat u = -\hat z$, so (6.1) reads $-E_z = ZH_v$ and $E_v = ZH_z$, which is T1's pair.

### 6.2 Fields of one family

In family $n$ ($k_v = n\pi/b$, $k_t^2 = k^2 - k_v^2$), write the field as an LSM part (electric Hertz vector
$\hat v\,P(u,z)\cos k_vv'$) plus an LSE part (magnetic Hertz vector $\hat v\,R(u,z)\sin k_vv'$):

| | LSM: $E = \nabla\nabla\cdot\Pi_e + k^2\Pi_e$, $H = i\omega\varepsilon_0\nabla\times\Pi_e$ | LSE: $H = \nabla\nabla\cdot\Pi_m + k^2\Pi_m$, $E = -i\omega\mu_0\nabla\times\Pi_m$ |
| --- | --- | --- |
| $E_v$ | $k_t^2P\cos k_vv'$ | 0 |
| $H_v$ | 0 | $k_t^2R\sin k_vv'$ |
| $E_s$ | $-k_v\,\partial_sP\,\sin k_vv'$ | $i\omega\mu_0\,\partial_nR\,\sin k_vv'$ |
| $H_s$ | $-i\omega\varepsilon_0\,\partial_nP\cos k_vv'$ | $k_v\,\partial_sR\cos k_vv'$ |

using $\nabla_tP\times\hat v = -\partial_nP\,\hat s + \partial_sP\,\hat n$. The solver's scalars are
$\psi = k_t^2P$ (the $E_v$ profile of §1) and $\phi = k_t^2R$ (the $H_v$ profile). Both satisfy
$\nabla^2 + k_t^2 = 0$, and the walls at $v = \pm b/2$ (perfect conductors) are satisfied by the $\sin/\cos$ factors
for any $\psi,\phi$: **the family index $n$ stays exact** with impedance walls parallel to $v$.

### 6.3 Boundary conditions

Inserting the table into (6.1), the $v$ factors cancel ($\sin k_vv'$ in the first, $\cos k_vv'$ in the second
equation) and, with $\omega\mu_0 = k\eta_0$, $\omega\varepsilon_0 = k/\eta_0$:

```latex
\boxed{\;\partial_n\psi = -\,\frac{i\eta_0}{k}\Big(\frac{k_t^2}{Z_s}\,\psi + k_v\,\partial_s\phi\Big),\qquad
\partial_n\phi = -\,\frac{i}{k\eta_0}\Big(Z_s\,k_t^2\,\phi + k_v\,\partial_s\psi\Big)\;}\tag{6.2}
```

* $Z_s\to0$: $\psi = 0$ (Dirichlet) and $\partial_n\phi = -ik_v\partial_s\psi/(k\eta_0) = 0$ (Neumann): the perfect walls
  of CONVENTIONS.md §7.
* $n = 0$ ($k_v = 0$): family 0 has only LSM modes (TE$_{m0}$), and (6.2) is the **Robin condition**
  $\partial_n\psi = -i\eta_0k\,\psi/Z_s$. At $u = +a/2$ this is T1 box B6 for LSE$^u$ ($Z X' = -ik\eta_0X$), as it
  must be: TE$_{m0}$ is both LSM$^v$ and LSE$^u$.
* $n\ge1$: the two conditions are **coupled** through the tangential derivatives. The split into two scalar
  problems (one Robin condition on $\psi$ for LSM and one on $\phi$ for LSE) is not exact: a pure LSM field has
  $H_v = 0$ but $E_s = -k_v\partial_s P\sin k_vv' \ne 0$ on the wall, which (6.1) forbids. This is why the exact modes
  of a guide with impedance walls at $u = \pm a/2$ are LS$^u$, not LS$^v$ modes (T1 §4–5), and why the low-loss
  eigenmode of the TE$_{mn}$/TM$_{mn}$ pair is close to, but not exactly, LSE$^v$ (T1 §5). The coupling is of order
  $Z_s$ (both $\psi$ on the wall and the $k_v\partial_s\psi$ term are $O(Z_s)$) but resonant between the degenerate
  TE/TM pair, so it cannot be dropped. Reversing its sign changes the TE$_{11}$ and
  TM$_{11}$ attenuation of a 5 cm lossy straight at 30 GHz by 22 % and 120 % against the exact `LossyWallFilter`
  (checked when writing `tests/test_fem2d.py::test_lossy_straight_matches_lossy_wall_filter`).

### 6.4 Weak form and symmetry

With test functions $w$ (for $\psi$) and $q$ (for $\phi$) and the walls $\Gamma_Z$ of the core:

```latex
\int\nabla w\cdot\nabla\psi - k_t^2w\psi - \oint_{\Gamma_Z}w\,\partial_n\psi + \text{ports} = \text{rhs},\qquad
-\oint_{\Gamma_Z}w\,\partial_n\psi = \frac{i\eta_0}{k}\Big(\frac{k_t^2}{Z_s}\oint w\psi + k_v\oint w\,\partial_s\phi\Big),
```

and the same for $\phi$ with $-\oint q\,\partial_n\phi = \frac{i}{k\eta_0}\big(Z_sk_t^2\oint q\phi + k_v\oint q\,\partial_s\psi\big)$.
Scaling the $\psi$ equation by $\kappa_\psi = 1/\eta_0$ and the $\phi$ equation by $\kappa_\phi = -\eta_0$:

```latex
\begin{pmatrix} \tfrac1{\eta_0}A_\psi + \tfrac{ik_t^2}{kZ_s}M_\Gamma & \tfrac{ik_v}{k}C\\[2pt]
-\tfrac{ik_v}{k}C & -\eta_0A_\phi - \tfrac{iZ_sk_t^2}{k}M_\Gamma\end{pmatrix}
\begin{pmatrix}\psi\\ \phi\end{pmatrix} = \begin{pmatrix}\tfrac1{\eta_0}r_\psi\\ -\eta_0r_\phi\end{pmatrix},\qquad
C_{ij} = \oint_{\Gamma_Z}w_i\,\partial_sw_j\,ds ,
```

with $A$ the lossless matrices of §4 (ports included) and $M_\Gamma$ the wall mass matrix. $C_{ij} + C_{ji} =
[w_iw_j]$ at the ends of each wall segment; at a corner between two impedance edges these end terms cancel, and
the impedance walls end where they meet the perfectly conducting arms, where $\psi$ is constrained to 0 (the
end node belongs to the arm wall). So $C$ is antisymmetric on the free $\psi$ unknowns, $-C = C^T$ there, and the
matrix is **complex symmetric**. With $\kappa$ the scaling of each excited port mode,
$S^{2D}_{ij} = 2i\kappa_j\beta_j\,b_i^TA^{-1}b_j - \delta_{ij}$, so $\kappa_i\beta_iS^{2D}_{ij}$ is symmetric. §3 then gives
$S^{LS}_{ij} = \sigma^{out}_i\sigma^{in}_j(f_j/f_i)S^{2D}_{ij}$ ($\sigma = 1$ for LSM; $\sigma^{in} = s_q$,
$\sigma^{out} = -s_p$ for LSE), which is symmetric iff $\sigma^{out}_i\sigma^{in}_i f_i^2\kappa_i\beta_i$ is the
same for every mode. From the closed forms of §2: LSM $g^2\beta/\eta_0 = 4k_t^2/(kb)$, LSE $-\eta_0h^2\beta =
-4k_t^2/(kb)$ times $\sigma^{out}\sigma^{in} = -1$: equal. The scaling $1/\eta_0 : -\eta_0$ that symmetrizes the
discrete coupling is therefore exactly the one that makes the LS amplitudes reciprocal, an independent check of
the relative normalization of $\psi$ and $\phi$ in (6.2).

Passivity: for $\operatorname{Re}Z_s\ge0$ the wall terms are absorbing ($+i\times$positive for real $Z_s$ in the
unscaled equations, the same sign as the outgoing port terms).

### 6.5 Choices in the implementation

* **Arms stay perfect.** The modal ports need the modes of the perfect guide, so only the core walls are impedance
  walls. The GSM is that of the lossy core between perfect guides, the definition of `LossyWallFilter` (T1 box
  B9) and of WP2's lossy straights (whose $\Lambda^-$ junction mismatch is the first-order version of the same
  junction). The jump of the wall condition at the junction makes the field singular there; the mesh is refined
  at these points like at re-entrant corners.
* **Walls at $v = \pm b/2$** are not included: they would couple $n$ with $n\pm2$ (CONVENTIONS.md §3).
* The term $k_t^2/Z_s$ is large for metal walls ($\eta_0k/|Z_s|\sim10^7$ m$^{-1}$); it acts as a penalty that keeps
  $\psi = O(Z_s)$ on the wall. With LU this costs no accuracy: the change of $S$ stays linear in $Z_s$ from
  $10^{-3}$ down to $10^{-9}\,\Omega$ (aluminium: 0.05–0.1 Ω). $Z_s = 0$ is treated as a perfect wall (the lossless
  code path, bitwise unchanged; with $Z_s\ne0$ the mesh is also refined at the ends of the impedance walls, so the
  limit $Z_s\to0$ differs from the lossless result by the discretization error, about $2\times10^{-6}$).
