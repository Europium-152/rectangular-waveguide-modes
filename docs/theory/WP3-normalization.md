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

Weak form (P2 or P3 Lagrange elements, scikit-fem), with $b_m = \int_{\text{port}}w\,\phi_m\,ds$:

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

## 5. Checks

| Check | What it pins down | Result (tests/test_fem2d.py) |
| --- | --- | --- |
| $g$, $h$ from the library fields vs the closed forms of §2, propagating and evanescent | sign and branch of $g$, $h$ | 2.5e-15 relative (test bound 1e-12) |
| Straight channel vs `gsm.straight` | de-embedding phases, LSE transmission sign, $Q$ | 1e-6, signs included |
| Smooth H- and E-plane bends vs wrmodes, 3 frequencies, every propagating mode | sign and gauge of every cross-mode coupling, LSM and LSE, TE/TM rotation | 1e-3 absolute |
| Smooth bends vs exact bend modes (Bessel functions of order $\nu = \beta_{\text{bend}}R$) | the phase of $S_{21}$ beyond the propagating-mode coupled-mode theory | see the report |
| Reciprocity, symmetric miter $S_{11} = S_{22}$ | §3 reciprocity condition, port frames | 1e-10 |
