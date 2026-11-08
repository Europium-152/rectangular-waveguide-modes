# T2 — Coupled local-mode equations of tapers, with curvature, regular through cutoff

*Theory note, wave 1 (card `tasks/T2-taper-coupling.md`). Status: derived; every formula checked numerically by
the scripts in `docs/theory/T2/` (section 12). Machine-checkable formulas: `docs/theory/T2/t2_coefficients.py`,
one function per coefficient.*

## Summary

For a rectangular guide whose cross-section $a(z)\times b(z)$, centred at $(u_0(z), v_0(z))$, moves along a planar
centre line of curvature $\kappa(z)$, the local-mode amplitudes obey, exactly (infinite system),

```latex
\boxed{\;\frac{dV}{dz} = -i\,Z\,I + N^{T} V ,\qquad \frac{dI}{dz} = -i\,Y\,V - N\,I\;}
```

* $N_{pq} = \iint_{S(z)} e_p\cdot\partial_z e_q\,ds$ is the overlap derivative **with the integration domain $S(z)$,
  the cross-section of the projecting mode** (other domain choices give $-N^T$, which differs at first order).
  It is purely geometric, linear in $a', b', u_0', v_0'$, and contains no $\beta$.
* The tilted-wall boundary term ($hE_z = -wE_n$ on a moving PEC wall) cancels exactly against the moving-domain
  (Leibniz) term in the $V$ equation; in the $I$ equation the moving-domain term survives and appears through
  $N + N^T = -W$.
* $Z$ and $Y$ are the T0 curvature matrices evaluated with the local $a(z), b(z)$, plus a diagonal $\kappa u_0$
  term. There are no $\kappa\times$slope cross terms.
* Every coefficient is a polynomial in $\omega$ and $\beta^2$ or a geometric integer formula: the V–I form is
  regular through cutoff. The power-wave form contains $\sqrt{Y_p/Y_q}$ and $Y_p'/Y_p\propto 1/\beta_p^2$ and
  diverges there.
* Cutoff crossings are solved with GSM slices built from V–I propagators (section 8, coded in
  `docs/theory/T2/vi_solver.py`). The truncated system converges as $O(1/M)$ in the number of modes, with a
  coefficient proportional to the wall slope; Richardson extrapolation in $1/M$ is recommended.

## 1. Setting and notation

Frame $(u, v, z)$ of `CONVENTIONS.md` §2, scale factor $h = 1 + \kappa(z)\,u$ (planar curvature in the $u$–$z$
plane; $u$ is measured from the reference curve whose curvature is κ). At each $z$ the cross-section is

```latex
S(z) = \{\,u_0 - a/2 \le u \le u_0 + a/2,\;\; v_0 - b/2 \le v \le v_0 + b/2\,\},
```

with $a, b, u_0, v_0, \kappa$ functions of $z$ (continuous, with piecewise continuous first derivatives; no second
derivatives enter). Local coordinates $\tilde u = u - u_0$, $\tilde v = v - v_0$, $\xi = \tilde u + a/2$,
$\eta = \tilde v + b/2$.

**Local modes.** $e_p(u, v; z)$ is the normalized transverse field of `CONVENTIONS.md` §4 for the local $a(z), b(z)$,
centred on $(u_0, v_0)$: `transverse_field(p, a, b, u - u0, v - v0)`. It is orthonormal over $S(z)$ with the flat
measure $du\,dv$. Outside $S(z)$ it is the analytic continuation of the same sines and cosines (needed only to
define $\partial_z e_q$ at the edges of a shrinking domain). $d_p = \nabla_t\cdot e_p$, $g_p = \nabla_t\cdot(\hat z\times e_p)$
as in T0.

**Walls.** On the contour $\partial S(z)$, $\hat n$ is the outward normal in the cross-section plane and
$\hat s = \hat z\times\hat n$, so that $(\hat n, \hat s, \hat z)$ is right-handed (Katsenelenbaum's orientation).
The **normal velocity** $w(s, z)$ is the outward displacement of the wall per unit $z$ (dimensionless):
on $u = u_0\pm a/2$, $w = \pm u_0' + a'/2$; on $v = v_0\pm b/2$, $w = \pm v_0' + b'/2$. The physical wall slope is
$w/h$ (the length element along $z$ is $h\,dz$).

**Expansion.** $E_t = \sum_q V_q e_q$, $H_t = \sum_q I_q\,\hat z\times e_q$, i.e.
$V_p = \iint_{S(z)} E_t\cdot e_p\,ds$, $I_p = \iint_{S(z)} H_t\cdot(\hat z\times e_p)\,ds$. With power waves
(`CONVENTIONS.md` §5) $V_p = (a^+_p + a^-_p)\sqrt{2/Y_p}$, $I_p = (a^+_p - a^-_p)\sqrt{2Y_p}$.

## 2. Derivation

Start from the transverse Maxwell equations of T0, valid for any $h(u, z)$:
$\partial_z E_t = \nabla_t(hE_z) + i\omega\mu_0 h\,\hat z\times H_t$,
$\partial_z H_t = \nabla_t(hH_z) - i\omega\varepsilon_0 h\,\hat z\times E_t$,
$i\omega\varepsilon_0E_z = \nabla_t\cdot(H_t\times\hat z)$, $-i\omega\mu_0H_z = \nabla_t\cdot(E_t\times\hat z)$.

### 2.1 Moving domain (Leibniz)

Because $S(z)$ moves,

```latex
\frac{dV_p}{dz} = \iint_{S}\partial_z E_t\cdot e_p
               + \iint_{S} E_t\cdot\partial_z e_p
               + \oint_{\partial S} w\,E_t\cdot e_p\,dl .
```

On the wall $e_p\cdot\hat s = 0$ (TM: $\Pi_p = 0$; TE: $\partial_n\Gamma_p = 0$) and $E_s = 0$ (PEC), so
$E_t\cdot e_p = E_n e_{p,n}$ there: **the moving-domain term $\oint w E_n e_{p,n}$ is first order in the slope and
does not vanish**, because normal field components are $O(1)$ at the walls.

### 2.2 The tilted-wall term

The wall is the surface $\{(u, v, z): (u, v)\in\partial S(z)\}$. Its tangent vectors are $\hat s$ and
$h\hat z + w\hat n$ (the displacement for $dz$ at a fixed contour point is $h\,dz\,\hat z + w\,dz\,\hat n$). A PEC wall
has $E\cdot\hat s = 0$ and $E\cdot(h\hat z + w\hat n) = 0$:

```latex
\boxed{\; h\,E_z = -\,w\,E_n \quad\text{on a moving PEC wall}\;}
```

T0 had $w = 0$ and dropped this term. In the projection of $\nabla_t(hE_z)$,

```latex
\iint_S \nabla_t(hE_z)\cdot e_p = \oint hE_z\,e_{p,n}\,dl - \iint_S hE_z\,d_p
 = -\oint w\,E_n e_{p,n}\,dl - \iint_S hE_z\,d_p .
```

The **tilted-wall term** $-\oint wE_ne_{p,n}$ is exactly minus the moving-domain term of 2.1: the two cancel in
the $V$ equation. (Each alone is $O(\text{slope})$; dropping only one of them gives a wrong equation, see 12.2.)
The remaining $\iint hE_z d_p$ is handled as in T0 ($d_p = 0$ on the wall), giving
$\frac{i}{\omega\varepsilon_0}\sum_q I_q\iint h\,d_pd_q$.

### 2.3 The $I$ equation

$dI_p/dz = \iint\partial_zH_t\cdot(\hat z\times e_p) + \iint H_t\cdot(\hat z\times\partial_z e_p) + \oint w\,H_t\cdot(\hat z\times e_p)$.
On the wall $\hat z\times e_p = e_{p,n}\hat s$, so the moving-domain term is $\oint w\,H_s e_{p,n}\,dl$: it involves the
tangential magnetic field, which does not vanish. The boundary term of $\nabla_t(hH_z)$ vanishes because
$(\hat z\times e_p)\cdot\hat n = -e_p\cdot\hat s = 0$, and the one of $\nabla_t\cdot(E_t\times\hat z)$ because
$(E_t\times\hat z)\cdot\hat n = E_s = 0$, on tilted walls too. With $H_s = \sum_q I_q e_{q,n}$ on the wall (the
tangential trace of the series; it converges, slowly, see 8.4) the moving-domain term is $\sum_q W_{pq}I_q$ with

```latex
\boxed{\; W_{pq} = \oint_{\partial S(z)} w\,e_{p,n}\,e_{q,n}\,dl \;}
```

### 2.4 Result

Collecting terms (no other boundary terms appear):

```latex
\frac{dV_p}{dz} = -i\sum_q Z_{pq} I_q + \sum_q N_{qp} V_q ,\qquad
\frac{dI_p}{dz} = -i\sum_q Y_{pq} V_q + \sum_q (N_{qp} + W_{pq}) I_q ,
```

```latex
Z_{pq} = \omega\mu_0\!\iint_S h\,e_p\cdot e_q - \frac{1}{\omega\varepsilon_0}\!\iint_S h\,d_pd_q ,\qquad
Y_{pq} = \omega\varepsilon_0\!\iint_S h\,e_p\cdot e_q - \frac{1}{\omega\mu_0}\!\iint_S h\,g_pg_q .
```

### 2.5 Identity of the moving domain; the integration-domain subtlety

Differentiating $\iint_{S(z)} e_p\cdot e_q = \delta_{pq}$ with the Leibniz rule gives

```latex
\boxed{\; N + N^{T} + W = 0 \;}
```

so $N_{qp} + W_{pq} = -N_{pq}$, and the system takes the boxed form of the summary,
$dV/dz = -iZI + N^TV$, $dI/dz = -iYV - NI$. The symmetric part of $N$ is a pure wall term, $-W/2$; the
antisymmetric part is the "interior" coupling.

**Which overlap derivative.** With $O_{pq}(z_1, z_2) = \iint_{S(z_1)} e_p(z_1)\cdot e_q(z_2)\,ds$ (domain: the
cross-section of the projecting mode),

* $\partial O_{pq}/\partial z_2\,|_{z_1=z_2=z} = N_{pq}$ — the coefficient of this note;
* the same derivative with the domain $S(z_2)$ (the cross-section of the differentiated mode) is $-N_{qp}$;
* $\partial O_{pq}/\partial z_1$ with domain $S(z_1)$ is $-N_{pq}$.

$N$ and $-N^T$ differ by $W$, which is first order in the slope (in the check of 12.1, $\max|W|/\max|N| = 0.42$
and $0.79$). A step-by-step mode-matching code that takes the overlap on the wrong side converges to the wrong
coefficient.

## 3. Closed forms

**Polarization vectors.** With orthonormal 1D functions $\phi^s_m = \sqrt{2/a}\sin(m\pi\xi/a)$ ($m\ge1$),
$\phi^c_m = \sqrt{\epsilon_m/a}\cos(m\pi\xi/a)$, and $\psi^s_n, \psi^c_n$ the same in $\eta$ over $b$, every mode is

```latex
e_p = \big(t^p_u\,\phi^c_m\psi^s_n,\;\; t^p_v\,\phi^s_m\psi^c_n\big),\qquad
t^{TM} = (\cos\theta, \sin\theta),\quad t^{TE} = (-\sin\theta, \cos\theta),\quad
\cos\theta = k_m/\chi,\ \sin\theta = k_n/\chi ,
```

$k_m = m\pi/a$, $k_n = n\pi/b$, $\chi^2 = k_m^2 + k_n^2$ (this reproduces `transverse_field`, signs included; for
$n = 0$ or $m = 0$ the vanishing component has $t = 0$). Also $\Pi_p = \phi^s_m\psi^s_n/\chi$,
$\Gamma_p = \phi^c_m\psi^c_n/\chi$, $d_p = -\chi\,\phi^s_m\psi^s_n$ (TM), $g_p = -\chi\,\phi^c_m\psi^c_n$ (TE).
$\partial_a\theta = \sin\theta\cos\theta/a$, $\partial_b\theta = -\sin\theta\cos\theta/b$.

**1D matrices** (dimensionless; indices $j$ = row mode, $l$ = column mode; $\epsilon_0 = 1$, $\epsilon_{j>0} = 2$):

```latex
\boxed{\begin{aligned}
S^{w}_{jl} &= a\langle\phi^s_j, \partial_a\phi^s_l\rangle = \frac{2jl}{j^2-l^2} \quad (j+l \text{ even},\ j\ne l;\ 0 \text{ otherwise})\\
C^{w}_{jl} &= a\langle\phi^c_j, \partial_a\phi^c_l\rangle = \frac{\sqrt{\epsilon_j\epsilon_l}\,l^2}{j^2-l^2}\quad (j+l\text{ even},\ j\ne l),\qquad C^w_{jj} = -\frac{\epsilon_j}{2},\qquad 0\ (j+l\text{ odd})\\
S^{sh}_{jl} &= a\langle\phi^s_j, \partial_{u_0}\phi^s_l\rangle = -\frac{4jl}{j^2-l^2}\quad (j+l\text{ odd};\ 0\text{ otherwise})\\
C^{sh}_{jl} &= a\langle\phi^c_j, \partial_{u_0}\phi^c_l\rangle = -\frac{2\sqrt{\epsilon_j\epsilon_l}\,l^2}{j^2-l^2}\quad (j+l\text{ odd};\ 0\text{ otherwise})
\end{aligned}}
```

($\partial_a$ at fixed $u$ and $u_0$: symmetric widening; $\partial_{u_0}$ at fixed $u$: rigid shift.) The same
matrices with $n$ and $b$ serve side $b$.

**Overlap derivative.** $N = a'N^a + b'N^b + u_0'N^u + v_0'N^v$ with ($p = (m, n)$, $q = (m', n')$)

```latex
\boxed{\begin{aligned}
N^a_{pq} &= \delta_{nn'}\Big[\tfrac1a\big(t^p_ut^q_u\,C^w_{mm'} + t^p_vt^q_v\,S^w_{mm'}\big)
           + \delta_{mm'}\,\sigma_{pq}\,\tfrac{\sin\theta\cos\theta}{a}\Big]\\
N^b_{pq} &= \delta_{mm'}\Big[\tfrac1b\big(t^p_ut^q_u\,S^w_{nn'} + t^p_vt^q_v\,C^w_{nn'}\big)
           - \delta_{nn'}\,\sigma_{pq}\,\tfrac{\sin\theta\cos\theta}{b}\Big]\\
N^u_{pq} &= \delta_{nn'}\,\tfrac1a\big(t^p_ut^q_u\,C^{sh}_{mm'} + t^p_vt^q_v\,S^{sh}_{mm'}\big)\\
N^v_{pq} &= \delta_{mm'}\,\tfrac1b\big(t^p_ut^q_u\,S^{sh}_{nn'} + t^p_vt^q_v\,C^{sh}_{nn'}\big)
\end{aligned}}
```

with $\sigma_{pq} = +1$ for $(p, q) = (\text{TE}_{mn}, \text{TM}_{mn})$, $-1$ for $(\text{TM}_{mn}, \text{TE}_{mn})$, $0$
otherwise (the rotation $\partial_z t^{TM} = \theta' t^{TE}$, $\partial_z t^{TE} = -\theta' t^{TM}$), and
$t^p$ evaluated with $(m, n)$, $t^q$ with $(m', n')$. There are no $1/(\chi_p^2-\chi_q^2)$ denominators: the
formulas stay finite for degenerate pairs (TE$_{mn}$/TM$_{mn}$, or accidental degeneracies when $a/b$ varies).

**Wall term** (equivalently $W = -N - N^T$):

```latex
W_{pq} = \delta_{nn'}\,t^p_ut^q_u\,\frac{\sqrt{\epsilon_m\epsilon_{m'}}}{a}\Big[\tfrac{a'}{2}(1+(-1)^{m+m'}) + u_0'((-1)^{m+m'}-1)\Big]
       + \delta_{mm'}\,t^p_vt^q_v\,\frac{\sqrt{\epsilon_n\epsilon_{n'}}}{b}\Big[\tfrac{b'}{2}(1+(-1)^{n+n'}) + v_0'((-1)^{n+n'}-1)\Big].
```

**Curvature integrals** (T0's $U$, $U^d$, $U^g$ with the centred $\tilde u$, local $a, b$):

```latex
\boxed{\begin{aligned}
U_{pq} &= \iint\tilde u\,e_p\cdot e_q = \delta_{nn'}\big(t^p_ut^q_u\,G^c_{mm'} + t^p_vt^q_v\,G^s_{mm'}\big),\quad
U^d_{pq} = \chi_p\chi_q\,G^s_{mm'}\delta_{nn'}\ (\text{TM, TM}),\quad U^g_{pq} = \chi_p\chi_q\,G^c_{mm'}\delta_{nn'}\ (\text{TE, TE}),\\
G^s_{jl} &= -\frac{8a\,jl}{\pi^2(j^2-l^2)^2},\qquad G^c_{jl} = -\frac{2a\sqrt{\epsilon_j\epsilon_l}\,(j^2+l^2)}{\pi^2(j^2-l^2)^2}\qquad(j+l\text{ odd; else }0).
\end{aligned}}
```

**V–I coefficients.** With $h_0 = 1 + \kappa u_0$ (the scale factor on the local centre line),

```latex
\boxed{\begin{aligned}
Z_{pq} &= h_0\,\delta_{pq}\Big(\omega\mu_0 - [p\in\text{TM}]\frac{\chi_p^2}{\omega\varepsilon_0}\Big)
         + \kappa\Big(\omega\mu_0\,U_{pq} - \frac{U^d_{pq}}{\omega\varepsilon_0}\Big)\\
Y_{pq} &= h_0\,\delta_{pq}\Big(\omega\varepsilon_0 - [p\in\text{TE}]\frac{\chi_p^2}{\omega\mu_0}\Big)
         + \kappa\Big(\omega\varepsilon_0\,U_{pq} - \frac{U^g_{pq}}{\omega\mu_0}\Big)
\end{aligned}}
```

so that $Z_{pp} = h_0\omega\mu_0$, $Y_{pp} = h_0\beta_p^2/(\omega\mu_0)$ (TE) and $Z_{pp} = h_0\beta_p^2/(\omega\varepsilon_0)$,
$Y_{pp} = h_0\omega\varepsilon_0$ (TM): $\beta$ enters only as $\beta^2$.

Python: `t_vector`, `S_width`, `C_width`, `S_shift`, `C_shift`, `G_s`, `G_c`, `N_a`, `N_b`, `N_u`, `N_v`,
`overlap_derivative`, `wall_term`, `U`, `Ud`, `Ug`, `impedance`, `admittance`, `vi_matrix`.

## 4. Power-wave form

With $D = \operatorname{diag}\sqrt{Y_p}$ (principal roots), $\mathcal Z = DZD$, $\mathcal Y = D^{-1}YD^{-1}$,
$P = DN^TD^{-1}$, $Q = D^{-1}ND$, $\Lambda = \operatorname{diag}(Y_p'/2Y_p)$:

```latex
\boxed{\begin{aligned}
\frac{da^+}{dz} &= \Big[\tfrac12(P-Q) - \tfrac i2(\mathcal Y+\mathcal Z)\Big]a^+ + \Big[\Lambda + \tfrac12(P+Q) - \tfrac i2(\mathcal Y-\mathcal Z)\Big]a^-\\
\frac{da^-}{dz} &= \Big[\Lambda + \tfrac12(P+Q) + \tfrac i2(\mathcal Y-\mathcal Z)\Big]a^+ + \Big[\tfrac12(P-Q) + \tfrac i2(\mathcal Y+\mathcal Z)\Big]a^-
\end{aligned}}
```

Element-wise:
$\tfrac12(P\mp Q)_{pq} = \tfrac12\big[N_{qp}\sqrt{Y_p}/\sqrt{Y_q} \mp N_{pq}\sqrt{Y_q}/\sqrt{Y_p}\big]$ and

```latex
\Lambda_p = \frac{Y_p'}{2Y_p} = \mp\frac{(\chi_p^2)'}{4\beta_p^2}\quad(\text{TE}: -,\ \text{TM}: +),\qquad
(\chi_p^2)' = -2\Big(k_m^2\frac{a'}{a} + k_n^2\frac{b'}{b}\Big).
```

The $\kappa$ parts of $-\tfrac i2(\mathcal Y\pm\mathcal Z)$ are $-i\kappa C^\pm$ of T0 (with the local $a, b$); the
diagonal of $-\tfrac i2(\mathcal Y+\mathcal Z)$ is $-ih_0\beta_p$. Python: `K_ff`, `K_fb`, `K_bf`, `K_bb`,
`power_wave_matrix`, `log_admittance_derivative`.

**Singular at cutoff.** $\sqrt{Y_p/Y_q}$ and $\Lambda_p\propto1/\beta_p^2$ diverge when $\beta_p\to0$ (for the
H-plane wedge of 12.2, $\Lambda_{TE30}$ is $-54$, $-684$ and $+48$ m⁻¹ at $\beta/k = -0.48i$, $-0.12i$, $0.39$). The V–I
coefficients stay finite. The power-wave form is only for propagating modes away from cutoff.

**Lossless structure.** For propagating modes $D$ is real, so $\tfrac12(P-Q)$ is real antisymmetric,
$\Lambda + \tfrac12(P+Q)$ is real symmetric, $\mathcal Y, \mathcal Z$ are real symmetric. Hence with
$G = \operatorname{diag}(I, -I)$ the matrix $K$ of $d[a^+;a^-]/dz = K[a^+;a^-]$ satisfies $K^HG + GK = 0$:
$\frac{d}{dz}(|a^+|^2 - |a^-|^2) = 0$, the net power flux is conserved. In V–I form, for **all** modes including
evanescent ones, $M = [[N^T, -iZ], [-iY, -N]]$ satisfies $M^H G' + G'M = 0$ with $G' = [[0, I], [I, 0]]$
(conservation of $\operatorname{Re}\sum V_pI_p^* = 2P$, the Poynting flux), because $Z, Y, N$ are real and $Z, Y$
symmetric. Reciprocity: $M^TJ + JM = 0$ with $J = [[0, I], [-I, 0]]$ ($M\in\mathfrak{sp}(2n,\mathbb C)$), so
$V_1^TI_2 - V_2^TI_1$ is conserved and every GSM computed from these equations is symmetric. Both identities need
only $Z = Z^T$, $Y = Y^T$ and the same $N$ in both equations; they hold for every truncation.

## 5. Selection rules and family structure

From the Kronecker deltas and the parity conditions of the 1D matrices (also $W$, $U$):

| Term | Couples | Within a family ($n = n'$)? |
| --- | --- | --- |
| $a'$ (symmetric width change) | $m+m'$ even (Δm even, Δm = 0 for TE$_{mn}$↔TM$_{mn}$), $n = n'$ | yes |
| $u_0'$ (offset along $u$) | $m+m'$ odd, $n = n'$ | yes |
| $\kappa$ (curvature in $u$–$z$) | $m+m'$ odd, $n = n'$ | yes |
| $b'$ (symmetric height change) | $n+n'$ even (Δn even), $m = m'$ | **no**: couples families $n$ and $n\pm2, \dots$ |
| $v_0'$ (offset along $v$) | $n+n'$ odd, $m = m'$ | **no** |

$a'$, $u_0'$, κ conserve the family structure of `CONVENTIONS.md` §3; a component with $b' \ne 0$ or $v_0'\ne0$
must use a basis with several families. For TE$_{10}$ incident on a symmetric taper ($u_0' = v_0' = \kappa = 0$),
only $m$ odd, $n$ even are reached (TE$_{30}$, TE$_{50}$ by $a'$; TE$_{12}$, TM$_{12}$, TE$_{14}$, … by $b'$; TE$_{32}$,
TM$_{32}$ … in second order). This is the statement of Butterweck & de Ronde (1968, p. 96) that "H$_{mn}$ modes with
$m$ even or $n$ odd cannot be excited"; their remark that height variation excites only modes with small amplitudes
is consistent with the large $\Delta\beta$ of the $n = 2$ modes. In the H-plane taper TE$_{m0}$ modes have $t_u = 0$
and $e_n = 0$ on the moving walls, so $W = 0$ and $N$ is antisymmetric there; in an E-plane taper $W\ne0$.

## 6. Curvature and taper together (strangled bend)

* The curvature coefficients take the local $a(z), b(z)$ (in $U$, $U^d$, $U^g$, χ and the $t$ vectors), with $u$
  measured from the local centre; an offset of the centre from the reference curve adds only the diagonal factor
  $h_0 = 1 + \kappa u_0$ (the propagation phase is accumulated along the actual centre line).
* No other cross terms: $N$ does not depend on κ, and $Z, Y$ do not depend on the slopes. In the power-wave form
  the coupling is the sum $K = K_{\text{uniform}}(a, b) + \kappa K_{\text{curv}}(a, b) + K_{\text{taper}}(a', b', u_0', v_0'; a, b)$
  with no products. (Derivation: the tilted-wall identity $hE_z = -wE_n$ carries the local $h$ on the wall, and
  it cancels against the Leibniz term pointwise, whatever $h$ is.)
* An asymmetric width change (one wall fixed) is $a'$ plus $u_0' = \pm a'/2$.
* Validity: $h > 0$ on the cross-section, $|\kappa|(|u_0| + a/2) < 1$.

## 7. Uniform-guide limit

For $a' = b' = u_0' = v_0' = 0$: $N = W = \Lambda = 0$, and with $u_0 = 0$ the power-wave matrix equals the T0
matrix $D + \kappa K$ of `coupled_mode_matrices` (checked to 6e-16, 12.1). For $\kappa = 0$ as well, each mode obeys
$dV/dz = -i\beta I/Y$, $dI/dz = -i\beta YV$.

## 8. Cutoff crossings: method for WP5

### 8.1 Why the V–I form is regular

$Z$ and $Y$ are polynomials in $\omega$ with geometric coefficients and contain β only as $\beta^2 = k^2-\chi^2$;
$N$ is geometric. The ODE $d[V;I]/dz = M(z)[V;I]$ therefore has coefficients as smooth as $a, b, u_0, v_0, \kappa$,
and its solutions pass through a cutoff plane without any special treatment. Near a turning point the single-mode
equation is Airy's ($V'' + \beta^2V = 0$ with $\beta^2$ linear in $z$, TE; $I'' + \beta^2I = 0$, TM). What is
singular at cutoff is only the *conversion* to power waves, and only at a plane where it is done.

### 8.2 Algorithm (GSM slices with V–I propagators)

1. **Mode list.** One ordered list $\mathcal M$ for the whole component, restricted by the selection rules of
   section 5 to the families/parities reachable from the port modes:
   $\mathcal M(f_c) = \{p : \min_{z\in[0,L]}\chi_p(z) < f_c\,k\}$, i.e. the cutoff factor $f_c\ge1$ is applied at the
   **widest** cross-section. A mode that propagates anywhere inside is therefore always kept, even if it is
   evanescent at both ports (it can carry power between two cutoff planes; the taper can trap it).
2. **Slices.** $z_0 = 0 < z_1 < \dots < z_S = L$ with (a) no interior boundary near a cutoff plane:
   $|k^2-\chi_p(z_j)^2|\ge\delta_c k^2$ for all $p\in\mathcal M$ ($\delta_c = 10^{-6}$: the condition number of $D$ is then at most $\sim\delta_c^{-1/4}$; move the boundary by a
   fraction of the slice if needed; the cutoff planes are the roots of $\chi_p(z) = k$); (b) bounded evanescent growth:
   $\max_p\operatorname{Im}|\beta_p|\,(z_{j+1}-z_j)\le\gamma_{\max}$ ($\gamma_{\max} = 3$), splitting a slice if needed;
   (c) accuracy: refine until the GSM changes by less than the target (4th order in the slice length).
3. **Propagator per slice.** $\Phi_j$ with $[V;I](z_{j+1}) = \Phi_j[V;I](z_j)$ by the 4th-order Magnus method,
   two Gauss points per sub-step ($h$ = sub-step, $c = \sqrt3/6$):
   $\Omega = \tfrac h2(M_1+M_2) + \tfrac{\sqrt3}{12}h^2[M_2, M_1]$, $M_{1,2} = M(z + (\tfrac12\mp c)h)$, $\Phi = \prod\exp\Omega$.
   $\Omega$ stays in the Lie algebra of section 4, so each $\Phi_j$ is exactly reciprocal and power-conserving.
4. **Slice GSM, built directly** (no global transfer matrix). With $P^\pm(z) = [\,D(z),\ \pm D(z)^{-1}]/(2\sqrt2)$
   ($a^\pm = P^\pm[V;I]$, $D = \operatorname{diag}\sqrt{Y_p(z)}$ with the local admittances, evanescent included):

   ```latex
   S_j = \begin{bmatrix}P^-(z_j)\\ P^+(z_{j+1})\Phi_j\end{bmatrix}
         \begin{bmatrix}P^+(z_j)\\ P^-(z_{j+1})\Phi_j\end{bmatrix}^{-1},
   ```

   with port bases `ModeBasis(a(z_j), b(z_j), M, sort=False)` and `ModeBasis(a(z_{j+1}), b(z_{j+1}), M, sort=False)`.
   Condition (b) keeps the inverted matrix well conditioned; (a) keeps $D$ and $D^{-1}$ bounded.
5. **Cascade** the slice GSMs with the Redheffer star product (`wrmodes.gsm.cascade`).
6. **Ports.** The component's port bases are its default bases (`ModeBasis.propagating(f, a, b, cutoff_factor)` at
   the end planes, same families). Modes of $\mathcal M$ outside a port basis are kept inside the component and
   dropped at that port (zero incident wave, outgoing amplitude discarded). A port plane exactly at (within
   $\delta_c$ of) the cutoff of a basis mode is an error: the amplitude of that mode is undefined there
   (`CONVENTIONS.md` §3).

The prototype `docs/theory/T2/vi_solver.py` implements steps 1–5 in about 100 lines.

### 8.3 Alternative: Riccati

The admittance $\mathcal Y_{\rm in}$ ($I = \mathcal Y_{\rm in}V$, looking towards $+z$) obeys
$\mathcal Y_{\rm in}' = -iY - N\mathcal Y_{\rm in} - \mathcal Y_{\rm in}N^T + i\,\mathcal Y_{\rm in}Z\,\mathcal Y_{\rm in}$,
integrated from $z = L$ ($\mathcal Y_{\rm in} = \operatorname{diag}Y_p(L)$, matched) towards $z = 0$; integrating in
this direction is stable for the evanescent modes. Its solutions have poles where a standing wave has a node of
$V$ (total reflection at a cutoff plane produces them), so it must be written for a reflection matrix with a
*constant, real, positive* reference admittance $Y_r$, $\Gamma = (I - \tilde y)(I+\tilde y)^{-1}$,
$\tilde y = Y_r^{-1/2}\mathcal Y_{\rm in}Y_r^{-1/2}$, which is a contraction for a passive load and therefore bounded.
It gives $S_{11}$ only; $S_{21}$ needs a second, linear sweep. The GSM-slice method is preferred because it
returns all four blocks and reuses the cascade.

### 8.4 Mode count and convergence

The equations are exact for the infinite system; the truncated system converges **algebraically**, as $O(1/M)$
in the number $M$ of modes per family (per index direction), with a coefficient proportional to the wall slope.
The cause is the Gibbs behaviour of the $H$ expansion: on a tilted wall $H_n = (w/h)H_z\ne0$, while every
basis field $\hat z\times e_q$ has zero normal component there, so the $I_q$ of the true field decay slowly and the
sums $\sum_q N_{pq}I_q$ converge as $1/M$. (The $V$ sums converge as $M^{-3}$: $E_t$ has no such mismatch.)
Observed (section 12): residual of the $I$ equation for exact fields $\propto M^{-1.0}$ in all seven configurations;
GSM of an H-plane wedge with $a' = 0.41$ (`study_convergence.py`, low-order blocks TE$_{1..4,0}$ at both ports):
successive differences $\max|S(2M) - S(M)|$ = 3.6e-2, 1.7e-2, 8.5e-3, 4.3e-3 for $M$ = 8→16→32→64→128, i.e. $S(M)$
is still about 4e-3 from its limit at $M = 128$.

Protocol for WP5:

* Count modes by $f_c$ (step 1). Report the result for at least two values, $f_c$ and $\approx 2f_c$ (which roughly
  doubles $M$), and the **Richardson extrapolation** $S_\infty\approx 2S(2M) - S(M)$ (for unequal counts,
  $S_\infty \approx (M_2S_2 - M_1S_1)/(M_2-M_1)$). On the same wedge the successive
  differences of the extrapolated values are 2.1e-3, 2.9e-4, 5.8e-5 ($R(32)-R(16)$, $R(64)-R(32)$, $R(128)-R(64)$),
  against 1.7e-2, 8.5e-3, 4.3e-3 raw: one Richardson step gains about two orders of magnitude at $M = 64$.
* For gentle tapers the error is small: for the smooth H-plane taper of 12.3 (slope ≤ 0.27, TE$_{30}$ crossing
  cutoff), the TE$_{10}\to$TE$_{30}$ conversion changes by 2.4e-5 (absolute) between $f_c = 3$ and $4$.
* $f_c = 1$ is not enough in general: it drops modes that are evanescent everywhere but carry the near field of
  the tilted walls.

### 8.5 Total reflection at a cutoff plane

A mode that reaches its cutoff plane $z_c$ is totally reflected. For a TE mode with $N_{qq} = 0$ (H-plane taper)
the turning-point (Airy) connection gives, at a plane $z > z_c$ where it propagates,

```latex
a^+_q(z) = i\,e^{-2i\int_{z_c}^{z}\beta_q\,dz'}\;a^-_q(z),
```

up to $O(\text{slope})$ corrections and conversion to other modes. (For TM modes the same argument applied to $I$,
which obeys Airy's equation there, gives $-i$; this was not checked numerically.) Checked with the solver: $|S_{22}| = 0.99998$,
phase within 0.013 rad of this formula for TE$_{30}$ in a linear H-plane taper with $a' = 0.028$ (12.3). This is
the "reflection at the cut-off plane" of Butterweck & de Ronde (p. 96).

## 9. Small-slope limit (single scattering)

To first order in the slopes and the curvature, with $\phi_p(z_1, z_2) = \int_{z_1}^{z_2}\beta_p\,dz$ and both
modes propagating on $[0, L]$, a unit forward mode $p$ incident at $z = 0$ produces

```latex
\boxed{\begin{aligned}
a^+_q(L) &\approx \int_0^L K^{++}_{qp}(z)\,e^{-i\phi_p(0,z)}\,e^{-i\phi_q(z,L)}\,dz ,\\
a^-_q(0) &\approx -\int_0^L K^{-+}_{qp}(z)\,e^{-i\phi_p(0,z)}\,e^{-i\phi_q(0,z)}\,dz ,
\end{aligned}}
```

with $K^{++}_{qp} = \tfrac12[N_{pq}\sqrt{Y_q/Y_p} - N_{qp}\sqrt{Y_p/Y_q}] - i\kappa C^+_{qp}$ and
$K^{-+}_{qp} = \Lambda_q\delta_{qp} + \tfrac12[N_{pq}\sqrt{Y_q/Y_p} + N_{qp}\sqrt{Y_p/Y_q}] + i\kappa C^-_{qp}$
($p\ne q$ for the first line; $q = p$ allowed for the second). Python: `first_order_conversion`. For an
H-plane taper (TE$_{m0}$, $N$ antisymmetric) the forward coupling is
$K^{++}_{qp} = -\frac{a'}{a}\,\frac{2mm'}{m'^2-m^2}\,\frac{\beta_m+\beta_{m'}}{2\sqrt{\beta_m\beta_{m'}}}$
($q = $ TE$_{m'0}$, $p = $ TE$_{m0}$) — proportional to the wall slope, as in Katsenelenbaum (§1.3.4) and Huang
(vol. II, eqs. 11-225, 11-258). Checked against the full solution: 0.26 % on TE$_{10}\to$TE$_{30}$ for a linear
taper 25→35 mm, $L = 0.15$ m at 20 GHz (12.3).

**Bottle shape.** The phase $\phi_p(0,z)+\phi_q(z,L)$ makes contributions from different $z$ interfere. As $a$ grows,
$\beta_1-\beta_3\to0$ and the integrand loses its oscillation, so the wide end must have the smallest slope
$a'$: this is the "bottle" profile of Butterweck & de Ronde (pp. 95–96). With a cutoff plane inside the taper, the
backward wave of mode $q$ generated near the narrow end is reflected at its cutoff plane (8.5) and returns as a
forward wave; its accumulated phase contains $\beta_p + \beta_q$ instead of $\beta_p - \beta_q$, so these secondary
contributions interfere destructively much more strongly and stay negligible, as Butterweck & de Ronde state.
The first-order formula is not valid across a cutoff plane of mode $p$ or $q$ (it
contains $1/\sqrt{\beta}$); use the full equations there.

## 10. Closed-form special cases

**Single mode, H-plane taper (TE$_{10}$).** $N_{11} = 0$, so the single-mode equations are
$dV/dz = -i\omega\mu_0I$, $dI/dz = -i(\beta^2/\omega\mu_0)V$, i.e. $V'' + \beta(z)^2V = 0$, and the power-wave
reflection density is $K^{-+}_{11} = \Lambda_1 = \beta'/(2\beta)$. This is Katsenelenbaum's result for a mode without
longitudinal wall currents, $P_{-m} = \delta h_m/(2h_m)$ (eq. 1.54, p. 39), whose backward waves have reversed $E_t$
(eq. I.6, p. 13; wrmodes backward waves keep $E_t$, so the reflection coefficients differ in sign and agree).
For a linear taper $a = a_0 + \alpha z$ the solutions are $V = \sqrt a\,Z_\nu(ka/\alpha)$,
$\nu^2 = \tfrac14 + \pi^2/\alpha^2$. The solver reproduces the exact single-mode GSM to 2.6e-9. (The exact
wedge field is a cylindrical wave of order $\pi/\varphi_0$, $\varphi_0 = 2\arctan(\alpha/2)$; it differs from ν at
$O(\alpha)$ because the single-mode truncation neglects TE$_{30}$, TE$_{50}$, ….)

**Single mode, E-plane taper (TE$_{10}$).** $N_{11} = -b'/(2b)$ (from $C^w_{00} = -1/2$), $\Lambda_1 = 0$, so the
reflection density is $-b'/(2b)$: TE$_{10}$ in an E-plane taper behaves as a line with characteristic impedance
$\propto b$ (at constant β).

**Through cutoff.** With $a(z) = \pi/\sqrt{k^2 - g(z - z_c)}$, $\beta^2 = g(z-z_c)$ exactly and the TE$_{10}$
single-mode equation is Airy's. The solver reproduces the exact GSM (port 1 below cutoff, port 2 above) to
1.6e-9 with 160 slices (4th-order convergence: 4.2e-7, 2.6e-8, 1.6e-9 for 40, 80, 160 slices).

## 11. Literature

* **Katsenelenbaum (1998).** §1.3 (pp. 35–41): equivalent boundary condition of a deformed wall and the
  step/taper coupling as a contour integral of the wall slope $v(s)$ (eq. 1.57). §2.3 (pp. 81–91): coupling
  coefficients of the cross-section method, eq. 2.95
  $S_{jm} = \oint v(E^j_nE^m_n + H^j_zH^m_z - H^j_sH^m_s)\,ds/(2h_j(h_j-h_m))$, closed forms 2.98a–c, and Kovalev's
  derivation (eqs. 2.100–2.103) in which the only wall term is $\oint vH_tE^m_n$ — the tangential component
  $E^m_t = E^m_nv$ of the reference field on the tilted wall, the counterpart of our tilted-wall term. Our
  coefficients are written as cross-section integrals with integer denominators instead of contour integrals with
  $h_j - h_m$ denominators (his §2.3.4 notes the 0/0 at degeneracies, which our form avoids). The normalization of
  his eigenfunctions (eq. 1.16) is not legible in the scan, so the coefficients were not compared term by term;
  the structure (wall-slope weighting, reflection density 1.54) agrees.
* **Butterweck & de Ronde (1968), pp. 95–97.** Bottle taper, reflection of the unwanted mode at its cutoff plane,
  excitable modes of a symmetric taper: all consistent with sections 5, 8.5 and 9.
* **Huang, vol. II (1965).** §11.17 (pp. 489–494) treats tapers ("过渡器") for circular H$_{01}$ guides only, in the
  weak-coupling approximation: eqs. 11-223–11-226 (single scattering,
  $A_i = \int K_{1i}e^{-i\int\Delta\beta}$, the same structure as section 9), with a coupling proportional to the slope,
  $c = c_0(a)\,da/dz$, $c_0 = 2\chi_1\chi_2/((\chi_2^2-\chi_1^2)a)$ (11-258, 11-259), and a Fourier-transform
  synthesis of the taper shape. §11.14 (sector-shaped launcher, pp. 474–479) introduces the "critical cross-section"
  (临界截面, where a mode reaches cutoff) and treats the integral by stationary phase (11-236–11-238), citing the
  coupling formula of §8.11 (vol. I) with the wall slope tan θ. Rectangular tapers and the V–I form are not treated.

## 12. Verification

All numbers below come from the scripts (run with `PYTHONPATH=src python3 docs/theory/T2/<script>`).

### 12.1 `check_closed_forms.py` — closed forms, domain, uniform limit

| Check | Result | Target |
| --- | --- | --- |
| 1D matrices $S^w, C^w, S^{sh}, C^{sh}, G^s, G^c$ vs sympy (exact integrals), $j, l = 0..4$ | equal to 1e-13 | exact |
| $N$ (all four slopes at once, 31 modes TE/TM $m\le4$, $n\le3$) vs quadrature of $e_p\cdot\partial_ze_q$ | 1.6e-12, 2.2e-12 (two geometries) | 1e-8 |
| $N$ vs numerical $\partial_{z_2}$ of the overlap on domain $S(z_1)$ (projecting mode) | 1.5e-12, 2.3e-12 | 1e-8 |
| same derivative on domain $S(z_2)$ equals $-N^T$ | 1.3e-12, 2.6e-12 | 1e-8 |
| $\partial_{z_1}$ of the overlap on domain $S(z_1)$ equals $-N$ | 1.3e-12, 2.6e-12 | 1e-8 |
| size of the domain effect, $\max|N(S(z_2)) - N(S(z_1))|/\max|N| = \max|W|/\max|N|$ | 0.42, 0.79 | (first order) |
| $W$ vs contour quadrature; identity $N + N^T + W = 0$ | 3e-15; 2e-16 | 1e-10; 1e-12 |
| $U$, $U^d$, $U^g$ vs quadrature | 1.1e-14, 7.7e-15, 1.9e-14 | 1e-10 |
| uniform limit: power-wave matrix = T0 $D + \kappa K$ (`coupled_mode_matrices`), 4 guides/families | ≤ 6.3e-16 | 1e-10 |

### 12.2 `check_exact_fields.py` — full equations and tilted-wall term against exact Maxwell fields

Exact fields (`exact_fields.py`): an E-plane wedge (Hankel waves, TE-to-$u$ and TM-to-$u$ potentials, tilted walls
with $E_n\ne0$); an H-plane wedge with Bessel standing waves, at three planes around the cutoff plane of TE$_{30}$
(evanescent, near cutoff, propagating); a straight guide seen in a curved frame ($R = 0.12$ m, $\kappa$, $a'$,
$u_0'$ together); straight guides tilted in the $v$–$z$ and $u$–$z$ planes ($v_0'$, $u_0'$). $V_p, I_p$ by quadrature on
each cross-section, $dV/dz$, $dI/dz$ by 4th-order differences, residuals of the truncated equations per test mode
(relative to the largest term of that mode's equation):

| Geometry | V-eq residual, largest truncation (modes) | rate | I-eq residual | rate | without tilted-wall / domain term |
| --- | --- | --- | --- | --- | --- |
| E-plane wedge, $b' = 0.30$ | 1.2e-5 (129) | $M^{-3.0}$ | 1.6e-3 | $M^{-1.0}$ | 2.4e2 / 0.73 (no convergence) |
| H-plane wedge, $a' = 0.41$, TE$_{30}$ evanescent ($\beta/k = -0.48i$) | 1.1e-4 (121) | $M^{-3.0}$ | 2.8e-2 | $M^{-1.0}$ | 2e4 / 1.6 |
| same, near cutoff ($\beta/k = -0.12i$) | 1.6e-5 (121) | $M^{-3.0}$ | 5.2e-2 | $M^{-1.0}$ | 3e4 / 1.7 |
| same, propagating ($\beta/k = 0.39$) | 9.5e-5 (121) | $M^{-3.0}$ | 0.40 | $M^{-1.0}$ | 4e4 / 1.9 |
| straight guide in a curved offset frame ($\kappa = 8.3$, $a' = 0.050$, $u_0' = 0.27$) | 1.1e-4 (121) | $M^{-2.9}$ | 5.5e-2 | $M^{-1.0}$ | 2.2 / 1.9 |
| guide tilted in $v$–$z$ ($v_0' = 0.26$) | 3.8e-5 (220) | $M^{-3.1}$ | 0.26 | $M^{-1.0}$ | 1.6 / 3.9 |
| guide tilted in $u$–$z$ ($u_0' = 0.26$) | 8.9e-5 (121) | $M^{-2.9}$ | 1.9e-2 | $M^{-1.0}$ | 2.3 / 1.9 |

(Residuals are per test mode, divided by the largest term of that mode's equation, maximum over the test modes;
the $I$-equation maxima come from weakly excited modes.)

Tilted-wall term, separately (E-plane wedge; curved frame):

| Check | E-plane wedge | curved frame |
| --- | --- | --- |
| $\max|hE_z + wE_n|/\max|E|$ on the moving walls | 2e-16 | 3e-14 |
| Leibniz decomposition of $dV_p/dz$ (interior + $\partial_ze_p$ + moving domain) | 1.1e-11 | 4.3e-12 |
| tilted-wall term $\oint hE_ze_{p,n}$ = $-\oint wE_t\cdot e_p$ (cancellation) | 5.3e-16 | 7.6e-14 |
| size of the moving-domain term relative to $dV/dz$ | 0.088 | 0.13 |
| modal form $\oint hE_ze_{p,n} = -\sum_qW_{pq}V_q$, truncation 9…129 / 16…121 | 6.2e-2 … 4.2e-3 ($M^{-1}$) | 0.26 … 3.2e-2 ($M^{-1}$) |

### 12.3 `check_solver.py` — cutoff-crossing method and special cases

| Check | Result | Target |
| --- | --- | --- |
| Airy: single mode through cutoff, GSM vs exact (port 1 at $\beta = -66i$ m⁻¹), 40/80/160 slices | 4.2e-7, 2.6e-8, 1.6e-9 (4th order) | 1e-8 |
| Bessel: linear H-plane taper single mode, GSM vs $\sqrt a\,Z_\nu$ | 2.6e-9 | 1e-8 |
| H-plane wedge (exact cylindrical waves, TE$_{30}$ crosses cutoff), $\max|b - Sa|/\max|a|$, $M$ = 6, 12, 24 | 9.9e-2, 4.7e-2, 2.3e-2 ($M^{-1.02}$) | $O(1/M)$ |
| reciprocity / unitarity ($\kappa, a', u_0', b'$ together, families 0 and 2, evanescent modes) | 3.9e-15 / 8.8e-15 | 1e-10 |
| total reflection of TE$_{30}$ at its cutoff plane: $1 - |S_{22}|$; phase vs $i\,e^{-2i\int\beta}$ | 2.1e-5; 0.013 rad | 1e-3; 0.05 |
| smooth taper with cutoff crossing ($|S_{21}(\text{TE}_3\leftarrow\text{TE}_1)| = 3.04\times10^{-2}$): change of $S_{21}(\text{TE}_3\leftarrow\text{TE}_1)$ relative to $f_c = 4$ (18 modes), for $f_c$ = 1, 1.25, 2, 3 (4, 5, 9, 13 modes) | 6.7e-4, 1.1e-4, 6.0e-5, 2.4e-5 | converging |
| same, slices 15/30/60 | change 6e-7 | 1e-4 relative |
| first order vs full: TE$_1\to$TE$_3$ conversion; TE$_1$ reflection | 0.26 %; 1.7 % | 5 % |
| $K^HG + GK = 0$ (20 propagating modes, all terms); $M^TJ + JM = 0$, $M^HG' + G'M = 0$ (with an evanescent mode) | 1.4e-18; 0; 0 | 1e-12 |

## 13. Validity limits and open issues

* Exact (infinite system) for perfectly conducting walls, rectangular local cross-sections, planar curvature in
  the $u$–$z$ plane, $h>0$, piecewise $C^1$ wall positions. Impedance walls (T1) add boundary terms not included
  here; the cancellation of 2.2 uses $E_s = 0$ on the wall.
* Truncation: $O(1/M)$ convergence with a slope-proportional constant (8.4). The WP4 staircase reference has its
  own slow convergence (edge singularities at each step); the comparison "as the step size → 0" must be made at
  converged mode counts on both sides, or with extrapolation.
* The power-wave form must not be used at or near cutoff. A port plane must not coincide with a cutoff plane.
* Twists (T3) and non-rectangular cross-sections are out of scope. A curved centre line with torsion is out of
  scope (the frame of T0 is orthogonal only for planar curvature).
* Not checked numerically: curvature combined with moving $v$-walls ($\kappa\ne0$ and $b'\ne0$ at once); no exact
  field was found for that geometry. The derivation treats it with no extra assumption (2.2 holds pointwise with
  the local $h$), and the combination $\kappa$ + moving $u$-walls is checked (curved frame).
