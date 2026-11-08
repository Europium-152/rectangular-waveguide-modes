# T2b — Coupled-mode equations of tapers with curvature, regular through cutoff (independent derivation)

*Theory note, wave 1, card T2 (second, independent derivation T2b, for comparison with T2). Status: derived;
closed forms checked symbolically and numerically; full equations checked against exact Maxwell solutions;
solution method checked through cutoff crossings. Check scripts: `docs/theory/T2b/`.*

**Route.** The card describes the cross-section method (projection on the local modes, as in Katsenelenbaum
et al. 1998). This note uses a different route: a coordinate transformation that maps the tapered, curved guide onto a uniform one, so that
the walls become coordinate surfaces and the taper appears as an anisotropic "medium". No tilted-wall boundary
term and no moving integration domain occur on this route. The results are then expressed in the conventions of
`CONVENTIONS.md` and in the T0 voltage–current form, so that they can be compared with T2 coefficient by
coefficient. Every boxed formula has a Python version in `docs/theory/T2b/t2b_formulas.py` (function named in
the box caption).

## 1. Geometry and notation

* $(u, v, z)$ is the frame of a planar centre line (CONVENTIONS.md §2): $z$ is the arc length along the
  **reference line** $u = v = 0$, the $z$ scale factor is $h = 1 + \kappa(z)u$, $\kappa > 0$ turns towards $-u$.
* At each $z$ the guide is the rectangle
  $R(z) = [u_0 - a/2,\, u_0 + a/2]\times[v_0 - b/2,\, v_0 + b/2]$, with $a(z), b(z), u_0(z), v_0(z)$ smooth.
  Primes are $d/dz$. For a guide whose axis is the reference line, $u_0 = v_0 = 0$ (the strangled bend).
* Local modes: $e_p(u, v; z) = e_p^{\rm CONV}(u - u_0, v - v_0; a, b)$, the normalized fields of
  `wrmodes.modes.transverse_field` with the local $a, b$; $d_p = \nabla_t\cdot e_p$, $g_p = \nabla_t\cdot(\hat z\times e_p)$
  (T0 notation); $\chi_p$, $\beta_p$, $Y_p$ local (CONVENTIONS.md §3, §5). For mode $p = (m, n)$ write

```latex
N_p = \sqrt{m^2b^2 + n^2a^2}\quad(\chi_p = \pi N_p/ab),\qquad \epsilon_0 = 1,\ \epsilon_{l>0} = 2 .
```

* **Wall slopes.** The outward normal velocities of the four walls are constant along each wall:
  $W_{u\pm} = a'/2 \pm u_0'$ (walls $u = u_0 \pm a/2$) and $W_{v\pm} = b'/2 \pm v_0'$. The closed forms use

```latex
\Omega_u(m, m') = W_{u-} + (-1)^{m+m'}W_{u+} =
  \begin{cases} a' & m + m' \text{ even}\\ -2u_0' & m + m' \text{ odd}\end{cases},\qquad
\Omega_v(n, n') = W_{v-} + (-1)^{n+n'}W_{v+} =
  \begin{cases} b' & n + n' \text{ even}\\ -2v_0' & n + n' \text{ odd}\end{cases}.
```

* Amplitudes: $E_t = \sum_q V_q e_q$, $H_t = \sum_q I_q\,\hat z\times e_q$ on the plane $z = $ const, with
  $V = \sqrt2(a^+ + a^-)/\sqrt Y$, $I = \sqrt2\sqrt Y(a^+ - a^-)$ (T0, CONVENTIONS.md §5).

## 2. The coordinate transformation

Map the guide onto the unit square: $u = u_0(\zeta) + a(\zeta)\,\xi$, $v = v_0(\zeta) + b(\zeta)\,\eta$, $z = \zeta$,
$\xi, \eta\in[-\tfrac12, \tfrac12]$. With $w_u = u_0' + a'\xi$, $w_v = v_0' + b'\eta$ (the transverse velocity of the
point of fixed $(\xi, \eta)$),

```latex
\frac{\partial\mathbf r}{\partial\xi} = a\,\hat u,\qquad \frac{\partial\mathbf r}{\partial\eta} = b\,\hat v,\qquad
\frac{\partial\mathbf r}{\partial\zeta} = w_u\hat u + w_v\hat v + h\,\hat z,\qquad \sqrt g = a\,b\,h .
```

Maxwell's equations keep their Cartesian form in $(\xi, \eta, \zeta)$ for the **covariant** components
$E_i = \mathbf E\cdot\partial_i\mathbf r$, $H_i$ and the **flux densities** through the coordinate faces
$B^i$, $D^i$ (e.g. $B^\xi = \mathbf B\cdot(\partial_\eta\mathbf r\times\partial_\zeta\mathbf r)$):
$\epsilon^{ijk}\partial_jE_k = -i\omega B^i$, $\epsilon^{ijk}\partial_jH_k = i\omega D^i$. Explicitly

```latex
E_\xi = aE_u,\quad E_\eta = bE_v,\quad E_\zeta = hE_z + w_uE_u + w_vE_v,\qquad
B^\xi = \mu_0 b\,(hH_u - w_uH_z),\quad B^\eta = \mu_0 a\,(hH_v - w_vH_z),\quad B^\zeta = \mu_0 ab\,H_z ,
```

and the same with $E\leftrightarrow H$, $\mu_0\to\varepsilon_0$ for $H_i$, $D^i$. Three facts make this route short:

1. **The walls are coordinate surfaces.** The tangential electric field on $\xi = \pm\tfrac12$ is
   $(E_\eta, E_\zeta)$, on $\eta = \pm\tfrac12$ it is $(E_\xi, E_\zeta)$. The PEC conditions are simply
   $E_\zeta = 0$ on all walls, $E_\eta = 0$ on $\xi = \pm\frac12$, $E_\xi = 0$ on $\eta = \pm\frac12$ — the tilted-wall
   condition $hE_z = -w_nE_n$ of the cross-section method is contained in $E_\zeta = 0$.
2. **Same amplitudes.** On a plane $\zeta = $ const, $(E_\xi, E_\eta) = (aE_u, bE_v)$ is the physical transverse
   field, rescaled. With $\varepsilon_q = (a e_{q,u}, b e_{q,v})$ and $\varepsilon^*_q = (b e_{q,u}, a e_{q,v})$, the
   expansion $(E_\xi, E_\eta) = \sum_qV_q\varepsilon_q$ has exactly the local-mode amplitudes $V_q$ of §1 (and
   $(H_\xi, H_\eta) = \sum_q I_q\,\hat z\times\varepsilon^*_q$ the amplitudes $I_q$). In $(\xi, \eta)$ the
   components of $\varepsilon_q$, $\varepsilon^*_q$ are fixed trigonometric products times scalar factors that
   depend on $a(\zeta)$, $b(\zeta)$ only (not on $u_0$, $v_0$, $\kappa$).
3. **Metric-free projection.** $\iint(\mathbf A\times\mathbf B)\cdot\hat z\,d\xi\,d\eta$ of covariant fields is the
   reaction (power) integral, and $\iint\varepsilon_p\cdot\varepsilon^*_q\,d\xi\,d\eta = \delta_{pq}$, so
   $V_p = \iint E_T\cdot\varepsilon_p^*\,d\xi d\eta$ and $I_p = \iint \varepsilon_p\cdot(H_T\times\hat z)\,d\xi d\eta$.

**Derivation.** The transverse Faraday and Ampère equations read
$\partial_\zeta E_T = \nabla_TE_\zeta + i\omega\,\hat z\times B_T$ and
$\partial_\zeta H_T = \nabla_TH_\zeta - i\omega\,\hat z\times D_T$ (same form as T0 with $hE_z\to E_\zeta$, $hH_t\to B_T$).
Project on $\varepsilon^*_p$ and integrate the gradient by parts on the fixed square: the boundary term is
$\oint E_\zeta\,\varepsilon^*_p\cdot n = 0$ because $E_\zeta = 0$ on every wall (in the $H$ equation the boundary
term contains the tangential component of $\varepsilon_p$, which vanishes). Then insert $E_\zeta$, $B_T$ from the
relations above, use $hE_z = h\sum_q I_qd_q/(i\omega\varepsilon_0)$ and $H_z = \sum_q V_qg_q/(i\omega\mu_0)$ (legitimate
term by term because $d_p = 0$ on the walls and $E_t$ has no tangential component there, see T0), and add the
derivative of the basis, $\iint E_T\cdot\partial_\zeta\varepsilon^*_p$. The terms with $h$ are exactly those of T0;
the terms with $w$ and $\partial_\zeta$ give a frequency-independent matrix $T$:

```latex
\boxed{\;
\frac{dV}{dz} = -i\,Z\,I + T\,V ,\qquad \frac{dI}{dz} = -i\,Y\,V - T^{\mathsf T} I\;}
\qquad\text{(B1, `system_matrix`)}
```

```latex
\boxed{\;
Z_{pq} = \omega\mu_0 \iint_{R(z)} h\,e_p\cdot e_q - \frac{1}{\omega\varepsilon_0}\iint_{R(z)} h\,d_pd_q ,\qquad
Y_{pq} = \omega\varepsilon_0 \iint_{R(z)} h\,e_p\cdot e_q - \frac{1}{\omega\mu_0}\iint_{R(z)} h\,g_pg_q\;}
\qquad\text{(B2, `ZY`)}
```

```latex
T_{pq} = -\iint (w\cdot e_q)\,d_p\,du\,dv + \iint g_q\,(w_v e_{p,u} - w_u e_{p,v})\,du\,dv
         + \iint \varepsilon_q\cdot\partial_\zeta\varepsilon^*_p\,d\xi\,d\eta
\qquad\text{(transformation route, `T_transformed`)}
```

The current equation gives $T^I_{pq} = \iint g_p(w_ue_{q,v} - w_ve_{q,u}) + \iint d_q(w\cdot e_p) + \iint\partial_\zeta\varepsilon_p\cdot\varepsilon^*_q$,
and $T^I_{pq} + T_{qp} = \iint\partial_\zeta(\varepsilon_p\cdot\varepsilon^*_q)\,d\xi d\eta = \partial_\zeta\delta_{pq} = 0$:
**$T^I = -T^{\mathsf T}$ exactly**, which is what makes B1 reciprocal and lossless (§6).

### 2.1 What $T$ is in the cross-section language, and the integration domain

The three integrals above add up to (checked to 4e-15 for 31 modes and three geometries with all four slopes
non-zero, `check_closed_forms.py`)

```latex
\boxed{\; T_{pq} = \iint_{R(z)} e_q\cdot\frac{\partial e_p}{\partial z}\Big|_{u,v}\,du\,dv
 = \frac{\partial}{\partial z'}\iint_{R(z)} e_q(u, v; z)\cdot e_p(u, v; z')\,du\,dv\,\Big|_{z' = z}\;}
\qquad\text{(B3a, `T_overlap`)}
```

with the trigonometric formulas of $e_p(\cdot; z')$ continued analytically outside their own rectangle and the
**integration domain fixed at $R(z)$**, the rectangle of the expansion. The derivative of the overlap over the
moving domain $R(z')$ differs at first order: it equals $-T_{qp}$ (difference
$\oint_{\partial R} w_n (e_p\cdot n)(e_q\cdot n)\,dl = -(T_{pq} + T_{qp})$, from $\partial_z\iint_{R(z)}e_p\cdot e_q = 0$),
which would put the wrong matrix in the voltage equation.

**Tilted-wall term (cross-section route, derived and checked separately).** Projecting
$\partial_zE_t = \nabla_t(hE_z) + i\omega\mu_0h\,\hat z\times H_t$ on $e_p$ over the moving rectangle produces two
boundary terms on the walls: the tilted-wall term $\oint hE_z\,(e_p\cdot n) = -\oint w_nE_n(e_p\cdot n)$
(because $hE_z + w_nE_n = 0$ on a wall moving with normal velocity $w_n$, i.e. $E_\zeta = 0$) and the Leibniz term
$+\oint w_n(E_t\cdot e_p) = +\oint w_nE_n(e_p\cdot n)$ from differentiating
$V_p = \iint_{R(z)}E_t\cdot e_p$ (the tangential parts vanish). They cancel exactly, leaving B3a. In the current
equation there is no tilted-wall term ($(\hat z\times e_p)\cdot n = 0$), but a Leibniz term
$\oint w_nH_s\,(e_p\cdot n)$ survives; it is what turns $-\iint e_p\cdot\partial_ze_q$ over the moving domain
into $-T_{qp}$. Both statements are checked by the exact-field residuals of §8: the equations with $T$ in the
voltage equation and $-T^{\mathsf T}$ in the current equation are satisfied by exact Maxwell fields; the
condition $hE_z + w_nE_n = 0$ holds for those fields on the walls.

## 3. Closed forms (all TE/TM pairs)

From B3a, Green's theorem and the perturbation of the Dirichlet ($\Pi$) and Neumann ($\Gamma$) eigenproblems
(Hadamard), with $s = \hat z\times n$ along the contour:

```latex
T^{TM,TM}_{p\ne q} = -\frac{\chi_p^2}{\chi_p^2 - \chi_q^2}\oint w_n\,\partial_n\Pi_p\,\partial_n\Pi_q ,\quad
T^{TE,TE}_{p\ne q} = -\frac{\chi_q^2}{\chi_p^2 - \chi_q^2}\oint w_n\,(\chi_p^2\Gamma_p\Gamma_q - \partial_s\Gamma_p\partial_s\Gamma_q),
```

```latex
T^{TM,TE}_{pq} = -\oint w_n\,\partial_n\Pi_p\,\partial_s\Gamma_q ,\qquad T^{TE,TM}_{pq} = 0 ,\qquad
T_{pp} = -\tfrac12\oint w_n\,(e_p\cdot n)^2 .
```

(the TE–TE form uses flat walls and $\partial_s\Gamma = 0$ at the corners, true for the rectangle). On the
rectangle $w_n$ is constant on each wall and the contour integrals reduce to the wall slopes. With
$p = (m, n)$, $q = (m', n')$:

```latex
\boxed{
\begin{aligned}
T_{pp}^{TE} &= -\frac{\epsilon_m\epsilon_n}{4N_p^2}\,\big(n^2\,a\,a' + m^2\,b\,b'\big), \qquad
T_{pp}^{TM} = -\frac{1}{N_p^2}\Big(m^2\frac{b^2}{a}a' + n^2\frac{a^2}{b}b'\Big),\\[2pt]
T^{TE,TE}_{pq} &= -\frac{N_q^2}{(N_p^2 - N_q^2)\,N_pN_q}\Big[\delta_{nn'}\sqrt{\epsilon_m\epsilon_{m'}}\,\frac{m^2b^2}{a}\,\Omega_u(m,m')
                 + \delta_{mm'}\sqrt{\epsilon_n\epsilon_{n'}}\,\frac{n^2a^2}{b}\,\Omega_v(n,n')\Big]\quad(p\ne q),\\[2pt]
T^{TM,TM}_{pq} &= -\frac{2N_p^2}{(N_p^2 - N_q^2)\,N_pN_q}\Big[\delta_{nn'}\,mm'\,\frac{b^2}{a}\,\Omega_u(m,m')
                 + \delta_{mm'}\,nn'\,\frac{a^2}{b}\,\Omega_v(n,n')\Big]\quad(p\ne q),\\[2pt]
T^{TM,TE}_{pq} &= \frac{ab\,\sqrt{\epsilon_{m'}\epsilon_{n'}}\;mn}{N_pN_q}\Big[\delta_{nn'}\frac{\Omega_u(m,m')}{a}
                 - \delta_{mm'}\frac{\Omega_v(n,n')}{b}\Big]\quad(\text{including } m' = m,\ n' = n),\qquad
T^{TE,TM}_{pq} = 0 .
\end{aligned}}
\qquad\text{(B3, `T_closed`)}
```

Superscripts give (kind of $p$, kind of $q$); $T_{pq}$ multiplies $V_q$ in $dV_p/dz$. All pairs with
$m\ne m'$ and $n\ne n'$ vanish; the denominators $N_p^2 - N_q^2$ never vanish for a coupled pair ($n = n'$,
$m\ne m'$ or the converse), so B3 has no degeneracy singularity, even in a square guide. $T$ is real and
independent of frequency. Examples (sympy, exact; `check_closed_forms.py` item 2):

| $p$, $q$ | $T_{pq}$ |
| --- | --- |
| TE10, TE30 | $3a'/(4a)$ |
| TE10, TE20 | $-8u_0'/(3a)$ |
| TE11, TE11 | $-(aa' + bb')/(a^2 + b^2)$ |
| TM11, TM11 | $-(a^3b' + b^3a')/(ab(a^2 + b^2))$ |
| TM11, TE11 | $2(ba' - ab')/(a^2 + b^2)$ |
| TE11, TM11 | $0$ |
| TM21, TE11 | $-8bu_0'/(\sqrt{a^2 + b^2}\sqrt{a^2 + 4b^2})$ |
| TM22, TE21 | $8av_0'/(\sqrt{a^2 + b^2}\sqrt{a^2 + 4b^2})$ |
| TM12, TM11 | $8v_0'\sqrt{4a^2 + b^2}/(3b\sqrt{a^2 + b^2})$ |
| TE12, TE10 | $-\sqrt2\,b'/\sqrt{4a^2 + b^2}$ |

For TE$_{m0}$ in an H-plane taper ($b' = u_0' = 0$): $T_{pq} = 2mm'a'/((m'^2 - m^2)a)$ for $m + m'$ even
($p = m$, $q = m'$), antisymmetric, and $T_{pp} = 0$.

## 4. Selection rules

Read off B3 (and from parity of the walls: $\Omega_u$ is $a'$ for $m + m'$ even and $-2u_0'$ for $m + m'$ odd):

| Wall motion | couples | conserves | families (CONVENTIONS.md §3) |
| --- | --- | --- | --- |
| symmetric width taper $a'$ | $\Delta m$ even, $\Delta n = 0$ (TE↔TE, TM↔TM, TM↔TE) | $n$ and the parity of $m$ | conserved: one GSM per family |
| width offset $u_0'$ | $\Delta m$ odd, $\Delta n = 0$ | $n$ | conserved |
| symmetric height taper $b'$ | $\Delta n$ even, $\Delta m = 0$ | $m$ and the parity of $n$ | **not** conserved: couples families $n$, $n\pm2$, … |
| height offset $v_0'$ | $\Delta n$ odd, $\Delta m = 0$ | $m$ | not conserved |
| curvature κ (T0) | $\Delta m$ odd, $\Delta n = 0$ | $n$ | conserved |

* To first order in the slopes, from TE$_{10}$ a symmetric taper reaches TE$_{m0}$ ($m$ odd, from $a'$) and
  TE$_{1n}$/TM$_{1n}$ ($n$ even, from $b'$). Butterweck & de Ronde (1968, p. 96) list H$_{30}$, H$_{50}$, …,
  H$_{12}$, H$_{32}$, …, H$_{14}$, H$_{34}$, … as the modes allowed by symmetry; the members with $m\ge3$ and $n\ge2$
  are reached only at second order (an $a'$ and a $b'$ step).
* A pure scaling ($a'/a = b'/b$, no offsets) does not couple TE$_{mn}$ and TM$_{mn}$ (their $T^{TM,TE}$ is
  $\propto ba' - ab'$); a change of aspect ratio does.
* Tapers of $a$, $u_0$ and κ only (H-plane tapers, strangled H-plane bends, offsets in the bend plane) can use the
  per-family GSMs of `CONVENTIONS.md`; any $b'$ or $v_0'$ requires the full basis.

## 5. Curvature and taper together (strangled bend)

B1–B3 are exact for arbitrary $\kappa(z)$, $a(z)$, $b(z)$, $u_0(z)$, $v_0(z)$; nothing was linearized. Therefore:

* The curvature terms are those of T0 **with the local $a(z)$, $b(z)$** and with $u$ measured from the reference
  line: $\iint_{R(z)}u\,(\cdot) = u_0\,(\text{straight integral}) + (\text{T0 integral of the local, centred rectangle})$.
  For $u_0 = 0$ (guide axis = reference line) the T0 coefficients simply take the local dimensions.
* **There are no cross terms** $\kappa a'$, $\kappa u_0'$, …: in the transformation $h$ multiplies only $E_z$ and
  $H_t$, and $w$ multiplies only $E_t$, $H_z$; the taper matrix $T$ does not depend on κ and the curvature part
  does not depend on the slopes. Equivalently, on a tilted wall in a bend $hE_z = -w_nE_n$, and $h$ drops out of
  the tilted-wall term. This is exact, not a first-order statement.
* With an offset in a bend the diagonal gains $\kappa u_0\beta_p$ (power-wave form): the local propagation
  constant is that of the guide axis, $\beta_p(1 + \kappa u_0)$ per unit reference length.
* The exact-solution check of §8 includes a curved reference line ($\kappa = 4$ m$^{-1}$) together with
  $a' \approx 0.29$ and $u_0' \approx 0.29$, and families with TE and TM modes.

## 6. Power-wave form and losslessness

With $s = \operatorname{diag}(\sqrt{Y_p})$ (principal roots, CONVENTIONS.md §5) and the local $\beta_p$, $Y_p$:

```latex
\boxed{
\begin{aligned}
\frac{da^+}{dz} &= -i\beta\,a^+ - i\kappa\,(C^+a^+ + C^-a^-) + A\,a^+ + B\,a^- ,\\
\frac{da^-}{dz} &= +i\beta\,a^- + i\kappa\,(C^-a^+ + C^+a^-) + B\,a^+ + A\,a^- ,\\
P &= s\,T\,s^{-1}\ \ \big(P_{pq} = \sqrt{Y_p}\,T_{pq}/\sqrt{Y_q}\big),\qquad A = \tfrac12(P - P^{\mathsf T}),\qquad
B = \tfrac12(P + P^{\mathsf T}) + \operatorname{diag}\Big(\frac{Y_p'}{2Y_p}\Big),\\
\frac{Y_p'}{Y_p} &= \mp\frac{(\chi_p^2)'}{2\beta_p^2}\ \ (\text{TE upper, TM lower}),\qquad
(\chi_p^2)' = -2\pi^2\Big(\frac{m^2a'}{a^3} + \frac{n^2b'}{b^3}\Big),
\end{aligned}}
\qquad\text{(B4, `power_wave_matrices`)}
```

$C^\pm$ are the T0 coefficients with the local $a, b$ plus $u_0\beta_p\delta_{pq}$ in $C^+$. B4 was checked against a
numerical change of variables of B1 (including $dY/dz$) to 4e-10 (finite-difference limited).

* **Lossless.** Above cutoff $T$, $Y$ are real: $A$ is real antisymmetric and $B$ real symmetric, so
  $\frac{d}{dz}\big(|a^+|^2 - |a^-|^2\big) = 0$ from the taper terms, as from the T0 terms. In V–I form the
  statement holds for all modes: $Z$, $Y$ real symmetric and the $T$, $-T^{\mathsf T}$ pair give
  $\frac{d}{dz}\operatorname{Re}(V^{\mathsf H}I) = 0$ and $\frac{d}{dz}(V^{\mathsf T}I' - I^{\mathsf T}V') = 0$
  (power and reciprocity), with evanescent modes included.
* **Single TE$_{10}$.** $T_{11} = -b'/(2b)$ and $Y'/2Y = \tfrac12(\ln\beta)'$ give
  $B_{11} = -\tfrac12\,(\ln(b/\beta))'$ for any $a(z), b(z), u_0(z), v_0(z)$: the coupling of Katsenelenbaum et al.
  (1998, eq. 4.47) with $W = Cb/\beta$ (eqs. 4.51–4.52), offsets not contributing (they state that 4.47 holds for TE$_{10}$ in an arbitrarily varying, not
  necessarily symmetric, cross-section, ibid. p. 161). The first-order reflection $a^-(0) = -\int_0^L B_{11}e^{-2i\int_0^z\beta}dz$ gives
  $\Delta W/2W$ for a small step (their eq. 4.50; their eqs. 4.46–4.47 carry the opposite sign convention for the
  reflected amplitude). Checked to 1e-15 (item 5 of `check_closed_forms.py`).
* The power-wave coefficients contain $\sqrt{Y_p/Y_q}$ and $Y_p'/Y_p\propto1/\beta_p^2$: they diverge at the cutoff
  of $p$ or $q$. B1–B3 contain no $\beta$ and no $\sqrt Y$.

## 7. Cutoff crossings: regularity and a stable solution method

**Regularity.** $Z$ and $Y$ are polynomials in $\omega$, $1/\omega$ with geometric coefficients and $T$ is purely
geometric; B1 is analytic in $z$ wherever the walls are smooth, at and through every cutoff plane, and at
degeneracies. Only the conversion to power waves at a plane where a mode is exactly at cutoff is singular, and
that is never needed inside the taper.

**Method (B5, `t2b_solver.solve`).** Keep a fixed list of modes $(kind, m, n)$ along the whole taper (the local
fields change continuously with $z$; $T$ already accounts for the change of basis, so $V$ and $I$ are continuous
and no junction matrices are needed between slices).

```latex
\boxed{
\begin{aligned}
&\text{1. slices } [z_j, z_{j+1}] \text{ with } |\beta_p|\,\Delta z\le \gamma \ (\gamma\approx 1\text{–}2) \text{ for every mode};\quad
\Phi_j = \exp\!\Big(\tfrac{\Delta z}{2}(M_1 + M_2) + \tfrac{\sqrt3\,\Delta z^2}{12}[M_2, M_1]\Big),\\
&\qquad M_{1,2} = M\big(z_j + (\tfrac12\mp\tfrac{\sqrt3}{6})\Delta z\big),\quad M = \begin{bmatrix} T & -iZ\\ -iY & -T^{\mathsf T}\end{bmatrix}
\ \ (\text{4th-order Magnus});\\
&\text{2. pseudo-waves with a constant real reference } G>0:\ V = \sqrt2(\alpha^+ + \alpha^-)/\sqrt G,\ I = \sqrt2\sqrt G(\alpha^+ - \alpha^-);\
 \Phi_j\to S_j;\ S = S_1\star S_2\star\cdots;\\
&\text{3. ports: junction to the modal waves, } S_{11} = \tfrac{Y - G}{Y + G},\ S_{12} = S_{21} = \tfrac{2\sqrt Y\sqrt G}{Y + G},\ S_{22} = -S_{11}
\ \text{(per mode, } Y \text{ at the port)}.
\end{aligned}}
\qquad\text{(B5)}
```

With a real reference, $|\alpha^+|^2 - |\alpha^-|^2$ is the exact power of every mode, propagating or evanescent,
so every $S_j$ of a lossless taper is unitary and the Redheffer cascade cannot blow up; the evanescent growth is
confined to one short slice. The Magnus step keeps $\Phi_j$ in the group of B1 (reciprocity and losslessness are
preserved to rounding: $|S - S^{\mathsf T}|\approx10^{-14}$, unitarity of the propagating block $\approx10^{-13}$ in
the checks). The result is the GSM of CONVENTIONS.md §6 (symmetric, evanescent port modes included). The ports
must not lie exactly at a cutoff (CONVENTIONS.md §3). A Riccati equation for an admittance matrix is an
alternative, but its solution has poles wherever a section behaves like a short circuit (e.g. at a cutoff plane
for a reflected mode), so the slice/star form is preferred.

**Counting evanescent modes.** Use all modes $(kind, m, n)$ of the coupled set (one family $n$ when only $a'$, $u_0'$,
κ vary) with $\chi_p < c_f\,k$ at the **widest** cross-section, $c_f$ = `cutoff_factor` ≥ 1, i.e. every mode that
propagates anywhere plus a margin of evanescent ones. Convergence is shown by increasing $c_f$ (equivalently the
largest index $M$).

**Convergence and the quasi-static tail correction.** Plain truncation converges only as $1/M$ (§8: 3.5e-2,
1.7e-2, 8.7e-3 for $M$ = 6, 12, 24 in a 7° wedge). The cause is the current series: on a tilted wall the normal
magnetic field is $H_n = w_nH_z/h\ne0$ while every basis field $\hat z\times e_q$ has $H_n = 0$, so
$I_q = O(1/q)$; and for TE modes the impedance $Z_{qq} = \omega\mu_0$ does not grow with $q$, so the high TE modes
answer the taper quasi-statically, $I_q\simeq\sum_pT_{qp}V_p/(iZ_{qq})$, without the $1/\beta_q$ suppression
evanescent modes usually have. Eliminating the modes outside the basis in this quasi-static way gives

```latex
\boxed{\; Y \;\to\; Y_{\rm eff} = Y - T_{\rm out}^{\mathsf T}\,Z_{\rm out}^{-1}\,T_{\rm out},\qquad
(T_{\rm out})_{qp} = T_{qp},\ q\notin\text{basis},\ p\in\text{basis},\quad Z_{{\rm out},qq} = \omega\mu_0\ (\text{TE}),\ \beta_q^2/\omega\varepsilon_0\ (\text{TM})\;}
\qquad\text{(B5', `tail_correction`)}
```

(sum over the modes of the coupled set outside the basis, up to a large index, e.g. 1000; terms decay as
$1/q^2$). $Y_{\rm eff}$ stays real symmetric, so reciprocity and losslessness are kept. In the exact-wedge tests of
§8 (error of the GSM on the first five modes of each port, relative to the incident amplitudes):

| case | plain, $M$ = 6 / 12 | with B5', $M$ = 6 / 12 |
| --- | --- | --- |
| symmetric H-plane wedge, family 0 (TE30, TE50 cut off inside) | 3.5e-2 / 1.7e-2 | 6.3e-3 / 3.7e-4 |
| asymmetric H-plane wedge, family 1 (TE and TM) | 8.9e-3 / 3.7e-3 | 6.8e-4 / 3.3e-5 |
| asymmetric E-plane wedge, column $m$ = 1 (families coupled) | 7.3e-3 / 3.9e-3 | 2.0e-4 / 1.3e-5 |
| curved reference line on the asymmetric wedge ($M$ = 4 / 8) | 1.9e-2 / 5.0e-3 | 1.1e-2 / 3.2e-4 |

Plain truncation halves the error per doubling of $M$; with B5' each doubling gains a factor 15–30. WP5 should
implement B5' (or at least extrapolate in $M$) and report convergence in $M$ with and without it.

## 8. Checks

`docs/theory/T2b/` (run with `PYTHONPATH=src`):

* `check_closed_forms.py`: B3 vs the transformation integrals vs the overlap derivative (fixed domain,
  complex step and Richardson); moving domain $= -T^{\mathsf T}$; sympy; B4 vs a numerical change of variables;
  structure; Katsenelenbaum eq. 4.47; uniform limit.
* `check_exact_residual.py`: **exact Maxwell fields** of wedge guides (sums of $H^{(2)}_\nu$, $J_\nu$ cylindrical waves,
  LSM and LSE to the uniform direction, `exact_wedge.py`) projected on the local modes of planar or curved
  cross-sections; residuals of both equations of B1 with $M$ modes. Cases: symmetric H-plane wedge (family 0);
  asymmetric H-plane wedge ($a'$, $u_0'$, family 1 with TE and TM); asymmetric E-plane wedge ($b'$, $v_0'$, families
  coupled); the asymmetric wedge seen from a curved reference line (κ, $a'$, $u_0'$, family 1). The voltage residual
  falls as $M^{-3}$ (e.g. 4.1e-7, 5.1e-8, 6.4e-9 for $M$ = 32, 64, 128), the current residual as $M^{-1}$ (4.6e-5,
  2.3e-5, 1.2e-5): the equations are exact, and the slow current series is the effect described in §7.
* `check_solver_exact.py`: B5 against exact wedge fields that contain modes totally reflected at their cutoff
  plane inside the section ($J_\nu$ standing waves): the computed GSM maps the exact incident amplitudes onto the
  exact scattered ones, with the errors of the table in §7 (they go to zero with $M$); reciprocity
  $|S - S^{\mathsf T}|\le 3\times10^{-14}$ and unitarity of the propagating block $\le 5\times10^{-14}$ in every run.
* In the residual check, replacing $T$ by the moving-domain matrix $-T^{\mathsf T}$ in the voltage equation leaves
  a residual of 4e-2, 1.6e-2 and 7.9e-2 (asymmetric H-plane, E-plane, curved cases at $M = 16$) instead of
  2e-6 to 4e-7: the domain choice matters at first order. (In the symmetric H-plane family-0 case $T$ is
  antisymmetric and the two coincide.) The tilted-wall condition $hE_z + w_nE_n = 0$ holds on the walls of the exact
  fields to 4e-14.
* `check_small_slope.py`: B6 against B5 in a linear H-plane taper where TE30 crosses cutoff.

Results are in `docs/reports/T2b.md`.

## 9. Small-slope limit (single scattering)

With the incident mode $p$ (unit forward amplitude, WKB phase $\varphi_p(z_1, z_2) = \int_{z_1}^{z_2}\beta_p$) and all
other amplitudes small, B4 integrates to first order in the slopes. When $q$ propagates over the whole taper:

```latex
\boxed{\; a_q^+(L) \simeq \int_0^L A_{qp}(z)\,e^{-i\varphi_p(0,z) - i\varphi_q(z,L)}\,dz ,\qquad
a_q^-(0) \simeq -\int_0^L B_{qp}(z)\,e^{-i\varphi_p(0,z) - i\varphi_q(0,z)}\,dz \;}
\qquad\text{(B6)}
```

($q = p$ in the second gives the reflection of Katsenelenbaum's eq. 4.46 form.) For TE$_{m0}$ in an H-plane taper,
$A_{qp} = T_{qp}(\beta_p + \beta_q)/(2\sqrt{\beta_p\beta_q})$, $B_{qp} = T_{qp}(\beta_q - \beta_p)/(2\sqrt{\beta_p\beta_q})$.

**When $q$ crosses cutoff inside the taper** the power-wave coefficients diverge at its cutoff plane $z_c$ and B6 is
not usable near it. The regular form follows from the reciprocity identity of B1 for mode $q$ alone,
$\frac{d}{dz}(V_q\hat I_q - I_q\hat V_q) = s_V\hat I_q - s_I\hat V_q$ with sources $s_V = T_{qp}V_p$,
$s_I = -T_{pq}I_p$:

```latex
\boxed{\; a_q^+(L) \simeq -\frac14\int_0^L\big[T_{qp}\,V_p\,\hat I_q + T_{pq}\,I_p\,\hat V_q\big]\,dz \;}
\qquad\text{(B6', `check_small_slope.first_order_regular`)}
```

where $(V_p, I_p)$ is the incident field and $(\hat V_q, \hat I_q)$ the single-mode solution of $q$ (B1 restricted
to $q$) with unit amplitude incident from port 2 ($\hat a^-_q(L) = 1$) that decays towards port 1 ($\hat a^+_q(0) = 0$);
it is regular through $z_c$ (an Airy function there). Away from $z_c$, WKB gives $\hat a^+_q/\hat a^-_q =
i\,e^{-2i\varphi_q(z_c, z)}$ (total reflection with phase $\pi/2$), and B6' splits into the direct term of B6 plus a
**secondary wave**: generated backwards at $z$, reflected at $z_c$, then forward,
$-i\int_{z_c}^LB_{qp}e^{-i\varphi_p(0,z) - i\varphi_q(z_c,z) - i\varphi_q(z_c,L)}dz$. The direct term oscillates
with $\beta_p - \beta_q$, the secondary term with $\beta_p + \beta_q$. This is Butterweck & de Ronde's explanation of
the "bottle" taper (1968, pp. 96–97): the direct contributions near the wide end have the largest weight because
$\beta_p - \beta_q\to0$ as $a$ grows, so the wide part must have a small slope; the secondary waves interfere
destructively. The WKB split is only asymptotic: for a linear taper most of the conversion comes from the Airy
region around $z_c$, where the power-wave coefficients diverge, so B6' must be used for numbers.

Check (`check_small_slope.py`): linear H-plane taper 22.86 → 45 mm, $b$ = 10.16 mm, 15 GHz, TE30 cutoff at
$a$ = 29.98 mm inside the taper, incident TE10; full solution B5 with $M$ = 10 and B5' (change at $M$ = 14 ≤ 4e-5):

| $L$ (m) | slope $a'$ | $S_{21}$(TE30←TE10), B5 | B6' rel. error | WKB direct + secondary, rel. error | abs. $S_{11}$(TE10) B5 | B6 rel. error |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.221 | 5.004e-2 − 5.35e-3i | 5.7e-2 | 0.53 | 2.009e-3 | 0.23 |
| 0.2 | 0.111 | 2.727e-3 + 2.552e-2i | 3.1e-3 | 0.66 | 8.50e-4 | 8.1e-2 |
| 0.4 | 0.055 | −1.156e-3 − 1.294e-2i | 5.4e-4 | 0.81 | 4.39e-4 | 4.2e-2 |
| 0.8 | 0.028 | −1.064e-3 − 6.444e-3i | 3.0e-4 | 0.99 | 2.25e-4 | 2.1e-2 |

B6' converges to the full solution as the slope falls (first-order theory, phase included); the TE10 reflection
from B6 has a relative error proportional to the slope.

## 10. Validity limits and notes for WP5

* Exact for PEC walls, rectangular local cross-sections, smooth $a, b, u_0, v_0, \kappa$ (continuous first
  derivatives inside a component; a slope discontinuity at a port is a junction between the taper and the
  straight guide, which B1 handles because $V$, $I$ are continuous there and $T$ is evaluated inside).
* No twist (T3), no wall loss (T1: the impedance boundary changes the boundary terms; on this route, an impedance
  wall gives $E_\zeta = Z_s(\ldots)$ instead of 0).
* Large slopes are allowed in principle (the wedge tests use 7°–10° half-angles), but the $1/M$ convergence of the
  plain truncation grows with the slope; use B5'.
* The reference line of κ is a modelling choice; with $u_0 \neq 0$ the lengths in the GSM are along the reference
  line (CONVENTIONS.md §6 measures along the centre line, so choose the centre line as the reference line when
  possible).

## References

* B. Z. Katsenelenbaum, L. Mercader del Río, M. Pereyaslavets, M. Sorolla Ayza, M. Thumm, *Theory of Nonuniform
  Waveguides: The Cross-Section Method*, IEE 1998: §2.3 (coupling coefficients of variable cross-section guides,
  eqs. 2.95–2.99, contour forms), §4.2.2 (TE$_{10}$ reflection, eqs. 4.46–4.52, pp. 161–162).
* H. J. Butterweck, F. C. de Ronde, "Oversized waveguide for mm-waves", Philips Tech. Rev. 29 (1968), pp. 95–97
  (tapers, bottle shape, modes excited by symmetric tapers).
* T0: `docs/theory/T0-field-convention.md`.
