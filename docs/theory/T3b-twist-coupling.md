# T3b — Coupled-mode equations of twisted rectangular guides (independent derivation)

*Theory note, wave 1, card T3 (second, independent derivation T3b). Derived in helical coordinates from the
covariant Maxwell equations, without differentiating rotated mode fields, and without looking at T3's work.
All formulas are in the conventions of `CONVENTIONS.md`. Machine-checkable versions and checks:
`docs/theory/T3b/` (section 10). Status: derived, verified symbolically and numerically, and checked against
exact modes of a uniformly twisted guide computed by an independent spectral method.*

## 0. Results at a glance

* The local-frame (co-rotating) Maxwell equations of a twisted guide are **exactly** those of the straight guide
  plus $-\tau\,G$ acting on $E_t$ and $H_t$, with $\tau=\theta'(z)$ and the rotation generator
  $G e = \hat z\times e-(u\partial_v-v\partial_u)e$. There is no $\tau^2$ term (box 1).
* The wall of a twisted guide is a helicoid, so $E\cdot\hat z\neq 0$ on it. This gives a wall term in the
  projection. As a result the coupling matrix is **not** $-\langle e_p, G e_q\rangle$ on both fields. The exact
  telegrapher form is $V' = \dots+\tau K V$, $I'=\dots-\tau K^T I$ with
  $K_{pq}=\iint e_q\cdot\partial_\theta e_p\,ds$ (boxes 2 and 3). An independent spectral solution of the twisted
  guide confirms this coupling and rules out the "naive" one (section 5).
* The closed forms for all TE/TM pairs are in boxes 2a and 2b. Selection rule: $m+m'$ and $n+n'$ both odd, so a
  twist always changes the family $n$. A twist couples TE to TM. $K_{\rm TE,TM}=0$ but $K_{\rm TM,TE}\neq0$.
* Power waves are in box 4: $\frac{d}{dz}[a^+;a^-] = [[-i\beta+\tau A,\ \tau B],[\tau B,\ i\beta+\tau A]]\,[a^+;a^-]$,
  with $A$ real antisymmetric and $B$ real symmetric for propagating modes.
* Reviewer statements checked independently:
  * **Generator.** Confirmed. It is what the metric gives, and $G(u,v)=0$.
  * **$c$.** $c=\iint e_{01}\cdot\partial_\theta e_{10}\,ds = 8/\pi^2 = 0.8105694691\ldots$, exactly. This is a
    sympy result for symbolic $a$, $b$, so it holds for every aspect ratio.
  * **Square-guide two-mode limit.** The polarization rotates by $-c\theta$ in the local frame, i.e. by
    $(1-8/\pi^2)\theta$ in the fixed frame. This is exact for any profile $\theta(z)$, with no reflection.
    Katsenelenbaum p. 146 states the same (full conversion every $\pi^3/16$ rad).
* End frames follow the updated `CONVENTIONS.md` §2 (PR #7): the line's global frame is used at both ports and no
  FrameChange follows a twist. A +90° then −90° twist is the identity at zero length (4e-15) (box 6).
* **Doane eq. 18** is transcribed correctly in the card. With the exact $\Delta\beta$ it gives the reviewer's
  0.30 / 0.47 / 0.68 %. The first-order triangular integral gives 0.0054 / 0.0001 / 0.045 % (TE₀₁), also as the
  reviewer found. Eq. 18 is a frequency-averaged estimate: the exact first-order result oscillates between 0 and
  4× eq. 18 and averages 1.5× eq. 18. It is not a strict upper envelope.
* **New finding for Fig. 9.** The full coupled-mode solution of the 20-inch triangular twist predicts 0.11–0.60 %
  spurious power, mostly TE₁,₂, which is reached in second order. That is well above Fig. 9's triangular curve
  (≈0.05–0.08 %). The constant and cosine curves of Fig. 9 are reproduced by the TE₀,₁ (and total) conversion
  (section 8). This is to be settled by COMSOL in wave 2.

## 1. Helical coordinates

Fixed Cartesian frame $(x, y, z)$. The cross-section turns about its **centre** (the origin of the $(u, v)$ frame
of `CONVENTIONS.md` §2) by $\theta(z)$. θ is positive counter-clockwise, from $\hat u$ towards $\hat v$, i.e.
right-handed about $+\hat z$, the direction of propagation:

$$x = u\cos\theta - v\sin\theta,\qquad y = u\sin\theta + v\cos\theta,\qquad z = z .$$

With $\tau = \theta'(z)$, $r=(u,v)$ and $w=(v,-u)$ (so that $\hat z\times w = r$), the metric is

$$g_{ij}=\begin{pmatrix}1&0&-\tau v\\0&1&\tau u\\-\tau v&\tau u&1+\tau^2(u^2+v^2)\end{pmatrix},\qquad
\det g = 1,\qquad g^{ij} = L L^T,\quad L=\begin{pmatrix}I&\tau w\\0&1\end{pmatrix}.$$

Useful facts (all checked symbolically in `check_metric_sympy.py`):

* $\partial\mathbf r/\partial u=\hat u$ and $\partial\mathbf r/\partial v=\hat v$. The covariant transverse
  components $E_u, E_v$ are therefore the Cartesian components in the rotating frame. They are the quantities
  that the mode expansion and the port amplitudes use, and they are continuous across any plane
  $z=\text{const}$, including where θ′ jumps.
* $\nabla z = \hat z$. The contravariant $E^z$ is the Cartesian $E_z$. The covariant $E_z = E^z - \tau\,w\cdot E_t$.
* Covariant Maxwell equations, with $\sqrt g = 1$:
  $\operatorname{curl} E = -i\omega\mu_0\,g^{ij}H_j$ and $\operatorname{curl} H = i\omega\varepsilon_0\,g^{ij}E_j$.
  Here curl is the plain coordinate curl of covariant components. The twisted guide is a straight guide filled
  with the anisotropic medium $\varepsilon_r=\mu_r=g^{ij}$.
* PEC walls: tangential covariant components vanish. On $u=\pm a/2$, $E_v = E_z = 0$; on $v=\pm b/2$,
  $E_u = E_z = 0$. In terms of $E^z$: $E^z = \tau v E_u$ on $u=\pm a/2$ and $E^z=-\tau u E_v$ on $v=\pm b/2$.
  The physical $E_z$ does not vanish on the helicoidal wall.

Katsenelenbaum (1998, p. 140, PDF p. 157) points out that the coupling depends on the rotation axis (centre or
vertex). Everything here is for rotation about the centre.

## 2. Exact local-frame equations

Eliminating the covariant $E_z$ and $H_z$ in favour of the contravariant $E^z$ and $H^z$ makes the $\tau^2$ terms
of $g^{ij}$ cancel identically. The result is exact for any $\theta(z)$:

$$\boxed{\begin{aligned}
\partial_z E_t &= \nabla_t E^z + i\omega\mu_0\,\hat z\times H_t - \tau\,G E_t,&
i\omega\varepsilon_0E^z &= \nabla_t\cdot(H_t\times\hat z),\\
\partial_z H_t &= \nabla_t H^z - i\omega\varepsilon_0\,\hat z\times E_t - \tau\,G H_t,&
-i\omega\mu_0H^z &= \nabla_t\cdot(E_t\times\hat z),
\end{aligned}}\qquad\text{(box 1)}$$

$$G e = \hat z\times e-(u\partial_v - v\partial_u)\,e = \nabla_t(w\cdot e) + r\,\nabla_t\cdot(e\times\hat z),$$

with the walls as in section 1. $G$ generates the active rotation $e_\theta(r)=R(\theta)e(R(-\theta)r)$:
$\partial_\theta e_\theta|_0 = Ge$. The term $-\tau G$ is the apparent turning of a field that is fixed in space,
seen from the turning frame. The rotation-invariant field $e=(u,v)$ gives $Ge = 0$. This confirms the reviewer's
generator, and it comes here out of the metric rather than being assumed.

## 3. Projection on the modes: the wall term

Expand as in T0, with the local mode fields $e_q(u,v)$ of `CONVENTIONS.md` §4:
$E_t=\sum_q V_q e_q$ and $H_t = \sum_q I_q\,\hat z\times e_q$. Then $V_p=\iint e_p\cdot E_t$ and
$I_p=\iint(\hat z\times e_p)\cdot H_t$, and the $e_p$ do not depend on z in helical coordinates. Box 1 is
equivalent to $\partial_zE_t=\nabla_tE_z+i\omega\mu_0\,\hat z\times(H_t+\tau\,w\,H^z)$, with the
**covariant** $E_z=E^z-\tau\,w\cdot E_t$, which vanishes on the walls. Projecting on $e_p$:

* $\iint e_p\cdot\nabla_tE_z=-\iint d_pE_z$, with no wall term since $E_z=0$ there. This equals
  $-\iint d_pE^z+\tau\iint d_p\,(w\cdot E_t)$.
* $\iint e_p\cdot\hat z\times(\tau wH^z)=\tau\iint(r\cdot e_p)H^z$.
* $E^z=\sum_qI_qd_q/(i\omega\varepsilon_0)$ and $H^z=\sum_qV_qg_q/(i\omega\mu_0)$ hold in $L^2$. Their
  projections on the complete Dirichlet set $\{\Pi_p\}$ and Neumann set $\{\Gamma_p\}$ can be computed with
  the true fields: the wall terms vanish because $d_p=0$ on the wall, and because
  $(E_t\times\hat z)\cdot n=0$.

Only $L^2$ convergence of the series is used: no series is differentiated, and no boundary value of a series
enters. The $I$ equation follows in the same way from
$\partial_zH_t=\nabla_tH_z-i\omega\varepsilon_0\,\hat z\times(E_t+\tau wE^z)$, using
$w\cdot(\hat z\times e_q)=-r\cdot e_q$ and $(\hat z\times e_p)\cdot r=w\cdot e_p$.

Here $d_p=\nabla_t\cdot e_p$ (TM: $-\chi_p^2\Pi_p$, TE: 0) and $g_p=\nabla_t\cdot(\hat z\times e_p)$
(TE: $-\chi_p^2\Gamma_p$, TM: 0), as in T0. With the straight-guide terms of T0 ($h=1$), the result is

$$\boxed{\frac{dV_p}{dz} = -\frac{i\beta_p}{Y_p}I_p + \tau\sum_q K_{pq}V_q,\qquad
\frac{dI_p}{dz} = -i\beta_pY_pV_p - \tau\sum_q K_{qp}I_q,}\qquad\text{(box 2)}$$

$$\boxed{K_{pq} = \iint\Big[(\nabla_t\cdot e_p)\,(w\cdot e_q) + (r\cdot e_p)\,\nabla_t\cdot(\hat z\times e_q)\Big]ds,
\qquad r=(u,v),\ w=(v,-u).}$$

$K$ is real and dimensionless, and it depends only on the cross-section. The **same** matrix appears in both
equations, transposed in the $I$ equation, for any $K$. This is what makes the system reciprocal and lossless.

**Relation to the rotation generator.** Let $\mathcal G_{pq}=\iint e_p\cdot Ge_q\,ds=\iint e_p\cdot\partial_\theta e_q\,ds$.
Integrating by parts and using $e\cdot\hat t=0$ on the walls gives

$$\boxed{K_{pq}=\mathcal G_{qp}=\iint e_q\cdot\partial_\theta e_p\,ds,\qquad
\mathcal G_{pq}+\mathcal G_{qp} = -\oint (e_p\cdot e_q)(u\,n_v - v\,n_u)\,dl.}\qquad\text{(box 3)}$$

$\mathcal G$ is not antisymmetric: its symmetric part is a wall integral, which is non-zero on the flat walls of a
rectangle. Two consequences follow.

1. The $H$ equation has the form one would guess from "a fixed field seen from a turning frame",
   $-\tau\mathcal G I$. The $E$ equation does **not**: it has $+\tau\mathcal G^TV$, not $-\tau\mathcal GV$. The
   difference is the wall term $\tau\oint E^z(e_p\cdot n)\,dl$ with $E^z=\tau\,w\cdot E_t$ on the helicoidal
   wall. A term-by-term projection of $\nabla_tE^z$ misses it, because every term of the series of $E^z$ vanishes
   on the wall.
2. The "naive" system $V'=\dots-\tau\mathcal GV$, $I'=\dots-\tau\mathcal GI$ is not even power-conserving: a lossless
   telegrapher system needs the $I$ matrix to be minus the transpose of the $V$ matrix. Its S matrix has a
   unitarity defect of order 1 for the Doane twist (`doane_fig9_extras.py`).

For the TE₁₀–TE₀₁ pair the wall term vanishes. $K$ is antisymmetric there, and
$\mathcal G_{01,10}=K_{10,01}=c=8/\pi^2$. This is the reviewer's $c$ (box 5).

### 3a. Closed forms

Write $L=u\partial_v-v\partial_u$ (rotation) and $D=u\partial_u+v\partial_v$ (dilation), with Π and Γ the
normalized potentials of `CONVENTIONS.md` §4 and $p=(m,n)$, $q=(m',n')$. Then

$$\boxed{\begin{aligned}
&\text{TE }p,\ \text{TE }q: & K_{pq}&=-\chi_q^2\iint\Gamma_q\,L\Gamma_p\\
&\text{TM }p,\ \text{TM }q: & K_{pq}&=\chi_p^2\iint\Pi_p\,L\Pi_q\\
&\text{TM }p,\ \text{TE }q: & K_{pq}&=(\chi_q^2-\chi_p^2)\iint\Pi_p\,D\Gamma_q+2\chi_q^2\iint\Pi_p\Gamma_q\\
&\text{TE }p,\ \text{TM }q: & K_{pq}&=0
\end{aligned}}\qquad\text{(box 2a)}$$

The integrals separate into the 1D integrals below, where $c_m(x)=\cos(m\pi x'/a)$, $s_m(x)=\sin(m\pi x'/a)$,
$x'=x+a/2$, over $[-a/2,a/2]$. Each is non-zero only for $m+m'$ odd:

$$Z(m,m';a)=\int s_mc_{m'}=\frac{a}{\pi}\frac{2m}{m^2-m'^2},\quad
X_c(m,m';a)=\int x\,c_mc_{m'}=-\frac{2a^2}{\pi^2}\frac{m^2+m'^2}{(m^2-m'^2)^2},\quad
X_s(m,m';a)=\int x\,s_ms_{m'}=-\frac{4a^2}{\pi^2}\frac{mm'}{(m^2-m'^2)^2}.$$

`twist_coupling.K_element` writes box 2a in these integrals. Carrying out the algebra (sympy) gives the fully
explicit form, with $P_p=b^2m^2+a^2n^2$, $\epsilon_0=1$, $\epsilon_{l>0}=2$, $\Delta_m = m^2-m'^2$ and
$\Delta_n=n^2-n'^2$, for $m+m'$ and $n+n'$ both odd (zero otherwise):

$$\boxed{\begin{aligned}
K^{\rm TE,TE}_{pq}&=-\frac{4\sqrt{\epsilon_m\epsilon_n\epsilon_{m'}\epsilon_{n'}}}{\pi^2ab}\sqrt{\frac{P_q}{P_p}}\;
\frac{a^2n^2(m^2+m'^2)\Delta_n-b^2m^2(n^2+n'^2)\Delta_m}{\Delta_m^2\Delta_n^2}\\
K^{\rm TM,TM}_{pq}&=-\frac{32\,mnm'n'}{\pi^2ab}\sqrt{\frac{P_p}{P_q}}\;\frac{a^2\Delta_n-b^2\Delta_m}{\Delta_m^2\Delta_n^2}\\
K^{\rm TM,TE}_{pq}&=-\frac{16\,mn\sqrt{\epsilon_{m'}\epsilon_{n'}}}{\pi^2\sqrt{P_pP_q}}
\left[\frac{a^2m'^2}{\Delta_m^2}+\frac{b^2n'^2}{\Delta_n^2}\right],\qquad K^{\rm TE,TM}_{pq}=0
\end{aligned}}\qquad\text{(box 2b)}$$

Examples (exact sympy integrals with symbolic $a$, $b$, `check_closed_forms.py`):

| $p$, $q$ | $K_{pq}$ |
| --- | --- |
| TE₁₀, TE₀₁ | $8/\pi^2$ |
| TE₀₁, TE₁₀ | $-8/\pi^2$ |
| TE₁₀, TE₂₁ | $-8\sqrt{2a^2+8b^2}/(3\pi^2a)$ |
| TM₂₁, TE₁₀ | $-32\sqrt2\,a^2/(9\pi^2b\sqrt{a^2+4b^2})$ |
| TE₁₀, TM₂₁ | 0 |
| TM₁₂, TM₂₁ | $-128(4a^4+5a^2b^2+b^4)/(27\pi^2ab\sqrt{4a^4+17a^2b^2+4b^4})$ |
| TM₁₁, TE₂₂ | $-64/(9\pi^2)$ |

**Selection rules.** $K_{pq}\neq0$ only if $m+m'$ **and** $n+n'$ are both odd. Every coupling therefore changes
the family $n$ (a twist mixes families, unlike every other component of wave 1) and keeps the parity of $m+n$.
The rule follows from the symmetries: the twisted guide is invariant under rotation by π about $z$, which
gives $(-1)^{m+n}$, and is odd under each mirror $u\to-u$ or $v\to-v$, since a mirror reverses τ. TE₁₀ couples to
TE₀₁, TE₂₁, TM₂₁, TE₀₃, … and not to TE$_{n0}$ or TE₂₀. This agrees with Katsenelenbaum (1998, p. 141, PDF
p. 158): "twists … cause coupling between modes separated by an odd integer in both indices", and the list TE₂₁,
TM₂₁, TE₀₃.

**Symmetries** (checked to ≤1e-15). $K^{\rm TM,TM}_{pq}/\chi_p^2$ is antisymmetric and $K^{\rm TE,TM}=0$.
$K^{\rm TE,TE}$ is **not** antisymmetric in general: $|K+K^T|$ reaches 0.21–0.33 of $\max|K|$, e.g.
$K_{\rm TE01,TE12}=0.854$ and $K_{\rm TE12,TE01}=-0.399$ in a square guide. Implementations must not
antisymmetrize $K$.

## 4. Power waves

With $V_p=(a^+_p+a^-_p)\sqrt{2/Y_p}$ and $I_p=(a^+_p-a^-_p)\sqrt{2Y_p}$ (`CONVENTIONS.md` §5), box 2 becomes

$$\boxed{\frac{d}{dz}\begin{pmatrix}a^+\\a^-\end{pmatrix}=
\begin{pmatrix}-i\beta+\tau A&\tau B\\ \tau B&i\beta+\tau A\end{pmatrix}\begin{pmatrix}a^+\\a^-\end{pmatrix},\quad
A_{pq}=\tfrac12\Big(K_{pq}\rho_{pq}-\frac{K_{qp}}{\rho_{pq}}\Big),\
B_{pq}=\tfrac12\Big(K_{pq}\rho_{pq}+\frac{K_{qp}}{\rho_{pq}}\Big),\
\rho_{pq}=\frac{\sqrt{Y_p}}{\sqrt{Y_q}}}\qquad\text{(box 4)}$$

The roots are principal roots, as in T0, so the same formulas hold for evanescent modes. τ = θ′(z) is in rad/m,
and β = diag(β_p).

* $A$ is the forward–forward and backward–backward block, $B$ the forward–backward block. The structure
  $[[A,B],[B,A]]$ differs from the bend's $[[-iC^+,-iC^-],[iC^-,iC^+]]$ of T0, because the twist coefficients
  are real.
* For propagating modes, $A$ is real antisymmetric and $B$ real symmetric, so the system is lossless. Checked:
  every S matrix in the scripts is unitary to 1e-13 and symmetric to 1e-13.
* $B_{pp}=0$, so there is no self-reflection, as in Katsenelenbaum p. 141. $B_{pq}\to0$ for degenerate modes.
* TE₁₀–TE₀₁ ($K_{01,10}=-c$, $K_{10,01}=c$):
  $A_{01,10}=-c\,(\beta_{01}+\beta_{10})/(2\sqrt{\beta_{01}\beta_{10}})$ and
  $B_{01,10}=-c\,(\beta_{01}-\beta_{10})/(2\sqrt{\beta_{01}\beta_{10}})$. The backward coupling is proportional to
  the difference of the wavenumbers. Katsenelenbaum eq. 4.6 (p. 141) gives
  $S_{jm}=-(8/\pi^2)\,\frac{h_j+h_m}{2h_j}\,\Phi'$ in his amplitude normalization. Its product $S_{jm}S_{mj}$ has the
  magnitude of our $A_{01,10}A_{10,01}$, $c^2(\beta_{01}+\beta_{10})^2/(4\beta_{01}\beta_{10})$. A sign-by-sign
  comparison would need his eqs. 2.51 and 2.71, which I did not transcribe.

`twist_coupling.M_matrix(basis, f, tau)` is the matrix of box 4.

## 5. Independent check: exact modes of a uniformly twisted guide

For constant τ the guide is z-invariant in helical coordinates. Fields ∝ $e^{-i\gamma z}$ then solve a 2D
eigenproblem:
$\iint(\operatorname{curl}_\gamma F)^*\cdot g_{ij}\operatorname{curl}_\gamma E = k^2\iint F^*\cdot g^{ij}E$,
for all admissible $F$. `check_helical_eigen.py` solves it with a spectral Nédélec (tensor Legendre) basis that
forms an exact discrete de Rham sequence, so spurious modes sit at $k^2=0$. It involves no mode expansion and no
integration by parts on mode fields: the wall terms of section 3 are handled by the variational form. It
converges to 1e-10 in $k$ (p = 16 → 22).

Box 4 is exact in τ, so its eigenvalues $-i\gamma$ at those $k$ must converge to the spectral γ as the mode basis
grows. Strong twists are used ($\tau a = 1$) so that the $O(\tau^2)$ differences between candidate couplings are
visible. The table gives max $|\gamma_{CM}/\gamma-1|$ over the 6 lowest modes:

| case | basis χ < 3k / 6k / 12k (modes) | this note (box 2) | naive ($-\mathcal G$ on E and H) | swapped ($-K^T$ on V, $K$ on I) |
| --- | --- | --- | --- | --- |
| a = 2, b = 1, τ = 0.5, γ = 4 | 89 / 359 / 1443 | 5.4e-3 / 2.9e-3 / 1.45e-3, Richardson 1.8e-4 | 2.5e-2 / 2.6e-2 / 2.7e-2 | 4.0e-2 / 4.1e-2 / 7.9e-2 |
| a = b = 1, τ = 1, γ = 6 | 106 / 414 / 1652 | 5.0e-3 / 2.5e-3 / 1.24e-3, Richardson 8.5e-5 | 2.1e-2 / 1.9e-2 / 2.0e-2 | 3.5e-2 / 4.6e-2 / 7.1e-2 |

The coupling of this note converges like $1/\chi_{max}$, and its Richardson extrapolation goes to 1e-4. The two
alternatives stall at 2–8 %. This is the decisive evidence for the wall term of section 3. **Note for WP6:** with a
strong twist, modal coupled-mode results converge only like $1/\chi_{max}$, which is slow. For practical twists
($\tau a\ll1$) the affected terms are $O(\tau^2)$ and small.

## 6. Square-guide limit (a = b)

TE₁₀ and TE₀₁ are degenerate, so ρ = 1. In the two-mode basis (order TE₀₁, TE₁₀), $A=\begin{pmatrix}0&-c\\c&0\end{pmatrix}$
and $B=0$. Since $\beta I$ commutes with $A$, the solution for **any** profile θ(z) is

$$\boxed{\begin{pmatrix}a_{01}\\a_{10}\end{pmatrix}(L)=e^{-i\beta L}
\begin{pmatrix}\cos c\theta&-\sin c\theta\\ \sin c\theta&\cos c\theta\end{pmatrix}
\begin{pmatrix}a_{01}\\a_{10}\end{pmatrix}(0),\qquad c=\frac{8}{\pi^2},\quad\theta=\theta(L)-\theta(0),}
\qquad\text{(box 5)}$$

with no reflection. For incident TE₁₀, $a_{10}=\cos c\theta$ and $a_{01}=-\sin c\theta$. Since $e_{10}\parallel+\hat v$
and $e_{01}\parallel-\hat u$, the field at the centre points at angle $-c\theta$ from $\hat v$ in the local frame.
In the fixed frame it points at $(1-c)\theta=(1-8/\pi^2)\theta$: the field does not follow the guide. For 90°:
−72.951° local and +17.049° fixed. Power is fully converted to TE₀₁ at $c\theta=\pi/2$, i.e.
θ = π³/16 = 111.03°, as in Katsenelenbaum p. 146 ("each π³/16 radians, i.e. after each 111°", "valid … for any
arbitrary function Φ(z)").

`square_and_frames.py` checks box 5 against the sliced transfer matrix for constant, triangular and irregular
profiles (1.5e-14). It also reads the polarization from the fields of `wrmodes.modes`.

**Effect of the other modes** (all propagating modes, forward and backward, 90° twists, a = b = 30 mm, incident TE₁₀):

| 2a/λ | L/λ | profile | local angle (−cθ = −72.951°) | ellipticity | power in other modes | reflected |
| --- | --- | --- | --- | --- | --- | --- |
| 2.5 | 5 | constant | −71.138° | 1.1e-2 | 1.1e-2 | 7.5e-4 |
| 2.5 | 20 | constant | −72.849° | 4.8e-4 | 4.7e-4 | 4.3e-5 |
| 2.5 | 80 | constant | −72.945° | 9.9e-6 | 2.0e-5 | 2.9e-6 |
| 2.5 | 80 | triangular | −72.940° | 5.8e-8 | 5.8e-8 | 8.1e-11 |
| 4.5 | 20 | triangular | −67.285° | 5.3e-2 | 5.1e-2 | 2.4e-8 |
| 4.5 | 80 | constant | −72.836° | 1.4e-4 | 2.0e-4 | 1.7e-6 |

The full table is in `square_and_frames.out`. The other modes are not degenerate with the pair. They change the
result through (i) leakage of order $(\tau/\Delta\beta)^2$, which is non-adiabatic and set by the ends of the
profile, and (ii) a slower rotation. In a uniform twist the pair splits into eigenmodes with
$\gamma_\pm=\beta\pm c_{\rm eff}\tau$, and the local rotation rate is $c_{\rm eff}\tau$. Second-order paths
TE₁₀ → q → TE₀₁ do not exist: q would need $m_q$ even and odd at the same time. By the 90° symmetry the
correction is therefore of relative order $(\tau/\Delta\beta)^2$:

| 2a/λ | τa = 0.01 | τa = 0.1 | τa = 0.3 |
| --- | --- | --- | --- |
| 1.4 | 0.810571 | 0.810727 | 0.812006 |
| 2.5 | 0.810557 | 0.809300 | 0.798727 |
| 4.5 | 0.810507 | 0.804136 | 0.180 (mixing with a third mode) |

$c_{\rm eff}$ is computed with a basis of all modes with χ < 3k. The deviation from c scales as τ²: ×100 from
τa = 0.01 to 0.1. Both tables are quasi-analytic tests for WP6. The two-mode result is exact. The full-basis
numbers are reference values to 1e-4 in angle.

## 7. End frames

This section follows `CONVENTIONS.md` §2 as updated by PR #7 (coordinator's message during this work). The
global guide frame is attached to the guide and rotates with the cross-section.

$$\boxed{\text{A twist's GSM uses the same port basis at both ports: }\texttt{ModeBasis}(a, b)\text{ in the line's
global frame (its own frame tag, no twist tag), with amplitudes in the co-rotating frame of section 1. No
FrameChange follows a twist. }\theta_{tot}\text{ only sets the orientation in space of the components after it.}}
\qquad\text{(box 6)}$$

* This is exactly what box 4 produces. The local co-rotating amplitudes at $z = L$ are the amplitudes in the
  rotated guide's own frame, with $(a, b)$ along $(u, v)$ at both ports.
* **Zero-length check** (`square_and_frames.py`, part 4). At zero length the transfer matrix of a twist by θ is
  $e^{\theta X}$, with $X=[[A,B],[B,A]]$. A +90° twist followed by a −90° twist is therefore the identity for any
  truncation. Through `from_transfer_matrix` and the star product, $\max|S-I_{GSM}|$ is 2e-16 … 4e-15 for a square
  30 mm guide and for tall WR187 at 9 and 16.5 GHz. At finite length the pair is not the identity, which is
  physical: two 20-λ constant-rate twists give $|S_{21}(\text{TE}_{10},\text{TE}_{10})|^2$ = 1.000000 (square,
  9 GHz, where the two-mode result of box 5 is exact and undoes itself), 0.996 (square, 16.5 GHz), 0.980 and 0.686
  (tall WR187, 9 and 16.5 GHz).
* **Caution on a single zero-length twist.** In a square guide, a single 90° twist of zero length should equal the
  overlap matrix of the field in a frame turned by 90°, i.e. `eplane_frame_signs`: the fields cannot follow, and
  the walls return onto themselves. In box 2 at zero length, V and I decouple: $V\to e^{\theta K}V$,
  $I\to e^{-\theta K^T}I$, independent of frequency. These products do **not** converge to that matrix as the
  basis grows. TE₁₀ → TE₀₁ comes out −1.005, −0.971, −0.966, −1.020, −0.967 for 18 … 2290 modes, and the max error
  over the m, n ≤ 2 block stays at 0.2–0.3. The abrupt limit sits outside the range where the local-mode series
  converges. WP6 should use the +90°/−90° identity and finite-length references (boxes 5 and 7, section 6), not
  this limit.

## 8. Rate profiles, Doane eq. 17 and eq. 18, Fig. 9

Profiles, total angle Θ over length L:

$$\boxed{\begin{aligned}
&\text{constant:} & \theta'&=\Theta/L, & |F(D)|&=\Theta\,|\operatorname{sinc}(DL/2)|\\
&\text{cosine:} & \theta&=\tfrac\Theta2\big(1-\cos\tfrac{\pi z}{L}\big), & |F(D)|&=\Theta\,\pi^2|\cos(DL/2)|/|\pi^2-(DL)^2|\\
&\text{triangular (Doane eq. 17):} & \theta'&=\tfrac{2\Theta}{L}\tfrac{2z}{L}\ (z<\tfrac L2),\ \tfrac{2\Theta}{L}\tfrac{2(L-z)}{L}\ (z>\tfrac L2), & |F(D)|&=\Theta\Big[\tfrac{\sin(DL/4)}{DL/4}\Big]^2
\end{aligned}}$$

$$\boxed{|a^+_q(L)|=|A_{qp}|\,|F(\beta_q-\beta_p)|,\qquad |a^-_q(0)|=|B_{qp}|\,|F(-(\beta_p+\beta_q))|,\qquad
F(D)=\int_0^L\theta'(z)e^{iDz}dz}\qquad\text{(box 7)}$$

These are the first-order (Born) amplitudes from incident forward mode p, with $\operatorname{sinc}x=\sin x/x$.
The closed forms of |F| agree with quadrature to 3e-13 (`rate_profiles.py`). "Cosine" is my reading of Fig. 9's
label (the angle varies as a half cosine). Katsenelenbaum p. 148 lists "cosine" and "cosine-squared" separately.

**Eq. 17** (Doane & Anderson 1989, p. 159, PDF p. 7):
$d\theta/dz=(2\theta_{tot}/L_t)(2z/L_t)$ for $0<z<L_t/2$ and $(2\theta_{tot}/L_t)(2(L_t-z)/L_t)$ for
$L_t/2<z<L_t$. This is the triangular profile above. Its integral is $\theta_{tot}$, and its maximum is
twice the constant rate (also Katsenelenbaum eq. 4.29).

**Eq. 18**, checked against the scan: "$P_c\simeq(8c\theta)^2/(\Delta\beta L_t)^4$", with
"$|\Delta\beta L_t|\simeq\frac{L_t\lambda}{a^2}\big|\frac{a^2}{b^2}-1\big|\frac\pi4$" and "c, is approximately
$8/\pi^2$". The card's transcription is correct. It derives from Unger (1957) eq. 13 (p. 1298),
$|w_2(z_1)|=|c_0(1-e^{j\Delta\beta z_1})/(\Delta\beta^2z_1)|$, for one linear taper from 0 to $c_0$. The exact
first-order triangular result (box 7, TE₀₁, $|A|\approx c$) is
$P=256\,c^2\Theta^2\sin^4(\Delta\beta L/4)/(\Delta\beta L)^4$. It oscillates between 0 and 4× eq. 18 and averages
$96c^2\Theta^2/(\Delta\beta L)^4 = 1.5\times$ eq. 18. Eq. 18 is therefore a frequency-averaged estimate, not a
strict envelope.

20-inch, 90° WR187 twist, "tall" orientation of Fig. 9 ($a$ = 0.872 in along $u$, $b$ = 1.872 in, TE₁₀ with E
along the long side), incident TE₁₀. Powers in % (`rate_profiles.out`):

| | 15 GHz | 16.5 GHz | 18 GHz |
| --- | --- | --- | --- |
| eq. 18, approximate Δβ | 0.395 | 0.579 | 0.820 |
| eq. 18, exact Δβ (reviewer: 0.30 / 0.47 / 0.68) | 0.302 | 0.465 | 0.684 |
| first order, triangular, TE₀₁ (reviewer: 0.0054 / 0.0001 / 0.045) | 0.0054 | 0.0001 | 0.0452 |
| first order, triangular, all modes | 0.0480 | 0.0284 (TE₀₃ 0.027) | 0.0662 |
| first order, constant, TE₀₁ / all | 0.874 / 0.940 | 0.128 / 0.980 | 2.361 / 2.715 |
| first order, cosine, TE₀₁ / all | 0.386 / 0.403 | 0.789 / 0.791 | 0.679 / 0.698 |

The acceptance test is met. The first-order triangular conversion is 0.0001–0.066 %, below 0.1 % and of the same
order as Fig. 9's triangular curve. The backward first-order conversion is ≤ 8e-3 % (constant) and ≤ 2e-6 %
(triangular).

**Full coupled-mode solution** (box 4, all propagating modes, forward and backward, 150 midpoint slices). Powers in
%, TE₀₁ / all spurious; Fig. 9 values are my rough reading of the scan (±0.05 %), not digitized data (L1 does
that):

| f (GHz) | constant | Fig. 9 | cosine | Fig. 9 | triangular | Fig. 9 |
| --- | --- | --- | --- | --- | --- | --- |
| 15.0 | 1.086 / 1.380 | ≈1.35 | 0.345 / 0.351 | ≈0.33 | 0.001 / 0.114 | ≈0.05 |
| 15.5 | 0.438 / 0.589 | ≈0.55 | 0.534 / 0.549 | ≈0.55 | 0.001 / 0.139 | ≈0.06 |
| 16.0 | 0.061 / 0.275 | ≈0.15 | 0.711 / 0.730 | ≈0.7 | 0.007 / 0.193 | ≈0.07 |
| 16.5 | 0.032 / 0.693 | ≈0.03 | 0.847 / 0.866 | ≈0.85 | 0.011 / 0.247 | ≈0.07 |
| 17.0 | 0.355 / 0.553 | ≈0.3 | 0.928 / 0.949 | ≈0.95 | 0.009 / 0.338 | ≈0.08 |
| 17.5 | 0.978 / 1.713 | ≈1.0 | 0.947 / 0.968 | ≈1.03 | 0.004 / 0.450 | ≈0.08 |
| 18.0 | 1.820 / 2.137 | ≈1.7 | 0.905 / 0.942 | ≈1.02 | 0.000 / 0.602 | ≈0.08 |

* The cosine curve is reproduced to within the reading accuracy. The constant curve follows our TE₀₁ conversion,
  including the deep minimum near 16.5 GHz. At 16.5 GHz our total includes 0.61 % of TE₂₁, which Fig. 9 does not
  show.
* The triangular twist's spurious power is dominated by **TE₁,₂** (0.08 → 0.38 %). TE₁₀ reaches it only at second
  order (TE₁₀ → TE₀₁/TE₀₃/TE₂₁/TM₂₁ → TE₁₂). This is 2–8× Fig. 9's triangular curve. Doane's text (p. 159) does
  name TE₀₃, TE₁₂ and TM₁₂ as important in tall guide.
* The result does not depend on evanescent modes. With cutoff_factor 1 → 3 (19 → 182 modes, GSM cascade of
  slices), TE₁₂ at 16.5 GHz goes 0.1685 → 0.1761 % and the total 0.2471 → 0.2614 % (`doane_fig9_extras.out`).
  In the ordinary orientation the triangular twist gives 0.017 / 0.012 / 0.001 %.
* **Open question for WP7/COMSOL:** is the second-order TE₁₂ conversion of the triangular twist real (Fig. 9 then
  counted only some modes), or does Fig. 9 include it and the coupled-mode model overestimates it? A COMSOL model
  of the 20-inch triangular twist at 16.5 GHz would settle it. The helical-coordinate check (section 5) makes an
  error in K unlikely.

## 9. Notes for WP6 (implementation)

* Port bases: the line's global frame at both ports, no FrameChange (box 6).
* Use `K_explicit` or `K_matrix_fast` (box 2b) and box 4. Do not antisymmetrize K, and do not use
  $-\langle e_p, Ge_q\rangle$ on both fields.
* A twist couples all families, so with cutoff_factor > 1 it needs the evanescent modes of every family. The
  line-level truncation rule of `CONVENTIONS.md` §3 (as updated in PR #7) keeps live H- and E-frame families only.
  The checks here used `evanescent_families='all'`. For Doane's twist, evanescent modes change the
  result by ≈6 % relative (section 8).
* The coefficients are frequency-independent in the V–I form. Only ρ, β and Y depend on f. Slicing with
  midpoint θ′ and `expm` converges fast: 150 → 600 slices changes TE₁₂ from 0.1685 to 0.1687 % at 16.5 GHz.
* Reference values for tests: box 5 (exact), the full-basis tables of section 6, the first-order integrals of
  box 7, and the reviewer's numbers, which are reproduced.

## 10. Scripts (`docs/theory/T3b/`, run with `PYTHONPATH=$PWD/src:$PWD/docs/theory/T3b`)

| script | checks | output |
| --- | --- | --- |
| `twist_coupling.py` | boxes 2, 2a, 2b, 4 (`K_element`, `K_explicit`, `K_matrix_fast`, `A_B_matrices`, `M_matrix`) | — |
| `check_metric_sympy.py` | section 1, box 1 (exact, symbolic), wall condition | `check_metric_sympy.out` |
| `check_closed_forms.py` | box 2 vs quadrature (8000 pairs, 2.1e-14), $K=\mathcal G^T$ with complex-step $\partial_\theta$ (1.2e-14), box 3 wall identity (2.3e-15), generator formula (4.9e-16), $G(u,v)=0$, selection rules, box 2b (6.9e-16), sympy exact forms, c = 8/π² | `check_closed_forms.out` |
| `check_helical_eigen.py` | section 5 | `check_helical_eigen.out` |
| `square_and_frames.py` | box 5, section 6 tables, zero-length limit, box 6 (+90°/−90° identity) | `square_and_frames.out` |
| `rate_profiles.py` | box 7, eq. 18, first order and full CM for Fig. 9 | `rate_profiles.out` |
| `doane_fig9_extras.py` | evanescent modes, ordinary orientation, coupling variants | `doane_fig9_extras.out` |

## References

* Doane, J. L., and Anderson, A., "Oversized rectangular waveguides with mode-free bends and twists for broadband
  applications", Microwave J. 32(3), 153–160 (1989), p. 159: eqs. 17, 18, Fig. 9.
* Katsenelenbaum, B. Z., et al., *Theory of Nonuniform Waveguides: the Cross-section Method* (IEE, 1998),
  pp. 139–148 (PDF pp. 156–165): eqs. 4.4–4.6, 4.16–4.24, 4.29.
* Unger, H.-G., "Normal mode bends for circular electric waves", BSTJ 36, 1292–1307 (1957), eq. 13, p. 1298.
* `docs/theory/T0-field-convention.md` (expansion and straight-guide terms).
