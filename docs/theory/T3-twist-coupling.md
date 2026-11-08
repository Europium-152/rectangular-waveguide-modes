# T3 — Coupled-mode equations of twisted rectangular guides

*Theory note, wave 1 (card `tasks/T3-twist-coupling.md`). Status: derived, closed forms checked by quadrature,
equations checked against an independent Galerkin solution of Maxwell's equations. Every boxed formula has a
Python form in `docs/theory/T3/twist_formulas.py`; the checks are `docs/theory/T3/check_*.py`.*

## Summary

* A twist is exactly a straight guide filled with the anisotropic medium $\varepsilon_0 G$, $\mu_0 G$ in
  co-rotating coordinates. The terms of second order in $\theta'$ cancel: the coupled-mode equations are
  **exactly linear in $\theta'(z)$** and contain no $\theta''$.
* The coupling matrix is $C_{pq} = \iint e_p\cdot\mathcal G e_q\,ds$ with the generator of the card,
  $\mathcal G e = \hat z\times e - (u\partial_v - v\partial_u)e$. **But it does not enter as the card implies.**
  $C$ is not antisymmetric on a rectangle: its symmetric part is a wall integral (surface charges times the
  normal velocity of the rotating wall). Projecting the field equations without dropping boundary terms gives
  $\boxed{dV/dz = \dots + \theta' C^{T}V,\quad dI/dz = \dots - \theta' C I}$. The "naive" form with $-\theta' C$
  in both equations is neither lossless nor reciprocal, and disagrees with Maxwell's equations (Galerkin check).
  For TE₁₀–TE₀₁ the symmetric part vanishes and nothing changes.
* $c = C_{\mathrm{TE01},\mathrm{TE10}} = 8/\pi^2$ for every aspect ratio (closed form, and quadrature to 1e-14).
* Selection rule: $C_{pq}\ne0$ only if $m_p+m_q$ **and** $n_p+n_q$ are odd. A twist always changes the family
  $n$. Also $C_{\mathrm{TM},\mathrm{TE}} = 0$ identically (while $C_{\mathrm{TE},\mathrm{TM}}\ne0$).
* Square guide, two-mode truncation: **exact** rotation of the polarization by $-c\theta$ in the local frame,
  $(1 - 8/\pi^2)\theta = 0.189\,\theta$ in the fixed frame, for any profile $\theta(z)$. Confirmed. The other
  modes change the splitting rate by $\approx +0.8(\tau/k)^2$ relative (uniform twist, 20 mm square at 10 GHz).
* End frames (CONVENTIONS §2 as updated by PR #7): the co-rotating frame is the guide frame, the twist's GSM has
  the same basis at both ports, no FrameChange. $Z(-90°)Z(+90°) = I$ at zero length (1e-15).
* Doane & Anderson eq. 18 transcribed as $P_c \simeq (8c\theta)^2/(\Delta\beta L_t)^4$ (confirmed on the scan; it
  is Unger 1957 eq. 20 applied to the triangle). Reviewer's numbers reproduced: 0.302 / 0.465 / 0.684 % (eq. 18)
  and 0.0054 / 0.0001 / 0.0452 % (first order, TE₁₀→TE₀₁) at 15 / 16.5 / 18 GHz. Eq. 18 is **not** an envelope of the
  first-order result: the exact first-order triangle is $4\sin^4(\Delta\beta L/4)$ times eq. 18 (envelope 4×,
  phase average 1.5×). At these three frequencies the first-order value happens to lie below eq. 18.
* Fig. 9: the full multimode solution reproduces the constant and cosine curves (e.g. cosine 0.35 / 0.87 / 0.94 %
  against about 0.35 / 0.85 / 1.03 %), but gives 0.11 / 0.25 / 0.60 % for the triangular twist against about
  0.05–0.08 %, through second-order conversion to TE₁₂, TE₀₃ and TM₁₂ (the modes Doane & Anderson name). Open
  issue for COMSOL in wave 2.

## 1. Twisted coordinates

The cross-section at $z$ is the port-1 rectangle rotated by $\theta(z)$ about $+\hat z$ (counterclockwise seen
looking along the propagation direction). Co-rotating coordinates $(u, v, z)$:

```latex
\mathbf r = R(\theta(z))\,(u\,\hat x + v\,\hat y) + z\,\hat z ,\qquad
\partial_u\mathbf r = \hat u,\quad \partial_v\mathbf r = \hat v,\quad
\partial_z\mathbf r = \hat z + \theta'\,\hat z\times\boldsymbol\rho ,
```

with $\boldsymbol\rho = u\hat u + v\hat v$. The walls are fixed at $u = \pm a/2$, $v = \pm b/2$. The metric and its
inverse, with $w = \theta'(-v, u)$ (the transverse velocity of the cross-section per unit $z$):

```latex
g_{ij} = \begin{pmatrix} 1 & 0 & w_u\\ 0 & 1 & w_v\\ w_u & w_v & 1+|w|^2\end{pmatrix},\qquad
G = g^{ij} = \begin{pmatrix} I + w w^T & -w\\ -w^T & 1\end{pmatrix},\qquad \sqrt g = 1 .
```

With covariant components, Maxwell's equations become those of a straight $a\times b$ guide filled with
$\varepsilon_0 G$, $\mu_0 G$:
$\nabla\times E = -i\omega\mu_0 G H$, $\nabla\times H = i\omega\varepsilon_0 G E$. The transverse covariant
components are the physical components on $\hat u, \hat v$ in the plane $z = $ const, so modal amplitudes defined in
this frame are the physical ones and are continuous at junctions. $E_z$ (covariant) $= E\cdot\partial_z\mathbf r$
vanishes on the walls because $\partial_z\mathbf r$ is tangent to them.

**Field equations.** Splitting into transverse and longitudinal parts and eliminating $E_z$, $H_z$:

```latex
E_z = w\cdot E_t + \tfrac{1}{i\omega\varepsilon_0}\nabla_t\!\cdot(H_t\times\hat z),\qquad
H_z = w\cdot H_t - \tfrac{1}{i\omega\mu_0}\nabla_t\!\cdot(E_t\times\hat z),
```

```latex
\partial_z E_t = [\text{straight}] + \nabla_t(w\cdot E_t) + (\hat z\times w)\,(\hat z\cdot\nabla_t\times E_t)
             = [\text{straight}] - \theta'\,\mathcal G E_t ,
```

and the same for $H_t$, where [straight] is the right-hand side of T0 with $h = 1$. The $ww^T$ terms cancel
exactly, so nothing of order $\theta'^2$ survives. Pointwise,
$\nabla_t(w\cdot F) + (\hat z\times w)(\hat z\cdot\nabla_t\times F) = -\theta'\mathcal G F$ with
$\mathcal G F = \hat z\times F - (u\partial_v - v\partial_u)F$. This is the generator of the active rotation
$F_\theta(\mathbf r) = R(\theta)F(R(-\theta)\mathbf r)$. It annihilates $F = (u, v)$, and the field of a fixed
pattern seen from the co-rotating frame obeys $\partial_z F = -\theta'\mathcal G F$. Both facts are checked in
`check_coefficients.py` (the generator against $d/d\theta$ of the rotated field, 2e-10). The card's generator is
correct.

## 2. Projection: where the boundary terms go

Expand $E_t = \sum_q V_q e_q$, $H_t = \sum_q I_q\,\hat z\times e_q$ with the CONVENTIONS fields. Project on
$e_p$ and $\hat z\times e_p$, and follow the rule of T0: integrate by parts only onto the mode functions. Use
only the boundary values of the true fields: $E_z = 0$ and tangential $E_t = 0$ on the walls. Never
differentiate the series term by term where a boundary trace is unknown. Then the series used are $L^2$
expansions: $\nabla\cdot(H_t\times\hat z)$ is expanded on the Dirichlet functions $\Pi_q$, and
$\nabla\cdot(E_t\times\hat z)$ on the Neumann functions $\Gamma_q$, with no boundary term in either.
The twist terms are

```latex
\frac{dV_p}{dz}\Big|_{tw} = \sum_q V_q\Big[-\!\iint d_p\,(w\cdot e_q) + \iint (e_p\cdot\hat z\times w)(\hat z\cdot\nabla\times e_q)\Big],
\qquad
\frac{dI_p}{dz}\Big|_{tw} = \sum_q I_q\Big[\iint (e_p\cdot w)\,d_q - \iint g_p\,(w\cdot\hat z\times e_q)\Big],
```

($d = \nabla\cdot e$, $g = \nabla\cdot(\hat z\times e)$). The $I$ bracket equals
$\iint(\hat z\times e_p)\cdot(-\theta'\mathcal G)(\hat z\times e_q) = -\theta' C_{pq}$, because the boundary term
contains the tangential part of $e_p$, which is zero. The $V$ bracket differs from $-\theta' C_{pq}$ by the wall
term $\oint (w\cdot e_q)(e_p\cdot n)\,dl$, which does not vanish: $e_p\cdot n$ is the surface charge. Using
$C + C^T = -\oint(e_p\cdot n)(e_q\cdot n)\,(\hat z\times\boldsymbol\rho)\cdot n\,dl$ (box T3-3), the $V$ bracket is
$+\theta' C_{qp}$. Hence:

> **Box T3-1 (coupled-mode equations of a twist, exact up to truncation).**
> ```latex
> \frac{dV_p}{dz} = -i\beta_p\frac{I_p}{Y_p} + \theta'(z)\sum_q C_{qp}V_q ,\qquad
> \frac{dI_p}{dz} = -i\beta_pY_pV_p - \theta'(z)\sum_q C_{pq} I_q ,\qquad
> C_{pq} = \iint e_p\cdot\mathcal G e_q\,ds .
> ```
> With $V = (a^+ + a^-)\sqrt{2/Y}$, $I = (a^+ - a^-)\sqrt{2Y}$ (CONVENTIONS §5), $r_{pq} = \sqrt{Y_p}/\sqrt{Y_q}$
> (principal roots):
> ```latex
> \frac{da^+}{dz} = -i\beta a^+ + \theta'(K^+a^+ + K^-a^-),\qquad
> \frac{da^-}{dz} = +i\beta a^- + \theta'(K^-a^+ + K^+a^-),\qquad
> K^\pm_{pq} = \tfrac12\big(r_{pq}C_{qp} \mp r_{qp}C_{pq}\big).
> ```
> `twist_formulas.system_matrix` (2N×2N, $y = [a^+; a^-]$) and `vi_system_matrix`.

Properties, for any $\theta'(z)$:

* **lossless**: $\frac{d}{dz}\mathrm{Re}\sum V_pI_p^* = 0$, because the $I$ matrix is minus the transpose of the
  $V$ matrix;
* **reciprocal**: $M_{22} = -M_{11}^T$ in $[V; I]$ form, the condition for a symmetric GSM;
* checked numerically: all GSMs built from it in `check_doane.py` and `check_square.py` have
  $|S - S^T|$ and $|S_p^HS_p - I|$ below 3e-13.

Writing $C = A + S$ (antisymmetric plus symmetric parts), the forward coupling is
$K^+_{pq} = -\tfrac12A_{pq}(r_{pq}+r_{qp}) + \tfrac12S_{pq}(r_{pq}-r_{qp})$. The S part is weighted by the
difference of admittances, so it matters near cutoff and for TE–TM pairs. The backward coupling is
$K^-_{pq} = -\tfrac12A_{pq}(r_{pq}-r_{qp}) + \tfrac12S_{pq}(r_{pq}+r_{qp})$. For TE₁₀–TE₀₁ ($S = 0$), $K^-$ is
proportional to the difference of the admittances. This is Katsenelenbaum's remark after eq. 4.6 (backward
coupling proportional to the difference of the wavenumbers).

**Why the naive form fails.** Using $-\theta'C$ in both equations (the projection of $-\theta'\mathcal G$ with
the series differentiated term by term) gives $M_{22} \ne -M_{11}^T$ whenever $S\ne0$. It is then neither
lossless nor reciprocal. Using only the antisymmetric part $-\theta'A$ in both equations is lossless and
reciprocal, but wrong. `check_galerkin.py` decides between the three (section 7).

## 3. Closed forms

Notation: $p = (m, n)$, $q = (m', n')$; $k = m\pi/a$, $l = n\pi/b$, $k' = m'\pi/a$, $l' = n'\pi/b$; $\chi$, $\chi'$
the cutoff wavenumbers; $K = k'^2 - k^2$, $\Lambda = l'^2 - l^2$; $\nu = \sqrt{\epsilon_m\epsilon_n}$ for TE and
$\nu = 2$ for TM; $N_{pq} = \nu_p\nu_q/(ab\,\chi\chi')$.

> **Box T3-2 (coupling matrix).** $C_{pq} = 0$ unless $m+m'$ and $n+n'$ are both odd. Otherwise
> ```latex
> C^{EE}_{pq} = -\frac{4N_{pq}\chi^2}{K^2\Lambda^2}\Big[(k^2+k'^2)\,l'^2\,\Lambda - k'^2(l^2+l'^2)\,K\Big],\qquad
> C^{EM}_{pq} = -8N_{pq}\,k'l'\Big(\frac{l^2}{\Lambda^2} + \frac{k^2}{K^2}\Big),
> ```
> ```latex
> C^{ME}_{pq} = 0,\qquad
> C^{MM}_{pq} = \frac{8N_{pq}\,kk'll'\,\chi'^2\,(K-\Lambda)}{K^2\Lambda^2},
> ```
> (superscripts: kind of $p$, kind of $q$; E = TE, M = TM). Dimensionless; independent of frequency.
> `twist_formulas.coupling_C`, `coupling_matrix`.

> **Box T3-3 (symmetric part).**
> $S_{pq} = \tfrac12(C_{pq}+C_{qp}) = -\tfrac12\oint(e_p\cdot n)(e_q\cdot n)\,(\hat z\times\boldsymbol\rho)\cdot n\,dl$.
> Equivalently, with $(A_p, B_p) = \frac{\nu_p}{\sqrt{ab}\chi}(-l, k)$ for TE and $\frac{2}{\sqrt{ab}\chi}(k, l)$ for TM:
> $S_{pq} = \frac{4b^2 nn'}{\pi^2(n'^2-n^2)^2}A_pA_q - \frac{4a^2mm'}{\pi^2(m'^2-m^2)^2}B_pB_q$.

Structure behind the boxes: $\mathcal G\nabla\Pi = -\nabla(L\Pi)$ with $L = u\partial_v - v\partial_u$, so
$C^{MM}_{pq} = \chi_q^2\iint\Pi_pL\Pi_q$, an antisymmetric matrix times $\chi_q^2$, and $C^{ME} = 0$ because
$\nabla\cdot(\nabla\Gamma\times\hat z) = 0$ and $\Pi_p = 0$ on the walls. The selection rule follows from parity:
$\mathcal G$ is odd under $u\to-u$ and under $v\to-v$, and a mode has parity $(-1)^{m+1}$, $(-1)^{n+1}$ under them.

**Particular values.** $C_{\mathrm{TE01},\mathrm{TE10}} = 4\nu\nu'\chi/(ab\chi' l^2) = 8/\pi^2$ for every $a/b$
(the $a$ and $b$ cancel). With the CONVENTIONS fields TE₁₀ points along $+\hat v$ and TE₀₁ along $-\hat u$, so
the sign is positive: rotating a $+\hat v$ field counterclockwise tilts it towards $-\hat u$. Other examples for
WR187 used tall ($a = 0.872''$, $b = 1.872''$), all from the closed form: $C(\mathrm{TE03},\mathrm{TE10}) = 0.2702$,
$C(\mathrm{TE05},\mathrm{TE10}) = 0.1621$, $C(\mathrm{TE21},\mathrm{TE10}) = -1.6845$ while
$C(\mathrm{TE10},\mathrm{TE21}) = 1.4534$ (S = −0.116), $C(\mathrm{TE10},\mathrm{TM21}) = -0.0538$. The TE₂₁ coupling
is *stronger* than TE₀₁'s.

**Check (acceptance test 1).** `check_coefficients.py`: closed forms against Gauss–Legendre quadrature of
$\iint e_p\cdot\mathcal Ge_q$ from `wrmodes.modes.transverse_field`, with complex-step derivatives. Five guides:
WR90 in both orientations, a 20 mm square, 44×30 mm, and WR187 tall. Each basis has 18–36 modes, all TE and TM
up to 3.2× the lowest cutoff. **Worst relative error 9.0e-15** (target 1e-10). The selection rule and
$C^{ME} = 0$ hold to 1e-11 in the quadrature values. $C + C^T$ matches the wall integral to 2.0e-15. It is
non-zero for 32–152 pairs per guide. $C_{\mathrm{TE01},\mathrm{TE10}} = 0.810569469138702$ for $a/b$ = 1, 2.25,
0.466, 10 and 0.1, against $8/\pi^2 = 0.810569469138702$.

**Literature.** Katsenelenbaum (1998) eq. 4.6 gives $S_{jm} = -\frac{8}{\pi^2}\frac{h_j+h_m}{2h_j}\Phi'$ and eq. 4.17
$A = \frac{8}{\pi^2}\frac{h_m+h_j}{2\sqrt{h_mh_j}}$. With power-normalized amplitudes this is
$|K^+| = c\,\tfrac12(r + 1/r)$ of box T3-1, magnitude identical. His p. 141 statement that twists couple "modes
separated by an odd integer in both indices" is the selection rule above. He also notes (p. 141) that for
TE₁₀–TE₀₁ "the additional contour integral … is equal to zero … these waves are coupled to each other by
transversal currents only". That contour integral is the wall term of box T3-3, which is zero for this pair and
not in general.

## 4. Square guide (a = b)

> **Box T3-4.** Two-mode truncation {TE₁₀, TE₀₁}, $a = b$: $\beta$ and $Y$ are equal, $S = 0$, so $K^- = 0$ and
> $K^+ = c\begin{pmatrix}0&1\\-1&0\end{pmatrix}$ (rows/columns TE₁₀, TE₀₁). For **any** $\theta(z)$ with TE₁₀
> incident:
> ```latex
> a^+_{10}(z) = \cos(c\,\theta(z))\,e^{-i\beta z},\qquad a^+_{01}(z) = -\sin(c\,\theta(z))\,e^{-i\beta z},\qquad c = 8/\pi^2,
> ```
> no reflection. The field $a_{10}e_{10} + a_{01}e_{01}$ is linearly polarized at the angle $\psi = -c\theta$ in the
> co-rotating frame, $\psi_{\text{fixed}} = (1 - 8/\pi^2)\theta = 0.18943\,\theta$ in the fixed frame. The
> polarization lags the guide. Full exchange TE₁₀ → TE₀₁ at $\theta = \pi/(2c) = \pi^3/16 = 111.4°$.

This agrees with Katsenelenbaum eq. 4.23 ($P_m = \cos(8\Phi/\pi^2)$, $P_j = -\sin(8\Phi/\pi^2)$, full conversion
every $\pi^3/16$ = 111°, "valid … for any arbitrary function Φ(z)"). It also confirms the corrected card: the
field does **not** follow the guide. It is not fixed in space either, as it would be in a circular guide.
`check_square.py` part 1: transfer-matrix integration for constant, raised-cosine and triangular profiles and
θ = 90°, 180°, 154.7°. Amplitude error 5.6e-15, no backward waves, ψ = −cθ (mod 180°) to 1e-12. The system matrix
equals box T3-4 to 1e-15.

**Other modes.** For a uniform twist $\theta' = \tau$ the system is $z$-invariant, and its eigenvalues give the
helical modes. The TE₁₀/TE₀₁ pair splits into $\gamma = \beta \mp c_{\text{eff}}\tau$. In the two-mode truncation
$c_{\text{eff}} = c$ exactly. Second-order couplings through the other modes cannot connect TE₁₀ to TE₀₁: TE₁₀
couples to $m'$ even, $n'$ odd, and TE₀₁ to $m'$ odd, $n'$ even, so no mode is shared. The diagonal shifts are
equal by symmetry. The correction is therefore even in τ. Full basis, 20×20 mm at 10 GHz (only TE₁₀/TE₀₁
propagate), evanescent modes included (`check_square.py` part 2), $c_{\text{eff}}/c - 1$:

| basis | τ/k = 0.01 | 0.03 | 0.1 | 0.2 |
| --- | --- | --- | --- | --- |
| below 2f (10 modes) | +2.0e-5 | +1.8e-4 | +2.0e-3 | +8.5e-3 |
| below 4f (44) | +6.0e-5 | +5.4e-4 | +6.0e-3 | +2.4e-2 |
| below 8f (178) | +7.3e-5 | +6.6e-4 | +7.2e-3 | +2.8e-2 |
| below 16f (716) | +8.3e-5 | +7.5e-4 | +8.2e-3 | +3.1e-2 |

So $c_{\text{eff}} \approx c\,[1 + \epsilon(\tau/k)^2]$ with $\epsilon\approx0.8$–0.9 here. It converges slowly
in the basis size (algebraically; see section 7). A practical twist has $\tau/k \lesssim 0.01$, where the
correction is below 1e-4.

Finite twists in an overmoded square guide (50.8 mm, raised-cosine 90° twists, all propagating modes, TE₁₀
incident, `check_square.py` part 3). Two-mode prediction: ψ = −72.95° (angles modulo 180°):

| f | modes | L | ψ_local | power in TE₁₀+TE₀₁ | reflected |
| --- | --- | --- | --- | --- | --- |
| 8 GHz | 10 | 0.1 m | −80.30° (ellipticity 0.38) | 0.593 | 3.4e-4 |
| 8 GHz | 10 | 0.3 m | −70.18° | 0.9964 | 1e-8 |
| 8 GHz | 10 | 1.0 m | −72.77° | 1.000000 | 6e-12 |
| 12 GHz | 24 | 0.1 m | +89.43° | 0.755 | 5e-5 |
| 12 GHz | 24 | 0.3 m | −65.98° | 0.694 | 1e-9 |
| 12 GHz | 24 | 1.0 m | −72.22° | 0.999999 | 4e-13 |

For slow twists (L = 1 m) the two-mode law holds within 0.2° (8 GHz) and 0.7° (12 GHz). For fast twists, conversion to the other
propagating modes dominates. These are the WP6 tests: the exact two-mode law (1e-12), and the full-basis
numbers above as regression values.

## 5. End frames

CONVENTIONS §2 (as updated by PR #7): *the global guide frame is attached to the guide and a twist rotates it
with the cross-section; a twist's GSM has the global frame at both ports and is not followed by a FrameChange.*
The co-rotating frame $(u, v)$ of section 1 is exactly this frame at every $z$. So:

> **Box T3-5.** A twist's GSM has `basis2 == basis1` (same $a$, $b$, modes and frame tag), with no frame
> bookkeeping. $\theta_{tot}$ changes only the orientation in space of the components after it. A twist given in
> the other plane frame (`plane = 'E'`, local sides swapped) has the coupling matrix $FCF^T$, with
> $F$ = `modes.eplane_frame_signs`. The same holds for the full system matrix, so the twist commutes with the
> H↔E frame change. Turning a twist around ($P\,S_{\text{swapped}}P$) gives the twist with profile
> $\theta'(L - z)$ and the **same** sign of θ: a helix keeps its handedness.

Checks (`check_frames.py`). For WR90, a 20 mm square and 30×17 mm, with 14–31 modes:
* zero length, $Z(\theta) = \exp(\theta M_1)$: $Z(-90°)Z(+90°) - I$ and $Z(+90°)Z(-90°) - I$ are at most 1.1e-15;
* H/E covariance of $C$ and of $M$ is at most 1.8e-16;
* reversal of a ramp twist: 3.6e-15 against the reversed profile with +θ, 1.7 against −θ.

**Caution for WP6 (information only).** Physically, a 90° twist of a *square* guide shrunk to zero length is the
90° relabeling of the modes (signed permutation $[[0,1],[-1,0]]$ on TE₁₀/TE₀₁). The truncated $\exp((\pi/2)M_1)$
does not converge to it. The TE₁₀/TE₀₁ block is $[[0.293, 0.956], [-0.956, 0.293]]$ with 4 modes,
$[[0.028, 1.015], [-1.015, 0.028]]$ with 96 and $[[0.026, 1.018], [-1.018, 0.026]]$ with 396. With L → 0 at fixed
N, θ′ exceeds every retained mode spacing, so the limits L → 0 and N → ∞ do not commute. Do not use a zero-length
twist as a physical frame change.

The superseded rule (twist followed by a FrameChange) is no longer needed. For the record, the overlap matrices
for rotations of the axes by 90°, 180° and 270° are signed permutations: (m, n) → (n, m) with $(-1)^m$ (TE) or
$(-1)^{m+1}$ (TM); (m, n) → (m, n) with $(-1)^{m+n}$; (m, n) → (n, m) with $(-1)^n$ (TE) or $(-1)^{n+1}$ (TM).
`twist_formulas.frame_change_matrix` gives them; they were checked against the field overlaps to 1e-12 before
PR #7.

## 6. Rate profiles and first-order conversion

First order, incident mode $q$: $a^+_p(L) = e^{-i\beta_pL}K^+_{pq}F(\beta_p - \beta_q)$ with
$F(\Delta\beta) = \int_0^L\theta'(z)e^{i\Delta\beta z}dz$.

> **Box T3-6 (profiles, total angle θ, length L).**
> ```latex
> P_{p} = |K^+_{pq}|^2\,|F|^2,\qquad
> \begin{array}{lll}
> \text{constant} & \theta' = \theta/L & |F| = \theta\,|\mathrm{sinc}(\Delta\beta L/2)|\\
> \text{triangular (Doane eq. 17)} & \theta' = \frac{2\theta}{L}\frac{2z}{L},\ \frac{2\theta}{L}\frac{2(L-z)}{L} & |F| = \theta\,\mathrm{sinc}^2(\Delta\beta L/4)\\
> \text{cosine (half-cosine)} & \theta' = \frac{\pi\theta}{2L}\sin\frac{\pi z}{L} & |F| = \theta\,\pi^2|\cos(\Delta\beta L/2)|/|\pi^2 - (\Delta\beta L)^2|\\
> \text{raised cosine} & \theta' = \frac{\theta}{L}(1-\cos\frac{2\pi z}{L}) & |F| = \theta\,|\mathrm{sinc}(\Delta\beta L/2)|\,\pi^2/|\pi^2-(\Delta\beta L/2)^2|
> \end{array}
> ```
> with $\mathrm{sinc}\,x = \sin x/x$. For TE₁₀–TE₀₁ far above cutoff, $|K^+|\simeq c$. `twist_formulas.rate_profile`,
> `profile_transform`, `first_order_power`. The closed forms agree with numerical integration to 1e-8.

Katsenelenbaum eq. 4.29 prints the second branch of the triangle as $4\theta(L-z)L^2$. It should read
$4\theta(L-z)/L^2$, as in Doane eq. 17.

**Eq. 18, transcribed from the scan** (Doane & Anderson 1989, p. 159 = PDF p. 7, right column):
"$P_c \simeq (8c\theta)^2/(\Delta\beta L_t)^4$. (18)". The next line reads
"$|\Delta\beta L_t| \simeq \frac{L_t\lambda}{a^2}\left|\frac{a^2}{b^2} - 1\right|\frac{\pi}{4}$", and then "The coupling
coefficient, c, is approximately $8/\pi^2$ (see reference 2)". The card's transcription is correct. Three checks:

1. **Origin.** Unger (1957) eq. 13 gives the unwanted-mode amplitude after one linear ramp of coupling,
   $c = c_0z/z_1$: $|w_2| = |c_0(1 - e^{j\Delta\beta z_1})/(\Delta\beta^2z_1)|$. His eq. 20 is the "average total
   conversion" of a bend with two ramps added in quadrature: $A_c = 4c_0^2/((\Delta\beta z_1)^2\Delta\beta^2)$. The
   triangle has two ramps of length $z_1 = L_t/2$ up to the peak coupling $c_0 = c\cdot2\theta/L_t$. This gives
   $A_c = 64c^2\theta^2/(\Delta\beta L_t)^4 = (8c\theta)^2/(\Delta\beta L_t)^4$, which is eq. 18. (Unger says
   "expressed in nepers"; it is the power fraction $|w_2|^2$.)
2. **Self-consistency.** With eq. 18 and $P_c < 0.01$, a 90° twist with $a = 2b$ needs
   $\Delta\beta L_t > 10.09$, i.e. $L_t > 4.3a^2/\lambda$. Doane & Anderson state "as long as … $5a^2/\lambda$". With a
   square instead of a fourth power the result would be $43a^2/\lambda$, so the fourth power is right.
3. **Relation to first order.** The exact first-order triangle is $c^2\theta^2\sin^4x/x^4$ with
   $x = \Delta\beta L_t/4$, i.e. $256c^2\theta^2\sin^4x/(\Delta\beta L_t)^4 = 4\sin^4x\cdot$(eq. 18). Its envelope is
   **4×** eq. 18 and its average over $x$ is 1.5× eq. 18. Eq. 18 is Unger's quadrature average: an
   order-of-magnitude estimate, **not an upper bound**. The card calls it "an envelope". That is true at 15, 16.5
   and 18 GHz only because $4\sin^4x < 1$ there.

**WR187 tall, 20-inch 90° twists** (`check_doane.py`; $a = 0.872''$, $b = 1.872''$, $c = 0.810569$):

| f | ΔβL (TE₁₀–TE₀₁) | eq. 18, exact Δβ | eq. 18, Doane's Δβ | 1st order TE₁₀→TE₀₁: const | tri | cosine | raised cos | 4× eq. 18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15 GHz | 13.612 | 0.302 % | 0.395 % | 0.874 % | 0.0054 % | 0.386 % | 0.064 % | 1.21 % |
| 16.5 GHz | 12.221 | 0.465 % | 0.579 % | 0.128 % | 0.0001 % | 0.789 % | 0.017 % | 1.86 % |
| 18 GHz | 11.100 | 0.684 % | 0.820 % | 2.361 % | 0.0452 % | 0.679 % | 0.525 % | 2.73 % |

The reviewer's values (0.30 / 0.47 / 0.68 % and 0.0054 / 0.0001 / 0.045 %) are reproduced.

**Against Fig. 9** (total spurious power). The Fig. 9 values are eyeball readings of the scan, ±0.05 % absolute;
the digitized curves are L1's job:

| f | profile | 1st order, all modes | full, all propagating modes | TE₁₀+TE₀₁ only | Fig. 9 | largest in full |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | constant | 0.940 | 1.387 | 1.212 | ≈1.35 | TE₀₁ 1.09 |
| 15 | cosine | 0.403 | 0.352 | 0.323 | ≈0.35 | TE₀₁ 0.35 |
| 15 | triangular | 0.048 | 0.114 | 0.0005 | ≈0.05 | TE₁₂ 0.080, TE₀₃ 0.019, TM₁₂ 0.011 |
| 16.5 | constant | 0.980 | 0.697 | 0.007 | ≈0.03 | TE₂₁ 0.61 |
| 16.5 | cosine | 0.791 | 0.867 | 0.870 | ≈0.85 | TE₀₁ 0.85 |
| 16.5 | triangular | 0.028 | 0.247 | 0.024 | ≈0.08 | TE₁₂ 0.169, TE₀₃ 0.043 |
| 18 | constant | 2.715 | 2.140 | 1.545 | ≈1.7 | TE₀₁ 1.82 |
| 18 | cosine | 0.698 | 0.943 | 1.038 | ≈1.03 | TE₀₁ 0.91 |
| 18 | triangular | 0.066 | 0.603 | 0.013 | ≈0.08 | TE₁₂ 0.384, TE₀₃ 0.158 |

(all in %). "Full" is the transfer-matrix solution of box T3-1 with all propagating modes (18–23), forward and
backward, 4th-order Magnus with 600 slices. Doubling the slices changes nothing in the 4th digit. Reciprocity
and unitarity are below 3e-13.

* The "cosine" of Fig. 9 is the half-cosine bump: it matches within the reading error at 15 and 16.5 GHz and
  about 0.09 % low at 18 GHz. The raised cosine does not match.
* Constant and cosine agree with Fig. 9 at about the 10–20 % level, except the constant twist at 16.5 GHz. There
  the full solution has 0.61 % in TE₂₁ (strong coupling, $|C|\approx1.6$, abrupt ends of a constant twist),
  while Fig. 9 shows ≈0.03 % and the TE₁₀+TE₀₁ solution 0.007 %.
* **Triangular: acceptance test met at first order** (all modes; maximum over 15–18 GHz **0.066 %**, below 0.1 %,
  the same order as Fig. 9). The full solution, however, gives 0.11–0.60 %. The excess is second-order
  conversion TE₁₀ → (TE₀₁, TE₀₃, TE₂₁) → TE₁₂, TE₀₃, TM₁₂. TE₁₂ and TM₁₂ are **not** coupled to TE₁₀ at first order
  (Δm = 0). Doane & Anderson write that in tall guide "other higher order modes such as TE₀₃, TE₁₂ and TM₁₂ become
  important", which are exactly these modes. Yet their triangular curve stays at 0.05–0.08 %, and their constant
  curve shows no TE₂₁ feature. Dropping the S term does not explain it: with only the antisymmetric part the
  triangular gives 0.111 / 0.238 / 0.580 %. **Open issue** for the wave-2 COMSOL campaign (WR187 tall, 20-inch
  triangular twist at 16.5 and 18 GHz is a good discriminating case).

## 7. Independent check against Maxwell's equations (Galerkin)

`check_galerkin.py`. For a uniform twist the problem in co-rotating coordinates is $z$-invariant. It is solved
directly by a Galerkin method on the weak form of $\nabla\times(G^{-1}\nabla\times E) = k^2GE$. Each Cartesian
component of $E$ is expanded in a trigonometric basis that satisfies tangential $E = 0$ wall by wall. The method
uses no modal expansion and no integration by parts of series. The coupled-mode eigenvalues of box T3-1 (forward
$\gamma$) are compared with the $\gamma$ at which the Galerkin problem has the eigenvalue $k^2$.

The test case is a strong twist, chosen to make the S term visible:

* guide 30×20 mm at 15 GHz, k = 314.38 m⁻¹;
* τ = 0.15k = 47.2 rad/m (90° in 33 mm);
* 9 propagating modes at τ = 0, 7 forward helical modes with real γ.

The Galerkin γ changes by 1.6e-5 k between 20² and 28² functions per component. Signed error
$(\gamma_{\text{CM}} - \gamma_{\text{Galerkin}})/k$ of the seven forward modes:

| coupled-mode equations | basis below 3f (81 modes) | 5f (235) | 8f (598) | extrapolated (1/cutoff → 0) |
| --- | --- | --- | --- | --- |
| **T3** (box T3-1) | up to 9.8e-3 | up to 6.2e-3 | up to 4.0e-3 | **all within 0.03–0.56e-3** |
| naive ($-\theta'C$ in both) | up to 5.0e-2 | 5.1e-2 | 5.2e-2 | −7e-3 to −5.3e-2 |
| antisymmetric part only | up to 3.3e-2 | 4.0e-2 | 5.4e-2 | −9e-3 to −6.4e-2 |

For T3 the error of every mode shrinks monotonically, roughly as 1/cutoff, and extrapolates to zero within
6e-4 k. That residual is consistent with a three-point extrapolation of a slowly converging series. The other
two forms converge to wrong values, off by 1–6 % of k. The slow (algebraic) convergence of the coupled-mode
series is expected: the true $H_t$ has a normal component $w_n H_z$ on the rotating walls, which no mode
function $\hat z\times e_q$ has. For practical twist rates ($\tau/k\lesssim10^{-2}$, i.e. 225 times weaker
than here) the truncation effect scales as $\tau^2$ and is negligible.

## 8. Notes for WP6

* Implement box T3-1 with `system_matrix`. It is linear in θ′. A constant-rate section is $\exp(LM)$; a variable
  rate needs a Magnus or ODE integrator (the 4th-order commutator-free Magnus scheme in `transfer_matrix` is
  enough: 600 slices for 0.5 m at 18 GHz).
* The basis must include the families $n\pm1$: a twist couples families, so per-family block structure does not
  apply. With `cutoff_factor = 1` the result converges only as fast as the omitted evanescent coupling allows.
  Section 4 shows effects of order $(\tau/k)^2$, negligible for practical twists.
* Tests: box T3-4 exactly; covariance and $Z(-\theta)Z(\theta) = I$ (box T3-5); reversal; reciprocity and
  unitarity; the WR187 numbers above as regression values; the Galerkin γ of section 7 as an independent
  reference.
