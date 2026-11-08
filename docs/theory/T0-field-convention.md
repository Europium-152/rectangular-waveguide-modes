# T0 — Mode-field convention behind the wrmodes 0.1 coupling coefficients

*Theory note, wave 0. Status: derived and verified numerically (`tests/test_conventions.py`).*

## Question

The closed-form coupling coefficients of wrmodes 0.1 (JINST paper, eqs. 5.3–5.11) are written with the
normalized potentials Π (TM) and Γ (TE), but the paper does not state the transverse fields, or their signs,
that go with them. Every new solver (2D FEM, mode matching, tapers, twists, losses) must produce amplitudes in
exactly the same basis, otherwise cascading silently mixes conventions. This note derives the coupled-mode
equations of a bend from Maxwell's equations for an *arbitrary* set of real transverse mode fields, and finds
the field definitions that reproduce the existing coefficients, signs included.

## Setting

Time dependence $e^{+i\omega t}$, forward modes $\propto e^{-i\beta z}$. Bend coordinates $(u, v, z)$,
right-handed, with scale factors $h_1 = h_2 = 1$ and $h_3 = h = 1 + \kappa(z) u$. The frame is orthogonal for any
planar curvature profile κ(z), and the transverse components of Maxwell's equations ($\nabla\times E = -i\omega\mu_0 H$,
$\nabla\times H = i\omega\varepsilon_0 E$) can be written

```latex
\partial_z E_t = \nabla_t (h E_z) + i\omega\mu_0\, h\, \hat z\times H_t ,\qquad
\partial_z H_t = \nabla_t (h H_z) - i\omega\varepsilon_0\, h\, \hat z\times E_t ,
```

```latex
i\omega\varepsilon_0 E_z = \nabla_t\cdot(H_t\times\hat z),\qquad
-i\omega\mu_0 H_z = \nabla_t\cdot(E_t\times\hat z).
```

## Expansion

Let $e_p(u,v)$ be real transverse fields of the straight guide with $\iint e_p\cdot e_q\,ds = \delta_{pq}$, wave
admittances $Y_p$ ($\beta_p/(\omega\mu_0)$ for TE, $\omega\varepsilon_0/\beta_p$ for TM) and

```latex
E_t = \sum_q V_q\, e_q ,\qquad H_t = \sum_q I_q\, \hat z\times e_q .
```

Projecting the two transverse equations on $e_p$ and $\hat z\times e_p$ and integrating the gradient terms by
parts gives the equations below. Two sets of boundary terms vanish on perfectly conducting walls:

* those of the first integration, because $E_z = 0$ and $e_p\cdot\hat t = 0$ on the walls;
* those hidden in writing $E_z$ (and $H_z$) as the term-by-term divergence of the $H_t$ (and $E_t$) series, which
  needs $d_p = -\chi_p^2\Pi_p = 0$ on the walls.

The second set is exactly what changes with impedance walls (T1) and with tilted walls (T2).

```latex
\frac{dV_p}{dz} = -i\omega\mu_0 \sum_q I_q \iint h\, e_p\cdot e_q
   + \frac{i}{\omega\varepsilon_0}\sum_q I_q \iint h\, d_p d_q ,
```

```latex
\frac{dI_p}{dz} = -i\omega\varepsilon_0 \sum_q V_q \iint h\, e_p\cdot e_q
   + \frac{i}{\omega\mu_0}\sum_q V_q \iint h\, g_p g_q ,
```

with $d_p = \nabla_t\cdot e_p$ and $g_p = \nabla_t\cdot(\hat z\times e_p)$. For $h = 1$ these reduce to the
transmission-line equations of each mode, $dV/dz = -i\beta I/Y$, $dI/dz = -i\beta Y V$ (checked for TE and TM).

## Power waves

With $V_p = (a^+_p + a^-_p)\sqrt{2/Y_p}$ and $I_p = (a^+_p - a^-_p)\sqrt{2Y_p}$ (so that $|a^\pm|^2$ is the power of
a propagating mode), the terms proportional to $\kappa$ give exactly the form of the paper,
$da^+_p/dz = -i\beta_p a^+_p - i\kappa\sum_q (C^+_{pq} a^+_q + C^-_{pq} a^-_q)$, with

```latex
C^{\pm}_{pq} = \tfrac12\Big[\pm\sqrt{Y_p}\sqrt{Y_q}\,\big(\omega\mu_0 U_{pq} - U^d_{pq}/(\omega\varepsilon_0)\big)
             + \big(\omega\varepsilon_0 U_{pq} - U^g_{pq}/(\omega\mu_0)\big)/\big(\sqrt{Y_p}\sqrt{Y_q}\big)\Big],
```

(written with $\sqrt{Y_p}\sqrt{Y_q}$, which differs from $\sqrt{Y_pY_q}$ by the branch when the admittances are
complex). The backward amplitudes obey $da^-_p/dz = +i\beta_p a^-_p + i\kappa\sum_q(C^-_{pq}a^+_q + C^+_{pq}a^-_q)$,
which is the lower half of the wrmodes matrix $K$ and is tested as well.

```latex
U_{pq} = \iint u\, e_p\cdot e_q\, ds,\quad U^d_{pq} = \iint u\, d_p d_q\, ds,\quad U^g_{pq} = \iint u\, g_p g_q\, ds .
```

## Result

* **TE–TE and TM–TM.** With $e_p$ proportional to $\nabla\Gamma_p$ or $\nabla\Pi_p$ (either sign), the formula
  reproduces the paper's TE–TE and TM–TM coefficients term by term. These are insensitive to the sign of the
  fields.
* **TM–TE.** For $e_{TM} = s_M\nabla\Pi$ and $e_{TE} = s_E\,\hat z\times\nabla\Gamma$ one finds
  $U_{pq} = -s_M s_E\, I_3$, where $I_3$ is the paper's integral
  $\iint u(\partial_x\Pi_p\,\partial_y\Gamma_q - \partial_y\Pi_p\,\partial_x\Gamma_q)$. The paper's
  $C^{\pm} = k(\beta_p\pm\beta_q)I_3/(2\sqrt{\beta_p\beta_q})$ requires $s_M s_E = -1$.

The convention adopted for wrmodes 1.0 is therefore

```latex
e_{TM} = \nabla_t\Pi ,\qquad e_{TE} = \nabla_t\Gamma\times\hat z\;(= -\hat z\times\nabla_t\Gamma),
```

with Π and Γ exactly as in eq. 5.4 of the paper (arguments $u + a/2$, $v + b/2$; positive prefactors).

## Verification

`tests/test_conventions.py::test_closed_form_coupling_follows_from_the_field_convention` evaluates $U$, $U^d$ and
$U^g$ by Gauss–Legendre quadrature from `wrmodes.modes.transverse_field` and compares $C^\pm$ with
`coupled_mode_matrices` for four guide/family/frequency cases with TE and TM modes, for all four blocks of $K$:
agreement to 1e-10 relative (quadrature-limited). `test_te_tm_sign_matters` confirms that the opposite relative sign fails. The same
derivation is an independent check of the structure of eqs. 5.3 of the paper.

## Uses beyond wave 0

The two projected equations hold for any $h(u, v, z)$ that multiplies the $z$ scale factor. Wall loss, tapers
and twists change the projection differently (boundary terms and moving cross-sections), so theory notes T1–T3
start from the same expansion but must re-derive the boundary terms.
