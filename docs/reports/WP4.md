# WP4 — Step junctions by mode matching: report

Branch `wave-1/WP4-steps`, from `main` at 223aeeb. Implementer card: `tasks/WP4-steps.md`.

## What was done

| Item | Files |
| --- | --- |
| Component protocol stub (byte-identical to the briefing; owned by WP1) | `src/wrmodes/components/__init__.py`, `src/wrmodes/components/base.py` |
| Closed-form overlaps of the normalized fields of two guides, one inside the other, any offset | `src/wrmodes/modematch/overlaps.py` |
| Symmetric mode-matching GSM of a junction | `src/wrmodes/modematch/junction.py` |
| `Step(a1, b1, a2, b2, du, dv, frame, cutoff_factor)` component, default truncation, `convergence_study` | `src/wrmodes/modematch/step.py` |
| Marcuvitz (1951) single-mode step circuits, and conversions circuit ↔ S | `src/wrmodes/modematch/marcuvitz.py` |
| Tests (72 + 1 expected failure) | `tests/test_steps.py` |
| Example: WR-90 → 22.86 × 1.6 mm → WR-90 | `examples/strangling_step.py` |
| exp(θK) study, data | `benchmarks/exp_theta_k_study.py`, `benchmarks/data/exp_theta_k_wr90.npz` |
| Public name | `wrmodes.Step` (one import line at the end of `src/wrmodes/__init__.py`) |

### Formulation

* **Overlaps.** Every field of `wrmodes.modes.transverse_field` is separable: $e_u = A_u\cos(k_m u')\sin(k_n v')$,
  $e_v = A_v\sin(k_m u')\cos(k_n v')$ with $(A_u, A_v) = c(k_m, k_n)$ for TM and $c(-k_n, k_m)$ for TE. So
  $X_{pq} = \iint_\text{small} e^\text{small}_p\cdot e^\text{large}_q\,ds = A_uA_u'\,I^u_{cc}I^v_{ss} + A_vA_v'\,I^u_{ss}I^v_{cc}$
  with 1D integrals $\int_0^{a_s}\cos(px)\cos(q(x+s))\,dx$ etc., evaluated as $A\cos(kA/2+\varphi)\,\mathrm{sinc}(kA/2)$
  (no division by zero for $p = q$). $s$ is the wall-to-wall offset. All TE/TM pairs and all families are covered.
* **Matching.** With $v = a^+ + a^-$, $i = a^+ - a^-$ (CONVENTIONS §5), the Galerkin projections of "$E_t$ continuous on
  the aperture, zero on the metal" and "$H_t$ continuous on the aperture" give $v_L = F v_S$, $i_S = F^T i_L$ with
  $F = \mathrm{diag}(\sqrt{Y_L})\,X^T\,\mathrm{diag}(1/\sqrt{Y_S})$ (principal roots). With $R = (I + F^TF)^{-1}$
  (size of the small guide's basis), large guide at port 1:
  $S_{11} = 2FRF^T - I$, $S_{12} = 2FR = S_{21}^T$, $S_{22} = 2R - I$. $R$ is symmetrized, so $S = S^T$ exactly. Since
  $X$ is real, the truncated equations conserve complex power exactly, so the propagating block is unitary at any
  truncation. Expansions use the same junction with the ports exchanged (same frame, no reversal signs).
* **Bases and truncation.** Each port has a `ModeBasis` of its own cross-section, both with the Step's `frame` tag
  (a mismatch raises). `default_basis(f, cutoff_factor)` keeps, on both sides, the modes with cutoff below
  `cutoff_factor·f` (same cutoff on both sides, so mode numbers are in proportion to the dimensions). A step that keeps
  $b$ conserves $n$ and a step that keeps $a$ conserves $m$: only the families $n$ (indices $m$) with a propagating
  mode on either side are then kept. Steps in $b$ or with $d_v \ne 0$ keep every family. With `cutoff_factor=None`
  the Step's own rule applies: $f_\max = \max(3f,\ M c/2d)$ with $M = 20$ half-periods across the smallest feature
  $d$ (the small side or the smallest wall gap, at least a quarter of the side) for steps in one dimension, and
  $M = 16$ across the smaller side of the small guide for steps in both dimensions.

## Results

### Overlaps and basic properties

* Overlaps vs Gauss-Legendre quadrature (160 × 160 points) of `transverse_field`, 70 GHz bases (up to 78 × 38
  modes): max |ΔX| / max|X| = 7.3e-15 (12 × 6 mm offset (3, −1) mm), 6.4e-15 (E-plane), 7.2e-15 (asymmetric
  H-plane), 8.8e-15 (corner offset), 1.0e-14 (identical guides, X = I).
* Identity step vs `straight(basis, f, 0)`: max |ΔS| = 1.6e-15 (default and explicit bases, all families, with
  evanescent modes).
* Reciprocity and unitarity, 6 geometries (E-plane reduction/expansion, symmetric and asymmetric H-plane, 2D steps
  with $d_u, d_v \ne 0$ in both directions) × 2 frequencies × up to 6 truncations (cutoff factor 1 to 30, up to
  ~1600 modes): reciprocity 0 (exact by construction), unitarity ≤ 2.2e-15. Also for unbalanced truncations
  (8:1 either way).
* `GSM.reversed()` of `Step(small→large, du, dv)` equals the directly built `Step(large→small, du, −dv)` to 1e-12,
  for $d_v \ne 0$ too, where omitting the signs $P = (-1)^{n+1}$ gives errors > 1e-3.

### Small → large → small at zero distance

`GSM.star` cannot do this cascade directly: the large-guide modes that do not couple to the aperture (null space of
$F^T$) are reflected with −1 on both sides, so $I - A_{22}B_{11}$ is exactly singular there and `np.linalg.solve`
returns garbage (no exception). These modes are neither excited nor seen by the outer ports. The test uses a
least-squares star product (`star_zero_length` in the test file) that drops them. It finds exactly
$N_L - N_S$ trapped modes in every case.

| Small guide, offset (mm) | f | Error, propagating block | Error, full S (evanescent modes too) |
| --- | --- | --- | --- |
| 22.86 × 2, (0, 0) | 10 GHz | 6.7e-16 | 6.7e-16 |
| 16 × 10.16, (2, 0) | 10 GHz | 9.0e-16 | 2.4e-15 |
| 18 × 8, (1, 0.5) | 10 GHz | 1.7e-15 | 3.7e-15 |
| 12 × 4, (−3, 2) | 20 GHz | 2.2e-15 | 3.6e-15 |

The maximum is over cutoff factors 1, 1.5, 3 and 8. The second junction is the step from large back to small at the
**same** position: `Step(small→large, du, −dv).reversed()`. For $d_v = 0$ this is `A.reversed()` itself. For
$d_v \ne 0$, `A @ A.reversed()` is not the identity, because turning the step around moves the small guide to
$-d_v$. Large → small → large is a zero-thickness iris: e.g. WR-90 → 22.86 × 2 mm → WR-90 at 10 GHz has
$|S_{21}| = 0.81$ (−1.89 dB with the default basis), reciprocal and unitary.

### Convergence (WR-90 → 22.86 × 2 mm, 10 GHz, TE10)

| Cutoff factor | Modes (port 1, port 2) | \|S11\| | Change |
| --- | --- | --- | --- |
| 20 | 27, 5 | 0.67556689 | — |
| 40 | 55, 11 | 0.67577027 | 2.0e-4 |
| 80 | 109, 21 | 0.67575724 | 1.3e-5 |
| 160 | 217, 43 | 0.67576413 | 6.9e-6 |
| 320 | 433, 85 | 0.67576274 | 1.4e-6 |
| 640 | 867, 171 | 0.67576327 | 5.3e-7 |
| 1280 | 1735, 341 | 0.67576322 | 5.6e-8 |

Unitarity ≤ 5.6e-16 at every truncation. The default basis (cutoff factor 150, 203 + 39 modes) gives 0.67576471.
Convergence is non-monotonic at low truncations (relative convergence) and smooth beyond a cutoff factor of about 80.
Other cases: a symmetric H-plane 22.86 → 16 mm step converges to ~2e-7 by cutoff factor 1280. A 2D step
(18 × 8 mm offset (1, 0.5) mm) is slower: change of |S| 1.6e-5 between cutoff factors 40 and 55 (4914 + 3053 modes).

### Marcuvitz (1951) single-mode circuits

**Transcribed formulas used.** Marcuvitz, *Waveguide Handbook*, Sec. 5.24 and 5.26, book pp. 296–310 (PDF pp.
310–324). Symbols: $\alpha$ = ratio of the small to the large dimension, $\beta = 1-\alpha$ (Sec. 5.24),
$\delta = 1-\alpha$ (Sec. 5.26), $\lambda_g$ / $\lambda_g'$ guide wavelengths, $\lambda$ free-space wavelength.
The time convention is $e^{j\omega t}$, the same as wrmodes.

* E-plane, symmetric, Sec. 5.26a, p. 307 (PDF 321), eqs. (1), (2a). Shunt $jB$ at the junction plane $T$ (both sides):
  $Y_0/Y_0' = b'/b = \alpha$;
  $\dfrac{B}{Y_0} = \dfrac{2b}{\lambda_g}\Big[\ln\Big(\dfrac{1-\alpha^2}{4\alpha}\Big(\dfrac{1+\alpha}{1-\alpha}\Big)^{\frac12(\alpha+\frac1\alpha)}\Big) + 2\dfrac{A+A'+2C}{AA'-C^2} + \Big(\dfrac{b}{4\lambda_g}\Big)^2\Big(\dfrac{1-\alpha}{1+\alpha}\Big)^{4\alpha}\Big(\dfrac{5\alpha^2-1}{1-\alpha^2}+\dfrac43\dfrac{\alpha^2C}{A}\Big)^2\Big]$,
  with (p. 308, PDF 322)
  $A = \big(\frac{1+\alpha}{1-\alpha}\big)^{2\alpha}\frac{1+\sqrt{1-(b/\lambda_g)^2}}{1-\sqrt{1-(b/\lambda_g)^2}} - \frac{1+3\alpha^2}{1-\alpha^2}$,
  $A' = \big(\frac{1+\alpha}{1-\alpha}\big)^{2/\alpha}\frac{1+\sqrt{1-(b'/\lambda_g)^2}}{1-\sqrt{1-(b'/\lambda_g)^2}} + \frac{3+\alpha^2}{1-\alpha^2}$,
  $C = \big(\frac{4\alpha}{1-\alpha^2}\big)^2$. Stated range and accuracy (p. 308): "$b/\lambda_g < 1$ … correct to
  within 1 per cent". Approximations (2b), (2c), (2d) on p. 308 were also transcribed and used only to check the
  transcription of (2a).
* E-plane, asymmetric, Sec. 5.26b, p. 310 (PDF 324): "Same as Sec. 5.26a except $\lambda_g$ is replaced by
  $\lambda_g/2$", restrictions included.
* H-plane, symmetric, Sec. 5.24a, p. 296 (PDF 310), wide-aperture forms: shunt inductive $jX$ at $T$ (wide-guide
  side), narrow-guide reference plane $T'$ at distance $l$ on the wide-guide side (Fig. 5-24-1).
  (1c) $\frac{Z_0'}{Z_0} \approx \frac{\lambda_g'a'}{\lambda_g a}\big(1+\beta+\frac{\beta^2}{2}\big)$;
  (2c) $\frac{Z_0}{X} \approx \frac{\lambda_g}{2a}\,\frac{\beta^2(1+\beta)\ln\frac2\beta}{1-\frac\beta2}\Big(1 - \frac{27}{8}\frac{Q+Q'}{1+8\ln\frac2\beta}\Big)$;
  (3c) $l/a \approx 0$; with $Q = 1-\sqrt{1-(2a/3\lambda)^2}$, $Q' = 1-\sqrt{1-(2a'/3\lambda)^2}$ (p. 297).
  Stated (p. 298): range $0.5 < a/\lambda < 1.5$, $2a'/\lambda > 1$; (1c), (2c), (3c) "are in error by less than 6 per
  cent for $0.5 < (a/\lambda) < 1$ and $\beta < 0.3$".
* H-plane, asymmetric, Sec. 5.24b, p. 301 (PDF 315):
  (4c) $\frac{Z_0'}{Z_0} \approx \frac{\lambda_g'/a'}{\lambda_g/a}\big(1-\beta-\frac{7\beta^2}{4}\big)$;
  (5c) $\frac{Z_0}{X} \approx \frac{\lambda_g}{2a}\frac{\beta^2}{1-\frac\beta2-\beta^2(\ln\frac2\beta+\frac32)}\Big\{\frac74\ln\frac2\beta\big[1+\beta-\frac54\beta^2\ln\frac2\beta\big]\big[1+\frac47(1-\beta)^2(\frac a\lambda)^2\big] - \frac89\big[1+3\beta\ln\frac2\beta\big][Q+Q']\Big\}$;
  (6c) $l/a \approx 0$; with $Q = 1-\sqrt{1-(a/\lambda)^2}$, $Q' = 1-\sqrt{1-(a'/\lambda)^2}$ (p. 302).
  Stated (p. 302): range $0.5 < a/\lambda < 1.0$; (4c), (5c), (6c) "in error by less than 10 per cent for
  $\beta < 0.2$ and $0.5 < a/\lambda < 0.8$".
* The full H-plane formulas (1a)–(3a), (4a)–(6a) with $X_{11}, X_{22}, X_0, A, A', C, R_1, R_2, T, N_{11}, N_{22}, N_{12}$
  (pp. 296–297, 301–302) are transcribed in `marcuvitz.hplane_step_circuit` but **not** used for acceptance (see open
  issues).

**Extraction.** E-plane: $y = (1-S_{11})/(1+S_{11})$ of TE10 in the high guide gives $Y_0'/Y_0 = \mathrm{Re}\,y$ and
$B/Y_0 = \mathrm{Im}\,y$. H-plane: $\mathrm{Re}\,y = Z_0/Z_0'$ and $\mathrm{Im}\,y = -Z_0/X$ from $S_{11}$, then $l$ from
the phase of $S_{22}$ moved from $T'$ to the junction plane. $S_{11}$ is the voltage reflection coefficient, since the
mode voltage is proportional to $a^+ + a^-$.

**E-plane (2a), default truncation.** In every case $\mathrm{Re}\,y = 1/\alpha$ to 1e-15, as in Marcuvitz's eq. (1).

| $b/\lambda_g$ | $b'/b$ | B/Y0 mode matching | B/Y0 (2a) | Difference |
| --- | --- | --- | --- | --- |
| 0.2 | 0.1 / 0.3 / 0.5 / 0.7 / 0.9 | 0.77701 / 0.34514 / 0.15979 / 0.05808 / 0.00771 | 0.77663 / 0.34505 / 0.15976 / 0.05807 / 0.00771 | +0.05 / +0.02 / +0.02 / +0.01 / −0.03 % |
| 0.5 | same | 2.08711 / 0.95779 / 0.44437 / 0.15810 / 0.02023 | 2.08153 / 0.95719 / 0.44395 / 0.15794 / 0.02016 | +0.27 / +0.06 / +0.09 / +0.10 / +0.32 % |
| 0.8 | same | 4.17800 / 2.09771 / 0.96868 / 0.32102 / 0.03732 | 4.15034 / 2.09435 / 0.96650 / 0.32001 / 0.03697 | +0.67 / +0.16 / +0.23 / +0.31 / +0.95 % |
| 0.2 (asymmetric, $\lambda_g/2$) | same | 1.61516 / 0.73021 / 0.33857 / 0.12167 / 0.01584 | 1.61251 / 0.73001 / 0.33836 / 0.12160 / 0.01580 | +0.16 / +0.03 / +0.06 / +0.06 / +0.26 % |

Converged values (m = 1 modes only, cutoff factor up to 800): at $b/\lambda_g = 0.8$, $b'/b = 0.9$ mode matching gives
+1.02 %, marginally outside the stated 1 %. At $b/\lambda_g = 0.95$, $b'/b = 0.7$ it gives +0.52 %, and at
$b/\lambda_g = 0.2$, $b'/b = 0.9$ it gives +0.06 %. The test allows 1.05 % for the first case only.

**H-plane, wide-aperture forms (default truncation).**

| Case | a/λ | β | Z0'/Z0 mode matching, (c), diff. | X/Z0 mode matching, (c), diff. | l/a mode matching |
| --- | --- | --- | --- | --- | --- |
| asym. | 0.6 | 0.10 | 1.4349, 1.4350, +0.0 % | 9.948, 10.186, +2.4 % | 0.0018 |
| asym. | 0.7 | 0.20 / 0.15 / 0.10 | 1.3973 / 1.2227 / 1.1267; +1.5 / +0.7 / +0.1 % | 4.210 / 7.158 / 14.678; +3.9 / +2.5 / +1.1 % | 0.0054 / 0.0037 / 0.0020 |
| asym. | 0.75 | 0.20 / 0.15 / 0.10 | 1.2092 / 1.1359 / 1.0859; +1.8 / +0.9 / +0.2 % | 4.920 / 8.261 / 16.721; +2.9 / +1.8 / +0.7 % | 0.0056 / 0.0038 / 0.0021 |
| sym. | 0.7 | 0.25 / 0.20 / 0.10 / 0.05 | 2.1797 / 1.5078 / 1.1440 / 1.0605; +1.2 / +0.6 / 0.0 / 0.0 % | 5.014 / 7.738 / 27.80 / 96.91; +9.0 / +6.6 / +4.5 / +4.3 % | 0.0015 / 0.0011 / 0.0004 / 0.0001 |
| sym. | 0.85 | 0.30 / 0.25 / 0.20 / 0.10 / 0.05 | 1.3747 / 1.2364 / 1.1567 / 1.0624 / 1.0288; +2.2 / +1.3 / +0.7 / 0.0 / 0.0 % | 5.000 / 7.209 / 11.02 / 38.81 / 134.3; +11.3 / +8.3 / +6.9 / +6.7 / +7.0 % | 0.0019 … 0.0001 |
| sym. | 0.95 | 0.30 / 0.25 / 0.20 / 0.10 / 0.05 | 1.1867 / 1.1308 / 1.0937 / 1.0420 / 1.0203; +2.3 / +1.4 / +0.8 / 0.0 / 0.0 % | 6.047 / 8.653 / 13.12 / 45.38 / 155.9; +9.9 / +7.7 / +7.1 / +8.6 / +9.5 % | 0.0020 … 0.0001 |

These cases are converged (e.g. a/λ = 0.95, β = 0.1: X/Z0 = 45.352, 45.274, 45.265 at cutoff factors 20, 100, 800).
The asymmetric step agrees within the stated 10 %. For the symmetric step, $Z_0'/Z_0$ agrees within the stated 6 %
(≤ 2.3 %). $X/Z_0$ from (2c) is 4–11 % above mode matching: within 6 % at a/λ = 0.7, β ≤ 0.1, and outside it
elsewhere. For a check beyond (2c), I read Marcuvitz's Fig. 5-24-3 (p. 300, two-mode equivalent static method)
by eye, roughly ±3 %, at a/λ = 1.4, outside the range of (2c). At α = 0.5 the figure gives $\frac{\lambda_g}{2a}\frac{X}{Z_0}$ ≈ 1.75
against 1.80 from mode matching, and at α = 0.6 it gives ≈ 3.35 against 3.17. Fig. 5-24-2 at α = 0.5 gives
$\frac{Z_0'\lambda_g a}{Z_0\lambda_g' a'}$ ≈ 1.45 against 1.48. $l/a$ from mode matching (1.1e-3 to 5.6e-3) is
positive, so $T'$ is on the wide-guide side as in the figure, and of the size of Fig. 5-24-2 (peak ≈ 2.5e-3 for the
symmetric case).

### exp(θK) study

`benchmarks/exp_theta_k_study.py` → `benchmarks/data/exp_theta_k_wr90.npz`. The study takes a WR-90 90° bend
(θ = π/2), positive κ, at 7–100 GHz in steps of 0.25 GHz, leaving out frequencies within 1e-4 of a cutoff (372
frequencies). The basis holds the propagating modes of the input family: H plane n = 0, input TE1,0; E plane local
n = 1, input TE0,1. The transfer matrix is $\Phi = e^{\theta K}$ with K from `coupled_mode_matrices`, converted with
`from_transfer_matrix`. The file stores the full GSM, the mode names, the defects and the power balance per frequency,
with `meta` as in CONVENTIONS §9.

* **Unitarity and reciprocity:** max 4.6e-15 and 4.0e-15 (H), 2.1e-15 and 2.1e-15 (E) over all frequencies. The
  power balance (kept + converted + reflected) is 1 to 1e-10. This is expected: $D + \kappa K$ conserves power for
  every κ, so K is skew-Hermitian with respect to diag(I, −I), and $e^{\theta K}$ preserves that form.
* **Single-mode band** (below 13.11 GHz for H, below 16.15 GHz for E): K restricted to one mode is zero, so the GSM
  is the identity: $S_{21} = 1$, $S_{11} = 0$ exactly.
* **Overmoded band**, power kept in the input mode / converted / reflected:

| f (GHz) | H plane: N, kept, converted, reflected | E plane: N, kept, converted, reflected |
| --- | --- | --- |
| 20 | 3, 0.075, 0.811, 0.114 | 3, 0.051, 0.898, 0.050 |
| 30 | 4, 0.056, 0.943, 0.0010 | 3, 0.891, 0.109, 0.0006 |
| 45 | 6, 0.014, 0.986, 0.0001 | 7, 0.211, 0.565, 0.225 |
| 70 | 10, 0.0042, 0.995, 0.0004 | 9, 0.349, 0.651, 0.0002 |
| 90 | 13, 0.00001, 0.9998, 0.0002 | 13, 0.331, 0.639, 0.030 |

  Largest total reflection: 0.475 (H, 26.25 GHz) and 0.712 (E, 30.25 GHz). In the H plane, 50–100 GHz, at most
  2.7 % of the power stays in TE10. In the E plane the power kept in the input mode varies between 0 and 0.89
  above 20 GHz.
  Whether and how far this differs from a real miter is for the comparison with WP3 at Gate A.

## Acceptance tests

| Check | Target | Result | Value |
| --- | --- | --- | --- |
| Overlaps vs quadrature | 1e-12 relative | pass | ≤ 1.0e-14 |
| Identity step = zero-length straight | 1e-12 | pass | 1.6e-15 |
| Reciprocity | 1e-10 | pass | 0 (exact by construction) |
| Unitarity, propagating modes, every truncation | 1e-10 | pass | ≤ 2.2e-15 |
| Small → large → small at zero distance = identity, propagating modes (full rank of $X^T$) | 1e-10 | pass, with a least-squares star product (plain `GSM.star` is singular here, see open issues) | ≤ 2.2e-15 (full S ≤ 3.7e-15) |
| Large → small → large is an iris, not the identity | — | pass | \|S21\| = 0.81 for WR-90 / 2 mm / WR-90 |
| E-plane step vs Marcuvitz (2a), symmetric and asymmetric, $b/\lambda_g < 1$ | 1 % | pass, with one marginal case | ≤ 0.95 % (default truncation), 1.02 % converged at $b/\lambda_g = 0.8$, $b'/b = 0.9$ |
| H-plane asymmetric step vs (4c), (5c) | 10 % | pass | Z ≤ 1.8 %, X ≤ 3.9 % |
| H-plane symmetric step vs (1c) | 6 % | pass | ≤ 2.3 % |
| H-plane symmetric step vs (2c) | 6 % | **fail** at most points | 4.3–11.3 % (xfail test) |
| Convergence of \|S11\|, WR-90 → 22.86 × 2 mm, two finest truncations | < 1e-4 | pass | 1.4e-6 (cutoff factors 160 → 320 in the test); 5.6e-8 at 640 → 1280 |

Full suite: 165 passed, 1 xfailed (`PYTHONPATH=$PWD/src python3 -m pytest -q`). `tests/test_steps.py` takes about
35 s on the shared container.

## Open issues

1. **Symmetric H-plane step vs Marcuvitz (2c):** mode matching gives a reactance X/Z0 4–11 % below (2c). Marcuvitz
   states 6 %. Mode matching is converged, the same code agrees with the E-plane (2a) to 1 % and with the asymmetric
   H-plane (5c) to 4 %, and it agrees roughly with Fig. 5-24-3. The deviation is not settled. It needs WP7 (D6
   transcription, checked by the second L1 agent) and the WP3 2D solver.
2. **Marcuvitz full H-plane formulas (1a)–(3a), (4a)–(6a)** as transcribed agree with his small-aperture forms in the
   quasi-static limit for α ≤ 0.2: $Z_0'/Z_0 \to 1.750$, (2b) within 4 %, (3b) within 2 %. But the frequency-dependent terms
   $N_{11}, N_{22}, N_{12}$ make $X/Z_0$ blow up and change sign for a/λ ≳ 0.7. For example, at α = 0.7, a/λ = 1.4
   they give −65 where Fig. 5-24-3 gives ≈ 5.8. I re-read every symbol at 400 dpi. Two printed details look
   suspicious: $\lambda_g$ (not $\lambda_g'$) in the denominator of (2a), and $1+3\alpha^2$ under $16\alpha^2R_2$ in
   $N_{22}$ of Sec. 5.24b. The book gives no expression for $X_{12}$, only $X_0 \approx A' - C^2/A$. Simple rescalings
   of the N terms do not fix it. This is either a misprint in the book or a transcription error I could not find,
   so these formulas are not used for acceptance. **L1 (D6) and WP7 should check them independently.**
3. **Zero-distance cascades in `GSM.star` (for WP1):** joining two junctions on the same plane (S → L → S with
   no straight between) makes $I - A_{22}B_{11}$ exactly singular: the trapped modes of the middle guide reflect with
   −1 on both sides. `np.linalg.solve` then returns wrong numbers without raising. The same can happen at the
   resonances of modes that are exactly decoupled from the outer ports. WP1 may want a condition check or a
   least-squares fallback (see `star_zero_length` in `tests/test_steps.py`).
4. **Default truncation in a cascade:** `Step.default_basis(f, cutoff_factor=None)` applies the Step's own rule. If
   WP1's cascade passes `cutoff_factor=1.0` (the default of the protocol stub), a step gets no evanescent modes, and
   its result is then only the zeroth-order mode-matching answer (e.g. |S11| = 0.6711 with cutoff factor 3 instead of 0.6758 for the 2 mm
   step). A cascade should leave the truncation of mode-matching components to them, or pass a large factor.
5. 2D steps (both sides change) converge slowly: about 1e-4 in |S| needs thousands of modes. The cost is dominated by
   $F^TF$. Symmetry reduction (parity classes for centred steps) is not implemented.
6. A frequency exactly at the cutoff of any basis mode raises (from `wave_admittance`, CONVENTIONS §3).

## Conventions issues

* None blocking. Two clarifications:
  * §3 defines families by $n$ only. A step that keeps $a$ (with $d_u = 0$) also conserves $m$, and the default basis
    uses that to drop the $m$ that cannot couple to a propagating mode. This is exact and is documented in `Step`.
  * The card's "step and its reverse" wording: with $d_v \ne 0$, `A @ A.reversed()` is not the identity, because
    turning a component around maps $v \to -v$ (§2) and moves the small guide to $-d_v$. The identity uses the step
    back at the same position, `Step(small→large, du, −dv).reversed()`. This follows from §2 as written.

## For other cards

* **WP1:** the component stub was created byte-identical (verified with `diff` against the briefing). `Step` derives
  from `Component`; `Step.default_basis(f, cutoff_factor=None)`. See open issues 3 and 4.
* **WP2:** `wrmodes.modematch.junction_gsm(basis_large, basis_small, X, f, large_port)` is the reusable kernel. Any
  overlap matrix $X$ (small × large) can be passed, e.g. for impedance-walled or sheet-loaded guides once their
  modes are written in the §5 normalization.
* **WP7 / L1:** Marcuvitz transcription notes above. The asymmetric E-plane case is Sec. 5.26b with $\lambda_g/2$.
  Asymmetric means one broad wall continuous, i.e. $d_v = \pm(b-b')/2$.
* **WP3 (Gate A):** the exp(θK) data are in `benchmarks/data/exp_theta_k_wr90.npz`; the layout is described in its
  `meta`.
