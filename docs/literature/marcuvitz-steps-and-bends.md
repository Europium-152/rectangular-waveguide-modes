# Marcuvitz, *Waveguide Handbook* (1951): steps, corners and circular bends (D6)

Source: N. Marcuvitz, *Waveguide Handbook*, MIT Radiation Laboratory Series vol. 10, McGraw-Hill (1951); the file in
Daniel's folder is the IET reprint. Book pages 296-335 = PDF pages 310-349 (offset 14). Sections covered: 5.24
(H-plane step, symmetric and asymmetric), 5.26 (E-plane step, symmetric and asymmetric), 5.28 (E-plane corners:
right-angle and arbitrary angle), 5.29 (H-plane corners: right-angle and arbitrary angle), 5.34 and 5.35 (circular
bends, E- and H-plane). Sections 5.25 (H-plane bifurcation), 5.27 (coaxial steps) and 5.30-5.33 (radial, coaxial,
circular-guide and aperture couplings) lie in the same page range but are not steps or bends and are not transcribed.

**No mitered (chamfered) bend is treated in this page range.** The "corners" of 5.28 and 5.29 are sharp
(unchamfered) junctions of two equal guides; the arbitrary-angle corners are drawn with the reference planes through
the inner corner. Alessandri et al. (1994) cite Marcuvitz [4] only for "approximate formulas" for chamfering.

Method: L1a and L1b transcribed these pages independently (raw files `docs/literature/raw/a/` and `raw/b/`), each
from renders at 220 dpi and every equation again at 400-600 dpi. L1 compared the two transcriptions symbol by symbol
(a normalized LaTeX diff of every equation: no difference in any symbol, exponent, index or sign; the differences
are layout and the order in which definitions are listed), re-read the doubtful places on 500-600 dpi renders, and
evaluated the formulas against the book's own curves (`benchmarks/literature/scripts/marcuvitz_checks.py`). This
file is L1b's text with L1a's notes and L1's corrections and checks merged; the *Checking record* at the end lists
every disagreement and how it was settled. Notation is the book's: `λ` free-space wavelength, `λ_g` guide
wavelength, `Y_0`, `Z_0` characteristic admittance/impedance, `j` = imaginary unit with `e^{jωt}`. Numerical
cross-checks test the transcription (and, for Sec. 5.24, the book's formulas), not wrmodes.

---

## 5.24 Junction of two rectangular guides, H-plane (change of width)

### a. Symmetrical case (pp. 296-298, PDF 310-312)

"A junction of two rectangular guides of unequal widths but of equal heights (H₁₀-mode in each rectangular guide)."
Fig. 5.24-1: widths `a` (larger) and `a'` (smaller), symmetric about the axis. Equivalent circuit: shunt inductive
reactance `jX` between a line `Z_0, λ_g` (reference plane `T`, larger guide) and a line `Z_0', λ_g'` (plane `T'`,
smaller guide); `T'` lies a distance `l` from `T`, on the side of the larger guide (top view).

"Equivalent-circuit Parameters.—At reference planes T and T' for the larger and smaller guide, respectively,"

$$\frac{Z_0'}{Z_0} = \frac{\lambda_g' a'}{\lambda_g a}\,\frac{X_{11}}{\alpha^2 (X_{22} - X_0)}\left[1 + \left(\frac{2a' X_0}{\lambda_g'}\right)^2\right], \tag{1a}$$

$$\frac{Z_0'}{Z_0} \approx \frac{\lambda_g' a'}{\lambda_g a}\,(1.750)(1 - 0.58\alpha^2), \qquad \alpha \ll 1, \tag{1b}$$

$$\frac{Z_0'}{Z_0} \approx \frac{\lambda_g' a'}{\lambda_g a}\left(1 + \beta + \frac{\beta^2}{2}\right), \qquad \beta = (1-\alpha) \ll 1. \tag{1c}$$

$$\frac{X}{Z_0} = \frac{2a}{\lambda_g}\, X_{11}\, \frac{1 + \left(\dfrac{2a' X_0}{\lambda_g'}\right)^2}{1 + \left(\dfrac{2a'}{\lambda_g}\right)^2 X_0 X_{22}}, \tag{2a}$$

$$\frac{X}{Z_0} \approx \frac{2a}{\lambda_g}\, 2.330\,\alpha^2 (1 + 1.56\alpha^2)(1 + 6.75\alpha^2 Q), \qquad \alpha \ll 1, \tag{2b}$$

$$\frac{Z_0}{X} \approx \frac{\lambda_g}{2a}\, \frac{\beta^2 (1+\beta)\ln\dfrac{2}{\beta}}{1 - \dfrac{\beta}{2}}\left(1 - \frac{27}{8}\,\frac{Q + Q'}{1 + 8\ln\dfrac{2}{\beta}}\right), \qquad \beta \ll 1. \tag{2c}$$

$$\frac{l}{a} = \frac{\alpha X_0}{\pi}, \tag{3a}$$

$$\frac{l}{a} \approx 0.0084\,\alpha(1 - 1.5\alpha^2), \qquad \alpha \ll 1, \tag{3b}$$

$$\frac{l}{a} \approx 0, \qquad \beta \ll 1 \tag{3c}$$

"where" (p. 297)

$$X_{11} = \frac{A}{1 - \dfrac{1}{4A}\left[(A+1)^2 N_{11} + 2(A+1) C N_{12} + C^2 N_{22}\right]},$$

$$X_{22} = \frac{A'}{1 - \dfrac{1}{4A'}\left[(A'+1)^2 N_{22} + 2(A'+1) C N_{12} + C^2 N_{11}\right]},$$

$$X_0 = X_{22} - \frac{X_{12}^2}{X_{11}} \approx A' - \frac{C^2}{A},$$

"and"

$$A = \frac{(1+R_1)(1-R_2) + T^2}{(1-R_1)(1-R_2) - T^2}, \qquad R_1 = -\left(\frac{1-\alpha}{1+\alpha}\right)^{\alpha},$$

$$A' = \frac{(1-R_1)(1+R_2) + T^2}{(1-R_1)(1-R_2) - T^2}, \qquad R_2 = \left(\frac{1-\alpha}{1+\alpha}\right)^{1/\alpha},$$

$$C = \frac{2T}{(1-R_1)(1-R_2) - T^2}, \qquad T = \frac{4\alpha}{1-\alpha^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\frac12\left(\alpha + \frac{1}{\alpha}\right)},$$

$$\begin{aligned}
N_{11} = 2\left(\frac{a}{\lambda}\right)^2 \Big\{ 1 &+ \frac{16 R_1}{\pi(1-\alpha^2)}\left[E(\alpha) - \alpha'^2 F(\alpha)\right]\left[E(\alpha') - \alpha^2 F(\alpha')\right] - R_1^2 - \alpha^2 T^2 \Big\} \\
&+ \frac{12\alpha^4}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{4\alpha}\left[Q - \frac12\left(\frac{2a}{3\lambda}\right)^2\right] \\
&+ \frac{48\alpha^2}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\alpha + \frac{3}{\alpha}}\left[Q' - \frac12\left(\frac23\,\frac{a'}{\lambda}\right)^2\right],
\end{aligned}$$

$$\begin{aligned}
N_{22} = 2\left(\frac{a}{\lambda}\right)^2 \Big\{ \alpha^2 &+ \frac{16\alpha^2 R_2 E(\alpha)}{\pi(1-\alpha^2)}\left[F(\alpha') - E(\alpha')\right] - \alpha^2 R_2^2 - T^2 \Big\} \\
&+ \frac{48\alpha^2}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{3\alpha + \frac{1}{\alpha}}\left[Q - \frac12\left(\frac23\,\frac{a}{\lambda}\right)^2\right] \\
&+ \frac{12}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{4/\alpha}\left[Q' - \frac12\left(\frac23\,\frac{a'}{\lambda}\right)^2\right],
\end{aligned}$$

$$\begin{aligned}
N_{12} = 2\left(\frac{a}{\lambda}\right)^2 \Big\{ \alpha'^2 &- \frac{4E(\alpha)}{\pi}\left[E(\alpha') - \alpha^2 F(\alpha')\right] - R_1 + \alpha^2 R_2 \Big\}\, T \\
&+ \frac{24\alpha^3}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\frac12\left(7\alpha + \frac{1}{\alpha}\right)}\left[Q - \frac12\left(\frac{2a}{3\lambda}\right)^2\right] \\
&+ \frac{24\alpha}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\frac12\left(\alpha + \frac{7}{\alpha}\right)}\left[Q' - \frac12\left(\frac23\,\frac{a'}{\lambda}\right)^2\right].
\end{aligned}$$

$$\alpha = \frac{a'}{a} = 1 - \beta, \qquad \alpha' = \sqrt{1 - \alpha^2}.$$

$$Q = 1 - \sqrt{1 - \left(\frac{2a}{3\lambda}\right)^2}. \qquad Q' = 1 - \sqrt{1 - \left(\frac23\,\frac{a'}{\lambda}\right)^2}.$$

"The functions F(α) and E(α) are complete elliptical integrals of the first and second kinds, respectively." (p. 298;
the modulus convention is not stated. L1 check: with α as the modulus k, i.e. F(α) = K(m = α²) in the parameter
notation of scipy and Abramowitz-Stegun, Eqs. (1a)-(3a) reproduce Fig. 5.24-2 at a/λ = 0.5 (1.529 and 1.332 at
α = 0.5 and 0.7, book 1.52 and 1.33; l/a = 2.59·10⁻³ near α = 0.48, book 2.53·10⁻³); with α as the parameter m
they give 2.59 and 49.4.)

Reading notes: in `N_11` the curly brace closes after `− α²T²`, so the prefactor `2(a/λ)²` multiplies only the first
line; the second and third lines are separate terms. The same holds for `N_22` and `N_12` (in `N_12` the brace closes
before the factor `T`). The square brackets in the second and third lines are printed `[Q − ½(2a/3λ)²]` in `N_11`
and `N_12` but `[Q − ½((2/3)(a/λ))²]` in `N_22`; these are equal.

**Restrictions (p. 298).** "The equivalent circuit is valid in the range 0.5 < a/λ < 1.5 and 2a'/λ > 1. Equations
(1a), (2a), and (3a) have been obtained by the equivalent static method using one static mode incident in each guide
and are estimated to be in error by less than 1 per cent over most of the wavelength range; the error may rise to as
much as 5 per cent at the limit a/λ = 1.5. Equations (1b), (2b), and (3b), valid in the small-aperture range, are
approximations that differ from the more accurate Eqs. (1a), (2a), and (3a) by less than 5 per cent for 0.5 < (a/λ)
< 1 and α < 0.4; for α < 0.5 the difference is less than 10 per cent. Equations (1c), (2c), and (3c) are
approximations in the wide-aperture range and are in error by less than 6 per cent for 0.5 < (a/λ) < 1 and β < 0.3.
The constants X₁₁, X₁₂, and X₂₂ are the impedance matrix elements of the T network referred to a common reference
plane T and correspond to a characteristic impedance choice, Z₀'/Z₀ = λ_g'a/λ_g a'."

Numerical results (p. 298): Fig. 5.24-2 (`Z₀'λ_g a/Z₀λ_g'a'` and `l/a` vs α for a/λ = 0.5 and 1.4) and Fig. 5.24-3
(`Xλ_g/Z₀2a` and `Z₀2a/Xλ_g` vs α) "have been plotted from data obtained by use of the equivalent static method with
two modes incident in each guide and are somewhat more accurate than the analytic results given in Eqs. (1a), (2a),
and (3a)." (Fig. 5.24-2 right axis, l/a, is labelled "3.0 × 10⁻³".)

### b. Asymmetrical case (pp. 298, 301-302, PDF 312, 315-316)

"A junction of two rectangular guides of unequal widths but of equal heights (H₁₀-mode in each rectangular guide)."
Fig. 5.24-4: one side wall continuous, the step on the other side; same equivalent circuit as in a.

$$\frac{Z_0'}{Z_0} = \text{same as in Eq. (1a), Sec. 5.24a}, \tag{4a}$$

$$\frac{Z_0'}{Z_0} \approx \frac{\lambda_g' a'^3}{\lambda_g a^3}\, 5.046 (1 - 1.44\alpha^2), \qquad \alpha \ll 1, \tag{4b}$$

$$\frac{Z_0'}{Z_0} \approx \frac{\lambda_g'/a'}{\lambda_g/a}\left(1 - \beta - \frac{7\beta^2}{4}\right), \qquad \beta = (1-\alpha) \ll 1. \tag{4c}$$

$$\frac{X}{Z_0} = \text{same as Eq. (2a), Sec. 5.24a}, \tag{5a}$$

$$\frac{X}{Z_0} \approx \frac{2a}{\lambda_g}\, 5.764\,\alpha^4 (1 - 1.33\alpha^2)(1 + 41.3\alpha^4 Q), \qquad \alpha \ll 1 \tag{5b}$$

$$\frac{Z_0}{X} \approx \frac{\lambda_g}{2a}\, \frac{\beta^2}{1 - \dfrac{\beta}{2} - \beta^2\left(\ln\dfrac{2}{\beta} + \dfrac32\right)} \left\{ \frac74 \ln\frac{2}{\beta}\left[1 + \beta - \frac54 \beta^2 \ln\frac{2}{\beta}\right] \cdot \left[1 + \frac47 (1-\beta)^2\left(\frac{a}{\lambda}\right)^2\right] - \frac89\left[1 + 3\beta\ln\frac{2}{\beta}\right]\left[Q + Q'\right] \right\}. \tag{5c}$$

$$\frac{l}{a} = \text{same as in Eq. (3a), Sec. 5.24a}, \tag{6a}$$

$$\frac{l}{a} \approx 0.0259\,\alpha(1 - 0.6\alpha^2), \qquad \alpha \ll 1, \tag{6b}$$

$$\frac{l}{a} \approx 0, \qquad \alpha \approx 1. \tag{6c}$$

"The parameters X₁₁, X₁₂, X₂₂, X₀, A, A', C, α, α' and β are defined as in Sec. 5.24a but with"

$$R_1 = -\left(\frac{1 + 3\alpha^2}{1 - \alpha^2}\right)\left(\frac{1-\alpha}{1+\alpha}\right)^{2\alpha},$$

$$R_2 = \left(\frac{3 + \alpha^2}{1 - \alpha^2}\right)\left(\frac{1-\alpha}{1+\alpha}\right)^{2/\alpha},$$

$$T = \frac{16\alpha^2}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\alpha + (1/\alpha)},$$

$$\begin{aligned}
N_{11} = 2\left(\frac{a}{\lambda}\right)^2 &\left[1 + \frac{8\alpha(1-\alpha^2)}{1 + 3\alpha^2} R_1 \ln\frac{1-\alpha}{1+\alpha} + \frac{16\alpha^2}{1 + 3\alpha^2} R_1 - R_1^2 - \alpha^2 T^2\right] \\
&+ 2\left[\frac{32}{3}\,\frac{\alpha^4}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{3\alpha}\right]^2\left[Q - \frac12\left(\frac{a}{\lambda}\right)^2\right] \\
&+ 2\left[\frac{32\alpha^2}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\alpha + (2/\alpha)}\right]^2\left[Q' - \frac12\left(\frac{a'}{\lambda}\right)^2\right],
\end{aligned}$$

$$\begin{aligned}
N_{22} = 2\left(\frac{a}{\lambda}\right)^2 &\left[\alpha^2 - \frac{8\alpha(1-\alpha^2)}{3 + \alpha^2} R_2 \ln\frac{1-\alpha}{1+\alpha} + \frac{16\alpha^2}{1 + 3\alpha^2} R_2 - \alpha^2 R_2^2 - T^2\right] \\
&+ 2\left[\frac{32\alpha^2}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{2\alpha + (1/\alpha)}\right]^2\left[Q - \frac12\left(\frac{a}{\lambda}\right)^2\right] \\
&+ 2\left[\frac{32}{3}\,\frac{1}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{3/\alpha}\right]^2\left[Q' - \frac12\left(\frac{a'}{\lambda}\right)^2\right],
\end{aligned}$$

$$\begin{aligned}
N_{12} = 2\left(\frac{a}{\lambda}\right)^2 &\left[\frac{(1-\alpha^2)^2}{2\alpha}\ln\frac{1-\alpha}{1+\alpha} - R_1 - \alpha^2 R_2\right] T \\
&+ \frac23\left(\frac{1-\alpha}{1+\alpha}\right)^{5\alpha + (1/\alpha)}\left[\frac{32\alpha^3}{(1-\alpha^2)^2}\right]^2\left[Q - \frac12\left(\frac{a}{\lambda}\right)^2\right] \\
&+ \frac23\left(\frac{1-\alpha}{1+\alpha}\right)^{\alpha + (5/\alpha)}\left[\frac{32\alpha}{(1-\alpha^2)^2}\right]^2\left[Q' - \frac12\left(\frac{a'}{\lambda}\right)^2\right],
\end{aligned}$$

$$Q = 1 - \sqrt{1 - \left(\frac{a}{\lambda}\right)^2}, \qquad Q' = 1 - \sqrt{1 - \left(\frac{a'}{\lambda}\right)^2}.$$

Reading notes: (4b) is printed with `a'³` (exponent hard to read but consistent with `a³` in the denominator). In
(4c) the printed form is a fraction of fractions, `(λ_g'/a')/(λ_g/a)`. In (5c) the two factors of the first term
are on two printed lines, joined by a centred dot at the end of the first line; the curly brace encloses both lines
and the subtracted term. `N_22` mixes `3 + α²` (second term) and `1 + 3α²` (third term) as printed (L1a and L1b; L1 confirmed on a
500 dpi render; see the evaluation below for the suspicion that it should read `3 + α²`). `Q` and `Q'` here
differ from those of 5.24a (no factor 2/3).

**Restrictions (p. 302).** "The equivalent circuit is valid in the range 0.5 < a/λ < 1.0. Equations (4a), (5a), and
(6a) have been obtained by the equivalent static method employing one static mode incident in each guide. These
equations are estimated to be in error by less than 1 per cent over most of the wavelength range and by as high as 5
per cent at the limit a/λ = 1.0. Equations (4b), (5b), and (6b), valid in the small-aperture range, are
approximations that differ from Eqs. (4a), (5a), and (6a) by less than 5 per cent for α < 0.4 and 0.5 < a/λ < 1.
Equations (4c), (5c), and ((6c) are approximations in the large-aperture range and are in error by less than 10 per
cent for β < 0.2 and 0.5 < a/λ < 0.8. The parameters X₁₁, X₁₂, and X₂₂ are the impedance matrix elements of the T
network referred to the common reference plane T and correspond to a characteristic impedance choice,
Z₀'/Z₀ = λ_g'a/λ_g a'." (The doubled parenthesis "((6c)" is in the original.)

Numerical results (p. 302): Fig. 5.24-5 (`Z₀'λ_g a³/Z₀λ_g'a'³` and `l/a` vs α for various λ/a) and Fig. 5.24-6
(`Xλ_g/Z₀2a` vs α), from the equivalent static method with two modes incident in each guide, "somewhat more accurate
than the analytic results indicated in Eqs. (4a), (5a), and (6a)."

### L1 evaluation of the frequency-dependent formulas (1a)-(3a), (4a)-(6a) against the book's curves

A wave-1 implementer (WP4) found that the frequency terms of Eqs. (1a)-(3a) blow up for a/λ ≳ 0.7 and suspected
two misprints: `λ_g` for `λ_g'` in the denominator of (2a), and `1 + 3α²` for `3 + α²` in `N_22` of 5.24b. L1 checked
the print and evaluated the formulas (`benchmarks/literature/scripts/marcuvitz_checks.py`; book values read by L1
from Figs. 5.24-3 and 5.24-6 by pixel cuts on 400 dpi renders, about ±1 % of full scale).

*Print.* The denominator of (2a) reads `1 + (2a'/λ_g)² X₀X₂₂` with an unprimed `λ_g` (600 dpi render; the primes
on `a'` and on `λ_g'` in the numerator are clearly printed, the denominator `λ_g` has none). `N_22` of 5.24b reads
`16α²/(1 + 3α²) R₂`. Both transcriptions follow the print.

*Symmetric step, Eq. (2a), `Xλ_g/(Z₀2a)`, as printed and with `λ_g'` in the denominator:*

| a/λ | α | Fig. 5.24-3 | (2a) as printed | (2a) with λ_g′ |
| --- | --- | --- | --- | --- |
| 0.5 | 0.304 | 0.254 | 0.252 (−1 %) | 0.261 (+3 %) |
| 0.5 | 0.404 | 0.524 | 0.514 (−2 %) | 0.532 (+1 %) |
| 0.5 | 0.504 | 0.972 | 0.977 (+0 %) | 1.007 (+4 %) |
| 0.5 | 0.604 | 1.787 | 1.856 (+4 %) | 1.908 (+7 %) |
| 0.5 | 0.654 | 2.480 | 2.620 (+6 %) | 2.687 (+8 %) |
| 1.4 | 0.304 | 0.377 | 0.409 (+8 %) | 0.431 (+14 %) |
| 1.4 | 0.404 | 0.888 | 1.108 (+25 %) | 1.183 (+33 %) |
| 1.4 | 0.504 | 1.816 | 3.116 (+72 %) | 3.383 (+86 %) |
| 1.4 | 0.604 | 3.303 | 16.58 (+402 %) | 18.84 (+470 %) |

The impedance ratio (1a) behaves the same way: 1.529 at a/λ = 0.5, α = 0.5 (book 1.52) but 1.855 at a/λ = 1.4
(book about 1.49). The blow-up comes from `X₁₁` and `X₂₂`, whose variational correction
`1 − (1/4A)[(A+1)²N₁₁ + 2(A+1)CN₁₂ + C²N₂₂]` passes through zero, not from the outer factor of (2a): the first α at
which `X₁₁` or `X₂₂` changes sign is 0.955, 0.91, 0.82, 0.75 and 0.63 for a/λ = 0.5, 0.7, 1.0, 1.2, 1.4 (symmetric
step; inside the stated range 2a'/λ > 1 from a/λ ≈ 0.7 on). Replacing `λ_g` by `λ_g'` in (2a) does not remove the
pole and moves the values further from the book's curves, so the print is not shown to be wrong there. Other simple
candidates (factor 2 of the first `N` term, (a/λ)² → (a/λ_g)², sign of the `Q` terms) do not restore agreement at
both a/λ either. Near α → 1 the wide-aperture form (2c) follows the book's descending curves (L1: 0.058 vs 0.059 at
α = 0.85, 0.0275 vs 0.030 at 0.90, a/λ = 1.4), while (2a) is negative there.

*Asymmetric step, Eq. (5a) with `N_22` as printed (`1 + 3α²`) and with `3 + α²`:*

| λ/a | α | Fig. 5.24-6 | printed | 3 + α² |
| --- | --- | --- | --- | --- |
| 1.05 | 0.404 | 0.207 | 0.192 (−7 %) | 0.186 (−10 %) |
| 1.05 | 0.504 | 0.613 | 0.558 (−9 %) | 0.535 (−13 %) |
| 1.05 | 0.654 | 2.283 | 1.905 (−17 %) | 1.822 (−20 %) |
| 1.05 | 0.704 | 3.254 | 2.714 (−17 %) | 2.568 (−21 %) |
| 2.00 | 0.404 | 0.138 | 0.132 (−5 %) | 0.131 (−5 %) |
| 2.00 | 0.654 | 1.130 | 1.100 (−3 %) | 1.084 (−4 %) |
| 2.00 | 0.704 | 1.707 | 1.649 (−3 %) | 1.622 (−5 %) |

The printed `1 + 3α²` agrees better with the book's curve than `3 + α²` at every point; the misprint is not
supported. In the asymmetric case no pole occurs for a/λ ≤ 0.9; it appears at α = 0.595 for a/λ = 1.0, the
limit of the stated range.

*Conclusion.* The transcription is right. As printed, the one-mode formulas (1a)-(3a) agree with the book's
two-mode curves within about 6 % at a/λ = 0.5 but are not usable at large a/λ and α (error +25 % at a/λ = 1.4,
α = 0.4, a pole at α ≈ 0.63), contrary to the stated "less than 1 per cent over most of the wavelength range". The
cause (an unidentified misprint in the `N` terms, or the stated accuracy being optimistic) is not settled by the
book. Use Figs. 5.24-2/3/5/6 (two-mode results) or (1c)/(2c) near α → 1 as references, not (1a)-(3a), for
a/λ ≳ 0.7; wrmodes' mode-matching steps (WP4) should be validated against mode matching or COMSOL, not against these
frequency corrections.

---

## 5.26 Change in height of rectangular guide (E-plane step)

### a. Symmetrical case (pp. 307-308, PDF 321-322)

"An axially symmetrical junction of two rectangular guides of equal widths and unequal heights (H₁₀-mode in
rectangular guides)." Fig. 5.26-1: heights `b` and `b'`; equivalent circuit: shunt capacitive susceptance `jB`
at the junction plane `T` between lines `Y_0, λ_g` and `Y_0', λ_g`.

"Equivalent-circuit Parameters.—At the terminal plane T"

$$\frac{Y_0}{Y_0'} = \frac{b'}{b} = \alpha = 1 - \delta, \tag{1}$$

$$\frac{B}{Y_0} = \frac{2b}{\lambda_g}\left[\ln\left(\frac{1-\alpha^2}{4\alpha}\right)\left(\frac{1+\alpha}{1-\alpha}\right)^{\frac12\left(\alpha + \frac{1}{\alpha}\right)} + 2\,\frac{A + A' + 2C}{AA' - C^2} + \left(\frac{b}{4\lambda_g}\right)^2\left(\frac{1-\alpha}{1+\alpha}\right)^{4\alpha}\left(\frac{5\alpha^2 - 1}{1 - \alpha^2} + \frac43\,\frac{\alpha^2 C}{A}\right)^2\right], \tag{2a}$$

$$\frac{B}{Y_0} \approx \frac{2b}{\lambda_g}\left[\ln\left(\frac{1-\alpha^2}{4\alpha}\right)\left(\frac{1+\alpha}{1-\alpha}\right)^{\frac12\left(\alpha + \frac{1}{\alpha}\right)} + \frac{2}{A}\right], \tag{2b}$$

$$\frac{B}{Y_0} \approx \frac{2b}{\lambda_g}\left[\ln\frac{e}{4\alpha} + \frac{\alpha^2}{3} + \frac12\left(\frac{b}{\lambda_g}\right)^2 (1 - \alpha^2)^4\right], \qquad \alpha \ll 1, \tag{2c}$$

$$\frac{B}{Y_0} \approx \frac{2b}{\lambda_g}\left(\frac{\delta}{2}\right)^2\left[\frac{2\ln\dfrac{2}{\delta}}{1 - \delta} + 1 + \frac{17}{16}\left(\frac{b}{\lambda_g}\right)^2\right], \qquad \delta \ll 1, \tag{2d}$$

"where"

$$A = \left(\frac{1+\alpha}{1-\alpha}\right)^{2\alpha}\,\frac{1 + \sqrt{1 - \left(\dfrac{b}{\lambda_g}\right)^2}}{1 - \sqrt{1 - \left(\dfrac{b}{\lambda_g}\right)^2}} - \frac{1 + 3\alpha^2}{1 - \alpha^2},$$

$$A' = \left(\frac{1+\alpha}{1-\alpha}\right)^{2/\alpha}\,\frac{1 + \sqrt{1 - \left(\dfrac{b'}{\lambda_g}\right)^2}}{1 - \sqrt{1 - \left(\dfrac{b'}{\lambda_g}\right)^2}} + \frac{3 + \alpha^2}{1 - \alpha^2},$$

$$C = \left(\frac{4\alpha}{1 - \alpha^2}\right)^2, \qquad e = 2.718.$$

Reading note: in (2a) and (2b) the logarithm's argument is printed as the product
`((1−α²)/4α)·((1+α)/(1−α))^{½(α+1/α)}` without enclosing brackets; Marcuvitz's equivalent coaxial formula (5.27a,
eq. 2a) prints `2 ln(...)(...)^{...}` the same way. L1b reads the power factor as part of the argument of `ln`,
which makes the bracket the familiar `ln[(1−α²)/4α ((1+α)/(1−α))^{(α+1/α)/2}]`. The scan does not settle this
typographically. L1a read it the same way, and two checks settle it (L1): (i) for α → 0 the power factor tends to
e, so the bracket starts with ln(e/4α), the first term of the asymptotic form (2c); with the factor outside the
logarithm this limit fails; (ii) in the static limit b/λ_g → 0, (2a) gives `Bλ_g/(Y₀b)` = 3.839, 2.473, 1.696,
0.785 at b'/b = 0.1, 0.2, 0.3, 0.5, as Fig. 5.26-3 (about 3.8, 2.45, 1.7, 0.78), and 5.19 and 3.64 at b'/b = 0.1
and 0.2 for b/λ_g = 0.8 (book about 5.4 and 3.6); with the factor outside the logarithm the values are 4.99, 1.05,
−1.70, −7.75 (negative capacitive susceptance).

**Restrictions (p. 308).** "The equivalent circuit is valid in the range b/λ_g < 1. Equation (2a) has been obtained
by the equivalent static method employing a static aperture field due to the incidence of the two lowest modes and
is correct to within 1 per cent in the range b/λ_g < 1. The approximate Eq. (2b) is correct to within 3 per cent for
(b/λ_g) < 0.7; for (b/λ_g) ≥ 0.7 it is still correct to within 3 per cent if α < 0.7. Equation (2c) is an asymptotic
expansion of Eq. (2a) correct to within 5 per cent if α < 0.6 and b/λ_g < 0.5 and to within 2 per cent if α < 0.4
and b/λ_g < 0.4. Equation (2d) is an asymptotic expansion of Eq. (2a) correct to within 5 per cent when δ < 0.5 and
b/λ_g < 0.5 and to within 3 per cent when δ < 0.4 and b/λ_g < 0.4."

Numerical results: Fig. 5.26-3, `Bλ_g/Y₀b` from Eq. (2a) vs b'/b with b/λ_g = 0, 0.4, 0.6, 0.7, 0.8, 0.9.
Experimental results: "The above results have been verified experimentally at least for the cases b/λ_g = 0.23 and
α > 0.15 to within a few per cent."

### b. Asymmetrical case (pp. 308, 310, PDF 322, 324)

"An axially asymmetrical junction of two rectangular guides of equal widths but unequal heights (H₁₀-modes in
rectangular guides)." Fig. 5.26-2: one broad wall continuous.

"Equivalent-circuit Parameters.—Same as Sec. 5.26a except λ_g is replaced by λ_g/2.
Restrictions.—Same as Sec. 5.26a except λ_g is replaced by λ_g/2.
Numerical Results.—If the λ_g in Fig. 5.26-3 is replaced by λ_g/2, one obtains a plot of Bλ_g/2Y₀b as a function of
b'/b with 2b/λ_g as a parameter."

---

## 5.28 E-plane corners (pp. 312-318, PDF 326-332)

### a. Right-angle bends

"A right-angle E-plane junction of two rectangular guides of equal dimensions (H₁₀-mode in rectangular guides)."
Fig. 5.28-1: side view with guide dimension `b` in the bend plane; the reference planes `T` of both arms lie at
distance `d` from the outer wall of the other arm (checked on a 600 dpi render); the alternative planes `T'` lie at
distance `b` from the outer walls, i.e. through the inner corner. Equivalent circuit at `T`: a shunt susceptance `−jB` between two lines `Y_0` (symbol drawn as an
inductor).

"Equivalent-circuit Parameters.—At reference planes T the equivalent circuit is pure shunt. The relative shunt
susceptance and location of the terminal planes are given by"

$$\frac{B}{Y_0} = \frac{\left(\dfrac{B_a}{Y_0}\right)^2 + 1}{\dfrac{B_b}{Y_0}} - 2\,\frac{B_a}{Y_0}, \tag{1}$$

$$\frac{2\pi(b - d)}{\lambda_g} = \cot^{-1}\left(2\,\frac{B_b}{Y_0} - \frac{B_a}{Y_0}\right), \tag{2}$$

"where"

$$\frac{B_a}{Y_0} = \frac{2b}{\lambda_g}\left\{ -\frac{\cot \pi x}{x} + \frac{1}{\pi x^2} + \frac{\pi}{6} - \ln 2 - \left[\frac{A_0 e^{-\frac{\pi}{2}} + (A_1 - A_2) e^{-\pi} + (1 + 5e^{-\pi})\dfrac{A_0^2}{16}}{1 - (1 + 5e^{-\pi})\left(\dfrac{A_1 - A_2}{4}\right)}\right]\right\}, \tag{3a}$$

$$\frac{B_a}{Y_0} \approx \frac{2b}{\lambda_g}\left[0.878 + 0.498\left(\frac{2b}{\lambda_g}\right)^2\right], \qquad \frac{2b}{\lambda_g} \ll 1, \tag{3b}$$

$$\frac{2B_b}{Y_0} - \frac{B_a}{Y_0} = \frac{\lambda_g}{2\pi b}\left\{1 + \pi x \cot \pi x - \pi x^2\left[5\ln 2 - \frac{7\pi}{6} - 8\sum_{n=1}^{\infty}\frac{1}{n(e^{2\pi n} - 1)}\right] + \pi x^2\left[\frac{A_0' e^{-\frac{\pi}{2}} - (A_1 + A_2)e^{-\pi} + (1 - 3e^{-\pi})\dfrac{A_0'^2}{16}}{1 + (1 - 3e^{-\pi})\dfrac{A_1 + A_2}{4}}\right]\right\}, \tag{4a}$$

$$\frac{B_b}{Y_0} \approx \frac{\lambda_g}{2\pi b}\left[1 - 0.114\left(\frac{2b}{\lambda_g}\right)^2\right], \qquad \frac{2b}{\lambda_g} \ll 1, \tag{4b}$$

"and"

$$x = \frac{2b}{\lambda_g},$$

$$A_0 = \frac{4}{\pi}\,\frac{x^2}{1 - x^2}, \qquad A_0' = A_0 - \frac{8}{\sinh \pi},$$

$$A_1 = \frac{1}{\pi}\,\frac{x^2}{1 - 0.5x^2}, \qquad A_2 = 4\left[\frac{1}{\sqrt{1 - x^2}\,\left(1 - e^{-2\pi\sqrt{1 - x^2}}\right)} - \frac{1}{1 - e^{-2\pi}}\right].$$

"An alternative circuit at the reference planes T' is shown in Fig. 5.28-2." (Fig. 5.28-2: at the planes `T'`, a
π network: a capacitive `jB_a` across each arm and an inductive `−jB_b` in series between them, the same network as
Fig. 5.28-5. In the small drawing one wire crosses the line through the first `jB_a` without a junction dot; read
literally it would be a T network of two series `jB_a` and a shunt `−jB_b`. L1 settled it as π: only the π network
turns into the pure shunt of Eqs. (1)-(2) when the planes are moved (for B_a = 0.9, B_b = 1.3: π gives the shunt
0.4077 at 2π(b − d)/λ_g = 0.5317, as Eqs. (1)-(2); the T network gives 0.6827), and the static B_a, B_b of (3b),
(4b) are those of Sec. 5.28b at θ = 90° (0.878, λ_g/2πb), whose Fig. 5.28-5 shows the π network unambiguously.)

Reading notes: in (4a) the term `A_0'²` is printed with the prime and the square stacked (`A₀'²`). The sign in front
of `B` in (1) is "− 2 B_a/Y_0" here but "+ 2 B_a/Y_0" in the H-plane formula 5.29a (1); both as printed.

**Restrictions (pp. 313, 316).** "The equivalent circuits are valid in the range 2b/λ_g < 1. The circuit parameters
obtained by the equivalent static method, employing two lowest modes incident in each guide, are accurate to within
1 per cent in the above range. Equations (3b) and (4b) are asymptotic approximations to B_a and B_b that agree with
the values from Eqs. (3a) and (4a) to within 8 per cent for 2b/λ_g < 0.6."

Numerical results: Fig. 5.28-3 (`d/b` and `Bλ_g/Y₀2b` vs `2b/λ_g`); Fig. 5.28-4 (`B_aλ_g/Y₀2b` and `B_b2πb/Y₀λ_g`
vs `2b/λ_g`). Experimental results: "A few measured values for B_a and B_b are indicated by the circled points in
Fig. 5.28-4. These data were taken at wavelengths in the vicinity of 3 cm."

L1b check (Python, own code from the transcription above): with B_b from (4a) as B_b = [(4a) + B_a]/2,
`d/b` = 0.499, 0.467, 0.392, 0.241, 0.145 and `Bλ_g/(Y₀2b)` = 1.391, 1.570, 2.066, 3.824, 7.135 at 2b/λ_g = 0.05,
0.3, 0.5, 0.7, 0.8; `B_aλ_g/(Y₀2b)` = 1.036 and `B_b2πb/(Y₀λ_g)` = 0.963 at 0.5. These agree with the curves of
Figs. 5.28-3 and 5.28-4 as read by eye (e.g. d/b ≈ 0.39 and Bλ_g/Y₀2b ≈ 2.07 at 0.5; d/b ≈ 0.24 at 0.7).
cot⁻¹ in (2) is taken in (0, π).

### b. Arbitrary angle bends (pp. 316-318, PDF 330-332)

"A symmetrical, arbitrary angle, E-plane junction of two rectangular guides of equal dimensions (H₁₀-mode in
rectangular guides)." Fig. 5.28-5, side view: guide height `b`; θ is the bend (deflection) angle, drawn between the
plane `T₁` and the plane `T₂`; both reference planes `T₁`, `T₂` pass through the inner corner (perpendicular to
their arms). Equivalent circuit (read from the drawing): a π network, a capacitive element `jB_a` connected across
the two conductors of each arm (shunt) and an inductive element `−jB_b` in series in one conductor between them
(confirmed by L1 on a 600 dpi render; L1a had described it as series `jB_a` and shunt `−jB_b`, which does not match
the drawing nor the limit θ → 0, where B_b = (λ_g/2πb) cot(θ/2) → ∞ must become a through connection).

"Equivalent-circuit Parameters.—At the reference planes T"

$$\frac{B_a}{Y_0} = \frac{2b}{\lambda_g}\left\{\Psi\left[-\frac12\left(1 - \frac{\theta}{\pi}\right)\right] - \Psi\left[-\frac12\right]\right\}, \tag{1}$$

$$\frac{B_b}{Y_0} = \frac{\lambda_g}{2\pi b}\cot\frac{\theta}{2}, \tag{2}$$

"where Ψ(x) is the derivative of the logarithm of x!"

Reading note: "x!" is the factorial, so Ψ(x) = d ln(x!)/dx = ψ(x + 1) with ψ the digamma function (L1a and L1b
agree; L1 re-computed the check below). L1b check:
ψ(1 − ½(1 − θ/π)) − ψ(½) = 0.361, 0.645, 0.878, 1.073 at θ = 30°, 60°, 90°, 120°, which reproduces the solid
`B_aλ_g/Y₀2b` line of Fig. 5.28-6 (0.65 at 60°, 0.88 at 90°, 1.07 at 120°), while the plain digamma ψ(x) gives
2.88 at 90° and does not. Eq. (2): B_b b/(Y₀λ_g) = 0.276, 0.159, 0.092 at 60°, 90°, 120°, as in Fig. 5.28-6.

**Restrictions (p. 316).** "The equivalent circuit is applicable in the wavelength range 2b/λ_g < 1. The circuit
parameters have been obtained by a simple equivalent static method and are rigorous only in the static limit
b/λ_g ≈ 0."

Numerical results: Fig. 5.28-6, `B_aλ_g/Y₀2b` and `B_b b/Y₀λ_g` vs θ (0-120°); "The solid curves represent the
static values of the parameters as given in Eqs. (1) and (2)." Dashed curves for 2b/λ_g = 0.40, 0.45, 0.51.
Experimental results (pp. 316-318): "Measured values of the circuit parameters are indicated by the circled points in
Fig. 5.28-6. These data apply to rectangular guides of dimensions a = 0.900 in. and b = 0.400 in. and were measured
for several values of b/λ_g. The measured values of B_b b/Y₀λ_g are within a few per cent of the static values given
in Eq. (2) for the bend angles θ = 60, 75, and 90° if 2b/λ_g < 0.51."

---

## 5.29 H-plane corners (pp. 318-322, PDF 332-336)

### a. Right-angle bends

"An H-plane right-angle junction of two rectangular guides with equal dimensions (H₁₀-mode in rectangular guides)."
Fig. 5.29-1 (top view, width `a`): planes `T` of both arms at distance `d` from the outer wall of the other arm,
alternative planes `T'` at distance `a`, through the inner corner (600 dpi render). Equivalent circuit at `T`: shunt `jB` (drawn as a capacitor) between two lines `Y_0`.

"Equivalent-circuit Parameters.—At the terminal planes T the equivalent circuit is pure shunt. The relative shunt
susceptance and the location of the terminal planes are given by"

$$\frac{B}{Y_0} = \frac{1 + \left(\dfrac{B_a}{Y_0}\right)^2}{\dfrac{B_b}{Y_0}} + 2\,\frac{B_a}{Y_0}, \tag{1}$$

$$\frac{2\pi(a - d)}{\lambda_g} = -\cot^{-1}\frac{B_a + 2B_b}{Y_0}, \tag{2}$$

"where"

$$\frac{B_a}{Y_0} = \frac{\lambda_g}{2a}\left[A_1 - x\cot\pi x + \frac{1}{\pi}\left(\frac{1 + x^2}{1 - x^2}\right) + A_2\right](1 + A_3), \tag{3a}$$

$$\frac{B_a}{Y_0} \approx 0.42 + \frac{0.64}{1 - x}, \qquad x \approx 1, \tag{3b}$$

$$\frac{B_a + 2B_b}{Y_0} = \frac{\lambda_g}{2a}\left[-A_1' - x\cot\pi x - \frac{1}{\pi}\left(\frac{1 + x^2}{1 - x^2}\right) + A_2'\right](1 + A_3'), \tag{4a}$$

$$\frac{B_a + 2B_b}{Y_0} \approx -2.04 + 1.63x, \qquad x \approx 1, \tag{4b}$$

"and"

$$x = \frac{2a}{\lambda_g}, \qquad \lambda_g = \frac{\lambda}{\sqrt{1 - \left(\dfrac{\lambda}{2a}\right)^2}}$$

$$A_1 = \frac{2}{1 - e^{-2\pi}} - \frac{4(1 + 63e^{-2\pi})}{\left[2(1 + 5e^{-\pi})(1 + 63e^{-2\pi}) - \left(\frac{64}{3}\right)^2 e^{-3\pi}\right]} = 0.3366,$$

$$A_1' = -\frac{2}{1 - e^{-2\pi}} + \frac{4(1 - 33e^{-2\pi})}{\left[2(1 - 3e^{-\pi})(1 - 33e^{-2\pi}) - \left(\frac{32}{3}\right)^2 e^{-3\pi}\right]} = 0.3071,$$

$$A_2 = \frac{\left[0.1801\left(\dfrac{1 + x^2}{4 - x^2}\right) + 0.2021\right]^2}{\dfrac{\sqrt{3 - x^2}}{1 - e^{-2\pi\sqrt{3 - x^2}}} - 0.1858 - 0.1592\left(\dfrac{1 + x^2}{7 - x^2}\right)},$$

$$A_2' = \frac{\left[0.1801\left(\dfrac{1 + x^2}{4 - x^2}\right) + 0.1669\right]^2}{\dfrac{\sqrt{3 - x^2}}{1 - e^{-2\pi\sqrt{3 - x^2}}} + 0.1432 + 0.1592\left(\dfrac{1 + x^2}{7 - x^2}\right)},$$

$$A_3 = 0.01 - 0.0095x \qquad \text{for } 0 \le x \le 1,$$

$$A_3 = -\frac{0.00416}{1.5 - x} + 0.0040 + 0.0019x + 0.0020x^2 \qquad \text{for } 1 \le x \le 1.436,$$

$$A_3' = -\frac{0.00146}{1.26 - x} - 0.0016 + 0.0034x - 0.0026x^2 \qquad \text{for } 0 \le x \le 1.2,$$

$$A_3' = \frac{0.00136}{x - 1.26} + 0.0076 - 0.0066x \qquad \text{for } 1.3 \le x \le 1.5.$$

"An alternative equivalent circuit at reference planes T' is in Fig. 5.29-2." (Same π topology as Fig. 5.28-2, with
the same crossing in the drawing; L1 checked that the π network with capacitive `jB_a`, `jB_b` reproduces Eqs.
(1)-(2) of this section, B = 3.1923 at |2π(a − d)/λ_g| = 0.2783 for B_a = 0.9, B_b = 1.3, and the T network does
not. B_b is negative over the plotted range, as Fig. 5.29-4 shows Y₀λ_g/B_b2a < 0.)

L1b arithmetic: the printed closed forms give A₁ = 0.336574 and A₁' = 0.307152, matching the printed 0.3366 and
0.3071 (this checks the transcription of 64/3, 32/3, 63, 33, 5 and 3).

**Restrictions (p. 319).** "The equivalent circuit is valid in the wavelength range a < λ < 2a. The circuit
parameters have been derived by the equivalent static method with two static modes incident in each guide. The
error in the susceptance values is estimated to be of the order of magnitude of 1 per cent in the range
0 ≦ 2a/λ_g ≦ 1.44, excluding the range 1.2 ≦ 2a/λ_g ≦ 1.3 in which the variational correction A₃' diverges.
Equations (3b) and (4b) are approximations to Eqs. (3a) and (4a), respectively, and agree with these to within 10 per
cent in the resonant range 0.7 ≦ 2a/λ_g ≦ 1.2. The quantity A₂ is less than 10 per cent of B_a/Y₀ for
2a/λ_g < 1.3; A₂' is less than 10 per cent of (B_a + 2B_b)/Y₀ for 2a/λ_g < 1.0."

Numerical results: Fig. 5.29-3 (`B2a/Y₀λ_g` and `d/a` vs `2a/λ_g`, 0-1.5); Fig. 5.29-4 (`Y₀λ_g/B_b2a` and
`(B_a + 2B_b)2a/Y₀λ_g`). Experimental results: "a number of experimental data are indicated by circled points in Fig.
5.29-4. These data were measured in rectangular guides of dimensions a = 0.900 in. and b = 0.400 in."

L1b check: at x = 0.01 and 0.5 the transcribed formulas give `B2a/(Y₀λ_g)` = 0.538, 0.570, `d/a` = 0.653, 0.647
(cot⁻¹ taken on the branch that gives 0 < d < a), `(B_a + 2B_b)2a/(Y₀λ_g)` = −0.918, −0.808 and
`Y₀λ_g/(B_b 2a)` = −1.541, −1.158; Fig. 5.29-3 starts at about 0.55 and 0.654, Fig. 5.29-4 at about −0.92
(lower curve) and −1.55 (upper curve, right axis). Consistent.

### b. Arbitrary angle H-plane corners (pp. 319, 322, PDF 333, 336)

"A symmetrical, arbitrary angle, H-plane junction of two rectangular guides of identical cross-sectional dimensions
(H₁₀-mode in rectangular guides)." Fig. 5.29-5, top view: width `a`, bend angle θ between the two reference planes
`T`, which pass through the inner corner. Equivalent circuit: a T network with series reactances `jX_b` in each arm
and a shunt `−jX_a` (capacitor symbol), referred to `Z_0`.

No formulas are given; only measured data (p. 322): "The equivalent circuit of Fig. 5.29-5 is valid in the
wavelength range a < λ < 2a. The circuit parameters Z₀/X_a and X_b/Z₀, relative to the reference planes T, were
measured in rectangular waveguide of dimensions a = 0.900 in. and b = 0.400 in. as a function of the bend angle θ at
various wavelengths. The resulting data are:"

| θ | λ = 3.0 cm (λ_g = 1.560″) Z₀/X_a | X_b/Z₀ | λ = 3.2 cm (λ_g = 1.762″) Z₀/X_a | X_b/Z₀ | λ = 3.4 cm (λ_g = 1.999″) Z₀/X_a | X_b/Z₀ |
| --- | --- | --- | --- | --- | --- | --- |
| 30° | 0.849 | 0.493 | 0.782 | 0.420 | 0.718 | 0.365 |
| 45° | | | | | 0.995 | 0.567 |
| 50° | 1.102 | 0.943 | | | 1.068 | 0.650 |
| 60° | 1.088 | 1.27 | 1.158 | 0.998 | 1.176 | 0.817 |
| 75° | 0.864 | 2.14 | | | | |
| 90° | 0.324 | 6.46 | 0.739 | 2.75 | 1.055 | 1.76 |
| 120° | | | −2.17 | −4.58 | 0.044 | 47.9 |

(Empty cells are empty in the book. Values checked digit by digit on a 450 dpi render.)

---

## 5.34 Circular bends, E-plane (pp. 333-334, PDF 347-348)

"An E-plane junction of a rectangular guide and a uniform circular bend of rectangular cross section (H₁₀-mode in
rectangular guide and in circular bend)." Fig. 5.34-1: height `b` in the bend plane, mean radius `R`; equivalent
circuit: series capacitive reactance `−jX` at plane `T`, between a line `Z₀, λ_g` and a line `Z₀', λ_gφ`.

"Equivalent-circuit Parameters.—At the reference plane T"

$$\frac{Z_0'}{Z_0} = 1 + \frac{1}{12}\left(\frac{b}{R}\right)^2\left[\frac12 - \frac15\left(\frac{2\pi b}{\lambda_g}\right)^2\right], \tag{1}$$

$$\frac{X}{Z_0} = \frac{32}{\pi^7}\left(\frac{2\pi b}{\lambda_g}\right)^3\left(\frac{b}{R}\right)^2 \sum_{n=1,3,\cdots}^{\infty} \frac{1}{n^7}\sqrt{1 - \left(\frac{2b}{n\lambda_g}\right)^2}. \tag{2}$$

"where"

$$\lambda_g = \frac{\lambda}{\sqrt{1 - \left(\dfrac{\lambda}{2a}\right)^2}},$$

$$\lambda_{g\phi} = \lambda_g\left\{1 - \frac{1}{12}\left(\frac{b}{R}\right)^2\left[-\frac12 + \frac15\left(\frac{2\pi b}{\lambda_g}\right)^2 \cdots\right]\right\}.$$

"The circular bend is a uniform angular guide with propagation wavelength λ_gφ, the latter being measured along the
arc of mean radius R. The electrical length corresponding to an angular distance φ or to a (mean) arc distance s is"

$$\frac{2\pi s}{\lambda_{g\phi}} = m\phi, \qquad \text{where } m = \frac{2\pi R}{\lambda_{g\phi}}.$$

**Restrictions (p. 334).** "The equivalent circuit is applicable in the wavelength range 2b/λ_g < 1. Equations (1)
and (2) have been derived by a variational method employing the form of the lowest mode in the angular guide for the
electric-field distribution in the junction plane. The circuit parameters have been evaluated to order (b/R)², but no
estimate of the range of accuracy is available."

Reading note: the summation in (2) is printed `n = 1, 3, ·` (odd n); the square root extends over
`1 − (2b/nλ_g)²`.

## 5.35 Circular bends, H-plane (pp. 334-335, PDF 348-349)

"An H-plane junction of a rectangular guide and a uniform circular bend, i.e., angular guide, of rectangular cross
section (H₁₀-mode in rectangular and angular guides)." Fig. 5.35-1: width `a` in the bend plane, mean radius `R`;
equivalent circuit: shunt inductive susceptance `−jB` at plane `T`, between a line `Y₀, λ_g` and a line
`Y₀', λ_gφ`.

"Equivalent-circuit Parameters.—At the reference plane T"

$$\frac{Y_0'}{Y_0} = 1 + \frac{\pi^2}{120}\left(\frac{a}{R}\right)^2\left[1 + \frac{\dfrac{15}{\pi^4}}{\left(\dfrac{2a}{\lambda}\right)^2 - 1} - \left(2 - \frac{5}{\pi^2} - \frac{60}{\pi^4}\right)\left(\frac{2a}{\lambda}\right)^2 + \left(1 + \frac{15}{6\pi^2} - \frac{105}{\pi^4}\right)\left(\frac{2a}{\lambda}\right)^4 \cdots\right], \tag{1}$$

$$\frac{B}{Y_0} = \frac{\lambda_g}{a}\,\frac{8}{\pi^4}\left(\frac{a}{R}\right)^2 \sum_{n=2,4,\cdots}^{\infty} \left\{\frac{n^3}{(n^2 - 1)^4}\left[1 - \frac{\left(\dfrac{4a}{\lambda}\right)^2}{n^2 - 1}\right]^2 \sqrt{1 - \left(\frac{2a}{n\lambda}\right)^2}\right\}, \tag{2}$$

"where"

$$\lambda_g = \frac{\lambda}{\sqrt{1 - \left(\dfrac{\lambda}{2a}\right)^2}}$$

$$\frac{1}{\lambda_{g\phi}^2} = \frac{1}{\lambda_g^2} + \frac{1}{24R^2}\left[1 - \left(\frac{12 + \pi^2}{2\pi^2}\right)\left(\frac{2a}{\lambda}\right)^2 + \left(\frac{15 - \pi^2}{2\pi^2}\right)\left(\frac{2a}{\lambda}\right)^4 \cdots\right].$$

"The propagation wavelength λ_gφ in the uniform angular guide (circular bend) is measured along the arc of mean radius
R. The electrical length corresponding to the angular distance φ or to a mean arc distance S is
2πs/λ_gφ = mφ, where m = 2πR/λ_gφ. With this interpretation the angular guide may be treated as a uniform line with
characteristic admittance Y₀' and wavelength λ_gφ."

**Restrictions (p. 335).** "The equivalent circuit is applicable in the wavelength range a < λ < 2a. The circuit
parameters have been derived by a variational method employing a sine function for the distribution of electric
field in the junction plane. The results have been evaluated to order (a/R)² but no estimate of the range of accuracy
is available."

Reading note: the book writes both "S" and "s" for the arc distance in the same sentence. The sum in (2) is printed
`n = 2, 4,` (even n). In (1) the term `15/π⁴` is the numerator over `(2a/λ)² − 1`, and the trailing dots after the
`(2a/λ)⁴` term are printed (truncated series). In (2) the bracket squared is `[1 − (4a/λ)²/(n² − 1)]²` with
`(4a/λ)²`, not `(2a/λ)²`, clearly printed (L1a, confirmed by L1b's transcription).

---

## Summary of stated parameter ranges and accuracies

| Section | Structure | Range of validity | Stated accuracy |
| --- | --- | --- | --- |
| 5.24a | symmetric H-plane step | 0.5 < a/λ < 1.5, 2a'/λ > 1 | (1a)-(3a) < 1 % over most of the range, up to 5 % at a/λ = 1.5; (1b)-(3b) within 5 % of (a) for 0.5 < a/λ < 1, α < 0.4 (10 % for α < 0.5); (1c)-(3c) < 6 % for 0.5 < a/λ < 1, β < 0.3 |
| 5.24b | asymmetric H-plane step | 0.5 < a/λ < 1.0 | (4a)-(6a) < 1 % over most of the range, up to 5 % at a/λ = 1.0; (b) within 5 % of (a) for α < 0.4, 0.5 < a/λ < 1; (c) < 10 % for β < 0.2, 0.5 < a/λ < 0.8 |
| 5.26a | symmetric E-plane step | b/λ_g < 1 | (2a) within 1 %; (2b) within 3 % for b/λ_g < 0.7 (and for b/λ_g ≥ 0.7 if α < 0.7); (2c) 5 % (α < 0.6, b/λ_g < 0.5), 2 % (α < 0.4, b/λ_g < 0.4); (2d) 5 % (δ < 0.5, b/λ_g < 0.5), 3 % (δ < 0.4, b/λ_g < 0.4); measured agreement "within a few per cent" for b/λ_g = 0.23, α > 0.15 |
| 5.26b | asymmetric E-plane step | as 5.26a with λ_g → λ_g/2 | as 5.26a with λ_g → λ_g/2 |
| 5.28a | right-angle E-plane corner | 2b/λ_g < 1 | 1 %; (3b), (4b) within 8 % of (3a), (4a) for 2b/λ_g < 0.6 |
| 5.28b | arbitrary-angle E-plane corner | 2b/λ_g < 1 | rigorous only in the static limit b/λ_g ≈ 0; measured B_b within a few % of Eq. (2) for θ = 60, 75, 90° if 2b/λ_g < 0.51 |
| 5.29a | right-angle H-plane corner | a < λ < 2a | about 1 % for 0 ≤ 2a/λ_g ≤ 1.44 except 1.2 ≤ 2a/λ_g ≤ 1.3; (3b), (4b) within 10 % for 0.7 ≤ 2a/λ_g ≤ 1.2 |
| 5.29b | arbitrary-angle H-plane corner | a < λ < 2a | measured data only (table above) |
| 5.34 | E-plane circular bend junction | 2b/λ_g < 1 | to order (b/R)², no accuracy estimate |
| 5.35 | H-plane circular bend junction | a < λ < 2a | to order (a/R)², no accuracy estimate |


---

## Checking record (L1, 2026-11-08)

| # | Place | L1a | L1b | Resolution |
| --- | --- | --- | --- | --- |
| 1 | All equations of Secs. 5.24, 5.26, 5.28, 5.29, 5.34, 5.35 | transcribed | transcribed | Normalized LaTeX diff: identical in every symbol, exponent, index and sign. Differences only in layout, order of the definitions (R₁, R₂, T, C) and punctuation. |
| 2 | 5.26a (2a), (2b): extent of `ln` | power factor inside the logarithm | same, noted as typographically ambiguous | Inside: α → 0 limit gives (2c); numbers reproduce Fig. 5.26-3; outside gives negative B (see the reading note). |
| 3 | 5.28b (1): Ψ | Ψ(x) = ψ(x+1) | same, with numerical check | Confirmed: ψ(x+1) reproduces Fig. 5.28-6 (0.645, 0.878, 1.073 at 60°, 90°, 120°); plain ψ gives 2.88 at 90°. |
| 4 | 5.24a (2a) denominator: λ_g or λ_g′ | λ_g | λ_g | Print has unprimed λ_g (600 dpi). λ_g′ does not cure the large-a/λ blow-up (evaluation above). Kept as printed. |
| 5 | 5.24b N₂₂: `1 + 3α²` | as printed, flagged | as printed | Print confirmed; `3 + α²` fits Fig. 5.24-6 worse at every point checked. Kept as printed. |
| 6 | 5.24a (1a)-(3a) at a/λ ≳ 0.7 | — | — | Formulas as printed diverge from the book's curves (pole of X₁₁ at α ≈ 0.63 for a/λ = 1.4); documented above; use the figures or (1c)/(2c). |
| 7 | Figs. 5.28-2, 5.28-5, 5.29-2: circuit topology | "series jB_a, shunt −jB_b" (5.28-5), "series jB_a, shunt −jB_b, shunt jB_a" (5.28-2, 5.29-2) | π network: shunt jB_a across each arm, −jB_b in series | L1b is right (5.28-5 drawing; ABCD check that only the π network gives Eqs. (1)-(2) of 5.28a and 5.29a). |
| 8 | Fig. 5.28-1, 5.29-1: planes T | "at distance d from the outer walls" | "at distance d from the outer wall of the other arm" | Same planes; L1b's wording kept (checked on the figure: the plane T of one arm is parallel to, and at distance d from, the outer wall of the other arm). |
| 9 | Elliptic integrals F(α), E(α) | modulus convention not stated; argument read as the modulus k | not stated | Modulus k confirmed numerically (Fig. 5.24-2 at a/λ = 0.5). |
| 10 | 5.24b (5c): product across the line break | product, brace closes after [Q + Q′] | same | Agreed; confirmed on the page. |
| 11 | 5.29b measured table | 21 values | 21 values | Identical; 120° at λ = 3.4 cm, X_b/Z₀ = 47.9, as printed. |
| 12 | 5.35 (1), (2) | `15/π⁴` over `(2a/λ)² − 1`; `(4a/λ)²` | same formulas | Agreed; L1a's reading notes added. |

No mitered (chamfered) bend is treated in pp. 296-335 (L1a and L1b agree; L1 found no occurrence of "miter",
"mitre", "chamfer" or "bevel" in the OCR text layer of PDF pp. 310-349).
