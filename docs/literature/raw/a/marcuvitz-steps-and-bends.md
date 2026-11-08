# Marcuvitz, *Waveguide Handbook* (1951): steps and bends (D6, transcription by L1a)

Source: N. Marcuvitz, *Waveguide Handbook*, MIT Radiation Laboratory Series vol. 10, McGraw-Hill, New York (1951),
Chapter 5, "Asymmetric structures; coupling of two guides", Secs. 5.24–5.35, pp. 296–335 (PDF pp. 310–349:
PDF page = book page + 14).

Transcribed by L1a on 2026-11-08 from page images rendered at 220 dpi (500 dpi for doubtful glyphs), symbol by
symbol, for checking by L1b. The PDF text layer is OCR and unusable for formulas. Equations keep the book's
numbering and symbols; the book's `ln` is the natural logarithm. Notes by the transcriber are marked *[L1a: ...]*.

**Scope found in pp. 296–335.** H-plane steps: Sec. 5.24 (symmetric and asymmetric). E-plane steps: Sec. 5.26
(symmetric and asymmetric). Sharp (unmitered) corners: Sec. 5.28 (E-plane, right-angle and arbitrary angle) and
Sec. 5.29 (H-plane, right-angle and arbitrary angle, the latter only as measured data). Curved bends: Sec. 5.34
(E-plane) and Sec. 5.35 (H-plane), as the junction of a straight guide with a uniform circular bend. **There is no
mitered-bend section in these pages**; the corners of Secs. 5.28–5.29 are sharp corners with the full outer corner.
Secs. 5.25 (H-plane bifurcation), 5.27 (coaxial steps) and 5.30–5.33 (radial guide, coaxial/circular couplings,
apertures) are in the page range but outside the card's list and are not transcribed.

Common notation (book's conventions): $\lambda$ free-space wavelength, $\lambda_g$ guide wavelength of the H₁₀
mode, $\lambda_g = \lambda/\sqrt{1-(\lambda/2a)^2}$; $Y_0 = 1/Z_0$ characteristic admittance of the H₁₀ line;
primed quantities refer to the second guide. Reactances/susceptances are normalized as written.

---

## Sec. 5.24 Junction of two rectangular guides, H-plane (H-plane step), pp. 296–302

### a. Symmetrical case (p. 296)

"A junction of two rectangular guides of unequal widths but of equal heights ($H_{10}$-mode in each rectangular
guide)." Widths $a$ (larger) and $a'$ (smaller), centred; Fig. 5.24-1. Equivalent circuit: shunt inductive
reactance $jX$ between lines $Z_0,\lambda_g$ and $Z_0',\lambda_g'$. "At reference planes $T$ and $T'$ for the
larger and smaller guide, respectively," ($T'$ lies a distance $l$ from the junction plane $T$ inside the larger
guide, Fig. 5.24-1 top view):

```latex
\frac{Z_0'}{Z_0} = \frac{\lambda_g' a'}{\lambda_g a}\,\frac{X_{11}}{\alpha^2 (X_{22}-X_0)}
  \left[1+\left(\frac{2a'X_0}{\lambda_g'}\right)^2\right],                                   \tag{1a}
\frac{Z_0'}{Z_0} \approx \frac{\lambda_g' a'}{\lambda_g a}\,(1.750)(1-0.58\alpha^2), \qquad \alpha \ll 1,   \tag{1b}
\frac{Z_0'}{Z_0} \approx \frac{\lambda_g' a'}{\lambda_g a}\left(1+\beta+\frac{\beta^2}{2}\right), \qquad \beta = (1-\alpha) \ll 1.   \tag{1c}
```

```latex
\frac{X}{Z_0} = \frac{2a}{\lambda_g}\,X_{11}\,
  \frac{1+\left(\dfrac{2a'X_0}{\lambda_g'}\right)^2}{1+\left(\dfrac{2a'}{\lambda_g}\right)^2 X_0 X_{22}},   \tag{2a}
\frac{X}{Z_0} \approx \frac{2a}{\lambda_g}\,2.330\,\alpha^2(1+1.56\alpha^2)(1+6.75\alpha^2 Q), \qquad \alpha \ll 1,   \tag{2b}
\frac{Z_0}{X} \approx \frac{\lambda_g}{2a}\,
  \frac{\beta^2(1+\beta)\ln\dfrac{2}{\beta}}{1-\dfrac{\beta}{2}}
  \left(1-\frac{27}{8}\,\frac{Q+Q'}{1+8\ln\dfrac{2}{\beta}}\right), \qquad \beta \ll 1.   \tag{2c}
```

```latex
\frac{l}{a} = \frac{\alpha X_0}{\pi},                                       \tag{3a}
\frac{l}{a} \approx 0.0084\,\alpha(1-1.5\alpha^2), \qquad \alpha \ll 1,    \tag{3b}
\frac{l}{a} \approx 0, \qquad \beta \ll 1                                   \tag{3c}
```

where (p. 297)

```latex
X_{11} = \frac{A}{1-\dfrac{1}{4A}\left[(A+1)^2 N_{11} + 2(A+1) C N_{12} + C^2 N_{22}\right]},
\qquad
X_{22} = \frac{A'}{1-\dfrac{1}{4A'}\left[(A'+1)^2 N_{22} + 2(A'+1) C N_{12} + C^2 N_{11}\right]},
\qquad
X_0 = X_{22} - \frac{X_{12}^2}{X_{11}} \approx A' - \frac{C^2}{A},
```

and

```latex
A = \frac{(1+R_1)(1-R_2)+T^2}{(1-R_1)(1-R_2)-T^2}, \qquad
A' = \frac{(1-R_1)(1+R_2)+T^2}{(1-R_1)(1-R_2)-T^2}, \qquad
C = \frac{2T}{(1-R_1)(1-R_2)-T^2},
R_1 = -\left(\frac{1-\alpha}{1+\alpha}\right)^{\alpha}, \qquad
R_2 = \left(\frac{1-\alpha}{1+\alpha}\right)^{1/\alpha}, \qquad
T = \frac{4\alpha}{1-\alpha^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\frac12\left(\alpha+\frac1\alpha\right)},
```

```latex
N_{11} = 2\left(\frac{a}{\lambda}\right)^2\left\{1+\frac{16R_1}{\pi(1-\alpha^2)}\,[E(\alpha)-\alpha'^2F(\alpha)]\,[E(\alpha')-\alpha^2F(\alpha')]
  - R_1^2 - \alpha^2T^2\right\}
  + \frac{12\alpha^4}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{4\alpha}\left[Q-\frac12\left(\frac{2a}{3\lambda}\right)^2\right]
  + \frac{48\alpha^2}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\alpha+\frac3\alpha}\left[Q'-\frac12\left(\frac23\,\frac{a'}{\lambda}\right)^2\right],
```

```latex
N_{22} = 2\left(\frac{a}{\lambda}\right)^2\left\{\alpha^2+\frac{16\alpha^2R_2E(\alpha)}{\pi(1-\alpha^2)}\,[F(\alpha')-E(\alpha')] - \alpha^2R_2^2 - T^2\right\}
  + \frac{48\alpha^2}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{3\alpha+\frac1\alpha}\left[Q-\frac12\left(\frac23\,\frac{a}{\lambda}\right)^2\right]
  + \frac{12}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{4/\alpha}\left[Q'-\frac12\left(\frac23\,\frac{a'}{\lambda}\right)^2\right],
```

```latex
N_{12} = 2\left(\frac{a}{\lambda}\right)^2\left\{\alpha'^2-\frac{4E(\alpha)}{\pi}\,[E(\alpha')-\alpha^2F(\alpha')] - R_1 + \alpha^2R_2\right\}T
  + \frac{24\alpha^3}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\frac12\left(7\alpha+\frac1\alpha\right)}\left[Q-\frac12\left(\frac{2a}{3\lambda}\right)^2\right]
  + \frac{24\alpha}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\frac12\left(\alpha+\frac7\alpha\right)}\left[Q'-\frac12\left(\frac23\,\frac{a'}{\lambda}\right)^2\right].
```

```latex
\alpha = \frac{a'}{a} = 1-\beta, \qquad \alpha' = \sqrt{1-\alpha^2}, \qquad
Q = 1-\sqrt{1-\left(\frac{2a}{3\lambda}\right)^2}, \qquad Q' = 1-\sqrt{1-\left(\frac23\,\frac{a'}{\lambda}\right)^2}.
```

"The functions $F(\alpha)$ and $E(\alpha)$ are complete elliptical integrals of the first and second kinds,
respectively." (p. 298) *[L1a: the modulus convention (k or m = k²) is not stated; the book's argument is the
modulus k.]*

**Restrictions (p. 298).** Valid for $0.5 < a/\lambda < 1.5$ and $2a'/\lambda > 1$. "Equations (1a), (2a), and
(3a) have been obtained by the equivalent static method using one static mode incident in each guide and are
estimated to be in error by less than 1 per cent over most of the wavelength range; the error may rise to as
much as 5 per cent at the limit $a/\lambda = 1.5$." Eqs. (1b), (2b), (3b) (small aperture) differ from (1a),
(2a), (3a) "by less than 5 per cent for $0.5 < (a/\lambda) < 1$ and $\alpha < 0.4$; for $\alpha < 0.5$ the
difference is less than 10 per cent." Eqs. (1c), (2c), (3c) (wide aperture) "are in error by less than 6 per
cent for $0.5 < (a/\lambda) < 1$ and $\beta < 0.3$." "$X_{11}$, $X_{12}$, and $X_{22}$ are the impedance matrix
elements of the T network referred to a common reference plane $T$ and correspond to a characteristic impedance
choice, $Z_0'/Z_0 = \lambda_g' a/\lambda_g a'$."

**Numerical results.** Fig. 5.24-2 (p. 299): $Z_0'\lambda_g a/Z_0\lambda_g' a'$ and $l/a$ vs $\alpha$ for
$a/\lambda = 0.5$ and 1.4; Fig. 5.24-3 (p. 300): $X\lambda_g/Z_0 2a$ and $Z_0 2a/X\lambda_g$ vs $\alpha$. "These
results have been plotted from data obtained by use of the equivalent static method with two modes incident in
each guide and are somewhat more accurate than the analytic results given in Eqs. (1a), (2a), and (3a)."

### b. Asymmetrical case (pp. 298, 301–302)

"A junction of two rectangular guides of unequal widths but of equal heights ($H_{10}$-mode in each rectangular
guide)", one side wall continuous (Fig. 5.24-4). Same circuit and reference planes.

```latex
\frac{Z_0'}{Z_0} = \text{same as in Eq. (1a), Sec. 5.24a},                                    \tag{4a}
\frac{Z_0'}{Z_0} \approx \frac{\lambda_g' a'^3}{\lambda_g a^3}\,5.046\,(1-1.44\alpha^2), \qquad \alpha \ll 1,   \tag{4b}
\frac{Z_0'}{Z_0} \approx \frac{\lambda_g'/a'}{\lambda_g/a}\left(1-\beta-\frac{7\beta^2}{4}\right), \qquad \beta = (1-\alpha) \ll 1.   \tag{4c}
```

```latex
\frac{X}{Z_0} = \text{same as Eq. (2a), Sec. 5.24a},                                           \tag{5a}
\frac{X}{Z_0} \approx \frac{2a}{\lambda_g}\,5.764\,\alpha^4(1-1.33\alpha^2)(1+41.3\alpha^4Q), \qquad \alpha \ll 1   \tag{5b}
\frac{Z_0}{X} \approx \frac{\lambda_g}{2a}\,
 \frac{\beta^2}{1-\dfrac{\beta}{2}-\beta^2\left(\ln\dfrac{2}{\beta}+\dfrac32\right)}
 \left\{\frac74\ln\frac2\beta\left[1+\beta-\frac54\beta^2\ln\frac2\beta\right]
 \cdot\left[1+\frac47(1-\beta)^2\left(\frac{a}{\lambda}\right)^2\right]
 -\frac89\left[1+3\beta\ln\frac2\beta\right][Q+Q']\right\}.   \tag{5c}
```

*[L1a: in (5c) the second bracket is printed on the next line after a centred dot ("·" implied by the line
break: "[1 + β − (5/4)β² ln 2/β]." followed by "[1 + (4/7)(1 − β)²(a/λ)²]"). It is read as a product of the two
brackets, both multiplying (7/4) ln(2/β); the closing brace after [Q + Q'] ends the expression.]*

```latex
\frac{l}{a} = \text{same as in Eq. (3a), Sec. 5.24a},   \tag{6a}
\frac{l}{a} \approx 0.0259\,\alpha(1-0.6\alpha^2), \qquad \alpha \ll 1,   \tag{6b}
\frac{l}{a} \approx 0, \qquad \alpha \approx 1.   \tag{6c}
```

"The parameters $X_{11}, X_{12}, X_{22}, X_0, A, A', C, \alpha, \alpha'$ and $\beta$ are defined as in Sec. 5.24a
but with" (p. 301–302)

```latex
R_1 = -\left(\frac{1+3\alpha^2}{1-\alpha^2}\right)\left(\frac{1-\alpha}{1+\alpha}\right)^{2\alpha}, \qquad
R_2 = \left(\frac{3+\alpha^2}{1-\alpha^2}\right)\left(\frac{1-\alpha}{1+\alpha}\right)^{2/\alpha}, \qquad
T = \frac{16\alpha^2}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\alpha+(1/\alpha)},
```

```latex
N_{11} = 2\left(\frac{a}{\lambda}\right)^2\left[1+\frac{8\alpha(1-\alpha^2)}{1+3\alpha^2}R_1\ln\frac{1-\alpha}{1+\alpha}
   + \frac{16\alpha^2}{1+3\alpha^2}R_1 - R_1^2 - \alpha^2T^2\right]
 + 2\left[\frac{32}{3}\,\frac{\alpha^4}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{3\alpha}\right]^2\left[Q-\frac12\left(\frac{a}{\lambda}\right)^2\right]
 + 2\left[\frac{32\alpha^2}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{\alpha+(2/\alpha)}\right]^2\left[Q'-\frac12\left(\frac{a'}{\lambda}\right)^2\right],
```

```latex
N_{22} = 2\left(\frac{a}{\lambda}\right)^2\left[\alpha^2-\frac{8\alpha(1-\alpha^2)}{3+\alpha^2}R_2\ln\frac{1-\alpha}{1+\alpha}
   + \frac{16\alpha^2}{1+3\alpha^2}R_2 - \alpha^2R_2^2 - T^2\right]
 + 2\left[\frac{32\alpha^2}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{2\alpha+(1/\alpha)}\right]^2\left[Q-\frac12\left(\frac{a}{\lambda}\right)^2\right]
 + 2\left[\frac{32}{3}\,\frac{1}{(1-\alpha^2)^2}\left(\frac{1-\alpha}{1+\alpha}\right)^{3/\alpha}\right]^2\left[Q'-\frac12\left(\frac{a'}{\lambda}\right)^2\right],
```

```latex
N_{12} = 2\left(\frac{a}{\lambda}\right)^2\left[\frac{(1-\alpha^2)^2}{2\alpha}\ln\frac{1-\alpha}{1+\alpha} - R_1 - \alpha^2R_2\right]T
 + \frac23\left(\frac{1-\alpha}{1+\alpha}\right)^{5\alpha+(1/\alpha)}\left[\frac{32\alpha^3}{(1-\alpha^2)^2}\right]^2\left[Q-\frac12\left(\frac{a}{\lambda}\right)^2\right]
 + \frac23\left(\frac{1-\alpha}{1+\alpha}\right)^{\alpha+(5/\alpha)}\left[\frac{32\alpha}{(1-\alpha^2)^2}\right]^2\left[Q'-\frac12\left(\frac{a'}{\lambda}\right)^2\right],
```

```latex
Q = 1-\sqrt{1-\left(\frac{a}{\lambda}\right)^2}, \qquad Q' = 1-\sqrt{1-\left(\frac{a'}{\lambda}\right)^2}.
```

*[L1a: in $N_{22}$ the third term inside the first bracket is printed with denominator $1+3\alpha^2$ (not
$3+\alpha^2$); transcribed as printed. Q and Q' of the asymmetric case have no factor 2/3, unlike Sec. 5.24a.]*

**Restrictions (p. 302).** Valid for $0.5 < a/\lambda < 1.0$. Eqs. (4a), (5a), (6a) (equivalent static method,
one static mode per guide) "are estimated to be in error by less than 1 per cent over most of the wavelength
range and by as high as 5 per cent at the limit $a/\lambda = 1.0$." Eqs. (4b), (5b), (6b) differ from (4a), (5a),
(6a) "by less than 5 per cent for $\alpha < 0.4$ and $0.5 < a/\lambda < 1$." Eqs. (4c), (5c), (6c) "are in error
by less than 10 per cent for $\beta < 0.2$ and $0.5 < a/\lambda < 0.8$." Same characteristic impedance choice
$Z_0'/Z_0 = \lambda_g' a/\lambda_g a'$. Numerical results: Figs. 5.24-5 and 5.24-6 ($Z_0'\lambda_g a^3/Z_0\lambda_g'
a'^3$, $l/a$ and $X\lambda_g/Z_0 2a$ vs $\alpha$ for various $\lambda/a$), from the two-mode equivalent static
method.

---

## Sec. 5.26 Change in height of rectangular guide (E-plane step), pp. 307–310

### a. Symmetrical case (pp. 307–308)

"An axially symmetrical junction of two rectangular guides of equal widths and unequal heights ($H_{10}$-mode in
rectangular guides)." Heights $b$ (larger) and $b'$; Fig. 5.26-1. Circuit: shunt capacitive susceptance $jB$
between $Y_0$ and $Y_0'$, same $\lambda_g$ on both sides. "At the terminal plane $T$" (the junction plane):

```latex
\frac{Y_0}{Y_0'} = \frac{b'}{b} = \alpha = 1-\delta,   \tag{1}
\frac{B}{Y_0} = \frac{2b}{\lambda_g}\left[\ln\left(\frac{1-\alpha^2}{4\alpha}\right)\left(\frac{1+\alpha}{1-\alpha}\right)^{\frac12\left(\alpha+\frac1\alpha\right)}
 + 2\,\frac{A+A'+2C}{AA'-C^2}
 + \left(\frac{b}{4\lambda_g}\right)^2\left(\frac{1-\alpha}{1+\alpha}\right)^{4\alpha}\left(\frac{5\alpha^2-1}{1-\alpha^2}+\frac43\,\frac{\alpha^2C}{A}\right)^2\right],   \tag{2a}
\frac{B}{Y_0} \approx \frac{2b}{\lambda_g}\left[\ln\left(\frac{1-\alpha^2}{4\alpha}\right)\left(\frac{1+\alpha}{1-\alpha}\right)^{\frac12\left(\alpha+\frac1\alpha\right)} + \frac{2}{A}\right],   \tag{2b}
\frac{B}{Y_0} \approx \frac{2b}{\lambda_g}\left[\ln\frac{e}{4\alpha}+\frac{\alpha^2}{3}+\frac12\left(\frac{b}{\lambda_g}\right)^2(1-\alpha^2)^4\right], \qquad \alpha \ll 1,   \tag{2c}
\frac{B}{Y_0} \approx \frac{2b}{\lambda_g}\left(\frac{\delta}{2}\right)^2\left[\frac{2\ln\dfrac{2}{\delta}}{1-\delta}+1+\frac{17}{16}\left(\frac{b}{\lambda_g}\right)^2\right], \qquad \delta \ll 1,   \tag{2d}
```

where

```latex
A = \left(\frac{1+\alpha}{1-\alpha}\right)^{2\alpha}\frac{1+\sqrt{1-\left(\dfrac{b}{\lambda_g}\right)^2}}{1-\sqrt{1-\left(\dfrac{b}{\lambda_g}\right)^2}} - \frac{1+3\alpha^2}{1-\alpha^2},
\qquad
A' = \left(\frac{1+\alpha}{1-\alpha}\right)^{2/\alpha}\frac{1+\sqrt{1-\left(\dfrac{b'}{\lambda_g}\right)^2}}{1-\sqrt{1-\left(\dfrac{b'}{\lambda_g}\right)^2}} + \frac{3+\alpha^2}{1-\alpha^2},
\qquad
C = \left(\frac{4\alpha}{1-\alpha^2}\right)^2, \qquad e = 2.718.
```

*[L1a: the print reads "ln((1−α²)/4α) ((1+α)/(1−α))^{½(α+1/α)}" with no bracket closing the logarithm
before the power term. It is transcribed with the power term inside the logarithm. Check: for α → 0 the power
term tends to e (since ((1+α)/(1−α))^{1/(2α)} → e), so the bracket of (2a) starts with ln(e/4α), the first term
of the printed asymptotic form (2c); with the power term outside the logarithm this limit would not hold. L1b
please check.]*

**Restrictions (p. 308).** "valid in the range $b/\lambda_g < 1$. Equation (2a) has been obtained by the
equivalent static method employing a static aperture field due to the incidence of the two lowest modes and is
correct to within 1 per cent in the range $b/\lambda_g < 1$. The approximate Eq. (2b) is correct to within 3 per
cent for $(b/\lambda_g) < 0.7$; for $(b/\lambda_g) \geq 0.7$ it is still correct to within 3 per cent if
$\alpha < 0.7$. Equation (2c) is an asymptotic expansion of Eq. (2a) correct to within 5 per cent if
$\alpha < 0.6$ and $b/\lambda_g < 0.5$ and to within 2 per cent if $\alpha < 0.4$ and $b/\lambda_g < 0.4$.
Equation (2d) is an asymptotic expansion of Eq. (2a) correct to within 5 per cent when $\delta < 0.5$ and
$b/\lambda_g < 0.5$ and to within 3 per cent when $\delta < 0.4$ and $b/\lambda_g < 0.4$."

**Numerical results.** Fig. 5.26-3 (p. 309): $B\lambda_g/Y_0 b$ from Eq. (2a) vs $b'/b$ for
$b/\lambda_g$ = 0, 0.4, 0.6, 0.7, 0.8, 0.9. **Experimental results:** "verified experimentally at least for the
cases $b/\lambda_g = 0.23$ and $\alpha > 0.15$ to within a few per cent."

### b. Asymmetrical case (p. 310)

"An axially asymmetrical junction of two rectangular guides of equal widths but unequal heights" (one broad wall
continuous, Fig. 5.26-2). "Equivalent-circuit Parameters.—Same as Sec. 5.26a except $\lambda_g$ is replaced by
$\lambda_g/2$. Restrictions.—Same as Sec. 5.26a except $\lambda_g$ is replaced by $\lambda_g/2$." Fig. 5.26-3 with
$\lambda_g \to \lambda_g/2$ gives $B\lambda_g/2Y_0b$ vs $b'/b$ with $2b/\lambda_g$ as parameter.

---

## Sec. 5.28 E-plane corners, pp. 312–318

### a. Right-angle bends (pp. 312–316)

"A right-angle E-plane junction of two rectangular guides of equal dimensions ($H_{10}$-mode in rectangular
guides)." Sharp corner; Fig. 5.28-1. "At reference planes $T$ the equivalent circuit is pure shunt" ($-jB$ at the
junction of the two lines). In the side view of Fig. 5.28-1 the planes $T$ lie at distance $d$ from the outer
walls of the corner (one plane per arm), and the planes $T'$ continue the inner walls, at distance $b$ from the
outer walls; $b-d$ is the distance from $T'$ to $T$. The relative shunt susceptance and location of the terminal
planes:

```latex
\frac{B}{Y_0} = \frac{\left(\dfrac{B_a}{Y_0}\right)^2+1}{\dfrac{B_b}{Y_0}} - 2\,\frac{B_a}{Y_0},   \tag{1}
\frac{2\pi(b-d)}{\lambda_g} = \cot^{-1}\left(2\frac{B_b}{Y_0}-\frac{B_a}{Y_0}\right),   \tag{2}
```

where

```latex
\frac{B_a}{Y_0} = \frac{2b}{\lambda_g}\left\{-\frac{\cot\pi x}{x}+\frac{1}{\pi x^2}+\frac{\pi}{6}-\ln 2
 - \left[\frac{A_0e^{-\frac{\pi}{2}}+(A_1-A_2)e^{-\pi}+(1+5e^{-\pi})\dfrac{A_0^2}{16}}{1-(1+5e^{-\pi})\left(\dfrac{A_1-A_2}{4}\right)}\right]\right\},   \tag{3a}
\frac{B_a}{Y_0} \approx \frac{2b}{\lambda_g}\left[0.878+0.498\left(\frac{2b}{\lambda_g}\right)^2\right], \qquad \frac{2b}{\lambda_g} \ll 1,   \tag{3b}
```

```latex
\frac{2B_b}{Y_0}-\frac{B_a}{Y_0} = \frac{\lambda_g}{2\pi b}\left\{1+\pi x\cot\pi x-\pi x^2\left[5\ln2-\frac{7\pi}{6}-8\sum_{n=1}^{\infty}\frac{1}{n(e^{2\pi n}-1)}\right]
 + \pi x^2\left[\frac{A_0'e^{-\frac{\pi}{2}}-(A_1+A_2)e^{-\pi}+(1-3e^{-\pi})\dfrac{A_0'^2}{16}}{1+(1-3e^{-\pi})\dfrac{A_1+A_2}{4}}\right]\right\},   \tag{4a}
\frac{B_b}{Y_0} \approx \frac{\lambda_g}{2\pi b}\left[1-0.114\left(\frac{2b}{\lambda_g}\right)^2\right], \qquad \frac{2b}{\lambda_g} \ll 1,   \tag{4b}
```

and

```latex
x = \frac{2b}{\lambda_g}, \qquad
A_0 = \frac{4}{\pi}\,\frac{x^2}{1-x^2}, \qquad A_0' = A_0-\frac{8}{\sinh\pi}, \qquad
A_1 = \frac1\pi\,\frac{x^2}{1-0.5x^2}, \qquad
A_2 = 4\left[\frac{1}{\sqrt{1-x^2}\,\left(1-e^{-2\pi\sqrt{1-x^2}}\right)}-\frac{1}{1-e^{-2\pi}}\right].
```

"An alternative circuit at the reference planes $T'$ is shown in Fig. 5.28-2" (series $jB_a$, shunt $-jB_b$,
second shunt $jB_a$; parameters $B_a$, $B_b$ above).

**Restrictions (pp. 313, 316).** "The equivalent circuits are valid in the range $2b/\lambda_g < 1$. The circuit
parameters obtained by the equivalent static method, employing two lowest modes incident in each guide, are
accurate to within 1 per cent in the above range. Equations (3b) and (4b) are asymptotic approximations to $B_a$
and $B_b$ that agree with the values from Eqs. (3a) and (4a) to within 8 per cent for $2b/\lambda_g < 0.6$."

**Numerical and experimental results (p. 316).** Fig. 5.28-3 (p. 314): $d/b$ and $B\lambda_g/Y_0 2b$ vs
$2b/\lambda_g$; Fig. 5.28-4 (p. 315): $B_a\lambda_g/Y_0 2b$ and $B_b 2\pi b/Y_0\lambda_g$ vs $2b/\lambda_g$. "A few
measured values for $B_a$ and $B_b$ are indicated by the circled points in Fig. 5.28-4. These data were taken at
wavelengths in the vicinity of 3 cm."

### b. Arbitrary angle bends (pp. 316–318)

"A symmetrical, arbitrary angle, E-plane junction of two rectangular guides of equal dimensions." Bend angle
$\theta$ (deflection angle between the arm axes, Fig. 5.28-5; $\theta = 90°$ is the right-angle bend); circuit of
Fig. 5.28-5 (series $jB_a$ in each arm, shunt $-jB_b$), "At the reference planes $T$" ($T_1$, $T_2$ through the
inner corner, perpendicular to each arm, as drawn):

```latex
\frac{B_a}{Y_0} = \frac{2b}{\lambda_g}\left\{\Psi\left[-\frac12\left(1-\frac{\theta}{\pi}\right)\right]-\Psi\left[-\frac12\right]\right\},   \tag{1}
\frac{B_b}{Y_0} = \frac{\lambda_g}{2\pi b}\cot\frac{\theta}{2},   \tag{2}
```

"where $\Psi(x)$ is the derivative of the logarithm of $x!$" *[L1a: $\Psi(x) = \frac{d}{dx}\ln x! =
\psi(x+1)$ in today's digamma notation.]*

**Restrictions (p. 316).** "applicable in the wavelength range $2b/\lambda_g < 1$. The circuit parameters have
been obtained by a simple equivalent static method and are rigorous only in the static limit
$b/\lambda_g \approx 0$." Fig. 5.28-6 (p. 317): $B_a\lambda_g/Y_0 2b$ and $B_b b/Y_0\lambda_g$ vs $\theta$
(0°–120°), static curves plus measured points. **Experimental results (pp. 316–318):** measured "in rectangular
guides of dimensions $a = 0.900$ in. and $b = 0.400$ in. ... for several values of $b/\lambda_g$. The measured
values of $B_b b/Y_0\lambda_g$ are within a few per cent of the static values given in Eq. (2) for the bend angles
$\theta$ = 60, 75, and 90° if $2b/\lambda_g < 0.51$."

---

## Sec. 5.29 H-plane corners, pp. 318–322

### a. Right-angle bends (pp. 318–319)

"An H-plane right-angle junction of two rectangular guides with equal dimensions." Sharp corner; Fig. 5.29-1.
"At the terminal planes $T$ the equivalent circuit is pure shunt" (capacitive $jB$). In the top view of
Fig. 5.29-1 the planes $T$ lie at distance $d$ from the outer walls and $T'$ continue the inner walls (distance $a$
from the outer walls). Relative shunt susceptance and location of the terminal planes:

```latex
\frac{B}{Y_0} = \frac{1+\left(\dfrac{B_a}{Y_0}\right)^2}{\dfrac{B_b}{Y_0}} + 2\,\frac{B_a}{Y_0},   \tag{1}
\frac{2\pi(a-d)}{\lambda_g} = -\cot^{-1}\frac{B_a+2B_b}{Y_0},   \tag{2}
```

where

```latex
\frac{B_a}{Y_0} = \frac{\lambda_g}{2a}\left[A_1-x\cot\pi x+\frac1\pi\left(\frac{1+x^2}{1-x^2}\right)+A_2\right](1+A_3),   \tag{3a}
\frac{B_a}{Y_0} \approx 0.42+\frac{0.64}{1-x}, \qquad x \approx 1,   \tag{3b}
\frac{B_a+2B_b}{Y_0} = \frac{\lambda_g}{2a}\left[-A_1'-x\cot\pi x-\frac1\pi\left(\frac{1+x^2}{1-x^2}\right)+A_2'\right](1+A_3'),   \tag{4a}
\frac{B_a+2B_b}{Y_0} \approx -2.04+1.63x, \qquad x \approx 1,   \tag{4b}
```

and

```latex
x = \frac{2a}{\lambda_g}, \qquad \lambda_g = \frac{\lambda}{\sqrt{1-\left(\dfrac{\lambda}{2a}\right)^2}},
A_1 = \frac{2}{1-e^{-2\pi}} - \frac{4(1+63e^{-2\pi})}{\left[2(1+5e^{-\pi})(1+63e^{-2\pi})-\left(\frac{64}{3}\right)^2e^{-3\pi}\right]} = 0.3366,
A_1' = -\frac{2}{1-e^{-2\pi}} + \frac{4(1-33e^{-2\pi})}{\left[2(1-3e^{-\pi})(1-33e^{-2\pi})-\left(\frac{32}{3}\right)^2e^{-3\pi}\right]} = 0.3071,
```

```latex
A_2 = \frac{\left[0.1801\left(\dfrac{1+x^2}{4-x^2}\right)+0.2021\right]^2}
           {\dfrac{\sqrt{3-x^2}}{1-e^{-2\pi\sqrt{3-x^2}}}-0.1858-0.1592\left(\dfrac{1+x^2}{7-x^2}\right)},
\qquad
A_2' = \frac{\left[0.1801\left(\dfrac{1+x^2}{4-x^2}\right)+0.1669\right]^2}
           {\dfrac{\sqrt{3-x^2}}{1-e^{-2\pi\sqrt{3-x^2}}}+0.1432+0.1592\left(\dfrac{1+x^2}{7-x^2}\right)},
```

```latex
A_3 = 0.01-0.0095x \quad \text{for } 0 \le x \le 1, \qquad
A_3 = -\frac{0.00416}{1.5-x}+0.0040+0.0019x+0.0020x^2 \quad \text{for } 1 \le x \le 1.436,
A_3' = -\frac{0.00146}{1.26-x}-0.0016+0.0034x-0.0026x^2 \quad \text{for } 0 \le x \le 1.2, \qquad
A_3' = \frac{0.00136}{x-1.26}+0.0076-0.0066x \quad \text{for } 1.3 \le x \le 1.5.
```

*[L1a: the printed constants A₁ = 0.3366 and A₁' = 0.3071 were recomputed from the transcribed expressions:
0.336574 and 0.307152 (Python, math.exp). This confirms the reading (64/3)² and (32/3)², which are small glyphs in
the print.]*

"An alternative equivalent circuit at reference planes $T'$ is in Fig. 5.29-2" (series $jB_a$, shunt $-jB_b$,
shunt $jB_a$).

**Restrictions (p. 319).** "valid in the wavelength range $a < \lambda < 2a$. The circuit parameters have been
derived by the equivalent static method with two static modes incident in each guide. The error in the
susceptance values is estimated to be of the order of magnitude of 1 per cent in the range
$0 \le 2a/\lambda_g \le 1.44$, excluding the range $1.2 \le 2a/\lambda_g \le 1.3$ in which the variational
correction $A_3'$ diverges. Equations (3b) and (4b) are approximations to Eqs. (3a) and (4a), respectively, and
agree with these to within 10 per cent in the resonant range $0.7 \le 2a/\lambda_g \le 1.2$. The quantity $A_2$
is less than 10 per cent of $B_a/Y_0$ for $2a/\lambda_g < 1.3$; $A_2'$ is less than 10 per cent of
$(B_a+2B_b)/Y_0$ for $2a/\lambda_g < 1.0$."

**Numerical and experimental results.** Fig. 5.29-3 (p. 320): $B2a/Y_0\lambda_g$ and $d/a$ vs $2a/\lambda_g$;
Fig. 5.29-4 (p. 321): $Y_0\lambda_g/B_b2a$ and $(B_a+2B_b)2a/Y_0\lambda_g$ vs $2a/\lambda_g$, with circled
measured points "measured in rectangular guides of dimensions $a = 0.900$ in. and $b = 0.400$ in."

### b. Arbitrary angle H-plane corners (pp. 319, 322): measured data only

"A symmetrical, arbitrary angle, H-plane junction of two rectangular guides of identical cross-sectional
dimensions." Circuit of Fig. 5.29-5: T network with series reactances $jX_b$ in each arm and shunt $-jX_a$,
reference planes $T$ through the inner corner perpendicular to each arm (as drawn), $\theta$ the deflection
angle. "The equivalent circuit of Fig. 5.29-5 is valid in the wavelength range $a < \lambda < 2a$. The circuit
parameters $Z_0/X_a$ and $X_b/Z_0$, relative to the reference planes $T$, were measured in rectangular waveguide
of dimensions $a = 0.900$ in. and $b = 0.400$ in. as a function of the bend angle $\theta$ at various
wavelengths. The resulting data are:" (table, p. 322, transcribed exactly; empty cells are blank in the book)

| θ | λ = 3.0 cm (λg = 1.560″): Z₀/Xa | Xb/Z₀ | λ = 3.2 cm (λg = 1.762″): Z₀/Xa | Xb/Z₀ | λ = 3.4 cm (λg = 1.999″): Z₀/Xa | Xb/Z₀ |
| --- | --- | --- | --- | --- | --- | --- |
| 30° | 0.849 | 0.493 | 0.782 | 0.420 | 0.718 | 0.365 |
| 45° | | | | | 0.995 | 0.567 |
| 50° | 1.102 | 0.943 | | | 1.068 | 0.650 |
| 60° | 1.088 | 1.27 | 1.158 | 0.998 | 1.176 | 0.817 |
| 75° | 0.864 | 2.14 | | | | |
| 90° | 0.324 | 6.46 | 0.739 | 2.75 | 1.055 | 1.76 |
| 120° | | | −2.17 | −4.58 | 0.044 | 47.9 |

---

## Sec. 5.34 Circular bends, E-plane, pp. 333–334

"An E-plane junction of a rectangular guide and a uniform circular bend of rectangular cross section ($H_{10}$-mode
in rectangular guide and in circular bend)." Mean radius $R$, bend in the plane of side $b$; Fig. 5.34-1. Circuit:
series capacitive reactance $-jX$ between $Z_0, \lambda_g$ and $Z_0', \lambda_{g\phi}$. "At the reference plane $T$"
(the junction plane):

```latex
\frac{Z_0'}{Z_0} = 1+\frac{1}{12}\left(\frac{b}{R}\right)^2\left[\frac12-\frac15\left(\frac{2\pi b}{\lambda_g}\right)^2\right],   \tag{1}
\frac{X}{Z_0} = \frac{32}{\pi^7}\left(\frac{2\pi b}{\lambda_g}\right)^3\left(\frac{b}{R}\right)^2\sum_{n=1,3,\ldots}^{\infty}\frac{1}{n^7}\sqrt{1-\left(\frac{2b}{n\lambda_g}\right)^2},   \tag{2}
```

where

```latex
\lambda_g = \frac{\lambda}{\sqrt{1-\left(\dfrac{\lambda}{2a}\right)^2}}, \qquad
\lambda_{g\phi} = \lambda_g\left\{1-\frac{1}{12}\left(\frac{b}{R}\right)^2\left[-\frac12+\frac15\left(\frac{2\pi b}{\lambda_g}\right)^2\cdots\right]\right\}.
```

"The circular bend is a uniform angular guide with propagation wavelength $\lambda_{g\phi}$, the latter being
measured along the arc of mean radius $R$. The electrical length corresponding to an angular distance $\phi$ or to
a (mean) arc distance $s$ is"

```latex
\frac{2\pi s}{\lambda_{g\phi}} = m\phi, \qquad \text{where } m = \frac{2\pi R}{\lambda_{g\phi}}.
```

**Restrictions (p. 334).** "applicable in the wavelength range $2b/\lambda_g < 1$. Equations (1) and (2) have been
derived by a variational method employing the form of the lowest mode in the angular guide for the
electric-field distribution in the junction plane. The circuit parameters have been evaluated to order
$(b/R)^2$, but no estimate of the range of accuracy is available."

## Sec. 5.35 Circular bends, H-plane, pp. 334–335

"An H-plane junction of a rectangular guide and a uniform circular bend, i.e., angular guide, of rectangular
cross section ($H_{10}$-mode in rectangular and angular guides)." Mean radius $R$, bend in the plane of side $a$;
Fig. 5.35-1. Circuit: shunt inductive susceptance $-jB$ between $Y_0, \lambda_g$ and $Y_0', \lambda_{g\phi}$. "At
the reference plane $T$":

```latex
\frac{Y_0'}{Y_0} = 1+\frac{\pi^2}{120}\left(\frac{a}{R}\right)^2\left[1+\frac{\dfrac{15}{\pi^4}}{\left(\dfrac{2a}{\lambda}\right)^2-1}
 -\left(2-\frac{5}{\pi^2}-\frac{60}{\pi^4}\right)\left(\frac{2a}{\lambda}\right)^2
 +\left(1+\frac{15}{6\pi^2}-\frac{105}{\pi^4}\right)\left(\frac{2a}{\lambda}\right)^4\cdots\right],   \tag{1}
\frac{B}{Y_0} = \frac{\lambda_g}{a}\,\frac{8}{\pi^4}\left(\frac{a}{R}\right)^2\sum_{n=2,4,\ldots}^{\infty}
 \left\{\frac{n^3}{(n^2-1)^4}\left[1-\frac{\left(\dfrac{4a}{\lambda}\right)^2}{n^2-1}\right]^2\sqrt{1-\left(\frac{2a}{n\lambda}\right)^2}\right\},   \tag{2}
```

where

```latex
\lambda_g = \frac{\lambda}{\sqrt{1-\left(\dfrac{\lambda}{2a}\right)^2}}, \qquad
\frac{1}{\lambda_{g\phi}^2} = \frac{1}{\lambda_g^2}+\frac{1}{24R^2}\left[1-\left(\frac{12+\pi^2}{2\pi^2}\right)\left(\frac{2a}{\lambda}\right)^2+\left(\frac{15-\pi^2}{2\pi^2}\right)\left(\frac{2a}{\lambda}\right)^4\cdots\right].
```

"The propagation wavelength $\lambda_{g\phi}$ in the uniform angular guide (circular bend) is measured along the
arc of mean radius $R$." Electrical length $2\pi s/\lambda_{g\phi} = m\phi$, $m = 2\pi R/\lambda_{g\phi}$. "With this
interpretation the angular guide may be treated as a uniform line with characteristic admittance $Y_0'$ and
wavelength $\lambda_{g\phi}$."

**Restrictions (p. 335).** "applicable in the wavelength range $a < \lambda < 2a$. The circuit parameters have been
derived by a variational method employing a sine function for the distribution of electric field in the
junction plane. The results have been evaluated to order $(a/R)^2$ but no estimate of the range of accuracy is
available."

*[L1a: in (1) of Sec. 5.35 the term $15/\pi^4$ is printed as a numerator over $(2a/\lambda)^2 - 1$, i.e.
$\frac{15/\pi^4}{(2a/\lambda)^2-1}$; the trailing dots after the $(2a/\lambda)^4$ term are printed in the book
(series truncated). In (2) the bracket squared is $\left[1-\frac{(4a/\lambda)^2}{n^2-1}\right]^2$ as printed;
the ratio $(4a/\lambda)^2$ (not $(2a/\lambda)^2$) is clearly printed.]*

---

## Items to check (for L1b)

1. Sec. 5.26a, Eqs. (2a)–(2b): placement of the power term relative to the logarithm (see note; transcribed inside).
2. Sec. 5.24b, Eq. (5c): product of the two brackets across the line break.
3. Sec. 5.24b, $N_{22}$: denominator $1+3\alpha^2$ in the $R_2$ term, as printed.
4. Sec. 5.28a, Fig. 5.28-1: definition of $d$ and of the planes $T$, $T'$ (figure only).
5. Sec. 5.29b table: values as printed (120° at 3.4 cm: Xb/Z₀ = 47.9).
