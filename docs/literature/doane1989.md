# Doane & Anderson (1989): bend and twist design rules (D7)

Source: J. L. Doane and T. N. Anderson, "Oversized rectangular waveguides with mode-free bends and twists for
broadband applications", *Microwave Journal* 32(3), 153–160 (March 1989). The PDF is a phone photograph of the
printed pages; PDF page 6 is journal p. 158, PDF page 7 is p. 159 (page number printed at the bottom right of
p. 159 and bottom left of p. 158).

Transcribed independently by L1a and L1b from the page images (2026-11-08; raw files in `docs/literature/raw/a/`
and `raw/b/`), compared symbol by symbol with each other and with the page images by L1 (see *Checking record* at
the end). This is the merged text. Equations are written in LaTeX with the paper's own symbols. Text in quotation
marks is copied verbatim (short quotes). Notes by the transcribers and the merger are marked *[L1a: ...]*,
*[L1b: ...]*, *[L1: ...]*.

## Context on p. 158 (PDF p. 6), right column

Eqs. 14–15, constant-radius bend (quoted for context, outside the card's range):

```latex
P_c = (c\theta)^2\,(\sin u/u)^2 \qquad (14)
\qquad\text{where}\qquad
u = \big[(\Delta\beta R/2)^2 + c^2\big]^{1/2}\,\theta . \qquad (15)
```

Criterion leading to eq. 16 (p. 158, right column, last paragraph):

> "To estimate the bend arc length required in a hyperbolic secant bend to ensure less than 1 percent mode
> conversion, we start by taking L_B as twice the length of a constant radius bend with R = R₀. That is, for a
> 90° bend L_B = πR₀, where R₀ is the minimum bend radius. (This results in a more severe truncation than
> L_B/R₀ = 4 considered above, but it is satisfactory unless the frequency is so far above cutoff that the
> coupling coefficients are extremely large.) P_c in Equation 7 will be less than 0.01 provided that
> cosh (ΔβL_B/4) is greater than 10, or for ΔβL_B = 12 or more."

> "E-plane bends couple TE₁₀ to the TE₁ₙ/TM₁ₙ combination having no E field perpendicular to that in the
> incident TE₁₀ mode; far above cutoff for these modes, we have"

## Unnumbered approximations, p. 159 (PDF p. 7), left column

E-plane bends (coupling TE₁₀ to the TE₁ₙ/TM₁ₙ combination):

```latex
\Delta\beta L_B \cong \frac{L_B\lambda}{b^2}\,\frac{n^2\pi}{4}, \qquad n \text{ odd.}
```

> "Generally, the TE₁ₙ/TM₁ₙ modes with n = 3 and above are much too separated from TE₁₀ to be important.
> H-plane bends couple TE₁₀ to TE_m0 with m even; here"

```latex
\Delta\beta L_B \cong \frac{L_B\lambda}{a^2}\cdot\frac{(m^2-1)\pi}{4}, \qquad m \text{ even.}
```

> "TE_m0 modes with m = 4 or more can be neglected."

*[L1a: for n = 1 the first expression gives ΔβL_B ≅ L_Bλπ/(4b²); with m = 2 the second gives
3πL_Bλ/(4a²). Setting ΔβL_B = 12 gives L_B = 48b²/(πλ) ≈ 15.3 b²/λ and L_B = 16a²/(πλ) ≈ 5.1 a²/λ, which the
paper rounds to eqs. 16a and 16b. This check is the transcriber's arithmetic, not a statement of the paper.]*

## Eq. 16: required bend length (p. 159, left column)

> "From the above approximations for Δβ, and requiring ΔβL_B = 12 for a 90° hyperbolic secant bend with less
> than 1 percent mode conversion, we find"

```latex
L_B \ge 16\,b^2/\lambda_{\min} \quad\text{(for E-plane bends)} \qquad (16a)
```

```latex
L_B \ge 5\,a^2/\lambda_{\min} \quad\text{(for H-plane bends).} \qquad (16b)
```

> "In ordinary waveguides with a = 2b, the required bend lengths are comparable."

> "In the case of "tall guide" with b > a (see Figure 1), however, the difference in required bend lengths
> becomes large. For example, for 90° bends in "tall" WR187 (a = 0.872", b = 1.872") waveguide for the 15.7 to
> 17.7 GHz band, the above considerations require L_B ≥ 5" for H-plane bends, but L_B ≥ 70" for E-plane
> bends." (the last word continues at the top of the middle column)

*[L1b: worked example, arithmetic only (not in the paper): λ_min = c₀/17.7 GHz = 16.94 mm = 0.667 in;
5a²/λ_min = 5(0.872)²/0.667 = 5.70 in; 16b²/λ_min = 16(1.872)²/0.667 = 84.1 in. The paper's 5" and 70" are lower by
12 % and 17 %; with λ at the band centre (16.7 GHz, 0.707 in) one gets 5.4" and 79". The paper does not state which
λ or rounding it used.]* *[L1: arithmetic re-done, same results.]*

Symbols: L_B bend arc length; λ free-space wavelength, λ_min at the highest frequency of operation; a, b the
guide dimensions (a along the TE₁₀ electric-field-free direction, i.e. the broad wall of an ordinary guide; in the
"tall" guide b > a); Δβ the difference of propagation constants between TE₁₀ and the coupled mode.

*[L1a: in this paper's notation a is the side perpendicular to the TE₁₀ electric field; in CONVENTIONS.md the
H-plane bend has local side a in the bend plane, which is the same side. For the tall WR187 guide the paper
writes a = 0.872", b = 1.872", while the Fig. 9 caption writes the guide as 1.872" x 0.872" ID.]*

## Twists, p. 159, middle and right columns

> "To avoid using an unfavorable bend (Figure 8) such as an E-plane bend in "tall" waveguide, we may substitute
> the combination of a favorable bend plus a twist. A nonconstant rate of twist is preferable to the uniform
> twists normally made commercially, except over very narrow bands."

> "In the case of ordinary waveguide with a > b, the closest unwanted mode in a twist is TE₀₁."

> "... a "linearly-tapered" or "triangular" variation in the rate of change of twist angle θ seems to work the
> best, where"

```latex
\frac{d\theta}{dz} =
\begin{cases}
(2\theta_{tot}/L_t)(2z/L_t), & 0 < z < L_t/2\\[2pt]
(2\theta_{tot}/L_t)\big(2(L_t - z)/L_t\big), & L_t/2 < z < L_t .
\end{cases}
\qquad (17)
```

> "Here θ_tot is the total desired twist angle over the twist length L_t. In Figure 9 we compare the
> theoretical predictions for 20" twists in WR187."

*[L1a: the integral of eq. 17 over 0 < z < L_t is θ_tot, as it should be; the peak rate, at z = L_t/2, is
2θ_tot/L_t, twice the rate of a uniform twist.]*

## Eq. 18: twist estimate (p. 159, right column)

> "To get a rough estimate for the required length for a linearly tapered twist, we can use the results of
> Unger¹¹ for the average mode conversion with linearly tapered coupling:"

```latex
P_c \simeq (8c\theta)^2/(\Delta\beta L_t)^4 . \qquad (18)
```

> "If we consider only TE₁₀-to-TE₀₁ coupling,"

```latex
|\Delta\beta L_t| \simeq \frac{L_t\lambda}{a^2}\,\left|\frac{a^2}{b^2} - 1\right|\,\frac{\pi}{4} .
```

> "The coupling coefficient, c, is approximately 8/π² (see reference 2). To ensure less than 1 percent mode
> converted power (P_c < 0.01), we find after using Unger's results that L_t for a 90° linearly-tapered twist
> with a = 2b must be as long as the bend length, L_B, for H-plane hyperbolic secant bends cited earlier in
> Equation 16b, namely, 5a²/λ, where a = 2b. (Note that L_B for linearly tapered bends would usually need to be
> much larger than this, since the coupling coefficients in bends are often several times the TE₁₀-TE₀₁
> coupling coefficient in twists.)"

Symbols: P_c converted power fraction; c coupling coefficient (TE₁₀–TE₀₁, ≈ 8/π²); θ the twist angle (θ_tot);
L_t twist length; Δβ the TE₁₀–TE₀₁ propagation-constant difference.

*[L1a: in eq. 18 the paper writes θ, not θ_tot; from the context it is the total twist angle. The printed
radicals and fraction bars were checked on the image at native resolution; the absolute-value bars in the
|ΔβL_t| expression enclose (a²/b² − 1) only, and a second pair encloses ΔβL_t on the left side.]*

## Figure 9

Digitized by L1a and L1b, merged in `benchmarks/literature/doane1989_fig9.csv` (20" 90° twists in WR187 tall guide;
constant, cosine and triangular rate of twist; theory of the paper's ref. 2).

## Checking record (L1, 2026-11-08)

* Every equation of this file (14, 15, the two unnumbered ΔβL_B approximations, 16a, 16b, 17, 18 and the unnumbered
  |ΔβL_t| relation) was compared symbol by symbol between the L1a and L1b transcriptions and with the page images
  (PDF pp. 6–7). There is no disagreement in any symbol, exponent, subscript or relation sign. Both read `≅` in the
  ΔβL_B relations and `≃` in (18) and in |ΔβL_t|; L1 confirms on the page.
* Typography only: the printed E-plane ΔβL_B relation has no multiplication dot between the two fractions, the
  H-plane one has a centred dot (L1a reproduces this; L1b wrote a dot in both). The meaning is the same.
* Both read the absolute-value bars of |ΔβL_t| as enclosing (a²/b² − 1) only, with a second pair around ΔβL_t;
  confirmed on the page.
* In (18) the printed symbol is θ without subscript; both transcribers and L1 read it as the total twist angle θ_tot
  of (17), from the context.
* Notation: both state that a is the side perpendicular to the TE₁₀ electric field (E parallel to b); in the tall
  guide b > a. The paper writes the tall WR187 as a = 0.872", b = 1.872"; the Fig. 9 caption gives "1.872" x 0.872" ID".
* Arithmetic checks (L1a, L1b, re-done by L1): ΔβL_B = 12 with n = 1 gives L_B = 48b²/(πλ) ≈ 15.3 b²/λ (paper: 16), with
  m = 2 gives L_B = 16a²/(πλ) ≈ 5.1 a²/λ (paper: 5); the integral of (17) over 0 < z < L_t is θ_tot.

## References cited in these passages

* 2: cited for c ≈ 8/π² (reference list of the paper, p. 160; not transcribed here).
* 10: Quine, formulas for coupling coefficients in E- and H-plane bends.
* 11: Unger, linearly tapered coupling.
