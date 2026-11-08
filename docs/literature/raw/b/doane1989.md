# Doane & Anderson (1989): design rules for bends and twists (eqs. 14-18), transcription by L1b

Source: J. L. Doane and T. N. Anderson, "Oversized rectangular waveguides with mode-free bends and twists for
broadband applications", *Microwave Journal* 32(3), 153-160 (1989). The PDF is a photograph of the printed pages
(no text layer); journal page 158 = PDF page 6, page 159 = PDF page 7. Read on 200-300 dpi renders by L1b. Every
symbol below was compared with the page image; uncertain readings are marked.

Notation of the paper: `a` and `b` are the guide sides with TE10 polarized along `b` (E parallel to side `b`); in
ordinary guide `a > b`. In "tall" guide `b > a`
(p. 159: "tall" WR187 has a = 0.872", b = 1.872"). `L_B` = bend arc length, `L_t` = twist length, `λ` = free-space
wavelength, `Δβ` = difference of propagation constants of TE10 and the coupled mode, `c` = coupling coefficient,
`θ` = bend (or twist) angle, `P_c` = converted power fraction.

## Context equations on p. 158 (PDF p. 6), right column

Constant-curvature bend (radius R), quoted from ref. 9 of the paper:

```
P_c = (cθ)^2 (sin u / u)^2                                   (14)
u = [(ΔβR/2)^2 + c^2]^(1/2) θ                                (15)
```

Hyperbolic-secant bend criterion, p. 158: "P_c in Equation 7 will be less than 0.01 provided that cosh (ΔβL_B/4) is
greater than 10, or for ΔβL_B = 12 or more." The bend length is taken as twice that of a constant-radius bend with
R = R_0: "for a 90° bend L_B = πR_0, where R_0 is the minimum bend radius."

## Δβ L_B far above cutoff (p. 158 bottom right, continued p. 159 left column, unnumbered)

E-plane bends ("E-plane bends couple TE10 to the TE1n/TM1n combination having no E field perpendicular to that in
the incident TE10 mode; far above cutoff for these modes, we have"):

```
Δβ L_B ≅ (L_B λ / b^2) · (n^2 π / 4),   n odd
```

"Generally, the TE1n/TM1n modes with n = 3 and above are much too separated from TE10 to be important."

H-plane bends ("H-plane bends couple TE10 to TEm0 with m even; here"):

```
Δβ L_B ≅ (L_B λ / a^2) · ((m^2 − 1) π / 4),   m even
```

"TEm0 modes with m = 4 or more can be neglected."

## Bend lengths (p. 159, left column)

"From the above approximations for Δβ, and requiring ΔβL_B = 12 for a 90° hyperbolic secant bend with less than 1
percent mode conversion, we find"

```
L_B ≥ 16 b^2 / λ_min     (for E-plane bends)      (16a)
L_B ≥  5 a^2 / λ_min     (for H-plane bends)      (16b)
```

Worked example in the text: for 90° bends in "tall" WR187 (a = 0.872", b = 1.872") for the 15.7 to 17.7 GHz band,
"the above considerations require L_B ≥ 5" for H-plane bends, but L_B ≥ 70" for E-plane bends".

L1b check of the example (arithmetic only, not in the paper): λ_min = c0/17.7 GHz = 16.94 mm = 0.667 in;
5 a²/λ_min = 5(0.872)²/0.667 = 5.70 in; 16 b²/λ_min = 16(1.872)²/0.667 = 84.1 in. The paper's 5" and 70" are
lower than these by 12 % and 17 %; with λ at the band centre (16.7 GHz, 0.707 in) one gets 5.4" and 79". The
paper's rounding or exact λ is not stated.

## Twist rate (p. 159, right column)

"a 'linearly-tapered' or 'triangular' variation in the rate of change of twist angle θ seems to work the best,
where"

```
dθ/dz = (2θ_tot/L_t)(2z/L_t),          0 < z < L_t/2
        (2θ_tot/L_t)(2(L_t − z)/L_t),  L_t/2 < z < L_t         (17)
```

"Here θ_tot is the total desired twist angle over the twist length L_t." (The integral of (17) over 0..L_t equals
θ_tot, checked by L1b.)

## Twist conversion estimate (p. 159, right column)

"To get a rough estimate for the required length for a linearly tapered twist, we can use the results of Unger [ref.
11] for the average mode conversion with linearly tapered coupling:"

```
P_c ≃ (8cθ)^2 / (ΔβL_t)^4                                    (18)
```

"If we consider only TE10-to-TE01 coupling,"

```
|ΔβL_t| ≃ (L_t λ / a^2) · | a^2/b^2 − 1 | · (π/4)
```

"The coupling coefficient, c, is approximately 8/π² (see reference 2). To ensure less than 1 percent mode converted
power (P_c < 0.01), we find after using Unger's results that L_t for a 90° linearly-tapered twist with a = 2b must be
as long as the bend length, L_B, for H-plane hyperbolic secant bends cited earlier in Equation 16b, namely, 5a²/λ,
where a = 2b."

Reading notes: in (18) the printed symbol is `θ` without subscript (the twist angle, θ_tot in (17)). The absolute
value bars in the |ΔβL_t| formula enclose `a²/b² − 1`; the printed formula uses `≃`. In the Δβ L_B formulas the
printed relation is `≅`.

## Figure 9 data

Digitized separately: `benchmarks/literature/raw/b/doane1989_fig9.csv` (20" 90° twists in WR187 tall guide,
constant / cosine / triangular rate of twist).
