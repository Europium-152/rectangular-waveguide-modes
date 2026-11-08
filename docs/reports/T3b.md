# T3b report — twist coupling, independent derivation (for comparison with T3)

Card: `tasks/T3-twist-coupling.md`, independent second derivation. Item 4 follows the coordinator's update
(PR #7, `CONVENTIONS.md` §2). Note: `docs/theory/T3b-twist-coupling.md`. Scripts and their outputs:
`docs/theory/T3b/`. I did not look at T3's worktree or branch.

## What was done

* I derived the coupled-mode equations from the covariant Maxwell equations in helical coordinates (metric
  $g_{ij}$, $\det g = 1$). Moving to contravariant $E^z, H^z$ makes the local-frame equations **exact and linear
  in τ = θ′**: straight guide $-\tau G$ on $E_t$ and $H_t$, with $G = \hat z\times-(u\partial_v-v\partial_u)$
  (box 1; symbolic check).
* I projected on the wrmodes mode fields using only $L^2$ convergence. The result is
  $V' = -i\beta I/Y + \tau KV$ and $I' = -i\beta YV - \tau K^TI$, with
  $K_{pq} = \iint[(\nabla\cdot e_p)(w\cdot e_q) + (r\cdot e_p)\nabla\cdot(\hat z\times e_q)] = \iint e_q\cdot\partial_\theta e_p$
  (boxes 2 and 3).
  * The helicoidal wall carries $E\cdot\hat z\neq0$. This produces a wall term, so the coupling is
    $+\tau\mathcal G^T$ on V, **not** $-\tau\mathcal G$ ($\mathcal G_{pq}=\iint e_p\cdot\partial_\theta e_q$).
  * The symmetric part of $\mathcal G$ is a wall integral that does not vanish for rectangles.
* Closed forms for all TE/TM pairs are in box 2a (potential form) and box 2b (fully explicit rational form).
  Selection rule: $m+m'$ and $n+n'$ both odd.
* Power-wave form (box 4): $d[a^+;a^-]/dz = [[-i\beta+\tau A,\ \tau B],[\tau B,\ i\beta+\tau A]]$, with
  * $A=\tfrac12(K\rho-K^T/\rho)$, antisymmetric;
  * $B=\tfrac12(K\rho+K^T/\rho)$, symmetric;
  * $\rho_{pq}=\sqrt{Y_p}/\sqrt{Y_q}$.
* **Independent validation.** I wrote a spectral Nédélec solver for the exact modes of a uniformly twisted guide
  (anisotropic medium $g^{ij}$, no mode expansion). The coupled-mode eigenvalues with this note's K converge to
  it. The "naive" coupling ($-\mathcal G$ on E and H) and a "swapped" variant do not.
* I covered the square-guide limit, end frames (updated rule), rate profiles, Doane eqs. 17 and 18 (checked
  against the scan, together with Unger eq. 13), and Fig. 9 at first order and in full coupled modes.

## Results (numbers from the scripts' `.out` files)

* $c = K_{10,01} = \mathcal G_{01,10} = 8/\pi^2$ exactly (sympy, symbolic a and b): independent of the aspect
  ratio. Confirmed.
* Generator confirmed. The complex-step rotation derivative agrees with the formula to 4.9e-16, and $G(u,v)=0$
  exactly.
* Square guide, two modes:
  * the rotation is $-c\theta$ in the local frame and $(1-8/\pi^2)\theta$ in the fixed frame, for any profile, with
    no reflection (1.5e-14);
  * 90° gives −72.951° local and +17.049° fixed;
  * TE₀₁ conversion is complete at θ = π³/16 = 111.03°, as in Katsenelenbaum p. 146.
* Square guide, full basis: the angle approaches −72.951° as L grows (−72.945° at 80λ, 2a/λ = 2.5). $c_{\rm eff}$
  of a uniform twist deviates from c as τ² (0.810557 at τa = 0.01 and 0.809300 at τa = 0.1, 2a/λ = 2.5).
* Exact twisted guide (τa = 1). This note's K gives max $|\gamma/\gamma_{\rm exact}-1|$ = 1.45e-3 / 1.24e-3 at
  χ < 12k, and the Richardson extrapolation gives 1.8e-4 / 8.5e-5. Naive: 2.7e-2 / 2.0e-2. Swapped: 7.9e-2 /
  7.1e-2.
* 20-inch 90° WR187 twist (tall), at 15 / 16.5 / 18 GHz:

  | quantity | 15 GHz | 16.5 GHz | 18 GHz |
  | --- | --- | --- | --- |
  | eq. 18, exact Δβ | 0.302 % | 0.465 % | 0.684 % |
  | eq. 18, approximate Δβ | 0.395 % | 0.579 % | 0.820 % |
  | first order, triangular, TE₀₁ | 0.0054 % | 0.0001 % | 0.0452 % |
  | first order, triangular, all modes | 0.048 % | 0.028 % | 0.066 % |

  * The eq. 18 values with exact Δβ and the first-order TE₀₁ values agree with the reviewer's numbers.
  * The exact first-order triangular result is $256c^2\Theta^2\sin^4(\Delta\beta L/4)/(\Delta\beta L)^4$: between
    0 and 4× eq. 18, average 1.5× eq. 18.
* Full coupled modes (all propagating modes) compared with Fig. 9 (rough reading of the scan):
  * The cosine curve is reproduced.
  * The constant curve follows our TE₀₁ conversion, including the deep minimum at 16.5 GHz.
  * The **triangular twist gives 0.11–0.60 %**, mostly TE₁,₂ reached in second order, against ≈0.05–0.08 % in
    Fig. 9.
  * The result is insensitive to evanescent modes (TE₁₂ 0.1685 → 0.1761 % with cutoff_factor 1 → 3) and to
    slicing (0.1685 → 0.1687 %, 150 → 600 slices).

## Acceptance tests

| Check | Target | Result |
| --- | --- | --- |
| Closed forms vs quadrature of $\iint e_p\cdot\partial_\theta e_q$ | 1e-10 relative | **pass**: 1.2e-14 vs complex-step $\partial_\theta$; 2.1e-14 vs box 2 definition (8000 pairs, 5 guides); box 2b 6.9e-16; sympy 3.6e-16 |
| Square guide, two-mode truncation | rotation $-c\theta$ local, exact | **pass**: exact closed form (box 5), transfer matrix 1.5e-14 for 3 profiles; angle from fields −72.951° at 90° |
| First-order integral, WR187 20-inch triangular | same order as Fig. 9, < 0.1 % at 15–18 GHz | **pass**: 0.0054 / 0.0001 / 0.045 % (TE₀₁), 0.048 / 0.028 / 0.066 % (all modes) |
| Zero length: +90° then −90° twist = identity (updated item 4) | identity | **pass**: max $|S-I|$ ≤ 4.4e-15 (square and WR187, 9 and 16.5 GHz) |
| T3 and T3b identical formulas | identical or discrepancy report | **not done here** (comparison is the coordinator's step). Watch for: K vs $-\mathcal G$ (section 3), $K^{\rm TE,TE}$ not antisymmetric, $K_{\rm TE,TM}=0\neq K_{\rm TM,TE}$ |

## Open issues

1. **Fig. 9 triangular curve vs full coupled modes** (0.11–0.60 % vs ≈0.05–0.08 %, mostly TE₁₂). The constant
   and cosine curves agree. This needs COMSOL (WP7 / wave 2): a 20-inch triangular twist in WR187 at 16.5 GHz.
   L1's digitization of Fig. 9 will firm up the comparison. My Fig. 9 values are eye readings.
2. **Zero-length single 90° twist.** The truncated coupled-mode product does not converge to the 90° frame-overlap
   matrix (errors 0.2–0.3 up to 2290 modes). It is not usable as a test; the +90°/−90° identity is.
3. **Slow convergence with strong twists.** The modal coupled-mode error falls only like $1/\chi_{max}$ at
   τa = 1. This is irrelevant for practical twists (τa ≪ 1), but WP6 should not promise fast convergence for
   very short twists.
4. Katsenelenbaum eq. 4.6 agrees with box 4 in magnitude ($|S_{jm}S_{mj}|$). A sign-by-sign comparison would need
   his eqs. 2.51 and 2.71 (not transcribed).

## Conventions issues

* The sign of the twist angle is not stated in `CONVENTIONS.md`. I used θ > 0 = counter-clockwise from û to v̂,
  right-handed about +ẑ (the propagation direction), with the cross-section rotating about its centre. The
  rotation axis matters: Katsenelenbaum p. 140 notes that a vertex-fixed twist has different coefficients.
  Suggest adding both to §2.
* The truncation rule of the updated §3 (live H- and E-frame families) is family-based. A twist couples every
  family, so with `cutoff_factor > 1` it needs evanescent modes of all families. For the Doane twist they matter
  at the 6 % relative level only.
* The T0 power-wave sign pattern ($[[-iC^+,-iC^-],[iC^-,iC^+]]$) does not fit a twist, whose real coefficients
  give $[[A,B],[B,A]]$. This is no conflict; WP6 just must not reuse the bend's assembly.

## For other cards

* WP6:
  * use box 2b and box 4 (`docs/theory/T3b/twist_coupling.py`: `K_explicit`, `K_matrix_fast`, `M_matrix`);
  * do not antisymmetrize K;
  * same global-frame basis at both ports;
  * reference values in sections 6–8 of the note.
* WP7: the Fig. 9 triangular discrepancy (open issue 1).
