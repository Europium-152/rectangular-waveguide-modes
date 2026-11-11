# WP4b — Mode matching for 90° miters; parity-reduced steps: report

Branch `wave-2/WP4b-miter-mm`, from `main` at ab3a496. Card `tasks/WP4b-miter-mode-matching.md`. No FEM code, FEM
test, `benchmarks/fem2d/`, WP3 report number or COMSOL result was read; the public interface of `fem2d.Miter` was
taken from its docstring only.

## What was done

| Item | Files |
| --- | --- |
| Scalar 2D corner solver: trigonometric interface bases, rectangle and triangle cell matrices (Neumann-to-Dirichlet, closed forms), networks of the miter (full two-port and bisector halves), resonance guard | `src/wrmodes/modematch/_corner2d.py` |
| `MiterMM(chamfer=None, plane='H', guide=None, n_modes=None, *, angle_deg=90, name='')`: TE/TM GSM from the LS problems, error-controlled default truncation, `convergence_study` | `src/wrmodes/modematch/miter.py`; exported as `wrmodes.modematch.MiterMM` and `wrmodes.MiterMM` |
| `Step` parity reduction (`Step.solve(..., reduce=True)`, `_class_key`) | `src/wrmodes/modematch/step.py` |
| Tests (62 fast, 10 slow) | `tests/test_modematch_miter.py`; `slow` marker registered in `pyproject.toml` |
| Benchmarks (npz with `meta`, CONVENTIONS.md 9) | `benchmarks/modematch/`: `miter_convergence_2d.py`, `miter_convergence_3d.py`, `miter_literature.py`, `fit_chamfer.py`, `fd_hplane_miter.py`, `miter_wr90_sweep.py`, `miter_timing.py`, `step_parity_timing.py`, data in `benchmarks/modematch/data/` |
| Registry | `benchmarks/registry.json`: Alessandri, Alimenti, Kotik entries updated; five `wp4b-*` entries |
| Discrepancy report | `docs/discrepancies/2026-11-11-alimenti-chamfered-miters.md` |

## Method

### Choice

A miter does not vary along v, so each family n is two independent scalar problems in the bend plane
(CONVENTIONS.md 7): LSM, ψ = E_v with ψ = 0 on the walls, and LSE, ψ = H_v with ∂ψ/∂n = 0, both with
∇²ψ + k_t²ψ = 0, k_t² = k² − (nπ/b)². Both conditions are exact on the oblique miter wall (E_v and the in-plane
tangential E ∝ ∂_sψ vanish for LSM; the in-plane tangential E ∝ ∂_nψ vanishes for LSE). The scalar problems are
solved by the segmentation of Alessandri, Mongiardo & Sorrentino (1994, Fig. 1): with the inner corner at the origin,
arm 1 in x ∈ [0, a], y < 0, arm 2 in y ∈ [0, a], x < 0, the outer walls x = a, y = a and the miter x + y = a + c
(c = √2 d − a, 0 ≤ c ≤ a), the corner region is the square C = [0, c]², the rectangles B = [c, a] × [0, c] and
E = [0, c] × [c, a] and the right isosceles triangle D under the miter. For c = 0 (optical mirror) only D remains, for
c = a (unmitered) only C.

Two departures from the paper, both for convergence:

1. **Impedance instead of admittance form.** Alessandri et al. take the tangential E on the interfaces as unknowns
   (cell admittance matrices). For the H-plane problem that is ψ itself, expanded on each interface segment in sines;
   at the points where three interfaces meet ((c, 0), (0, c), (c, c)) the expansion forces ψ = 0 where the true field
   is not zero. In the H^{1/2} norm of the problem a point constraint has zero capacity, so the solution still
   converges, but only logarithmically. Here the unknowns are the normal derivatives g = ∂ψ/∂n on the interfaces
   (cosine-type bases, no constraint at junction points) and each cell is described by its Neumann-to-Dirichlet
   matrix Z[k, l] = ∫ w_k ψ[w_l]. This is the same segmentation with the roles of E and H exchanged for LSM; for LSE
   it is Alessandri's own form.
2. **Bisector symmetry.** The miter is symmetric about y = x, and the reflection maps arm mode m of port 1 onto mode
   m of port 2 with sign +1 (both u′ coordinates run from the inner wall, CONVENTIONS.md 2). The even and odd halves
   (Neumann or Dirichlet on the bisector) give S11 = S22 = (Γ_e + Γ_o)/2, S21 = S12 = (Γ_e − Γ_o)/2. The full
   two-port network is also implemented (`symmetry=False`) and agrees to 1e-15, which checks the frame of port 2.

The half problem has three cells: the half of C (a triangle with its hypotenuse on the bisector), B, and the half of
D (a triangle with its hypotenuse on the interface y = c and its legs on the bisector and the miter).

Kotik & Sivov (1965) is the same method for the optical mirror: the image system of a mirror on the diagonal of the
crossing square is the square with image data used for D here (their Green's-theorem projection on the square is
another projection of the same image problem). It is therefore not an independent third method in this code; the
independent checks are Marcuvitz's corner circuits (phases) and a finite-difference solution (below).

### Cells

**Rectangle** [0, L_x] × [0, L_y], each side a data side (inward normal derivative h given) or a wall ('D' or 'N');
data sides count as homogeneous Neumann for the other sides' sub-problems. The field is the sum of one separable
sub-problem per data side s: ψ = Σ_j c_j X_j(t) Y_j(η), X_j the orthonormal eigenfunctions along s with the boundary
types of the two adjacent sides (cos, cos((j+½)πt/L), sin((j+½)πt/L) or sin), η the distance from s,
γ_j² = λ_j² − k_t², Y_j = cosh γ_j(H − η) (opposite side Neumann) or sinh γ_j(H − η) (Dirichlet), c_j = h_j / Y_j′(0).
In the natural basis of the side the blocks are:

* own side: diag(Y_j(0)/Y_j′(0)) = −coth(γH)/γ or −tanh(γH)/γ;
* opposite (data) side: diag(−1/(γ sinh γH));
* adjacent side s′ meeting s at its end e (end e′ of s′), test functions w_k of s′ with eigenvalues p_k:
  Z[k, j] = X_j(e) ∫ w_k Y_j dt′ / Y_j′(0). If w is the natural basis of s′ (Neumann at e′, and at its far end the
  type of the side opposite s), Green's identity gives the closed form
  **Z[k, j] = −w_k(e′) X_j(e) / (λ_j² + p_k² − k_t²)**, symmetric under the exchange of s and s′ as reciprocity
  requires. Otherwise the integral is evaluated (`ce_int`).

All blocks are real for real k_t² (γ real or imaginary), so the cells are lossless.

**Triangles** by images (Alessandri's Fig. 3). The triangle with data on its legs and a wall on the hypotenuse
(half of C, and D in the full network) is the square with the leg data reflected across the hypotenuse (sign +1 for a
Neumann, −1 for a Dirichlet wall, reversed parameter where the reflection reverses it). The triangle with data on its
hypotenuse and walls on its legs (half of D) is the square with that data on all four sides, reflected across both
legs: Z = Z_BB + σ₁Z_BL + σ₂Z_BR J + σ₁σ₂Z_BT J, J = diag((−1)^k). For the optical mirror this is the closed form
Z_kl = δ_kl[−coth(γ_l a)/γ_l − σ₁σ₂(−1)^l/(γ_l sinh γ_l a)] − (σ₁ + σ₂(−1)^{k+l}) ν_k ν_l / (λ_k² + λ_l² − k_t²).

**The one mismatched side.** For LSM the interface y = c between B and the half of D is natural to D in cosines but to
B in cos((j+½)πt/L) (Dirichlet wall x = a). Its basis is cosines; B's own block on that side is then
Σ_j Π_kj Π_lj ρ_j over J = 8(K + k_t L/π) eigenfunctions plus the exact tail of the leading terms
(−1/μ_j − k_t²/2μ_j³ with Π_kj Π_lj = ν_kν_l(2/L)(−1)^{k+l}μ_j²/((μ_j² − p_k²)(μ_j² − p_l²)), summed with
polygamma functions; error 1e-8 at J = 64 against 1e-5 without the tail). All its other blocks follow from
reciprocity.

### Arms and the network

In each arm ψ = Σ_m (α_m e^{−iβζ} + ω_m e^{iβζ}) φ_m/√β_m, ζ towards the corner, φ_m the orthonormal sin (LSM) or cos
(LSE) modes, β with the branch of CONVENTIONS.md 3. Continuity of ∂ψ/∂ζ (projected on φ_m) and of ψ (tested with the
interface functions) give (Z + iAᵀA)G = 2Aᵀα, ω = α − iAG with A = diag(β^{−1/2})P, P_mk = ⟨φ_m, w_k⟩:

  **S = I − 2iA(Z + iAᵀA)⁻¹Aᵀ**,

symmetric, and over the propagating modes S_pp = (I − iW)(I + iW)⁻¹ with W real symmetric, i.e. exactly unitary at
every truncation (tests: 1e-15).

### LS amplitudes and the TE/TM GSM

For a live family (n = 0 with k > π/a, or n ≥ 1 with k_t > 0) the 3D LS amplitudes of `ls_rotation` are the 2D
amplitudes up to one factor per family and type: a unit LSM amplitude has H_v = 0 and
E_v = C_n √(2/a) sin(mπu′/a) cos(nπv′/b)/√β, a unit LSE amplitude E_v = 0 and
H_v = C′_n √(ε_m/a) cos(mπu′/a) sin(nπv′/b)/√β, with C_n > 0 and C′_n < 0 independent of m, evanescent modes included
(derived, and checked numerically from `transverse_field`: spread 6e-14). The factors cancel in S, but one sign does
not: H_v is a transverse magnetic component, which reverses for backward waves (CONVENTIONS.md 5), so

  **S_LSE,11 = −S_2D,11, S_LSE,22 = −S_2D,22, S_LSE,21 = S_2D,21** (a short circuit: S = −1, ∂H_v/∂z = 0).

This sign is invisible to every magnitude, reciprocity and unitarity check; it was found by the phase comparison with
Marcuvitz's E-plane corner (below), where without it S11 came out exactly opposite. Then S_TE/TM = QᵀS_LS Q per port
(Q from `ls_rotation`). Since Q is orthogonal on each propagating TE/TM pair, every TE/TM entry error is bounded by
the largest LS entry error. Families without a propagating mode are uncoupled straight modes over the centre-line
length a (CONVENTIONS.md 3). `angle_deg=-90` returns D S D, D = (−1)^{m+1}.

Interface (as `fem2d.Miter` documents it): reference planes through the inner corner, `arm_length` = a/2, port 2 frame
= port 1 frame rotated about v, bases tagged `plane`, `default_basis(f, cutoff_factor=None)` = the guide's line rule
with cutoff factor 1, `with_guide`, lossy guides refused (`require_lossless`). `chamfer` is d/a; d ≥ √2 is the
unmitered corner; d < 1/√2 (mirror cutting the arms before the reference planes) raises (open issue 2).

### Cell resonances

Each cell matrix has poles at the cell's eigenvalues (k_t² = λ_x² + λ_y²); they cancel in the network, at a loss of
accuracy of about ε/δ^{1.3} at relative distance δ. Two measures keep the error below 5e-8 at any frequency (test with
exact hits): the closed form above is used only farther than 1e-4 (relative) from a cell eigenvalue (nearer, its pole
is rounded independently of the self blocks and the cancellation fails; the integral form shares γ with them), and
within 1e-9 the result is the mean of the solutions at k_t²(1 ± 1e-8) (`RESONANCE_GUARD`). γ = 0 exactly is
evaluated at |γ|H = 1e-7 (removable for a Dirichlet opposite side).

## Convergence and the default truncation

Truncation of a scalar problem: M arm modes; ⌈M/R⌉ interface functions per width a, in proportion to the segment
lengths. The errors of the two truncations have opposite signs and nearly cancel at R = 1.15 (LSM) and 0.7 (LSE),
found by scanning R with references extrapolated along four ratios that agree to 6e-8–3e-7. The fitted order is
**p = 4/3** (1.33–1.40 in the clean cases), the exponent of the r^{2/3} field at the 270° inner corner.

Scalar study (`miter_convergence_2d.py`; error = max |S − S_ref| over the propagating block, S_ref = Aitken
extrapolation of M = 250, 500, 1000, whose own error is below 1e-5):

| wall | d/a | k_t a/π | M = 60 | 100 | 160 | 240 |
| --- | --- | --- | --- | --- | --- | --- |
| D (R 1.15) | 0.707 | 4.3 / 13.7 | 8.1e-5 / 3.9e-4 | 2.6e-5 / 1.0e-4 | 1.6e-5 / 3.8e-5 | 8.2e-6 / 1.7e-5 |
| D | 0.87 | 4.3 / 13.7 | 1.4e-4 / 3.5e-4 | 5.7e-5 / 8.3e-5 | 2.6e-5 / 3.0e-5 | 1.4e-5 / 1.4e-5 |
| D | 1.0 | 4.3 / 13.7 | 4.6e-5 / 2.8e-4 | 2.0e-5 / 7.3e-5 | 6.5e-6 / 2.2e-5 | 3.5e-6 / 1.1e-5 |
| D | 1.414 | 4.3 / 13.7 | 5.1e-5 / 2.9e-4 | 1.6e-5 / 7.8e-5 | 8.8e-6 / 3.3e-5 | 4.4e-6 / 1.5e-5 |
| N (R 0.7) | 0.707 | 4.3 / 13.7 | 1.0e-4 / 1.9e-4 | 5.5e-5 / 1.0e-4 | 2.9e-5 / 5.4e-5 | ≈1.4e-5 / 2.8e-5 (M 260) |
| N | 0.87 | 4.3 / 13.7 | 1.4e-4 / 2.4e-4 | 7.2e-5 / 1.3e-4 | 3.6e-5 / 6.4e-5 | ≈1.9e-5 / 3.3e-5 (M 260) |
| N | 1.0 | 4.3 / 13.7 | 1.1e-4 / 2.0e-4 | 5.6e-5 / 9.7e-5 | 3.3e-5 / 5.6e-5 | ≈1.6e-5 / 2.8e-5 (M 260) |

(full tables: d/a = 0.707, 0.80, 0.87, 0.95, 1.0, √2; k_t a/π = 1.3–13.7, LSE also 0.5; in the npz files). The
chamfered cases converge less regularly (fitted order 0.5–1.1 at d/a = 0.80 and 0.95).

**Default** (`MiterMM.default_n_modes`): M = ⌈6x + 100⌉ (LSM), ⌈12x + 70⌉ (LSE), x = k_t a/π, which puts the
scalar error below about 6e-5 in this study, **with error control**: every scalar problem is also solved at M/1.5
and M/2.25 (about 40 % extra), the error is estimated from each pair with the order 4/3,
|S(M) − S(M′)| / ((M/M′)^{4/3} − 1), the larger estimate is taken, and M is raised until it is below `ERR_TARGET`
= 5e-5 (at most 8 times the rule; a warning otherwise). The truncation used is returned in `gsm(...).truncation`;
an integer `n_modes` fixes M (no control).

The control is needed because the chamfered corner has sharp resonances just below arm cutoffs. Example: E plane,
d/a = 1.0, WR-90 at 30 GHz, family n = 1 at k_t a/π = 1.98: |S21| of the LSM m = 1 mode falls from 0.99 to 0.44
within 0.02 in k_t a/π, so an error in the resonance position is multiplied by |dS/dx| ≈ 100; the rule alone gave
3.4e-4, the control raises M to 896 and the error to 4.5e-5. Before the asymptotic range the convergence there is
erratic (E plane d/a = 0.95, 55 GHz, n = 5 at k_t a/π = 2.993: 1.6e-3 at M = 53, 2.7e-4 at 79, 2.9e-4 at 118, then
smoothly 1.0e-4, 8.5e-5, 5.1e-5 at 177, 266, 399), and an estimate from a single pair can be fooled: a first version
(M/2 only) missed H plane d/a = 0.95, 30 GHz, n = 1 at k_t a/π = 3.984 (1.1e-4), a second (M/1.5 only) the case above (1.9e-4); with both pairs no case of the 3D table
exceeds 5e-5. A scan of chamfered LSM problems around the m = 2 and 3 cutoffs (x = m(1 + δ), δ = −3 % … +1 %,
c = 0.23 a and 0.414 a, against M = 1500) found true errors up to 5.8e-5 with the single-pair estimate.

3D check (`miter_convergence_3d.py`, WR-90, propagating block of the full GSM, against a reference with
`ERR_TARGET` = 1.5e-5):

Entry: max |S_default − S_tight| over the propagating block (largest M used by the default in brackets).

| plane | d/a | 12 GHz | 30 GHz | 55 GHz | 70 GHz | 90 GHz |
| --- | --- | --- | --- | --- | --- | --- |
| H | 0.707 (mirror) | 0.0e+00 (111) | 2.2e-05 (128) | 2.4e-05 (227) | 2.6e-05 (248) | 2.4e-05 (276) |
| H | 0.800 | 0.0e+00 (111) | 4.4e-05 (208) | 2.9e-05 (364) | 4.4e-05 (338) | 3.3e-05 (437) |
| H | 0.870 | 0.0e+00 (111) | 2.1e-05 (192) | 3.9e-05 (400) | 4.1e-05 (245) | 2.5e-05 (409) |
| H | 0.950 | 0.0e+00 (111) | 1.0e-05 (496) | 2.7e-05 (251) | 2.5e-05 (255) | 2.6e-05 (300) |
| H | 1.000 | 0.0e+00 (111) | 1.5e-05 (378) | 3.7e-05 (286) | 3.0e-05 (245) | 2.4e-05 (312) |
| H | 1.414 | 2.8e-05 (111) | 2.1e-05 (200) | 3.2e-05 (227) | 3.2e-05 (245) | 2.5e-05 (275) |
| E | 0.707 (mirror) | 9.1e-06 (105) | 1.9e-05 (113) | 3.0e-05 (177) | 3.6e-05 (189) | 3.4e-05 (213) |
| E | 0.800 | 2.1e-05 (105) | 2.3e-05 (170) | 4.4e-05 (231) | 3.7e-05 (245) | 4.9e-05 (491) |
| E | 0.870 | 7.5e-06 (105) | 1.1e-05 (318) | 4.0e-05 (287) | 3.1e-05 (297) | 3.8e-05 (239) |
| E | 0.950 | 1.1e-05 (105) | 1.9e-05 (269) | 3.5e-05 (932) | 3.5e-05 (214) | 4.4e-05 (309) |
| E | 1.000 | 1.0e-05 (105) | 4.5e-05 (896) | 3.5e-05 (333) | 3.7e-05 (299) | 3.9e-05 (491) |
| E | 1.414 | 9.2e-06 (105) | 2.2e-05 (168) | 3.1e-05 (281) | 3.5e-05 (186) | 3.4e-05 (247) |

Largest difference 4.9e-5; largest error estimate of the defaults 5.0e-5 (the target). The reference itself is
within about 1.5e-5.

## Literature

All values with `MiterMM` (default truncation) in a guide of the stated aspect ratio, at the digitized abscissae
(`miter_literature.py`; z = (ours − digitized)/uncertainty; for markers the x reading error is added through the
local slope).

| Figure | curve | n | rms z | share \|z\| ≤ 1 | share \|z\| ≤ 2 | max \|z\| (at x) | mean diff (dB; Kotik: \|S21\|²) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Alessandri 1994 Fig. 4 | d/b=1.41 | 191 | 0.29 | 100 % | 100 % | 0.8 (1.025) | +0.057 |
| Alessandri 1994 Fig. 4 | d/b=0.707 | 139 | 0.22 | 100 % | 100 % | 0.5 (1.155) | -0.031 |
| Alessandri 1994 Fig. 4 | d/b=0.87 | 38 | 0.89 | 71 % | 100 % | 1.3 (1.615) | +0.140 |
| Alessandri 1994 Fig. 4 (ref. [7]) | ref7 d/b=1.41 | 13 | 0.36 | 100 % | 100 % | 0.7 (1.266) | +0.068 |
| Alessandri 1994 Fig. 4 (ref. [7]) | ref7 d/b=0.707 | 13 | 0.67 | 85 % | 100 % | 1.2 (1.265) | -0.149 |
| Alessandri 1994 Fig. 4 (ref. [7]) | ref7 d/b=0.87 | 12 | 8.92 | 8 % | 8 % | 14.3 (1.738) | +0.375 |
| Alessandri 1994 Fig. 5 | d/b=0.9 | 192 | 0.43 | 100 % | 100 % | 0.8 (1.030) | +0.118 |
| Alessandri 1994 Fig. 5 | d/b=0.87 | 189 | 0.87 | 79 % | 99 % | 2.0 (1.815) | +0.230 |
| Alessandri 1994 Fig. 5 | d/b=0.85 | 184 | 1.54 | 43 % | 97 % | 15.0 (1.650) | +0.131 |
| Alessandri 1994 Fig. 5 | d/b=0.83 | 171 | 2.13 | 54 % | 63 % | 5.9 (1.275) | +0.081 |
| Alimenti 1994 Fig. 2 (unmitered) | MMFD | 151 | 0.76 | 76 % | 100 % | 1.7 (1.770) | -0.008 |
| Alimenti 1994 Fig. 2 (unmitered) | MM markers | 10 | 1.12 | 40 % | 100 % | 1.9 (1.793) | +0.003 |
| Alimenti 1994 Fig. 3 (optical mirror) | MM | 142 | 0.72 | 85 % | 99 % | 2.1 (1.505) | -0.233 |
| Alimenti 1994 Fig. 3 (optical mirror) | MMFD markers | 10 | 0.96 | 70 % | 100 % | 1.5 (1.487) | -0.227 |
| Alimenti 1994 Fig. 4 | t/a=0.30 | 150 | 5.11 | 5 % | 15 % | 11.4 (1.300) | +0.940 |
| Alimenti 1994 Fig. 4 | t/a=0.34 | 147 | 7.56 | 13 % | 29 % | 45.3 (1.470) | +0.627 |
| Alimenti 1994 Fig. 4 | t/a=0.38 | 83 | 12.39 | 13 % | 31 % | 52.1 (1.635) | -0.973 |
| Alimenti 1994 Fig. 6 | MMFD | 118 | 34.55 | 7 % | 19 % | 206.7 (1.665) | -1.975 |
| Alimenti 1994 Fig. 6 | BCMM | 99 | 13.49 | 6 % | 18 % | 33.1 (1.885) | -0.659 |
| Alimenti 1994 Fig. 6 | BCMM markers | 10 | 13.36 | 10 % | 10 % | 24.2 (1.794) | -0.470 |
| Kotik 1965 Fig. 3 | rigorous | 416 | 1.30 | 81 % | 96 % | 13.2 (12.600) | -0.003 |
| Ma 1997 Fig. 11 | c/a=0 | 155 | 0.19 | 100 % | 100 % | 0.6 (1.025) | -0.024 |
| Ma 1997 Fig. 11 | c/a=0.60 | 189 | 31.48 | 2 % | 4 % | 149.2 (1.755) | -6.449 |
| Ma 1997 Fig. 11 | c/a=0.631 | 181 | 19.06 | 2 % | 4 % | 83.2 (1.585) | +0.715 |
| Ma 1997 Fig. 11 | c/a=0.70 | 188 | 16.79 | 0 % | 1 % | 80.5 (1.300) | -0.108 |
| Ma 1997 Fig. 11 | c/a=1.0 | 181 | 1.66 | 25 % | 73 % | 2.6 (1.375) | +0.403 |

* **Alessandri, b/a.** The paper does not give the broad side. b/a = 0.5 reproduces every curve; with WR-90's 0.444
  the unmitered and mirror curves are off by 5–7 uncertainties (rms z 6.7 and 5.4) and the chamfered ones by 5–25.
  The paper used b/a = 0.5 (e.g. WR-28 or a 2:1 guide).
* **Alessandri outliers**: d/b = 0.85 and 0.83 near their zeros (|S11| below −38 dB): up to 8 dB at the zero of
  0.85 (f/fc 1.650: −57.4 vs −49.1 dB) and +1 to +3 dB along the flat bottom of 0.83, i.e. at most 2.2e-3 in |S11|;
  the zeros agree within about 0.005 in f/fc, which a 0.3 % change of d or b/a accounts for. The crosses of their
  ref. [7] (Reiter & Arndt, BCMM) for d/b = 0.87 lie 1.7–5.8 dB below both Alessandri's line and ours.
* **Kotik outliers**: just around the TE40, TE60, TE80 cutoffs (kd − nπ ∈ [−0.1, 0.4]), where our dips are deeper
  and sharper (e.g. 0.665 vs 0.758 at kd = 12.6) than Kotik's N ≤ 18 solution as drawn ("no drastic changes" near
  the critical frequencies, his text), and above kd = 30.4, where we are 0.010–0.011 lower (0.960 vs 0.971; his text:
  "3 % loss at kd ≈ 30").
* **Alimenti Figs. 2 and 3** (unmitered, optical mirror) and **Ma Fig. 11 c/a = 0** agree. Ma c/a = 1.0 lies 0.40 dB
  below us (rms z 1.66) and Alimenti Fig. 3 0.23 dB above: L2's 0.68 dB difference between the papers is split.
  At Ma's Table 1 point (optical mirror, f/fc = 1.686) MiterMM and FD give |S11| = 0.2779 / 0.2778 against Ma's
  0.2622 (Table 1, "Ref.", 60 steps).
* **Alimenti Figs. 4–6 and Ma Fig. 11, chamfered**: not reproduced. See
  `docs/discrepancies/2026-11-11-alimenti-chamfered-miters.md`: an independent FD solution agrees with MiterMM
  (|S11| = 0.037100 vs 0.037103 at f/fc = 1.5 and 0.013867 vs 0.013867 at 1.69 for t/a = 0.38), Ma's curves are reproduced at t/a + 0.016…0.020 (rms z 0.4–1.6), the BCMM line of Fig. 6 at
  t/a + 0.005 (rms z 4.3), Alimenti's MMFD curves at no single t/a.

### Phase check: Marcuvitz's right-angle corners

Marcuvitz Secs. 5.28a and 5.29a (eqs. (3a), (4a), L1's transcription) give the unmitered corner as a π network
(shunt jB_a at each plane, series −jB_b (E) or jB_b (H)) at planes T′ through the inner corner, i.e. our reference
planes. S11 = (Γ_e + Γ_o)/2, S21 = (Γ_e − Γ_o)/2 with Γ_e = (1 − jB_a)/(1 + jB_a), Γ_o for jB_a + 2(series):

| plane | f (GHz) | 2b/λ_g or 2a/λ_g | S11 MiterMM | S11 Marcuvitz | S21 MiterMM | S21 Marcuvitz | ΔS11 | ΔS21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 7.5 | 0.555 | −0.4587+0.0685i | −0.4582+0.0681i | −0.1309−0.8762i | −0.1302−0.8766i | 6.8e-4 | 7.9e-4 |
| H | 8.5 | 0.825 | −0.2870+0.2470i | −0.2873+0.2470i | −0.6038−0.7015i | −0.6033−0.7018i | 3.2e-4 | 5.9e-4 |
| H | 10.0 | 1.151 | 0.0474+0.4298i | 0.0468+0.4302i | −0.8963+0.0988i | −0.8962+0.0976i | 6.7e-4 | 1.2e-3 |
| H | 11.0 | 1.347 (outside his 1 % range) | 0.5323+0.3410i | 0.5267+0.3424i | −0.4180+0.6524i | −0.4241+0.6523i | 5.7e-3 | 6.0e-3 |
| E | 7.5 | 0.247 | 0.1069+0.1482i | 0.1069+0.1480i | 0.7973−0.5753i | 0.7972−0.5754i | 1.2e-4 | 1.9e-4 |
| E | 8.5 | 0.367 | 0.2434+0.1665i | 0.2432+0.1662i | 0.5396−0.7886i | 0.5391−0.7890i | 4.1e-4 | 6.6e-4 |
| E | 10.0 | 0.512 | 0.4746+0.0446i | 0.4740+0.0436i | 0.0822−0.8752i | 0.0805−0.8757i | 1.1e-3 | 1.7e-3 |
| E | 11.0 | 0.599 | 0.5931−0.1748i | 0.5918−0.1764i | −0.2222−0.7538i | −0.2247−0.7538i | 2.0e-3 | 2.5e-3 |

Within Marcuvitz's stated accuracy (about 1 % on the susceptances), phases included: this fixes the reference planes,
the port-2 frame (sign of S21) and the LSE sign.

### Independent finite differences

`fd_hplane_miter.py`: 5-point stencil, h = a/N, exact discrete modal ports, Richardson from N = 200, 400.

| t/a (d/a) | f/fc | FD N = 50 / 100 / 200 / 400 | FD extrapolated (order) | MiterMM 2D (M = 400) |
| --- | --- | --- | --- | --- |
| 0.38 (0.976) | 1.5000 | 0.036498 / 0.036910 / 0.037038 / 0.037080 | 0.037100 (1.61) | 0.037103 |
| 0.38 (0.976) | 1.6360 | 0.000378 / 0.000085 / 0.000012 / 0.000007 | 0.000007 (4.01) | 0.000014 |
| 0.38 (0.976) | 1.6900 | 0.014285 / 0.013971 / 0.013893 / 0.013874 | 0.013867 (2.01) | 0.013867 |
| 0.0 (0.707) | 1.6863 | 0.280907 / 0.279072 / 0.278311 / 0.278000 | 0.277786 (1.29) | 0.277793 |

At the zero (f/fc = 1.636) both are at the 1e-5 level. The FD order of 1.3-1.6 is the corner singularities'.

## Acceptance

| Check | Target | Result |
| --- | --- | --- |
| Alessandri Figs. 4–5 (E plane) | digitizing uncertainty; state b/a | **b/a = 0.5**: rms z 0.22–0.89 for d/b = 1.41, 0.707, 0.87, 0.9 (100 % within 2u except one point at 2.0u); d/b = 0.85 / 0.83 rms z 1.55 / 2.14, outliers at their zeros (≤ 2.2e-3 in \|S11\|) |
| Alimenti Figs. 2–3 (H) | digitizing uncertainty | rms z 0.76 (Fig. 2), 0.71 (Fig. 3); markers 1.12, 0.96 |
| Alimenti Figs. 4–6 (H, chamfered) | digitizing uncertainty | **not met**: rms z 5–35; discrepancy report (FD sides with MiterMM) |
| Kotik Fig. 3 (H, optical mirror, overmoded) | digitizing uncertainty | rms z 1.30 over 416 points, 96 % within 2u; outliers next to the TE40/60/80 cutoffs and kd > 30.4 |
| Reciprocity; unitarity; S11 = S22 | CONVENTIONS.md 6 (1e-10) | ≤ 1e-16 (relative), ≤ 5e-15; S11 = S22 to 4e-13 from the full two-port network (not by construction) |
| Convergence in the mode counts | reported; defaults reach 1e-4 in every propagating entry | 2D and 3D tables above; default error ≤ 4.9e-5 (60 WR-90 cases, H and E, 6 chamfers, 12-90 GHz) against the tight reference |
| FEM agreement | WP7b | not run (data: `benchmarks/modematch/data/miter_wr90_sweep.npz`) |
| Parity-reduced Step | equal to the full solve to 1e-12 | ≤ 1.4e-14 (8 geometries); speed-ups below |
| Performance | WR-90 H-plane plain miter, 90 GHz, full default basis, < 1 s | 0.66 s (129 port modes) |

## Timings

`miter_timing.py`, seconds per frequency (median of 3), default truncation, this container (2 cores, nothing else
running):

| plane | d/a | 30 GHz (12 port modes) | 60 GHz (57 port modes) | 90 GHz (129 port modes) |
| --- | --- | --- | --- | --- |
| H | 0.707 (mirror) | 0.06 | 0.25 | 0.66 |
| H | 0.870 | 0.26 | 2.23 | 3.42 |
| H | 1.000 | 0.34 | 0.94 | 2.07 |
| H | 1.414 | 0.07 | 0.25 | 0.57 |
| E | 0.707 (mirror) | 0.08 | 0.29 | 0.78 |
| E | 0.870 | 0.61 | 1.30 | 2.93 |
| E | 1.000 | 2.78 | 1.77 | 2.98 |
| E | 1.414 | 0.10 | 0.38 | 0.54 |

The error control costs about 40 % (the two coarser solves) plus the refinements; the chamfered cases are
slower (three cells, the J-sum of the mismatched side, and refinements near resonances).

## Step parity reduction

For a centred step (d_u = 0, resp. d_v = 0) the fields of both guides have the parity (−1)^{m+1} under u → −u (resp.
(−1)^{n+1} under v → −v), so the overlap matrix is block diagonal in the parity classes; a step that keeps a with
d_u = 0 conserves m itself, one that keeps b conserves n. `Step.solve` now solves each class with `junction_gsm`
separately (modes of a class seen by no mode on the other side: S = −I on the large side, +I on the small side, as
in the full formula). `Step.gsm` uses it by default; `reduce=False` is the full solve. Same GSM to ≤ 1.4e-14
(centred 1D and 2D steps, offsets in u or v only, both offsets: no reduction).

`step_parity_timing.py` (best of 2-3 runs, default truncation):

| case | modes (port 1, port 2) | classes | full solve (s) | reduced (s) | speed-up | max \|ΔS\| |
| --- | --- | --- | --- | --- | --- | --- |
| WR-90 -> 22.86 x 1.6 mm, 90 GHz | 3415, 526 | 28 | 1.607 | 0.064 | 25.0 | 1.4e-14 |
| 22.86 x 1.6 mm -> WR-90, 90 GHz | 526, 3415 | 28 | 1.646 | 0.071 | 23.1 | 1.4e-14 |
| 2D: WR-90 -> 18 x 8 mm centred, 30 GHz | 1463, 899 | 4 | 0.822 | 0.104 | 7.9 | 6.5e-15 |
| 2D: WR-90 -> 18 x 8 mm centred, 60 GHz | 1463, 899 | 4 | 0.993 | 0.136 | 7.3 | 6.5e-15 |
| 2D: 12 x 6 mm -> WR-90 centred, 25 GHz | 796, 2595 | 4 | 1.779 | 0.186 | 9.5 | 9.3e-15 |

The 1D step conserves m (28 classes: 14 values of m times 2 parities of n); the 2D steps have the 4 parity
classes, about 1/16 of the solve and 1/4 of the overlaps.

## Open issues

1. **Chamfered H-plane literature** (Alimenti Figs. 4–6, Ma Fig. 11): discrepancy report; Daniel / a theory agent
   should decide how WP3b and WP7b use these figures. Related: registry `wp3-ma1997-table1` (0.57 % on 0.2622) is 6 %
   from MiterMM and FD; `tests/adversarial/test_wp3_adversarial.py::test_h_miter_against_ma1997_table1` fails on main
   as well (with the three `test_arm_length_is_half_a_tan_half_angle` cases and one more WP3 adversarial test: the
   same 5 failures on main and on this branch; WP4b did not look into them).
2. **d < 1/√2** (a miter cutting the arms before the reference planes) is refused. It would need the analytic
   continuation of CONVENTIONS.md 6 (a shorter arm, then a uniform piece), straightforward with this machinery.
3. **Order 4/3.** The inner-corner singularity limits convergence to M^{-4/3}; 1e-4 needs M ≈ 100–300 per scalar
   problem. Singular interface functions near the inner corner (or extrapolation of the reduced reactance matrix
   W, which keeps unitarity) would cut the cost several-fold.
4. **Resonances below cutoff.** Near the sharp corner resonances the error control raises M (up to 8× the rule;
   warning beyond). The estimate assumes the asymptotic order; single-pair estimates were fooled twice, the
   two-pair estimate holds on the 60 WR-90 cases of the 3D table, but it remains an estimate, not a bound. The cost
   near such resonances is up to about 5 s per frequency (E plane, d/a = 0.95, 55 GHz: M = 932).
5. **Dead families** (no propagating mode) are uncoupled straight modes over the centre line (CONVENTIONS.md 3
   allows it). Their evanescent entries are approximate; a line needs the usual distance to the next component.
6. **No wall loss** (as `fem2d.Miter` in wave 2): a lossy guide raises.
7. **Kotik & Sivov is not an independent method here** (the image square is the same construction); the card lists
   it as a possible third method. FD and Marcuvitz serve as the independent checks.

## For other cards and the coordinator

* **WP7b**: `miter_wr90_sweep.npz` holds, for H and E plane, d/a = mirror, 0.80, 0.87, 0.90, 0.95, 1.0, WR-90,
  10–90 GHz in 2 GHz steps (frequencies within 0.2 % of a cutoff skipped, listed in `meta`), the TE10 excitation
  column of the propagating GSM (reflected and transmitted amplitudes of every propagating mode: what the card's
  0.01 dB / 1e-4 criteria need) and the full propagating GSMs at 20 and 40 GHz (190 kB in all; rerun the script for
  full GSMs elsewhere, about 15 min). E-plane modes are in the local frame (global TE10 = −local TE0,1).
* **WP3b**: MiterMM is a drop-in for `fem2d.Miter(90, ...)` in a `Line` (same interface, also `angle_deg=-90`).
* **WP5**: `Step` is now parity-reduced by default.
* **CONVENTIONS.md (proposals, not changed)**: (a) §7 could state that LS solvers built on ψ = H_v must negate the
  LSE reflections (the H_v amplitude convention differs from the voltage convention of §5); (b) §3 "dead families"
  could prefer a rigorous solve when the family's k_t² is not near 0 (the LS rotation is fine there).
