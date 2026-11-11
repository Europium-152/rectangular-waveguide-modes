# Discrepancy: attenuation of lossy straights, WP2 implementation vs WP7 adversarial reference

*Opened by WP2 on 2026-11-08. To be settled by a theory agent (T1) and Daniel, not by WP2.*

## Observation

Two of the eight WP7 adversarial tests of WP2 (`tests/adversarial/test_wp2_adversarial.py` on branch
`wave-1/WP7-verification`) fail:

| test | case | tolerance | WP2 deviation | worst group |
| --- | --- | --- | --- | --- |
| `test_20m_lossy_straight_at_90GHz` | WR-90, aluminium, 20 m, 90 GHz | 1e-6 | 3.6e-3 | TE2,6/TM2,6 (fc/f = 0.994) |
| `test_square_guide_degenerate_groups_in_a_lossy_straight` | 20 mm square, aluminium, 1 m, 30 GHz | 1e-6 | 2.6e-2 | TE0,4/TE4,0 (fc/f = 0.9993) |

The tests compare the decay rates -ln(sv)/L of the degenerate groups of S21 with the eigenvalues of the WP7
wall-integral matrix A restricted to each group. The per-mode attenuation constants agree (WP7
`compare_wp2_loss.py`: max relative deviation 2.0e-12 for all modes, WR-90 and square guide, aluminium and copper),
and so does the loss matrix itself.

## Cause

The WP7 reference assumes that each group decays with its own block of Lambda^+ only. WP2 implements
docs/theory/T1-wall-impedance.md, boxes B4 and section 5: all propagating modes coupled through the full
Lambda_full = [[Lambda^+, Lambda^-], [-Lambda^-, -Lambda^+]]. Two effects separate the two (numbers from the WP2
code, scratch checks reproduced in tests/test_loss.py):

1. **Coupling between groups** (forward, Lambda^+ off the group blocks). Non-synchronous, but summed over the many
   propagating modes it shifts the decay rates. Removing Lambda^- and keeping the full Lambda^+ leaves a deviation
   of 3.3e-3 (TE0,1, WR-90, 90 GHz) and 8.9e-4 (TE0,2/TE2,0, square guide). With group-block Lambda^+ only, the WP2
   machinery reproduces the WP7 reference to 9e-12 and 4e-13. The shift is physical: with one pair of lossy walls,
   where the exact Leontovich modes are known (T1 box B6), the coupled eigenvalue is within 3.5e-4 of the exact
   attenuation, while the diagonal (Marcuvitz) value is off by 3.8e-3 (TE0,1, side walls, 90 GHz) and 1.7e-3
   (TE1,0, broad walls, 90 GHz) (`test_coupled_attenuation_approaches_the_exact_leontovich_value`). T1 section 5
   calls it "a genuine O(ka Z/eta0) correction of the overmoded guide".
2. **Forward-backward coupling Lambda^-** near cutoff. For one mode the 2 x 2 system has
   lambda^2 = (i beta + Lambda^+)^2 - (Lambda^-)^2 = -beta^2 + i beta (Y N + G/Y) + N G, i.e. exactly the transmission
   line with series impedance i beta/Y + N and shunt admittance i beta Y + G (T1 section 3). Near cutoff
   Lambda^- ~ Lambda^+ ~ beta/40 and the decay differs from Re Lambda^+: TE0,4 of the square guide at 30 GHz
   (beta = 23.4 rad/m, Lambda^+ = Lambda^- = 0.62 (1 + i) 1/m) decays at 0.606 Np/m instead of 0.622 (-2.6 %).

## Proposal

Keep WP2 as specified by T1. Change the two WP7 tests to compare with the exact exponential of
(D - Lambda_full) built from the WP7 wall-integral matrix (or restrict the group test to modes with fc/f < 0.9 and
drop the inter-group terms consciously, with a tolerance of about 5e-3). The coordinator and T1 decide.

## Resolution (WP7, 2026-11-08; for Daniel's sign-off)

WP7 did not read WP2's code; it read this file and `docs/theory/T1-wall-impedance.md`. Every number below comes
from `benchmarks/validation_v1/ref_loss.py` on branch `wave-1/WP7-verification`.

**1. Independent derivation of the first-order system.** I started from the unconjugated reciprocity theorem
between the lossy field and an ideal (PEC) test mode: d/dz ∬(E×H_q − E_q×H)·ẑ = −∮(E×H_q)·n̂. With Leontovich,
E_tan = Z H_tan×n̂, the right-hand side becomes −Z∮H_tan·H_q,tan. Replacing the wall field by the modal series and
testing with the backward and the forward mode q gives

* da⁺/dz = −iβa⁺ − Λ⁺a⁺ − Λ⁻a⁻ and da⁻/dz = +iβa⁻ + Λ⁻a⁺ + Λ⁺a⁻;
* Λ⁺ = P − Q and Λ⁻ = −(P + Q);
* P_qp = (Z/4)∮h_t,q h_t,p, with h_t = √(2Y)(e·n̂) (transverse wall current);
* Q_qp = (Z/4)∮h_z,q h_z,p, with h_z = (i/ωμ₀)√(2/Y)(∇×e)_z (longitudinal wall current).

Written in T1's N and G, this is exactly T1 B3–B4. It couples every pair of propagating modes through the walls,
across degenerate groups, and has a forward–backward block. The new functions `first_order_matrices` (wall
integrals by quadrature of `transverse_field`) and `straight_gsm_blocks` (lossy straight solved by its eigenmodes,
which stays well conditioned for any length) implement it. Checks:

* diag Λ⁺ = (1+i)α of the earlier surface-integral reference, to 2e-15;
* Λ± symmetric;
* the eigenmode solution equals the matrix exponential to 1e-10 (1 m);
* S21 symmetric to 7e-11 for 20 m at 90 GHz with 129 modes;
* the result is passive.

**2. Check against the exact modes of a guide with one pair of impedance walls.** I derived the dispersion
relation myself:

* LSE^u from E = −∇×(ûX cos k_n v′ e^{−iγz}), and LSM^u from H = ∇×(ûX sin k_n v′ e^{−iγz});
* Leontovich gives X′(0) = qX(0) and X′(a) = −qX(a), with q = ikη₀/Z (LSE^u) or ikZ/η₀ (LSM^u);
* hence (q² − k_u²) sin k_u a + 2qk_u cos k_u a = 0, and γ² = k² − k_n² − k_u². This agrees with T1 B6.

Its root (`exact_gamma`) tends to mπ/a, and for small Z to Butterweck's eq. (21) (1e-4). Relative error of the
attenuation, aluminium (1+i)R_s on one pair of WR-90 walls. For broad walls, the modes are named in the swapped frame
(a ↔ b). "group" is the WP7 reference before this resolution (each degenerate group's own Λ⁺ block); "group + Λ⁻"
adds its own forward–backward block.

| case | f_c/f | groups | full first-order system: max (median) | group: max (median) | group + Λ⁻: max |
| --- | --- | --- | --- | --- | --- |
| side walls, 90 GHz | < 0.9 | 61 | 6.2e-4 (3.6e-4) | 3.8e-3 (6.9e-5), TE0,1 | 3.8e-3 |
| side walls, 90 GHz | ≥ 0.9 | 11 | 2.3e-3 (5.5e-4), TE1,6 at 0.986 | 3.6e-3 | 3.8e-3 |
| broad walls, 90 GHz | < 0.9 | 61 | 9.5e-4 (3.6e-4) | 1.7e-3 (2.2e-4), global TE1,0 | 1.7e-3 |
| broad walls, 90 GHz | ≥ 0.9 | 11 | 1.7e-3 (3.3e-4) | 2.5e-3 | 1.7e-3 |
| side walls, 45 GHz | < 0.9 / ≥ 0.9 | 17 / 4 | 3.3e-4 / 1.3e-3 | 1.3e-3 / 1.1e-3 | 1.4e-3 / 1.3e-3 |
| broad walls, 45 GHz | < 0.9 / ≥ 0.9 | 17 / 4 | 5.4e-4 / 2.3e-4 | 5.8e-4 / 3.5e-3 | 6.0e-4 / 2.3e-4 |
| 20 mm square, one pair, 30 GHz | ≥ 0.9 (TE0,4 at 0.9993) | 2 | 6.0e-4 | **1.8e-2** | 6.0e-4 |

Findings:

* **WP2's two claims are confirmed.** First, inter-group coupling through the walls shifts the attenuation by up to
  3.8e-3 (TE0,1 with lossy side walls at 90 GHz). There it brings the first-order result from 3.8e-3 to within
  3.5e-4 of the exact value. Second, near cutoff the forward–backward block matters: TE0,4 in the square guide is
  off by 1.8 % without it and by 6e-4 with it.
* **Neither first-order model is exact.** The full system has a nearly uniform error of about 3–4e-4. That is of
  the order of |Z|/η₀ = 3.8e-4 at 90 GHz: the O(Z²) remainder of first-order theory, including the spurious far-mode
  terms that T1 section 5 describes. Its worst case is 2.3e-3, within 2 % of cutoff. The group-only model is better
  for many individual modes (median 7e-5 on the side walls), but its worst cases are 4–30 times larger.
* **The WP7 tolerance was wrong.** The old tests compared WP2 with a different first-order truncation at 1e-6. The
  two models are 1e-3 apart, so no correct code could have passed.

**3. Decision.**

* **Reference model.** The adversarial tests of WP2's lossy straights now use the full first-order coupled system,
  T1 B4, as derived independently above: `test_20m_lossy_straight_at_90GHz`,
  `test_square_guide_degenerate_groups_in_a_lossy_straight`, and `compare_wp2_loss.py`.
* **Tolerance: 1e-8 absolute on every entry of S11 and S21 over the propagating modes.** This is the same theory
  implemented independently. The reference's own numerical noise is ≤ 1e-10 (eigenmodes vs exponential, S21 symmetry,
  quadrature with 96 vs 160 points per wall: 1e-13 on Λ).
* **Per-mode α unchanged.** The α tests (definition P_loss/2P, = Re Λ⁺_pp) keep 1e-6.
* **What this means physically.** The first-order model is within 1e-3 of the exact Leontovich attenuation for
  modes with f_c/f < 0.9, and within about 2.5e-3 closer to cutoff (aluminium WR-90, 45 and 90 GHz, one pair of
  walls). For conductivity, the uncertainty of σ itself dominates. No test may ask for more than this from the
  first-order theory.
* **New self-checks.** `tests/adversarial/test_wp7_references.py` has four new tests: the coupled reference's
  self-consistency, the exact root against Butterweck's eq. (21), coupled vs group against the exact modes, and
  backward coupling near cutoff.

Status: resolved on the WP7 side. WP2 is unchanged and its specification (T1) stands.

**Signed off by Daniel on 9 Nov 2026:** resolution accepted (WP2 and T1 stand; WP7 tests against the coupled
first-order reference).
