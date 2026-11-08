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
