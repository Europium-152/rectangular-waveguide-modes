# Discrepancy: chamfered H-plane miters, MiterMM (WP4b) vs Alimenti et al. 1994 Figs. 4-6 and Ma et al. 1997 Fig. 11

*Opened by WP4b on 2026-11-11. To be settled by a theory agent and Daniel, not by WP4b. WP3b and WP7b (FEM) are
the next independent evidence.*

## Observation

`wrmodes.modematch.MiterMM` reproduces, within the digitizing uncertainty, the unmitered corner and the optical
mirror of both papers (Alimenti Fig. 2: rms z 0.76; Fig. 3: rms z 0.71; Ma Fig. 11 c/a = 0: rms z 0.19), Kotik &
Sivov Fig. 3 (optical mirror, overmoded) and every curve of Alessandri et al. (E plane, b/a = 0.5) including the
chamfered ones (docs/reports/WP4b.md). It does **not** reproduce the chamfered H-plane curves:

| Curve (geometry, CONVENTIONS.md 6) | rms z | reflection zero, paper | zero, MiterMM |
| --- | --- | --- | --- |
| Alimenti Fig. 4, t/a = 0.30 (d/a = 0.919) | 5.1 | 1.29 | 1.302 |
| Alimenti Fig. 4, t/a = 0.34 (d/a = 0.948) | 7.6 | 1.50 | 1.47 |
| Alimenti Fig. 4, t/a = 0.38 (d/a = 0.976) | 12.4 | 1.69 | 1.636 |
| Alimenti Fig. 6, t/a = 0.386 (d/a = 0.980), MMFD | 34.6 | 1.68-1.78 and 1.943 | 1.666 and 1.943 |
| Alimenti Fig. 6, t/a = 0.386, BCMM | 13.5 | (below -38 dB from 1.64 to 1.77) | 1.666 and 1.943 |
| Alimenti Fig. 5, matching frequency at t/a = 0.30 / 0.32 / 0.34 / 0.36 / 0.38 / 0.40 | | 1.299 / 1.401 / 1.501 / 1.593 / 1.690 / 1.821 | 1.302 / 1.390 / 1.471 / 1.550 / 1.636 / 1.757 |
| Ma Fig. 11, c/a = 0.70 (d/a = 0.919, same geometry as Alimenti t/a = 0.30) | 16.8 | 1.38 | 1.302 |
| Ma Fig. 11, c/a = 0.631 (d/a = 0.968) | 19.1 | 1.68 and 1.93 | 1.587 and 1.971 |
| Ma Fig. 11, c/a = 0.60 (d/a = 0.990) | 31.5 | (two zeros, below the frame) | 1.757 and 1.892 |

The two papers also disagree with each other: Ma's c/a = 0.70 and Alimenti's t/a = 0.30 are the same miter (L2's
reading of t) and their zeros are 0.09 apart in f/fc. MiterMM agrees with Alimenti at t/a = 0.30-0.32 and drifts
below for larger t/a.

## Evidence that MiterMM is right for the stated geometry

1. **Independent finite differences** (`benchmarks/modematch/fd_hplane_miter.py`, shares only the geometry): for
   t/a = 0.38, |S11| of TE10 = 0.03710 at f/fc = 1.5 (MiterMM 0.037103), about 1e-5 at 1.636 (MiterMM: the zero) and
   0.01387 at 1.69 (MiterMM 0.013867, i.e. -37 dB where Alimenti has his zero). For the optical mirror at Ma's Table 1
   point (f/fc = 16/9.488) FD gives 0.27779 (MiterMM 0.27779; Ma Table 1: 0.2622; Alimenti Fig. 3: about 0.288).
   (`data/fd_hplane_miter.npz`: FD at N = 50, 100, 200, 400 and the Richardson value; at t/a = 0.38, f/fc = 1.5 FD
   0.037100 vs MiterMM 0.037103, at 1.69 0.013867 vs 0.013867.)
2. **Convergence**: the zero of t/a = 0.38 is 1.63605 for M = 20 to 400 arm modes (to 1e-5).
3. **Phase checks** against Marcuvitz's corner circuits (unmitered, both planes): complex S11 and S21 within
   1.7e-3.
4. **The same method reproduces Alessandri's chamfered E-plane curves** (d/b = 0.83-0.9) within their uncertainty
   or, at the deep zeros, within 2.2e-3 in |S11|.

## What fits the papers

A one-parameter fit of the miter position to each published curve (`benchmarks/modematch/fit_chamfer.py`,
data/fit_chamfer.npz):

| Curve | nominal t/a | best-fit t/a | rms z at the best fit |
| --- | --- | --- | --- |
| Ma c/a = 0.70 | 0.300 | 0.316 | 1.63 |
| Ma c/a = 0.631 | 0.369 | 0.389 | 1.11 |
| Ma c/a = 0.60 | 0.400 | 0.419 | 0.42 |
| Alimenti Fig. 6, BCMM (Reiter & Arndt) | 0.386 | 0.391 | 4.3 |
| Alimenti Fig. 4, t/a = 0.30 / 0.34 / 0.38 | | 0.301 / 0.346 / 0.376 | 5.0 / 3.9 / 11.9 |
| Alimenti Fig. 6, MMFD | 0.386 | 0.391 | 15.2 |

Ma's chamfered curves are reproduced almost perfectly once the cut is moved by +0.016 to +0.020 a in t (c/a smaller
by the same amount): a consistent offset, plausibly the effective wall position of their 60-step staircase (their
Table 1 shows that the staircase alone moves |S11| by up to 1.7 %). The BCMM line of Fig. 6 fits at t/a + 0.005.
Alimenti's own MMFD curves fit no single geometry.

## Questions for the theory agent / Daniel

1. Is L2's reading of Alimenti's t (d/a = (1 + t/a)/sqrt(2)) right? MiterMM matches t/a = 0.30 exactly, so a
   different reading would have to coincide there and differ by 0.005-0.05 at larger t/a; no simple reading does.
2. Should Alimenti Figs. 4-6 and Ma Fig. 11 (chamfered curves) be demoted from acceptance references to qualitative
   ones (WP3b and WP7b cards, registry entries `wave2-alimenti1994-fig4`, `-fig5`, `-fig6`, `wp3-ma1997-figures`)?
3. The registry entry `wp3-ma1997-table1` sets 0.57 % on |S11| = 0.2622 for the optical mirror at 16 GHz; MiterMM and
   the FD solution both give 0.2778 (+6 %). The test `tests/adversarial/test_wp3_adversarial.py::
   test_h_miter_against_ma1997_table1` already fails on main (WP4b did not look at why, by the independence rule).
