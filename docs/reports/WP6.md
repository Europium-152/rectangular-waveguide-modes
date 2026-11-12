# WP6 — Twists: report

Branch `wave-2/WP6-twists`, from `main` at a0aaae0. Card `tasks/WP6-twists.md`.

## What was done

| Item | Files |
| --- | --- |
| `Twist(angle_deg, length, profile='constant', guide=None, cutoff_factor=None, atol=1e-6, ...)`, `TwistProfile`, coupling matrix, system matrices, slice solver, helical modes, zero-length limit, `convergence_study` | `src/wrmodes/components/twist.py`; exported in `wrmodes.components` and as `wrmodes.Twist` |
| Tests (57 fast, 3 slow) | `tests/test_twist.py` |
| Example: Doane's three twists, convergence, a twist–bend–twist line | `examples/twist.py` |
| Benchmarks (npz with `meta`, CONVENTIONS §9) | `benchmarks/twist/`: `doane_fig9.py`, `e4_predictions.py`, `near_cutoff_16p5.py`, `checks.py`; data in `benchmarks/twist/data/` |
| Registry | `benchmarks/registry.json`: eight `wp6-*` entries |
| Discrepancy report | `docs/discrepancies/2026-11-12-doane-fig9-twists.md` |

Read: CONVENTIONS.md, T3, T3b and their reports and scripts, `loss.py`, the component code, Doane & Anderson 1989
p. 159 (PDF p. 7; eq. 17, 18 and Fig. 9 checked on the page image).

## Method and decisions

* **Closed forms: T3b only.** `twist_coupling_matrix` is T3b box 2a/2b in the separated 1D integrals (the
  vectorized `K_matrix_fast` form), and `twist_system_matrices` is T3b box 4: dy/dz = (D + θ′(z) X) y,
  X = [[A, B], [B, A]]. Against T3b's `K_matrix_fast` the difference is 0 (bitwise), against T3's Cᵀ 1.6e-14
  (182 modes). Not antisymmetrized; K_TE,TM = 0 ≠ K_TM,TE.
* **Sign and axis.** Positive angle: the cross-section turns from +x towards +y (from u towards v at port 1),
  right-handed about the direction of propagation; rotation about the centre (T3 and T3b's θ).
* **Profiles.** 'constant'; 'cosine' θ′ = (πθ/2L) sin(πz/L); 'triangular' Doane eq. 17; or a callable r(t),
  normalized by quadrature, with optional `breakpoints`. Doane & Anderson only define the triangular rate
  (eq. 17); "cosine" is the half-cosine rate of T3/T3b, the reading that reproduces their curve. Stated in the
  docstrings.
* **Integration.** Constant rate, propagating modes, no loss: one `expm` of the whole section (exact).
  Variable rate: 4th-order commutator-free Magnus slices (`transfer_matrix._slice_exponentials`, reused), with the
  bend solver's adaptive doubling and asymptotic-ratio test, error control on every complex entry of S
  (atol 1e-6), slice edges on the kinks (L/2 for the triangle) and a minimum slicing from the beat lengths.
* **GSM built directly** (CONVENTIONS §6) when the basis has evanescent modes or the guide is lossy: every slice's
  transfer matrix is converted to a GSM and the slices are joined by Redheffer products (batched tree reduction),
  with |Im β| h ≤ 4 and ‖Λ‖ h ≤ 0.5 per slice; for a constant rate one slice GSM is doubled. Both routes agree to
  1e-11 (test).
* **Basis.** All families are coupled. A port basis is completed before solving: with the propagating modes it
  lacks and, if it has evanescent modes, with every mode of every family up to its highest cutoff (the line's
  'live' rule omits families a twist couples). `cutoff_factor=c` solves on all modes below c·f. `default_basis`
  uses `evanescent_families='all'` for c > 1 (card item 4). The port sub-block is returned; dropped modes are taken
  as matched.
* **Frames.** `plane = 'H'`; `basis2 == basis1` for any angle; no FrameChange. `Line` needed no change.
* **Loss.** The T1 matrix `loss_matrix_full` of the cross-section is added in the co-rotating frame. Neglected:
  helicoidal-wall corrections, O((τρ)²) for the wall area and O(τρ) in the tangential H. The latter is odd under
  each mirror, like the twist coupling, so it cannot change any mode's attenuation at first order in τ; it perturbs
  the twist coupling by a relative O(R_s/η0) ≈ 1e-4 (an argument, not a derivation).

## Acceptance tests

| Check | Target | Result |
| --- | --- | --- |
| Exact two-mode law (T3 box T3-4, T3b box 5) | 1e-12 | **pass**: 9.3e-14 (20 mm square at 10 GHz, constant / cosine / triangular / callable, 90°, 154.7°, −60°; S11 included) |
| Z(−θ)Z(θ) = I at zero length | 1e-12 | **pass**: 4.2e-15 (transfer matrices and GSM cascade; 30 mm square at 9 GHz, tall WR187 at 9 and 16.5 GHz) |
| +90° then −90° of finite length vs identity | reported | TE1,0 power lost: tall WR187 9 GHz 0.28 / 2.0e-2 / 1.6e-4 / 1.4e-5 at L = 5 / 20 / 80 / 320 λ (T3b: 0.980 transmitted at 20 λ, reproduced). Slower for degenerate pairs: max ‖S21‖−I\| 1.7e-2 (9 GHz) and 0.21 (16.5 GHz) at 320 λ. Table below |
| θ → 0, lossless and aluminium | `Straight` to 1e-12 | **pass**: 3.8e-14 (cutoff_factor 1 and 1.5, constant and triangular) |
| Reciprocity, unitarity (lossless) | 1e-10 | **pass**: ≤ 6e-14 for Doane's twists, ≤ 1e-12 for 320 λ pairs; with evanescent modes (direct GSM) also |
| Passivity (lossy) | σ_max ≤ 1 | **pass**: margin 2.3–2.5e-3 (aluminium, Doane's twists) |
| T3/T3b full-basis tables, WR187 15 / 16.5 / 18 GHz, three profiles | 1e-3 relative | **pass**: 3.4e-7 against T3's prototype (total and the three largest modes, 9 cases); T3's printed table within its last digit; T3b's square-guide angle table within 1.2e-3° (6 rows) |
| Doane Fig. 9, constant and cosine | digitizing uncertainty | **not met** (discrepancy report): cosine within 1 σ only at 15.1–16.35 GHz (up to 9 σ below above 16.5 GHz); constant within 1 σ at 9/119 points because of TE2,1 lobes (up to 0.71 %) absent from Fig. 9; our TE0,1 conversion alone follows it (rms 0.11 %) |
| Fig. 9 triangular | reported vs 0.11 / 0.25 / 0.60 % | ours 0.114 / 0.247 / 0.603 % (cutoff_factor 1), 0.120 / 0.262 / 0.609 % (1.5), against the digitized 0.049 / 0.099 / 0.149 % (not 0.05–0.08 % as eye-read in T3): 2.2–4.0×. Settled by E4 |
| Near-cutoff sensitivity at 16.5 GHz | report | see below: TE2,1 0.61 % robust; triangular +5 % when TE2,3/TM2,3 are kept |
| Galerkin propagation constants (T3 §7, T3b §5) | within the Galerkin's convergence (state it) | **pass**: the spectral solution is converged to 3e-10, so the comparison measures the modal truncation. max \|γ_CM/γ − 1\| = 5.37e-3 / 2.93e-3 / 1.45e-3 at 89 / 359 / 1443 modes, Richardson (1/χ_max) 7.5e-4 / 1.8e-4, matching T3b's numbers to 3 digits. T3's own Galerkin values are not stored in the repository, so only T3b's spectral reference is used |

## Near-cutoff sensitivity at 16.5 GHz

`benchmarks/twist/near_cutoff_16p5.py`. TE2,3/TM2,3 (cutoff 16.5121 GHz) are coupled to TE1,0 at first order.
Spurious power in % (number of modes), cutoff_factor = 1 means propagating modes only:

| cutoff_factor | constant 16.45 / 16.50 / 16.55 GHz | of which TE2,1 | triangular 16.45 / 16.50 / 16.55 GHz | cosine 16.45 / 16.50 / 16.55 GHz |
| --- | --- | --- | --- | --- |
| 1 | 0.688 / 0.697 / 0.668 | 0.610 / 0.609 / 0.559 | 0.2418 / 0.2474 / 0.2667 | 0.856 / 0.867 / 0.878 |
| 1.001 | 0.688 / 0.699 / 0.668 | 0.610 / 0.610 / 0.559 | 0.2418 / **0.2603** / 0.2667 | 0.856 / 0.867 / 0.878 |
| 1.01 | 0.690 / 0.699 / 0.668 | 0.611 / 0.610 / 0.559 | **0.2542** / 0.2603 / 0.2667 | 0.856 / 0.867 / 0.878 |
| 1.5 | 0.693 / 0.691 / 0.643 | 0.615 / 0.603 / 0.542 | 0.2554 / 0.2616 / 0.2682 | 0.858 / 0.869 / 0.881 |
| 2 | 0.694 / 0.692 / 0.645 | 0.616 / 0.604 / 0.543 | 0.2570 / 0.2633 / 0.2700 | 0.857 / 0.869 / 0.880 |
| 3 (182–184 modes) | 0.692 / 0.694 / 0.648 | 0.614 / 0.606 / 0.548 | 0.2554 / 0.2617 / 0.2683 | 0.858 / 0.870 / 0.881 |

Findings:

1. **The constant twist's 0.61 % of TE2,1 (T3 §6) is real**, not a near-cutoff artifact. It changes by at most
   1 % relative with the truncation (0.603–0.616 %) and smoothly with frequency (0.61 → 0.56 % at +0.05 GHz).
   It is the main lobe of first-order TE1,0 → TE2,1 conversion, |K| = 1.68, abrupt ends. It is absent from Fig. 9.
2. **The triangular twist jumps by 5 % at the TE2,3/TM2,3 cutoff when these modes are dropped.** With
   cutoff_factor 1 they are omitted below their cutoff. 16.5 GHz then gives 0.2474 %, and 0.2603 % once they are
   kept (cutoff_factor ≥ 1.001). With them kept, the curve is continuous through the cutoff (0.2542 / 0.2603 /
   0.2667 %). This step accounts for T3b's "6 % from evanescent modes" at 16.5 GHz. Further modes change
   < 1.2 %.
3. **Evanescent modes matter elsewhere too:** at 15.0 GHz cutoff_factor 1.3 adds TE0,5 (cutoff 15.76 GHz,
   first-order coupled) and the triangular result rises 0.1142 → 0.1204 % (+5 %). Rule of thumb from these
   runs: a variable-rate twist in tall WR187 needs cutoff_factor ≈ 1.1–1.5 for 1 % relative accuracy in the
   spurious power. A constant twist gets 2–4 % from cutoff_factor 1.
4. The cosine twist (dominated by TE0,1, far from cutoff) is insensitive (< 0.4 %).
5. The default for a `Line` stays cutoff_factor 1 (CONVENTIONS §3 for coupled-mode solvers). Users who need
   better than ~5 % on second-order conversion must pass cutoff_factor > 1, to the twist or to the line.

## Doane & Anderson Fig. 9

`benchmarks/twist/doane_fig9.py`, against L1's digitized curves (119 points each). Selected points, % of input:

| f (GHz) | constant ours / TE0,1 / Fig. 9 | cosine ours / two-mode / Fig. 9 | triangular ours / Fig. 9 |
| --- | --- | --- | --- |
| 15.025 | 1.296 / 1.050 / 1.303 ± 0.030 | 0.362 / 0.332 / 0.351 ± 0.010 | 0.116 / 0.049 ± 0.010 |
| 15.50 | 0.594 / 0.438 / 0.603 ± 0.012 | 0.550 / 0.518 / 0.550 ± 0.010 | 0.139 / 0.062 ± 0.010 |
| 16.00 | 0.275 / 0.061 / 0.160 ± 0.015 | 0.731 / 0.709 / 0.726 ± 0.011 | 0.193 / 0.080 ± 0.014 |
| 16.50 | 0.697 / 0.032 / 0.051 ± 0.011 | 0.867 / 0.870 / 0.885 ± 0.010 | 0.247 / 0.099 ± 0.011 |
| 17.00 | 0.554 / 0.355 / 0.346 ± 0.010 | 0.950 / 0.984 / 1.008 ± 0.010 | 0.338 / 0.116 ± 0.010 |
| 17.50 | 1.715 / 0.978 / 0.948 ± 0.013 | 0.969 / 1.040 / 1.063 ± 0.011 | 0.450 / 0.133 ± 0.010 |
| 17.975 | 2.061 / 1.774 / 1.685 ± 0.046 | 0.946 / 1.039 / 1.048 ± 0.012 | 0.595 / 0.149 ± 0.014 |

Summary (details and the "what it is not" list in `docs/discrepancies/2026-11-12-doane-fig9-twists.md`):

* **Cosine:** within 1 σ at 15.1–16.35 GHz, then up to 9 σ (0.10 %) below. The two-mode model (TE1,0 + TE0,1)
  follows the whole curve within 3.5 σ (rms 0.022 %).
* **Constant:** Fig. 9 follows our TE0,1 conversion (rms 0.11 %). It does not follow our total, whose TE2,1 lobes
  reach 0.48 % at 15.25, 0.53 % at 15.78, 0.61 % at 16.45 and 0.71 % at 17.35 GHz.
* **Triangular:** ours is 2.2–4.0× Fig. 9, 95–98 % of it in TE1,2, TE0,3 and TM1,2. The digitized curve rises
  from 0.049 to 0.149 % (L1). T3's eye reading of 0.05–0.08 % is superseded, and the gap at 18 GHz is 4×, not 7×.
* Our reading: Fig. 9 looks like a reduced-mode calculation for the cosine and constant curves. For the triangular
  curve no subset of ours matches (the two-mode model gives 0.0003–0.027 %). Not fudged; E4 decides.

## Predictions for COMSOL campaign 2, case E4

`benchmarks/twist/e4_predictions.py` → `benchmarks/twist/data/e4_predictions.npz`. 20-inch 90° twist, tall WR187,
TE1,0 incident, E4's output ports (T3 names: m half-sines across the short side, n across the long side).
|S21| and power. Each cell gives cutoff_factor 1 / 1.5 / 2 (evanescent modes of all families):

| port mode | triangular 15 GHz | triangular 18 GHz | cosine 18 GHz |
| --- | --- | --- | --- |
| TE1,0 | 0.99943 / 0.99940 / 0.99940 (99.886 / 99.880 / 99.880 %) | 0.99698 / 0.99695 / 0.99692 (99.397 / 99.392 / 99.385 %) | 0.99527 / 0.99523 / 0.99523 (99.057 / 99.049 / 99.048 %) |
| TE0,1 | 0.00351 / 0.00330 / 0.00328 (0.0012 / 0.0011 / 0.0011 %) | 0.00012 / 0.00092 / 0.00092 (0.0000 / 0.0001 / 0.0001 %) | 0.09515 / 0.09557 / 0.09558 (0.905 / 0.913 / 0.914 %) |
| TE0,3 | 0.01372 / 0.01422 / 0.01416 (0.0188 / 0.0202 / 0.0200 %) | 0.03969 / 0.04037 / 0.04061 (0.158 / 0.163 / 0.165 %) | 0.00678 / 0.00687 / 0.00695 (0.0046 / 0.0047 / 0.0048 %) |
| TE1,2 | 0.02824 / 0.02894 / 0.02887 (0.0797 / 0.0837 / 0.0834 %) | 0.06193 / 0.06184 / 0.06212 (0.384 / 0.383 / 0.386 %) | 0.01514 / 0.01516 / 0.01535 (0.0229 / 0.0230 / 0.0236 %) |
| TM1,2 | 0.01029 / 0.01056 / 0.01067 (0.0106 / 0.0111 / 0.0114 %) | 0.02154 / 0.02168 / 0.02194 (0.0464 / 0.0470 / 0.0481 %) | 0.00680 / 0.00675 / 0.00692 (0.0046 / 0.0046 / 0.0048 %) |
| TE2,1 | 0.00616 / 0.00604 / 0.00606 (0.0038 / 0.0037 / 0.0037 %) | 0.01196 / 0.01219 / 0.01218 (0.0143 / 0.0149 / 0.0148 %) | 0.00726 / 0.00701 / 0.00703 (0.0053 / 0.0049 / 0.0049 %) |
| TM2,1 | 0.00015 / 0.00017 / 0.00017 | 0.00023 / 0.00036 / 0.00036 | 0.00020 / 0.00026 / 0.00025 |
| reflected (all modes) | 4–6e-9 | 3–4e-9 | 1–2e-9 |
| other propagating modes (not ported) | < 1e-7 | 1.1e-5 (TE1,4 0.9e-5) | 3e-6 |

* The truncation spread is ≤ 4e-5 of the input power per converted mode, and 1.2e-4 for TE1,0 (0.0005 dB),
  below the 1e-4 per-mode criterion of CONVENTIONS §8. E4's seven ports carry all but 1.1e-5 of the power, so the
  unported modes cannot distort the comparison.
* The decisive numbers are the per-mode powers: TE1,2 at 8.0e-4 (15 GHz) and 3.8e-3 (18 GHz), where Fig. 9 allows
  at most 4.9e-4 and 1.5e-3 in total. The TE1,0 transmission differs from Fig. 9's by only 0.003 dB at 15 GHz and
  0.020 dB at 18 GHz, at or below the 0.02 dB criterion. COMSOL therefore needs a per-mode power accuracy of about
  2e-4 of the input, i.e. port amplitudes |S| to about 3e-3 absolute for TE1,2.
* Phases depend on the reference planes and the 15 mm legs; only magnitudes are predicted.

## Line [Twist(+90°), SmoothBend(plane='H'), Twist(−90°)]

It runs with no new rule: the line inserts no FrameChange. Its GSM equals the cascade of the three GSMs in the
global basis to 1e-13 and is unitary to 5e-14 (test). **Physically:** the first twist turns the cross-section so
that side w lies along the input's y axis. The H-plane bend then bends the guide in the plane of side w, which in
space is the input's y–z plane, i.e. the E-plane direction of the input guide. The field still sees an H-plane
bend: TE1,0 couples to TE2,0, not to TE1,1/TM1,1. The second twist returns the cross-section to its original
orientation relative to the bent axis. This is Doane & Anderson's "favorable bend plus a twist" (p. 159), used to
turn an overmoded line in its E-plane direction without an E-plane bend. With an E-plane bend between the twists,
the line inserts the two frame changes around the bend only (test).

## Finite +90° / −90° pairs (constant rate), `benchmarks/twist/checks.py`

| guide, f | modes | TE1,0 lost: L = 5 / 20 / 80 / 320 λ | max \|\|S21\| − I\| at 320 λ |
| --- | --- | --- | --- |
| 30 mm square, 16.5 GHz | 18 | 0.60 / 4.3e-3 / 1.7e-4 / 1.0e-5 | 4.1e-2 |
| tall WR187, 9 GHz | 5 | 0.28 / 2.0e-2 / 1.6e-4 / 1.4e-5 | 1.7e-2 |
| tall WR187, 16.5 GHz | 19 | 0.93 / 0.31 / 2.0e-2 / 5.6e-4 | 0.21 |

TE1,0's own conversion falls like L⁻² (adiabatic). The slowest entries belong to near-degenerate pairs, which
the twist mixes even when slow. The pair tends to the identity in magnitude; the phases include the twist's
O(τ²) shift of the propagation constants, which is even in τ and does not cancel.

## Other numbers

* **Slices** (18 GHz, cutoff_factor 1): cosine and triangular change by < 2e-9 in power from 512 to 4096 slices.
  The adaptive solver stops at 1024 slices with an error estimate of 1.2–1.4e-9.
* **Wall loss** (aluminium, Doane's twists): ohmic loss 0.506 / 0.471 % at 15 / 18 GHz, against 0.489 / 0.442 % for
  the 20-inch straight (+3.5–7 %). Inside the twist the local field contains a few % of the lossier
  near-cutoff modes. The converted power changes by ≤ 0.012 % absolute.
* **Timing** (this 2-core machine, 18 GHz, one twist): constant < 0.01 s; cosine / triangular 1.0 / 0.7 s
  (23 modes, 1024 slices); with cutoff_factor 1.5 (51 modes) 13 s; cutoff_factor 3 (182 modes) 3–5 min. The timing
  test asserts 2 s only with `WRMODES_PERF=1`, otherwise 20 s.
* **Full fast suite**: 614 passed, 3 skipped, 1 xfailed (15 min on this machine); twist slow tests: 3 passed.

## Open issues

1. **Fig. 9** (discrepancy report): constant and cosine not met over the band; triangular 2.2–4× above. E4
   settles it.
2. **Cost with evanescent modes for variable rates.** The direct-GSM route converts and stars every slice. With
   182 modes (cutoff_factor 3) one 20-inch twist takes minutes. An interaction-picture or exponential integrator
   could cut this if wide studies are needed; not done (not required by the card).
3. **Loss in the helicoidal wall** is first-order T1 in the co-rotating frame. The neglected O(τρ) term is argued,
   not derived, to be O(R_s/η0) relative to the twist coupling; a theory check would close it.
4. **Strong twists** converge like 1/χ_max (T3 §7, T3b §5): `Twist` is accurate for τa ≪ 1, which covers practical
   twists.

## Proposals for CONVENTIONS.md (not changed here)

* §2: state the twist sign and axis: "a positive twist angle turns the cross-section from +x towards +y, right-handed
  about the propagation direction, about the centre of the cross-section; turning a twist around keeps the angle
  and reverses the profile" (T3/T3b conventions issues, confirmed by WP6's reversal test).
* §3: add that components coupling families (twists) complete the line's 'live' basis internally with all families
  up to its highest cutoff. Add that for second-order conversion near a cutoff, cutoff_factor 1 can be off by
  ~5 % (a first-order-coupled mode just below cutoff is dropped).
* §10: a twist commutes with the H ↔ E frame change (T3 box T3-5; also with equal-wall loss), so `plane = None`
  would be admissible. WP6 keeps `'H'` as the card says; `None` would save two FrameChanges around twists next to
  E-plane components.

## Decisions for the coordinator or Daniel

1. Accept the Fig. 9 constant/cosine rows as "not met, explained", pending E4 (discrepancy report).
2. E4 acceptance: compare per-mode powers (TE1,2, TE0,3, TM1,2), not the TE1,0 dB, and confirm in CAMPAIGN_2.md
   that the mesh check targets ~2e-4 of input power per mode.
3. Whether `Line` should default to a larger cutoff_factor when it contains a twist, or leave it to the user as
   now.
4. The CONVENTIONS proposals above.
