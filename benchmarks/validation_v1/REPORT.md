# WP7 — Independent verification, wave 1: report

*Written by the coordinator from the WP7 agent's final report (the agent's environment did not allow it to write
report files), plus the coordinator's integration runs. Branch `wave-1/WP7-verification`.*

## Independence

The WP7 agent built every reference from `CONVENTIONS.md`, the task cards, the wrmodes 0.1 code on `main` and the
literature, and did not read the WP1–WP4 worktrees or branches.

## References built

| Reference | File | Result |
| --- | --- | --- |
| Global line reference: product of 0.1 transfer matrices (atol 1e-10, `error_control='full'`) over straights and bends, own frame relabelling | `benchmarks/validation_v1/ref_line.py`, `ref_frames.py` | vs wrmodes 0.1: max \|Δ\|S\|²\| = 3.2e-11 over S11, S21, S12, S22 and every mode (target 1e-9); 129 modes at 90 GHz in 75 s |
| E-plane frame relabelling derived from the potentials | `ref_frames.py` | global (M,N) → local (N,M) × (−1)^M (TE), (−1)^(M+1) (TM); equals `eplane_frame_signs` and the field overlaps to 1e-12 (three guides, one square) |
| Surface-integral wall loss | `ref_loss.py` | vs textbook α: 2.0e-12 for every propagating mode (WR-90, 20 mm square; aluminium and copper). TE10, aluminium WR-90: 0.1227 dB/m at 18 GHz, 0.2303 dB/m at 90 GHz. TE11/TM11 couple through the walls (WR-90, 60 GHz: [[0.02686, 0.00886], [0.00886, 0.04054]] Np/m) |
| First-order coupled wall loss (added after the WP2 discrepancy) | `ref_loss.py`: `first_order_matrices`, `straight_gsm_blocks` | WP7's own derivation from unconjugated reciprocity, equal to T1 B3–B4: all propagating pairs coupled through the walls, forward and backward blocks. diag Λ⁺ = (1+i)α to 2e-15; eigenmode solution = matrix exponential to 1e-10; S21 symmetric to 7e-11 (129 modes, 20 m, 90 GHz) |
| Exact Leontovich modes, one pair of impedance walls | `ref_loss.py`: `exact_gamma`, `one_pair_comparison` | own LSE^u/LSM^u dispersion relation (= T1 B6); tends to Butterweck eq. (21) to 1e-4. Aluminium WR-90 at 90 GHz: full first-order system within 1e-3 of exact for f_c/f < 0.9 (median 3.6e-4), 2.3e-3 within 2 % of cutoff; group-block-only up to 3.8e-3, and 1.8e-2 near cutoff (TE0,4, square guide) |
| Literature comparisons (L1 CSV format) | `ref_literature.py`, `compare_literature.py` | ready; tested on synthetic CSV |
| COMSOL post-processing | `benchmarks/comsol/wp7_postprocess.py` | recovers planted GSM columns to 1e-9 (C1, C3 geometries) |
| Marcuvitz §5.26a eq. (2a) | `ref_literature.py` | consistent with its asymptotic forms (2c) 0.5 %, (2d) 1.3 % at δ = 0.2 |

Long-line phase floor: a 20 m straight at 90 GHz holds phase only to about 1e-10 (one ulp of β times βL ≈ 3.8e4
rad); no tolerance below this is meaningful.

## Comparisons run by the coordinator after integration (WP1 + WP4 + conventions update 1)

* `compare_wp1_lines.py`: **10 of 10 line cases pass**: same plane (20/40/60/90 GHz), mixed planes (20/30/40 GHz),
  §7 sequence with 10 m straights (20/50/90 GHz). Largest |ΔP| = 3.2e-11 (target 1e-6).
* FrameChange vs the derived relabelling: S21 and S12 exact (0.0), S11 = S22 = 0, H → E → H identity exact. The
  script now passes the guide explicitly (WP1's `FrameChange` needs it).
* Adversarial tests: 48 passed, 17 skipped (interfaces not yet merged), 0 failed after two test corrections:
  * WP2 tests skip until WP2 lands (WP1's `Guide` raises `NotImplementedError` for `sigma`).
  * The WP4 offset small → large → small identity uses `Step.solve` (the mode-matching problem truncated on
    exactly the given bases, identity to 1.5e-15). `Step.gsm` solves on its own converged basis and returns a
    sub-block (CONVENTIONS §10), which assumes neighbours far enough for the dropped evanescent modes to decay;
    at zero distance that is not the case. Measured consequence: two offset junctions 2 cm apart in WR-90 at
    35 GHz, cascaded with propagating-only intermediate ports, differ by 0.075 in S21 from the full solution,
    independent of truncation. Lines with closely spaced discontinuities need `cutoff_factor > 1`.

## Discrepancy WP2–WP7 on lossy straights (resolved on the WP7 side, awaiting Daniel)

`docs/discrepancies/2026-11-08-wp2-wp7-straight-attenuation.md`. Two WP2 adversarial tests failed: 3.6e-3 and
2.6e-2 against a tolerance of 1e-6. Cause: the WP7 tests compared WP2 with each degenerate group's own Λ⁺ block.
That is a different first-order truncation from the one T1 specifies and WP2 implements: T1 couples every pair
through the walls and includes the forward–backward block. Against the exact modes of a guide with one pair of
impedance walls, both truncations are first-order approximations, about 1e-3 apart. The full system is the more
uniform one (about 4e-4, at most 1e-3 for f_c/f < 0.9); the group-only model is off by up to 3.8e-3, and by 1.8e-2
near cutoff. The 1e-6 tolerance was therefore unachievable.

The tests now compare WP2's lossy straight with WP7's independently derived full first-order system, at 1e-8
absolute on S11 and S21. The per-mode α tests keep 1e-6. The physical accuracy of the first-order model is stated as
1e-3 (f_c/f < 0.9) and 2.5e-3 (near cutoff).

## Acceptance

| Test | Result |
| --- | --- |
| References reproduce 0.1 where they overlap (1e-9 in power) | pass, 3.2e-11 |
| COMSOL scripts run on Daniel's version without edits | not yet run: written for COMSOL 6.2.0.415 (version of the existing export); compiled and dry-run against stand-ins |
| Every WP1–WP4 acceptance number citing WP7 has a registry entry | pass, 16 entries in `benchmarks/registry.json` |

## COMSOL campaign 1 — Gate A (10 Nov 2026)

Daniel ran `benchmarks/comsol/CAMPAIGN_1.md` in COMSOL 6.2 from the GUI (the WP7 Java scripts were not used).
Data, Java models and the comparison script are in `benchmarks/comsol/campaign_1/`; the full tables are in
`campaign_1/RESULTS.md` (`python benchmarks/comsol/campaign_1/compare_campaign_1.py --out ...`). Criterion: within
the COMSOL error (mesh check where available) + 0.02 dB in |S21|.

| Case | Points | Result |
| --- | --- | --- |
| C1 90° plain H-plane miter (2D) | 23 | 19 within 0.02 dB (14 within 0.005); at 30 and 90 GHz, against the fine mesh, ≤ 0.0012 dB. **4 open**: 26.3 (0.12 dB), 52.5 (0.05), 53.0 (0.04), 46 (0.03 dB) |
| C1b 60° and 120° optical-mirror miters (2D) | 6 | pass, ≤ 0.005 dB against the fine mesh: confirms the optical-mirror definition (CONVENTIONS §6) |
| C2 sech 60° 15 cm E-plane bend, Al 3.774e7 S/m | 22 | pass: total ≤ 0.0074 dB (the 30 GHz PEC Fig. 6 deviation, unchanged); ohmic part (Al − PEC) ≤ 0.0004 dB |
| C2 constant 60° 15 cm E-plane bend, Al | 22 | ohmic part pass: same-model Al − PEC 0.0178 / 0.0274 / 0.0414 dB at 18 / 40 / 60 GHz, wrmodes 0.0178 / 0.0272 / 0.0412. The lossless conversion of this model is 0.021–0.026 dB below the paper's PEC model, which wrmodes matches to ≤ 0.0035 dB (open, COMSOL side) |
| C3a WR-90 → h = 2.0, 1.6 mm centred steps (3D) | 6 | pass: \|S21\| ≤ 0.005 dB, \|S11\| 0.010–0.023 dB against the fine mesh, which moves COMSOL towards wrmodes; COMSOL \|S11\|²+\|S21\|² = 1.001–1.003 (fine), 1.002–1.005 (coarse) |
| C3b 1.6 mm × 100 mm strangled section (3D) | 3 | explained: \|S11\| 0.06–0.11 dB off; the section is a resonator with \|Γ\| ≈ 0.73 at each step, and a 1° round-trip phase error (wrmodes, length changed by 55 µm) reproduces COMSOL at 10 GHz exactly (0.7527 / 0.6584 vs COMSOL 0.7548 / 0.6586); C3a shows 0.3–0.4° per reflection. COMSOL's power sum is 1.0034 (10 GHz) and 1.014 (30 GHz, unphysical) |

Findings:

* **Solver tolerance is not a factor**: the stationary relative tolerance 1e-2 → 1e-4 changes the C2 and C3
  values by < 2e-5 (the coordinator's hypothesis was wrong); the C3 power excess is discretization error.
* **C1 open points**: all four lie less than 0.1 GHz above a TE_m0 cutoff (TE40 26.23, TE70 45.90, TE80
  52.46 GHz), where that mode's guide wavelength (40–155 mm) is much longer than the 2 cm PML, whose scaling is
  set by the free-space wavelength. wrmodes moves by < 0.001 dB under mesh density ×2, order 2 and arm length a.
  Check requested: the same frequencies with 15 cm PMLs.
* **C2 constant, lossless offset**: the new constant-curvature model and the paper's model
  (`constant_E_bend_60_deg_lowfreq_15cm.mph`) differ by 0.02–0.03 dB with PEC walls; not a wall-loss question.
  Requested: the paper model's Java, to diff.
* Phase: S21 agrees modulo 180° (COMSOL's port-mode sign) to 0.01–1.2°, growing with frequency in the 15 cm
  bends (COMSOL mesh dispersion, about 1e-4 of the total phase).
* C3 was solved as a full model (not the quarter model of the instructions); no effect on the comparison.

## Open issues

* COMSOL scripts untested in COMSOL (property names are discovered at run time; C0 dumps them).
* Marcuvitz H-plane step (§5.24a) not implemented: L1 found the printed formulas unusable for a/λ ≳ 0.7.
* Alessandri 1994 does not state the broad side a; the E-plane result depends on b/a (`--b-over-a`).
* No independent lossy-bend reference; the WP2 power-balance test uses a first-order 2αL check (tolerance 2e-3). The coupled straight reference could be extended to bends (add κK to the same system) if wanted.
* Conventions: the reactive part of Z_s also shifts β by about α; the WP2 card asks only for α.
