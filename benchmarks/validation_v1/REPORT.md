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

## Acceptance

| Test | Result |
| --- | --- |
| References reproduce 0.1 where they overlap (1e-9 in power) | pass, 3.2e-11 |
| COMSOL scripts run on Daniel's version without edits | not yet run: written for COMSOL 6.2.0.415 (version of the existing export); compiled and dry-run against stand-ins |
| Every WP1–WP4 acceptance number citing WP7 has a registry entry | pass, 16 entries in `benchmarks/registry.json` |

## COMSOL campaign 1 (for Daniel)

See `benchmarks/comsol/README_WP7.md`. Scripts: `WP7_C0_probe.java` (dumps port property names, run first),
`WP7_C1_h_miter.java` (WR-90 plain 90° H-plane miter), `WP7_C2_sech_bend_aluminium.java` (18–38 GHz only; at
90 GHz the 60 cm model is too large), `WP7_C3_strangling_step.java`. S-parameters are taken by projecting E and
H on wrmodes' own mode fields, so COMSOL's port sign conventions do not matter.

## Open issues

* COMSOL scripts untested in COMSOL (property names are discovered at run time; C0 dumps them).
* Marcuvitz H-plane step (§5.24a) not implemented: L1 found the printed formulas unusable for a/λ ≳ 0.7.
* Alessandri 1994 does not state the broad side a; the E-plane result depends on b/a (`--b-over-a`).
* No independent lossy-bend reference; the WP2 power-balance test uses a first-order 2αL check (tolerance 2e-3).
* Conventions: the reactive part of Z_s also shifts β by about α; the WP2 card asks only for α.
