# Discrepancy: 20-inch 90° twists in tall WR187, wrmodes vs Doane & Anderson (1989) Fig. 9

*Opened by WP6 on 2026-11-12. To be settled by COMSOL campaign 2 (case E4), a theory agent and Daniel, not by WP6.*

## Observation

Card WP6 asks that the constant and cosine curves of Fig. 9 be reproduced within the digitizing uncertainty, and
that the triangular curve be reported against T3's finding. Data: `benchmarks/literature/doane1989_fig9.csv`
(L1, 119 grid points per curve, uncertainty 0.010–0.046 %). Ours: `benchmarks/twist/data/doane_fig9.npz`
(`benchmarks/twist/doane_fig9.py`), total spurious power 1 − |S21(TE1,0; TE1,0)|² of `Twist(90, 0.508,
profile)` in WR187 used tall (a = 22.149 mm along u, b = 47.549 mm, TE1,0 with E along b), all propagating modes.

| curve | ours vs Fig. 9 (full multimode) | best-matching subset of ours |
| --- | --- | --- |
| constant | within the uncertainty at 9 of 119 points; max \|z\| = 84; rms difference 0.44 % | our TE0,1 conversion alone: rms 0.11 %, max \|z\| 22. The two-mode model (TE1,0 + TE0,1 only) is 3–14 σ low everywhere |
| cosine | within 1 σ at 15.1–16.35 GHz; below Fig. 9 above, by up to 0.10 % (9 σ) at 17.75–18 GHz | two-mode model: within 3.5 σ at every point, rms 0.022 % |
| triangular | 2.2–4.0 times Fig. 9 at every point (0.116 % vs 0.049 % at 15.0 GHz, 0.595 % vs 0.149 % at 18.0 GHz) | none: the two-mode model gives 0.0003–0.027 %, below Fig. 9 |

Where the differences come from (all in % of the input power):

* **Constant twist: TE2,1.** Our total has lobes of TE2,1 conversion, 0.48 % at 15.25 GHz, 0.53 % at 15.78, 0.61 %
  at 16.45, 0.71 % at 17.35 GHz. TE2,1 (cutoff 13.90 GHz) is coupled to TE1,0 at first order with |K| = 1.68, more
  strongly than TE0,1 (8/π² = 0.81); a constant rate switches the coupling on and off abruptly, so the converted
  power follows |sinc(Δβ L/2)|². Fig. 9 shows none of these lobes (0.05 % at 16.5 GHz where we have 0.70 %).
  Without TE2,1, our total is closer to Fig. 9 but not within its uncertainty (e.g. 0.09 % vs 0.051 % at 16.5 GHz,
  1.14 % vs 0.95 % at 17.5 GHz).
* **Cosine twist: TE0,1 and the multimode correction.** The cosine twist converts almost only to TE0,1; Fig. 9 is
  matched by the two-mode model better than by the full one. Above 16.5 GHz the other modes lower the TE0,1
  conversion by up to 0.10 %.
* **Triangular twist: second-order conversion.** 95–98 % of our spurious power is in TE1,2, TE0,3 and TM1,2, which
  TE1,0 reaches mostly at second order (TE1,2 and TM1,2 are not coupled to TE1,0 at first order: Δm = 0). These are
  the modes Doane & Anderson name as "important" in tall guide (p. 159). The rise of Fig. 9's triangular curve
  with frequency (×3 over 15–18 GHz) is real (L1) and has the sign of ours (×5).

What it is not (checked):

* **Not the coupling coefficients or the system.** Two independent derivations (T3, T3b) agree to 8e-18; the
  helical eigenvalues converge to an independent spectral solution of Maxwell's equations (T3b section 5,
  reproduced here: 5.37e-3 / 2.93e-3 / 1.45e-3 at 89 / 359 / 1443 modes, Richardson 1.8e-4). The "naive" and
  antisymmetric-only couplings give 0.17 / 0.38 / 1.03 % and 0.11 / 0.23 / 0.55 % for the triangular twist
  (T3, T3b): no variant reaches Fig. 9.
* **Not the implementation.** WP6 reproduces T3's prototype (an independent code with T3's C^T form) to 3.4e-7
  relative, and T3b's square-guide tables to 1.2e-3°.
* **Not slicing or truncation.** Slices: 128 → 4096 changes the 18 GHz triangular result in the 7th digit.
  Evanescent modes of all families (cutoff_factor 1.5–3) *raise* the triangular result by 1–7 % relative.
* **Not the profile reading.** "Triangular" is Doane eq. 17, transcribed from the scan (T3, T3b, and checked
  again by WP6 on PDF p. 7). "Cosine" is not defined in the paper; the half-cosine rate matches the curve, the
  raised cosine does not (T3).
* **Not the orientation.** In the ordinary orientation (a = 1.872 in) the triangular twist gives 0.017 / 0.012 /
  0.001 % (T3b), not Fig. 9's rising curve either.

## Assessment (WP6's view, not a ruling)

Fig. 9 shows "theoretical predictions" of their reference 2, which we do not have. Its cosine curve looks like a
calculation with TE1,0 and TE0,1 only, its constant curve like one without TE2,1, and its triangular curve like
neither. A reduced mode set in Doane's calculation would explain the constant and cosine curves; it does not
explain the triangular one, where a two-mode model gives far less than Fig. 9. COMSOL campaign 2, case E4 (15 and
18 GHz, triangular, cosine control at 18 GHz) measures the per-mode powers directly. WP6's predictions for E4 are
in `benchmarks/twist/data/e4_predictions.npz` and `docs/reports/WP6.md`.

**Note for the E4 comparison.** The decisive quantities are the converted powers per mode (TE1,2 0.080 % at 15 GHz
and 0.384 % at 18 GHz; Fig. 9 allows at most 0.05 % and 0.15 % in total). The TE1,0 transmission does not decide:
0.005 dB (15 GHz) and 0.026 dB (18 GHz) for us, 0.002 dB and 0.0065 dB for Fig. 9; at 15 GHz the difference is
below the 0.02 dB COMSOL criterion of CONVENTIONS section 8.

## Requested decision

1. Accept the WP6 acceptance rows "Fig. 9 constant and cosine" as *not met, explained above*, pending E4.
2. After E4: if COMSOL agrees with wrmodes (per-mode powers within 1e-4 of the input), close this report with
   "Fig. 9 is a reduced-mode calculation"; if COMSOL agrees with Fig. 9, reopen T3/T3b.
