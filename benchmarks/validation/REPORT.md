# Validation of the transfer-matrix solver (wrmodes 0.1.0)

This folder validates `wrmodes.solve_transfer_matrix`, the solver that replaced the collocation
(`scipy.integrate.solve_bvp`) solver as the default of `solve_mode_coupling` in wrmodes 0.1.0 and that is
described in Section 6.1 of the revised JINST paper. Every number below is produced by the scripts in this
folder. They write their output to `results/`, which is not under version control.

```
cd benchmarks/validation
python check_coupling_integrals.py     # 1. closed-form coupling integrals vs quadrature        (seconds)
python run_accuracy_suite.py           # 2. 1792-case accuracy suite vs reference solution      (~1 h, all cores)
python analyze_accuracy_suite.py       #    summary -> results/accuracy_summary.json
python check_properties.py             # 3. physical properties and edge cases                  (~1 min)
python check_convergence_order.py      # 4. observed order of the Magnus slicing                (~5 min)
python check_paper_dataset.py          # 5. full data set of the SPARC figure (2320 points)     (~30 min, all cores)
python analyze_paper_dataset.py        #    summary -> results/paper_dataset_summary.json
python run_timings.py                  # 6. run times, serial                                   (~10 min)
```

## The reference solution

`vcommon.reference` integrates the fundamental matrix, Φ' = (D + κ(z)K)Φ with Φ(0) = I, using the 8th-order
Runge–Kutta method DOP853 at rtol = 1e-11. It integrates piecewise between curvature discontinuities and uses
the exact matrix exponential when the curvature is constant. T and R follow from Φ through the same algebraic
boundary conditions as the solver. The reference shares only the coupling matrices D and K with the solver
under test, and those matrices are validated separately (test 1). It shares none of the solver's numerics: no
Magnus slicing, no adaptive doubling and no batched exponentials. The accuracy floor of the reference is about
1e-9 in power, which is visible at the bottom of the convergence curves in test 4.

## 1. Coupling integrals (`check_coupling_integrals.py`)

The closed-form integrals I1–I6 (eqs. 5.6–5.11 of the paper) were compared with 120×120-node Gauss–Legendre
quadrature of their definitions for WR-90 in both orientations and for a square guide. In total, 2848 integrals
were checked.

* The maximum relative error is **1.1e-12**.
* The selection rules hold to rounding:
  * Integrals between modes whose m_p and m_q have the same parity are below 1e-16 relative to scale.
  * Integrals between modes with n_p ≠ n_q are below 1e-16.

## 2. Accuracy suite (`run_accuracy_suite.py`)

The suite covers 8 guides/orientations × 4 bend geometries × 7 curvature profiles × 8 frequencies = **1792
cases**.

* **Guides:**
  * WR-90 and WR-28, each in the H- and E-plane orientation.
  * A 44×30 mm guide and a 20×20 mm square guide, each in both orientations.
* **Geometries:**
  * 30° bend with length 2D (the sharp case: κ/β up to 1.16 for the input mode).
  * 90° bends with lengths 7D and 26D.
  * 180° bend with length 90D.
* **Profiles:**
  * constant;
  * hyperbolic secant;
  * half-sine;
  * triangular (kink at L/2, given as a breakpoint);
  * S-bend (sign reversal at L/2);
  * piecewise-constant step at L/3 (given as a breakpoint);
  * the same step without the breakpoint, kept to document that limitation.
* **Frequencies:** 8 frequencies across each guide's band, giving up to 13 coupled modes and losses up to
  27 dB.

**Default settings** (`error_control='input'`, `atol=1e-6`; 1536 cases, all profiles except the step without
breakpoint):

| quantity | result |
|---|---|
| converged | 1536 / 1536 |
| max error in the output power of any mode | **9.6e-7** (≤ atol in every case) |
| loss error of the input mode | max 2.6e-5 dB, 99th percentile 4.8e-6 dB, median 8.8e-10 dB |
| error estimate vs actual error (218 cases with actual error > 1e-8) | within 10 % in 99.5 % of cases; the actual error never exceeds the estimate by more than 6 % |
| slices per case | median 128, max 4096 |
| unitarity defect of [T; R] (`error_control='full'`) | max 9.5e-13 |
| `error_control='full'`, max error in any entry of T / R | 5.5e-7 / 6.7e-7 (1527 / 1536 converged, see below) |

**Step profile without a breakpoint.** A curvature discontinuity inside a slice reduces the method to first
order. In 208 of the 256 cases the solver stops at `max_slices` and warns, with a message that suggests passing
`breakpoints`. The worst power error is 5.6e-4, and the loss error is at most 3e-3 dB. With the breakpoint,
the same cases behave like every other profile.

**`error_control='full'` in long bends.** In nine cases, all 180° WR-90 bends 2 m long between 68 and 90 GHz
(βL ≈ 3400–3900 rad), the solver reaches the default `max_slices=4096` before it can confirm the stricter
'full' error control. It then warns, although the actual errors are 1e-8. These cases start from the minimum
slice count, which is capped at `max_slices // 8 = 512`. With `max_slices=16384`, they converge at 8192 slices
with clean fourth-order ratios.

**Previous collocation solver** (`method='bvp'`, paper settings, 768 cases, 90° geometries):

* Maximum loss error 0.58 dB; median 5.5e-4 dB; 0.035 dB where the loss is below 1 dB.
* About 6 times slower than the transfer-matrix solver on the same cases (65 s vs 11.5 s in total).

## 3. Physical properties and edge cases (`check_properties.py`)

The six cases cover four guides, 60°–180° bends, all profile types and 5–13 coupled modes.

* **Power conservation:**
  * The columns of [T; R] are orthonormal to ≤ 1.5e-13.
  * The net power flow Σ|A⁺|² − |A⁻|² is constant along the bend to ≤ 4e-14, and equals 1 − (reflected
    power).
* **Reciprocity:** traversing the bend backwards, with curvature −κ(L−z), gives |T_rev| = |T|ᵀ to ≤ 3.4e-15.
* **Mirror symmetry:** bending the other way (κ → −κ) leaves |T| unchanged; the difference is exactly 0.
* **Straight guide:** T = diag(e^{−iβL}) and R = 0 to 2e-15.
* **Edge cases** (all finite, without warnings, power conserved to 1e-14):

  | case | error in output power vs reference |
  |---|---|
  | highest retained mode 1 ppm above cutoff | 3.7e-10 |
  | highest retained mode 1 ppb above cutoff | 3.3e-10 |
  | higher-order input mode TM1,1 | 3.3e-10 |
  | higher-order input mode TE2,1 | 3.2e-10 |
  | 90° bend with R0 = 1.5 cm | 3e-15 |
  | single-mode family | 3e-12 |

* **Amplitude profile along z** (`z_eval`): the power in every mode agrees with a DOP853 integration of the state
  vector to ≤ 1.5e-10.
* **Reflected power:** for the paper's 15 cm 60° E-plane bend at 60 GHz, the power reflected at the input is
  4.5e-9.

## 4. Convergence order (`check_convergence_order.py`)

This test uses uniform slicing with 8 to 2048 slices, without adaptivity, on five cases with N = 5 to 27 modes
(βL up to 1100 rad). The fitted order of the output-power error, over the asymptotic range
1e-8 < err < 1e-4, is:

| case | N | order |
|---|---|---|
| WR-90 H, 0.6 m 60° sech, 89.5 GHz | 13 | 3.98 |
| WR-90 tall-guide E, 0.6 m 60° sech, 89.5 GHz | 27 | 3.87 |
| WR-90 E, 0.15 m 60° sech, 60 GHz | 9 | 4.15 |
| WR-90 H, 0.4 m 90° half-sine, 80 GHz | 12 | 4.01 |
| 44×30 H, 1.0 m 90° triangular, 18 GHz | 5 | 4.07 |

These orders match the 4th order expected of the two-point Gauss Magnus integrator. Figure 8(c) of the paper
shows the first two cases.

## 5. The paper's data set (`check_paper_dataset.py`)

The test recomputes all 16 curves of the SPARC bend figure: 2 bends × 2 profiles × 4 orientations × 145
frequencies, from 18 to 90 GHz in 0.5 GHz steps. That is **2320 points** with up to 27 coupled modes.

* **Transfer-matrix solver (default settings):**
  * 2320 / 2320 converged.
  * Maximum loss error vs the reference 8.5e-7 dB; median 3e-11 dB.
  * Maximum power error 2.0e-7.
* **Previous collocation solver (paper V1/V2 settings):**
  * 0.032 dB for the hyperbolic secant TE10 bends.
  * 0.046 dB wherever the loss is below 1 dB.
  * 2.05 dB at the 32 dB resonance of the 40 cm constant-curvature tall-guide E-plane bend.
* **Peak-loss table:** Table 1 of the paper (peak losses and the > 0.5, 1 and 3 dB intervals) is identical with
  both solvers.
* **Run time:**
  * Total (run in parallel on 2 cores): transfer matrix 179 s, collocation 1299 s, reference 329 s, i.e.
    **7.3×** faster than collocation.
  * Per point: 2.3–5× faster for N ≤ 8 and 5–10× for N ≥ 9.

## 6. Run times (`run_timings.py`)

Serial, best of 3 runs. Machine: Linux x86-64 cloud VM (2 cores), Python 3.13, NumPy 2.5, SciPy 1.18, with
BLAS at one thread (`OPENBLAS_NUM_THREADS=1` in that environment; the solver now enforces this itself,
see `blas_threads`).

**a) The COMSOL case of the paper** (22 frequencies, 18–60 GHz, 15 cm 60° sech E-plane bend):

* transfer matrix **0.15 s**;
* collocation (V1 settings) 0.74 s, a factor of 4.9.

**b) Time per frequency point, 60 cm 60° sech bend:**

| orientation | f (GHz) | N | transfer matrix (s) | collocation (s) | ratio |
|---|---|---|---|---|---|
| TE10 H-plane | 19.5 | 2 | 0.005 | 0.004 | 0.9 |
| TE10 H-plane | 39.5 | 6 | 0.013 | 0.080 | 6.0 |
| TE10 H-plane | 59.5 | 9 | 0.061 | 0.295 | 4.9 |
| TE10 H-plane | 89.5 | 13 | 0.138 | 1.022 | 7.4 |
| tall-guide TE01 E-plane | 29.5 | 7 | 0.019 | 0.093 | 4.8 |
| tall-guide TE01 E-plane | 59.5 | 17 | 0.252 | 1.561 | 6.2 |
| tall-guide TE01 E-plane | 89.5 | 27 | 0.806 | 7.702 | 9.6 |

**c) Bend length** (60° sech, TE10 H-plane). The number of slices grows slowly with length, while the
collocation grid grows in proportion to it:

| f (GHz) | L (m) | slices | transfer matrix (s) | collocation (s) | ratio |
|---|---|---|---|---|---|
| 60 | 0.15 | 512 | 0.029 | 0.112 | 3.9 |
| 60 | 0.6 | 1024 | 0.056 | 0.291 | 5.2 |
| 60 | 2.0 | 2048 | 0.128 | 0.834 | 6.5 |
| 60 | 5.0 | 2048 | 0.139 | 1.997 | 14.3 |
| 89.5 | 0.15 | 512 | 0.067 | 0.370 | 5.5 |
| 89.5 | 0.6 | 1024 | 0.133 | 0.960 | 7.2 |
| 89.5 | 5.0 | 2048 | 0.330 | 9.115 | 27.6 |

The full table is in `results/timings.json`.

## Choice of the minimum number of slices

The solver starts from a minimum number of slices such that every pair of coupled modes that can exchange a
relevant amount of power slips by at most `max_phase_per_slice` (default π, i.e. half a beat length) of
relative phase per slice. Below that, the Richardson estimate is unreliable: coarse results can be stuck at the
same wrong value and agree with each other.

* **No minimum.** The first version started at 8 slices and doubled. 34 of the 1536 cases (2.2 %) ended above
  atol, the worst at 3e-5, with estimates too low by up to ~100×.
* **Rule variants.** These were tested on all 1536 cases: limits of 8, 6 and 4 rad, and none, each with single
  or double ratio acceptance. 4 rad with double acceptance was the most relaxed clean combination.
* **π vs 4 rad.** π is the natural scale (half a beat length; the convergence scale of the Magnus series). With
  starting counts rounded up to 8·2^k, as for 4 rad, it gives the same maximum errors as 4 rad and a slightly
  more reliable estimate (99.5 % within 10 %, never more than 6 % optimistic). It costs ~15–19 % more time.
  π is the default.
* **Rejected: starting exactly at the minimum.** Starting exactly at the minimum, rather than rounding up to
  8·2^k, was also tried. It saved no time, because the final count is set mostly by the acceptance rules. It
  gave 3 optimistic estimates (one by ×120, still 37× below atol). It also accepted one step-without-breakpoint
  case with an error of 6.6e-6. It was therefore not adopted.

## Change of the acceptance rule during the validation

The first version of the solver also accepted a result after a *single* comparison when the two results
already agreed to within atol·1e-3. The accuracy suite showed that this shortcut can be fooled: in 4 of the
1536 cases, two coarse results agreed by coincidence while both were still off by up to 2e-7 in power. That
was still inside atol, but the error estimate was too low by a factor of up to 300.

The shortcut now requires two successive differences below atol·1e-3, i.e. three agreeing results. All numbers
in this report were obtained with the final rule. The cost is about 1.8× more slices on average, mostly in
cases that converge quickly. Each of the 4 cases now reaches the accuracy floor of the reference.

## Known limitations

* **Curvature discontinuities.** A discontinuity in κ(z) or its derivative that does not fall on a slice
  boundary reduces the convergence order. Pass its position in `breakpoints`. When the solver hits
  `max_slices`, it warns with this suggestion.
* **Validity of the theory.** The first-order coupled-mode theory itself, as opposed to its numerical solution,
  assumes κ/β ≪ 1. The accuracy suite includes cases up to κ/β = 1.16 to stress the numerics, but such cases
  are outside the physical validity range of the model. Section 6.1 of the paper discusses this.
* **Evanescent modes.** Only propagating modes are retained, which keeps the algebraic boundary conditions well
  conditioned. Section 6.3 / figure 8(a) of the paper shows that modes near cutoff change the loss by about
  1e-4 dB at most.
