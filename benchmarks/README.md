# Benchmarks for the JINST paper

These scripts reproduce the figures and numbers of *Optimizing rectangular waveguide transmission lines for a
plasma reflectometry diagnostic using Coupled-Mode Theory* (D. Hachmeister et al., JINST 2026) with `wrmodes`.

Since version 0.1.0, `wrmodes.solve_mode_coupling` uses the transfer-matrix (Magnus) solver by default. The
previous collocation solver is still available as `method='bvp'`. Setting the environment variable
`WRMODES_METHOD=bvp` makes the scripts that import `common.py` use the collocation solver instead.

## Figures and tables of the revised paper

| Script | Paper item | Runtime (laptop) |
|---|---|---|
| `fig5_zhang_comparison.py` | Fig. 5: comparison with Zhang et al. (2012), 44x30 mm guide, 9.5 GHz | seconds |
| `fig6_comsol_comparison_markers.py` | Fig. 6: 15 cm 60° E-plane bends vs COMSOL; triangles at the 22 COMSOL frequencies (data in `comsol/`) | seconds |
| `paper_figures.py power` | Fig. 7: power along the bend and net power flow | seconds |
| `paper_figures.py convergence` | Fig. 8: (a) truncation, (b) error vs tolerance, (c) convergence order | ~5 min |
| `paper_figures.py sparc` | Fig. 9 and Table 1: 60 cm 60° and 40 cm 90° WR-90 bends, 4 orientations, 18–90 GHz | ~2 min |
| `paper_figures.py truncation`, `length`, `sequence` | Section 7: truncation sensitivity, 90° E-plane bend length, bend sequences | ~1–5 min each |
| `runtime_comsol_case.py` | Run time for the COMSOL model of Fig. 6 (same bend, same 22 frequencies). `--repeat N` reports the best of N runs; `--method {transfer-matrix,bvp}` selects the solver | < 1 s |
| `diagnose_runtime.py` | Times the solver and its building blocks with and without multithreaded BLAS on the current machine (needs `threadpoolctl`); writes `diagnose_runtime.txt` | seconds |

`paper_figures.py all` runs every subcommand. Figures go to `figures/` (or `--out DIR`). The numbers quoted in
the paper are printed and saved to `results/paper_numbers.json`.

The original V1 scripts, `fig6_comsol_comparison.py` (dense line) and `fig7_sparc_bends.py`, are kept for
reference.

`common.py` holds the WR-90 dimensions, the curvature profiles (eqs. 4.6–4.7 of the paper) and the solver
settings.

`comsol/` contains the COMSOL 6.2 model (Java export, `hyperbolic_secant_E_bend_60_deg_lowfreq_parametric.java`)
and the exported `emw.S21dB` data for the two bends of Fig. 6.

## Validation of the solver

`validation/` holds a comprehensive validation of the transfer-matrix solver. It covers:

* coupling integrals;
* a 1792-case accuracy suite against an independent reference solution;
* physical properties (power conservation, reciprocity, symmetry);
* edge cases;
* the convergence order;
* the full data set of the paper;
* run times.

See [`validation/REPORT.md`](validation/REPORT.md) for the results and the commands to reproduce them.
