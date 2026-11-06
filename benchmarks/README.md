# Benchmarks for the JINST paper

Scripts reproducing the figures of *Optimizing rectangular waveguide transmission lines for a plasma
reflectometry diagnostic using Coupled-Mode Theory* (D. Hachmeister et al., JINST 2026) with the published
`wrmodes` package (`pip install wrmodes`).

| Script | Paper item | Runtime (laptop) |
|---|---|---|
| `fig5_zhang_comparison.py` | Fig. 5, comparison with Zhang et al. (2012), 44x30 mm guide, 9.5 GHz | seconds |
| `fig6_comsol_comparison.py` | Fig. 6, 15 cm 60° E-plane bends vs COMSOL (data in `comsol/`) | seconds |
| `fig7_sparc_bends.py` | Fig. 7 and Table 1, 60 cm 60° and 40 cm 90° WR-90 bends, 4 orientations | ~30 min (`--quick` for 2 GHz steps) |

`common.py` holds the WR-90 dimensions, the curvature profiles (eqs. 4.6–4.7 of the paper) and the solver
settings used throughout the paper (all propagating modes, 10 grid points per guided wavelength, `tol=0.5`).

`comsol/` contains the COMSOL 6.2 model (Java export, `hyperbolic_secant_E_bend_60_deg_lowfreq_parametric.java`)
and the exported `emw.S21dB` data for the two bends of Fig. 6.
| `runtime_comsol_case.py` | Runtime comparison with the COMSOL model of Fig. 6 (same bend, same 22 frequencies); `--repeat N` for best-of-N | < 1 s |
