# Rectangular Waveguide Modes (`wrmodes`)

Calculate all rectangular modes below a given frequency and for a given waveguide size.
Calculate mode-conversion losses in curved rectangular waveguides with coupled-mode theory.

# Install

```
pip install wrmodes
```

# Quick start

```python
import numpy as np
import wrmodes as wr

# WR-90, E-plane bend of the TE10 mode: side a (0.4 in) lies in the bend plane
a, b = 0.4 * 0.0254, 0.9 * 0.0254
L = 0.15                                  # bend length (m)
R = L / np.radians(60)                    # 60 degree constant-curvature bend
curvature = lambda z: np.ones_like(z) / R # curvature (1/m) along the bend, z in [0, L]

modes, result = wr.solve_mode_coupling(50e9, a, b, ('TE', 0, 1), L, curvature)
j = result.input_index
print("mode-conversion loss: %.3f dB" % (-20 * np.log10(abs(result.T[j, j]))))
for (name, *_), t in zip(modes, result.T[:, j]):
    print(f"{name:8s} output power {abs(t)**2:.4f}")
```

Mode names follow the convention of the paper: `a` is the side in the bend plane, `b` the side
perpendicular to it, and the vertical index `n` refers to `b`. The input mode `('TE', 0, 1)` above is the
usual TE10 mode of a guide whose short side lies in the bend plane (E-plane bend).

# Solvers

`solve_mode_coupling(..., method=...)` offers two numerical methods for the same coupled-mode equations.

* **`'transfer-matrix'` (default since 0.1.0).** The coupled-mode equations are linear,
  `dy/dz = (D + kappa(z) K) y`, so a bend is fully described by its transfer matrix. It is computed as a
  product of matrix exponentials over slices of the bend (fourth-order Magnus integrator). The exponentials
  propagate the phase `exp(-i beta z)` of every mode exactly, so the slices only have to resolve the
  curvature profile and the beat lengths between modes; a constant curvature is exact. The number of slices
  is chosen adaptively until the estimated error is below `atol` (default `1e-6` of the input power for the
  power of every output mode, about `4e-6` dB on the loss). The result also contains the full transmission
  (`result.T`) and reflection (`result.R`) matrices for all input modes, and the amplitude profile along the
  bend (`result.x`, `result.y`, or at chosen positions with `z_eval=`).
* **`'bvp'`.** Collocation with `scipy.integrate.solve_bvp`, the method of `wrmodes <= 0.0.5`. Extra keyword
  arguments (`tol`, `max_nodes`, ...) are passed to `solve_bvp`; `resolution` sets the initial grid.

The transfer-matrix solver works with many small matrices, for which a multithreaded BLAS only adds overhead
(several times slower on machines with many cores). It therefore limits BLAS to one thread while it runs
(`blas_threads=1`, through `threadpoolctl`; `blas_threads=None` leaves the threading unchanged). To use several
cores, solve different frequencies in parallel processes. `benchmarks/diagnose_runtime.py` shows the effect on
a given machine.

For both methods, `result.y[p, -1]` is the output amplitude of forward mode `p` and `result.y[N + p, 0]` the
reflected amplitude at the input, for a unit-amplitude input mode.

Cascading bends and straight sections (all modes, phases included):

```python
modes, T_bend, R_bend, beta = wr.transmission_matrix(f, a, b, n, L, curvature)   # family with index n
T_line = T_bend @ wr.straight_section(beta, 2.0) @ T_bend
```

# Validation

The `tests/` folder contains unit tests (`pytest`), and `benchmarks/` the scripts that reproduce the
figures of the accompanying paper (comparison with Zhang et al. 2012 and with COMSOL Multiphysics) and the
validation of the transfer-matrix solver.
