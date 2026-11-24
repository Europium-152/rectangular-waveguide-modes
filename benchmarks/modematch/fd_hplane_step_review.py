"""WF4: the independent finite-difference values of the symmetric H-plane step from the review of WP4 (PR #4), as a
data file with `meta` (CONVENTIONS.md section 9).

The reviewer solved the symmetric H-plane step a'/a = 0.8 with a 2D finite-difference model (scalar E_y, Dirichlet
walls, discrete modal ports, Richardson extrapolation in the grid step; the review's script fd2d.py, not in the
repository) and reported (docs/reports/WP4.md, "H-plane, independent FD reference"):

  * a/lambda = 0.85: X/Z0 = 11.07 +- 0.05 (1 sigma, stated by the reviewer);
  * a/lambda = 1.4:  2 a Z0 / (lambda_g X) = 0.1077 +- 0.0003 (Richardson estimates of the N = 100/200/400 runs
    with two junction-plane placements lie in 0.10756-0.10761; the +- 0.0003 is WP4's bound).

X is the shunt reactance of Marcuvitz's circuit (sec. 5.24a) at the junction plane, Z0 the impedance of the wide
guide. Used by tests/test_steps.py::test_hplane_symmetric_step_against_fd_reference (registry entry
wp4-review-fd-hplane-step).

    python benchmarks/modematch/fd_hplane_step_review.py   ->  data/fd_hplane_step_review.npz
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

if __name__ == "__main__":
    m = meta("benchmarks/modematch/fd_hplane_step_review.py",
             "Symmetric H-plane step a'/a = 0.8: independent 2D finite-difference values of the WP4 review (PR #4)",
             a_prime_over_a=0.8, method="2D finite differences, scalar E_y, Dirichlet walls, discrete modal ports, "
             "Richardson extrapolation in the grid step (review of PR #4, script not in the repository)",
             source="docs/reports/WP4.md, 'H-plane, independent FD reference'", sigma="1 sigma",
             quantities={"X/Z0": "shunt reactance at the junction plane / Z0 of the wide guide (Marcuvitz 5.24a)",
                         "2aZ0/(lambda_g X)": "Marcuvitz Fig. 5-24-3 ordinate"})
    print(save("fd_hplane_step_review.npz", m, a_over_lambda=np.array([0.85, 1.4]),
               quantity=np.array(["X/Z0", "2aZ0/(lambda_g X)"]), value=np.array([11.07, 0.1077]),
               sigma=np.array([0.05, 0.0003])))
