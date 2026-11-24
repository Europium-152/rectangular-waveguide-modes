"""2D modal-port finite-element solver for coplanar components (wrmodes 1.0, WP3).

Components whose geometry does not vary along v (side b) are solved family by family (n) as two scalar
Helmholtz problems in the u-z plane (LSM: E_v, Dirichlet; LSE: H_v, Neumann; CONVENTIONS.md section 7), with
modal (Dirichlet-to-Neumann) ports and P2 Lagrange elements, and returned as GSMs in the TE/TM basis.

Requires the optional extra ``wrmodes[fem]`` (scikit-fem and gmsh)::

    pip install "wrmodes[fem]"

Determinism (WF4, `benchmarks/fem2d/determinism.py`). gmsh meshes with a fixed algorithm (Delaunay-Frontal, 6) and
a fixed random seed, so the mesh, and the GSM, do not depend on the gmsh version: gmsh 4.10.5, 4.12.2, 4.13.1 and
4.15.2 give bitwise identical GSMs for the test cases (H- and E-plane miters, a chamfered miter, two chained 45 degree
miters; Linux x86-64, scikit-fem 12.0.2). The node order of the mesh, which another platform or build could change,
moves S by at most 4e-14 (a fixed random renumbering), and a repeated solve in the same process is bitwise equal.
The remaining platform dependence is round-off in BLAS and the sparse solver (about 1e-13). Figure scripts may
therefore promise FEM results to 1e-10 in every entry of S across gmsh >= 4.10 and platforms
(`tests/test_fem2d.py::test_gsm_is_reproducible_across_gmsh_versions` checks a stored GSM to that bound); they
cannot promise more than the discretization error against the exact solution (at the default mesh density 6, a
1.5 times finer mesh changes the powers by up to 2.5e-4, the chained miters, and 5e-5 for single miters).

Public names: `Miter`, `Coplanar2D`, `FEMOptions`, `convergence_study`, and the outline helpers
`Outline`, `PortPlane`, `miter_outline`, `chained_miter_outline`, `arc_bend_outline`, `straight_outline`,
`mirror_outline`, and `mode_parity`.
"""
try:
    import skfem  # noqa: F401
    import gmsh  # noqa: F401
except (ImportError, OSError) as exc:          # gmsh raises OSError when its shared libraries are missing
    raise ImportError(
        "wrmodes.fem2d needs the optional dependencies scikit-fem and gmsh: pip install \"wrmodes[fem]\" "
        f"(import failed: {exc}). On Linux, the gmsh wheel also needs the system libraries libGLU and libXft "
        "(e.g. apt-get install libglu1-mesa libxft2)."
    ) from exc

from .geometry import Outline, PortPlane
from .components import (Coplanar2D, Miter, FEMOptions, miter_outline, chained_miter_outline, arc_bend_outline,
                         straight_outline, mirror_outline, mode_parity)
from .convergence import convergence_study

__all__ = ["Coplanar2D", "Miter", "FEMOptions", "convergence_study", "Outline", "PortPlane", "miter_outline",
           "chained_miter_outline", "arc_bend_outline", "straight_outline", "mirror_outline", "mode_parity"]
