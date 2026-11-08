"""2D modal-port finite-element solver for coplanar components (wrmodes 1.0, WP3).

Components whose geometry does not vary along v (side b) are solved family by family (n) as two scalar
Helmholtz problems in the u-z plane (LSM: E_v, Dirichlet; LSE: H_v, Neumann; CONVENTIONS.md section 7), with
modal (Dirichlet-to-Neumann) ports and P2 Lagrange elements, and returned as GSMs in the TE/TM basis.

Requires the optional extra ``wrmodes[fem]`` (scikit-fem and gmsh)::

    pip install "wrmodes[fem]"

Public names: `Miter`, `Coplanar2D`, `FEMOptions`, `convergence_study`, and the outline helpers
`Outline`, `PortPlane`, `miter_outline`, `arc_bend_outline`, `straight_outline`.
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
from .components import Coplanar2D, Miter, FEMOptions, miter_outline, arc_bend_outline, straight_outline
from .convergence import convergence_study

__all__ = ["Coplanar2D", "Miter", "FEMOptions", "convergence_study", "Outline", "PortPlane", "miter_outline",
           "arc_bend_outline", "straight_outline"]
