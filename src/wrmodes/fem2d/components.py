"""Coplanar components solved with the 2D modal-port finite-element method (WP3).

`Coplanar2D` takes any core outline (straight segments and arcs) in the plane of side a; `Miter` builds the outline
of a plain or chamfered miter bend. Both return GSMs in the TE/TM basis of CONVENTIONS.md, in the local frame of the
component (frame tag 'H' or 'E', CONVENTIONS.md section 2). They only couple modes within a family n (no variation
along v), and solve each family as two scalar problems (LSM and LSE, CONVENTIONS.md section 7).
"""
import dataclasses
import numpy as np
import scipy.constants as cst

from ..components import Component
from ..gsm import GSM
from ..modes import Mode, ModeBasis, propagation_constant
from .geometry import Outline, PortPlane, build_mesh
from .normalization import ls_amplitude_factors
from .solver import Discretization, solve_scalar

__all__ = ["Coplanar2D", "Miter", "FEMOptions"]


@dataclasses.dataclass(frozen=True)
class FEMOptions:
    """Discretization settings of the 2D solver.

    mesh_density : elements per transverse wavelength 2 pi / k_t; each family n gets its own mesh; default 10.
    max_element : largest element as a fraction of the smallest port width a; default 1/4.
    corner_refinement : element size at re-entrant corners as a fraction of the nominal size; default 0.05.
    n_evanescent : evanescent port modes per port and polarization in the modal condition, M; default 10.
    port_distance : distance (m) from each port plane to the modal port; None means one guide width a.
    order : polynomial order of the Lagrange elements (2 or 3); default 2.
    """
    mesh_density: float = 10.0
    max_element: float = 0.25
    corner_refinement: float = 0.05
    n_evanescent: int = 10
    port_distance: float = None
    order: int = 2


class Coplanar2D(Component):
    """A two-port component whose geometry does not vary along v, given by its core outline.

    Parameters
    ----------
    outline : wrmodes.fem2d.geometry.Outline
        Core outline in the u-z plane, with its two port planes (and their de-embedding distances).
    b : float
        Side b (m), the dimension along v, common to both ports.
    frame : str or None
        Frame tag of the port bases ('H', 'E' or None), CONVENTIONS.md section 2.
    mirror : (origin, direction), optional
        Symmetry line of the outline (port 2 the mirror image of port 1). The mesh is then built symmetric, so
        that S11 = S22 holds to rounding.
    name : str
    options : FEMOptions or keyword arguments of FEMOptions

    Notes
    -----
    Default basis: the propagating modes (`cutoff_factor` = 1). The solver does not need evanescent modes in the
    returned basis: the evanescent fields of the discontinuity are carried by the arms and by the M evanescent
    modes of the modal ports. Evanescent modes requested in a basis are de-embedded from the modal ports over the
    arms with exp(|beta| L): their entries lose accuracy roughly as exp(2 |beta| L) times the discretization error,
    so keep `port_distance` short (e.g. a/4, with enough evanescent port modes) when they are needed.
    """

    def __init__(self, outline, b, frame=None, mirror=None, name="", options=None, **kw):
        self.outline = outline
        self.b = float(b)
        self.frame = frame
        self.mirror = mirror
        self.name = name or "coplanar 2D"
        opts = options or FEMOptions()
        self.options = dataclasses.replace(opts, **kw) if kw else opts

    # -------------------------------------------------------------------------------------- interface
    @property
    def port_widths(self):
        return self.outline.ports[0].a, self.outline.ports[1].a

    def with_options(self, **kw):
        """A copy of this component with other FEMOptions."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.options = dataclasses.replace(self.options, **kw)
        return new

    def default_basis(self, f, cutoff_factor=1.0):
        """Propagating modes of each port at f (Hz), plus evanescent ones below cutoff_factor * f."""
        a1, a2 = self.port_widths
        return (ModeBasis.propagating(f, a1, self.b, cutoff_factor=cutoff_factor, frame=self.frame),
                ModeBasis.propagating(f, a2, self.b, cutoff_factor=cutoff_factor, frame=self.frame))

    def gsm(self, f, basis1=None, basis2=None):
        """GSM at f (Hz) in the TE/TM basis of CONVENTIONS.md, at the reference planes of the outline.

        The families n of the bases are solved independently; a family with k_t^2 <= 0 has no propagating mode
        and cannot be in the bases (ValueError). The returned GSM carries the solver diagnostics in `fem_info`.
        """
        if basis1 is None or basis2 is None:
            d1, d2 = self.default_basis(f)
            basis1 = d1 if basis1 is None else basis1
            basis2 = d2 if basis2 is None else basis2
        self._check_basis(basis1, 0)
        self._check_basis(basis2, 1)
        S, info = self._solve(f, basis1, basis2)
        g = GSM.from_matrix(S, basis1, basis2, f, self.name)
        g.fem_info = info
        return g

    # -------------------------------------------------------------------------------------- internals
    def _check_basis(self, basis, k):
        a = self.port_widths[k]
        if not (np.isclose(basis.a, a, rtol=1e-9) and np.isclose(basis.b, self.b, rtol=1e-9)):
            raise ValueError(f"basis of port {k + 1} has a x b = {basis.a} x {basis.b}, the component "
                             f"{a} x {self.b}")
        if basis.frame not in (None, self.frame):
            raise ValueError(f"basis of port {k + 1} has frame {basis.frame!r}, the component {self.frame!r}")

    def element_size(self, f, n):
        """Nominal element size (m) for family n at f (Hz): 2 pi / k_t / mesh_density, at most max_element * a."""
        o = self.options
        k = 2 * np.pi * f / cst.c
        kt = np.sqrt(k * k - (n * np.pi / self.b) ** 2)
        return min(2 * np.pi / kt / o.mesh_density, o.max_element * min(self.port_widths))

    def _mesh(self, h):
        o = self.options
        Lp = min(self.port_widths) if o.port_distance is None else o.port_distance
        md = build_mesh(self.outline, h, Lp, refine_factor=o.corner_refinement, mirror=self.mirror)
        return md, Discretization(md, order=o.order)

    def _solve(self, f, basis1, basis2):
        o = self.options
        k = 2 * np.pi * f / cst.c
        bases = (basis1, basis2)
        families = sorted(set(basis1.families()) | set(basis2.families()))
        for n in families:
            if k * k - (n * np.pi / self.b) ** 2 <= 1e-10 * k * k:
                raise ValueError(f"family n = {n} has k_t^2 <= 0 at {f / 1e9:.6g} GHz (no propagating mode); "
                                 "leave it out of the bases (CONVENTIONS.md section 7)")
        rot = [ls_amplitude_factors(bs, f) for bs in bases]
        N1, N2 = len(basis1), len(basis2)
        off = (0, N1)
        S_ls = np.zeros((N1 + N2, N1 + N2), dtype=complex)
        info = {"h": {}, "ndof": [], "elements": {}}
        meshes = {}
        for n in families:
            kt2 = k * k - (n * np.pi / self.b) ** 2
            # one mesh per family, sized for its own k_t (meshes of equal size are shared)
            h = self.element_size(f, n)
            key = round(h / self.element_size(f, families[0]), 9)
            if key not in meshes:
                meshes[key] = self._mesh(h)
            md, disc = meshes[key]
            Lp = md.port_distance
            info["h"][n], info["elements"][n], info["port_distance"] = md.h, md.t.shape[1], Lp
            for pol in ("LSM", "LSE"):
                rows = [[l for l, lab in enumerate(rot[p][1]) if lab[0] == pol and lab[2] == n] for p in range(2)]
                if not (rows[0] or rows[1]):
                    continue
                port_modes, port_betas, excite = [], [], []
                for p in range(2):
                    a = self.port_widths[p]
                    m0 = 1 if pol == "LSM" else 0
                    m_prop = int(np.floor(np.sqrt(kt2) * a / np.pi - 1e-12))
                    m_hi = max([m_prop + o.n_evanescent] + [rot[p][1][l][1] for l in rows[p]])
                    ms = list(range(m0, m_hi + 1))
                    port_modes.append(ms)
                    port_betas.append([propagation_constant(Mode("TE", m, n), a, self.b, f) for m in ms])
                    excite.append([ms.index(rot[p][1][l][1]) for l in rows[p]])
                S2, inf = solve_scalar(disc, kt2, pol, port_modes, port_betas, excite)
                info["ndof"].append(((n, pol), inf["ndof"]))
                # 2D amplitudes -> LS amplitudes, then from the modal ports to the reference planes
                idx, fac, beta, sgn, dist = [], [], [], [], []
                for p in range(2):
                    dk = self.outline.ports[p].deembed + Lp
                    for l, e in zip(rows[p], excite[p]):
                        idx.append(off[p] + l)
                        fac.append(rot[p][2][l])
                        beta.append(port_betas[p][e])
                        sgn.append(1.0 if p == 0 else -1.0)
                        dist.append(dk)
                fac, beta, sgn, dist = map(np.asarray, (fac, beta, sgn, dist))
                T = S2 * fac[None, :] / fac[:, None]
                if pol == "LSE":
                    T = -T * sgn[:, None] * sgn[None, :]
                ph = np.exp(1j * beta * dist)
                T = ph[:, None] * T * ph[None, :]
                S_ls[np.ix_(idx, idx)] = T
        Q = np.zeros((N1 + N2, N1 + N2), dtype=complex)
        Q[:N1, :N1] = rot[0][0]
        Q[N1:, N1:] = rot[1][0]
        return Q.T @ S_ls @ Q, info


class Miter(Coplanar2D):
    """Plain or chamfered miter bend, solved with the 2D modal-port FEM.

    Parameters
    ----------
    angle : float
        Bend angle theta (rad), 0 < theta < pi. The guide turns towards -u (outer wall at u = +a/2, CONVENTIONS.md
        section 2), and the mirror is perpendicular to the bisector of the two arms.
    chamfer : float or None
        Alessandri's d (1994, Fig. 1): distance from the inner corner to the mirror, along the bisector, divided by
        the in-plane side a. None is the plain miter, whose mirror joins the two outer corners at the reference
        planes: d = cos(theta/2). d = 1/cos(theta/2) is the unmitered (sharp) corner. For 90 degrees, Ma et al.
        (1997) use the cut length c along the outer walls: d = (2 - c/a)/sqrt(2).
    plane : 'H' or 'E'
        Bend plane of the global TE10 mode. 'H': a = w, b = h. 'E': a = h, b = w (local frame, CONVENTIONS.md
        section 2; the GSM is in that local frame, tagged 'E').
    guide : (w, h) or an object with attributes w and h (m), keyword only.
    options / keyword arguments : FEMOptions.

    Reference planes (CONVENTIONS.md section 6): perpendicular to each arm through the inner corner, at
    `arm_length` = (a/2) tan(theta/2) from the intersection of the arm axes. The GSM refers to these planes. When the
    mirror cuts the outer wall before a reference plane (d < cos(theta/2)), the field is solved from the last
    uniform cross-section of the arm and shifted analytically to the reference plane.
    """

    def __init__(self, angle, chamfer=None, plane="H", *, guide, name="", options=None, **kw):
        w, h = (guide.w, guide.h) if hasattr(guide, "w") else guide
        if plane not in ("H", "E"):
            raise ValueError("plane must be 'H' or 'E'")
        a, b = (w, h) if plane == "H" else (h, w)
        theta = float(angle)
        if not 0 < theta < np.pi:
            raise ValueError("the bend angle must be in (0, pi) rad")
        d = np.cos(theta / 2) if chamfer is None else float(chamfer)
        if not 0 < d:
            raise ValueError("chamfer d must be positive")
        self.angle, self.chamfer, self.plane, self.a = theta, chamfer, plane, a
        outline, mirror = miter_outline(a, theta, d)
        super().__init__(outline, b, frame=plane, mirror=mirror,
                         name=name or f"{plane}-plane miter {np.degrees(theta):.4g} deg"
                         + ("" if chamfer is None else f" d/a={d:.4g}"), options=options, **kw)

    @property
    def arm_length(self):
        """Distance (m) from the intersection of the arm axes to each reference plane: (a/2) tan(theta/2)."""
        return 0.5 * self.a * np.tan(self.angle / 2)


def miter_outline(a, theta, d):
    """Core outline and symmetry line of a miter of in-plane side a, angle theta and chamfer d (units of a).

    Coordinates: arm 1 runs along +y with u = +x; the arm axes intersect at the origin, so the inner corner is
    O = (-a/2, -(a/2) tan(theta/2)). Returns (Outline, (O, bisector direction)).
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    t = np.tan(theta / 2)
    O = np.array([-a / 2, -a / 2 * t])
    u2 = np.array([np.cos(theta), np.sin(theta)])
    bis = np.array([c, s])
    P1, P2 = O + np.array([a, 0.0]), O + a * u2
    dd, d_plain, d_max = d * a, a * c, a / c
    tol = 1e-12 * a

    def mirror_pt(x):
        r = x - O
        return O + 2 * bis * (bis @ r) - r

    if dd >= d_max - tol:                         # sharp corner, no cut
        Pout = np.array([a / 2, a / 2 * t])
        verts, ports = [O, P1, Pout, P2], (PortPlane(tuple(O), tuple(P1)), PortPlane(tuple(O), tuple(P2)))
    elif abs(dd - d_plain) <= tol:                # plain miter: the mirror joins P1 and P2
        verts, ports = [O, P1, P2], (PortPlane(tuple(O), tuple(P1)), PortPlane(tuple(O), tuple(P2)))
    elif dd > d_plain:                            # partial cut
        Q1 = np.array([a / 2, O[1] + (dd - d_plain) / s])
        Q2 = mirror_pt(Q1)
        verts, ports = [O, P1, Q1, Q2, P2], (PortPlane(tuple(O), tuple(P1)), PortPlane(tuple(O), tuple(P2)))
    else:                                         # deep cut: the mirror meets the outer wall before the plane
        delta = (d_plain - dd) / s
        E1, Q1 = O - np.array([0.0, delta]), np.array([a / 2, O[1] - delta])
        E2, Q2 = mirror_pt(E1), mirror_pt(Q1)
        verts = [E1, Q1, Q2, E2, O]
        ports = (PortPlane(tuple(E1), tuple(Q1), deembed=delta), PortPlane(tuple(E2), tuple(Q2), deembed=delta))
    return Outline(vertices=[tuple(v) for v in verts], ports=ports), (tuple(O), tuple(bis))


def arc_bend_outline(a, radius, angle):
    """Core outline of a circular bend of centre-line radius R and angle (rad), turning towards -u.

    Port 1 lies on y = 0 with u = +x; the centre of curvature is (-R, 0), i.e. kappa = 1/R > 0 in the convention of
    CONVENTIONS.md section 2 (outer wall at u = +a/2). Centre-line length R * angle.
    """
    C = np.array([-radius, 0.0])
    e = np.array([np.cos(angle), np.sin(angle)])
    p1i, p1o = np.array([-a / 2, 0.0]), np.array([a / 2, 0.0])
    p2i, p2o = C + (radius - a / 2) * e, C + (radius + a / 2) * e
    return Outline(vertices=[tuple(p1i), tuple(p1o), tuple(p2o), tuple(p2i)],
                   ports=(PortPlane(tuple(p1i), tuple(p1o)), PortPlane(tuple(p2i), tuple(p2o))),
                   arcs={1: tuple(C), 3: tuple(C)})


def straight_outline(a, length):
    """Core outline of a straight guide of the given length (m)."""
    p = [(-a / 2, 0.0), (a / 2, 0.0), (a / 2, length), (-a / 2, length)]
    return Outline(vertices=p, ports=(PortPlane(p[0], p[1]), PortPlane(p[3], p[2])))
