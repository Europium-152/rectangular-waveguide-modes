"""Coplanar components solved with the 2D modal-port finite-element method (WP3).

`Coplanar2D` takes any core outline (straight segments and arcs) in the plane of side a; `Miter` builds the outline
of a plain or chamfered miter bend. Both return GSMs in the TE/TM basis of CONVENTIONS.md, in the local frame of
their `plane` ('H' or 'E', CONVENTIONS.md section 2). They only couple modes within a family n (no variation along
v), and solve each family as two scalar problems (LSM and LSE, CONVENTIONS.md section 7). Both follow the
component protocol of `wrmodes.components.Component` and can be used in a `wrmodes.Line`.
"""
import dataclasses
import numpy as np
import scipy.constants as cst

from ..components import Component, Guide
from ..components.guide import _check_plane
from ..gsm import GSM
from ..modes import Mode, ModeBasis, propagation_constant
from .geometry import Outline, PortPlane, build_mesh
from .normalization import ls_amplitude_factors
from .solver import Discretization, solve_scalar

__all__ = ["Coplanar2D", "Miter", "FEMOptions", "miter_outline", "arc_bend_outline", "straight_outline",
           "mirror_outline", "mode_parity"]


@dataclasses.dataclass(frozen=True)
class FEMOptions:
    """Discretization settings of the 2D solver.

    order : polynomial order of the Lagrange elements (2 or 3); default 3.
    mesh_density : elements per wavelength 2 pi / k_eff, where k_eff is the larger of k_t and the largest |beta| of
        the evanescent basis modes of the family; each family n gets its own mesh; default 6.
    max_element : largest element as a fraction of the smallest port width a; default 1/4.
    corner_refinement : element size at re-entrant corners as a fraction of the nominal size; default 0.05.
    port_distance : length (m) of the straight arms between the port planes and the modal ports; None (default)
        means a/4, shortened automatically when the bases contain evanescent modes (see `evanescent_gain`).
    n_evanescent : minimum number of evanescent port modes per port and polarization in the modal condition; default
        10. More are added so that the first port mode left out decays by `port_decay` e-folds over the arm.
    port_decay : see n_evanescent; default 16.
    evanescent_gain : largest factor exp(2 |beta| L) by which the shift of an evanescent mode from the modal port
        to the reference plane may amplify the discretization error; with port_distance=None the arms are
        shortened to respect it; default 10.
    """
    order: int = 3
    mesh_density: float = 6.0
    max_element: float = 0.25
    corner_refinement: float = 0.05
    port_distance: float = None
    n_evanescent: int = 10
    port_decay: float = 16.0
    evanescent_gain: float = 10.0


def _as_guide(guide):
    if guide is None or isinstance(guide, Guide):
        return guide
    w, h = (guide.w, guide.h) if hasattr(guide, "w") else guide
    return Guide(w, h)


def mode_parity(basis):
    """Signs D_p = (-1)^(m+1) of the modes under the mirror u -> -u (TE and TM alike).

    The mirror image of a component (e.g. a miter turning towards +u instead of -u) has the GSM D S D.
    """
    return np.array([(-1.0) ** (m.m + 1) for m in basis.modes])


class Coplanar2D(Component):
    """A two-port component whose geometry does not vary along v, given by its core outline.

    Parameters
    ----------
    outline : wrmodes.fem2d.geometry.Outline
        Core outline in the u-z plane (coordinates in m), with its two port planes (and their de-embedding
        distances).
    b : float, optional
        Side b (m), the dimension along v, common to both ports. Default: from `guide`.
    plane : 'H' or 'E'
        Local frame of the component (CONVENTIONS.md section 2): the outline is drawn in the plane of side
        a = guide.dims(plane)[0]; the port bases carry this frame tag.
    guide : Guide, optional
        Global cross-section. Needed by `wrmodes.Line` (which attaches its own guide with `with_guide`) and to
        check that the outline fits it. A guide with a wall conductivity raises NotImplementedError (WP2).
    mirror : (origin, direction), optional
        Symmetry line of the outline (port 2 the mirror image of port 1). The mesh is then built symmetric, so that
        S11 = S22 holds to rounding.
    length : float, optional
        Centre-line length (m) between the reference planes, used only for a family exactly at k_t = 0, which is
        returned as uncoupled straight modes over this length (the LS rotation does not exist there).
    name : str
    options : FEMOptions, or keyword arguments of FEMOptions

    Notes
    -----
    * Families: the component couples only modes of the same family n of its frame. Every family of the bases is
      solved, also those without a propagating mode (k_t^2 <= 0, CONVENTIONS.md section 3), except a family
      exactly at k_t = 0 (see `length`).
    * Default basis: the propagating modes (`cutoff_factor` = 1). The solver does not need evanescent modes in the
      returned basis: the evanescent fields of the discontinuity are carried by the arms and the modal ports.
    * Evanescent modes requested in a basis are shifted from the modal ports to the reference planes with
      exp(|beta| L); with `port_distance=None` the arms are shortened so that this amplifies the discretization
      error by at most `evanescent_gain` (and more port modes are used, see FEMOptions). A reference plane inside
      the discontinuity (`PortPlane.deembed` > 0) adds exp(|beta| deembed), which cannot be avoided: it is part
      of the definition of the GSM at such a plane (docs/reports/WP3.md).
    * Perfectly conducting walls only (WP2 adds wall loss).
    """

    def __init__(self, outline, b=None, plane="H", guide=None, mirror=None, length=None, name="", options=None,
                 **kw):
        self.outline = outline
        self.plane = _check_plane(plane)
        self.guide = _as_guide(guide)
        if b is None:
            if self.guide is None:
                raise ValueError("give b or a guide")
            b = self.guide.dims(self.plane)[1]
        self.b = float(b)
        self.mirror = mirror
        self.length = None if length is None else float(length)
        self.name = name or "coplanar 2D"
        opts = options or FEMOptions()
        self.options = dataclasses.replace(opts, **kw) if kw else opts
        if self.guide is not None:
            self._check_guide(self.guide)

    # -------------------------------------------------------------------------------------- interface
    @property
    def frame(self):
        """Frame tag of the port bases (= plane)."""
        return self.plane

    @property
    def port_widths(self):
        return self.outline.ports[0].a, self.outline.ports[1].a

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r}, plane={self.plane!r}>"

    def _check_guide(self, guide):
        a, b = guide.dims(self.plane)
        if not (np.isclose(self.b, b, rtol=1e-9) and all(np.isclose(x, a, rtol=1e-9) for x in self.port_widths)):
            raise ValueError(f"{self.name}: the outline ({self.port_widths} x {self.b}) does not fit {guide!r} in "
                             f"the {self.plane} frame ({a} x {b})")

    def _copy(self, **attrs):
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.__dict__.update(attrs)
        return new

    def with_options(self, **kw):
        """A copy of this component with other FEMOptions."""
        return self._copy(options=dataclasses.replace(self.options, **kw))

    def with_guide(self, guide):
        """This component in `guide` (used by wrmodes.Line); the outline must fit the guide."""
        guide = _as_guide(guide)
        if self.guide is not None:
            if guide != self.guide:
                raise ValueError(f"{self.name} belongs to {self.guide!r}, not to {guide!r}")
            return self
        self._check_guide(guide)
        return self._copy(guide=guide)

    def default_basis(self, f, cutoff_factor=None):
        """Port bases at f (Hz). cutoff_factor=None means 1: the propagating modes only.

        With a guide, the bases are `guide.basis(f, cutoff_factor, plane)` (the line's rule); without one,
        `ModeBasis.propagating` of each port.
        """
        cf = 1.0 if cutoff_factor is None else cutoff_factor
        if self.guide is not None:
            bb = self.guide.basis(f, cf, plane=self.plane)
            return bb, bb
        a1, a2 = self.port_widths
        return (ModeBasis.propagating(f, a1, self.b, cutoff_factor=cf, frame=self.plane),
                ModeBasis.propagating(f, a2, self.b, cutoff_factor=cf, frame=self.plane))

    def gsm(self, f, basis1=None, basis2=None):
        """GSM at f (Hz) in the TE/TM basis of CONVENTIONS.md, at the reference planes of the outline.

        The returned GSM carries the solver diagnostics in `fem_info`.
        """
        if self.guide is not None:
            self.guide.require_lossless(self.name)
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
        if basis.frame not in (None, self.plane):
            raise ValueError(f"basis of port {k + 1} has frame {basis.frame!r}, the component {self.plane!r}")

    def element_size(self, f, k_eff):
        """Nominal element size (m) for an effective wavenumber k_eff (1/m) at f (Hz)."""
        o = self.options
        return min(2 * np.pi / k_eff / o.mesh_density, o.max_element * min(self.port_widths))

    def _mesh(self, h, Lp):
        o = self.options
        md = build_mesh(self.outline, h, Lp, refine_factor=o.corner_refinement, mirror=self.mirror)
        return md, Discretization(md, order=o.order)

    def _port_distance(self, f, bases):
        """Arm length (m): the option, or a/4 shortened for evanescent basis modes (FEMOptions)."""
        o = self.options
        if o.port_distance is not None:
            return float(o.port_distance)
        Lp = 0.25 * min(self.port_widths)
        bev = 0.0
        for bs in bases:
            prop = bs.propagating_mask(f)
            beta = bs.beta(f)
            bev = max([bev] + [abs(beta[i].imag) for i in range(len(bs)) if not prop[i]])
        if bev > 0:
            Lp = min(Lp, np.log(o.evanescent_gain) / (2 * bev))
        return Lp

    def _solve(self, f, basis1, basis2):
        o = self.options
        k = 2 * np.pi * f / cst.c
        bases = (basis1, basis2)
        N1, N2 = len(basis1), len(basis2)
        off = (0, N1)
        S_ls = np.zeros((N1 + N2, N1 + N2), dtype=complex)
        Q = np.zeros((N1 + N2, N1 + N2), dtype=complex)
        families = sorted(set(basis1.families()) | set(basis2.families()))
        Lp = self._port_distance(f, bases)
        info = {"port_distance": Lp, "h": {}, "ndof": [], "elements": {}, "uncoupled_families": []}
        meshes = {}
        for n in families:
            kt2 = k * k - (n * np.pi / self.b) ** 2
            idx = [bs.family(n) for bs in bases]
            subs = [bs.subset(ix) for bs, ix in zip(bases, idx)]
            try:
                rot = [ls_amplitude_factors(sb, f) for sb in subs]
            except ValueError:
                # k_t = 0 (f = n c / 2b): the LS rotation does not exist (CONVENTIONS.md section 7)
                self._uncoupled(f, n, subs, idx, off, S_ls, Q, info)
                continue
            for p in range(2):
                gi = [off[p] + i for i in idx[p]]
                Q[np.ix_(gi, gi)] = rot[p][0]
            # one mesh per family, sized for its wavelength and for the decay of its evanescent basis modes
            bev = [abs(b_.imag) for sb in subs for b_, pr in zip(sb.beta(f), sb.propagating_mask(f)) if not pr]
            k_eff = max([np.sqrt(max(kt2, 0.0))] + bev + [1e-3 * k])
            h = self.element_size(f, k_eff)
            key = round(float(np.log(h)), 6)
            if key not in meshes:
                meshes[key] = self._mesh(h, Lp)
            md, disc = meshes[key]
            info["h"][n], info["elements"][n] = md.h, md.t.shape[1]
            for pol in ("LSM", "LSE"):
                rows = [[l for l, lab in enumerate(rot[p][1]) if lab[0] == pol] for p in range(2)]
                if not (rows[0] or rows[1]):
                    continue
                port_modes, port_betas, excite = [], [], []
                for p in range(2):
                    a = self.port_widths[p]
                    m0 = 1 if pol == "LSM" else 0
                    m_req = max([0] + [rot[p][1][l][1] for l in rows[p]])
                    m_prop = int(np.floor(np.sqrt(max(kt2, 0.0)) * a / np.pi - 1e-12))
                    m_hi = max(m_prop + o.n_evanescent, m_req)
                    if Lp > 0:      # the first port mode left out decays by port_decay e-folds over the arm
                        m_hi = max(m_hi, int(np.ceil(a / np.pi * np.sqrt(max((o.port_decay / Lp) ** 2 + kt2, 0)))))
                    m_hi = min(m_hi, m_req + 400)
                    ms = list(range(m0, m_hi + 1))
                    port_modes.append(ms)
                    port_betas.append([propagation_constant(Mode("TE", m, n), a, self.b, f) for m in ms])
                    excite.append([ms.index(rot[p][1][l][1]) for l in rows[p]])
                S2, inf = solve_scalar(disc, kt2, pol, port_modes, port_betas, excite)
                info["ndof"].append(((n, pol), inf["ndof"]))
                # 2D amplitudes -> LS amplitudes, then from the modal ports to the reference planes
                gidx, fac, beta, sgn, dist = [], [], [], [], []
                for p in range(2):
                    dk = self.outline.ports[p].deembed + Lp
                    for l, e in zip(rows[p], excite[p]):
                        gidx.append(off[p] + idx[p][l])
                        fac.append(rot[p][2][l])
                        beta.append(port_betas[p][e])
                        sgn.append(1.0 if p == 0 else -1.0)
                        dist.append(dk)
                fac, beta, sgn, dist = map(np.asarray, (fac, beta, sgn, dist))
                T = S2 * fac[None, :] / fac[:, None]
                if pol == "LSE":
                    T = -T * sgn[:, None] * sgn[None, :]
                ph = np.exp(1j * beta * dist)
                S_ls[np.ix_(gidx, gidx)] = ph[:, None] * T * ph[None, :]
        return Q.T @ S_ls @ Q, info

    def _uncoupled(self, f, n, subs, idx, off, S_ls, Q, info):
        """Family exactly at k_t = 0: uncoupled straight modes over the centre-line length (documented)."""
        if self.length is None:
            raise ValueError(f"{self.name}: family n = {n} is at k_t = 0 at {f / 1e9:.9g} GHz, where the LS rotation "
                             "does not exist; give the centre-line length (length=...) to treat it as uncoupled")
        if subs[0].modes != subs[1].modes:
            raise ValueError(f"{self.name}: family n = {n} at k_t = 0 with different port widths")
        P = np.exp(-1j * subs[0].beta(f) * self.length)
        i1 = [off[0] + i for i in idx[0]]
        i2 = [off[1] + i for i in idx[1]]
        S_ls[i2, i1] = P
        S_ls[i1, i2] = P
        for gi in (i1, i2):
            Q[gi, gi] = 1.0
        info["uncoupled_families"].append(n)


class Miter(Coplanar2D):
    """Plain or chamfered miter bend, solved with the 2D modal-port FEM.

    Parameters
    ----------
    angle_deg : float
        Bend angle in **degrees**, 0 < |angle_deg| < 180 (as `SechCurvature` and `ConstantCurvature.from_angle`).
        A positive angle turns the guide towards -u (outer wall at u = +a/2, like kappa > 0 in CONVENTIONS.md
        section 2); a negative angle turns it towards +u, and the GSM is then D S D, with S the GSM of the positive
        angle and D = (-1)^(m+1) the parity of the modes under u -> -u (`mode_parity`).
    chamfer : float or None
        Alessandri's d (1994, Fig. 1): distance from the inner corner to the mirror, along the bisector, divided by
        the in-plane side a; the mirror is perpendicular to the bisector.

        * None (default): the optical mirror, through the intersection of the two arm axes,
          d = 1 / (2 cos(theta/2)).
        * d = cos(theta/2): the mirror through the points where the outer walls cross the reference planes. For
          90 degrees both are d = 1/sqrt(2), the classical plain miter (Ma et al. 1997, c/a = 1; Alessandri 1994,
          d/b = 0.707); for other angles they differ, and only the optical one is a good bend.
        * d >= 1/cos(theta/2): the unmitered (sharp) corner.

        For 90 degrees, Ma et al. (1997) use the cut length c along the outer walls: d = (2 - c/a)/sqrt(2).
    plane : 'H' or 'E'
        Bend plane of the global TE10 mode. 'H': a = w, b = h. 'E': a = h, b = w (local frame, CONVENTIONS.md
        section 2; the GSM is in that local frame, tagged 'E').
    guide : Guide, (w, h) or None
        Global cross-section; None leaves it to `wrmodes.Line` (`with_guide`).
    options / keyword arguments : FEMOptions.

    Reference planes (CONVENTIONS.md section 6): perpendicular to each arm through the inner corner, at
    `arm_length` = (a/2) tan(|theta|/2) from the intersection of the arm axes; centre-line length 2 arm_length.
    When the mirror cuts the outer wall before a reference plane (d < cos(theta/2), e.g. the optical mirror of
    angles below 90 degrees), the field is solved from the last uniform cross-section of the arm and shifted
    analytically to the reference plane.
    """

    def __init__(self, angle_deg, chamfer=None, plane="H", guide=None, name="", options=None, **kw):
        angle_deg = float(angle_deg)
        if not 0 < abs(angle_deg) < 180:
            raise ValueError("the bend angle must satisfy 0 < |angle_deg| < 180 (degrees)")
        self.angle_deg = angle_deg
        self.chamfer = None if chamfer is None else float(chamfer)
        if self.chamfer is not None and not self.chamfer > 0:
            raise ValueError("chamfer d must be positive")
        self.plane = _check_plane(plane)
        self.guide = _as_guide(guide)
        self.name = name or (f"{plane}-plane miter {angle_deg:g} deg"
                             + ("" if chamfer is None else f" d/a={self.chamfer:.4g}"))
        opts = options or FEMOptions()
        self.options = dataclasses.replace(opts, **kw) if kw else opts
        self.mirror = None
        if self.guide is not None:
            self._build()

    @property
    def theta(self):
        """|bend angle| (rad)."""
        return np.radians(abs(self.angle_deg))

    @property
    def d(self):
        """Chamfer d/a actually used."""
        return 0.5 / np.cos(self.theta / 2) if self.chamfer is None else self.chamfer

    def _build(self):
        a, b = self.guide.dims(self.plane)
        self.b = b
        self.outline, self.mirror = miter_outline(a, self.theta, self.d)
        self.length = 2 * self.arm_length

    def __repr__(self):
        return (f"Miter({self.angle_deg:g}, chamfer={self.chamfer!r}, plane={self.plane!r}"
                + ("" if self.guide is None else f", guide={self.guide!r}") + ")")

    def with_guide(self, guide):
        guide = _as_guide(guide)
        if self.guide is not None:
            if guide != self.guide:
                raise ValueError(f"{self.name} belongs to {self.guide!r}, not to {guide!r}")
            return self
        new = self._copy(guide=guide)
        new._build()
        return new

    def _need_guide(self):
        if self.guide is None:
            raise ValueError(f"{self.name} has no guide: give guide=Guide(w, h), or use it in a wrmodes.Line")

    @property
    def arm_length(self):
        """Distance (m) from the intersection of the arm axes to each reference plane: (a/2) tan(|theta|/2)."""
        self._need_guide()
        return 0.5 * self.guide.dims(self.plane)[0] * np.tan(self.theta / 2)

    @property
    def port_widths(self):
        self._need_guide()
        a = self.guide.dims(self.plane)[0]
        return a, a

    def default_basis(self, f, cutoff_factor=None):
        self._need_guide()
        return super().default_basis(f, cutoff_factor)

    def gsm(self, f, basis1=None, basis2=None):
        """GSM at f (Hz) at the reference planes, in the local frame of `plane`."""
        self._need_guide()
        g = super().gsm(f, basis1, basis2)
        if self.angle_deg < 0:          # mirror image u -> -u of the bend turning towards -u
            D1, D2 = mode_parity(g.basis1), mode_parity(g.basis2)
            info = g.fem_info
            g = GSM(D1[:, None] * g.S11 * D1[None, :], D1[:, None] * g.S12 * D2[None, :],
                    D2[:, None] * g.S21 * D1[None, :], D2[:, None] * g.S22 * D2[None, :],
                    g.basis1, g.basis2, f, self.name)
            g.fem_info = info
        return g


def miter_outline(a, theta, d):
    """Core outline and symmetry line of a miter of in-plane side a (m), angle theta (rad, > 0, turning towards
    -u) and chamfer d (units of a).

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
    elif abs(dd - d_plain) <= tol:                # the mirror joins P1 and P2
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


def mirror_outline(outline):
    """The mirror image x -> -x of an outline, with the port planes re-oriented (u from the new u = -a/2 end).

    In the coordinates of `miter_outline` and `arc_bend_outline` (u = +x at port 1), this turns a component
    turning towards -u into one turning towards +u.
    """
    m = lambda p: (-p[0], p[1])
    ports = tuple(PortPlane(m(p.outer), m(p.inner), p.deembed) for p in outline.ports)
    arcs = {k: m(c) for k, c in outline.arcs.items()}
    return Outline(vertices=[m(v) for v in outline.vertices], ports=ports, arcs=arcs)


def arc_bend_outline(a, radius, angle):
    """Core outline of a circular bend of centre-line radius R (m) and angle (rad), turning towards -u.

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
