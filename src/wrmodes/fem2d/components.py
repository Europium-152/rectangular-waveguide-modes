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
from .solver import Discretization, solve_scalar, solve_impedance

__all__ = ["Coplanar2D", "Miter", "FEMOptions", "miter_outline", "chained_miter_outline", "arc_bend_outline",
           "straight_outline", "mirror_outline", "mode_parity"]


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


def _check_wall_impedance(z):
    if z is None or callable(z):
        return z
    if isinstance(z, str):
        if z != "guide":
            raise ValueError("wall_impedance must be None, a number (ohm), a callable Z_s(f) or 'guide'")
        return z
    z = complex(z)
    if not np.isfinite(z) or z.real < 0:
        raise ValueError("wall_impedance must be finite with Re Z_s >= 0 (a passive wall)")
    return z


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
    wall_impedance : None, complex, callable or 'guide'
        Surface impedance Z_s (ohm, e^{+i omega t}: Z_s = (1 + i) R_s for a metal, CONVENTIONS.md section 1) of the
        **in-plane walls of the core** (the walls parallel to v: every edge of the outline except the port planes),
        as a number, a function Z_s(f) of the frequency (Hz), or 'guide' for the conductivity of the guide
        (`Guide.wall_impedance`). None (default): perfectly conducting walls. See Notes.
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
    * Wall loss (`wall_impedance`). The Leontovich condition E_tan = Z_s H_tan x n on the in-plane walls is
      imposed exactly (docs/theory/WP3-normalization.md, section 6): a Robin condition on psi = E_v for family
      n = 0 (TE_m0, LSM only), and for n >= 1 a condition that couples the LSM and LSE problems through the
      tangential derivatives (the LS split of CONVENTIONS.md section 7 is exact only for perfect walls). The
      family index n stays exact, because the walls at v = +-b/2 are **not** included: they are perfect
      conductors here (a lossy wall at v = +-b/2 would couple n with n +- 2, CONVENTIONS.md section 3; for metal
      walls their loss is the part `wrmodes.loss` attributes to the walls 'v+', 'v-'). With wall_impedance='guide'
      in a line whose guide has a conductivity, the broad-wall loss of the component (H plane: walls y = +-h/2;
      E plane: x = +-w/2) is therefore left out. The straight arms behind the reference planes stay perfectly
      conducting, so the GSM is that of the lossy core between perfect guides (as `wrmodes.filters.LossyWallFilter`
      and the first-order lossy straights of WP2, whose junction mismatch Lambda^- cancels between adjacent lossy
      components). The discrete system stays complex symmetric (reciprocity to rounding); it is passive for
      Re Z_s >= 0. A family exactly at k_t = 0 is returned lossless (uncoupled straight modes).
    """

    def __init__(self, outline, b=None, plane="H", guide=None, mirror=None, length=None, name="", options=None,
                 wall_impedance=None, **kw):
        self.outline = outline
        self.wall_impedance = _check_wall_impedance(wall_impedance)
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
        if self.guide is not None and self.wall_impedance is None:
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

    def surface_impedance(self, f):
        """Surface impedance Z_s (ohm) of the in-plane core walls at f (Hz), or None for perfect walls."""
        z = self.wall_impedance
        if z is None:
            return None
        if isinstance(z, str):                        # 'guide'
            if self.guide is None:
                raise ValueError(f"{self.name}: wall_impedance='guide' needs a guide")
            z = self.guide.wall_impedance(f)
            if z is None:
                return None
        elif callable(z):
            z = complex(z(f))
            if not np.isfinite(z) or z.real < 0:
                raise ValueError("wall_impedance(f) must be finite with Re Z_s >= 0")
        z = complex(z)
        return None if z == 0 else z

    def element_size(self, f, k_eff):
        """Nominal element size (m) for an effective wavenumber k_eff (1/m) at f (Hz)."""
        o = self.options
        return min(2 * np.pi / k_eff / o.mesh_density, o.max_element * min(self.port_widths))

    def _mesh(self, h, Lp, lossy=False):
        o = self.options
        # impedance walls: the wall condition jumps where the core walls meet the perfectly conducting arms
        extra = [pt for p in self.outline.ports for pt in (p.inner, p.outer)] if lossy else ()
        md = build_mesh(self.outline, h, Lp, refine_factor=o.corner_refinement, mirror=self.mirror,
                        extra_points=extra)
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
        Zs = self.surface_impedance(f)
        if Zs is not None and not Lp > 0:
            raise ValueError(f"{self.name}: impedance walls need straight perfectly conducting arms before the modal "
                             "ports (port_distance > 0)")
        info = {"port_distance": Lp, "h": {}, "ndof": [], "elements": {}, "uncoupled_families": [],
                "wall_impedance": Zs}
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
                meshes[key] = self._mesh(h, Lp, lossy=Zs is not None)
            md, disc = meshes[key]
            info["h"][n], info["elements"][n] = md.h, md.t.shape[1]
            ky = n * np.pi / self.b
            lossy = Zs is not None
            pol_list = [p for p in ("LSM", "LSE") if not (p == "LSE" and n == 0)]
            setups = {}
            for pol in pol_list:
                rows = [[l for l, lab in enumerate(rot[p][1]) if lab[0] == pol] for p in range(2)]
                if not (rows[0] or rows[1] or (lossy and n > 0)):
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
                setups[pol] = (rows, port_modes, port_betas, excite)
            if lossy:
                order = [p for p in pol_list if p in setups]
                S2, inf = solve_impedance(disc, k, kt2, ky, Zs,
                                          [(p,) + tuple(setups[p][1:]) for p in order])
                info["ndof"].append(((n, "+".join(order)), inf["ndof"]))
                solved = [(order, S2)]
            else:
                solved = []
                for pol in pol_list:
                    if pol not in setups:
                        continue
                    S2, inf = solve_scalar(disc, kt2, pol, *setups[pol][1:])
                    info["ndof"].append(((n, pol), inf["ndof"]))
                    solved.append(([pol], S2))
            for order, S2 in solved:
                # 2D amplitudes -> LS amplitudes, then from the modal ports to the reference planes.
                # alpha_in = s_in f a_in, alpha_out = s_out f a_out: LSM s_in = s_out = 1; LSE (H_v reverses with
                # the direction of travel) s_in = +1 / -1 at port 1 / 2 and s_out = -s_in (WP3-normalization.md 3).
                gidx, fac, beta, s_in, s_out, dist = [], [], [], [], [], []
                for pol in order:
                    rows, _, port_betas, excite = setups[pol]
                    for p in range(2):
                        dk = self.outline.ports[p].deembed + Lp
                        sp_ = 1.0 if p == 0 else -1.0
                        for l, e in zip(rows[p], excite[p]):
                            gidx.append(off[p] + idx[p][l])
                            fac.append(rot[p][2][l])
                            beta.append(port_betas[p][e])
                            s_in.append(1.0 if pol == "LSM" else sp_)
                            s_out.append(1.0 if pol == "LSM" else -sp_)
                            dist.append(dk)
                fac, beta, s_in, s_out, dist = map(np.asarray, (fac, beta, s_in, s_out, dist))
                T = S2 * (s_out[:, None] * s_in[None, :]) * fac[None, :] / fac[:, None]
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
    wall_impedance : None, complex, callable or 'guide'
        Impedance of the in-plane walls (inner wall, outer walls and mirror), as in `Coplanar2D` (the walls at
        v = +-b/2 stay perfect conductors; see the notes there).
    options / keyword arguments : FEMOptions.

    Reference planes (CONVENTIONS.md section 6): perpendicular to each arm through the inner corner, at
    `arm_length` = (a/2) tan(|theta|/2) from the intersection of the arm axes; centre-line length 2 arm_length.
    When the mirror cuts the outer wall before a reference plane (d < cos(theta/2), e.g. the optical mirror of
    angles below 90 degrees), the field is solved from the last uniform cross-section of the arm and shifted
    analytically to the reference plane.
    """

    def __init__(self, angle_deg, chamfer=None, plane="H", guide=None, name="", options=None, wall_impedance=None,
                 **kw):
        self.wall_impedance = _check_wall_impedance(wall_impedance)
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
                + ("" if self.guide is None else f", guide={self.guide!r}")
                + ("" if self.wall_impedance is None else f", wall_impedance={self.wall_impedance!r}") + ")")

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


def _line_intersection(p, d, q, e):
    """Intersection of the lines p + s d and q + t e (2D); None if parallel."""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-14 * np.hypot(*d) * np.hypot(*e):
        return None
    r = q - p
    s = (r[0] * e[1] - r[1] * e[0]) / den
    return p + s * d


def _seg_dist(p1, p2, q1, q2):
    """Smallest distance between two closed segments in the plane."""
    def pt_seg(p, a, b):
        ab = b - a
        t = np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0)
        return np.hypot(*(p - a - t * ab))
    if _segments_cross(p1, p2, q1, q2, 0.0):
        return 0.0
    return min(pt_seg(p1, q1, q2), pt_seg(p2, q1, q2), pt_seg(q1, p1, p2), pt_seg(q2, p1, p2))


def _check_simple(outline, a):
    """Raise ValueError unless the core plus straight arms of length a is a simple polygon: no two non-adjacent
    edges touch or cross and no edge turns back on the previous one (the guide must not overlap itself)."""
    V, _ = outline.full_boundary(a)
    V = [np.asarray(v, float) for v in V]
    m = len(V)
    tol = 1e-9 * a
    for q in range(m):
        e0, e1 = V[q] - V[q - 1], V[(q + 1) % m] - V[q]
        if abs(e0[0] * e1[1] - e0[1] * e1[0]) <= 1e-12 * a * a and e0 @ e1 < 0:
            raise ValueError("the chained outline turns back on itself: spacers too short for these miters")
    for q in range(m):
        for r in range(q + 2, m):
            if q == 0 and r == m - 1:
                continue
            if _seg_dist(V[q], V[(q + 1) % m], V[r], V[(r + 1) % m]) <= tol:
                raise ValueError("the chained outline (with its arms) intersects or touches itself: spacers too "
                                 "short for these miters")


def _segments_cross(p1, p2, q1, q2, tol):
    """True if the open segments p1-p2 and q1-q2 cross (proper intersection)."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 * d2 < -tol) and (d3 * d4 < -tol)


def chained_miter_outline(angles_deg, spacers, chamfers=None, *, a, port_extension=0.0):
    """Core outline of n miters joined by straight spacers, with the reference planes of the first and last miter.

    Parameters
    ----------
    angles_deg : sequence of float
        Bend angle of each miter in degrees, signed as in `Miter` (positive: turns towards -u).
    spacers : sequence of float
        n - 1 centre-line lengths (m, >= 0) between consecutive miters: the distance between the reference plane
        of miter i (its port 2) and that of miter i + 1 (its port 1), i.e. the length of the `Straight` that joins
        them in the equivalent `Line` [Miter, Straight, Miter, ...].
    chamfers : None or sequence of (float or None)
        d/a of each miter as in `Miter` (None: the optical mirror, d = 1 / (2 cos(theta/2))).
    a : float
        In-plane side (m).
    port_extension : float
        Length (m) of straight guide added beyond the last uniform cross-section at both ends (default 0); the port
        planes then lie at the ends of these pieces and are de-embedded to the reference planes. Collinear vertices
        are removed. Used to reproduce drawn geometries such as COMSOL campaign 2, case E5.

    Returns
    -------
    (Outline, mirror, length)
        The outline (for `Coplanar2D`), its symmetry line (origin, direction) when the chain is symmetric (angles,
        chamfers and spacers palindromic; else None), and the centre-line length between the reference planes,
        sum(2 arm_length_i) + sum(spacers).

    Geometry (CONVENTIONS.md section 6): coordinates as in `miter_outline` (input arm along +y, u = +x, the arm
    axes of the first miter cross at the origin). Each miter is the miter of `miter_outline`: the inner walls meet at
    the inner corner, the outer walls are cut by the mirror perpendicular to the bisector at d a from the inner
    corner. Where two mirrors overlap (the spacer is shorter than the sum of the lengths by which they reach into
    it, e.g. two optical 45 degree miters closer than s = 2a), the outer wall between them disappears and the
    outline follows the two mirrors to their intersection; if that point is not inside the guide, or the mirror
    of one miter reaches the inner corner of its neighbour (opposite turns), a ValueError is raised. The reference
    planes are perpendicular to the end arms through the inner corners of the first and last miter, at
    `arm_length` = (a/2) tan(|theta|/2) from their apexes; when a mirror cuts the outer wall of an end arm before
    the reference plane (d < cos(theta/2)), the port plane lies at the last uniform cross-section and the GSM is
    continued analytically to the reference plane (`PortPlane.deembed`, CONVENTIONS.md section 6).
    """
    th = [np.radians(float(x)) for x in angles_deg]
    n = len(th)
    if n < 1:
        raise ValueError("give at least one miter")
    if any(not 0 < abs(t) < np.pi for t in th):
        raise ValueError("every bend angle must satisfy 0 < |angle_deg| < 180 (degrees)")
    spacers = [float(s) for s in spacers]
    if len(spacers) != n - 1:
        raise ValueError(f"{n} miters need {n - 1} spacers")
    if any(s < 0 for s in spacers):
        raise ValueError("spacers must be >= 0")
    if chamfers is None:
        chamfers = [None] * n
    if len(chamfers) != n:
        raise ValueError("give one chamfer per miter (None for the optical mirror)")
    dd = [0.5 / np.cos(abs(t) / 2) if c is None else float(c) for t, c in zip(th, chamfers)]
    if any(not d > 0 for d in dd):
        raise ValueError("chamfer d must be positive")
    a = float(a)
    tol = 1e-12 * a
    rot = lambda v, t: np.array([np.cos(t) * v[0] - np.sin(t) * v[1], np.sin(t) * v[0] + np.cos(t) * v[1]])
    ell = [0.5 * a * np.tan(abs(t) / 2) for t in th]
    # centre line: directions t_j, u_j of segment j (0 .. n) and apexes A_i (0 .. n-1)
    tdir, udir = [np.array([0.0, 1.0])], [np.array([1.0, 0.0])]
    for t in th:
        tdir.append(rot(tdir[-1], t))
        udir.append(rot(udir[-1], t))
    A = [np.zeros(2)]
    for i in range(n - 1):
        A.append(A[-1] + tdir[i + 1] * (ell[i] + spacers[i] + ell[i + 1]))
    # events of every miter on the sides sigma = -1 (u = -a/2) and +1 (u = +a/2)
    # each event: list of points; ev_in[i][sig] = point on segment i's side line, ev_out[i][sig] = on segment i+1's
    corners = []
    for i in range(n):
        s_in = -1.0 if th[i] > 0 else 1.0
        s_out = -s_in
        O = A[i] + s_in * 0.5 * a * udir[i] - ell[i] * tdir[i]
        half = abs(th[i]) / 2
        bis = np.cos(half) * s_out * udir[i] + np.sin(half) * tdir[i]
        info = {"inner": s_in, "O": O, "bis": bis}
        p_out_i = A[i] + s_out * 0.5 * a * udir[i]          # a point of the outer side line of segment i
        p_out_o = A[i] + s_out * 0.5 * a * udir[i + 1]      # ... of segment i + 1
        if dd[i] * a >= a / np.cos(half) - tol:              # sharp corner
            P = _line_intersection(p_out_i, tdir[i], p_out_o, tdir[i + 1])
            info["outer_pts"] = [P]
            info["mirror"] = None
        else:
            M0 = O + dd[i] * a * bis                          # a point of the mirror; direction perpendicular to bis
            md = np.array([-bis[1], bis[0]])
            Qi = _line_intersection(p_out_i, tdir[i], M0, md)
            Qo = _line_intersection(p_out_o, tdir[i + 1], M0, md)
            info["outer_pts"] = [Qi, Qo]
            info["mirror"] = (M0, md)
        corners.append(info)

    def side_points(sig):
        """Vertices of side sig from port 1 to port 2, and the de-embedding of both ends (m)."""
        # per segment j: entry point (from miter j-1, on segment j's line) and exit point (to miter j)
        entries, exits = [None] * (n + 1), [None] * (n + 1)
        for i, c in enumerate(corners):
            if c["inner"] == sig:
                exits[i], entries[i + 1] = ("corner", c["O"], i), ("corner", c["O"], i)
            else:
                op = c["outer_pts"]
                exits[i] = ("outer", op[0], i)
                entries[i + 1] = ("outer", op[-1], i)
        # resolve overlaps on the inner segments
        seq = []          # list of vertices in order
        for j in range(n + 1):
            en, ex = entries[j], exits[j]
            if en is not None and ex is not None:
                pos_en = tdir[j] @ en[1]
                pos_ex = tdir[j] @ ex[1]
                if pos_ex < pos_en - tol:
                    ce, cx = corners[en[2]], corners[ex[2]]
                    if en[0] == "corner" or ex[0] == "corner" or ce["mirror"] is None or cx["mirror"] is None:
                        raise ValueError(f"miters {en[2] + 1} and {ex[2] + 1} overlap: spacer too short for this "
                                         "geometry (a wall would cut through the guide)")
                    P = _line_intersection(ce["mirror"][0], ce["mirror"][1], cx["mirror"][0], cx["mirror"][1])
                    off = None if P is None else udir[j] @ (P - (A[ex[2]] - ell[ex[2]] * tdir[j]))
                    if P is None or abs(off) > 0.5 * a + tol:
                        raise ValueError(f"the mirrors of miters {en[2] + 1} and {ex[2] + 1} cross outside the guide: "
                                         "spacer too short")
                    entries[j] = ("joint", P, en[2])
                    exits[j] = ("joint", P, ex[2])
        for j in range(n + 1):
            en, ex = entries[j], exits[j]
            if j > 0:
                c = corners[j - 1]
                if en[0] == "corner":
                    pass                          # the corner was added as the exit of segment j - 1
                elif en[0] == "outer":
                    if len(c["outer_pts"]) == 2:
                        seq.append(en[1])         # Qo (Qi added as exit of segment j-1); sharp: single point
                else:
                    pass                          # joint, added as exit of segment j - 1
            if j < n:
                c = corners[j]
                if ex[0] == "corner":
                    seq.append(ex[1])
                elif ex[0] == "outer":
                    seq.append(ex[1])             # Qi, or the sharp outer corner
                else:
                    seq.append(ex[1])             # joint point
        return seq

    left, right = side_points(-1.0), side_points(1.0)
    # port planes: reference planes through the inner corners of the first and last miter, moved back to the last
    # uniform cross-section when a mirror cuts the end arm before them
    def end_plane(k):
        j = 0 if k == 0 else n
        c = corners[0] if k == 0 else corners[-1]
        sgn = 1.0 if k == 0 else -1.0                         # + t_j points into the core at port 1, out at 2
        ref = tdir[j] @ c["O"]
        side_first = [left[0], right[0]] if k == 0 else [left[-1], right[-1]]
        pos = min(sgn * (tdir[j] @ p) for p in side_first)
        plane = min(sgn * ref, pos)                           # in the direction into the core
        delta = sgn * ref - plane
        return sgn * plane, delta

    p1, d1 = end_plane(0)
    p2, d2 = end_plane(1)
    ext = float(port_extension)
    if ext < 0:
        raise ValueError("port_extension must be >= 0")
    on_line = lambda base, j, pos, sig: (base + sig * 0.5 * a * udir[j]
                                         + (pos - tdir[j] @ (base + sig * 0.5 * a * udir[j])) * tdir[j])
    P1i = on_line(A[0], 0, p1 - ext, -1.0)
    P1o = on_line(A[0], 0, p1 - ext, 1.0)
    P2i = on_line(A[-1], n, p2 + ext, -1.0)
    P2o = on_line(A[-1], n, p2 + ext, 1.0)
    verts = [P1o] + right + [P2o, P2i] + left[::-1] + [P1i]
    # drop duplicates and collinear points (the port end points are kept)
    keep_ids = {0, len(right) + 1, len(right) + 2, len(verts) - 1}
    out, ids = [], []
    for idx, v in enumerate(verts):
        if out and np.hypot(*(v - out[-1])) <= tol:
            if idx in keep_ids:
                ids[-1] = idx
            continue
        out.append(v)
        ids.append(idx)
    if np.hypot(*(out[0] - out[-1])) <= tol:
        out.pop(); ids.pop()
    changed = True
    while changed:
        changed = False
        m = len(out)
        for q in range(m):
            if ids[q] in keep_ids:
                continue
            a0, a1, a2 = out[q - 1], out[q], out[(q + 1) % m]
            cr = (a1[0] - a0[0]) * (a2[1] - a0[1]) - (a1[1] - a0[1]) * (a2[0] - a0[0])
            if abs(cr) <= 1e-12 * a * a and (a1 - a0) @ (a2 - a1) > 0:
                del out[q]; del ids[q]
                changed = True
                break
    ports = (PortPlane(tuple(P1i), tuple(P1o), deembed=d1 + ext), PortPlane(tuple(P2i), tuple(P2o), deembed=d2 + ext))
    outline = Outline(vertices=[tuple(v) for v in out], ports=ports)
    _check_simple(outline, a)
    length = 2 * sum(ell) + sum(spacers)
    # symmetry line of palindromic chains
    mirror = None
    if (np.allclose(th, th[::-1], rtol=0, atol=1e-15) and np.allclose(dd, dd[::-1], rtol=1e-15, atol=0)
            and np.allclose(spacers, spacers[::-1], rtol=1e-15, atol=0)):
        if n % 2 == 1:
            c = corners[n // 2]
            mirror = (tuple(c["O"]), tuple(c["bis"]))
        else:
            i = n // 2 - 1
            mid = A[i] + tdir[i + 1] * (ell[i] + 0.5 * spacers[i])
            mirror = (tuple(mid), tuple(udir[i + 1]))
    return outline, mirror, length
