"""90-degree plain and chamfered miters by mode matching (segmentation), in the H or E plane.

The method and its derivation are in docs/reports/WP4b.md; the 2D solver is ``wrmodes.modematch._corner2d``.
In short: a miter does not vary along v, so each family n splits into an LSM (Dirichlet, psi = E_v) and an LSE
(Neumann, psi = H_v) scalar problem in the bend plane with k_t^2 = k^2 - (n pi / b)^2 (CONVENTIONS.md section 7).
Each scalar problem is solved by the segmentation of Alessandri, Mongiardo & Sorrentino (1994, Fig. 1): the corner
region is cut into rectangles and right isosceles triangles, whose cell matrices are closed forms of separable
expansions (the triangles by images, as in their Fig. 3, which for the optical mirror is the image system of Kotik &
Sivov 1965). The unknowns on the interfaces are the normal derivatives of psi (impedance form), which converges for
both wall types. The symmetry of the miter about its bisector halves each problem (even / odd parity). The LS
amplitudes of the 2D solution are the LS amplitudes of CONVENTIONS.md section 7 (a common factor per family and
type, which cancels), and the GSM is returned in the TE/TM basis, S = Q^T S_LS Q per port (`modes.ls_rotation`).
"""
import warnings

import numpy as np
import scipy.constants as cst

from ..components import Component
from ..components.guide import Guide
from ..modes import Mode, ModeBasis, ls_rotation, propagation_constant, CUTOFF_RTOL
from ..gsm import GSM
from . import _corner2d

__all__ = ["MiterMM"]

_SQRT2 = np.sqrt(2.0)


class MiterMM(Component):
    """90-degree plain or chamfered miter bend, by mode matching (segmentation of the corner).

    Interchangeable with ``wrmodes.fem2d.Miter(90, chamfer, plane, guide)`` in a `wrmodes.Line`: same port bases,
    frame tags, reference planes and `arm_length`; it needs no FEM install.

    Parameters
    ----------
    chamfer : float or None
        Alessandri's d (1994, Fig. 1): distance from the inner corner to the miter wall, along the bisector,
        divided by the in-plane side a. None (default) is the optical mirror of CONVENTIONS.md section 6,
        d = 1/sqrt(2) (Alessandri's d/b = 0.707, Ma et al.'s c/a = 1, Alimenti et al.'s t/a = 0). d = sqrt(2) and
        larger is the unmitered (sharp) corner. Ma et al. (1997) cut c along the outer walls:
        d = (2 - c/a)/sqrt(2); Alimenti et al. (1994) leave t of the outer wall standing: d = (1 + t/a)/sqrt(2).
        Miters cutting into the arms before the reference planes (d < 1/sqrt(2)) are not supported (use
        ``wrmodes.fem2d.Miter``).
    plane : 'H' or 'E'
        Bend plane of the global TE10 mode (CONVENTIONS.md section 2). 'H': a = w, b = h. 'E': a = h, b = w; the
        GSM is in the local E frame (bases tagged 'E'); `Line` inserts the frame changes.
    guide : Guide, (w, h) or None
        Global cross-section; None leaves it to `wrmodes.Line` (`with_guide`). Perfectly conducting walls only.
    n_modes : int or None
        Truncation: number of arm modes M of every scalar 2D problem; the interface expansions have
        ceil(M / RATIO[wall]) functions per width a, in proportion to their lengths. None (default): the rule
        `default_n_modes` (per family and wall type, from the convergence study in docs/reports/WP4b.md) with error
        control: each scalar problem is also solved at M/1.5, and M is increased until the estimated error of every
        propagating entry is below ERR_TARGET (`_solve_2d`); this matters near sharp resonances of the corner just
        below a cutoff. The GSM carries the truncation used in `gsm(...).truncation`. An integer fixes M.
    angle_deg : +90 or -90 (keyword)
        +90 turns towards -u (outer wall at u = +a/2, as kappa > 0 in CONVENTIONS.md section 2), -90 towards +u;
        the GSM of -90 is D S D with D = (-1)^(m+1) (mirror u -> -u).
    name : str (keyword)

    Reference planes (CONVENTIONS.md section 6): perpendicular to each arm through the inner corner, at
    `arm_length` = a/2 from the intersection of the arm axes; the port 2 frame is the port 1 frame rotated about v,
    so that S11 = S22. The component couples modes within each family n only (no wall loss). Families without a
    propagating mode are treated as uncoupled straight modes over the centre-line length 2 `arm_length` = a
    (CONVENTIONS.md section 3).

    Port bases: `gsm(f, basis1, basis2)` returns exactly the requested modes (any truncation of the port bases,
    evanescent modes included); the internal truncation is `n_modes`, independent of the port bases. Port modes
    left out are treated as matched: evanescent modes need the neighbouring component far enough away for them to
    decay (at least about a / pi per neper of the lowest evanescent mode left out).
    """

    name = "miter (mode matching)"
    plane = "H"
    #: arm modes per interface function across the width a, per wall type (relative convergence; the errors of the
    #: two truncations have opposite signs and nearly cancel at these ratios, docs/reports/WP4b.md)
    RATIO = {"D": 1.15, "N": 0.7}
    #: default truncation of a scalar problem with k_t a / pi = x: M = ceil(N_PER_MODE[wall] x + N_ADD[wall])
    N_PER_MODE = {"D": 6.0, "N": 12.0}
    N_ADD = {"D": 100.0, "N": 70.0}
    #: error control of the default truncation (`_solve_2d`): target of the estimated error of every propagating
    #: entry of each scalar problem (the TE/TM entries are bounded by the largest of them), order of convergence,
    #: and the largest M as a multiple of the rule
    ERR_TARGET = 5e-5
    ORDER = 4.0 / 3.0
    M_MAX_FACTOR = 8
    #: the error estimate compares M with M / CHECK_RATIO (M / 2 proved too coarse next to resonances below cutoff)
    CHECK_RATIO = 1.5

    def __init__(self, chamfer=None, plane="H", guide=None, n_modes=None, *, angle_deg=90.0, name=""):
        if plane not in ("H", "E"):
            raise ValueError(f"plane must be 'H' or 'E', got {plane!r}")
        self.plane = plane
        if float(abs(angle_deg)) != 90.0:
            raise ValueError("MiterMM is for 90-degree miters only (angle_deg = +90 or -90); use wrmodes.fem2d.Miter")
        self.angle_deg = float(angle_deg)
        if chamfer is None:
            d = 1 / _SQRT2
        else:
            d = float(chamfer)
            if d < (1 - 1e-12) / _SQRT2:
                raise ValueError(f"chamfer d/a = {d:g} < 1/sqrt(2): the miter cuts into the arms before the reference "
                                 f"planes; not supported by MiterMM (use wrmodes.fem2d.Miter)")
            d = min(max(d, 1 / _SQRT2), _SQRT2)
        self.chamfer = chamfer
        self._d = d
        if guide is not None and not isinstance(guide, Guide):
            guide = Guide(*guide)
        self.guide = guide
        if n_modes is not None and int(n_modes) < 2:
            raise ValueError("n_modes must be at least 2")
        self.n_modes = None if n_modes is None else int(n_modes)
        self.name = name or repr(self)

    # ------------------------------------------------------------------------------------- geometry
    def __repr__(self):
        ch = "optical mirror" if self.chamfer is None else f"d/a={self._d:.6g}"
        g = "" if self.guide is None else f", {self.guide!r}"
        sgn = "" if self.angle_deg > 0 else ", angle_deg=-90"
        return f"MiterMM({ch}, plane={self.plane!r}{g}{sgn})"

    @property
    def d(self):
        """Chamfer d/a actually used (1/sqrt(2) <= d <= sqrt(2))."""
        return self._d

    @property
    def theta(self):
        """|bend angle| (rad)."""
        return np.pi / 2

    @property
    def frame(self):
        """Frame tag of the port bases (= plane)."""
        return self.plane

    def _dims(self, basis=None):
        if self.guide is not None:
            return self.guide.dims(self.plane)
        if basis is None:
            raise ValueError("MiterMM without a guide: pass port bases or a guide")
        return basis.a, basis.b

    @property
    def arm_length(self):
        """Distance (m) from the intersection of the arm axes to each reference plane: a/2 (CONVENTIONS.md 6)."""
        if self.guide is None:
            raise ValueError("arm_length needs the guide (MiterMM(guide=...) or inside a Line)")
        return 0.5 * self._dims()[0]

    @property
    def port_widths(self):
        """In-plane width a (m) at port 1 and port 2."""
        a = self._dims()[0]
        return a, a

    def with_guide(self, guide):
        """This miter in `guide` (used by wrmodes.Line)."""
        if self.guide is not None:
            if self.guide != guide:
                raise ValueError(f"{self!r} belongs to {self.guide!r}, not to {guide!r}")
            return self
        return MiterMM(self.chamfer, self.plane, guide, self.n_modes, angle_deg=self.angle_deg,
                       name=self.name if self.name != repr(self) else "")

    # ------------------------------------------------------------------------------------- bases
    def default_basis(self, f, cutoff_factor=None):
        """Port bases at f (Hz). cutoff_factor=None means 1: the propagating modes only (the internal truncation is
        `n_modes`, not the port bases). The bases are `guide.basis(f, cutoff_factor, plane)` (the line's rule)."""
        if self.guide is None:
            raise ValueError("default_basis needs the guide (MiterMM(guide=...))")
        cf = 1.0 if cutoff_factor is None else cutoff_factor
        b = self.guide.basis(f, cf, plane=self.plane)
        return b, b

    def default_n_modes(self, f, n=0, wall="D"):
        """Default number of arm modes M of the scalar problem of family n and wall type ('D': LSM, 'N': LSE) at
        f (Hz): ceil(N_PER_MODE[wall] x + N_ADD[wall]) with x = k_t a / pi, the number of half-periods of the
        family across a (convergence study in docs/reports/WP4b.md)."""
        a, b = self._dims()
        k = 2 * np.pi * f / cst.c
        kt2 = k * k - (np.pi * n / b) ** 2
        return self._M(np.sqrt(max(kt2, 0.0)) * a, wall, None, 0)

    def _M(self, kta, wall, n_modes, mmax):
        if n_modes is None:
            n_modes = self.n_modes
        if n_modes is None:
            M = int(np.ceil(self.N_PER_MODE[wall] * kta / np.pi + self.N_ADD[wall]))
        else:
            M = int(n_modes)
        return max(M, mmax + 1)

    # ------------------------------------------------------------------------------------- GSM
    def _check(self, basis, port):
        if basis.frame not in (None, self.plane):
            raise ValueError(f"basis{port} is in the {basis.frame!r} frame, but the miter is in the {self.plane!r} "
                             f"frame")
        if self.guide is not None:
            a, b = self.guide.dims(self.plane)
            if not (np.isclose(basis.a, a, rtol=1e-12, atol=0) and np.isclose(basis.b, b, rtol=1e-12, atol=0)):
                raise ValueError(f"basis{port} is for a {basis.a:g} x {basis.b:g} cross-section, the miter's guide is "
                                 f"{a:g} x {b:g} in the {self.plane} frame")

    def gsm(self, f, basis1=None, basis2=None, n_modes=None, symmetry=True):
        """GSM at f (Hz) at the reference planes, in the local frame of `plane`.

        basis1, basis2 : ModeBasis, optional (default `default_basis(f)`); any modes of the a x b guide.
        n_modes : int, optional; overrides the internal truncation (used by `convergence_study`).
        symmetry : bool; True (default) solves the even and odd halves about the bisector, False the full
            two-port network (twice the cost; S11 = S22 then holds only to round-off, a check of the port frames).
        """
        if self.guide is not None:
            self.guide.require_lossless(repr(self))
        if basis1 is None or basis2 is None:
            d1, d2 = self.default_basis(f)
            basis1 = d1 if basis1 is None else basis1
            basis2 = d2 if basis2 is None else basis2
        self._check(basis1, 1)
        self._check(basis2, 2)
        a, b = self._dims(basis1)
        if not (np.isclose(basis2.a, a, rtol=1e-12, atol=0) and np.isclose(basis2.b, b, rtol=1e-12, atol=0)):
            raise ValueError("basis1 and basis2 are for different cross-sections")
        k = 2 * np.pi * f / cst.c
        union = sorted(set(basis1.modes) | set(basis2.modes), key=lambda m: (m.n, m.m, m.kind))
        for m in union:
            chi2 = (np.pi * m.m / a) ** 2 + (np.pi * m.n / b) ** 2
            if abs(k * k - chi2) <= CUTOFF_RTOL * k * k:
                raise ValueError(f"{m.name} is exactly at cutoff at {f} Hz; its wave admittance is 0 or infinite")
        idx = {m: i for i, m in enumerate(union)}
        N = len(union)
        info = {}
        blocks = [np.zeros((N, N), dtype=complex) for _ in range(4)]       # S11, S21, S12, S22 on `union`
        c = min(max(_SQRT2 * self._d - 1.0, 0.0), 1.0)    # miter line x + y = a (1 + c), c in units of a
        for n in sorted({m.n for m in union}):
            fam = [m for m in union if m.n == n]
            kt2 = k * k - (np.pi * n / b) ** 2
            live = kt2 > (np.pi / a) ** 2 if n == 0 else kt2 > 0
            if not live:
                for m in fam:                                # uncoupled straight modes over the centre line, a
                    blocks[1][idx[m], idx[m]] = blocks[2][idx[m], idx[m]] = \
                        np.exp(-1j * propagation_constant(m, a, b, f) * a)
                continue
            full = set(fam)                                  # the family with every TE/TM partner (LS rotation)
            for m in fam:
                if m.m >= 1 and n >= 1:
                    full |= {Mode("TE", m.m, n), Mode("TM", m.m, n)}
            fb = ModeBasis(a, b, full)
            Q, labels = ls_rotation(fb, f)
            kta = np.sqrt(kt2) * a
            mmax = max(m.m for m in fam)
            S2 = {}
            for typ, wall in (("LSM", "D"), ("LSE", "N")):
                if typ == "LSE" and n == 0:
                    continue
                sol, M, est = self._solve_2d(c, wall, kt2 * a * a, kta, n_modes, mmax, symmetry)
                info[(n, typ)] = (M, est)
                s11, s21 = sol[0], sol[1]
                s12, s22 = (s21, s11) if symmetry else (sol[2], sol[3])
                if typ == "LSE":
                    # psi = H_v is a transverse magnetic component: it reverses for backward waves (CONVENTIONS.md
                    # section 5), so the LSE reflections change sign (a short circuit has S = -1 but dH_v/dz = 0)
                    s11, s22 = -s11, -s22
                S2[typ] = (s11, s21, s12, s22)
            # S_LS on the rows of Q (labels), then S_TE/TM = Q^T S_LS Q per port
            nl = len(labels)
            L = [np.zeros((nl, nl), dtype=complex) for _ in range(4)]
            types = np.array([t for t, _, _ in labels])
            ms = np.array([m for _, m, _ in labels])
            for typ in S2:
                sel = np.where(types == typ)[0]
                off = 1 if typ == "LSM" else 0                # arm index of mode m: m - 1 (sin) or m (cos)
                ii = ms[sel] - off
                for blk in range(4):
                    L[blk][np.ix_(sel, sel)] = S2[typ][blk][np.ix_(ii, ii)]
            T = [Q.T @ Lb @ Q for Lb in L]
            if self.angle_deg < 0:
                D = np.array([(-1.0) ** (m.m + 1) for m in fb.modes])
                T = [D[:, None] * Tb * D[None, :] for Tb in T]
            sel = [fb.index(m) for m in fam]
            gi = [idx[m] for m in fam]
            for blk in range(4):
                blocks[blk][np.ix_(gi, gi)] = T[blk][np.ix_(sel, sel)]
        S11u, S21u, S12u, S22u = blocks
        i1 = [idx[m] for m in basis1.modes]
        i2 = [idx[m] for m in basis2.modes]
        B1 = basis1 if basis1.frame is not None else basis1.with_frame(self.plane)
        B2 = basis2 if basis2.frame is not None else basis2.with_frame(self.plane)
        g = GSM(S11u[np.ix_(i1, i1)], S12u[np.ix_(i1, i2)], S21u[np.ix_(i2, i1)], S22u[np.ix_(i2, i2)],
                B1, B2, f, self.name)
        g.truncation = info          # {(n, 'LSM'|'LSE'): (M used, error estimate or None)}
        return g

    def _solve_2d(self, c, wall, kap2a2, kta, n_modes, mmax, symmetry):
        """Scalar problem with the fixed truncation `n_modes`, or with error control (n_modes None).

        Error control: solve at M and at M' = ceil(M / CHECK_RATIO) (about 30 % of the cost); the truncation error
        decays as M^-p with p = ORDER = 4/3 (the r^(2/3) field at the inner corner), so err(M) ~ |S(M) - S(M')| /
        ((M / M')^p - 1) over the propagating block. While the estimate exceeds ERR_TARGET, M grows by max(1.5, 1.1 (err /
        ERR_TARGET)^(1/p)) (re-estimated from the last two solutions) up to M_MAX_FACTOR times the rule; the result
        is the last plain Galerkin solution (exactly unitary and reciprocal)."""
        R = self.RATIO[wall]

        def solve(M):
            return _corner2d.miter_2d(1.0, c, wall, kap2a2, M, int(np.ceil(M / R)), symmetric=symmetry)

        M = self._M(kta, wall, n_modes, mmax)
        fixed = n_modes is not None or self.n_modes is not None
        if fixed:
            return solve(M), M, None
        m0 = 1 if wall == "D" else 0
        P = int(np.sum(np.arange(m0, m0 + M) * np.pi < np.sqrt(kap2a2)))
        if P == 0:
            return solve(M), M, None
        # the coarser solution must still hold every requested and every propagating mode
        M = max(M, int(np.ceil(self.CHECK_RATIO * (max(mmax, P) + 2))))
        sol = solve(M)

        def prop(x):
            return np.concatenate([b[:P, :P].ravel() for b in x[:2]])

        M_prev = int(np.ceil(M / self.CHECK_RATIO))
        prev = solve(M_prev)
        M_cap = self.M_MAX_FACTOR * M
        while True:
            est = np.max(np.abs(prop(sol) - prop(prev))) / ((M / M_prev) ** self.ORDER - 1)
            if est <= self.ERR_TARGET or M >= M_cap:
                break
            grow = max(1.5, 1.1 * (est / self.ERR_TARGET) ** (1 / self.ORDER))
            M_prev, prev = M, sol
            M = int(min(np.ceil(M * grow), M_cap))
            sol = solve(M)
        if est > self.ERR_TARGET:
            warnings.warn(f"MiterMM: truncation error estimate {est:.1e} above {self.ERR_TARGET:g} at M = {M} "
                          f"(k_t a / pi = {kta / np.pi:.4f}, wall {wall}): a sharp resonance of the corner?",
                          RuntimeWarning, stacklevel=3)
        return sol, M, est

    def convergence_study(self, f, n_modes_list, basis=None):
        """Convergence of the propagating block with the truncation n_modes (arm modes per scalar problem).

        Returns a dict like `wrmodes.modematch.convergence_study`: 'n_modes' (K,), 'modes' (propagating mode names),
        'S' (K, P, P) propagating block (ports 1 then 2), 'change' max ||S_k| - |S_k-1||, 'change_complex'
        max |S_k - S_k-1|, 'unitarity', 'reciprocity'.
        """
        if basis is None:
            basis = self.default_basis(f)[0]
        S_list, uni, rec = [], [], []
        for nm in n_modes_list:
            g = self.gsm(f, basis, basis, n_modes=nm)
            m1, m2 = g.propagating()
            keep = np.concatenate([m1, m2])
            S_list.append(g.S[np.ix_(keep, keep)])
            uni.append(g.unitarity_defect())
            rec.append(g.reciprocity_defect() / np.max(np.abs(g.S)))
        S = np.array(S_list)
        K = len(n_modes_list)
        change, change_c = np.full(K, np.nan), np.full(K, np.nan)
        for i in range(1, K):
            change[i] = np.max(np.abs(np.abs(S[i]) - np.abs(S[i - 1])))
            change_c[i] = np.max(np.abs(S[i] - S[i - 1]))
        names = [m.name for m, p in zip(basis.modes, basis.propagating_mask(f)) if p]
        return {"n_modes": np.asarray(n_modes_list), "modes": names, "S": S, "change": change,
                "change_complex": change_c, "unitarity": np.array(uni), "reciprocity": np.array(rec)}
