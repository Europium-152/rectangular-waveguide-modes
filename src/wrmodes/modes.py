"""Mode basis and mode fields of the rectangular waveguide.

Everything in this module follows ``CONVENTIONS.md`` at the root of the repository. In short:

* Local cross-section coordinates (u, v) with u in [-a/2, a/2] and v in [-b/2, b/2]; (u, v, z) is right-handed.
  Side ``a`` lies along u (the bend plane of a bend), side ``b`` along v. Mode (m, n) has m half-periods
  along u and n along v.
* Time dependence exp(+i omega t); a forward mode varies as exp(-i beta z). Below cutoff, beta = -i|beta|, so
  that forward modes decay in +z.
* Transverse electric mode fields are real and normalized to unit integral of |e|^2:
  TM modes  e = grad Pi,
  TE modes  e = grad Gamma x z_hat,
  with the normalized potentials Pi and Gamma of the JINST paper (eq. 5.4). These are the fields for which the
  closed-form curvature coupling coefficients of ``wrmodes.mode_coupling`` hold, signs included
  (see ``docs/theory/T0-field-convention.md``).
* Amplitudes a are normalized with the unconjugated reaction, (1/2) int (E x H) . z_hat ds = a^2, i.e.
  E_t = sum (a+ + a-) sqrt(2/Y) e,  H_t = sum (a+ - a-) sqrt(2 Y) z_hat x e,  with the principal square root of
  the wave admittance Y. For propagating modes |a|^2 is the power carried by the mode.
"""
from dataclasses import dataclass
import numpy as np
import scipy.constants as cst

__all__ = ["ETA0", "Mode", "ModeBasis", "reversal_signs", "eplane_frame_signs", "cutoff_wavenumber", "propagation_constant", "wave_admittance",
           "transverse_field", "potential", "divergence", "ls_rotation"]

_KIND_ORDER = {"TE": 0, "TM": 1}


@dataclass(frozen=True)
class Mode:
    """A rectangular-waveguide mode: kind 'TE' or 'TM', m half-periods along side a (u), n along side b (v)."""
    kind: str
    m: int
    n: int

    def __post_init__(self):
        kind = str(self.kind).upper()
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "m", int(self.m))
        object.__setattr__(self, "n", int(self.n))
        if kind not in _KIND_ORDER:
            raise ValueError(f"mode kind must be 'TE' or 'TM', got {self.kind!r}")
        if self.m < 0 or self.n < 0:
            raise ValueError("mode numbers must be non-negative")
        if kind == "TE" and self.m == 0 and self.n == 0:
            raise ValueError("TE0,0 does not exist")
        if kind == "TM" and (self.m == 0 or self.n == 0):
            raise ValueError(f"TM{self.m},{self.n} does not exist (TM modes need m, n >= 1)")

    @property
    def name(self):
        """Name in the style of wrmodes 0.1, e.g. 'TE1,0'."""
        return f"{self.kind}{self.m},{self.n}"

    def __str__(self):
        return self.name

    @classmethod
    def parse(cls, obj):
        """Mode from a Mode, a ('TE', m, n) tuple, a wrmodes 0.1 tuple ('TE1,0', m, n, cutoff) or a name 'TE1,0'."""
        if isinstance(obj, Mode):
            return obj
        if isinstance(obj, str):
            kind, rest = obj[:2], obj[2:]
            m, n = rest.split(",")
            return cls(kind, int(m), int(n))
        kind = str(obj[0])[:2]
        return cls(kind, int(obj[1]), int(obj[2]))


# ----------------------------------------------------------------------------------------- scalars
def cutoff_wavenumber(mode, a, b):
    """chi = pi sqrt((m/a)^2 + (n/b)^2) (1/m)."""
    mode = Mode.parse(mode)
    return np.pi * np.hypot(mode.m / a, mode.n / b)


CUTOFF_RTOL = 1e-13
"""A mode is treated as exactly at cutoff when |k^2 - chi^2| <= CUTOFF_RTOL k^2. This makes the classification
independent of the platform's rounding (a frequency built to coincide with a cutoff gives beta = 0 everywhere)."""


def _cutoff_offset(k, chi):
    """k^2 - chi^2, snapped to 0 within CUTOFF_RTOL k^2."""
    d = k * k - chi * chi
    return 0.0 if abs(d) <= CUTOFF_RTOL * k * k else d


def propagation_constant(mode, a, b, f):
    """beta (1/m), real and positive above cutoff, -i|beta| below cutoff (forward modes decay in +z), exactly 0 at
    cutoff (within CUTOFF_RTOL)."""
    k = 2 * np.pi * f / cst.c
    d = _cutoff_offset(k, cutoff_wavenumber(mode, a, b))
    return np.sqrt(d) + 0j if d >= 0 else -1j * np.sqrt(-d)


ETA0 = cst.mu_0 * cst.c
"""Free-space impedance used throughout (ohm). It is defined as mu0 c so that k = omega / c, omega mu0 = k ETA0 and
omega eps0 = k / ETA0 hold exactly; the CODATA eps0 of scipy satisfies mu0 eps0 c^2 = 1 only to about 1e-12."""


def wave_admittance(mode, a, b, f):
    """Wave admittance Y (S): beta / (k eta0) for TE, k / (eta0 beta) for TM (complex below cutoff)."""
    mode = Mode.parse(mode)
    k = 2 * np.pi * f / cst.c
    beta = propagation_constant(mode, a, b, f)
    if beta == 0:
        raise ValueError(f"{mode.name} is exactly at cutoff at {f} Hz; its wave admittance is 0 or infinite")
    return beta / (k * ETA0) if mode.kind == "TE" else k / (ETA0 * beta)


# ----------------------------------------------------------------------------------------- fields
def _eps(l):
    return 1.0 if l == 0 else 2.0


def _norm(mode, a, b):
    m, n = mode.m, mode.n
    s = np.pi * np.sqrt(m * m * b / a + n * n * a / b)
    return (np.sqrt(_eps(m) * _eps(n)) if mode.kind == "TE" else 2.0) / s


def potential(mode, a, b, u, v):
    """Normalized Helmholtz potential: Gamma (TE, cos-cos) or Pi (TM, sin-sin), eq. 5.4 of the JINST paper."""
    mode = Mode.parse(mode)
    up, vp = mode.m * np.pi * (np.asarray(u) + a / 2) / a, mode.n * np.pi * (np.asarray(v) + b / 2) / b
    c = _norm(mode, a, b)
    if mode.kind == "TE":
        return c * np.cos(up) * np.cos(vp)
    return c * np.sin(up) * np.sin(vp)


def transverse_field(mode, a, b, u, v):
    """Normalized transverse electric field (e_u, e_v) of the mode: grad Pi (TM) or grad Gamma x z_hat (TE)."""
    mode = Mode.parse(mode)
    km, kn = mode.m * np.pi / a, mode.n * np.pi / b
    up, vp = km * (np.asarray(u) + a / 2), kn * (np.asarray(v) + b / 2)
    c = _norm(mode, a, b)
    if mode.kind == "TM":
        return c * km * np.cos(up) * np.sin(vp), c * kn * np.sin(up) * np.cos(vp)
    du = -c * km * np.sin(up) * np.cos(vp)        # d Gamma / du
    dv = -c * kn * np.cos(up) * np.sin(vp)        # d Gamma / dv
    return dv, -du                                 # grad Gamma x z_hat = (dGamma/dv, -dGamma/du)


def divergence(mode, a, b, u, v):
    """(div e, div(z_hat x e)) of the normalized field: (-chi^2 Pi, 0) for TM, (0, -chi^2 Gamma) for TE."""
    mode = Mode.parse(mode)
    chi2 = cutoff_wavenumber(mode, a, b) ** 2
    p = potential(mode, a, b, u, v)
    zero = np.zeros_like(p)
    return (-chi2 * p, zero) if mode.kind == "TM" else (zero, -chi2 * p)


# ----------------------------------------------------------------------------------------- basis
def _sort_key(mode, a, b):
    return (cutoff_wavenumber(mode, a, b), _KIND_ORDER[mode.kind], mode.m, mode.n)


class ModeBasis:
    """An ordered set of modes of an a x b guide; the basis of one port of a GSM.

    Modes are ordered by cutoff frequency, then TE before TM, then m, then n (see CONVENTIONS.md). Two bases
    are equal when a, b, the ordered mode list and the frame tag are equal. The optional `frame` tag (e.g. 'H' or
    'E', CONVENTIONS.md section 2) lets the cascade detect frame mismatches that the dimensions alone cannot
    reveal, as in a square guide.
    """

    def __init__(self, a, b, modes, sort=True, frame=None):
        self.a, self.b = float(a), float(b)
        self.frame = frame
        modes = [Mode.parse(m) for m in modes]
        if len(set(modes)) != len(modes):
            raise ValueError("duplicate modes in basis")
        if sort:
            modes = sorted(modes, key=lambda m: _sort_key(m, self.a, self.b))
        self.modes = tuple(modes)
        self._index = {m: i for i, m in enumerate(self.modes)}

    # construction ------------------------------------------------------------------------------
    @classmethod
    def below(cls, f_max, a, b, families=None, kinds=("TE", "TM"), frame=None):
        """All modes with cutoff frequency strictly below f_max (Hz), optionally restricted to vertical
        indices n in `families`. A mode is below f_max when chi < 2 pi f_max / c (the same test as
        `propagating_mask`)."""
        chi_max = 2 * np.pi * f_max / cst.c
        m_max, n_max = int(chi_max * a / np.pi) + 1, int(chi_max * b / np.pi) + 1
        modes = []
        for m in range(m_max + 1):
            for n in range(n_max + 1):
                if families is not None and n not in families:
                    continue
                for kind in kinds:
                    if (kind == "TE" and m + n == 0) or (kind == "TM" and m * n == 0):
                        continue
                    mode = Mode(kind, m, n)
                    if _cutoff_offset(chi_max, cutoff_wavenumber(mode, a, b)) > 0:
                        modes.append(mode)
        return cls(a, b, modes, frame=frame)

    @classmethod
    def propagating(cls, f, a, b, families=None, cutoff_factor=1.0, frame=None, evanescent_families="propagating"):
        """Propagating modes at f, plus the evanescent modes with cutoff below cutoff_factor * f.

        By default (`evanescent_families='propagating'`), evanescent modes are only added to families that have at
        least one propagating mode: components that conserve n cannot couple the others to anything that
        propagates, and leaving them out avoids the singular LS rotation at k = n pi / b (see ls_rotation). Use
        'all' to keep every family.
        """
        basis = cls.below(cutoff_factor * f, a, b, families=families, frame=frame)
        if cutoff_factor > 1 and evanescent_families == "propagating":
            prop = basis.propagating_mask(f)
            keep_n = {m.n for m, p in zip(basis.modes, prop) if p}
            basis = cls(a, b, [m for m in basis.modes if m.n in keep_n], frame=frame)
        elif evanescent_families not in ("propagating", "all"):
            raise ValueError("evanescent_families must be 'propagating' or 'all'")
        return basis

    # access -------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.modes)

    def __iter__(self):
        return iter(self.modes)

    def __getitem__(self, i):
        return self.modes[i]

    def index(self, mode):
        return self._index[Mode.parse(mode)]

    def __contains__(self, mode):
        try:
            return Mode.parse(mode) in self._index
        except (ValueError, TypeError, IndexError):
            return False

    def __eq__(self, other):
        return (isinstance(other, ModeBasis) and self.modes == other.modes and self.frame == other.frame
                and np.isclose(self.a, other.a, rtol=1e-12, atol=0) and np.isclose(self.b, other.b, rtol=1e-12, atol=0))

    def __hash__(self):
        # a and b are compared with a tolerance, so they cannot enter the hash
        return hash((self.modes, self.frame))

    def __repr__(self):
        fr = "" if self.frame is None else f", frame={self.frame!r}"
        return f"ModeBasis(a={self.a:g}, b={self.b:g}{fr}, {len(self)} modes: {', '.join(self.names)})"

    @property
    def names(self):
        return [m.name for m in self.modes]

    def cutoff_frequencies(self):
        return np.array([cutoff_wavenumber(m, self.a, self.b) * cst.c / (2 * np.pi) for m in self.modes])

    def beta(self, f):
        """Propagation constants (1/m) at f, with the branch of propagation_constant."""
        return np.array([propagation_constant(m, self.a, self.b, f) for m in self.modes], dtype=complex)

    def admittance(self, f):
        return np.array([wave_admittance(m, self.a, self.b, f) for m in self.modes], dtype=complex)

    def propagating_mask(self, f):
        """True for the modes that propagate at f (chi < k beyond CUTOFF_RTOL; a mode at cutoff does not propagate)."""
        k = 2 * np.pi * f / cst.c
        return np.array([_cutoff_offset(k, cutoff_wavenumber(m, self.a, self.b)) > 0 for m in self.modes], dtype=bool)

    def family(self, n):
        """Indices of the modes with vertical index n."""
        return [i for i, m in enumerate(self.modes) if m.n == n]

    def families(self):
        return sorted({m.n for m in self.modes})

    def subset(self, indices):
        return ModeBasis(self.a, self.b, [self.modes[i] for i in indices], sort=False, frame=self.frame)

    def with_frame(self, frame):
        """The same modes with another frame tag."""
        return ModeBasis(self.a, self.b, self.modes, sort=False, frame=frame)

    def legacy_modes(self, f=None):
        """The modes as wrmodes 0.1 tuples (name, m, n, cutoff), as used by coupled_mode_matrices.

        The 0.1 functions only handle propagating modes (their branch of beta below cutoff is +i|beta|, opposite to
        CONVENTIONS.md). If f is given, a ValueError is raised when any mode is evanescent at f.
        """
        if f is not None and not np.all(self.propagating_mask(f)):
            bad = [m.name for m, p in zip(self.modes, self.propagating_mask(f)) if not p]
            raise ValueError(f"the wrmodes 0.1 coupled-mode functions only take propagating modes; evanescent at "
                             f"{f / 1e9:.4g} GHz: {', '.join(bad)}")
        fc = self.cutoff_frequencies()
        return [(m.name, m.m, m.n, c) for m, c in zip(self.modes, fc)]


# ----------------------------------------------------------------------------------------- LS modes
def reversal_signs(basis):
    """Signs P (+1/-1 per mode) for turning a component around (CONVENTIONS.md section 2).

    Turning a component around is the rotation by pi about u: (u, v, z) -> (u, -v, -z). It keeps the sign of the
    curvature (u stays on the outer side of a bend) and maps mode p to P_p times itself, P = (-1)^(n+1) for TE and
    TM alike.
    """
    return np.array([(-1.0) ** (m.n + 1) for m in basis.modes])


def eplane_frame_signs(global_basis, local_basis):
    """Signed permutation from the global (H-plane) frame to the E-plane local frame (CONVENTIONS.md section 2).

    Local frame u = y, v = -x: global mode (M, N) of a w x h guide is local mode (N, M) of the h x w guide, with
    the factor (-1)^M for TE and (-1)^(M+1) for TM. Returns the matrix F with a_local = F a_global.
    """
    F = np.zeros((len(local_basis), len(global_basis)))
    for j, g in enumerate(global_basis.modes):
        loc = Mode(g.kind, g.n, g.m)
        sign = (-1.0) ** g.m if g.kind == "TE" else (-1.0) ** (g.m + 1)
        F[local_basis.index(loc), j] = sign
    return F


def ls_rotation(basis, f):
    """Rotation from TE/TM amplitudes to longitudinal-section amplitudes with respect to v (side b).

    Returns (Q, labels). Q is complex-orthogonal (Q Q^T = I; real orthogonal for propagating modes), with
    a_LS = Q a_TETM, and labels[i] = ('LSE' | 'LSM', m, n) for row i:

    * LSE (TE^v): E_v = 0 everywhere, built from H_v (Neumann on the walls parallel to v).
    * LSM (TM^v): H_v = 0 everywhere, built from E_v (Dirichlet on the walls parallel to v).

    TE_m0 modes are LSM, TE_0n modes are LSE. For m, n >= 1 the pair (TE_mn, TM_mn) rotates into
    LSE_mn = cos(phi) TE - sin(phi) TM and LSM_mn = sin(phi) TE + cos(phi) TM, with
    tan(phi) = (m b k) / (n a beta). Both members of every such pair must be in the basis.
    The rows of Q follow the order of the TE member of each pair (TM rows take the LSM of the pair).

    The normalization is cos^2 + sin^2 = 1 with r^2 = (n a beta)^2 + (m b k)^2 = (n^2 a^2 + m^2 b^2) k_t^2, where
    k_t^2 = k^2 - (n pi / b)^2. At k_t = 0 (f = n c / (2 b)) the LS amplitudes of evanescent pairs with m >= 1 do
    not exist and Q is singular; near it, Q is ill-conditioned (entries of order k / |k_t|). A ValueError is
    raised when |k_t^2| < 1e-10 k^2. Families with k_t^2 <= 0 have no propagating mode, so components that conserve
    n can leave them out (ModeBasis.propagating does so by default).
    """
    N = len(basis)
    Q = np.zeros((N, N), dtype=complex)
    labels = [None] * N
    k = 2 * np.pi * f / cst.c
    a, b = basis.a, basis.b
    for i, mode in enumerate(basis.modes):
        m, n = mode.m, mode.n
        if mode.kind == "TE" and n == 0:
            Q[i, i], labels[i] = 1.0, ("LSM", m, n)
        elif mode.kind == "TE" and m == 0:
            Q[i, i], labels[i] = 1.0, ("LSE", m, n)
        elif mode.kind == "TE":
            partner = Mode("TM", m, n)
            if partner not in basis:
                raise ValueError(f"{mode.name} needs {partner.name} in the basis for the LS rotation")
            j = basis.index(partner)
            beta = propagation_constant(mode, a, b, f)
            kt2 = k * k - (n * np.pi / b) ** 2
            if abs(kt2) < 1e-10 * k * k:
                raise ValueError(f"LS rotation of {mode.name}/{partner.name} is singular at {f / 1e9:.6g} GHz "
                                 f"(k_t = 0 for family n = {n}); leave this family out of the LS basis")
            # tan(phi) = (m b k) / (n a beta); cos/sin from the complex-orthogonal normalization
            x, y = n * a * beta, m * b * k
            r = np.sqrt(x * x + y * y)
            c, s = x / r, y / r
            Q[i, i], Q[i, j], labels[i] = c, -s, ("LSE", m, n)
            Q[j, i], Q[j, j], labels[j] = s, c, ("LSM", m, n)
    return Q, labels
