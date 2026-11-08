"""Generalized scattering matrices (GSM) and their cascade.

A GSM describes a two-port waveguide component in terms of the mode amplitudes of CONVENTIONS.md. Port 1 is the
input plane (z = 0 of the component) and port 2 the output plane. Incident waves are a1 = A+(port 1) and
a2 = A-(port 2); scattered waves are b1 = A-(port 1) and b2 = A+(port 2):

    [b1]   [S11 S12] [a1]
    [b2] = [S21 S22] [a2]

S11 has the shape (N1, N1), S22 (N2, N2), S21 (N2, N1) and S12 (N1, N2), where N1 and N2 are the numbers of modes
of the two port bases. With the amplitude normalization of CONVENTIONS.md, every reciprocal component has a
symmetric S (S^T = S), lossless components have a unitary S restricted to the propagating modes, and a cascade
is the Redheffer star product, which stays well conditioned with evanescent modes.
"""
import numpy as np

from .modes import ModeBasis, reversal_signs

__all__ = ["GSM", "cascade", "straight", "from_transfer_matrix", "block_diag"]


class GSM:
    """Generalized scattering matrix of a two-port component.

    Parameters
    ----------
    S11, S12, S21, S22 : array_like
        The four blocks (see the module docstring).
    basis1, basis2 : ModeBasis
        Mode bases of port 1 and port 2; their order defines the rows and columns of the blocks.
    f : float
        Frequency (Hz).
    name : str, optional
        Label used in reprs and error messages.
    """

    def __init__(self, S11, S12, S21, S22, basis1, basis2=None, f=None, name=""):
        basis2 = basis1 if basis2 is None else basis2
        self.S11 = np.asarray(S11, dtype=complex)
        self.S12 = np.asarray(S12, dtype=complex)
        self.S21 = np.asarray(S21, dtype=complex)
        self.S22 = np.asarray(S22, dtype=complex)
        self.basis1, self.basis2 = basis1, basis2
        self.f = f
        self.name = name
        N1, N2 = len(basis1), len(basis2)
        for blk, shape, label in ((self.S11, (N1, N1), "S11"), (self.S12, (N1, N2), "S12"),
                                  (self.S21, (N2, N1), "S21"), (self.S22, (N2, N2), "S22")):
            if blk.shape != shape:
                raise ValueError(f"{label} has shape {blk.shape}, expected {shape} from the port bases")

    # ---------------------------------------------------------------------------------- views
    @property
    def S(self):
        """The full (N1 + N2) x (N1 + N2) matrix [[S11, S12], [S21, S22]]."""
        return np.block([[self.S11, self.S12], [self.S21, self.S22]])

    @classmethod
    def from_matrix(cls, S, basis1, basis2=None, f=None, name=""):
        basis2 = basis1 if basis2 is None else basis2
        N1 = len(basis1)
        S = np.asarray(S, dtype=complex)
        return cls(S[:N1, :N1], S[:N1, N1:], S[N1:, :N1], S[N1:, N1:], basis1, basis2, f, name)

    def __repr__(self):
        lbl = f" {self.name!r}" if self.name else ""
        fs = "" if self.f is None else f" at {self.f / 1e9:.4g} GHz"
        return f"<GSM{lbl}{fs}: {len(self.basis1)} -> {len(self.basis2)} modes>"

    def propagating(self):
        """Boolean masks of the propagating modes of port 1 and port 2 (requires f)."""
        if self.f is None:
            raise ValueError("this GSM has no frequency")
        return self.basis1.propagating_mask(self.f), self.basis2.propagating_mask(self.f)

    def reversed(self):
        """The same component turned around, so that its port 2 becomes port 1 (CONVENTIONS.md section 2).

        Turning around is the rotation by pi about u, (u, v, z) -> (u, -v, -z): the ports are swapped and every
        mode picks up the sign P = (-1)^(n+1), so S' = P S_swapped P. Within one family P is a common sign and
        cancels; it matters for components that couple different families.
        """
        P1, P2 = reversal_signs(self.basis2), reversal_signs(self.basis1)    # new port 1 = old port 2
        return GSM(P1[:, None] * self.S22 * P1[None, :], P1[:, None] * self.S21 * P2[None, :],
                   P2[:, None] * self.S12 * P1[None, :], P2[:, None] * self.S11 * P2[None, :],
                   self.basis2, self.basis1, self.f, self.name + " (reversed)" if self.name else "")

    def transmission(self, mode_in, mode_out=None):
        """Complex transmission amplitude from mode_in at port 1 to mode_out (default: the same mode) at port 2."""
        j = self.basis1.index(mode_in)
        i = self.basis2.index(mode_in if mode_out is None else mode_out)
        return self.S21[i, j]

    def loss_db(self, mode_in, mode_out=None):
        """-20 log10 |S21| for the given modes: the transmission loss (mode conversion, reflection and ohmic)."""
        return -20 * np.log10(np.abs(self.transmission(mode_in, mode_out)))

    # ---------------------------------------------------------------------------------- checks
    def reciprocity_defect(self):
        """max |S - S^T| (zero for a reciprocal component)."""
        S = self.S
        return float(np.max(np.abs(S - S.T)))

    def unitarity_defect(self):
        """max |S_p^H S_p - I| over the propagating modes (zero for a lossless component)."""
        m1, m2 = self.propagating()
        keep = np.concatenate([m1, m2])
        Sp = self.S[np.ix_(keep, keep)]
        return float(np.max(np.abs(Sp.conj().T @ Sp - np.eye(Sp.shape[0])))) if Sp.size else 0.0

    def passivity_margin(self):
        """1 - (largest singular value of S over the propagating modes); >= 0 for a passive component."""
        m1, m2 = self.propagating()
        keep = np.concatenate([m1, m2])
        Sp = self.S[np.ix_(keep, keep)]
        return float(1.0 - np.linalg.norm(Sp, 2)) if Sp.size else 1.0

    # ---------------------------------------------------------------------------------- cascade
    def star(self, other):
        """Redheffer star product: this component followed by `other` (port 2 of self joined to port 1 of other)."""
        if self.basis2 != other.basis1:
            raise ValueError(f"cannot join {self!r} to {other!r}: port bases differ "
                             f"({self.basis2!r} vs {other.basis1!r}); insert a junction or change basis")
        if (self.f is None) != (other.f is None):
            raise ValueError("cannot join a GSM with a frequency to one without")
        if self.f is not None and not np.isclose(self.f, other.f, rtol=1e-12, atol=0):
            raise ValueError("cannot join GSMs computed at different frequencies")
        A11, A12, A21, A22 = self.S11, self.S12, self.S21, self.S22
        B11, B12, B21, B22 = other.S11, other.S12, other.S21, other.S22
        n = A22.shape[0]
        I = np.eye(n)
        # multiple reflections between the two components: L (I - B11 A22)^-1 R and L (I - A22 B11)^-1 R
        M1 = I - B11 @ A22
        M2 = I - A22 @ B11
        S11 = A11 + _outer_solve(A12, M1, B11 @ A21)
        S12 = _outer_solve(A12, M1, B12)
        S21 = _outer_solve(B21, M2, A21)
        S22 = B22 + _outer_solve(B21, M2, A22 @ B12)
        f = self.f if self.f is not None else other.f
        name = " + ".join(x for x in (self.name, other.name) if x)
        return GSM(S11, S12, S21, S22, self.basis1, other.basis2, f, name)

    def __matmul__(self, other):
        """`A @ B` is the cascade A followed by B (same as A.star(B))."""
        return self.star(other)


RCOND_MIN = 1e-12
"""Reciprocal condition number below which the star product treats I - A22 B11 as singular."""


def _outer_solve(L, M, R):
    """L M^-1 R, robust to an exactly singular M (trapped modes between two components).

    M = I - B11 A22 (or I - A22 B11) is singular when a mode between the two components is totally reflected
    on both sides without coupling to either outer port, e.g. two junctions on the same plane. Such a trapped
    mode is invisible from outside if its null directions satisfy L V0 = 0 or U0^H R = 0 (M = U S V^H); then the
    pseudo-inverse gives the correct product. Otherwise the cascade is at an exact resonance of a mode that couples
    to the ports, which has no finite scattering matrix, and a LinAlgError is raised.
    """
    n = M.shape[0]
    if n == 0:
        return np.zeros((L.shape[0], R.shape[1]), dtype=complex)
    lu = np.linalg.cond(M, 1) if n else 1.0
    if np.isfinite(lu) and 1.0 / lu >= RCOND_MIN:
        return L @ np.linalg.solve(M, R)
    U, sv, Vh = np.linalg.svd(M)
    tol = max(sv[0], 1.0) * 1e-10
    null = sv <= tol
    V0, U0 = Vh[null].conj().T, U[:, null]
    scale_L = max(np.max(np.abs(L)) if L.size else 0.0, 1.0)
    scale_R = max(np.max(np.abs(R)) if R.size else 0.0, 1.0)
    hidden = (np.max(np.abs(L @ V0)) if L.size else 0.0) <= 1e-8 * scale_L or \
             (np.max(np.abs(U0.conj().T @ R)) if R.size else 0.0) <= 1e-8 * scale_R
    if not hidden:
        raise np.linalg.LinAlgError("star product: exact resonance of a mode trapped between the two components that "
                                    "couples to the outer ports (I - A22 B11 is singular); the cascade has no finite "
                                    "scattering matrix at this frequency")
    inv = (Vh[~null].conj().T / sv[~null]) @ U[:, ~null].conj().T
    return L @ inv @ R


def cascade(*gsms):
    """Cascade of components in order of propagation (first element at the input)."""
    if not gsms:
        raise ValueError("nothing to cascade")
    out = gsms[0]
    for g in gsms[1:]:
        out = out.star(g)
    return out


def straight(basis, f, length, name=""):
    """Straight lossless section of the given length: S21 = S12 = diag(exp(-i beta L)), no reflection.

    Evanescent modes decay as exp(-|beta| L) (branch of wrmodes.modes.propagation_constant).
    """
    N = len(basis)
    P = np.diag(np.exp(-1j * basis.beta(f) * length))
    Z = np.zeros((N, N), dtype=complex)
    return GSM(Z, P, P, Z.copy(), basis, basis, f, name or f"straight {length:g} m")


def from_transfer_matrix(phi, basis, f=None, name=""):
    """GSM of a component given by its 2N x 2N transfer matrix y(L) = phi y(0), y = [A+, A-].

    This is how the coupled-mode solvers (propagating modes only) hand their result to the cascade. It must not
    be used with evanescent modes in the basis: the transfer matrix then contains growing exponentials.
    """
    phi = np.asarray(phi, dtype=complex)
    N = len(basis)
    if phi.shape != (2 * N, 2 * N):
        raise ValueError(f"transfer matrix has shape {phi.shape}, expected {(2 * N, 2 * N)}")
    P11, P12, P21, P22 = phi[:N, :N], phi[:N, N:], phi[N:, :N], phi[N:, N:]
    # A-(L) = P21 a1 + P22 b1 = a2  ->  b1 = P22^-1 (a2 - P21 a1);  b2 = A+(L) = P11 a1 + P12 b1
    S11 = -np.linalg.solve(P22, P21)
    S12 = np.linalg.inv(P22)
    S21 = P11 + P12 @ S11
    S22 = P12 @ S12
    return GSM(S11, S12, S21, S22, basis, basis, f, name)


def block_diag(*gsms, basis1=None, basis2=None, name=""):
    """Combine GSMs of disjoint mode sets (e.g. the families n of a coplanar component) into one GSM.

    The result uses `basis1` / `basis2` if given (they must contain exactly the union of the modes), otherwise the
    union of the port bases in the standard order.
    """
    if not gsms:
        raise ValueError("nothing to combine")
    g0 = gsms[0]
    for g in gsms[1:]:
        for port in ("basis1", "basis2"):
            p, q = getattr(g0, port), getattr(g, port)
            if not (np.isclose(p.a, q.a, rtol=1e-12, atol=0) and np.isclose(p.b, q.b, rtol=1e-12, atol=0)
                    and p.frame == q.frame):
                raise ValueError(f"block_diag: {port} of {g!r} has another cross-section or frame than {g0!r}")
        if (g.f is None) != (g0.f is None) or (g.f is not None and not np.isclose(g.f, g0.f, rtol=1e-12, atol=0)):
            raise ValueError("block_diag: GSMs at different frequencies")
    a, b = g0.basis1.a, g0.basis1.b
    a2, b2 = g0.basis2.a, g0.basis2.b
    modes1 = [m for g in gsms for m in g.basis1]
    modes2 = [m for g in gsms for m in g.basis2]
    basis1 = ModeBasis(a, b, modes1, frame=g0.basis1.frame) if basis1 is None else basis1
    basis2 = ModeBasis(a2, b2, modes2, frame=g0.basis2.frame) if basis2 is None else basis2
    if sorted(map(str, basis1)) != sorted(map(str, modes1)) or sorted(map(str, basis2)) != sorted(map(str, modes2)):
        raise ValueError("basis1/basis2 must contain exactly the modes of the GSMs being combined")
    N1, N2 = len(basis1), len(basis2)
    S11 = np.zeros((N1, N1), complex); S12 = np.zeros((N1, N2), complex)
    S21 = np.zeros((N2, N1), complex); S22 = np.zeros((N2, N2), complex)
    for g in gsms:
        i1 = [basis1.index(m) for m in g.basis1]
        i2 = [basis2.index(m) for m in g.basis2]
        S11[np.ix_(i1, i1)] = g.S11
        S12[np.ix_(i1, i2)] = g.S12
        S21[np.ix_(i2, i1)] = g.S21
        S22[np.ix_(i2, i2)] = g.S22
    f = gsms[0].f
    return GSM(S11, S12, S21, S22, basis1, basis2, f, name)
