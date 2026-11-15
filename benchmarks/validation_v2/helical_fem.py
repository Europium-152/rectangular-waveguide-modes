"""Full-wave reference for twists (WP7b, verifies WP6): eigenmodes of a uniformly twisted rectangular guide by a 2D
finite-element eigenproblem in helical coordinates, matched to the straight guide at the end planes.

Method (independent of the T3/T3b coupled-mode matrices; no modal expansion of the fields inside the twist):

* Helical (co-rotating) coordinates (u, v, z) with r = R(tau z)(u x + v y) + z z_hat. The twisted guide becomes a
  straight a x b guide filled with eps0 G, mu0 G, G = g^{ij} = [[I + w w^T, -w], [-w^T, 1]], w = tau (-v, u),
  sqrt(g) = 1, for the covariant field components (T3 section 1; standard transformation optics). Its transverse
  covariant components are the physical components on the rotating axes, so they are continuous at a plane
  z = const where the guide changes (end of the twist, change of rate).
* For constant tau the problem is z-invariant: E = E(u, v) exp(-i gamma z). The weak form
  int (curl_{-gamma} F)^T G^{-1} (curl_gamma E) - k^2 F^T G E = 0 (bilinear, no conjugation) is discretized with
  second- or third-order Nedelec (first kind) elements for (E_u, E_v) and Lagrange elements of the same order for
  E_z (an exact discrete de Rham pair, no spurious modes at gamma != 0), on a structured triangulation of the
  rectangle, PEC on its boundary. This gives the quadratic eigenproblem
      [K(tau) - k^2 M(tau)] x + gamma B(tau) x + gamma^2 C x = 0,
  linearized on [x; gamma x] and solved by shift-invert Arnoldi about gamma = sigma (the modes of smallest |gamma -
  sigma|).
* H_t of each eigenmode is H = (i/(k eta0)) G^{-1} curl_gamma E (covariant). Projecting E_t and H_t on the
  CONVENTIONS straight-guide fields e_s and z x e_s (quadrature on the FEM mesh) gives the straight-guide
  amplitudes a^+_s, a^-_s (CONVENTIONS section 5) of every twisted eigenmode at a plane z = const.
* A section of constant rate and length l between two planes is solved by mode matching on those planes
  (Galerkin with the N straight modes as test functions, N forward and N backward twisted modes as unknowns): its
  GSM in the straight-guide basis follows from a 2N x 2N linear system (`section_gsm`). A variable rate is a
  cascade of constant-rate sections (midpoint rates) joined at planes; the result converges as O(1/n^2) in the
  number n of sections and is extrapolated (`twist_gsm`).

Port frame: the basis at both ports is the straight guide in the co-rotating frame (u along side a), which is
the global frame of CONVENTIONS.md section 2 after a twist (the frame rotates with the guide). A positive tau turns
the cross-section from +x towards +y (right-handed about +z), the sense of `wrmodes.Twist`.

Only merged wave 1 code is used: `wrmodes.modes` (fields, admittances), `wrmodes.gsm` (GSM, cascade).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import Basis, BilinearForm, ElementTriN2, ElementTriN3, ElementTriP2, ElementTriP3, MeshTri

from wrmodes.gsm import GSM
from wrmodes.modes import ETA0, ModeBasis, transverse_field, wave_admittance

C0 = 299792458.0


def _curl_terms():
    """Bilinear forms of the tau-polynomial pieces. Trial (Et, Ez), test (Ft, Fz); rho = (u, v), w = tau (-v, u)."""

    def rho(w):
        return w.x[0], w.x[1]

    def dotrho(u, v, vec):        # rho . vec
        return u * vec[0] + v * vec[1]

    def dotw1(u, v, vec):         # (w / tau) . vec = -v vec_u + u vec_v
        return -v * vec[0] + u * vec[1]

    # K(tau) = K0 + tau K1 + tau^2 K2  (A0 F)^T g (A0 E), g = G^{-1} = [[1, 0, w_u], [0, 1, w_v], [w_u, w_v, 1 + |w|^2]]
    @BilinearForm
    def K0(Et, Ez, Ft, Fz, w):
        return Ez.grad[0] * Fz.grad[0] + Ez.grad[1] * Fz.grad[1] + Et.curl * Ft.curl

    @BilinearForm
    def K1(Et, Ez, Ft, Fz, w):
        u, v = rho(w)
        # w x grad(phi) = w_u d_v phi - w_v d_u phi = -tau rho . grad(phi)
        return -Et.curl * dotrho(u, v, Fz.grad) - Ft.curl * dotrho(u, v, Ez.grad)

    @BilinearForm
    def K2(Et, Ez, Ft, Fz, w):
        u, v = rho(w)
        return (u * u + v * v) * Et.curl * Ft.curl

    # B(tau) = i (B0 + tau B1)  [antisymmetric real parts]
    @BilinearForm
    def B0(Et, Ez, Ft, Fz, w):
        return Fz.grad[0] * Et.value[0] + Fz.grad[1] * Et.value[1] - Ft.value[0] * Ez.grad[0] - Ft.value[1] * Ez.grad[1]

    @BilinearForm
    def B1(Et, Ez, Ft, Fz, w):
        u, v = rho(w)
        # curlF (w x E_t) - (w x F_t) curlE, with w x E = -tau rho . E
        return -Ft.curl * dotrho(u, v, Et.value) + dotrho(u, v, Ft.value) * Et.curl

    # mass F^T G E = F_t.(I + w w^T) E_t - F_z (w.E_t) - (w.F_t) E_z + F_z E_z
    @BilinearForm
    def M0(Et, Ez, Ft, Fz, w):
        return Et.value[0] * Ft.value[0] + Et.value[1] * Ft.value[1] + Ez.value * Fz.value

    @BilinearForm
    def M1(Et, Ez, Ft, Fz, w):
        u, v = rho(w)
        return -Fz.value * dotw1(u, v, Et.value) - dotw1(u, v, Ft.value) * Ez.value

    @BilinearForm
    def M2(Et, Ez, Ft, Fz, w):
        u, v = rho(w)
        return dotw1(u, v, Ft.value) * dotw1(u, v, Et.value)

    @BilinearForm
    def Ctt(Et, Ez, Ft, Fz, w):
        return Et.value[0] * Ft.value[0] + Et.value[1] * Ft.value[1]

    return dict(K0=K0, K1=K1, K2=K2, B0=B0, B1=B1, M0=M0, M1=M1, M2=M2, C=Ctt)


class HelicalGuide:
    """FEM discretization of the a x b cross-section in helical coordinates (any constant rate tau).

    Parameters
    ----------
    a, b : float
        Sides (m) along u and v (CONVENTIONS frame; side a along u).
    h : float
        Target element size (m) of the structured triangulation (squares cut into two triangles).
    order : 2 or 3
        Polynomial order of the Nedelec / Lagrange pair.
    """

    def __init__(self, a, b, h, order=2):
        self.a, self.b, self.order = a, b, order
        nx, ny = max(2, int(np.ceil(a / h))), max(2, int(np.ceil(b / h)))
        self.mesh = MeshTri.init_tensor(np.linspace(-a / 2, a / 2, nx + 1), np.linspace(-b / 2, b / 2, ny + 1))
        el = (ElementTriN2() * ElementTriP2()) if order == 2 else (ElementTriN3() * ElementTriP3())
        self.basis = Basis(self.mesh, el, intorder=2 * order + 2)
        forms = _curl_terms()
        self.mat = {k: f.assemble(self.basis).tocsr() for k, f in forms.items()}
        bnd = self.basis.get_dofs().all()
        self.free = np.setdiff1d(np.arange(self.basis.N), bnd)
        self.matf = {k: m[self.free][:, self.free].tocsc() for k, m in self.mat.items()}
        self.n = len(self.free)
        self._quad = None

    # ---------------------------------------------------------------- eigenproblem
    def operators(self, tau, k):
        m = self.matf
        A = (m["K0"] + tau * m["K1"] + tau ** 2 * m["K2"]) - k ** 2 * (m["M0"] + tau * m["M1"] + tau ** 2 * m["M2"])
        B = 1j * (m["B0"] + tau * m["B1"])
        return A.astype(complex), B.tocsc(), m["C"].astype(complex)

    def modes(self, tau, f, n_eig, sigma=None, tol=1e-12):
        """Eigenpairs (gamma, x) of the twisted guide at f (Hz), the n_eig closest to sigma (default 1e-3 k (1 - i))."""
        k = 2 * np.pi * f / C0
        A, B, C = self.operators(tau, k)
        if sigma is None:
            sigma = 1e-3 * k * (1 - 1j)
        P = (A + sigma * B + sigma ** 2 * C).tocsc()
        lu = spla.splu(P)
        n = self.n
        BsC = (B + sigma * C).tocsr()
        Cc = C.tocsr()

        # L z = gamma Mz z with L = [[0, I], [-A, -B]], Mz = diag(I, C); operator (L - sigma Mz)^{-1} Mz
        def op(zz):
            x, y = zz[:n], zz[n:]
            r1, r2 = x, Cc @ y
            xs = -lu.solve(r2 + BsC @ r1)
            ys = r1 + sigma * xs
            return np.concatenate([xs, ys])

        Op = spla.LinearOperator((2 * n, 2 * n), matvec=op, dtype=complex)
        nu, V = spla.eigs(Op, k=n_eig, which="LM", tol=tol, ncv=min(2 * n, max(2 * n_eig + 20, 60)))
        gam = sigma + 1.0 / nu
        X = V[:n, :]
        # residual check of the quadratic problem
        res = []
        for j in range(len(gam)):
            x = X[:, j]
            r = A @ x + gam[j] * (B @ x) + gam[j] ** 2 * (C @ x)
            res.append(np.linalg.norm(r) / max(np.linalg.norm(A @ x), 1e-300))
        return gam, X, np.array(res)

    def forward_modes(self, tau, f, n_prop, n_evan, kmax, margin=12, tol=1e-12):
        """Forward eigenpairs only, from two shift-invert runs: sigma = k/2 (every forward propagating mode lies
        within k/2 of it, every other mode farther) and sigma = -i kmax/2 (forward evanescent modes with
        |gamma| <= kmax). Returns (gamma, X, residuals) of the forward modes found (duplicates merged)."""
        k = 2 * np.pi * f / C0
        out_g, out_x, out_r = [], [], []
        runs = [(0.5 * k * (1 + 1e-7j), n_prop + margin)]
        if n_evan > 0:
            runs.append((-0.5j * kmax + 1e-7 * k, n_evan + margin))
        for sigma, nev in runs:
            nev = min(nev, 2 * self.n - 2)
            g, X, r = self.modes(tau, f, nev, sigma=sigma, tol=tol)
            for j in range(len(g)):
                fwd = (abs(g[j].imag) < 1e-9 * k and g[j].real > 0) or g[j].imag < -1e-9 * k
                if fwd and all(abs(g[j] - h) > 1e-9 * k for h in out_g):
                    out_g.append(g[j])
                    out_x.append(X[:, j])
                    out_r.append(r[j])
        return np.array(out_g), np.array(out_x).T, np.array(out_r)

    # ---------------------------------------------------------------- fields and projections
    def _full(self, x):
        z = np.zeros(self.basis.N, dtype=complex)
        z[self.free] = x
        return z

    def quad(self):
        if self._quad is None:
            xg = self.basis.global_coordinates().value  # (2, nel, nqp)
            self._quad = (xg[0].ravel(), xg[1].ravel(), np.asarray(self.basis.dx).ravel())
        return self._quad

    def fields(self, x, gamma, tau, k):
        """(E_t, H_t) at the quadrature points, H normalized as (i/(k eta0)) G^{-1} curl_gamma E."""
        z = self._full(x)
        ft, fz = self.basis.interpolate(z.real)
        gt, gz = self.basis.interpolate(z.imag)
        Et = ft.value + 1j * gt.value          # (2, nel, nqp)
        curl = ft.curl + 1j * gt.curl
        Ez = fz.value + 1j * gz.value
        gEz = fz.grad + 1j * gz.grad
        u, v, _ = self.quad()
        Et = Et.reshape(2, -1)
        curl, Ez, gEz = curl.ravel(), Ez.ravel(), gEz.reshape(2, -1)
        cu = gEz[1] + 1j * gamma * Et[1]
        cv = -1j * gamma * Et[0] - gEz[0]
        cz = curl
        wu, wv = -tau * v, tau * u
        Hu = (1j / (k * ETA0)) * (cu + wu * cz)
        Hv = (1j / (k * ETA0)) * (cv + wv * cz)
        return Et, np.vstack([Hu, Hv]), Ez

    def straight_fields(self, basis):
        if getattr(self, "_sf_key", None) is not basis:
            u, v, _ = self.quad()
            self._sf = np.array([transverse_field(m, self.a, self.b, u, v) for m in basis.modes])  # (N, 2, nq)
            self._sf_key = basis
        return self._sf

    def amplitudes(self, x, gamma, tau, f, basis):
        """Straight-guide amplitudes (a^+, a^-) of the twisted eigenmode x at a plane z = const."""
        k = 2 * np.pi * f / C0
        Et, Ht, _ = self.fields(x, gamma, tau, k)
        e = self.straight_fields(basis)
        _, _, dx = self.quad()
        V = np.einsum("sjq,jq,q->s", e, Et, dx)
        zxe = np.stack([-e[:, 1], e[:, 0]], axis=1)
        I = np.einsum("sjq,jq,q->s", zxe, Ht, dx)
        Y = np.array([wave_admittance(m, self.a, self.b, f) for m in basis.modes], dtype=complex)
        sp_ = V * np.sqrt(Y / 2)
        si = I / np.sqrt(2 * Y)
        resid = None
        return 0.5 * (sp_ + si), 0.5 * (sp_ - si)

    def norm_t(self, x, gamma, tau, f):
        """L2 norm of E_t (for checking the completeness of the projections)."""
        k = 2 * np.pi * f / C0
        Et, _, _ = self.fields(x, gamma, tau, k)
        _, _, dx = self.quad()
        return np.sqrt(np.sum(np.abs(Et) ** 2 * dx))


def select_modes(gam, X, n_fwd, k):
    """Split eigenpairs into forward and backward sets of n_fwd modes each, ordered by k^2 - gamma^2 (the
    'effective cutoff'); forward = Re gamma > 0 for propagating, Im gamma < 0 for evanescent."""
    tol = 1e-9 * k
    fwd = [j for j in range(len(gam)) if (abs(gam[j].imag) < tol and gam[j].real > 0) or gam[j].imag < -tol]
    bwd = [j for j in range(len(gam)) if (abs(gam[j].imag) < tol and gam[j].real < 0) or gam[j].imag > tol]
    key = lambda j: (k ** 2 - gam[j] ** 2).real
    fwd = sorted(fwd, key=key)[:n_fwd]
    bwd = sorted(bwd, key=key)[:n_fwd]
    if len(fwd) < n_fwd or len(bwd) < n_fwd:
        raise RuntimeError(f"not enough eigenmodes: {len(fwd)} forward, {len(bwd)} backward, need {n_fwd}")
    return fwd, bwd


class Section:
    """Eigenmodes of one constant-rate section, projected on a straight basis.

    Forward modes are chosen by assignment: each straight mode s of the basis gets the forward eigenmode whose E_t
    has the largest fraction of its L2 norm in s (Hungarian assignment over the candidates), so that a truncation
    never pairs a straight mode with a wrong twisted one. The backward modes are not taken from the eigensolver:
    they follow exactly from the forward ones by the symmetry of the twisted guide under the rotation by pi about
    the u axis, (u, v, z) -> (u, -v, -z), which maps a forward mode with gamma to a backward mode with -gamma and
    straight-mode amplitudes a^+_b = P a^-_f, a^-_b = P a^+_f, P = (-1)^(n+1) (CONVENTIONS.md section 2, turning a
    component around)."""

    def __init__(self, guide, tau, f, basis, extra=20):
        from scipy.optimize import linear_sum_assignment
        self.tau, self.f, self.basis = tau, f, basis
        k = 2 * np.pi * f / C0
        N = len(basis.modes)
        prop = basis.propagating_mask(f)
        n_prop = int(np.sum(prop))
        chi = np.array([np.pi * np.hypot(m.m / basis.a, m.n / basis.b) for m in basis.modes])
        kmax = np.sqrt(max(chi.max() ** 2 - k ** 2, 0.0)) * 1.05 + 1e-3 * k
        gam, X, res = guide.forward_modes(tau, f, n_prop, N - n_prop, kmax, margin=extra)
        self.residual = float(np.max(res))
        fwd = sorted(range(len(gam)), key=lambda j: (k ** 2 - gam[j] ** 2).real)[:N + extra]
        if len(fwd) < N:
            raise RuntimeError(f"only {len(fwd)} forward eigenmodes for {N} straight modes")
        Pc, Qc, frac = [], [], np.zeros((N, len(fwd)))
        e = guide.straight_fields(basis)
        _, _, dx = guide.quad()
        for c, j in enumerate(fwd):
            ap, am = guide.amplitudes(X[:, j], gam[j], tau, f, basis)
            Pc.append(ap)
            Qc.append(am)
            Et, _, _ = guide.fields(X[:, j], gam[j], tau, k)
            V = np.einsum("sjq,jq,q->s", e, Et, dx)
            frac[:, c] = np.abs(V) ** 2 / np.sum(np.abs(Et) ** 2 * dx)
        rows, cols = linear_sum_assignment(-frac)
        order = cols[np.argsort(rows)]
        fr = frac[np.arange(N), order]
        self.assignment_min_fraction = float(np.min(fr))
        self.assignment_worst = basis.names[int(np.argmin(fr))]
        self.assignment_fractions = fr
        self.gamma_f = gam[np.array(fwd)[order]]
        self.Pf = np.array(Pc).T[:, order]
        self.Qf = np.array(Qc).T[:, order]
        Psign = np.array([(-1.0) ** (m.n + 1) for m in basis.modes])
        self.gamma_b = -self.gamma_f
        self.Pb = Psign[:, None] * self.Qf
        self.Qb = Psign[:, None] * self.Pf
        self.k = k

    def gsm_blocks(self, length):
        """(S11, S21, S12, S22) of the section of this rate between two planes `length` apart."""
        N = self.Pf.shape[0]
        Df = np.exp(-1j * self.gamma_f * length)        # forward modes, decay for evanescent (Im < 0)
        Db = np.exp(-1j * self.gamma_f * length)        # backward modes: exp(+i gamma_f z), gamma_b = -gamma_f
        # unknowns c+ (forward amplitudes at z = 0) and c-' (backward amplitudes at z = l)
        # A+(0) = Pf c+ + Pb Db c-' ;  A-(l) = Qf Df c+ + Qb c-'
        M = np.block([[self.Pf, self.Pb * Db[None, :]], [self.Qf * Df[None, :], self.Qb]])
        Minv = np.linalg.inv(M)
        # outputs: A-(0) = Qf c+ + Qb Db c-' ;  A+(l) = Pf Df c+ + Pb c-'
        O = np.block([[self.Qf, self.Qb * Db[None, :]], [self.Pf * Df[None, :], self.Pb]])
        S = O @ Minv
        return S[:N, :N], S[N:, :N], S[:N, N:], S[N:, N:]


def gsm_from_blocks(blocks, basis, f):
    S11, S21, S12, S22 = blocks
    return GSM(S11, S12, S21, S22, basis, basis, f)


def constant_twist(guide, angle, length, f, basis):
    """GSM of a constant-rate twist of `angle` (rad) over `length` (m)."""
    sec = Section(guide, angle / length, f, basis)
    return gsm_from_blocks(sec.gsm_blocks(length), basis, f), sec


def piecewise_twist(guide, rate, length, f, basis, n_sections, cache=None):
    """Cascade of n_sections constant-rate sections, rate(z) at the midpoints (rad/m)."""
    from wrmodes.gsm import cascade
    dz = length / n_sections
    zs = (np.arange(n_sections) + 0.5) * dz
    cache = {} if cache is None else cache
    gs = []
    for z in zs:
        tau = float(rate(z))
        key = round(tau, 12)
        if key not in cache:
            cache[key] = Section(guide, tau, f, basis)
        gs.append(gsm_from_blocks(cache[key].gsm_blocks(dz), basis, f))
    return cascade(*gs)


def straight_basis(a, b, f, cutoff_factor):
    """All modes of every family with cutoff below cutoff_factor * f (a twist couples every family)."""
    return ModeBasis.propagating(f, a, b, cutoff_factor=cutoff_factor, evanescent_families="all")
