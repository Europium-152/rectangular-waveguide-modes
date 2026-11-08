"""Transfer-matrix solver for the coupled-mode equations of a bent rectangular waveguide.

The coupled-mode equations for the forward (A+) and backward (A-) mode amplitudes are linear,

    dy/dz = M(z) y,     y = [A+_1, ..., A+_N, A-_1, ..., A-_N],     M(z) = D + kappa(z) K,

where D = diag(-i beta, +i beta) contains the propagation constants and the constant matrix K the
coupling coefficients. The bend is therefore completely described by its 2N x 2N transfer matrix
Phi, with y(L) = Phi y(0).

Phi is computed as a product of matrix exponentials over short slices of the bend, using the
fourth-order Magnus integrator with two Gauss-Legendre points per slice:

    Omega_k = h/2 (M1 + M2) + sqrt(3)/12 h^2 [M2, M1],     Phi = exp(Omega_n) ... exp(Omega_1).

Because the exponential propagates the fast phase exp(-i beta z) of every mode exactly, the slices only
have to resolve the variation of the curvature kappa(z) and the beat lengths 2 pi / |beta_p - beta_q| of the
pairs of modes that exchange power, not the guide wavelength. For a constant curvature the result is exact
for any number of slices. The number of slices is chosen adaptively: starting from a minimum set by the beat
lengths, it is doubled until the Richardson estimate of the error of the fourth-order method is below the
requested tolerance, with a check that the convergence is in its asymptotic regime.

Once Phi is known, the boundary conditions (prescribed incident forward amplitudes at z = 0, no
backward wave incident at z = L) are imposed algebraically, which gives the forward transmission
matrix T and the reflection matrix R of the bend for every input mode at once:

    A+(L) = T A+(0),     A-(0) = R A+(0).

This approach is well conditioned because only propagating modes are retained (all solutions
oscillate); it must not be used with evanescent modes in the expansion.
"""
import contextlib
import warnings
import numpy as np
import scipy.constants as cst
from scipy.linalg import expm

from .main import list_propagating_modes
from .mode_coupling import coupling_coefficient

_GAUSS_1 = 0.5 - np.sqrt(3.0) / 6.0
_GAUSS_2 = 0.5 + np.sqrt(3.0) / 6.0
_MAGNUS_C = np.sqrt(3.0) / 12.0


# ----------------------------------------------------------------------------------------- mode set
def _mode_name(mode):
    """('TE', 1, 0) -> 'TE1,0' (the naming used by list_propagating_modes)."""
    return f"{mode[0]}{int(mode[1])},{int(mode[2])}"



_THREADPOOL_CONTROLLER = None


def _blas_limit(n_threads):
    """Context manager limiting the BLAS/LAPACK threads to n_threads (no-op if None or threadpoolctl is missing)."""
    global _THREADPOOL_CONTROLLER
    if n_threads is None:
        return contextlib.nullcontext()
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:
        return contextlib.nullcontext()
    if _THREADPOOL_CONTROLLER is None:
        _THREADPOOL_CONTROLLER = ThreadpoolController()      # inspects the loaded libraries once (a few ms)
    return _THREADPOOL_CONTROLLER.limit(limits=n_threads, user_api='blas')


def select_coupled_modes(f, a, b, mode_in=None, n=None, coupled_modes=None, maximum_modes=None):
    """Propagating modes that can couple to `mode_in` (same vertical index n), sorted by cutoff.

    Either `mode_in` (type, m, n) or the vertical index `n` must be given. `coupled_modes` overrides the
    automatic selection; `maximum_modes` truncates the list to the lowest-order modes.
    """
    if coupled_modes is None:
        if n is None:
            n = int(mode_in[2])
        coupled_modes = [mode for mode in list_propagating_modes(f, a, b) if mode[2] == n]
    coupled_modes = list(coupled_modes)
    if maximum_modes is not None and len(coupled_modes) > maximum_modes:
        coupled_modes = coupled_modes[:maximum_modes]
    return coupled_modes


def propagation_constants(f, a, b, modes):
    """Propagation constants beta_p (1/m) of the given modes at frequency f."""
    k = 2 * cst.pi * f / cst.c
    chi2 = np.array([cst.pi ** 2 * ((m / a) ** 2 + (n / b) ** 2) for _, m, n, *_ in modes])
    return np.sqrt(k ** 2 - chi2 + 0j)


def coupled_mode_matrices(f, a, b, modes):
    """Matrices of the coupled-mode system dy/dz = (D + kappa(z) K) y.

    Returns
    -------
    D : (2N, 2N) complex ndarray
        diag(-i beta, +i beta).
    K : (2N, 2N) complex ndarray
        Coupling matrix (to be multiplied by the local curvature in 1/m).
    beta : (N,) complex ndarray
        Propagation constants (1/m).
    """
    N = len(modes)
    beta = propagation_constants(f, a, b, modes)
    if np.any(beta.imag != 0):
        warnings.warn("coupled_mode_matrices: evanescent modes in the expansion. The coupled-mode equations of "
                      "wrmodes are written for propagating modes; below cutoff this function's beta is +i|beta|, "
                      "opposite to the branch of CONVENTIONS.md (wrmodes.modes.propagation_constant).", RuntimeWarning)
    Cp = np.zeros((N, N), dtype=complex)
    Cm = np.zeros((N, N), dtype=complex)
    for i, p in enumerate(modes):
        for j, q in enumerate(modes):
            # coupling_coefficient already contains the factor i of eq. (5.2)
            Cp[i, j] = coupling_coefficient(tuple(p[:3]), tuple(q[:3]), 1, f, a, b)
            Cm[i, j] = coupling_coefficient(tuple(p[:3]), tuple(q[:3]), -1, f, a, b)
    D = np.zeros((2 * N, 2 * N), dtype=complex)
    D[:N, :N] = np.diag(-1j * beta)
    D[N:, N:] = np.diag(+1j * beta)
    K = np.empty((2 * N, 2 * N), dtype=complex)
    K[:N, :N] = -Cp
    K[:N, N:] = -Cm
    K[N:, :N] = Cm
    K[N:, N:] = Cp
    return D, K, beta


# ----------------------------------------------------------------------------------------- Magnus
def _curvature_at(curvature, z):
    z = np.asarray(z, dtype=float)
    return np.broadcast_to(np.asarray(curvature(z), dtype=float), z.shape)


def _slice_exponentials(D, K, curvature, edges):
    """exp(Omega_k) for each slice [edges[k], edges[k+1]] (fourth-order Magnus, two Gauss points).

    Returns an array of shape (n_slices, 2N, 2N); the exponentials are computed in one batched call.
    """
    h = np.diff(edges)
    k1 = _curvature_at(curvature, edges[:-1] + _GAUSS_1 * h)
    k2 = _curvature_at(curvature, edges[:-1] + _GAUSS_2 * h)
    DK = D @ K - K @ D   # [M2, M1] = (k2 - k1) [K, D] = (k1 - k2) [D, K], since D and K are constant
    omega = (h[:, None, None] * D[None]
             + (0.5 * h * (k1 + k2))[:, None, None] * K[None]
             + (_MAGNUS_C * h * h * (k1 - k2))[:, None, None] * DK[None])
    return expm(omega)


def _slice_edges(length, base_counts, points, level):
    """Slice boundaries: each segment [points[i], points[i+1]] is split uniformly into base_counts[i] * 2**level."""
    parts = [np.linspace(z0, z1, c * 2 ** level + 1)[:-1] for z0, z1, c in zip(points[:-1], points[1:], base_counts)]
    return np.concatenate(parts + [np.array([length])])


def _product(exps):
    """exps[n-1] @ ... @ exps[0], by pairwise (tree) multiplication."""
    mats = exps
    while len(mats) > 1:
        if len(mats) % 2:
            last, mats = mats[-1:], mats[:-1]
        else:
            last = None
        mats = mats[1::2] @ mats[0::2]
        if last is not None:
            mats = np.concatenate([mats, last])
    return mats[0]


def _scattering_from_transfer(phi, N):
    """T and R (forward incidence from z = 0) from the transfer matrix."""
    Pff, Pfb, Pbf, Pbb = phi[:N, :N], phi[:N, N:], phi[N:, :N], phi[N:, N:]
    R = -np.linalg.solve(Pbb, Pbf)
    T = Pff + Pfb @ R
    return T, R


# ----------------------------------------------------------------------------------------- result
class TransferMatrixResult:
    """Result of :func:`solve_transfer_matrix`.

    Attributes
    ----------
    modes : list
        Modes of the expansion, as (name, m, n, cutoff) tuples.
    beta : ndarray (N,)
        Propagation constants (1/m).
    T, R : ndarray (N, N)
        Transmission (A+(L) = T A+(0)) and reflection (A-(0) = R A+(0)) matrices of the bend.
    transfer_matrix : ndarray (2N, 2N)
        Full transfer matrix, y(L) = transfer_matrix @ y(0).
    x : ndarray (M,)
        Positions (m) at which the amplitude profile is given.
    y : ndarray (2N, M)
        Amplitudes of the forward (rows 0..N-1) and backward (rows N..2N-1) modes at the positions x, for
        the input mode `mode_in` with unit amplitude (same layout as scipy.integrate.solve_bvp results).
    input_index : int
        Index of the input mode in `modes`.
    n_slices : int
        Number of Magnus slices of the returned solution.
    error_estimate : float
        Estimated maximum absolute error of the quantities selected by `error_control`.
    reflection_error_estimate : float
        Estimated maximum absolute error of the reflection amplitudes (column of R for the input mode, or all
        of R with error_control='full').
    history : list of (n_slices, error_estimate)
        Convergence history of the adaptive slicing.
    success, status, message
        Convergence flag (status 0: converged; 1: maximum number of slices reached), as in solve_bvp.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getitem__(self, key):  # dict-like access, as for scipy OptimizeResult
        return getattr(self, key)

    def __repr__(self):
        return (f"TransferMatrixResult(N={len(self.modes)} modes, n_slices={self.n_slices}, "
                f"error_estimate={self.error_estimate:.1e}, success={self.success})")


# ----------------------------------------------------------------------------------------- solver
def solve_transfer_matrix(f, a, b, mode_in, length, curvature, coupled_modes=None, maximum_modes=None,
                          atol=1e-6, error_control='input', initial_slices=8, max_slices=4096, breakpoints=None,
                          max_phase_per_slice=np.pi, z_eval=None, profile=True, verbose=False, blas_threads=1,
                          loss_matrix=None):
    """Solve the coupled-mode equations of a bent rectangular waveguide with the transfer-matrix method.

    Parameters
    ----------
    f : float
        Frequency (Hz).
    a, b : float
        Waveguide sides (m); the bend is in the plane of side `a` (curvature applied along `a`).
    mode_in : tuple (str, int, int)
        Input mode, e.g. ('TE', 1, 0). It selects the coupled mode family (same vertical index n) and the
        column of T used for the amplitude profile `y`.
    length : float
        Length of the bend (m).
    curvature : callable
        kappa(z) in 1/m for z in [0, length]; must accept a NumPy array.
    coupled_modes, maximum_modes : optional
        Override / truncate the automatic mode selection (all propagating modes of the family).
    atol : float
        Target absolute accuracy. With error_control='input' (default): accuracy of the output power of every
        mode, as a fraction of the input power; the default 1e-6 corresponds to about 4e-6 dB on the
        mode-conversion loss. With error_control='full': accuracy of the complex entries of T and R.
    error_control : {'input', 'full'}
        Which quantities the error estimate covers. 'input' (default): the transmitted power of every mode for
        the given input mode, which determines the mode-conversion loss and the converted powers. Tiny
        amplitudes in strongly phase-mismatched (e.g. near-cutoff) modes converge more slowly but carry
        negligible power; their phases are not error-controlled in this mode. 'full': all complex entries of
        T and R, as needed to cascade bends or to use the reflections. The
        reflections of gently curved bends are tiny (typically -100 dB) and converge more slowly, because the
        forward-backward coupling oscillates with beta_p + beta_q; they are only error-controlled with 'full'.
    initial_slices, max_slices : int
        Number of slices of the first Magnus evaluation, and upper limit of the adaptive doubling.
    breakpoints : array_like, optional
        Positions (m) inside the bend where the curvature is discontinuous (e.g. a step in curvature, or the
        junction of the two halves of an S-bend). Slice boundaries are placed on them, which keeps the
        fourth-order convergence. A discontinuity inside a slice degrades the convergence to first order: the
        result is still correct, but many more slices are needed (and a warning is issued if max_slices is
        reached).
    max_phase_per_slice : float
        Upper limit (rad) of the relative phase |beta_p - beta_q| h accumulated over one slice by any pair of
        coupled modes that can exchange a relevant amount of power. It sets the starting (minimum) number of
        slices, so that the error estimate operates in the asymptotic (fourth-order) regime. The default, pi, means
        that every relevant pair of modes slips by at most half a beat length per slice; it was validated over
        1536 test cases (benchmarks/validation). None disables the rule.
    z_eval : array_like, optional
        Positions (m) at which to return the amplitude profile. By default, the slice boundaries.
    profile : bool
        Compute the amplitude profile y(z) (cheap). If False, x = [0, L] and y holds the end values only.
    verbose : bool
        Print the mode set and the convergence history.
    blas_threads : int or None
        Number of threads the BLAS/LAPACK library may use during the solution (default 1). The solver performs
        thousands of operations on small (2N x 2N) matrices, for which a multithreaded BLAS only adds overhead:
        on machines with many cores it can make the solver several times slower. The limit is applied with
        threadpoolctl (if installed) and restored on return; None leaves the BLAS threading unchanged. To use
        several cores, run different frequencies in parallel processes instead.
    loss_matrix : array or (array, array), optional
        Ohmic wall loss (wrmodes 1.0, docs/theory/T1-wall-impedance.md box B4b): Lambda0_full, or the pair
        (Lambda0_full, Lambda1_full), 2N x 2N in the order of the mode set (e.g. from
        `wrmodes.loss.bend_loss_matrices`). The system becomes
        dy/dz = [(D - Lambda0_full) + kappa(z) (K - Lambda1_full)] y; it keeps the form "constant + kappa(z) x
        constant", so the Magnus integrator, the error estimate and the adaptive slicing apply unchanged (the
        minimum number of slices also counts the pairs coupled by kappa Lambda1). None (default) is the lossless
        wrmodes 0.1 solver, bit for bit.

    Returns
    -------
    modes : list
        Modes of the expansion.
    result : TransferMatrixResult
        See :class:`TransferMatrixResult`.
    """
    with _blas_limit(blas_threads):
        return _solve_transfer_matrix(f, a, b, mode_in, length, curvature, coupled_modes, maximum_modes, atol,
                                      error_control, initial_slices, max_slices, breakpoints, max_phase_per_slice,
                                      z_eval, profile, verbose, loss_matrix)


def _solve_transfer_matrix(f, a, b, mode_in, length, curvature, coupled_modes, maximum_modes, atol, error_control,
                           initial_slices, max_slices, breakpoints, max_phase_per_slice, z_eval, profile, verbose,
                           loss_matrix=None):
    if error_control not in ('input', 'full'):
        raise ValueError("error_control must be 'input' or 'full'")
    modes = select_coupled_modes(f, a, b, mode_in, coupled_modes=coupled_modes, maximum_modes=maximum_modes)
    names = [m[0] for m in modes]
    name_in = _mode_name(mode_in)
    if name_in not in names:
        raise ValueError(f"Input mode {name_in} is not among the propagating coupled modes {names} at "
                         f"{f/1e9:.3f} GHz")
    j = names.index(name_in)
    N = len(modes)
    D, K, beta = coupled_mode_matrices(f, a, b, modes)
    if loss_matrix is not None:
        if isinstance(loss_matrix, tuple) and len(loss_matrix) == 2 and loss_matrix[1] is None:
            loss_matrix = loss_matrix[0]
        lm = np.asarray(loss_matrix, dtype=complex)
        if lm.shape == D.shape:                                  # Lambda0_full alone (array or nested lists)
            L0, L1 = lm, None
        elif lm.shape == (2,) + D.shape:                         # the pair (Lambda0_full, Lambda1_full)
            L0, L1 = lm[0], lm[1]
        else:
            raise ValueError(f"loss_matrix must be Lambda0_full {D.shape} or the pair (Lambda0_full, Lambda1_full), "
                             f"got shape {lm.shape} (the mode set has {N} modes)")
        D = D - L0
        if L1 is not None:
            K = K - np.asarray(L1, dtype=complex)
    if verbose:
        print(f"{N} coupled modes at {f/1e9:.3f} GHz: {', '.join(names)}")

    # Adaptive doubling of the number of slices. The error of the finer result is estimated from the difference
    # of two successive results (Richardson estimate for a 4th-order method: |X_2n - X_n| / 15). The estimate is
    # only accepted once two successive ratios of differences show (at least) fourth-order convergence (>= 8,
    # theoretical value 16), which protects against inaccurate pre-asymptotic results that agree by chance (a single
    # large ratio followed by a small one). Results whose differences are close to round-off, where the ratios are
    # erratic, are accepted without the ratio test when two successive differences are both below atol * 1e-3
    # (three successive results agree); a single small difference is not enough, because two pre-asymptotic results
    # can agree by chance.
    def measure(T, R):
        if error_control == 'input':
            return np.abs(T[:, j]) ** 2, R[:, j]    # output power of every mode for the given input
        return T, R

    points = [0.0] + sorted(float(p) for p in np.atleast_1d(breakpoints if breakpoints is not None else [])
                            if 0.0 < p < length) + [float(length)]
    seg = np.diff(points)
    base_counts = [max(1, int(np.ceil(initial_slices * s_ / length))) for s_ in seg]
    # Minimum number of slices: the fourth-order (asymptotic) error behaviour, on which the error estimate relies,
    # only sets in when the relative phase accumulated over one slice by every pair of coupled modes is small,
    # |beta_p - beta_q| h <= max_phase_per_slice (forward-forward coupling; with error_control='full' also
    # (beta_p + beta_q) h for the forward-backward coupling that produces the reflections).
    # Only pairs that can exchange a relevant amount of power are considered: a pair with phase mismatch dbeta and
    # coupling rate kappa_max |K_pq| exchanges an amplitude of order kappa_max |K_pq| / dbeta (a power of order its
    # square), so pairs for which this is below atol (amplitude, error_control='full') or whose square is below atol
    # (power, error_control='input') cannot produce errors above atol (this excludes e.g. strongly mismatched
    # near-cutoff modes). At most an eighth of max_slices is used, to leave room for the error estimate.
    br = beta.real
    zz = np.linspace(0.0, length, 513)
    kmax = float(np.max(np.abs(_curvature_at(curvature, zz))))
    def max_relevant(phase, Kblock):
        rate = kmax * np.abs(Kblock)
        size = rate / np.maximum(phase, 1e-300)
        rel = (rate > 0) & ((size if error_control == 'full' else size ** 2) >= atol)
        return float(np.max(phase[rel])) if np.any(rel) else 0.0
    dphase = max_relevant(np.abs(br[:, None] - br[None, :]), K[:N, :N])
    if error_control == 'full':
        dphase = max(dphase, max_relevant(br[:, None] + br[None, :], K[:N, N:]))
    n_min = int(np.ceil(dphase * length / max_phase_per_slice)) if max_phase_per_slice else 0
    level = 0
    while sum(base_counts) * 2 ** level < min(n_min, max_slices // 8) and sum(base_counts) * 2 ** (level + 1) <= max_slices:
        level += 1
    edges = _slice_edges(length, base_counts, points, level)
    n = len(edges) - 1
    exps = _slice_exponentials(D, K, curvature, edges)
    phi = _product(exps)
    T, R = _scattering_from_transfer(phi, N)
    status, err, err_R, diff_prev, ratio_prev, small_prev = 1, np.inf, np.inf, None, None, False
    history = []
    while 2 * n <= max_slices:
        T_prev, R_prev = measure(T, R)
        level += 1
        edges = _slice_edges(length, base_counts, points, level)
        n = len(edges) - 1
        exps = _slice_exponentials(D, K, curvature, edges)
        phi = _product(exps)
        T, R = _scattering_from_transfer(phi, N)
        Tm, Rm = measure(T, R)
        diff_T = np.max(np.abs(Tm - T_prev))
        diff = max(diff_T, np.max(np.abs(Rm - R_prev))) if error_control == 'full' else diff_T
        err = diff / 15.0
        err_R = np.max(np.abs(Rm - R_prev)) / 15.0     # reflection amplitudes (reported, not controlled)
        ratio = np.inf if diff == 0 else (diff_prev / diff if diff_prev is not None else None)
        asymptotic = diff == 0 or (ratio is not None and ratio >= 8.0 and ratio_prev is not None
                                   and ratio_prev >= 8.0)
        ratio_prev = ratio
        history.append((n, err))
        if verbose:
            print(f"  {n:5d} slices: error estimate {err:.2e}" + ("" if asymptotic else "  (pre-asymptotic)"))
        small = err <= atol * 1e-3
        if err <= atol and (asymptotic or (small and small_prev)):
            status = 0
            break
        small_prev = small
        diff_prev = diff
    if status != 0:
        warnings.warn(f"transfer-matrix solver: tolerance {atol:g} not reached with {n} slices "
                      f"(error estimate {err:.1e}) at {f/1e9:.3f} GHz. If the curvature is discontinuous inside "
                      f"the bend, pass the positions of the discontinuities with breakpoints=[...]; otherwise "
                      f"increase max_slices.", RuntimeWarning)

    # amplitude profile for the input mode
    y0 = np.zeros(2 * N, dtype=complex)
    y0[j] = 1.0
    y0[N:] = R[:, j]
    if not profile:
        x = np.array([0.0, length])
        yL = phi @ y0
        y = np.stack([y0, yL], axis=1)
    else:
        if z_eval is None:
            x, seg = edges, exps
        else:
            z_eval = np.asarray(z_eval, dtype=float)
            if np.any(z_eval < 0) or np.any(z_eval > length):
                raise ValueError("z_eval must lie within [0, length]")
            x = np.unique(np.concatenate([edges, z_eval]))
            seg = _slice_exponentials(D, K, curvature, x)
        y = np.empty((2 * N, len(x)), dtype=complex)
        y[:, 0] = y0
        for i, E in enumerate(seg):
            y[:, i + 1] = E @ y[:, i]
        if z_eval is not None:
            idx = np.searchsorted(x, z_eval)
            x, y = z_eval, y[:, idx]

    message = ("The algorithm converged to the desired accuracy." if status == 0 else
               "The maximum number of slices was reached before the desired accuracy.")
    result = TransferMatrixResult(modes=modes, beta=beta, T=T, R=R, transfer_matrix=phi, x=x, y=y,
                                  input_index=j, n_slices=n, error_estimate=err, reflection_error_estimate=err_R,
                                  error_control=error_control, history=history, status=status,
                                  success=status == 0, message=message)
    return modes, result


def transmission_matrix(f, a, b, n, length, curvature, **kwargs):
    """Full transmission and reflection matrices of a bend for the mode family with vertical index n.

    Convenience wrapper around :func:`solve_transfer_matrix` (the input mode is the lowest mode of the
    family; T and R do not depend on it). Returns (modes, T, R, beta).
    """
    modes = select_coupled_modes(f, a, b, n=n, coupled_modes=kwargs.pop('coupled_modes', None),
                                 maximum_modes=kwargs.pop('maximum_modes', None))
    kwargs.setdefault('profile', False)
    kwargs.setdefault('error_control', 'full')
    first = modes[0]
    mode_in = (first[0][:2], int(first[1]), int(first[2]))   # e.g. ('TE', 1, 0)
    modes, res = solve_transfer_matrix(f, a, b, mode_in, length, curvature, coupled_modes=modes, **kwargs)
    return modes, res.T, res.R, res.beta


def straight_section(beta, length):
    """Transmission matrix of a straight waveguide section: diag(exp(-i beta L))."""
    return np.diag(np.exp(-1j * np.asarray(beta) * length))
