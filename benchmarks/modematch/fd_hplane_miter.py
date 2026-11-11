"""Independent check of MiterMM for the H-plane TE10 problem (WP4b): second-order finite differences.

The scalar problem psi = E_y, laplacian(psi) + k^2 psi = 0, psi = 0 on the walls, of the 90-degree miter (arm 1
x in [0, a], y < 0; arm 2 y in [0, a], x < 0; outer walls x = a, y = a; miter x + y = a + c, c = t a, i.e.
d/a = (1 + t)/sqrt(2)), on a square grid h = a / N with the 5-point stencil. The miter line passes through grid
nodes (c / h integer), so no boundary interpolation is needed. Each arm is closed a few rows from the corner by
the exact discrete radiation condition of the uniform discrete guide (all N - 1 discrete modes sin(m pi i / N),
outgoing root of rho + 1/rho = 2 - h^2 (k^2 - mu_m)). |S11| of TE10 is read at y = 0. It shares nothing with
MiterMM but the geometry; the error decays as h^p with p about 1.3-1.6 (corner singularities), and the result is
Richardson-extrapolated from N = 200 and 400.
Usage: python benchmarks/modematch/fd_hplane_miter.py  (writes data/fd_hplane_miter.npz; about 10 min)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def fd_s11(N, cnodes, kapa, arm_rows=3, modes_out=(1,)):
    """a = 1, h = 1/N; miter line i + j = N + cnodes (cnodes = c/h, 0..N). kapa = kappa * a."""
    h = 1.0 / N
    k2h2 = (kapa * h) ** 2
    s = N + cnodes
    # node set: arm1 i=1..N-1, j=-arm_rows..0 ; corner: i>=1? careful at the inner corner
    nodes = {}
    def inside(i, j):
        if j <= 0:   # arm 1 region (and its port plane j=0)
            return 1 <= i <= N - 1 and j >= -arm_rows
        if i <= 0:   # arm 2
            return 1 <= j <= N - 1 and i >= -arm_rows
        return i <= N - 1 and j <= N - 1 and i + j < s
    # also nodes on x=0 (i=0) with 1<=j<=N-1 and on y=0 (j=0) with 1<=i<=N-1 are interior (port planes)
    for i in range(-arm_rows, N):
        for j in range(-arm_rows, N):
            if inside(i, j):
                nodes[(i, j)] = len(nodes)
    n = len(nodes)
    rows, cols, vals = [], [], []
    for (i, j), p in nodes.items():
        rows.append(p); cols.append(p); vals.append(-4.0 + k2h2)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = nodes.get((i + di, j + dj))
            if q is not None:
                rows.append(p); cols.append(q); vals.append(1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=complex).tolil()
    # discrete modes on i = 1..N-1
    ii = np.arange(1, N)
    m = np.arange(1, N)
    V = np.sqrt(2.0 / N) * np.sin(np.outer(ii, m) * np.pi / N)          # columns orthonormal
    mu = 2 - 2 * np.cos(m * np.pi / N)
    B = 2 - (k2h2 - mu)
    disc = np.sqrt(B * B - 4 + 0j)
    r1, r2 = (B + disc) / 2, (B - disc) / 2
    # outgoing root (travels/decays towards the far end, j -> -inf): |rho| > 1, or Im rho > 0 on |rho| = 1
    rout = np.where(np.abs(r1) > 1 + 1e-12, r1, np.where(np.abs(r2) > 1 + 1e-12, r2, np.where(r1.imag > 0, r1, r2)))
    rin = 1 / rout
    j0 = -arm_rows
    # closure at the port rows: u_{j0-1} = sum_m v_m [(v_m . u_j0 - a_m rin^j0)/rout + a_m rin^(j0-1)]
    idx1 = np.array([nodes[(i, j0)] for i in ii])
    idx2 = np.array([nodes[(j0, i)] for i in ii])
    D = (V / rout[None, :]) @ V.T
    A = A.tocsr() + sp.csr_matrix((D.ravel(), (np.repeat(idx1, N - 1), np.tile(idx1, N - 1))), shape=(n, n)) \
        + sp.csr_matrix((D.ravel(), (np.repeat(idx2, N - 1), np.tile(idx2, N - 1))), shape=(n, n))
    out = []
    for mo in modes_out:
        alpha = np.zeros(N - 1, complex); alpha[mo - 1] = 1.0
        # incident in arm 1 (amplitude 1 at j = 0): u_inc_j = rin^j v
        src = V @ (alpha * (rin ** (j0 - 1) - rin ** j0 / rout))
        rhs = np.zeros(n, complex)
        rhs[idx1] = -src
        u = spla.spsolve(A.tocsc(), rhs)
        c = V.T @ u[idx1]
        omega = (c - alpha * rin ** j0) / rout ** j0      # amplitude at j = 0
        out.append(omega)
    return out


CASES = [(0.38, 1.5), (0.38, 1.636), (0.38, 1.69), (0.0, 16 / 9.488)]   # (t/a, f/fc)
NS = (50, 100, 200, 400)


def main():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from wrmodes.modematch._corner2d import miter_2d
    from _meta import meta, save
    vals = np.zeros((len(CASES), len(NS)))
    extrap, mm, order = np.zeros(len(CASES)), np.zeros(len(CASES)), np.zeros(len(CASES))
    for i, (t, x) in enumerate(CASES):
        for j, N in enumerate(NS):
            vals[i, j] = abs(fd_s11(N, int(round(t * N)), np.pi * x)[0][0])
        q = np.log((vals[i, 1] - vals[i, 2]) / (vals[i, 2] - vals[i, 3])) / np.log(2.0)
        order[i] = q if np.isfinite(q) and q > 0.5 else 4 / 3
        extrap[i] = vals[i, 3] + (vals[i, 3] - vals[i, 2]) / (2 ** order[i] - 1)
        S11, _ = miter_2d(1.0, t, "D", (np.pi * x) ** 2, 400, 348)
        mm[i] = abs(S11[0, 0])
        print(f"t/a={t} f/fc={x:.4f}: FD {' '.join(f'{v:.6f}' for v in vals[i])} -> {extrap[i]:.6f} (p {order[i]:.2f});"
              f" MM (M=400) {mm[i]:.6f}", flush=True)
    m = meta("benchmarks/modematch/fd_hplane_miter.py",
             "Finite-difference |S11| of TE10, H-plane 90-degree miter, vs MiterMM's 2D solver",
             cases=[{"t_over_a": t, "f_over_fc": x, "d_over_a": (1 + t) / np.sqrt(2)} for t, x in CASES], N=list(NS),
             layout="fd[case, N]; extrap = Richardson from N = 200, 400 with the fitted order; mm = MiterMM 2D (M=400)")
    save("fd_hplane_miter.npz", m, fd=vals, extrap=extrap, order=order, mm=mm)


if __name__ == "__main__":
    main()
