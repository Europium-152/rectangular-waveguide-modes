"""Independent check of the first-order loss matrix against exact modes of impedance-walled guides (T1b).

Run from the repository root:   PYTHONPATH=src python3 docs/theory/T1b/check_exact_modes.py

With impedance walls on ONE pair of opposite walls, the guide separates: its exact modes are the LS modes with
respect to the wall normal, whose transverse wavenumber solves a Robin problem (t1b_coefficients.lsu_dispersion).
To first order in Z, the propagation constants of the coupled-mode system dy/dz = (D - Lambda_full) y must agree
with the exact ones, with an error O(Z^2). For the degenerate pairs (TE_mn, TM_mn) this requires the TE-TM
coefficients with the right sign: degenerate perturbation theory diagonalizes the 2x2 block of Lambda^{++}, and its
eigenvectors must be the exact LS combinations.

  (a) walls u = -+a/2 lossy (Z_u-, Z_u+ different), v-walls PEC: exact LSE^u / LSM^u.
  (b) walls v = -+b/2 lossy, u-walls PEC: exact LSE^v / LSM^v (same dispersion function with a -> b).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1b_coefficients as T  # noqa: E402

a, b, f = 22.86e-3, 10.16e-3, 60e9
k = T.wavenumber(f)


def first_order_gammas(modes, Z):
    """Eigenvalues of the forward block -i beta - Lambda^{++}, returned as gamma = i * eigenvalue (so that the
    mode varies as exp(-i gamma z)), and the eigenvectors."""
    beta = np.array([T.beta(p, a, b, f) for p in modes])
    L = np.array([[T.lambda_pp(p, q, a, b, f, Z) for q in modes] for p in modes])
    w, V = np.linalg.eig(-1j * np.diag(beta) - L)
    return 1j * w, V


def family_u(n, M):
    out = [("TE", m, n) for m in range(0, M + 1) if m + n > 0]
    out += [("TM", m, n) for m in range(1, M + 1) if n > 0]
    return out


def family_v(m, Nmax):
    out = [("TE", m, n) for n in range(0, Nmax + 1) if m + n > 0]
    out += [("TM", m, n) for n in range(1, Nmax + 1) if m > 0]
    return out


def run_case(label, walls, side, fam_index, scale):
    Zm0, Zp0 = (0.7 + 0.9j), (1.3 + 0.6j)          # two different complex wall impedances (ohm)
    rows = []
    for s in scale:
        Zm, Zp = s * Zm0, s * Zp0
        if side == "u":
            Z = {"u-": Zm, "u+": Zp}
            modes = family_u(fam_index, 8)
            L, kt_fixed = a, fam_index * np.pi / b
        else:
            Z = {"v-": Zm, "v+": Zp}
            modes = family_v(fam_index, 8)
            L, kt_fixed = b, fam_index * np.pi / a
        g1, V = first_order_gammas(modes, Z)
        errs = []
        # exact: LSE (Dirichlet limit, index >= 1) and LSM (Neumann limit, index >= 0, needs the other index >= 1)
        exact = []
        idx_e = range(1, 6)
        rts = T.lsu_kx_roots("LSE", list(idx_e), L, k, Zm, Zp)
        exact += [("LSE", i, r) for i, r in zip(idx_e, rts)]
        if fam_index >= 1:
            idx_m = range(0, 6)
            rts = T.lsu_kx_roots("LSM", list(idx_m), L, k, Zm, Zp)
            exact += [("LSM", i, r) for i, r in zip(idx_m, rts)]
        for kind, i, kx in exact:
            g2 = k * k - kx * kx - kt_fixed ** 2
            ge = np.sqrt(g2 + 0j)
            ge = -ge if ge.imag > 0 else ge
            if abs(ge.real) < 0.05 * k:      # skip modes close to cutoff (first order is not uniform there)
                continue
            j = np.argmin(abs(g1 - ge))
            beta0 = T.beta(("TE", i, fam_index) if side == "u" else ("TE", fam_index, i), a, b, f)
            errs.append((kind, i, abs(g1[j] - ge), abs(ge - beta0)))
        rows.append((s, errs))
    print(f"{label}")
    print("   scale   max |gamma_1st - gamma_exact|   max |gamma_exact - beta_PEC|   ratio")
    prev = None
    for s, errs in rows:
        e = max(x[2] for x in errs)
        d = max(x[3] for x in errs)
        print(f"   {s:7.0e}   {e:.3e} 1/m                  {d:.3e} 1/m                  {e/d:.2e}")
        if prev is not None:
            order = np.log(prev / e) / np.log(10)
            print(f"            error falls by 10^{order:.2f} per decade of Z (2 expected)")
        prev = e
    return rows


def check_degenerate_vectors():
    """The 2x2 block of Lambda^{++} on (TE_mn, TM_mn) with u-walls lossy has the LSE^u / LSM^u combinations as
    eigenvectors; with v-walls lossy, the LSE^v / LSM^v combinations (rotation of CONVENTIONS.md section 7)."""
    print("Eigenvectors of the degenerate 2x2 blocks")
    worst = 0.0
    for (m, n) in ((1, 1), (2, 1), (1, 2), (3, 2)):
        pE, pM = ("TE", m, n), ("TM", m, n)
        NE, NM = T.norm(pE, a, b), T.norm(pM, a, b)
        km, kn = m * np.pi / a, n * np.pi / b
        sE, sM = np.sqrt(T.admittance(pE, a, b, f)), np.sqrt(T.admittance(pM, a, b, f))
        for side, Z in (("u", {"u-": 1.0, "u+": 1.0}), ("v", {"v-": 1.0, "v+": 1.0})):
            Lb = np.array([[T.lambda_pp(p, q, a, b, f, Z) for q in (pE, pM)] for p in (pE, pM)])
            w, V = np.linalg.eig(Lb)
            if side == "u":
                # LSE^u: E_u = 0 with V = sqrt(2/Y) a:  a_E (-NE kn)/sE + a_M (NM km)/sM = 0
                vec = np.array([NM * km / sM, NE * kn / sE])
            else:
                # LSE^v: E_v = 0:  a_E (NE km)/sE + a_M (NM kn)/sM = 0  (= CONVENTIONS.md section 7 rotation)
                vec = np.array([NM * kn / sM, -NE * km / sE])
            vec = vec / np.linalg.norm(vec)
            best = min(abs(1 - abs(np.vdot(vec, V[:, j]) / np.linalg.norm(V[:, j]))) for j in range(2))
            worst = max(worst, best)
            print(f"   TE/TM{m},{n}, {side}-walls lossy: 1 - |<LSE, eigenvector>| = {best:.1e};"
                  f"  Lambda^{{++}}_TE,TM / Lambda^{{++}}_TE,TE = {Lb[0,1]/Lb[0,0]:+.4f}")
    # the LS^v rotation of CONVENTIONS.md: LSE = cos(phi) TE - sin(phi) TM, tan(phi) = m b k / (n a beta)
    return worst


if __name__ == "__main__":
    sc = [1e-2, 1e-1, 1e0]
    ru = run_case("(a) u-walls lossy (Z_u- = 0.7+0.9i, Z_u+ = 1.3+0.6i, times scale), family n = 1, WR-90, 60 GHz",
                  None, "u", 1, sc)
    run_case("    same, family n = 0 (TE_m0 = LSE^u, Butterweck's case)", None, "u", 0, sc)
    rv = run_case("(b) v-walls lossy (same impedances), family m = 1", None, "v", 1, sc)
    run_case("    same, family m = 2", None, "v", 2, sc)
    worst = check_degenerate_vectors()
    assert worst < 1e-10
    print("all checks passed")
