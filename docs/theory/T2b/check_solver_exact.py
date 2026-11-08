"""Check of the full solution method (box B5) against exact wedge fields, through cutoff crossings (card T2b).

A section s1 <= s <= s2 of a wedge guide is solved with t2b_solver.solve (V-I Magnus slices, real-reference
pseudo-waves, Redheffer cascade). The exact field gives the modal amplitudes a+-(s) at both ports by quadrature;
the GSM must map the exact incident amplitudes [a+(s1); a-(s2)] onto the exact scattered ones [a-(s1); a+(s2)].
The exact fields include J_nu standing waves: a mode incident from the wide end, totally reflected at its cutoff
cross-section inside the section, so modes cross cutoff inside the taper. The error is reported for the modes of
the lowest indices (propagating and evanescent at the ports) as the truncation M grows.
Run: PYTHONPATH=src python3 docs/theory/T2b/check_solver_exact.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from wrmodes.modes import Mode
from t2b_formulas import beta_Y
from t2b_solver import solve, full
from exact_wedge import WedgeField
from check_exact_residual import Frame, geometry, amplitudes, family_basis, column_basis, FD


def make_geom(case, s1):
    frame = case["frame"]
    h = 1e-4

    def geom(z):
        s = s1 + z
        G = np.array([geometry(frame, s + j * h, case) for j in range(-4, 5)])
        dG = FD @ G / h
        a, b, u0, v0 = G[4]
        return dict(a=a, b=b, u0=u0, da=dG[0], db=dG[1], du0=dG[2], dv0=dG[3], kappa=frame.kappa)
    return geom


def modal(case, s, modes, f):
    V, I = amplitudes(case["field"], case["frame"], s, case, modes)
    a, b, u0, v0 = geometry(case["frame"], s, case)
    _, Y = beta_Y(modes, f, a, b)
    sY = np.sqrt(Y)
    S, D = V * sY / np.sqrt(2), I / (np.sqrt(2) * sY)
    return (S + D) / 2, (S - D) / 2


def run(case, s1, s2, f, Ms, ncheck=5, growth=1.0, m_tail=0):
    print("\n" + case["name"] + f"  (section {s1} .. {s2} m), tail correction up to index {m_tail}")
    for M in Ms:
        modes = case["basis"](M)
        geom = make_geom(case, s1)
        S = full(solve(modes, f, geom, s2 - s1, growth=growth, m_tail=m_tail))
        p1, m1 = modal(case, s1, modes, f)
        p2, m2 = modal(case, s2, modes, f)
        inc = np.concatenate([p1, m2])
        out = np.concatenate([m1, p2])
        pred = S @ inc
        N = len(modes)
        idx = np.r_[0:ncheck, N:N + ncheck]
        err = np.max(np.abs(pred[idx] - out[idx])) / np.max(np.abs(inc))
        recip = np.max(np.abs(S - S.T)) / np.max(np.abs(S))
        g1, g2 = geom(0), geom(s2 - s1)
        prop1 = np.real(beta_Y(modes, f, g1["a"], g1["b"])[0]) > 0
        prop2 = np.real(beta_Y(modes, f, g2["a"], g2["b"])[0]) > 0
        pidx = np.r_[np.where(prop1)[0], N + np.where(prop2)[0]]
        Sp = S[np.ix_(pidx, pidx)]
        unit = np.max(np.abs(Sp.conj().T @ Sp - np.eye(len(pidx))))
        print(f"  M={M:3d} N={N:3d}: max|S a_inc - a_out| / max|a_inc| = {err:.2e}  (first {ncheck} modes each port); "
              f"|S-S^T| = {recip:.1e}, |Sp^H Sp - I| = {unit:.1e}")
    print("  modes propagating at s1: " + ", ".join(p.name for p, q in zip(modes, prop1) if q))
    print("  modes propagating at s2: " + ", ".join(p.name for p, q in zip(modes, prop2) if q))
    return err


def cases(f):
    X, Yv, Zv = np.eye(3)
    b0, a0 = 0.0102, 0.02286
    out = []
    ph = 0.12
    apex = np.array([0.0, 0.0, -0.10])
    fld = WedgeField(f, apex, -ph, ph, b0, Yv, X, [(1.0, "LSM", 1, 0, 2), (1.0, "LSM", 3, 0, 0), (0.5, "LSM", 5, 0, 0)])
    out.append((dict(name="H-plane symmetric wedge, n=0: TE10 out + TE30, TE50 reflected at cutoff", frame=Frame(),
                     field=fld, along="u", apex=apex, phis=(-ph, ph), width=b0, basis=family_basis(0)), -0.06, 0.0))
    apex = np.array([-0.004, 0.0, -0.12])
    phs = (-0.05, 0.17)
    fld = WedgeField(f, apex, *phs, b0, Yv, X, [(1.0, "LSM", 1, 1, 2), (0.7, "LSE", 3, 1, 0), (0.5, "LSM", 3, 1, 0)])
    out.append((dict(name="H-plane asymmetric wedge, n=1: LSM1 out + LSE3, LSM3 reflected at cutoff", frame=Frame(),
                     field=fld, along="u", apex=apex, phis=phs, width=b0, basis=family_basis(1)), -0.06, 0.0))
    apex = np.array([0.0, 0.003, -0.08])
    phs = (-0.04, 0.10)
    fld = WedgeField(f, apex, *phs, a0, X, Yv, [(1.0, "LSE", 0, 1, 2), (0.6, "LSE", 2, 1, 0), (0.6, "LSM", 2, 1, 0)])
    out.append((dict(name="E-plane asymmetric wedge, m=1: TE10 out + LSE/LSM 2 reflected at cutoff", frame=Frame(),
                     field=fld, along="v", apex=apex, phis=phs, width=a0, basis=column_basis(1)), -0.04, 0.03))
    apex = np.array([-0.004, 0.0, -0.12])
    phs = (-0.05, 0.17)
    fld = WedgeField(f, apex, *phs, b0, Yv, X, [(1.0, "LSM", 1, 1, 2), (0.7, "LSE", 3, 1, 0)])
    out.append((dict(name="curved centre line (kappa = 4 /m) on the asymmetric wedge, n=1", frame=Frame(kappa=4.0),
                     field=fld, along="u", apex=apex, phis=phs, width=b0, basis=family_basis(1)), -0.04, 0.0))
    return out


if __name__ == "__main__":
    f = 30e9
    cs = cases(f)
    which = [int(x) for x in sys.argv[1:]] or range(len(cs))
    for i in which:
        case, s1, s2 = cs[i]
        Ms = (6, 12) if i < 3 else (4, 8)
        run(case, s1, s2, f, Ms, m_tail=0, growth=2.0)
        run(case, s1, s2, f, Ms, m_tail=1000, growth=2.0)
