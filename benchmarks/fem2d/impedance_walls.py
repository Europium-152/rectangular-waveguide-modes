"""Impedance (Leontovich) walls of the 2D solver against T1 and WP2 (WP3b).

1. Per-mode attenuation of a straight guide with impedance walls at u = +-a/2 (H plane: the narrow walls of
   WR-90, aluminium sigma = 3.5e7 S/m) against the exact roots of T1 box B6 (`wrmodes.filters.impedance_wall_modes`)
   and the first-order `wrmodes.loss` (T1 boxes B2-B3, for those walls only). The FEM attenuation is extracted
   from the GSM of a straight outline of length L: the transfer matrix J^-1 D J of a uniform section between two
   identical junctions has the eigenvalues exp(-+i gamma L), so alpha = -ln|lambda| / L is free of the junction to
   the extent that the discrete junctions are inverse to each other; their discretization error enters as c / L,
   and is removed by extrapolating from L1 and L2 (alpha = (L2 alpha_2 - L1 alpha_1) / (L2 - L1)).
2. Butterweck's lossy-wall filter (75 GHz, 250 ohm on both narrow walls, 15 cm, WP2 report): FEM against the exact
   `LossyWallFilter` (WP2) and Butterweck's 0.6 / 2.5 / 5.4 dB (Philips Tech. Rev. 29, pp. 94-95).
3. Lossy straight sections, all propagating families, against `LossyWallFilter` (same physical problem: lossy
   section between perfect guides), complex GSM entries.

Usage: python benchmarks/fem2d/impedance_walls.py [--processes 2]  -> data/impedance_walls.npz
"""
import argparse
import time
import numpy as np
import scipy.constants as cst

from _lit import meta, save, pmap

W, H = 22.86e-3, 10.16e-3
SIGMA = 3.5e7


def gsm_straight(args):
    """Propagating GSM of family n of a lossy straight outline (H plane)."""
    f, L, Z, n, dens = args
    from wrmodes.fem2d import Coplanar2D, straight_outline
    from wrmodes import Guide
    B = Guide(W, H).basis(f, 1.0, plane="H")
    Bn = B.subset(B.family(n)) if n is not None else B
    c = Coplanar2D(straight_outline(W, L), H, wall_impedance=Z, mesh_density=dens)
    t = time.time()
    G = c.gsm(f, Bn, Bn)
    return G.S11, G.S12, G.S21, G.S22, [str(m) for m in Bn.modes], time.time() - t, G.fem_info["ndof"]


def eig_gamma(S11, S12, S21, S22, L):
    iS12 = np.linalg.inv(S12)
    Phi = np.block([[S21 - S22 @ iS12 @ S11, S22 @ iS12], [-iS12 @ S11, iS12]])
    lam = np.linalg.eigvals(Phi)
    return lam[np.abs(lam) < 1]


def exact_gammas(f, n, Z, count=12):
    from wrmodes.filters import impedance_wall_modes
    k = 2 * np.pi * f / cst.c
    ky = n * np.pi / H
    out = []
    for kind in (["LSE"] if n == 0 else ["LSE", "LSM"]):
        s = impedance_wall_modes(f, W, Z, Z, kind, count)
        g = np.sqrt(k * k - ky * ky - s + 0j)
        g = np.where(g.imag > 0, -g, g)
        out += [(kind, x) for x in g if x.real > 3 * abs(x.imag)]
    return out


def first_order_alphas(f, n, Z):
    """Eigen-attenuations of Lambda^+ (u walls) per degenerate TE/TM pair, family n (wrmodes.loss)."""
    from wrmodes.loss import loss_matrices
    from wrmodes import Guide
    B = Guide(W, H).basis(f, 1.0, plane="H")
    Bn = B.subset(B.family(n))
    Lp, _ = loss_matrices(Bn, f, {"u+": Z, "u-": Z})
    beta = Bn.beta(f).real
    out = []
    done = set()
    for i in range(len(Bn)):
        if i in done:
            continue
        grp = [j for j in range(len(Bn)) if abs(beta[j] - beta[i]) < 1e-9 * abs(beta[i])]
        done.update(grp)
        ev = np.linalg.eigvals(Lp[np.ix_(grp, grp)])
        out += [(beta[i], e.real) for e in ev]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--density", type=float, default=12.0)
    args = ap.parse_args()
    arrays, summ = {}, {}
    # ---------------------------------------------------------------- 1. attenuation per mode
    L1, L2 = 0.2, 0.6
    print("1. Attenuation, aluminium narrow walls (u = +-a/2), WR-90; FEM extrapolated from L = %g and %g m, "
          "mesh_density %g" % (L1, L2, args.density))
    print("| f (GHz) | n | beta (1/m) | alpha exact B6 (Np/m) | FEM L1 rel | FEM L2 rel | FEM extrap. rel | "
          "first order rel | FEM extrap. vs first order |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = []
    jobs = [(10e9, 0), (30e9, 0), (30e9, 1)]
    res = pmap(gsm_straight, [(f, L, complex(np.sqrt(np.pi * f * cst.mu_0 / SIGMA) * (1 + 1j)), n, args.density)
                              for f, n in jobs for L in (L1, L2)], args.processes)
    for q, (f, n) in enumerate(jobs):
        Z = complex(np.sqrt(np.pi * f * cst.mu_0 / SIGMA) * (1 + 1j))
        lams = [eig_gamma(*res[2 * q + r][:4], L) for r, L in enumerate((L1, L2))]
        fo = first_order_alphas(f, n, Z)
        for kind, g in exact_gammas(f, n, Z):
            al = []
            for lam, L in zip(lams, (L1, L2)):
                j = np.argmin(np.abs(lam - np.exp(-1j * g * L)))
                al.append(-np.log(abs(lam[j])) / L)
            ext = (L2 * al[1] - L1 * al[0]) / (L2 - L1)
            ax = -g.imag
            jf = np.argmin([abs(b - g.real) + 1e3 * abs(a - ax) for b, a in fo])
            afo = fo[jf][1]
            rows.append((f, n, g.real, ax, al[0] / ax - 1, al[1] / ax - 1, ext / ax - 1, afo / ax - 1, ext / afo - 1))
            print("| %.0f | %d | %.3f | %.6e | %+.1e | %+.1e | %+.1e | %+.1e | %+.1e |" % (
                f / 1e9, n, g.real, ax, *rows[-1][4:]))
    rows = np.array(rows)
    arrays["alpha_table"] = rows
    summ["alpha"] = dict(max_abs_rel_extrapolated=float(np.max(np.abs(rows[:, 6]))),
                         max_abs_rel_first_order_vs_exact=float(np.max(np.abs(rows[:, 7]))))
    print("max |FEM extrapolated / exact - 1| = %.2e; max |first order / exact - 1| = %.2e" % (
        summ["alpha"]["max_abs_rel_extrapolated"], summ["alpha"]["max_abs_rel_first_order_vs_exact"]))
    # ---------------------------------------------------------------- 2. Butterweck
    from wrmodes import Guide
    from wrmodes.filters import LossyWallFilter
    print("\n2. Butterweck lossy-wall filter: 75 GHz, 250 ohm on both narrow walls, 15 cm (dB)")
    f = 75e9
    B = Guide(W, H).basis(f, 1.0, plane="H")
    out = {}
    for dens in (6.0, 10.0):
        r0 = gsm_straight((f, 0.15, 250.0, 0, dens))
        r1 = gsm_straight((f, 0.15, 250.0, 1, dens))
        out[dens] = (r0, r1)
    B0, B1 = B.subset(B.family(0)), B.subset(B.family(1))
    ref = {M: (LossyWallFilter(0.15, 250.0, guide=Guide(W, H), modes_per_family=M).gsm(f, B0, B0),
               LossyWallFilter(0.15, 250.0, guide=Guide(W, H), modes_per_family=M).gsm(f, B1, B1)) for M in (80, 160)}
    names = [("TE1,0", 0), ("TE2,0", 0), ("TE3,0", 0), ("TE0,1", 1), ("TE1,1", 1), ("TM1,1", 1)]
    butter = {"TE1,0": 0.6, "TE2,0": 2.5, "TE3,0": 5.4}
    print("| mode | FEM (6/lambda) | FEM (10/lambda) | LossyWallFilter M=80 | M=160 | FEM vs M=160 | Butterweck | FEM vs Butterweck |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    bt = []
    for nm, fam in names:
        vals = []
        for dens in (6.0, 10.0):
            r = out[dens][fam]
            i = r[4].index(nm)
            vals.append(-20 * np.log10(abs(r[2][i, i])))
        Bf = (B0, B1)[fam]
        i = [str(m) for m in Bf.modes].index(nm)
        rv = [-20 * np.log10(abs(ref[M][fam].S21[i, i])) for M in (80, 160)]
        bw = butter.get(nm, np.nan)
        bt.append([vals[0], vals[1], rv[0], rv[1], bw])
        print("| %s | %.4f | %.4f | %.4f | %.4f | %+.2f %% | %s | %s |" % (
            nm, vals[0], vals[1], rv[0], rv[1], 100 * (vals[1] / rv[1] - 1), "%.1f" % bw if bw == bw else "-",
            "%+.1f %%" % (100 * (vals[1] / bw - 1)) if bw == bw else "-"))
    r = out[10.0][0]
    i1, i3 = r[4].index("TE1,0"), r[4].index("TE3,0")
    conv = 20 * np.log10(abs(r[2][i3, i1]))
    i1r, i3r = [str(m) for m in B0.modes].index("TE1,0"), [str(m) for m in B0.modes].index("TE3,0")
    print("TE1,0 -> TE3,0: FEM %.3f dB, LossyWallFilter M=160 %.3f dB (Butterweck: -36 ... -24.5 dB)" % (
        conv, 20 * np.log10(abs(ref[160][0].S21[i3r, i1r]))))
    arrays["butterweck"] = np.array(bt)
    summ["butterweck"] = dict(modes=[n for n, _ in names], fem6=[b[0] for b in bt], fem10=[b[1] for b in bt],
                              filter80=[b[2] for b in bt], filter160=[b[3] for b in bt], conversion_fem=conv)
    # ---------------------------------------------------------------- 3. complex GSM vs LossyWallFilter
    print("\n3. Lossy straight (5 cm) vs LossyWallFilter, every propagating family, max |dS| (complex)")
    print("| f (GHz) | Z (ohm) | modes | max \\|dS\\| (FEM 8/lambda) | reciprocity | largest singular value |")
    print("| --- | --- | --- | --- | --- | --- |")
    cmp_rows = []
    for f in (18e9, 54e9):
        for Z in (complex(np.sqrt(np.pi * f * cst.mu_0 / SIGMA) * (1 + 1j)), 2.5 + 0j, 250.0 + 0j):
            r = gsm_straight((f, 0.05, Z, None, 8.0))
            Bf = Guide(W, H).basis(f, 1.0, plane="H")
            Rf = LossyWallFilter(0.05, Z, guide=Guide(W, H)).gsm(f, Bf, Bf)
            S = np.block([[r[0], r[1]], [r[2], r[3]]])
            d = np.max(np.abs(S - Rf.S))
            rec = np.max(np.abs(S - S.T)) / np.max(np.abs(S))
            sv = np.linalg.svd(S, compute_uv=False).max()
            cmp_rows.append([f, Z.real, Z.imag, len(Bf), d, rec, sv])
            print("| %.0f | %.4g%+.4gi | %d | %.1e | %.1e | %.6f |" % (f / 1e9, Z.real, Z.imag, len(Bf), d, rec, sv))
    arrays["filter_compare"] = np.array(cmp_rows)
    m = meta("benchmarks/fem2d/impedance_walls.py", "Impedance walls of wrmodes.fem2d against T1 B6 roots, "
             "wrmodes.loss and LossyWallFilter", guide_w=W, guide_h=H, sigma=SIGMA, L1=L1, L2=L2,
             mesh_density=args.density, summary=summ,
             columns=dict(alpha_table="f, n, beta, alpha_exact, rel L1, rel L2, rel extrapolated, rel first order, "
                                      "extrapolated/first order - 1",
                          butterweck="FEM 6/lambda, FEM 10/lambda, filter M=80, filter M=160, Butterweck (dB), "
                                     "rows TE1,0 TE2,0 TE3,0 TE0,1 TE1,1 TM1,1",
                          filter_compare="f, Re Z, Im Z, modes, max |dS|, reciprocity, largest sv"))
    print("saved", save("impedance_walls.npz", m, **arrays))


if __name__ == "__main__":
    main()
