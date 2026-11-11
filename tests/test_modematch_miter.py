"""Tests of wrmodes.modematch.MiterMM (WP4b): the 2D segmentation core, the 3D GSM, the literature, and the
parity-reduced Step. Slow cases are marked `slow`. Report: docs/reports/WP4b.md."""
import os
import time

import numpy as np
import pytest
import scipy.constants as cst

from wrmodes import Guide, Line, Straight
from wrmodes.gsm import cascade
from wrmodes.modes import Mode, ModeBasis, ls_rotation, transverse_field
from wrmodes.modematch import MiterMM, Step
from wrmodes.modematch import _corner2d as c2

HERE = os.path.dirname(os.path.abspath(__file__))
LIT = os.path.join(HERE, "..", "benchmarks", "literature")
WR90 = Guide(22.86e-3, 10.16e-3)
SQ2 = np.sqrt(2.0)


def load_csv(name):
    rows = [r for r in open(os.path.join(LIT, name)) if not r.startswith("#")][1:]
    out = {}
    for r in rows:
        v = r.strip().split(",")
        out.setdefault(v[0], []).append([float(v[1]), float(v[2]), float(v[3])])
    return {k: np.array(v) for k, v in out.items()}


# ============================================================================================ 2D core
def test_basis1d_orthonormal_and_integrals():
    L = 0.37
    for kind in ("cos", "nd", "dn", "sin"):
        b = Basis1D = c2.Basis1D(kind, L, 9)
        G = c2.cc_int(b.funcs(), b.funcs(), L)
        assert np.allclose(G, np.eye(9), atol=1e-13)
    # cc_int and ce_int against Gauss-Legendre quadrature
    t, w = np.polynomial.legendre.leggauss(400)
    t, w = 0.5 * L * (t + 1), 0.5 * L * w
    f1, f2 = c2.Basis1D("nd", L, 7), c2.Basis1D("cos", L, 5)
    A, p, h = f1.funcs()
    F1 = A[:, None] * np.cos(p[:, None] * t + h[:, None])
    A, p, h = f2.reversed_funcs()
    F2 = A[:, None] * np.cos(p[:, None] * t + h[:, None])
    assert np.allclose(c2.cc_int(f1.funcs(), f2.reversed_funcs(), L), (F1 * w) @ F2.T, atol=1e-13)
    gam = np.array([0.0, 3.0, 40.0, 2j, 15j, 4 + 1e-9j])
    E = np.exp(-gam[None, :] * t[:, None])
    assert np.allclose(c2.ce_int(f1.funcs(), gam, L), (F1 * w) @ E, atol=1e-13)


@pytest.mark.parametrize("wall", ["D", "N"])
@pytest.mark.parametrize("kap", [3.0, 20.0, 45.0])
def test_rectangle_closed_forms_against_integrals(wall, kap):
    """Adjacent blocks: Green's-identity closed form = explicit integral; cells real for real kappa^2."""
    Lx, Ly = 0.6, 0.35
    kb = "nd" if wall == "D" else "cos"
    for sides in ({"B": c2.Basis1D(kb, Lx, 25), "L": c2.Basis1D("cos", Ly, 15), "T": c2.Basis1D(kb, Lx, 25), "R": wall},
                  {"B": c2.Basis1D(kb, Lx, 25), "L": c2.Basis1D(kb, Ly, 15), "T": wall, "R": wall},
                  {s: c2.Basis1D("cos", Lx if s in "BT" else Ly, 20) for s in "BTLR"}):
        c2.rect_ntd(Lx, Ly, kap * kap, sides, check=True)
        assert c2.rect_ntd.last_check < 1e-12
        assert c2.rect_ntd.last_imag < 1e-12


def test_mismatched_side_tail_correction():
    """The J-sum with the polygamma tail equals the converged sum (cos data on an 'nd' side)."""
    L, K, kap2, H = 0.7, 12, 15.0 ** 2, 0.4
    W = c2.Basis1D("cos", L, K)

    def part(J):
        X = c2.Basis1D("nd", L, J)
        Pi = c2.cc_int(W.funcs(), X.funcs(), L)
        g = c2._gamma(X.p, kap2)
        return ((Pi * (-1 / np.tanh(g * H) / g)) @ Pi.T).real

    ref = part(200000) + c2._tail_cos_on_nd(K, 200000, L, kap2)
    for J in (64, 256):
        assert np.max(np.abs(part(J) - ref)) > 5e-7                      # the tail matters
        assert np.max(np.abs(part(J) + c2._tail_cos_on_nd(K, J, L, kap2) - ref)) < 5e-8


def test_optical_mirror_triangle_closed_form():
    """T2 triangle (image square) = the closed form Z_kl = diag - (s1 + s2 (-1)^(k+l)) nu_k nu_l / (l_k^2 + l_l^2 - k^2)."""
    a, kap, K = 1.0, 17.3, 30
    k = np.arange(K)
    lam = k * np.pi / a
    nu = np.where(k == 0, 1.0, np.sqrt(2.0)) / np.sqrt(a)
    g = np.sqrt(lam ** 2 - kap ** 2 + 0j)
    for s1 in (1, -1):
        for s2 in (1, -1):
            diag = (-1 / np.tanh(g * a) / g - s1 * s2 * (-1.0) ** k / (g * np.sinh(g * a))).real
            Zc = np.diag(diag) - (s1 + s2 * (-1.0) ** (k[:, None] + k[None, :])) * np.outer(nu, nu) \
                / (lam[:, None] ** 2 + lam[None, :] ** 2 - kap ** 2)
            Z = c2.triangle_t2(c2.Basis1D("cos", a, K), s1, s2, kap * kap)
            assert np.max(np.abs(Z - Zc)) < 1e-13 * np.max(np.abs(Zc))


@pytest.mark.parametrize("wall", ["D", "N"])
@pytest.mark.parametrize("c", [0.0, 0.05, 0.3, 0.7, 0.97, 1.0])
def test_full_network_equals_bisector_halves(wall, c):
    """The two-port network (port 2 in its own frame) equals the even/odd halves: S11 = S22, S12 = S21."""
    for kap in (5.0, 17.3, 31.0):
        S11, S21 = c2.miter_2d(1.0, c, wall, kap * kap, 50, 40)
        F11, F21, F12, F22 = c2.miter_2d(1.0, c, wall, kap * kap, 50, 40, symmetric=False)
        for X, Y in ((F11, S11), (F22, S11), (F21, S21), (F12, S21)):
            assert np.max(np.abs(X - Y)) < 1e-12


@pytest.mark.parametrize("wall", ["D", "N"])
def test_2d_unitary_and_symmetric_at_every_truncation(wall):
    for c in (0.0, 0.25, 1.0):
        for M in (6, 13, 40):
            for kap in (4.0, 12.0):
                S11, S21 = c2.miter_2d(1.0, c, wall, kap * kap, M, max(3, M // 2))
                m0 = 1 if wall == "D" else 0
                p = np.arange(m0, m0 + M) * np.pi < kap
                Sp = np.block([[S11[np.ix_(p, p)], S21[np.ix_(p, p)]], [S21[np.ix_(p, p)], S11[np.ix_(p, p)]]])
                assert np.max(np.abs(Sp.conj().T @ Sp - np.eye(len(Sp)))) < 1e-12
                assert np.max(np.abs(S11 - S11.T)) < 1e-12 and np.max(np.abs(S21 - S21.T)) < 1e-12


@pytest.mark.parametrize("wall,c,kap", [("D", 0.3, np.pi / 0.3), ("N", 0.3, np.pi / 0.3),
                                         ("N", 0.6, np.pi * np.sqrt(5) / 0.4), ("D", 1.0, 1.5 * np.pi)])
def test_cell_resonances_are_harmless(wall, c, kap):
    """At (and near) a cell eigenvalue the result is continuous (resonance guard, removable gamma = 0)."""
    ds = np.array([-4e-3, -2e-3, -1e-3, 1e-3, 2e-3, 4e-3])
    vals = []
    for d in ds:
        S11, S21 = c2.miter_2d(1.0, c, wall, (kap * (1 + d)) ** 2, 30, 24)
        vals.append(np.concatenate([S11[:3, :3].ravel(), S21[:3, :3].ravel()]))
    coef = np.polyfit(ds, np.array(vals), 4)
    for d in (0.0, 1e-12, 1e-10, 1e-9):
        S11, S21 = c2.miter_2d(1.0, c, wall, (kap * (1 + d)) ** 2, 30, 24)
        v = np.concatenate([S11[:3, :3].ravel(), S21[:3, :3].ravel()])
        ref = np.array([np.polyval(coef[:, i], d) for i in range(v.size)])
        assert np.max(np.abs(v - ref)) < 1e-6


# ============================================================================================ LS modes
def test_ls_amplitudes_are_the_2d_amplitudes():
    """Unit LSM amplitude: H_v = 0, E_v = C sqrt(2/a) sin(m pi u'/a) cos(n pi v'/b) / sqrt(beta); unit LSE amplitude:
    E_v = 0, H_v = C' sqrt(eps_m/a) cos(m pi u'/a) sin(n pi v'/b) / sqrt(beta), with C, C' independent of m
    (evanescent modes included). This is what makes S_LS equal to the 2D GSM of psi = E_v (LSM) or psi = H_v (LSE)."""
    a, b, f = 22.86e-3, 10.16e-3, 40e9
    k = 2 * np.pi * f / cst.c
    B = ModeBasis.below(4 * f, a, b, frame="H")
    B = B.subset([i for i, m in enumerate(B.modes) if k * k - (m.n * np.pi / b) ** 2 > 0])
    Q, labels = ls_rotation(B, f)
    Y, beta = B.admittance(f), B.beta(f)
    U, V = np.meshgrid(np.array([-0.31, 0.07, 0.22]) * a, np.array([-0.4, 0.13, 0.35]) * b, indexing="ij")
    up, vp = U + a / 2, V + b / 2
    fields = [transverse_field(m, a, b, U, V) for m in B.modes]
    consts = {}
    for i, (typ, m, n) in enumerate(labels):
        idx = np.nonzero(Q[i])[0]
        Ev = sum(Q[i, p] * np.sqrt(2 / Y[p]) * fields[p][1] for p in idx)
        Hv = sum(Q[i, p] * np.sqrt(2 * Y[p]) * fields[p][0] for p in idx)     # (z x e)_v = e_u
        bt = beta[idx[0]]
        if typ == "LSM":
            shape = np.sqrt(2 / a) * np.sin(m * np.pi * up / a) * np.cos(n * np.pi * vp / b) / np.sqrt(bt)
            C, other = Ev / shape, Hv
        else:
            shape = np.sqrt((1 if m == 0 else 2) / a) * np.cos(m * np.pi * up / a) * np.sin(n * np.pi * vp / b) \
                / np.sqrt(bt)
            C, other = Hv / shape, Ev
        assert np.max(np.abs(other)) < 1e-10 * np.max(np.abs(C))
        consts.setdefault((typ, n), []).append(C.ravel())
    assert len(consts) == 5
    for key, Cs in consts.items():
        Cs = np.array(Cs)
        assert np.max(np.abs(Cs - Cs[0, 0])) < 1e-12 * abs(Cs[0, 0]), key


# ============================================================================================ MiterMM
MITERS = [None, 0.8, 0.87, 0.95, 1.0, SQ2]


@pytest.mark.parametrize("plane", ["H", "E"])
@pytest.mark.parametrize("chamfer", MITERS)
def test_gsm_properties(plane, chamfer):
    """Reciprocity and unitarity (1e-10), S11 = S22 from the full two-port network (CONVENTIONS.md 6)."""
    m = MiterMM(chamfer, plane, WR90, n_modes=60)
    for f in (9e9, 21e9, 38e9):
        b, _ = m.default_basis(f, cutoff_factor=2.0)       # evanescent port modes too
        g = m.gsm(f, b, b)
        assert g.reciprocity_defect() <= 1e-10 * np.max(np.abs(g.S))
        assert g.unitarity_defect() <= 1e-10
        gf = m.gsm(f, b, b, symmetry=False)
        assert np.max(np.abs(gf.S11 - gf.S22)) < 1e-11
        assert np.max(np.abs(gf.S12 - gf.S21)) < 1e-11
        assert np.max(np.abs(gf.S - g.S)) < 1e-11
        assert gf.reciprocity_defect() <= 1e-10 * np.max(np.abs(gf.S))
        # families are not coupled
        n1 = np.array([md.n for md in b.modes])
        assert np.max(np.abs(g.S21[n1[:, None] != n1[None, :]]), initial=0.0) == 0.0


def _marcuvitz_pi(Ba, Bs):
    """S11, S21 of the symmetric pi network (shunt jBa at both planes, series admittance Bs), normalized."""
    ge = (1 - Ba) / (1 + Ba)
    yo = Ba + 2 * Bs
    go = (1 - yo) / (1 + yo)
    return (ge + go) / 2, (ge - go) / 2


def _marcuvitz_eplane(x):
    """Marcuvitz (1951) Sec. 5.28a eqs. (3a), (4a): B_a/Y0, B_b/Y0 of the right-angle E-plane corner at the planes
    T' through the inner corner; x = 2b/lambda_g (b in the bend plane). Transcription: docs/literature/
    marcuvitz-steps-and-bends.md."""
    E = np.exp
    A0 = 4 / np.pi * x ** 2 / (1 - x ** 2)
    A0p = A0 - 8 / np.sinh(np.pi)
    A1 = 1 / np.pi * x ** 2 / (1 - 0.5 * x ** 2)
    A2 = 4 * (1 / (np.sqrt(1 - x ** 2) * (1 - E(-2 * np.pi * np.sqrt(1 - x ** 2)))) - 1 / (1 - E(-2 * np.pi)))
    Ba = x * (-1 / np.tan(np.pi * x) / x + 1 / (np.pi * x ** 2) + np.pi / 6 - np.log(2)
              - (A0 * E(-np.pi / 2) + (A1 - A2) * E(-np.pi) + (1 + 5 * E(-np.pi)) * A0 ** 2 / 16)
              / (1 - (1 + 5 * E(-np.pi)) * (A1 - A2) / 4))
    ssum = sum(1 / (n * (E(2 * np.pi * n) - 1)) for n in range(1, 20))
    t4 = (1 / (np.pi * x)) * (1 + np.pi * x / np.tan(np.pi * x)
                              - np.pi * x ** 2 * (5 * np.log(2) - 7 * np.pi / 6 - 8 * ssum)
                              + np.pi * x ** 2 * (A0p * E(-np.pi / 2) - (A1 + A2) * E(-np.pi)
                                                  + (1 - 3 * E(-np.pi)) * A0p ** 2 / 16)
                              / (1 + (1 - 3 * E(-np.pi)) * (A1 + A2) / 4))
    return Ba, (t4 + Ba) / 2


def _marcuvitz_hplane(x):
    """Marcuvitz Sec. 5.29a eqs. (3a), (4a): B_a/Y0, B_b/Y0 of the right-angle H-plane corner at T';
    x = 2a/lambda_g, 0 <= x <= 1.2."""
    E = np.exp
    A1, A1p = 0.3366, 0.3071
    r = np.sqrt(3 - x ** 2) / (1 - E(-2 * np.pi * np.sqrt(3 - x ** 2)))
    A2 = (0.1801 * (1 + x ** 2) / (4 - x ** 2) + 0.2021) ** 2 / (r - 0.1858 - 0.1592 * (1 + x ** 2) / (7 - x ** 2))
    A2p = (0.1801 * (1 + x ** 2) / (4 - x ** 2) + 0.1669) ** 2 / (r + 0.1432 + 0.1592 * (1 + x ** 2) / (7 - x ** 2))
    A3 = 0.01 - 0.0095 * x if x <= 1 else -0.00416 / (1.5 - x) + 0.0040 + 0.0019 * x + 0.0020 * x ** 2
    A3p = -0.00146 / (1.26 - x) - 0.0016 + 0.0034 * x - 0.0026 * x ** 2
    Ba = (1 / x) * (A1 - x / np.tan(np.pi * x) + 1 / np.pi * (1 + x ** 2) / (1 - x ** 2) + A2) * (1 + A3)
    Bab = (1 / x) * (-A1p - x / np.tan(np.pi * x) - 1 / np.pi * (1 + x ** 2) / (1 - x ** 2) + A2p) * (1 + A3p)
    return Ba, (Bab - Ba) / 2


@pytest.mark.parametrize("plane", ["H", "E"])
def test_unmitered_corner_against_marcuvitz_complex(plane):
    """Complex S11 and S21 of TE10 at the planes through the inner corner vs Marcuvitz's pi networks
    (Secs. 5.28a, 5.29a; stated accuracy about 1 % on the susceptances). This fixes the reference planes, the
    port-2 frame (sign of S21) and the sign of the LSE reflection (E plane)."""
    w, h = WR90.w, WR90.h
    m = MiterMM(SQ2, plane, WR90)
    for f in (7.5e9, 8.5e9, 10e9):
        k = 2 * np.pi * f / cst.c
        beta = np.sqrt(k * k - (np.pi / w) ** 2)
        lg = 2 * np.pi / beta
        if plane == "H":
            Ba, Bb = _marcuvitz_hplane(2 * w / lg)
            r11, r21 = _marcuvitz_pi(1j * Ba, 1j * Bb)
            mode = "TE1,0"
        else:
            Ba, Bb = _marcuvitz_eplane(2 * h / lg)
            r11, r21 = _marcuvitz_pi(1j * Ba, -1j * Bb)
            mode = "TE0,1"
        g = m.gsm(f)
        i = g.basis1.index(mode)
        assert abs(g.S11[i, i] - r11) < 3e-3, (f, g.S11[i, i], r11)
        assert abs(g.S21[i, i] - r21) < 3e-3, (f, g.S21[i, i], r21)


def test_plane_frames_and_line():
    """In a Line: the E-plane miter gets the frame changes; the line equals the manual cascade; lossy guide refused."""
    from wrmodes.components.guide import map_basis
    from wrmodes.components.frames import FrameChange
    from wrmodes.gsm import straight
    f = 21e9
    bH = WR90.basis(f, 1.0, plane="H")
    for plane in ("H", "E"):
        mm = MiterMM(0.87, plane, n_modes=40)
        line = Line(WR90, [Straight(0.01), mm, Straight(0.02)])
        g = line.gsm(f)
        assert g.unitarity_defect() < 1e-10 and g.reciprocity_defect() < 1e-10
        if plane == "H":
            ref = cascade(straight(bH, f, 0.01), mm.with_guide(WR90).gsm(f, bH, bH), straight(bH, f, 0.02))
        else:
            assert sum(line.inserted) == 2
            bE = map_basis(bH, "H", "E")
            ref = cascade(straight(bH, f, 0.01), FrameChange("H", "E", WR90).gsm(f, bH, bE),
                          mm.with_guide(WR90).gsm(f, bE, bE), FrameChange("E", "H", WR90).gsm(f, bE, bH),
                          straight(bH, f, 0.02))
        assert np.max(np.abs(ref.S - g.S)) < 1e-12
    with pytest.raises(NotImplementedError):
        MiterMM(None, "H", Guide(22.86e-3, 10.16e-3, sigma=3.5e7)).gsm(10e9)


def test_interface_attributes():
    m = MiterMM(None, "E", WR90)
    assert m.plane == "E" and m.frame == "E"
    assert np.isclose(m.arm_length, WR90.h / 2)
    assert np.isclose(m.d, 1 / SQ2) and np.isclose(m.theta, np.pi / 2)
    b1, b2 = m.default_basis(30e9)
    assert b1 == WR90.basis(30e9, 1.0, plane="E") and b1.frame == "E"
    assert MiterMM(2.0, "H").d == SQ2                      # beyond sqrt(2): unmitered
    with pytest.raises(ValueError):
        MiterMM(0.6)
    with pytest.raises(ValueError):
        MiterMM(None, angle_deg=60)
    with pytest.raises(ValueError):
        MiterMM(None, "H", WR90).gsm(10e9, WR90.basis(10e9, plane="E"), WR90.basis(10e9, plane="E"))
    assert MiterMM(None, "H").with_guide(WR90).guide == WR90


def test_turning_the_other_way():
    """angle_deg = -90 is the mirror image u -> -u: S' = D S D with D = (-1)^(m+1); TE10 unchanged in magnitude."""
    f = 30e9
    p, q = MiterMM(0.9, "H", WR90, n_modes=40), MiterMM(0.9, "H", WR90, n_modes=40, angle_deg=-90)
    gp, gq = p.gsm(f), q.gsm(f)
    D = np.array([(-1.0) ** (m.m + 1) for m in gp.basis1.modes])
    assert np.max(np.abs(gq.S21 - D[:, None] * gp.S21 * D[None, :])) < 1e-14
    assert np.max(np.abs(gq.S11 - D[:, None] * gp.S11 * D[None, :])) < 1e-14
    # +90 then -90 in the same plane is an offset; +90 then +90 a U-turn: both unitary
    g = (gp @ gq)
    assert g.unitarity_defect() < 1e-10


def test_dead_families_are_straight():
    """A family without a propagating mode is treated as uncoupled straight modes over the centre line a."""
    f = 12e9
    b = ModeBasis.propagating(f, WR90.w, WR90.h, cutoff_factor=3.0, evanescent_families="all", frame="H")
    g = MiterMM(None, "H", WR90, n_modes=30).gsm(f, b, b)
    k = 2 * np.pi * f / cst.c
    for i, md in enumerate(b.modes):
        if k * k - (md.n * np.pi / WR90.h) ** 2 <= 0:
            assert g.S11[i, i] == 0
            assert np.isclose(g.S21[i, i], np.exp(-1j * b.beta(f)[i] * WR90.w))


def test_requested_port_modes_and_truncation_independence():
    """gsm on a sub-basis returns the sub-block of the gsm on the larger basis."""
    m = MiterMM(0.87, "H", WR90, n_modes=50)
    f = 33e9
    big = WR90.basis(f, 2.5, plane="H")
    small = WR90.basis(f, 1.0, plane="H")
    gb, gs = m.gsm(f, big, big), m.gsm(f, small, small)
    ii = [big.index(md) for md in small.modes]
    assert np.max(np.abs(gb.S21[np.ix_(ii, ii)] - gs.S21)) < 1e-13
    assert np.max(np.abs(gb.S11[np.ix_(ii, ii)] - gs.S11)) < 1e-13


# ============================================================================================ literature
def _s11_2d_db(d, kap_a, wall, m=0, M=80):
    c = min(max(SQ2 * d - 1, 0.0), 1.0)
    S11, _ = c2.miter_2d(1.0, c, wall, kap_a ** 2, M, int(np.ceil(M / MiterMM.RATIO[wall])))
    return 20 * np.log10(abs(S11[m, m]))


def _zscores(fn, curves, model, step=3):
    D = load_csv(fn)
    out = {}
    for cur in curves:
        arr = D[cur][::step]
        out[cur] = np.array([(model(cur, x) - y) / u for x, y, u in arr])
    return out


def test_alessandri_eplane_b_over_a_half():
    """Alessandri et al. (1994) Figs. 4-5 (E plane, TE10 = local LSE m = 0, n = 1) with b/a = 0.5.

    Within the digitizing uncertainty (rms z < 1, |z| <= 2.5) for d/b = 1.41, 0.707, 0.9, 0.87 (line and dots).
    The deep curves d/b = 0.85 and 0.83 (|S11| < -38 dB) differ near their zeros by up to a few dB, i.e. by less
    than 3e-3 in |S11| (zero positions within about 0.005 in f/fc; docs/reports/WP4b.md)."""
    r = 0.5
    dmap = {"1.41": SQ2, "0.707": 1 / SQ2}

    def model(cur, x):
        d = float(dmap.get(cur.split("=")[1], cur.split("=")[1]))
        return _s11_2d_db(d, np.pi * r * np.sqrt(x * x - 1), "N")

    for fn, curves in (("alessandri1994_fig4.csv", ["d/b=1.41", "d/b=0.707"]),
                       ("alessandri1994_fig5.csv", ["d/b=0.9", "d/b=0.87"])):
        for cur, z in _zscores(fn, curves, model, step=4).items():
            assert np.sqrt(np.mean(z ** 2)) < 1.0, (fn, cur)
            assert np.max(np.abs(z)) < 2.5, (fn, cur)
    D = load_csv("alessandri1994_fig5.csv")
    for cur in ("d/b=0.85", "d/b=0.83"):
        for x, y, u in D[cur][::4]:
            assert abs(10 ** (model(cur, x) / 20) - 10 ** (y / 20)) < 3e-3, (cur, x)


def test_alessandri_rules_out_wr90_ratio():
    """With WR-90's b/a = 0.444 the unmitered and optical-mirror curves are off by many digitizing uncertainties."""
    z = _zscores("alessandri1994_fig4.csv", ["d/b=1.41"],
                 lambda cur, x: _s11_2d_db(SQ2, np.pi * (10.16 / 22.86) * np.sqrt(x * x - 1), "N"), step=6)
    assert np.sqrt(np.mean(z["d/b=1.41"] ** 2)) > 4


def test_alimenti_unmitered_and_optical_mirror():
    """Alimenti et al. (1994) Fig. 2 (unmitered) and Fig. 3 (optical mirror), H plane: rms z < 1.5."""
    for fn, cur, d in (("alimenti1994_fig2_mmfd.csv", "MMFD", SQ2), ("alimenti1994_fig3_mm.csv", "MM", 1 / SQ2)):
        z = _zscores(fn, [cur], lambda c_, x: _s11_2d_db(d, np.pi * x, "D"), step=3)[cur]
        assert np.sqrt(np.mean(z ** 2)) < 1.5, fn


def test_alimenti_chamfered_disagreement_is_documented():
    """Alimenti Figs. 4 and 6 (chamfered, t/a = 0.30-0.386) are NOT reproduced (docs/discrepancies/
    2026-11-11-alimenti-chamfered-miters.md). Pinned here: our zero of t/a = 0.38 lies at f/fc = 1.636, where an
    independent finite-difference solution (benchmarks/modematch/fd_hplane_miter.py) agrees, not at 1.69."""
    d = (1 + 0.38) / SQ2
    xs = np.linspace(1.60, 1.70, 41)
    v = [_s11_2d_db(d, np.pi * x, "D", M=60) for x in xs]
    assert abs(xs[int(np.argmin(v))] - 1.636) < 0.004
    # FD extrapolated |S11| (benchmarks/modematch/data/fd_hplane_miter.npz): 0.03710 at 1.5, 0.01387 at 1.69
    for x, ref in ((1.5, 0.03710), (1.69, 0.01387)):
        assert abs(10 ** (_s11_2d_db(d, np.pi * x, "D") / 20) - ref) < 1e-4


def test_kotik_optical_mirror_overmoded():
    """Kotik & Sivov (1965) Fig. 3: |B1|^2 of TE10 through the optical mirror vs kd = 2 pi a / lambda, within the
    uncertainty (reading + 0.005 drafting; rms z < 1.5, |z| < 2.5), except the points just above the TE40, TE60 and
    TE80 cutoffs (kd - n pi in [-0.1, 0.4], n = 4, 6, 8), where our dips are sharper than Kotik's N <= 18 solution as drawn, and
    kd > 30.4, where we lie 0.011 lower (docs/reports/WP4b.md)."""
    D = load_csv("kotik1965_fig3_rigorous.csv")["rigorous"][::3]
    z = []
    for x, y, u in D:
        if any(-0.1 <= x - n * np.pi <= 0.4 for n in (4, 6, 8)) or x > 30.4:
            continue
        S11, S21 = c2.miter_2d(1.0, 0.0, "D", x * x, 80, 70)
        z.append((abs(S21[0, 0]) ** 2 - y) / u)
    z = np.array(z)
    assert len(z) > 100
    assert np.sqrt(np.mean(z ** 2)) < 1.5 and np.max(np.abs(z)) < 2.5


# ============================================================================================ Step parity
STEPS = [(Step(22.86e-3, 10.16e-3, 22.86e-3, 1.6e-3), 30e9),
         (Step(22.86e-3, 1.6e-3, 22.86e-3, 10.16e-3), 30e9),
         (Step(22.86e-3, 10.16e-3, 18e-3, 8e-3), 20e9),
         (Step(12e-3, 6e-3, 22.86e-3, 10.16e-3), 25e9),
         (Step(22.86e-3, 10.16e-3, 16e-3, 10.16e-3), 20e9),
         (Step(22.86e-3, 10.16e-3, 16e-3, 8e-3, du=1e-3), 20e9),
         (Step(22.86e-3, 10.16e-3, 16e-3, 8e-3, dv=1e-3), 20e9),
         (Step(22.86e-3, 10.16e-3, 16e-3, 8e-3, du=1e-3, dv=0.5e-3), 15e9)]


@pytest.mark.parametrize("step,f", STEPS)
def test_step_parity_reduction_equals_full_solve(step, f):
    b1, b2 = step.default_basis(f, cutoff_factor=6.0)
    full = step.solve(f, b1, b2, reduce=False)
    red = step.solve(f, b1, b2)
    assert np.max(np.abs(full.S - red.S)) < 1e-12
    assert red.reciprocity_defect() == 0.0 and red.unitarity_defect() < 1e-10


def test_step_gsm_uses_reduction_and_keys():
    assert Step(22.86e-3, 10.16e-3, 18e-3, 8e-3)._class_key()(Mode("TE", 3, 2)) == (1, 0)
    assert Step(22.86e-3, 10.16e-3, 22.86e-3, 1.6e-3)._class_key()(Mode("TM", 3, 5)) == (3, 1)
    assert Step(22.86e-3, 10.16e-3, 16e-3, 8e-3, du=1e-3, dv=1e-3)._class_key() is None


# ============================================================================================ slow
@pytest.mark.slow
@pytest.mark.parametrize("plane", ["H", "E"])
@pytest.mark.parametrize("chamfer", [None, 0.87, 1.0, SQ2])
def test_default_truncation_reaches_1e4(plane, chamfer):
    """Default (error-controlled) truncation: every propagating entry within 1e-4 of a reference computed with a
    3.3 times tighter error target (WR-90, 30 and 55 GHz; the E-plane d/a = 1.0 case at 30 GHz sits on a sharp
    resonance of the corner just below a cutoff, which the error control resolves)."""

    class Tight(MiterMM):
        ERR_TARGET = 1.5e-5

    m, mt = MiterMM(chamfer, plane, WR90), Tight(chamfer, plane, WR90)
    for f in (30e9, 55e9):
        b, _ = m.default_basis(f)
        prop = np.concatenate([b.propagating_mask(f)] * 2)
        sel = np.ix_(prop, prop)
        g0 = m.gsm(f)
        assert max(v[1] for v in g0.truncation.values() if v[1] is not None) <= MiterMM.ERR_TARGET
        assert np.max(np.abs(g0.S[sel] - mt.gsm(f).S[sel])) < 1e-4


def test_error_control_refines_near_a_resonance():
    """E-plane d/a = 1.0, WR-90, 30 GHz: family n = 1 has k_t a / pi = 1.98, just below the cutoff of its LSM m = 2,
    where the corner has a sharp resonance; the error control raises M there and nowhere else is it needed."""
    m = MiterMM(1.0, "E", WR90)
    g = m.gsm(30e9)
    M, est = g.truncation[(1, "LSM")]
    assert M > 3 * m.default_n_modes(30e9, 1, "D") and est <= MiterMM.ERR_TARGET
    assert g.unitarity_defect() < 1e-10


@pytest.mark.slow
def test_performance_plain_hplane_90ghz():
    """WR-90 H-plane plain miter at 90 GHz, full default basis: under 1 s per frequency (card WP4b)."""
    m = MiterMM(None, "H", WR90)
    m.gsm(90e9)
    t = time.perf_counter()
    for f in (89.7e9, 90e9, 90.3e9):
        m.gsm(f)
    assert (time.perf_counter() - t) / 3 < 1.0


@pytest.mark.slow
def test_alimenti_fd_cross_check():
    """The finite-difference check of benchmarks/modematch/fd_hplane_miter.py, re-run at a coarse grid pair."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fd", os.path.join(HERE, "..", "benchmarks", "modematch",
                                                                      "fd_hplane_miter.py"))
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    t, x = 0.38, 1.5
    r = [abs(fd.fd_s11(N, int(round(t * N)), np.pi * x)) for N in (100, 200)]
    extrap = r[1] + (r[1] - r[0]) / (2 ** 1.6 - 1)
    mm = 10 ** (_s11_2d_db((1 + t) / SQ2, np.pi * x, "D", M=160) / 20)
    assert abs(extrap - mm) < 3e-4
