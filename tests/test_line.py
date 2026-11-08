"""Tests of wrmodes.line (WP1): cascades in the global frame, frame changes, the JINST section 7 sequence, sweeps."""
import multiprocessing as mp
import sys

import numpy as np
import pytest
from scipy.linalg import block_diag as _bd

import wrmodes as wr
from wrmodes import Line, Guide, Straight, SmoothBend, FrameChange, SechCurvature, ConstantCurvature
from wrmodes.components import map_basis
from wrmodes.gsm import from_transfer_matrix
from wrmodes.modes import eplane_frame_signs

WR90 = Guide(0.9 * 0.0254, 0.4 * 0.0254)


# ------------------------------------------------------------------------------------------ references
def bend_phi(f, basis, L, kappa, atol=1e-6):
    """Full 2N x 2N transfer matrix of a bend in the order of a local basis of propagating modes, assembled from
    the per-family transfer matrices of solve_transfer_matrix (wrmodes 0.1 method, error_control='full')."""
    N = len(basis)
    phi = np.zeros((2 * N, 2 * N), complex)
    for n in basis.families():
        idx = np.array(basis.family(n))
        sub = basis.subset(idx)
        m0 = sub.modes[0]
        _, r = wr.solve_transfer_matrix(f, basis.a, basis.b, (m0.kind, m0.m, m0.n), L, kappa,
                                        coupled_modes=sub.legacy_modes(f), atol=atol, error_control="full",
                                        profile=False)
        k = len(idx)
        P = r.transfer_matrix
        for bi, rows in enumerate((idx, idx + N)):
            for bj, cols in enumerate((idx, idx + N)):
                phi[np.ix_(rows, cols)] = P[bi * k:(bi + 1) * k, bj * k:(bj + 1) * k]
    return phi


def straight_phi(f, basis, L):
    b = basis.beta(f)
    return np.diag(np.concatenate([np.exp(-1j * b * L), np.exp(1j * b * L)]))


def to_global(phi_local, F):
    """Transfer matrix in the global frame from one in a local frame with a_local = F a_global."""
    FF = _bd(F, F)
    return FF.T @ phi_local @ FF


def reference_phi(f, gbasis, items):
    """Product of full transfer matrices for a list of ('bend', plane, L, kappa) / ('straight', L) in the global basis."""
    phi = np.eye(2 * len(gbasis), dtype=complex)
    for it in items:
        if it[0] == "straight":
            phi = straight_phi(f, gbasis, it[1]) @ phi
            continue
        _, plane, L, kappa = it
        if plane == "H":
            phi = bend_phi(f, gbasis, L, kappa) @ phi
        else:
            lb = map_basis(gbasis, "H", "E")
            F = eplane_frame_signs(gbasis, lb)
            phi = to_global(bend_phi(f, lb, L, kappa), F) @ phi
    return phi


def components(items):
    return [Straight(it[1]) if it[0] == "straight" else SmoothBend(it[2], it[3], plane=it[1]) for it in items]


def assert_lossless(g, tol=1e-10):
    assert g.reciprocity_defect() <= tol * np.abs(g.S).max()
    assert g.unitarity_defect() <= tol


# ------------------------------------------------------------------------------------------ cascades
def test_same_plane_cascade_equals_transfer_matrix_product():
    f = 40e9
    items = [("bend", "H", 0.15, ConstantCurvature(1 / 0.25)), ("straight", 0.7),
             ("bend", "H", 0.12, ConstantCurvature(-1 / 0.3))]
    line = Line(WR90, components(items))
    g = line.gsm(f)
    ref = from_transfer_matrix(reference_phi(f, g.basis1, items), g.basis1, f)
    assert np.max(np.abs(g.S - ref.S)) < 1e-12
    assert_lossless(g)


def test_mixed_plane_line_equals_transfer_matrix_product_in_the_global_frame():
    f = 33e9
    items = [("bend", "H", 0.3, SechCurvature(60, 0.3)), ("straight", 0.5),
             ("bend", "E", 0.4, SechCurvature(90, 0.4)), ("straight", 2.0),
             ("bend", "E", 0.4, -SechCurvature(45, 0.4)), ("bend", "H", 0.3, -SechCurvature(90, 0.3))]
    line = Line(WR90, components(items))
    assert sum(line.inserted) == 2
    g = line.gsm(f)
    assert g.basis1 == WR90.basis(f) and g.basis1.frame == "H"
    ref = from_transfer_matrix(reference_phi(f, g.basis1, items), g.basis1, f)
    assert np.max(np.abs(g.S - ref.S)) < 1e-10
    assert_lossless(g)
    # the E-plane bends couple different global families
    i, j = g.basis1.index("TE1,0"), g.basis1.index("TE1,1")
    assert abs(g.S21[i, j]) > 1e-3


@pytest.mark.parametrize("f", [25e9, 40e9])
def test_e_plane_bend_in_a_line_equals_wrmodes_0_1(f):
    L, kappa = 0.4, SechCurvature(90, 0.4)
    g = Line(WR90, [SmoothBend(L, kappa, plane="E")]).gsm(f)
    gb = g.basis1
    lb = map_basis(gb, "H", "E")
    F = eplane_frame_signs(gb, lb)
    S21_local, S11_local = F @ g.S21 @ F.T, F @ g.S11 @ F.T
    for n in lb.families():
        idx = lb.family(n)
        modes, T, R, beta = wr.transmission_matrix(f, WR90.h, WR90.w, n, L, kappa)     # a = h, b = w
        assert [m[0] for m in modes] == [lb[i].name for i in idx]
        assert np.max(np.abs(S21_local[np.ix_(idx, idx)] - T)) < 1e-10
        assert np.max(np.abs(S11_local[np.ix_(idx, idx)] - R)) < 1e-10
    # TE1,0 (global) -> TE0,1 (local) with the sign -1 on both sides: the transmission keeps its sign
    j = gb.index("TE1,0")
    modes, T, R, beta = wr.transmission_matrix(f, WR90.h, WR90.w, 1, L, kappa)
    assert np.isclose(g.S21[j, j], T[0, 0], atol=1e-10, rtol=0)
    assert_lossless(g)


# ------------------------------------------------------------------------------------------ JINST section 7
SEQ_FREQS = np.linspace(18e9, 90e9, 10)


def sequence_items(plane):
    return [("bend", plane, 0.6, SechCurvature(60, 0.6)), ("straight", 10.0),
            ("bend", plane, 0.4, SechCurvature(90, 0.4)), ("straight", 1.0),
            ("bend", plane, 0.4, -SechCurvature(90, 0.4))]


@pytest.mark.parametrize("plane", ["H", "E"])
def test_paper_sequence_of_bends(plane):
    """JINST section 7: 60 deg / 10 m / 90 deg / 1 m / -90 deg, TE1,0 in the H or E plane, at 10 frequencies, in
    the input family (global n = 0 for H, global m = 1 for E). Reference: the product of the full 2N x 2N transfer
    matrices (1e-10). The T-only product of benchmarks/paper_figures.py ignores the reflections between the
    bends and agrees to about 1e-6."""
    items = sequence_items(plane)
    line = Line(WR90, components(items))
    worst_full, worst_T = 0.0, 0.0
    for f in SEQ_FREQS:
        full = WR90.basis(f)
        gb = full.subset([i for i, m in enumerate(full) if (m.n == 0 if plane == "H" else m.m == 1)])
        g = line.gsm(f, basis1=gb)
        assert_lossless(g)
        # full transfer matrices of the bends, in the local frame (one family)
        lb = map_basis(gb, "H", plane)
        F = eplane_frame_signs(gb, lb) if plane == "E" else np.eye(len(gb))
        phis = [bend_phi(f, lb, it[2], it[3]) if it[0] == "bend" else straight_phi(f, lb, it[1]) for it in items]
        phi = np.linalg.multi_dot(phis[::-1])
        ref = from_transfer_matrix(to_global(phi, F), gb, f)
        worst_full = max(worst_full, np.max(np.abs(g.S - ref.S)))
        # T-only product (benchmarks/paper_figures.py, fig_sequence, nominal straight lengths): forward blocks only
        N = len(gb)
        Ts = [P[:N, :N] - P[:N, N:] @ np.linalg.solve(P[N:, N:], P[N:, :N]) for P in phis]
        Tseq = np.linalg.multi_dot(Ts[::-1])
        worst_T = max(worst_T, np.max(np.abs(F @ g.S21 @ F.T - Tseq)))
    assert worst_full < 1e-10, worst_full
    assert worst_T < 1e-5, worst_T


# ------------------------------------------------------------------------------------------ frames in a line
def test_frame_changes_are_inserted_and_described():
    line = Line(WR90, [Straight(1.0), SmoothBend(0.3, 1.0, plane="E"), Straight(2.0), SmoothBend(0.3, 1.0),
                       SmoothBend(0.3, 1.0, plane="E")])
    kinds = [type(e).__name__ for e in line.elements]
    assert kinds == ["Straight", "FrameChange", "SmoothBend", "Straight", "FrameChange", "SmoothBend",
                     "FrameChange", "SmoothBend", "FrameChange"]
    assert line.inserted == (False, True, False, False, True, False, True, False, True)
    text = line.describe()
    assert text.count("inserted by Line") == 4 and "H->E" in text and "E->H" in text
    # every element is attached to the line's guide
    assert all(e.guide == WR90 for e in line.elements)


def test_explicit_frame_changes_and_port_plane():
    f = 30e9
    bend = SmoothBend(0.4, SechCurvature(90, 0.4), plane="E")
    auto = Line(WR90, [bend]).gsm(f)
    explicit = Line(WR90, [FrameChange("H", "E"), bend, FrameChange("E", "H")])
    assert not any(explicit.inserted)
    assert np.array_equal(explicit.gsm(f).S, auto.S)
    with pytest.raises(ValueError):
        Line(WR90, [FrameChange("E", "H")])
    # a line with its ports in the E frame needs no frame change around an E-plane bend
    eline = Line(WR90, [bend], plane="E")
    assert len(eline.elements) == 1
    ge = eline.gsm(f)
    assert ge.basis1.frame == "E" and np.array_equal(ge.S, bend.with_guide(WR90).gsm(f, ge.basis1).S)


def test_h_e_h_in_a_line_is_the_identity():
    f = 50e9
    g = Line(WR90, [FrameChange("H", "E"), FrameChange("E", "H")]).gsm(f, cutoff_factor=1.5)
    N = len(g.basis1)
    assert np.array_equal(g.S, np.block([[np.zeros((N, N)), np.eye(N)], [np.eye(N), np.zeros((N, N))]]))


def test_nested_lines_and_empty_line():
    f = 30e9
    inner = Line(WR90, [SmoothBend(0.4, SechCurvature(90, 0.4), plane="E"), Straight(0.5)])
    outer = Line(WR90, [Straight(1.0), inner, SmoothBend(0.3, SechCurvature(30, 0.3))])
    flat = Line(WR90, [Straight(1.0), SmoothBend(0.4, SechCurvature(90, 0.4), plane="E"), Straight(0.5),
                       SmoothBend(0.3, SechCurvature(30, 0.3))])
    assert np.allclose(outer.gsm(f).S, flat.gsm(f).S, atol=1e-13, rtol=0)
    e = Line(WR90, []).gsm(f)
    assert np.array_equal(e.S21, np.eye(len(e.basis1))) and not e.S11.any()


def test_line_refuses_components_of_another_guide():
    with pytest.raises(ValueError):
        Line(WR90, [Straight(1.0, guide=Guide(0.02, 0.01))])


def test_evanescent_modes_in_a_mixed_line_do_not_change_the_propagating_result():
    f = 33e9
    line = Line(WR90, components([("bend", "H", 0.3, SechCurvature(60, 0.3)), ("straight", 0.05),
                                  ("bend", "E", 0.4, SechCurvature(90, 0.4))]))
    g1, g15 = line.gsm(f), line.gsm(f, cutoff_factor=1.5)
    assert len(g15.basis1) > len(g1.basis1)
    idx = [g15.basis1.index(m) for m in g1.basis1]
    assert np.allclose(g15.S[np.ix_(idx + [i + len(g15.basis1) for i in idx], idx + [i + len(g15.basis1) for i in idx])],
                       g1.S, atol=1e-13, rtol=0)
    assert_lossless(g15)


# ------------------------------------------------------------------------------------------ sweeps
def sweep_line():
    return Line(WR90, [SmoothBend(0.3, SechCurvature(90, 0.3)), Straight(1.0),
                       SmoothBend(0.3, SechCurvature(60, 0.3), plane="E")])


SWEEP_FREQS = np.array([12e9, 17e9, 21e9, 26e9, 30e9])


def test_sweep_with_1_and_4_processes_is_identical():
    line = sweep_line()
    r1 = line.sweep(SWEEP_FREQS, "TE1,0", processes=1, keep_gsm=True)
    r4 = line.sweep(SWEEP_FREQS, ("TE", 1, 0), processes=4, keep_gsm=True)
    assert np.array_equal(r1.loss_db, r4.loss_db) and np.array_equal(r1.reflected_power, r4.reflected_power)
    assert list(r1.converted_power) == list(r4.converted_power)
    for m in r1.converted_power:
        assert np.array_equal(r1.converted_power[m], r4.converted_power[m], equal_nan=True)
    for g1, g4 in zip(r1.gsms, r4.gsms):
        assert np.array_equal(g1.S, g4.S)
    # contents
    g = line.gsm(SWEEP_FREQS[-1])
    j = g.basis1.index("TE1,0")
    assert np.isclose(r1.loss_db[-1], -20 * np.log10(abs(g.S21[j, j])), rtol=1e-14)
    assert np.isclose(r1.converted_power["TE2,0"][-1], abs(g.S21[g.basis2.index("TE2,0"), j]) ** 2, rtol=1e-14)
    assert np.isnan(r1.converted_power["TE2,0"][0])                 # TE2,0 is cut off at 12 GHz
    # power balance (lossless): transmitted + converted + reflected = 1
    total = 10 ** (-r1.loss_db / 10) + np.nansum([v for v in r1.converted_power.values()], axis=0) + r1.reflected_power
    assert np.allclose(total, 1, atol=1e-10, rtol=0)


@pytest.mark.skipif(sys.platform != "linux" or "fork" not in mp.get_all_start_methods(),
                    reason="fork start method (tested on Linux only; unsafe with macOS system libraries)")
def test_sweep_with_unpicklable_curvature_falls_back_to_fork():
    line = Line(WR90, [SmoothBend(0.3, lambda z: np.full(np.shape(z), 2.0))])
    r1 = line.sweep(SWEEP_FREQS[:2], "TE1,0", processes=1)
    r2 = line.sweep(SWEEP_FREQS[:2], "TE1,0", processes=2)
    assert np.array_equal(r1.loss_db, r2.loss_db)


def test_sweep_below_cutoff_of_the_input_mode():
    r = Line(WR90, [Straight(1.0)]).sweep([5e9, 10e9], "TE1,0", processes=1)
    assert np.isnan(r.loss_db[0]) and abs(r.loss_db[1]) < 1e-12
