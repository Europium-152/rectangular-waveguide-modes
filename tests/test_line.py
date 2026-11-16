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


def test_evanescent_modes_in_a_mixed_line_change_the_propagating_result_little():
    """WF2: SmoothBend couples the evanescent modes of the basis (before WF2 they were uncoupled and the propagating
    result was unchanged to 1e-13). For these sech bends the TE1,0 column moves by 2e-6 in power; entries between
    high-order modes, which couple strongly to the evanescent ones inside the bends, by up to 1.3e-3."""
    f = 33e9
    line = Line(WR90, components([("bend", "H", 0.3, SechCurvature(60, 0.3)), ("straight", 0.05),
                                  ("bend", "E", 0.4, SechCurvature(90, 0.4))]))
    g1, g15 = line.gsm(f), line.gsm(f, cutoff_factor=1.5)
    assert len(g15.basis1) > len(g1.basis1)
    idx = [g15.basis1.index(m) for m in g1.basis1]
    n = len(g15.basis1)
    S = g15.S[np.ix_(idx + [i + n for i in idx], idx + [i + n for i in idx])]
    D = np.abs(np.abs(S) ** 2 - np.abs(g1.S) ** 2)
    assert D[:, g1.basis1.index("TE1,0")].max() < 1e-5 and D.max() < 3e-3
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


# ------------------------------------------------------------------------------------------ review fixes
from wrmodes.components.base import Component           # noqa: E402
from wrmodes.gsm import GSM, straight                   # noqa: E402


class IdealJunction(Component):
    """Test double for a component that changes the cross-section (e.g. a step): zero length, transmits the modes
    common to both port bases and totally reflects the others. Symmetric and unitary, not physical."""

    def __init__(self, g_in, g_out, plane):
        self.guides, self.plane, self.name = (g_in, g_out), plane, "ideal junction"

    def default_basis(self, f, cutoff_factor=1.0):
        return tuple(g.basis(f, cutoff_factor, plane=self.plane) for g in self.guides)

    def gsm(self, f, basis1=None, basis2=None):
        if basis1 is None:
            basis1, basis2 = self.default_basis(f)
        S11, S22 = np.zeros((len(basis1),) * 2), np.zeros((len(basis2),) * 2)
        S21 = np.zeros((len(basis2), len(basis1)))
        for j, m in enumerate(basis1):
            if m in basis2:
                S21[basis2.index(m), j] = 1
            else:
                S11[j, j] = 1
        for i, m in enumerate(basis2):
            if m not in basis1:
                S22[i, i] = 1
        return GSM(S11, S21.T, S21, S22, basis1, basis2, f, self.name)


class NoPlane(Component):
    def default_basis(self, f, cutoff_factor=1.0):
        raise NotImplementedError

    def gsm(self, f, basis1=None, basis2=None):
        raise NotImplementedError


def test_line_requires_a_declared_plane():
    with pytest.raises(TypeError):
        Line(WR90, [Straight(1.0), NoPlane()])


def test_line_checks_the_cross_section_of_basis1():
    f = 30e9
    line = Line(WR90, [Straight(1.0)])
    with pytest.raises(ValueError):
        line.gsm(f, basis1=WR90.basis(f, plane="E").with_frame("H"))         # a = h, b = w, tagged H
    with pytest.raises(ValueError):
        line.gsm(f, basis1=Guide(0.02, 0.01).basis(f))


def test_line_through_a_change_of_cross_section():
    f = 30e9
    g2 = Guide(0.03, 0.012)
    bend = SmoothBend(0.3, SechCurvature(60, 0.3), plane="E")
    line = Line(WR90, [Straight(0.3), IdealJunction(WR90, g2, "H"), Straight(0.2), bend, Straight(0.1)])
    assert line.guides == (WR90, g2)
    assert line.elements[2].guide == g2 and line.elements[4].guide == g2
    assert "(to Guide(w=0.03" in line.describe()
    for cf in (1.0, 1.5):
        g = line.gsm(f, cutoff_factor=cf)
        b1, b2, b2e = WR90.basis(f, cf), g2.basis(f, cf), g2.basis(f, cf, plane="E")
        assert g.basis1 == b1 and g.basis2 == b2 and len(b2) > len(b1)
        manual = (straight(b1, f, 0.3) @ IdealJunction(WR90, g2, "H").gsm(f, b1, b2) @ straight(b2, f, 0.2)
                  @ FrameChange("H", "E").gsm(f, b2) @ bend.with_guide(g2).gsm(f, b2e) @ straight(b2e, f, 0.1)
                  @ FrameChange("E", "H").gsm(f, b2e))
        assert np.array_equal(g.S, manual.S)
        assert g.reciprocity_defect() < 1e-12      # (the test double is not unitary: it transmits evanescent modes)
    # the junction in the E plane gets E-frame bases in both guides
    gE = Line(WR90, [IdealJunction(WR90, g2, "E")]).gsm(f)
    assert gE.basis2 == g2.basis(f)
    with pytest.raises(ValueError):
        Line(WR90, [IdealJunction(g2, WR90, "H")])                         # starts in the wrong guide
    with pytest.raises(ValueError):
        Line(WR90, [IdealJunction(WR90, g2, "H"), Straight(1.0, guide=WR90)])


def test_nested_line_with_a_change_of_cross_section_is_flattened():
    f = 30e9
    g2 = Guide(0.03, 0.012)
    inner = Line(WR90, [IdealJunction(WR90, g2, "H"), SmoothBend(0.3, SechCurvature(60, 0.3), plane="E")])
    outer = Line(WR90, [Straight(0.5), inner, Straight(0.5)])
    assert outer.guides == (WR90, g2) and not any(isinstance(e, Line) for e in outer.elements)
    flat = Line(WR90, [Straight(0.5), IdealJunction(WR90, g2, "H"), SmoothBend(0.3, SechCurvature(60, 0.3), plane="E"),
                       Straight(0.5)])
    assert np.allclose(outer.gsm(f, cutoff_factor=1.3).S, flat.gsm(f, cutoff_factor=1.3).S, atol=1e-14, rtol=0)


def test_sweep_from_below_cutoff_with_a_mixed_plane_line():
    line = Line(WR90, [SmoothBend(0.3, SechCurvature(90, 0.3)), Straight(1.0),
                       SmoothBend(0.3, SechCurvature(60, 0.3), plane="E")])
    freqs = [4e9, 6e9, 8e9, 12e9, 18e9]
    assert line.gsm(4e9).S.shape == (0, 0)
    r = line.sweep(freqs, "TE1,0", processes=1, cutoff_factor=1.5, keep_gsm=True)
    assert np.all(np.isnan(r.loss_db[:2])) and np.all(np.isnan(r.reflected_power[:2]))
    assert np.all(np.isfinite(r.loss_db[2:]))
    assert len(r.gsms[0].basis1) == 0 and len(r.gsms[1].basis1) == 0      # nothing propagates: no live family
    assert len(r.gsms[2].basis1) >= 1
    r2 = line.sweep(freqs, "TE1,0", processes=1, cutoff_factor=1.5, keep_gsm=True, evanescent_families="all")
    assert len(r2.gsms[-1].basis1) >= len(r.gsms[-1].basis1)
    assert r2.gsms[-1].basis1 == WR90.basis(18e9, 1.5, evanescent_families="all")


class CountingBend(SmoothBend):
    calls = 0

    def gsm(self, f, basis1=None, basis2=None):
        CountingBend.calls += 1
        return super().gsm(f, basis1, basis2)


def test_cache_for_ensembles_of_straight_lengths():
    f = 30e9
    b60, b90 = CountingBend(0.6, SechCurvature(60, 0.6)), CountingBend(0.4, SechCurvature(90, 0.4), plane="E")
    lengths = [9.6, 10.0, 10.3]
    CountingBend.calls = 0
    plain = [Line(WR90, [b60, Straight(L), b90]).gsm(f) for L in lengths]
    assert CountingBend.calls == 3 * 2
    cache = {}
    CountingBend.calls = 0
    cached = [Line(WR90, [b60, Straight(L), b90]).gsm(f, cache=cache) for L in lengths]
    assert CountingBend.calls == 2
    for p, c in zip(plain, cached):
        assert np.array_equal(p.S, c.S)
    # a component used twice in one line is computed once
    CountingBend.calls = 0
    Line(WR90, [b60, Straight(1.0), b60]).gsm(f)
    assert CountingBend.calls == 1
    # another guide is another cache entry
    CountingBend.calls = 0
    Line(Guide(0.03, 0.012), [b60]).gsm(f, cache=cache)
    assert CountingBend.calls == 1


# ------------------------------------------------------------------------------------------ WF2: near-field warning
def test_warns_once_for_non_uniform_components_joined_directly_without_evanescent_modes():
    """WF2 item 2 (wrmodes.line, Evanescent near fields): a bend directly at a step, with the default
    cutoff_factor = 1, warns once per line; no warning with evanescent modes in the basis, or with a straight of at
    least NEAR_FIELD_DECAY_LENGTHS decay lengths of the least-attenuated evanescent mode between them."""
    from wrmodes.line import EvanescentJunctionWarning, NEAR_FIELD_DECAY_LENGTHS, _least_attenuated
    import warnings
    from wrmodes.modematch import Step
    f = 20e9
    W, H = WR90.w, WR90.h
    bend = SmoothBend(0.05, ConstantCurvature(1 / 0.1))
    step = Step(W, H, 1.1 * W, H)
    ln = Line(WR90, [bend, step])
    with pytest.warns(EvanescentJunctionWarning, match="joined directly"):
        ln.gsm(f)
    with warnings.catch_warnings():
        warnings.simplefilter("error", EvanescentJunctionWarning)
        ln.gsm(f)                                                  # once per line
        Line(WR90, [bend, step]).gsm(f, cutoff_factor=3)           # evanescent modes carried
        alpha, mode = _least_attenuated(WR90, f)
        d = NEAR_FIELD_DECAY_LENGTHS / alpha
        Line(WR90, [bend, Straight(1.01 * d), step]).gsm(f)       # far enough apart
        Line(WR90, [bend, Straight(0.2), SmoothBend(0.05, ConstantCurvature(-1 / 0.1))]).gsm(f)
        Line(WR90, [Straight(0.1), bend, Straight(0.1)]).gsm(f)    # one non-uniform component only
    with pytest.warns(EvanescentJunctionWarning, match=mode.name):
        Line(WR90, [bend, Straight(0.99 * d), FrameChange("H", "E"), FrameChange("E", "H"), step]).gsm(f)


# ------------------------------------------------------------------------------------------ WF3: buried planes
from wrmodes.components import Component
from wrmodes.gsm import GSM, BuriedPlanes, shift_planes, straight as _straight
from wrmodes.line import BuriedPlaneWarning, NonPassiveCascadeWarning, PASSIVITY_TOL, _largest_singular_value


class _Buried(Component):
    """Test component: `inner` with its reference planes moved `depth` (m) into it, as a component whose planes lie
    inside its discontinuity (GSM = shift_planes(inner GSM, d1, d2), `buried` = the inner GSM)."""

    def __init__(self, inner, depth, plane="H", name="buried"):
        self.inner, self.depth, self.plane, self.name = inner, tuple(depth), plane, name
        self.guide = getattr(inner, "guide", None)

    def with_guide(self, guide):
        return _Buried(self.inner.with_guide(guide), self.depth, self.plane, self.name)

    def default_basis(self, f, cutoff_factor=None):
        return self.inner.default_basis(f, cutoff_factor)

    def gsm(self, f, basis1=None, basis2=None):
        u = self.inner.gsm(f, basis1, basis2)
        g = shift_planes(u, *self.depth)
        g.buried = BuriedPlanes(self.depth, u)
        return g


class _Noisy(Component):
    """Test component: `inner` with eps (1 + i) added to every evanescent-evanescent reflection (a GSM whose
    propagating block is passive but whose evanescent entries are not physical, as a badly converged solver)."""
    plane = "H"

    def __init__(self, inner, eps, name="noisy"):
        self.inner, self.eps, self.name = inner, eps, name
        self.guide = getattr(inner, "guide", None)

    def with_guide(self, guide):
        return type(self)(self.inner.with_guide(guide), self.eps, self.name)

    def default_basis(self, f, cutoff_factor=None):
        return self.inner.default_basis(f, cutoff_factor)

    def gsm(self, f, basis1=None, basis2=None):
        g = self.inner.gsm(f, basis1, basis2)
        ev = ~g.basis1.propagating_mask(f)
        E = np.outer(ev, ev) * self.eps * (1 + 1j)
        return GSM(g.S11 + E, g.S12, g.S21, g.S22 + E, g.basis1, g.basis2, f, self.name)


def _close_relative(a, b, tol):
    """Entry-wise |a - b| <= tol max(|a|, |b|, 1e-300): for GSMs whose evanescent entries span many decades."""
    return np.all(np.abs(a - b) <= tol * np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300))


def test_shift_planes_is_a_straight_of_negative_length():
    f = 30e9
    b = WR90.basis(f, 2.0)
    g = SmoothBend(0.05, ConstantCurvature(1 / 0.1)).with_guide(WR90).gsm(f, b)
    L = 0.013
    assert np.allclose(shift_planes(g, -L, 0).S, (_straight(b, f, L) @ g).S, rtol=0, atol=1e-15)
    assert np.allclose(shift_planes(g, 0, -L).S, (g @ _straight(b, f, L)).S, rtol=0, atol=1e-15)
    back = shift_planes(shift_planes(g, L, 2 * L), -L, -2 * L)
    assert _close_relative(back.S, g.S, 1e-13)
    d = np.arange(len(b)) * 1e-3                                      # one distance per mode
    s = shift_planes(g, d, 0)
    assert np.allclose(s.S11, np.exp(1j * b.beta(f) * d)[:, None] * g.S11 * np.exp(1j * b.beta(f) * d)[None, :])
    with pytest.raises(ValueError):
        BuriedPlanes((-1e-3, 0.0), g)


@pytest.mark.parametrize("cf", [1.0, 2.0, 3.0])
def test_buried_planes_are_absorbed_by_the_straight(cf):
    """WF3: [X, Straight(s), X] with X's planes a depth d inside equals [X_u, Straight(s - 2d), X_u] (X_u at the
    uniform planes) to rounding, the line's own ports lying d inside as well (`buried` of the result). With
    buried_planes='keep' (before WF3) the cascade fails at cf >= 2 (factors exp(|beta| d) ~ 1e13)."""
    f, W = 30e9, WR90.w
    d = W
    mm = wr.MiterMM()
    X = _Buried(mm, (d, d))
    ref = Line(WR90, [mm, Straight(0.5 * W), mm]).gsm(f, cutoff_factor=cf)
    with pytest.warns(wr.line.EvanescentJunctionWarning) if cf == 1 else _nowarn(BuriedPlaneWarning):
        g = Line(WR90, [X, Straight(2 * d + 0.5 * W), X]).gsm(f, cutoff_factor=cf)
    assert g.buried is not None and g.buried.depth == (d, d)
    assert np.max(np.abs(g.buried.gsm.S - ref.S)) < 1e-13
    assert _close_relative(g.S, shift_planes(ref, d, d).S, 1e-12)
    if cf >= 2:
        with pytest.raises(np.linalg.LinAlgError):
            Line(WR90, [X, Straight(2 * d + 0.5 * W), X]).gsm(f, cutoff_factor=cf, buried_planes="keep")


class _nowarn:
    """Context manager: fail on the given warning category."""

    def __init__(self, cat):
        self.cat = cat

    def __enter__(self):
        import warnings
        self._c = warnings.catch_warnings()
        self._c.__enter__()
        warnings.simplefilter("error", self.cat)

    def __exit__(self, *exc):
        return self._c.__exit__(*exc)


def test_buried_planes_through_frame_changes_and_several_straights():
    """WF3: the depth is taken from the straights nearest to the component, across frame changes (an E-plane miter
    after an H-plane one) and across several straights; equal to the cascade at the uniform planes."""
    f, W = 12e9, WR90.w
    cf = 2.0
    d1, d2 = 0.6 * WR90.h, 0.4 * W
    XE = _Buried(wr.MiterMM(plane="E"), (0.0, d1), plane="E")
    XH = _Buried(wr.MiterMM(), (d2, 0.0))
    s = (0.2 * W, 0.3 * W, 0.1 * W + d1 + d2)
    g = Line(WR90, [XE, Straight(s[0]), Straight(s[1]), FrameChange("E", "H"), Straight(s[2]), XH]).gsm(
        f, cutoff_factor=cf)
    ref = Line(WR90, [wr.MiterMM(plane="E"), Straight(sum(s) - d1 - d2), wr.MiterMM()]).gsm(f, cutoff_factor=cf)
    assert g.buried is None
    assert np.max(np.abs(g.S - ref.S)) < 1e-12


def test_buried_planes_with_a_lossy_straight_keep_the_loss_of_the_whole_straight():
    """WF3: the depth moves only the evanescent modes into a lossy straight (where they are uncoupled), so the
    result is the cascade at the reference planes with the loss of the whole length s: equal to buried_planes='keep'
    where that is well conditioned, passive and reciprocal."""
    f = 18e9
    lossy = Guide(WR90.w, WR90.h, 3.5e7)
    bend = SmoothBend(0.05, ConstantCurvature(1 / 0.15))
    d = 2e-3
    X = _Buried(bend, (d, d))
    ln = Line(lossy, [X, Straight(0.03), X])
    g = ln.gsm(f, cutoff_factor=1.5)
    k = ln.gsm(f, cutoff_factor=1.5, buried_planes="keep")
    assert np.max(np.abs(g.S - k.S)) < 1e-10 * np.max(np.abs(k.S))
    assert g.reciprocity_defect() < 1e-10 * np.max(np.abs(g.S))
    assert 0 < g.passivity_margin() < 1e-2


def test_overlapping_buried_planes_warn_and_keep_the_reference_plane():
    """WF3: s < d2 + d1' (the discontinuities overlap): the first port takes its depth, the second keeps its GSM
    at the reference plane, and the line warns once (BuriedPlaneWarning) when the basis has evanescent modes;
    with propagating modes only it does not warn (the shift is a phase)."""
    f, W = 30e9, WR90.w
    d = 0.3 * W
    X = _Buried(wr.MiterMM(), (d, d))
    ln = Line(WR90, [X, Straight(1.5 * d), X])
    with pytest.warns(BuriedPlaneWarning, match="overlap"):
        g = ln.gsm(f, cutoff_factor=2.0)
    with _nowarn(BuriedPlaneWarning):
        ln.gsm(f, cutoff_factor=2.0)                                     # once per line
        Line(WR90, [X, Straight(1.5 * d), X]).gsm(f)                     # cf 1
    k = Line(WR90, [X, Straight(1.5 * d), X]).gsm(f, cutoff_factor=2.0, buried_planes="keep")
    p = np.concatenate(g.propagating())
    assert np.max(np.abs(g.S - k.S)[np.ix_(p, p)]) < 1e-10


def test_partial_depth_at_the_line_port_is_deferred_to_it():
    """WF3: a buried component at the start of a line, after a straight shorter than its depth: the line's port
    lies inside the discontinuity, and the line's GSM is the straight followed by the component, as defined."""
    f, W = 30e9, WR90.w
    d = 0.5 * W
    mm = wr.MiterMM()
    X = _Buried(mm, (d, 0.0))
    g = Line(WR90, [Straight(0.2 * d), X]).gsm(f, cutoff_factor=2.0)
    ref = (_straight(g.basis1, f, 0.2 * d) @ X.with_guide(WR90).gsm(f, g.basis1, g.basis2))
    assert g.buried.depth == pytest.approx((0.8 * d, 0.0), rel=1e-12)
    assert _close_relative(g.S, ref.S, 1e-12)
    assert np.max(np.abs(g.buried.gsm.S - mm.with_guide(WR90).gsm(f, g.basis1, g.basis2).S)) < 1e-13


def test_near_field_warning_uses_the_uniform_guide_left():
    """WF3: two components whose planes lie d inside, with a straight of 2 d + a bit, are joined directly."""
    f, W = 30e9, WR90.w
    d = W
    X = _Buried(wr.MiterMM(), (d, d))
    with pytest.warns(wr.line.EvanescentJunctionWarning, match="joined directly"):
        Line(WR90, [X, Straight(2 * d + 1e-3), X]).gsm(f)


def test_passivity_check_names_the_junction_or_the_element():
    """WF3: Line.gsm warns when the propagating block of the cascade has a singular value above 1 + 1e-8, naming
    the junction where the partial cascade stops being passive, or the element that is not passive itself."""
    f, W = 30e9, WR90.w
    N = _Noisy(wr.MiterMM(), 1e-6)
    with pytest.warns(NonPassiveCascadeWarning, match=r"junction of element 0 \(noisy\) and element 2 \(noisy\)"):
        g = Line(WR90, [N, Straight(0.5 * W), N]).gsm(f, cutoff_factor=2.0)
    assert _largest_singular_value(g, f) > 1 + PASSIVITY_TOL

    class _Gain(_Noisy):
        def gsm(self, f, basis1=None, basis2=None):
            g = self.inner.gsm(f, basis1, basis2)
            return GSM(g.S11, 1.001 * g.S12, 1.001 * g.S21, g.S22, g.basis1, g.basis2, f, self.name)

    with pytest.warns(NonPassiveCascadeWarning, match=r"element 1 \(gain\) itself"):
        Line(WR90, [Straight(0.01), _Gain(wr.MiterMM(), 0.0, name="gain"), Straight(0.01)]).gsm(f)
    with _nowarn(NonPassiveCascadeWarning):                              # the same lines without the defects
        Line(WR90, [wr.MiterMM(), Straight(0.5 * W), wr.MiterMM()]).gsm(f, cutoff_factor=2.0)
        Line(WR90, [Straight(0.01), wr.MiterMM(), Straight(0.01)]).gsm(f)


def test_lines_without_buried_planes_are_bitwise_unchanged():
    """WF3: lines without buried planes give the GSMs frozen before WF3 (tests/data/make_lines_unburied.py; bitwise
    on the machine that froze them, 1e-11 absolute across platforms as test_components)."""
    import json
    import os
    here = os.path.join(os.path.dirname(__file__), "data")
    sys.path.insert(0, here)
    try:
        import make_lines_unburied as mk
    finally:
        sys.path.pop(0)
    with np.load(os.path.join(here, "lines_unburied.npz")) as z:
        ref = {k: z[k] for k in z.files}
    assert json.loads(str(ref.pop("meta")))["git_commit"].startswith("d795a12")
    new = mk.compute()
    assert sorted(new) == sorted(ref)
    for k in ref:
        assert np.array_equal(new[k], ref[k]) or np.max(np.abs(new[k] - ref[k])) <= 1e-11, k


def test_passivity_check_cost_is_small():
    """WF3: the passivity check (one SVD of the propagating block) costs little next to the cascade."""
    import time
    f = 56e9
    ln = Line(WR90, [SmoothBend(0.05, ConstantCurvature(1 / 0.2)), Straight(0.05),
                     SmoothBend(0.05, ConstantCurvature(1 / 0.2), plane="E"), Straight(0.05)])
    g = ln.gsm(f, cutoff_factor=1.5)
    t0 = time.perf_counter()
    for _ in range(5):
        _largest_singular_value(g, f)
    t_check = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    ln.gsm(f, cutoff_factor=1.5)
    t_line = time.perf_counter() - t0
    assert t_check < 0.2 * t_line, (t_check, t_line)
