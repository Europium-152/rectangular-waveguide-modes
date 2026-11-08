"""Tests of wrmodes.components (WP1): Guide, Straight, SmoothBend, FrameChange."""
import numpy as np
import pytest

import wrmodes as wr
from wrmodes.components import Guide, Straight, SmoothBend, FrameChange, SechCurvature, ConstantCurvature, map_basis
from wrmodes.components.frames import frame_overlap
from wrmodes.gsm import straight, from_transfer_matrix
from wrmodes.modes import ModeBasis, Mode, eplane_frame_signs

WR90 = Guide(0.9 * 0.0254, 0.4 * 0.0254)
TOL = 1e-10


def assert_lossless(g):
    S = np.abs(g.S).max()
    assert g.reciprocity_defect() <= TOL * S, g.reciprocity_defect()
    assert g.unitarity_defect() <= TOL, g.unitarity_defect()


# ------------------------------------------------------------------------------------------ Guide
def test_guide_bases_and_frames():
    f = 40e9
    bh = WR90.basis(f)
    be = WR90.basis(f, plane="E")
    assert bh.frame == "H" and be.frame == "E"
    assert (bh.a, bh.b) == (WR90.w, WR90.h) and (be.a, be.b) == (WR90.h, WR90.w)
    assert bh == ModeBasis.propagating(f, WR90.w, WR90.h, frame="H")
    assert sorted((m.kind, m.n, m.m) for m in bh) == sorted((m.kind, m.m, m.n) for m in be)
    assert map_basis(be, "E", "H") == bh
    # with evanescent modes the E basis is the mapped global basis, not the E-frame default
    be15 = WR90.basis(f, 1.5, plane="E")
    assert map_basis(be15, "E", "H") == WR90.basis(f, 1.5)


def test_guide_is_immutable_and_sigma_is_a_hook():
    with pytest.raises(AttributeError):
        WR90.w = 1.0
    lossy = Guide(WR90.w, WR90.h, sigma=3.5e7)
    assert lossy.sigma == 3.5e7 and not lossy.lossless
    with pytest.raises(NotImplementedError):
        Straight(0.1, guide=lossy).gsm(30e9)
    with pytest.raises(NotImplementedError):
        SmoothBend(0.3, 1.0, guide=lossy).gsm(30e9)
    with pytest.raises(NotImplementedError):
        wr.Line(lossy, [Straight(0.1)]).gsm(30e9)


# ------------------------------------------------------------------------------------------ Straight
def test_straight_component():
    f = 30e9
    s = Straight(0.25, guide=WR90)
    b1, b2 = s.default_basis(f, 1.5)
    g = s.gsm(f, b1)
    assert np.array_equal(g.S, straight(b1, f, 0.25).S)
    assert_lossless(g)
    assert s.gsm(f).basis1 == WR90.basis(f)
    with pytest.raises(ValueError):
        Straight(0.1).default_basis(f)                    # no guide


# ------------------------------------------------------------------------------------------ SmoothBend
def legacy_family_phi(f, a, b, sub, L, kappa, atol=1e-6):
    m0 = sub.modes[0]
    _, r = wr.solve_transfer_matrix(f, a, b, (m0.kind, m0.m, m0.n), L, kappa, coupled_modes=sub.legacy_modes(f),
                                    atol=atol, error_control="full", profile=False)
    return r


@pytest.mark.parametrize("plane,f", [("H", 30e9), ("H", 45e9), ("E", 30e9), ("E", 45e9)])
def test_smooth_bend_matches_wrmodes_0_1_per_family(plane, f):
    L, kappa = 0.4, SechCurvature(90, 0.4)
    bend = SmoothBend(L, kappa, plane=plane, guide=WR90)
    g = bend.gsm(f)
    basis = g.basis1
    assert basis.frame == plane
    assert_lossless(g)
    for n in basis.families():
        idx = basis.family(n)
        modes, T, R, beta = wr.transmission_matrix(f, basis.a, basis.b, n, L, kappa)
        assert [m[0] for m in modes] == [basis[i].name for i in idx]
        assert np.allclose(g.S21[np.ix_(idx, idx)], T, atol=1e-10, rtol=0)
        assert np.allclose(g.S11[np.ix_(idx, idx)], R, atol=1e-10, rtol=0)
    # different families do not couple
    fam = np.array([m.n for m in basis])
    assert np.all(g.S21[fam[:, None] != fam[None, :]] == 0)


def test_single_mode_family_is_a_straight_section():
    f = 10e9                                    # TE1,0 only
    g = SmoothBend(0.3, 1 / 0.2, guide=WR90).gsm(f)
    assert len(g.basis1) == 1 and g.info[0] is None
    assert np.array_equal(g.S, straight(g.basis1, f, 0.3).S)


def test_evanescent_modes_are_exact_in_the_straight_limit():
    f = 40e9
    basis = WR90.basis(f, 1.6)
    assert (~basis.propagating_mask(f)).sum() >= 3
    g = SmoothBend(0.3, ConstantCurvature(0.0), guide=WR90).gsm(f, basis)
    assert np.allclose(g.S, straight(basis, f, 0.3).S, atol=1e-12, rtol=0)


@pytest.mark.parametrize("plane", ["H", "E"])
def test_propagating_result_does_not_depend_on_evanescent_modes(plane):
    f = 40e9
    bend = SmoothBend(0.4, SechCurvature(90, 0.4), plane=plane, guide=WR90)
    g1 = bend.gsm(f)
    b15 = WR90.basis(f, 1.6, plane=plane)
    g15 = bend.gsm(f, b15)
    ev = ~b15.propagating_mask(f)
    assert ev.sum() >= 3
    idx = [b15.index(m) for m in g1.basis1]
    assert np.array_equal(g15.S21[np.ix_(idx, idx)], g1.S21) and np.array_equal(g15.S11[np.ix_(idx, idx)], g1.S11)
    # evanescent modes: uncoupled, straight, no reflection
    beta = b15.beta(f)
    e = np.where(ev)[0]
    assert np.allclose(np.diag(g15.S21)[e], np.exp(-1j * beta[e] * 0.4), rtol=1e-14, atol=0)
    assert np.all(g15.S11[e] == 0) and np.all(g15.S21[np.ix_(e, ~ev)] == 0)
    assert_lossless(g15)
    assert g15.reciprocity_defect() == 0 or g15.reciprocity_defect() < 1e-12


def test_e_plane_bend_with_families_without_propagating_modes():
    """The line's E-frame basis can contain local families that are entirely evanescent."""
    f = 40e9
    be = WR90.basis(f, 1.6, plane="E")
    prop = be.propagating_mask(f)
    dead = [n for n in be.families() if not prop[be.family(n)].any()]
    assert dead
    g = SmoothBend(0.4, SechCurvature(60, 0.4), plane="E", guide=WR90).gsm(f, be)
    for n in dead:
        i = be.family(n)
        assert np.allclose(g.S21[np.ix_(i, i)], np.diag(np.exp(-1j * be.beta(f)[i] * 0.4)), rtol=1e-14, atol=0)
    assert_lossless(g)


def test_bend_refuses_a_basis_of_another_frame():
    f = 30e9
    with pytest.raises(ValueError):
        SmoothBend(0.3, 1.0, plane="E").gsm(f, WR90.basis(f, plane="H"))
    with pytest.raises(ValueError):
        SmoothBend(0.3, 1.0, plane="X")


def test_breakpoints_are_passed_on():
    """A step in curvature at mid-length: the bend uses the breakpoint exactly as solve_transfer_matrix does."""
    f, L = 25e9, 0.4
    kappa = lambda z: np.where(np.asarray(z) < L / 2, 1 / 0.3, -1 / 0.3)
    b = WR90.basis(f, plane="H")
    g = SmoothBend(L, kappa, breakpoints=[L / 2], guide=WR90).gsm(f, b)
    for n in b.families():
        idx = b.family(n)
        sub = b.subset(idx)
        m0 = sub.modes[0]
        _, r = wr.solve_transfer_matrix(f, b.a, b.b, (m0.kind, m0.m, m0.n), L, kappa, coupled_modes=sub.legacy_modes(f),
                                        error_control="full", breakpoints=[L / 2], profile=False)
        assert r.success and g.info[n]["n_slices"] == r.n_slices
        assert np.array_equal(g.S21[np.ix_(idx, idx)], from_transfer_matrix(r.transfer_matrix, sub, f).S21)


# ------------------------------------------------------------------------------------------ FrameChange
@pytest.mark.parametrize("f,cf", [(30e9, 1.0), (70e9, 1.0), (40e9, 1.7)])
def test_frame_change_is_the_signed_permutation_of_conventions(f, cf):
    fc = FrameChange("H", "E", guide=WR90)
    b1, b2 = fc.default_basis(f, cf)
    g = fc.gsm(f, b1, b2)
    F = g.S21
    assert np.array_equal(F, eplane_frame_signs(b1, b2))
    assert set(np.unique(F)) <= {-1.0, 0.0, 1.0}
    assert np.array_equal(np.abs(F).sum(0), np.ones(len(b1))) and np.array_equal(np.abs(F).sum(1), np.ones(len(b2)))
    assert np.array_equal(g.S12, F.T) and not g.S11.any() and not g.S22.any()
    assert_lossless(g)
    # the quadrature overlap is a signed permutation to 1e-12 before rounding
    M = frame_overlap(b1, "H", b2, "E")
    assert np.max(np.abs(M - F)) < 1e-12
    # examples of CONVENTIONS.md section 2
    for glob, loc, s in [(("TE", 1, 0), ("TE", 0, 1), -1), (("TE", 2, 0), ("TE", 0, 2), 1)]:
        if Mode(*glob) in b1:
            assert F[b2.index(loc), b1.index(glob)] == s


@pytest.mark.parametrize("f", [30e9, 70e9])
def test_h_e_h_frame_changes_compose_to_identity(f):
    he, eh = FrameChange("H", "E", guide=WR90), FrameChange("E", "H", guide=WR90)
    b1, b2 = he.default_basis(f, 1.4)
    g = he.gsm(f, b1, b2) @ eh.gsm(f, b2)
    N = len(b1)
    assert g.basis1 == b1 and g.basis2 == b1
    assert np.array_equal(g.S21, np.eye(N)) and np.array_equal(g.S12, np.eye(N))
    assert not g.S11.any() and not g.S22.any()
    # E -> H by quadrature is the transpose of H -> E
    assert np.array_equal(eh.gsm(f, b2).S21, he.gsm(f, b1).S21.T)


def test_frame_tags_prevent_joining_square_guides_without_frame_change():
    sq = Guide(0.02, 0.02)
    f = 30e9
    h = sq.basis(f)
    e = sq.basis(f, plane="E")
    assert h.modes == e.modes and h != e
    gh = Straight(0.1, guide=sq).gsm(f, h)
    ge = SmoothBend(0.2, 1.0, plane="E", guide=sq).gsm(f, e)
    with pytest.raises(ValueError):
        gh @ ge
    g = gh @ FrameChange("H", "E").gsm(f, h) @ ge
    assert g.basis2.frame == "E"
