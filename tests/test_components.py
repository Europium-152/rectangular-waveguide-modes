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
    # WP2: lossy straights, bends and lines are passive (the loss itself is tested in tests/test_loss.py)
    for g in (Straight(0.1, guide=lossy).gsm(30e9), SmoothBend(0.3, 1.0, guide=lossy).gsm(30e9),
              wr.Line(lossy, [Straight(0.1)]).gsm(30e9)):
        assert 0 < g.passivity_margin() < 1e-2
    with pytest.raises(NotImplementedError):
        lossy.require_lossless("a lossless-only component")


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
    assert g.info["route"] == "VariableGuide"
    assert np.allclose(g.S, straight(basis, f, 0.3).S, atol=1e-12, rtol=0)


def test_propagating_path_is_bitwise_unchanged():
    """WF2 item 2: with propagating-mode bases SmoothBend keeps its transfer-matrix route, bitwise (frozen outputs
    of origin/main before WF2, tests/data/make_smoothbend_propagating.py)."""
    import json
    import os
    import sys
    here = os.path.join(os.path.dirname(__file__), "data")
    sys.path.insert(0, here)
    try:
        import make_smoothbend_propagating as mk
    finally:
        sys.path.pop(0)
    with np.load(os.path.join(here, "smoothbend_propagating.npz")) as z:
        ref = {k: z[k] for k in z.files}
    assert json.loads(str(ref.pop("meta")))["git_commit"].startswith("7cb7cf1")
    new = mk.compute()
    assert sorted(new) == sorted(ref)
    for k in ref:
        assert np.array_equal(new[k], ref[k]), k


@pytest.mark.parametrize("plane", ["H", "E"])
def test_evanescent_modes_are_coupled(plane):
    """WF2 item 2: with evanescent modes in the basis the bend is solved as a VariableGuide (curvature kappa(z),
    constant a, b): the evanescent modes are coupled to the propagating ones, the GSM is reciprocal, lossless and
    symmetric. For the 40 cm 90 deg sech bend at 40 GHz the TE1,0 (H) / TE0,1 (E) column of the propagating block
    moves by < 1.4e-7 in power from the propagating-only result; entries between high-order modes near the
    degenerate TE2,2 / TM2,2 pair (H) or TM1,5 (E) move by 2.8e-4 / 4.8e-4 (converged in the internal list)."""
    f = 40e9
    bend = SmoothBend(0.4, SechCurvature(90, 0.4), plane=plane, guide=WR90)
    g1 = bend.gsm(f)
    b16 = WR90.basis(f, 1.6, plane=plane)
    g16 = bend.gsm(f, b16)
    assert g16.info["route"] == "VariableGuide" and g16.basis1 == b16 == g16.basis2
    ev = ~b16.propagating_mask(f)
    assert ev.sum() >= 3
    assert np.abs(g16.S21[np.ix_(ev, ~ev)]).max() > 1e-8          # coupled now (WP1: exactly zero)
    assert_lossless(g16)
    assert np.max(np.abs(g16.S11 - g16.S22)) < 1e-9                # symmetric sech profile, one family per class
    idx = [b16.index(m) for m in g1.basis1]
    D = np.abs(np.abs(g16.S21[np.ix_(idx, idx)]) ** 2 - np.abs(g1.S21) ** 2)
    j = g1.basis1.index("TE1,0" if plane == "H" else "TE0,1")
    assert D[:, j].max() < 1e-6 and D.max() < 1e-3, (D[:, j].max(), D.max())


def test_evanescent_route_is_variable_guide():
    """The delegation is exact: the GSM equals VariableGuide(L, a, b, kappa) on the same bases, internal cutoff
    factor twice that of the port basis."""
    f = 30e9
    L, R = 0.05, 0.05
    bend = SmoothBend(L, ConstantCurvature(1 / R), guide=WR90)
    b = WR90.basis(f, 2.0)
    g = bend.gsm(f, b)
    cf = max(np.pi * np.hypot(m.m / b.a, m.n / b.b) for m in b.modes) / (2 * np.pi * f / 299792458.0)
    vg = wr.VariableGuide(L, b.a, b.b, kappa=1 / R, plane="H", guide=WR90, uniform=[(0, L)], mode_factor=2 * cf)
    h = vg.gsm(f, b, b)
    assert np.array_equal(g.S, h.S)
    assert g.info["mode_factor"] == pytest.approx(2 * cf)


def test_lossy_bend_with_evanescent_modes_is_passive_and_reciprocal():
    lossy = Guide(WR90.w, WR90.h, 3.5e7)
    f = 30e9
    g = SmoothBend(0.1, ConstantCurvature(1 / 0.1), plane="E", guide=lossy).gsm(f, lossy.basis(f, 1.5, plane="E"))
    assert g.reciprocity_defect() <= TOL * np.abs(g.S).max() and 0 < g.passivity_margin() < 0.05


def test_e_plane_bend_with_families_without_propagating_modes():
    """The line's E-frame basis can contain local families that are entirely evanescent; they are coupled within
    the family by the curvature, and decay like the straight evanescent modes."""
    f = 40e9
    be = WR90.basis(f, 1.6, plane="E")
    prop = be.propagating_mask(f)
    dead = [n for n in be.families() if not prop[be.family(n)].any()]
    assert dead
    g = SmoothBend(0.4, SechCurvature(60, 0.4), plane="E", guide=WR90).gsm(f, be)
    for n in dead:
        i = be.family(n)
        assert np.abs(g.S21[np.ix_(i, i)]).max() < 1e-30            # decayed over 0.4 m, as e^(-|beta| L)
        assert np.abs(g.S21[np.ix_(i, prop)]).max() == 0             # no coupling to other families
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


# ------------------------------------------------------------------------------------------ review fixes
def test_live_rule_keeps_evanescent_modes_of_the_live_families_of_both_frames():
    f = 10e9                                     # TE1,0 only propagates
    be = WR90.basis(f, 3, plane="E")
    fam1 = [m.name for m in be if m.n == 1]      # local family n = 1 = global m = 1
    assert "TE1,1" in fam1 and "TM1,1" in fam1 and "TE0,1" in fam1
    assert [m.name for m in WR90.basis(f, 3, plane="E", evanescent_families="propagating") if m.n == 1] == ["TE0,1"]
    gb = WR90.basis(f, 3)
    prop = gb.propagating_mask(f)
    live_n = {m.n for m, p in zip(gb, prop) if p}
    live_m = {m.m for m, p in zip(gb, prop) if p}
    full = ModeBasis.below(3 * f, WR90.w, WR90.h)
    assert set(gb.modes) == {m for m in full if m.n in live_n or m.m in live_m}
    assert len(WR90.basis(f, 3, evanescent_families="all")) == len(full)
    with pytest.raises(ValueError):
        WR90.basis(f, 3, evanescent_families="none")


@pytest.mark.parametrize("f", [12e9, 20e9])
def test_live_rule_is_symmetric_in_a_square_guide(f):
    sq = Guide(0.02, 0.02)
    assert sq.basis(f, 3).modes == sq.basis(f, 3, plane="E").modes


def test_plane_is_part_of_the_protocol():
    from wrmodes.components.base import Component, UNDECLARED
    assert Component.plane is UNDECLARED
    assert Straight.plane is None and FrameChange.plane is None
    assert SmoothBend(0.3, 1.0, plane="E").plane == "E"


def test_empty_bases_below_the_first_cutoff():
    f = 5e9
    assert len(WR90.basis(f)) == 0
    g = SmoothBend(0.3, 1.0, guide=WR90).gsm(f)
    assert g.S.shape == (0, 0)
    g = FrameChange("H", "E", guide=WR90).gsm(f)
    assert g.S.shape == (0, 0) and g.basis2.frame == "E"
    assert Straight(1.0, guide=WR90).gsm(f).S.shape == (0, 0)


class LossyStraight(Straight):
    """Test double: a straight with a given (symmetric) Lambda_full = [[L+, L-], [-L-, -L+]], to check the hook."""

    def _loss_matrix(self, basis, f):
        n = len(basis)
        lp = np.diag(np.linspace(0.1, 0.3, n)).astype(complex)
        lm = np.zeros((n, n), complex)
        if n > 1:
            lp[0, 1] = lp[1, 0] = 0.02
            lm[0, 1] = lm[1, 0] = 0.01j
        return np.block([[lp, lm], [-lm, -lp]])


def test_loss_matrix_hook_of_straight():
    from scipy.linalg import expm
    f, L = 30e9, 2.0
    s = LossyStraight(L, guide=WR90)
    g = s.gsm(f)
    b = g.basis1
    lam = s._loss_matrix(b, f)
    D = np.diag(np.concatenate([-1j * b.beta(f), 1j * b.beta(f)]))
    assert np.allclose(g.S, from_transfer_matrix(expm((D - lam) * L), b, f).S, atol=1e-13, rtol=0)
    assert g.reciprocity_defect() < 1e-12 and g.passivity_margin() > 0
    assert not np.any(Straight(L, guide=WR90)._loss_matrix(b, f))


def test_smooth_bend_loss_hook():
    assert SmoothBend(0.3, 1.0, guide=WR90)._loss_matrix(WR90.basis(30e9), 30e9) is None
    lossy = Guide(WR90.w, WR90.h, sigma=3.5e7)
    L0, L1 = SmoothBend(0.3, 1.0, guide=lossy)._loss_matrix(lossy.basis(30e9), 30e9)
    assert L0.shape == L1.shape == (2 * len(lossy.basis(30e9)),) * 2 and np.any(L0) and np.any(L1)