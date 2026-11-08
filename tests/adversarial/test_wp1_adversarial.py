"""Adversarial tests of WP1 (components and cascade), written from tasks/WP1-cascade.md and CONVENTIONS.md only.

They skip until WP1's interfaces are merged. Weak points targeted: long straights (phase over 20 m at 90 GHz),
zero-length elements, cutoff proximity, many modes per family, extreme aspect ratios, evanescent modes with large
|beta| L, frame changes (signs, square guides), and the global-ODE reference of WP7.
"""
import os

import numpy as np
import pytest
import scipy.constants as cst

import ref_common
import ref_frames
import ref_interfaces as I
import ref_line
from wrmodes.modes import Mode, ModeBasis

WR90 = ref_line.WR90
TOL_P = 1e-6          # WP1 acceptance against the WP7 reference: 1e-6 of input power


def _line(segs, w=WR90[0], h=WR90[1], atol=1e-10):
    I.require("Guide", "Straight", "SmoothBend", "Line")
    return I.build_line(segs, w, h, atol=atol)


def _blocks(g, names):
    i1 = np.array([g.basis1.index(Mode.parse(n)) for n in names])
    i2 = np.array([g.basis2.index(Mode.parse(n)) for n in names])
    return {"S11": g.S11[np.ix_(i1, i1)], "S21": g.S21[np.ix_(i2, i1)],
            "S12": g.S12[np.ix_(i1, i2)], "S22": g.S22[np.ix_(i2, i2)]}


def _max_dP(A, B):
    return max(np.max(np.abs(np.abs(A[k]) ** 2 - np.abs(B[k]) ** 2)) for k in A)


def _beta(mode, a, b, f):
    """beta from first principles, not from wrmodes: sqrt(k^2 - (m pi/a)^2 - (n pi/b)^2)."""
    k = 2 * np.pi * f / cst.c
    return np.sqrt(k * k - (mode.m * np.pi / a) ** 2 - (mode.n * np.pi / b) ** 2)


# ----------------------------------------------------------------------------------------- reference lines
@pytest.mark.parametrize("name", ["same-plane", "mixed-planes", "long-straights"])
def test_lines_match_global_ode_reference(name):
    """The three WP7 reference lines (stored data, every input mode, both ports) at their first frequency."""
    I.require("Guide", "Straight", "SmoothBend", "Line")
    path = os.path.join(ref_common.DATA, "line_reference.npz")
    data = ref_common.load_npz(path)
    f = ref_line.REFERENCE_FREQUENCIES[name][0]
    key = f"{name}/{f / 1e9:g}"
    names = [str(m) for m in data[f"{key}/modes"]]
    ref = {k: data[f"{key}/{k}"] for k in ("S11", "S21", "S12", "S22")}
    g = _line(ref_line.reference_lines()[name]).gsm(f, cutoff_factor=1.0)
    assert _max_dP(_blocks(g, names), ref) <= TOL_P


def test_mixed_plane_line_at_40GHz_all_families_coupled():
    """26 modes, H and E bends: every family is coupled to every other by the frame changes."""
    segs = ref_line.reference_lines()["mixed-planes"]
    gb, _, S, info = ref_line.line_transfer(40e9, *WR90, segs)
    assert info["groups"] == 1
    g = _line(segs).gsm(40e9, cutoff_factor=1.0)
    assert _max_dP(_blocks(g, gb.names), S) <= TOL_P


# ----------------------------------------------------------------------------------------- long straights
def test_phase_of_20m_straight_at_90GHz():
    """beta L is about 3.8e4 rad; every propagating mode's phase must hold to 1e-10 (a few ulp of beta)."""
    f, L = 90e9, 20.0
    g = _line([ref_line.straight(L)]).gsm(f, cutoff_factor=1.0)
    for i, m in enumerate(g.basis1.modes):
        j = g.basis2.index(m)
        assert abs(g.S21[j, i] - np.exp(-1j * _beta(m, *WR90, f) * L)) < 1e-10, m
    assert np.max(np.abs(g.S11)) == 0 and np.max(np.abs(g.S22)) == 0


def test_long_line_equals_reference_at_90GHz():
    """Paper section 7 sequence with two 10 m straights at 90 GHz (129 modes): WP1 vs global ODE."""
    I.require("Guide", "Straight", "SmoothBend", "Line")
    path = os.path.join(ref_common.DATA, "line_reference.npz")
    data = ref_common.load_npz(path)
    key = "long-straights/90"
    if f"{key}/S21" not in data:
        pytest.skip("90 GHz reference not stored")
    names = [str(m) for m in data[f"{key}/modes"]]
    ref = {k: data[f"{key}/{k}"] for k in ("S11", "S21", "S12", "S22")}
    g = _line(ref_line.reference_lines()["long-straights"]).gsm(90e9, cutoff_factor=1.0)
    assert _max_dP(_blocks(g, names), ref) <= TOL_P


# ----------------------------------------------------------------------------------------- zero length
@pytest.mark.parametrize("comp", ["straight", "bendH", "bendE"])
def test_zero_length_elements_are_identity(comp):
    Straight, SmoothBend, Guide, Line = I.require("Straight", "SmoothBend", "Guide", "Line")
    c = {"straight": lambda: Straight(0.0),
         "bendH": lambda: SmoothBend(0.0, lambda z: np.full(np.shape(z), 5.0), plane="H"),
         "bendE": lambda: SmoothBend(0.0, lambda z: np.full(np.shape(z), 5.0), plane="E")}[comp]
    try:
        comp_obj = c()
        g = Line(Guide(*WR90), [comp_obj]).gsm(60e9)
    except ValueError:
        pytest.skip("zero-length element refused with a ValueError (acceptable, but say so in the docs)")
    n = len(g.basis1)
    assert np.allclose(g.S21, np.eye(n), atol=1e-12) and np.allclose(g.S11, 0, atol=1e-12)


# ----------------------------------------------------------------------------------------- cutoff proximity
@pytest.mark.parametrize("rel", [1e-3, 1e-6])
def test_bend_just_above_a_cutoff(rel):
    """TE3,0 just above cutoff: the 0.1 coupling coefficients grow like beta^-1/2. The GSM must stay finite,
    reciprocal and unitary, and equal the global-ODE reference in power."""
    fc = 3 * cst.c / (2 * WR90[0])
    f = fc * (1 + rel)
    segs = [ref_line.bend(0.1, ref_line.sech_curvature(30, 0.1), "H")]
    g = _line(segs).gsm(f, cutoff_factor=1.0)
    assert np.all(np.isfinite(g.S))
    assert g.reciprocity_defect() <= 1e-10 * np.max(np.abs(g.S))
    assert g.unitarity_defect() <= 1e-10
    gb, _, S, _ = ref_line.line_transfer(f, *WR90, segs)
    assert _max_dP(_blocks(g, gb.names), S) <= TOL_P


# ----------------------------------------------------------------------------------------- many modes
def test_bend_with_35_modes_in_one_family():
    """44 x 1.2 mm guide at 120 GHz: 35 TE_m0 modes in family 0 and no other family (aspect ratio 37)."""
    w, h, f = 0.044, 0.0012, 120e9
    segs = [ref_line.bend(0.1, ref_line.sech_curvature(20, 0.1), "H")]
    basis = ModeBasis.propagating(f, w, h)
    assert len(basis) == len(basis.family(0)) >= 30
    g = _line(segs, w, h).gsm(f, cutoff_factor=1.0)
    assert g.unitarity_defect() <= 1e-10
    gb, _, S, _ = ref_line.line_transfer(f, w, h, segs)
    assert _max_dP(_blocks(g, gb.names), S) <= TOL_P


# ----------------------------------------------------------------------------------------- aspect ratio
def test_extreme_aspect_ratio_eplane_bend():
    """22.86 x 1.6 mm guide (aspect 14), E-plane bend: the local guide is 1.6 x 22.86 mm."""
    w, h, f = 22.86e-3, 1.6e-3, 60e9
    segs = [ref_line.bend(0.2, ref_line.sech_curvature(45, 0.2), "E")]
    g = _line(segs, w, h).gsm(f, cutoff_factor=1.0)
    gb, _, S, _ = ref_line.line_transfer(f, w, h, segs)
    assert _max_dP(_blocks(g, gb.names), S) <= TOL_P


# ----------------------------------------------------------------------------------------- evanescent modes
def test_evanescent_modes_through_a_long_straight_do_not_overflow():
    """cutoff_factor 3 on a 10 m line at 20 GHz: evanescent S21 ~ exp(-|beta| 10 m) underflows to 0; nothing may be
    NaN or inf, and the propagating block must not change (WP1 card)."""
    segs = [ref_line.straight(5.0), ref_line.bend(0.4, ref_line.sech_curvature(90, 0.4), "H"), ref_line.straight(5.0)]
    line = _line(segs)
    g3 = line.gsm(20e9, cutoff_factor=3.0)
    g1 = line.gsm(20e9, cutoff_factor=1.0)
    assert np.all(np.isfinite(g3.S))
    names = g1.basis1.names
    b3, b1 = _blocks(g3, names), _blocks(g1, names)
    assert max(np.max(np.abs(b3[k] - b1[k])) for k in b1) < 1e-10
    assert len(g3.basis1) > len(g1.basis1)


# ----------------------------------------------------------------------------------------- frame changes
def _frame_change_gsm(frm, to, w, h, f):
    """FrameChange(frm, to).gsm with an explicit port-1 basis (the card gives FrameChange no guide)."""
    FrameChange = I.require("FrameChange")
    dims = (w, h) if frm == "H" else (h, w)
    b1 = ModeBasis.propagating(f, *dims, frame=frm)
    fc = FrameChange(frm, to)
    try:
        return fc.gsm(f, b1)
    except (TypeError, ValueError):
        dims2 = (h, w) if frm == "H" else (w, h)
        return fc.gsm(f, b1, ModeBasis.propagating(f, *dims2, frame=to))


def test_frame_change_is_the_derived_signed_permutation():
    f = 70e9
    g = _frame_change_gsm("H", "E", *WR90, f)
    gb = ModeBasis(g.basis1.a, g.basis1.b, g.basis1.modes, sort=False)
    lb = ref_frames.eplane_local_basis(gb)
    F = ref_frames.eplane_matrix(gb, lb)
    i2 = np.array([g.basis2.index(m) for m in lb.modes])
    assert np.array_equal(g.S21[i2, :], F) and np.array_equal(g.S12[:, i2], F.T)   # exact, not approximately
    assert not np.any(g.S11) and not np.any(g.S22)
    back = _frame_change_gsm("E", "H", *WR90, f)
    hh = g @ back
    assert np.array_equal(np.abs(hh.S21), np.eye(len(hh.basis1))) and np.allclose(hh.S21, np.eye(len(hh.basis1)))


def test_square_guide_frame_tags_prevent_silent_joins():
    """In a 20 x 20 mm guide the H and E local bases have equal dimensions; only the frame tags tell them apart.
    Joining an E-frame output directly to an H-frame input must be refused."""
    g = _frame_change_gsm("H", "E", 0.02, 0.02, 30e9)
    assert g.basis1.frame is not None and g.basis2.frame is not None and g.basis1.frame != g.basis2.frame
    with pytest.raises(ValueError):
        g @ g


def test_square_guide_mixed_line_matches_reference():
    """Mixed-plane line in a square guide, where TE_mn and TE_nm are degenerate: frame signs decide the result."""
    w = h = 0.02
    segs = [ref_line.bend(0.3, ref_line.sech_curvature(60, 0.3), "H"), ref_line.straight(0.25),
            ref_line.bend(0.3, ref_line.sech_curvature(60, 0.3), "E")]
    f = 25e9
    g = _line(segs, w, h).gsm(f, cutoff_factor=1.0)
    gb, _, S, _ = ref_line.line_transfer(f, w, h, segs)
    assert _max_dP(_blocks(g, gb.names), S) <= TOL_P
